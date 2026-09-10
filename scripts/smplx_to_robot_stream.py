import argparse
import csv
import inspect
import os
import pickle
import pathlib
import sys
import threading
import time
import warnings
from collections import deque

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from rich import print

# Ensure repository root is on sys.path when launching via
# `python scripts/smplx_to_robot_stream.py`.
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def _exp_smooth(curr, prev, alpha):
    if prev is None or alpha >= 0.999:
        return curr
    return alpha * curr + (1.0 - alpha) * prev


def _resolve_torch_device(requested):
    req = str(requested).strip().lower()
    if req == "cpu":
        return "cpu"

    allow_cuda_postprocessor = os.environ.get("GMR_ALLOW_CUDA_POSTPROCESSOR", "0") == "1"
    if req in ("cuda", "cuda:0", "auto") and not allow_cuda_postprocessor:
        print(
            "[Stream] GMR postprocessor uses CPU by default to avoid CUDA NVRTC "
            "architecture errors. Set GMR_ALLOW_CUDA_POSTPROCESSOR=1 to override."
        )
        return "cpu"

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda:0"
        print("[Stream] Warning: --torch_device=cuda but CUDA is unavailable, fallback to CPU.")
        return "cpu"

    return "cpu"


def _is_nvrtc_arch_error(exc):
    msg = str(exc).lower()
    return "nvrtc" in msg and (
        "gpu-architecture" in msg or
        "--gpu-architecture" in msg or
        "-arch" in msg
    )


def _extract_qpos(retarget_result):
    if isinstance(retarget_result, tuple):
        return retarget_result[0]
    return retarget_result


def _segment_segment_distance(p1, p2, p3, p4):
    """Minimum distance between line segments p1-p2 and p3-p4 (Ericson 2005)."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b
    sN = tN = D
    sD = tD = D
    if D < 1e-8:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a
    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD
    dp = w + sc * u - tc * v
    return float(np.linalg.norm(dp))


class OnlineQposPostprocessor:
    X02LITE_R_ELBOW_QPOS_INDEX = 7 + 7
    X02LITE_R_ELBOW_SOFT_MAX_RAD = 1.35

    def __init__(
        self,
        xml_file,
        root_body_name=None,
        smooth_alpha=0.25,
        arm_smooth_alpha=0.3,
        height_adjust=True,
        root_origin_offset=True,
        torch_device="auto",
    ):
        self.smooth_alpha = float(smooth_alpha)
        self.arm_smooth_alpha = float(arm_smooth_alpha)
        self.height_adjust = bool(height_adjust)
        self.root_origin_offset = bool(root_origin_offset)
        self.prev_qpos = None
        self.xy_origin = None
        self.xml_file = xml_file
        self.root_body_name = root_body_name
        self.is_x02lite = "x02lite" in str(xml_file).replace("\\", "/").lower()
        self._arm_nudge_smoothed = {}

        device = _resolve_torch_device(torch_device)
        self._set_kinematics_device(device)

        # Distance from the lowest body frame (e.g. the ankle) down to the actual
        # sole of the foot mesh.  height_adjust aligns the lowest *frame* to the
        # ground, which leaves the sole buried underground, so we add this back.
        self.sole_offset = self._compute_sole_offset()

        # Detect foot bodies and their ankle-pitch/knee/hip-pitch joint indices so
        # we can flatten the foot onto the ground during stance (fixes the
        # "hovering" look when WHAM reports the foot toe-down).  Empty -> disabled.
        self._foot_pitch_chain = self._detect_foot_pitch_chain()

        # Detect arms and their shoulder-roll (abduction) dof plus the thigh they
        # can clip, so we can push the forearm clear of the hip during sit->stand.
        self._arm_thigh_chain = self._detect_arm_thigh_chain()

    def _detect_foot_pitch_chain(self):
        """Map foot body name -> ankle-pitch dof index for flat-foot grounding.

        A foot is the last link of a leg (``<prefix>_ankle_roll_link`` here), and
        its ankle-pitch joint is the hinge on the parent
        ``<prefix>_ankle_pitch_link`` body.  Returns [] when the naming convention
        is not recognized (feature silently disabled)."""
        names = self.kinematics_model.body_names
        dof_indices = self.kinematics_model.joint_dof_idx
        dof_of = {
            name: dof_indices[i]
            for i, name in enumerate(names)
            if dof_indices[i] >= 0
        }
        chain = []
        for name in names:
            if not name.endswith("_ankle_roll_link"):
                continue
            prefix = name[: -len("_ankle_roll_link")]
            ankle_pitch = dof_of.get(prefix + "_ankle_pitch_link")
            if ankle_pitch is None:
                continue
            chain.append({"foot_body": name, "ankle_pitch": ankle_pitch})
        if chain:
            print(f"[Stream] Flat-foot grounding enabled for: "
                  f"{', '.join(c['foot_body'] for c in chain)}")
        return chain

    def _detect_arm_thigh_chain(self):
        """Map each arm to its shoulder-roll (abduction) dof and the segments it
        clips.

        An arm is recognised by its ``<prefix>_elbow_link``; the forearm runs from
        there to ``<prefix>_wrist_yaw_link``.  Its obstacles are the thigh
        (``<prefix>_hip_yaw_link`` -> ``<prefix>_knee_link``) and, for the left arm
        only, the hip/pelvis (``base_link`` -> ``<prefix>_hip_pitch_link``).
        Returns [] when the naming convention is not recognised (feature silently
        disabled)."""
        names = self.kinematics_model.body_names
        dof_indices = self.kinematics_model.joint_dof_idx
        dof_of = {
            name: dof_indices[i]
            for i, name in enumerate(names)
            if dof_indices[i] >= 0
        }
        lo, hi = self._dof_limits()
        chain = []
        for name in names:
            if not name.endswith("_elbow_link"):
                continue
            prefix = name[: -len("_elbow_link")]
            shoulder_roll = dof_of.get(prefix + "_shoulder_roll_link")
            wrist = prefix + "_wrist_yaw_link"
            thigh = prefix + "_hip_yaw_link"
            knee = prefix + "_knee_link"
            hip = prefix + "_hip_pitch_link"
            if shoulder_roll is None:
                continue
            if wrist not in names or thigh not in names or knee not in names:
                continue
            # Every shoulder/elbow dof we may rotate to clear the thigh.
            arm_dofs = [
                dof_of[prefix + "_" + seg + "_link"]
                for seg in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
                if prefix + "_" + seg + "_link" in dof_of
            ]
            # Abduction is the side of the shoulder-roll joint with the wider
            # travel (for a symmetric humanoid the sign flips between arms).
            abduct_sign = 1.0 if hi[shoulder_roll] >= -lo[shoulder_roll] else -1.0
            # The forearm can also clip through the hip/pelvis, not just the
            # thigh.  The hip segment runs from the pelvis (root body) down to the
            # hip-pitch joint; the left arm (the side seen clipping) avoids it too.
            hip_a = hip_b = None
            if "left" in prefix and hip in names:
                hip_a = self.kinematics_model.body_names[0]  # pelvis / root body
                hip_b = hip
            chain.append({
                "shoulder_roll": shoulder_roll,
                "abduct_sign": abduct_sign,
                "arm_dofs": arm_dofs,
                "elbow_body": name,
                "wrist_body": wrist,
                "thigh_body": thigh,
                "knee_body": knee,
                "hip_a_body": hip_a,
                "hip_b_body": hip_b,
            })
        if chain:
            print(f"[Stream] Arm-thigh collision avoidance enabled for: "
                  f"{', '.join(c['elbow_body'] for c in chain)}")
        return chain

    def _get_mj_model(self):
        if getattr(self, "_mj", None) is None:
            import mujoco as mj
            self._mj = mj
            self._mj_model = mj.MjModel.from_xml_path(self.xml_file)
            self._mj_data = mj.MjData(self._mj_model)
            self._mj_body_id_cache = {}
        return self._mj_model, self._mj_data

    def _mj_body_id(self, name):
        self._get_mj_model()
        if name not in self._mj_body_id_cache:
            self._mj_body_id_cache[name] = self._mj.mj_name2id(
                self._mj_model, self._mj.mjtObj.mjOBJ_BODY, name
            )
        return self._mj_body_id_cache[name]

    def _sole_lowest_z(self, m, d, body_name):
        """Lowest world z of a body's visual (group 0) mesh vertices."""
        bid = self._mj_body_id(body_name)
        zmin = float("inf")
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid:
                continue
            if m.geom_type[g] != self._mj.mjtGeom.mjGEOM_MESH:
                continue
            if m.geom_group[g] != 0:
                continue
            mid = m.geom_dataid[g]
            start = m.mesh_vertadr[mid]
            count = m.mesh_vertnum[mid]
            verts = m.mesh_vert[start:start + count]
            w = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            zmin = min(zmin, float(np.min(w[:, 2])))
        return zmin

    def _dof_limits(self):
        if getattr(self, "_dof_lo", None) is None:
            lo, hi = self.kinematics_model.get_dof_limits()
            self._dof_lo = lo.detach().cpu().numpy().astype(np.float64)
            self._dof_hi = hi.detach().cpu().numpy().astype(np.float64)
        return self._dof_lo, self._dof_hi

    def _flatten_stance_feet(self, q, stance_clearance=0.03, blend=0.6, max_step=0.3):
        """Rotate each planted foot's ankle-pitch so its sole sits flat on the ground.

        WHAM (especially from a top-down drone view) estimates the foot with a
        toe-down tilt during stance, which reads as the heel hovering and the toe
        digging in.  For a planted foot we rotate the ankle-pitch joint by the
        amount that brings the foot's world forward axis back to horizontal, using
        a numerical Jacobian of that axis w.r.t. the ankle-pitch dof (so the
        joint's axis/rotation sign conventions are handled automatically).
        Swing feet (sole clearly above the ground) are left untouched."""
        if not self._foot_pitch_chain or self.sole_offset <= 0:
            return q
        m, d = self._get_mj_model()
        d.qpos[:] = q
        self._mj.mj_forward(m, d)
        lower, upper = self._dof_limits()
        for c in self._foot_pitch_chain:
            # Planted when the lowest point of the sole mesh is near the ground.
            if self._sole_lowest_z(m, d, c["foot_body"]) > stance_clearance:
                continue
            bid = self._mj_body_id(c["foot_body"])
            dof = c["ankle_pitch"]
            fwd = d.xmat[bid].reshape(3, 3)[:, 0]  # world forward (local +x)
            fz = float(fwd[2])
            if abs(fz) < 1e-4:
                continue
            # Numerical Jacobian: d(fwd_z) / d(ankle_pitch).
            q_test = q.copy()
            q_test[7 + dof] += 0.02
            d.qpos[:] = q_test
            self._mj.mj_forward(m, d)
            fz_p = float(d.xmat[bid].reshape(3, 3)[:, 0][2])
            gain = (fz_p - fz) / 0.02
            if abs(gain) < 1e-4:
                continue
            dtheta = float(np.clip(-fz / gain, -max_step, max_step))
            new = float(np.clip(float(q[7 + dof]) + dtheta * blend, lower[dof], upper[dof]))
            q[7 + dof] = new
            # Re-forward so the next foot sees the updated pose.
            d.qpos[:] = q
            self._mj.mj_forward(m, d)
        return q

    def _arm_obstacles(self, c):
        """Segments the forearm must stay clear of: the thigh, and (for the left
        arm) the hip/pelvis as well."""
        segs = [(c["thigh_body"], c["knee_body"])]
        if c.get("hip_a_body") is not None:
            segs.append((c["hip_a_body"], c["hip_b_body"]))
        return segs

    def _arm_clearance(self, d, c):
        """Nearest (dist, a_body, b_body) of the forearm's obstacles to its
        centerline."""
        a = d.xpos[self._mj_body_id(c["elbow_body"])]
        b = d.xpos[self._mj_body_id(c["wrist_body"])]
        best = None
        for pa, pb in self._arm_obstacles(c):
            dist = _segment_segment_distance(
                a, b, d.xpos[self._mj_body_id(pa)], d.xpos[self._mj_body_id(pb)]
            )
            if best is None or dist < best[0]:
                best = (dist, pa, pb)
        return best

    def _arm_gain(self, m, d, c, qw, dof, sign, obs, dist, step=0.05):
        """Forward-difference of the forearm->obstacle distance w.r.t. ``dof`` in
        direction ``sign`` (or -inf if the move leaves the joint range)."""
        lower, upper = self._dof_limits()
        new_val = float(qw[7 + dof]) + step * sign
        if new_val < lower[dof] or new_val > upper[dof]:
            return float("-inf")
        q_probe = qw.copy()
        q_probe[7 + dof] = new_val
        d.qpos[:] = q_probe
        self._mj.mj_forward(m, d)
        a = d.xpos[self._mj_body_id(c["elbow_body"])]
        b = d.xpos[self._mj_body_id(c["wrist_body"])]
        pa, pb = obs
        d_new = _segment_segment_distance(
            a, b, d.xpos[self._mj_body_id(pa)], d.xpos[self._mj_body_id(pb)]
        )
        return (d_new - dist) / step

    def _avoid_arm_thigh_collision(self, q, clearance=0.13, blend=0.9, max_step=0.15, max_iters=6):
        """Keep the forearms clear of the thighs (and, for the left arm, the hip).

        When the arm hangs by the side the forearm can pass through the thigh or
        the hip/pelvis (reading as the hand/forearm intersecting the hip/butt
        during sit->stand).  For any arm whose forearm centerline is closer than
        ``clearance`` to the nearest obstacle segment we rotate the arm away.
        Normally this is shoulder-roll abduction (raising the arm out to the
        side), but during sit->stand the hand naturally presses onto the front of
        the thigh, where abduction would push the forearm straight through it; in
        that case we search the whole shoulder+elbow chain for whichever joint
        clears it best.  The correction (offset from the raw pose) is
        exponentially smoothed per joint across frames so the arm eases out and
        back instead of snapping."""
        if not self._arm_thigh_chain:
            return q
        m, d = self._get_mj_model()
        lower, upper = self._dof_limits()
        for c in self._arm_thigh_chain:
            # Solve for the collision-free pose on a working copy of the pose.
            qw = q.copy()
            d.qpos[:] = qw
            self._mj.mj_forward(m, d)
            for _ in range(max_iters):
                dist, pa, pb = self._arm_clearance(d, c)
                if dist >= clearance:
                    break
                deficit = clearance - dist
                obs = (pa, pb)
                # Primary: abduction.  This keeps the ordinary by-the-side
                # correction unchanged.
                gain_abd = self._arm_gain(m, d, c, qw, c["shoulder_roll"], c["abduct_sign"], obs, dist)
                best = (gain_abd, c["shoulder_roll"], c["abduct_sign"]) if gain_abd > 1e-4 else None
                if best is None:
                    # Fallback: abduction would push the forearm further into the
                    # thigh/hip (e.g. hands pressing down on the thighs).  Pick
                    # whichever shoulder/elbow dof clears it best.
                    for dof in c["arm_dofs"]:
                        for sign in (+1.0, -1.0):
                            gain = self._arm_gain(m, d, c, qw, dof, sign, obs, dist)
                            if gain > 1e-4 and (best is None or gain > best[0]):
                                best = (gain, dof, sign)
                if best is None:
                    break
                gain, dof, sign = best
                dtheta = float(np.clip(deficit / gain, 0.0, max_step))
                new = float(np.clip(float(qw[7 + dof]) + sign * dtheta * blend, lower[dof], upper[dof]))
                if abs(new - qw[7 + dof]) < 1e-5:
                    break
                qw[7 + dof] = new
                d.qpos[:] = qw
                self._mj.mj_forward(m, d)
            # Smooth the correction (offset from the raw pose) per joint so the
            # arm eases out/back gradually and switching joints stays smooth.
            for dof in c["arm_dofs"]:
                nudge = float(qw[7 + dof] - q[7 + dof])
                prev = self._arm_nudge_smoothed.get(dof, 0.0)
                smoothed = self.arm_smooth_alpha * nudge + (1.0 - self.arm_smooth_alpha) * prev
                self._arm_nudge_smoothed[dof] = smoothed
                q[7 + dof] = float(q[7 + dof]) + smoothed
        d.qpos[:] = q
        self._mj.mj_forward(m, d)
        return q

    def _compute_sole_offset(self, clearance=0.005):
        try:
            import mujoco as mj
            m = mj.MjModel.from_xml_path(self.xml_file)
            d = mj.MjData(m)
            d.qpos[:] = 0.0
            if m.nq >= 7:
                d.qpos[3] = 1.0  # identity quaternion (w=1)
            mj.mj_forward(m, d)
            lowest_frame = float(np.min(d.xpos[1:, 2]))  # exclude world body
            lowest_vert = float("inf")
            for g in range(m.ngeom):
                if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                    continue
                if m.geom_group[g] != 0:
                    continue
                mid = m.geom_dataid[g]
                start = m.mesh_vertadr[mid]
                count = m.mesh_vertnum[mid]
                verts = m.mesh_vert[start:start + count]
                w = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                lowest_vert = min(lowest_vert, float(np.min(w[:, 2])))
            if lowest_vert == float("inf"):
                return 0.0
            return max(0.0, float(lowest_frame - lowest_vert) + float(clearance))
        except Exception as e:
            print(f"[Stream] Warning: could not compute foot sole offset ({e}); using 0.")
            return 0.0

    def _set_kinematics_device(self, device):
        self.device = device
        self.kinematics_model = KinematicsModel(
            self.xml_file,
            device=device,
            root_body_name=self.root_body_name,
        )

    def _forward_kinematics_for_height(self, q):
        root_pos = torch.from_numpy(q[:3][None]).to(self.device, dtype=torch.float32)
        root_rot_xyzw = torch.from_numpy(q[3:7][[1, 2, 3, 0]][None]).to(self.device, dtype=torch.float32)
        dof_pos = torch.from_numpy(q[7:][None]).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            body_pos, _ = self.kinematics_model.forward_kinematics(root_pos, root_rot_xyzw, dof_pos)
            return torch.min(body_pos[..., 2]).item()

    def process(self, qpos):
        q = np.asarray(qpos, dtype=np.float32).copy()

        if self.prev_qpos is not None and self.smooth_alpha < 0.999:
            q[:3] = _exp_smooth(q[:3], self.prev_qpos[:3], self.smooth_alpha)

            quat = q[3:7].copy()
            prev_quat = self.prev_qpos[3:7]
            if np.dot(quat, prev_quat) < 0.0:
                quat = -quat
            quat = _exp_smooth(quat, prev_quat, self.smooth_alpha)
            quat /= (np.linalg.norm(quat) + 1e-8)
            q[3:7] = quat

            q[7:] = _exp_smooth(q[7:], self.prev_qpos[7:], self.smooth_alpha)

        if self.height_adjust:
            try:
                lowest_height = self._forward_kinematics_for_height(q)
            except RuntimeError as e:
                if self.device != "cpu" and _is_nvrtc_arch_error(e):
                    print("[Stream] Warning: CUDA NVRTC architecture error in GMR postprocessor; fallback to CPU.")
                    self._set_kinematics_device("cpu")
                    lowest_height = self._forward_kinematics_for_height(q)
                else:
                    raise
            q[2] -= (lowest_height - self.sole_offset)

            # After grounding, flatten each planted foot so its sole is parallel to
            # the ground (fixes heel-hover / toe-dig from a toe-down WHAM estimate).
            self._flatten_stance_feet(q)

            # Keep the forearms clear of the thighs (fixes hand/forearm clipping the
            # hip during sit->stand when the arms hang by the side).
            self._avoid_arm_thigh_collision(q)

        if self.root_origin_offset:
            if self.xy_origin is None:
                self.xy_origin = q[:2].copy()
            q[:2] -= self.xy_origin

        if self.is_x02lite and q.shape[0] > self.X02LITE_R_ELBOW_QPOS_INDEX:
            q[self.X02LITE_R_ELBOW_QPOS_INDEX] = min(
                q[self.X02LITE_R_ELBOW_QPOS_INDEX],
                self.X02LITE_R_ELBOW_SOFT_MAX_RAD,
            )

        self.prev_qpos = q.copy()
        return q


def write_motion_pkl(save_path, qpos_seq, fps, xml_file, root_body_name=None, torch_device="auto"):
    qpos_seq = np.asarray(qpos_seq, dtype=np.float32)
    root_pos = qpos_seq[:, :3].copy()
    root_rot = qpos_seq[:, 3:7][:, [1, 2, 3, 0]].copy()  # wxyz -> xyzw
    dof_pos = qpos_seq[:, 7:].copy()

    device = _resolve_torch_device(torch_device)

    def _compute_local_body_pos(dev):
        model = KinematicsModel(xml_file, device=dev, root_body_name=root_body_name)
        with torch.no_grad():
            fk_root_pos = torch.zeros((dof_pos.shape[0], 3), device=dev)
            fk_root_rot = torch.zeros((dof_pos.shape[0], 4), device=dev)
            fk_root_rot[:, -1] = 1.0
            body_pos, _ = model.forward_kinematics(
                fk_root_pos,
                fk_root_rot,
                torch.from_numpy(dof_pos).to(device=dev, dtype=torch.float32),
            )
            return body_pos.detach().cpu().numpy(), model

    try:
        local_body_pos, kinematics_model = _compute_local_body_pos(device)
    except RuntimeError as e:
        if device != "cpu" and _is_nvrtc_arch_error(e):
            print("[Stream] Warning: CUDA NVRTC architecture error while writing motion PKL; fallback to CPU.")
            device = "cpu"
            local_body_pos, kinematics_model = _compute_local_body_pos(device)
        else:
            raise

    motion_data = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": kinematics_model.body_names,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    import pickle

    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_npz_dir", type=str, required=True, help="Directory containing stream artifacts (tail/chunk).")
    parser.add_argument(
        "--stream_mode",
        choices=["chunk", "tail", "hybrid"],
        default="tail",
        help="Stream transport mode. Default is tail (append-only stream_tail.pkl).",
    )
    parser.add_argument(
        "--stream_tail_path",
        type=str,
        default=None,
        help="Path to append-only tail stream file. Defaults to <stream_npz_dir>/stream_tail.pkl.",
    )
    parser.add_argument(
        "--torch_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device used by GMR postprocessing and FK routines.",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "x02lite",
            "openloong",
            "lite_11_v1",
            "tienkung",
            "fourier_gr3",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--robot_path",
        type=str,
        default=None,
        help="Optional robot XML path. If set, overrides built-in XML for the selected --robot.",
    )
    parser.add_argument("--coord_fix", choices=["auto", "none", "yup_to_zup"], default="auto")
    parser.add_argument("--save_path", type=str, default="pkl_outputs/live_motion.pkl")
    parser.add_argument("--csv_path", type=str, default="pkl_outputs/csv/live_motion.csv")
    parser.add_argument("--record_gmrvideo", action="store_true")
    parser.add_argument("--record_video", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--video_path", type=str, default="videos/live_stream_robot.mp4")
    parser.add_argument(
        "--viewer_warmup_frames",
        type=int,
        default=0,
        help="Skip rendering first N frames in viewer while still writing csv/pkl.",
    )
    parser.add_argument(
        "--camera_follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Follow robot base with viewer camera.",
    )
    parser.add_argument(
        "--camera_lookat_height_offset",
        type=float,
        default=0.45,
        help="Raise camera look-at point by this many meters.",
    )
    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=12.0,
        help="Viewer camera elevation angle in degrees.",
    )
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=0.85,
        help="Scale factor for default robot camera distance.",
    )
    parser.add_argument(
        "--camera_azimuth",
        type=float,
        default=None,
        help="Optional fixed camera azimuth angle in degrees.",
    )

    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--height_adjust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--root_origin_offset", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--poll_interval", type=float, default=0.05)
    parser.add_argument("--idle_timeout", type=float, default=0.0, help="Exit after idle timeout (seconds). 0 disables timeout.")
    parser.add_argument("--done_flag_name", type=str, default="stream_done.flag")
    parser.add_argument(
        "--ready_flag_path",
        type=str,
        default=None,
        help="Optional file path to touch when consumer startup is ready.",
    )
    parser.add_argument(
        "--viewer_drop_old_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop oldest buffered viewer frames when lagging to keep playback responsive.",
    )
    parser.add_argument(
        "--viewer_ready_timeout_sec",
        type=float,
        default=8.0,
        help="Max wait time for async viewer initialization before continuing.",
    )
    parser.add_argument(
        "--viewer_thread_join_timeout_sec",
        type=float,
        default=10.0,
        help="Timeout when waiting async viewer thread to flush queue and exit.",
    )
    parser.add_argument(
        "--done_grace_sec",
        type=float,
        default=2.0,
        help="Extra wait time after done flag appears before exiting, to avoid missing late chunk files.",
    )
    parser.add_argument(
        "--max_chunk_retry",
        type=int,
        default=120,
        help="Maximum retries per chunk before exiting with an error to avoid infinite loops.",
    )

    args = parser.parse_args()
    stream_mode = str(args.stream_mode).strip().lower()
    if stream_mode not in ("chunk", "tail", "hybrid"):
        print(f"[Stream] Warning: unknown stream_mode={stream_mode}, fallback to tail.")
        stream_mode = "tail"
    if stream_mode == "hybrid":
        print("[Stream] Warning: hybrid mode is not supported in consumer yet, fallback to tail.")
        stream_mode = "tail"

    record_gmrvideo = bool(args.record_gmrvideo or args.record_video)
    robot_path = args.robot_path if args.robot_path is not None and str(args.robot_path).strip() != "" else None
    viewer_async = True
    viewer_rate_limit = False
    viewer_startup_buffer_sec = 0.2
    viewer_max_buffer_sec = 0.6

    gmr_init_params = set(inspect.signature(GMR.__init__).parameters.keys())
    viewer_init_params = set(inspect.signature(RobotMotionViewer.__init__).parameters.keys())
    gmr_supports_robot_path = "robot_path" in gmr_init_params

    if robot_path is not None and not gmr_supports_robot_path:
        print(
            "[Stream] Fatal: current GeneralMotionRetargeting does not support --robot_path. "
            "Please update the code/environment or unset ROBOT_PATH."
        )
        sys.exit(2)

    stream_npz_dir = os.path.abspath(args.stream_npz_dir)
    done_flag = os.path.join(stream_npz_dir, args.done_flag_name)
    stream_tail_path = os.path.abspath(args.stream_tail_path) if args.stream_tail_path else os.path.join(stream_npz_dir, "stream_tail.pkl")
    os.makedirs(stream_npz_dir, exist_ok=True)

    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    csv_file = open(args.csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    processed = set()
    qpos_history = []
    aligned_fps_holder = [30.0]

    state = {
        "retarget": None,
        "postprocessor": None,
        "viewer": None,
        "tail_body_model": None,
        "tail_joint_names": None,
        "tail_parents": None,
    }
    viewer_init_attempted = [False]
    tail_coord_fix_logged = [False]
    viewer_queue = deque()
    viewer_started = [False]
    viewer_drop_count = [0]
    viewer_queue_lock = threading.Lock()
    viewer_stop_event = threading.Event()
    viewer_ready_event = threading.Event()
    viewer_thread = [None]
    viewer_failed = [False]
    viewer_error = [None]

    def _align_tail_betas(raw_betas, body_model):
        betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
        target_dim = int(getattr(body_model, "num_betas", betas.shape[0]))
        if betas.shape[0] < target_dim:
            betas = np.pad(betas, (0, target_dim - betas.shape[0]))
        elif betas.shape[0] > target_dim:
            betas = betas[:target_dim]
        return betas

    def _tail_apply_yup_to_zup(root_orient, trans):
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        rot_fix = R.from_matrix(rotation_matrix)
        root_orient = (rot_fix * R.from_rotvec(root_orient)).as_rotvec().astype(np.float32)
        trans = trans @ rotation_matrix.T
        return root_orient, trans

    def tail_params_to_smplx_frame(params):
        if state["tail_body_model"] is None:
            body_model = smplx.create(
                str(SMPLX_FOLDER),
                "smplx",
                gender="neutral",
                use_pca=False,
            )
            state["tail_body_model"] = body_model
            state["tail_joint_names"] = JOINT_NAMES[: len(body_model.parents)]
            state["tail_parents"] = body_model.parents

        body_model = state["tail_body_model"]
        joint_names = state["tail_joint_names"]
        parents = state["tail_parents"]

        body_pose = np.asarray(params["body_pose"], dtype=np.float32).reshape(1, 63)
        root_orient = np.asarray(params["global_orient_global"], dtype=np.float32).reshape(1, 3)
        trans = np.asarray(params["transl_global"], dtype=np.float32).reshape(1, 3)

        if args.coord_fix in ("auto", "yup_to_zup"):
            root_orient, trans = _tail_apply_yup_to_zup(root_orient, trans)
            if not tail_coord_fix_logged[0]:
                print("[SMPL] Applied coordinate fix: y-up -> z-up")
                tail_coord_fix_logged[0] = True

        betas = _align_tail_betas(params["betas"], body_model)

        with torch.no_grad():
            smplx_output = body_model(
                betas=torch.from_numpy(betas).float().view(1, -1),
                global_orient=torch.from_numpy(root_orient).float(),
                body_pose=torch.from_numpy(body_pose).float(),
                transl=torch.from_numpy(trans).float(),
                left_hand_pose=torch.zeros(1, 45).float(),
                right_hand_pose=torch.zeros(1, 45).float(),
                jaw_pose=torch.zeros(1, 3).float(),
                leye_pose=torch.zeros(1, 3).float(),
                reye_pose=torch.zeros(1, 3).float(),
                return_full_pose=True,
            )

        single_global_orient = smplx_output.global_orient[0].detach().cpu().numpy()
        single_full_body_pose = smplx_output.full_pose[0].detach().cpu().numpy().reshape(-1, 3)
        single_joints = smplx_output.joints[0].detach().cpu().numpy()

        result = {}
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(single_full_body_pose[i].squeeze())
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat())

        return result, betas

    def _build_viewer_kwargs(motion_fps):
        viewer_kwargs = dict(
            robot_type=args.robot,
            motion_fps=float(max(1.0, motion_fps)),
            transparent_robot=0,
            record_video=True,
            video_path=args.video_path,
        )
        if robot_path is not None and "robot_path" in viewer_init_params:
            viewer_kwargs["robot_path"] = robot_path
        if "camera_follow" in viewer_init_params:
            viewer_kwargs["camera_follow"] = args.camera_follow
        if "camera_lookat_height_offset" in viewer_init_params:
            viewer_kwargs["camera_lookat_height_offset"] = args.camera_lookat_height_offset
        if "camera_elevation" in viewer_init_params:
            viewer_kwargs["camera_elevation"] = args.camera_elevation
        if "camera_distance_scale" in viewer_init_params:
            viewer_kwargs["camera_distance_scale"] = args.camera_distance_scale
        if "camera_azimuth" in viewer_init_params:
            viewer_kwargs["camera_azimuth"] = args.camera_azimuth
        return viewer_kwargs

    def _viewer_target_fps():
        return float(max(1.0, aligned_fps_holder[0]))

    def _viewer_required_frames():
        return max(0, int(round(max(0.0, viewer_startup_buffer_sec) * _viewer_target_fps())))

    def _viewer_worker_loop(initial_motion_fps):
        viewer = None
        try:
            viewer = RobotMotionViewer(**_build_viewer_kwargs(initial_motion_fps))
            state["viewer"] = viewer
            print(
                "[Stream] MuJoCo viewer enabled in async thread "
                f"(DISPLAY={os.environ.get('DISPLAY', '<unset>')}, rate_limit={int(bool(viewer_rate_limit))})"
            )
        except Exception as e:
            viewer_failed[0] = True
            viewer_error[0] = str(e)
            state["viewer"] = None
            print(
                "[Stream] Warning: failed to initialize MuJoCo viewer "
                f"(DISPLAY={os.environ.get('DISPLAY', '<unset>')}): {e}. "
                "Continue without on-screen window. "
                "If you expect a window: ensure DISPLAY is valid, run `xhost +local:docker` for Docker, "
                "and avoid USE_XVFB_GMR=1."
            )
            viewer_ready_event.set()
            return

        viewer_ready_event.set()

        try:
            while True:
                qpos = None
                with viewer_queue_lock:
                    qlen = len(viewer_queue)
                    required = _viewer_required_frames()
                    if not viewer_started[0] and (qlen >= required or (viewer_stop_event.is_set() and qlen > 0)):
                        viewer_started[0] = True
                        print(
                            f"[Stream] Viewer playback starts (buffer={qlen}, required={required}, "
                            f"drop_old={int(bool(args.viewer_drop_old_frames))}, async=1)"
                        )
                    if viewer_started[0] and qlen > 0:
                        qpos = viewer_queue.popleft()

                if qpos is None:
                    if viewer_stop_event.is_set():
                        with viewer_queue_lock:
                            if len(viewer_queue) == 0:
                                break
                    time.sleep(0.001)
                    continue

                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=None,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    rate_limit=bool(viewer_rate_limit),
                    follow_camera=args.camera_follow,
                )
        except Exception as e:
            viewer_failed[0] = True
            viewer_error[0] = str(e)
            print(f"[Stream] Warning: async viewer loop stopped due to error: {e}")
        finally:
            try:
                viewer.close()
            except Exception as e:
                print(f"[Stream] Warning: closing viewer failed: {e}")
            state["viewer"] = None

    def init_viewer_if_needed(motion_fps):
        if not record_gmrvideo:
            return
        if state["viewer"] is not None:
            return
        if viewer_thread[0] is not None and viewer_thread[0].is_alive():
            return
        if viewer_init_attempted[0]:
            return

        viewer_init_attempted[0] = True
        viewer_failed[0] = False
        viewer_error[0] = None
        viewer_started[0] = False
        viewer_ready_event.clear()
        viewer_stop_event.clear()

        initial_motion_fps = float(max(1.0, motion_fps))
        if viewer_async:
            t = threading.Thread(
                target=_viewer_worker_loop,
                args=(initial_motion_fps,),
                daemon=True,
                name="gmr-viewer-thread",
            )
            viewer_thread[0] = t
            t.start()

            ready_timeout = max(1.0, float(args.viewer_ready_timeout_sec))
            if not viewer_ready_event.wait(timeout=ready_timeout):
                viewer_failed[0] = True
                viewer_error[0] = f"viewer init timeout ({ready_timeout:.1f}s)"
                print(
                    "[Stream] Warning: async MuJoCo viewer init timeout; "
                    "continue pipeline without waiting for on-screen viewer."
                )
            return

        try:
            state["viewer"] = RobotMotionViewer(**_build_viewer_kwargs(initial_motion_fps))
            print(
                "[Stream] MuJoCo viewer enabled in main thread "
                f"(DISPLAY={os.environ.get('DISPLAY', '<unset>')}, rate_limit={int(bool(viewer_rate_limit))})"
            )
        except Exception as e:
            state["viewer"] = None
            viewer_failed[0] = True
            viewer_error[0] = str(e)
            print(
                "[Stream] Warning: failed to initialize MuJoCo viewer "
                f"(DISPLAY={os.environ.get('DISPLAY', '<unset>')}): {e}. "
                "Continue without on-screen window. "
                "If you expect a window: ensure DISPLAY is valid, run `xhost +local:docker` for Docker, "
                "and avoid USE_XVFB_GMR=1."
            )
        finally:
            viewer_ready_event.set()

    def init_retarget_if_needed(actual_human_height=None):
        if state["retarget"] is not None:
            return

        retarget_kwargs = dict(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=args.robot,
        )
        if robot_path is not None and gmr_supports_robot_path:
            retarget_kwargs["robot_path"] = robot_path
        state["retarget"] = GMR(**retarget_kwargs)

        state["postprocessor"] = OnlineQposPostprocessor(
            state["retarget"].xml_file,
            root_body_name=state["retarget"].robot_root_name,
            smooth_alpha=max(0.0, min(1.0, args.smooth_alpha)),
            height_adjust=args.height_adjust,
            root_origin_offset=args.root_origin_offset,
            torch_device=args.torch_device,
        )

        init_viewer_if_needed(aligned_fps_holder[0])

    def enqueue_viewer_qpos(qpos):
        if not record_gmrvideo:
            return
        if viewer_failed[0]:
            return
        if state["viewer"] is None and (viewer_thread[0] is None or not viewer_thread[0].is_alive()):
            return
        qpos_copy = np.asarray(qpos, dtype=np.float32).copy()
        with viewer_queue_lock:
            viewer_queue.append(qpos_copy)
            if args.viewer_drop_old_frames:
                max_frames = max(1, int(round(max(0.1, viewer_max_buffer_sec) * _viewer_target_fps())))
                overflow = len(viewer_queue) - max_frames
                if overflow > 0:
                    for _ in range(overflow):
                        viewer_queue.popleft()
                    viewer_drop_count[0] += int(overflow)

    def process_chunk(chunk_path):
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            chunk_path, SMPLX_FOLDER, coord_fix=args.coord_fix
        )
        smplx_frames, aligned_fps_local = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
        aligned_fps_holder[0] = float(aligned_fps_local)

        if state["retarget"] is None:
            init_retarget_if_needed(actual_human_height=actual_human_height)

        for frame in smplx_frames:
            qpos = _extract_qpos(state["retarget"].retarget(frame))
            qpos = state["postprocessor"].process(qpos)
            qpos_history.append(qpos.copy())

            # CSV uses root_rot in xyzw order.
            row = np.concatenate([qpos[:3], qpos[3:7][[1, 2, 3, 0]], qpos[7:]], axis=0)
            csv_writer.writerow(row.tolist())

            if record_gmrvideo and (len(qpos_history) - 1) >= max(0, int(args.viewer_warmup_frames)):
                enqueue_viewer_qpos(qpos)

        csv_file.flush()

    def process_tail_record(record):
        frame_id = int(record.get("frame_id", -1))
        params = record.get("params")
        if frame_id < 0 or params is None:
            return None

        smplx_frame, betas = tail_params_to_smplx_frame(params)

        if state["retarget"] is None:
            human_height = None
            if betas is not None and betas.size > 0:
                human_height = 1.66 + 0.1 * float(betas[0])
            init_retarget_if_needed(actual_human_height=human_height)

        qpos = _extract_qpos(state["retarget"].retarget(smplx_frame))
        qpos = state["postprocessor"].process(qpos)
        qpos_history.append(qpos.copy())

        row = np.concatenate([qpos[:3], qpos[3:7][[1, 2, 3, 0]], qpos[7:]], axis=0)
        csv_writer.writerow(row.tolist())

        if record_gmrvideo and (len(qpos_history) - 1) >= max(0, int(args.viewer_warmup_frames)):
            enqueue_viewer_qpos(qpos)

        csv_file.flush()
        return frame_id

    print(f"[Stream] Watching {stream_npz_dir}")
    print(f"[Stream] Mode: {stream_mode}")
    if stream_mode in ("tail", "hybrid"):
        print(f"[Stream] Tail path: {stream_tail_path}")
    print(f"[Stream] CSV output: {args.csv_path}")

    # Initialize MuJoCo viewer before signaling readiness so upstream can
    # wait until the GMR visualization path is actually ready.
    init_viewer_if_needed(aligned_fps_holder[0])

    if args.ready_flag_path:
        try:
            ready_flag_path = os.path.abspath(args.ready_flag_path)
            ready_dir = os.path.dirname(ready_flag_path)
            if ready_dir:
                os.makedirs(ready_dir, exist_ok=True)
            with open(ready_flag_path, "w") as f:
                f.write(f"pid={os.getpid()} ts={time.time()}\n")
            print(f"[Stream] Ready flag created: {ready_flag_path}")
        except Exception as e:
            print(f"[Stream] Warning: failed to create ready flag {args.ready_flag_path}: {e}")

    last_activity = time.time()
    fatal_error = None
    done_seen_at = None
    done_expected_chunks = None
    chunk_retry_count = {}
    tail_file = None
    tail_done_seen = False
    while True:
        progressed = False

        if stream_mode == "chunk":
            chunk_names = sorted(
                [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
            )
            new_chunk_names = [n for n in chunk_names if n not in processed]

            if len(new_chunk_names) > 0:
                done_seen_at = None
                for name in new_chunk_names:
                    chunk_path = os.path.join(stream_npz_dir, name)
                    try:
                        process_chunk(chunk_path)
                    except Exception as e:
                        retry = int(chunk_retry_count.get(name, 0)) + 1
                        chunk_retry_count[name] = retry

                        err_msg = str(e)
                        if "unexpected keyword argument 'robot_path'" in err_msg:
                            fatal_error = (
                                "GeneralMotionRetargeting in current runtime does not accept robot_path. "
                                "Please update environment or unset ROBOT_PATH."
                            )
                            print(f"[Stream] Fatal on {name}: {fatal_error}")
                            break

                        if retry <= 3 or retry % 20 == 0:
                            print(f"[Stream] Waiting for valid chunk {name} (retry={retry}): {e}")

                        if retry >= max(1, int(args.max_chunk_retry)):
                            fatal_error = f"Chunk {name} failed {retry} times. Last error: {e}"
                            print(f"[Stream] Fatal on {name}: {fatal_error}")
                            break
                        continue

                    processed.add(name)
                    chunk_retry_count.pop(name, None)
                    print(f"[Stream] Processed {name} (total chunks: {len(processed)})")
                    last_activity = time.time()
                    progressed = True

                if fatal_error is not None:
                    break

            if not progressed:
                if os.path.exists(done_flag):
                    if done_expected_chunks is None:
                        try:
                            with open(done_flag, "r") as f:
                                done_expected_chunks = int((f.read() or "").strip())
                                print(f"[Stream] Done flag detected with expected chunks={done_expected_chunks}")
                        except Exception:
                            done_expected_chunks = None

                    if done_expected_chunks is not None and len(processed) >= done_expected_chunks:
                        final_chunk_names = sorted(
                            [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
                        )
                        final_new_chunks = [n for n in final_chunk_names if n not in processed]
                        if len(final_new_chunks) == 0:
                            break

                    if done_expected_chunks is not None:
                        time.sleep(max(0.01, args.poll_interval))
                        continue

                    if done_seen_at is None:
                        done_seen_at = time.time()
                    elif (time.time() - done_seen_at) >= max(0.0, args.done_grace_sec):
                        final_chunk_names = sorted(
                            [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
                        )
                        final_new_chunks = [n for n in final_chunk_names if n not in processed]
                        if len(final_new_chunks) == 0:
                            break
                else:
                    done_seen_at = None

        else:  # tail mode
            if tail_file is None:
                if os.path.exists(stream_tail_path):
                    try:
                        tail_file = open(stream_tail_path, "rb")
                    except Exception as e:
                        fatal_error = f"Failed to open tail stream {stream_tail_path}: {e}"
                        print(f"[Stream] Fatal: {fatal_error}")
                        break
                else:
                    if args.idle_timeout > 0 and (time.time() - last_activity) > args.idle_timeout:
                        print(f"[Stream] Idle timeout ({args.idle_timeout}s), exiting.")
                        break
                    time.sleep(max(0.01, args.poll_interval))
                    continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=DeprecationWarning,
                        message=r"numpy\.core\.numeric is deprecated.*",
                    )
                    record = pickle.load(tail_file)
            except EOFError:
                if tail_done_seen:
                    break
            except Exception as e:
                fatal_error = f"Tail stream read failed: {e}"
                print(f"[Stream] Fatal: {fatal_error}")
                break
            else:
                if isinstance(record, dict) and bool(record.get("done", False)):
                    tail_done_seen = True
                elif isinstance(record, dict):
                    try:
                        frame_id = process_tail_record(record)
                    except Exception as e:
                        fatal_error = f"Tail record process failed: {e}"
                        print(f"[Stream] Fatal: {fatal_error}")
                        break
                    if frame_id is not None:
                        if frame_id not in processed:
                            processed.add(frame_id)
                            if len(processed) % 20 == 0:
                                print(f"[Stream] Processed tail frames: {len(processed)}")
                        last_activity = time.time()
                        progressed = True

        if fatal_error is not None:
            break

        if not progressed:
            if args.idle_timeout > 0 and (time.time() - last_activity) > args.idle_timeout:
                print(f"[Stream] Idle timeout ({args.idle_timeout}s), exiting.")
                break
            time.sleep(max(0.01, args.poll_interval))

    if tail_file is not None:
        tail_file.close()

    csv_file.flush()
    csv_file.close()

    if stream_mode == "chunk":
        print(
            f"[Stream] Summary: processed_chunks={len(processed)} "
            f"expected_chunks={done_expected_chunks if done_expected_chunks is not None else -1} "
            f"qpos_frames={len(qpos_history)}"
        )
    else:
        print(f"[Stream] Summary: processed_tail_frames={len(processed)} qpos_frames={len(qpos_history)}")

    if record_gmrvideo:
        viewer_stop_event.set()
        if viewer_async and viewer_thread[0] is not None:
            join_timeout = max(1.0, float(args.viewer_thread_join_timeout_sec))
            viewer_thread[0].join(timeout=join_timeout)
            if viewer_thread[0].is_alive():
                with viewer_queue_lock:
                    remaining = len(viewer_queue)
                print(
                    "[Stream] Warning: async viewer thread did not exit before timeout; "
                    f"remaining_queue={remaining}"
                )
        elif state["viewer"] is not None:
            while True:
                qpos = None
                with viewer_queue_lock:
                    qlen = len(viewer_queue)
                    required = _viewer_required_frames()
                    if not viewer_started[0] and (qlen >= required or qlen > 0):
                        viewer_started[0] = True
                    if viewer_started[0] and qlen > 0:
                        qpos = viewer_queue.popleft()
                if qpos is None:
                    break
                state["viewer"].step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=None,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    rate_limit=bool(viewer_rate_limit),
                    follow_camera=args.camera_follow,
                )
            state["viewer"].close()
            state["viewer"] = None

        with viewer_queue_lock:
            remaining = len(viewer_queue)
        print(
            f"[Stream] Viewer playback stopped. dropped_frames={viewer_drop_count[0]} "
            f"remaining_queue={remaining} async={int(bool(viewer_async))}"
        )
        if viewer_failed[0] and viewer_error[0] is not None:
            print(f"[Stream] Viewer warning: {viewer_error[0]}")

    if len(qpos_history) > 0 and args.save_path:
        write_motion_pkl(
            args.save_path,
            qpos_history,
            aligned_fps_holder[0],
            state["retarget"].xml_file,
            root_body_name=state["retarget"].robot_root_name,
            torch_device=args.torch_device,
        )
        print(f"[Stream] Saved motion pkl: {args.save_path}")
    else:
        print("[Stream] No qpos generated; skip pkl save.")

    if fatal_error is not None:
        print(f"[Stream] Exit with error: {fatal_error}")
        sys.exit(2)

    print("[Stream] Done.")

"""Verify the two WHAM-network optimizations are numerically equivalent, and measure speed.

1. reset_root_velocity: feet under trajectory root = rot_delta @ feet  (vs a full LBS pass)
2. modules.py  x[:, [i]] -> x[:, i:i+1]  slice change (no numerical change expected)

Run:  D:\\Anaconda\\envs\\wham_gmr\\python.exe scripts/verify_wham_opt.py
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.utils.kp_utils import root_centering
from lib.data.utils.normalizer import Normalizer
from lib.utils.imutils import compute_cam_intrinsics
import lib.utils.transforms as pt_transforms
from lib.models.layers.utils import reset_root_velocity


def old_reset_root_velocity(smpl, output, stationary, pred_ori, pred_vel, thr=0.7):
    """The pre-optimization implementation (full second LBS forward)."""
    b, f = pred_vel.shape[:2]
    stationary_mask = (stationary.clone().detach() > thr).unsqueeze(-1).float()
    poses_root = pt_transforms.rotation_6d_to_matrix(pred_ori.clone().detach())
    vel_world = (poses_root[:, 1:] @ pred_vel.clone().detach().unsqueeze(-1)).squeeze(-1)
    o = smpl.get_output(body_pose=output.body_pose.clone().detach(),
                        global_orient=poses_root[:, 1:].reshape(-1, 1, 3, 3),
                        betas=output.betas.clone().detach(),
                        pose2rot=False)
    feet = o.feet.reshape(b, f, 4, 3)
    feet_vel = feet[:, 1:] - feet[:, :-1] + vel_world[:, 1:].unsqueeze(-2)
    feet_vel = torch.cat((torch.zeros_like(feet_vel[:, :1]), feet_vel), dim=1)
    stationary_vel = feet_vel * stationary_mask
    del_vel = stationary_vel.sum(dim=2) / ((stationary_vel != 0).sum(dim=2) + 1e-4)
    vel_world_update = vel_world - del_vel
    vel_root = (poses_root[:, 1:].mT @ vel_world_update.unsqueeze(-1)).squeeze(-1)
    return vel_root


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(ROOT, "configs", "yamls", "demo.yaml"))
    device = cfg.DEVICE.lower()

    print("[verify] loading SMPL + WHAM network ...", flush=True)
    smpl = build_body_model("cpu")
    network = build_network(cfg, smpl)
    network.eval()
    smpl = network.smpl  # the copy that lives on `device`

    keypoints_normalizer = Normalizer(cfg)
    F = 16

    feature_t = torch.randn(1, F, 1024, device=device)
    init_data = {
        "init_global_orient": torch.eye(3, device=device).reshape(1, 1, 3, 3),
        "init_body_pose": torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, 23, 1, 1),
        "init_betas": torch.zeros(1, 10, device=device),
    }
    kp2d = np.random.rand(F, 17, 3).astype(np.float32)
    kp2d[..., 2] = 1.0
    kp2d_t = torch.from_numpy(kp2d).float()
    bbox = np.array([128.0, 128.0, 1.0], dtype=np.float32)
    bbox_t = torch.from_numpy(np.tile(bbox, (F, 1))).float()
    mask = (kp2d_t[..., -1] < 0.3).unsqueeze(0).to(device)
    res = torch.tensor([256.0, 256.0]).float()
    intrinsics = compute_cam_intrinsics(res)
    intrinsics_batch = intrinsics.unsqueeze(0).to(device)
    res_batch = res.unsqueeze(0).to(device)
    norm_kp2d, _ = keypoints_normalizer(kp2d_t[..., :-1].clone(), res, intrinsics, 224, 224, bbox_t)
    norm_kp2d = norm_kp2d.unsqueeze(0).to(device)

    with torch.inference_mode():
        init_output = smpl.get_output(
            global_orient=init_data["init_global_orient"],
            body_pose=init_data["init_body_pose"],
            betas=init_data["init_betas"],
            pose2rot=False, return_full_pose=True)
    init_kp3d = root_centering(init_output.joints[:, :17], "coco")
    init_kp = torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0, 0].reshape(1, -1)), dim=-1).unsqueeze(0).to(device)
    init_smpl = pt_transforms.matrix_to_rotation_6d(init_output.full_pose).unsqueeze(0).to(device)
    init_root = pt_transforms.matrix_to_rotation_6d(init_output.global_orient).to(device)
    inits = (init_kp, init_smpl)
    cam_angvel = torch.zeros((1, F, 6), device=device)
    bbox_b = bbox_t.unsqueeze(0).to(device)

    def run_network():
        with torch.inference_mode():
            return network(norm_kp2d, inits, feature_t, mask=mask, init_root=init_root,
                           cam_angvel=cam_angvel, return_y_up=True,
                           cam_intrinsics=intrinsics_batch, bbox=bbox_b, res=res_batch,
                           refine_traj=False, states=None)

    out = run_network()  # warmup

    output = network.output
    pred_root, pred_vel, pred_contact = network.pred_root, network.pred_vel, network.pred_contact
    b, f = pred_vel.shape[:2]

    # --- 1. feet equivalence (rotation vs full LBS) ---
    poses_root = pt_transforms.rotation_6d_to_matrix(pred_root.clone().detach())
    with torch.inference_mode():
        old_out = smpl.get_output(body_pose=output.body_pose.clone().detach(),
                                  global_orient=poses_root[:, 1:].reshape(-1, 1, 3, 3),
                                  betas=output.betas.clone().detach(), pose2rot=False)
    old_feet = old_out.feet.reshape(b, f, 4, 3)

    r1 = output.global_orient.detach().reshape(b, f, 3, 3)
    rot_delta = poses_root[:, 1:] @ r1.transpose(-1, -2)
    new_feet = output.feet.detach().reshape(b, f, 4, 3) @ rot_delta.transpose(-1, -2)

    feet_diff = (old_feet - new_feet).abs().max().item()
    print(f"[verify] feet  max abs diff (full-LBS vs rotation): {feet_diff:.3e}", flush=True)

    # --- 2. full vel_root equivalence (new impl vs inlined old impl) ---
    with torch.inference_mode():
        old_vr = old_reset_root_velocity(smpl, output, pred_contact, pred_root, pred_vel, thr=0.5)
    new_vr = out["vel_root_reset"]
    vr_diff = (old_vr - new_vr).abs().max().item()
    print(f"[verify] vel_root  max abs diff (old vs new):      {vr_diff:.3e}", flush=True)

    ok = feet_diff < 1e-3 and vr_diff < 1e-3
    print(f"[verify] {'PASS' if ok else 'FAIL'}  (tolerance 1e-3)", flush=True)

    # --- 3. timing ---
    torch.cuda.synchronize()
    N = 30
    t0 = time.time()
    for _ in range(N):
        run_network()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N * 1000
    print(f"[verify] network forward avg: {dt:.2f} ms  (F={F}, refine_traj=False)", flush=True)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

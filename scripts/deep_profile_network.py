"""Pinpoint the source of aten::copy_ / aten::_local_scalar_dense in the WHAM network.

Skips the ViT extractor & detector (not needed) and uses the legacy autograd
profiler with with_stack=True, which records CPU launch stacks for CUDA ops in
torch 1.11 (the new torch.profiler needs use_cpu=True there).

Run:  python scripts/deep_profile_network.py
"""
import os
import sys

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


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(ROOT, "configs", "yamls", "demo.yaml"))
    device = cfg.DEVICE.lower()

    print("[deep] loading SMPL + WHAM network ...")
    smpl = build_body_model("cpu")          # moved to cuda by build_network .to(device)
    network = build_network(cfg, smpl)
    network.eval()

    keypoints_normalizer = Normalizer(cfg)
    F = 16

    # --- synthetic feature (1, F, 1024) + init params (standard HMR2 shapes) ---
    feature_t = torch.randn(1, F, 1024, device=device)

    # HMR2 predict_init outputs rotation MATRICES (pose2rot=False), not axis-angle:
    #   init_global_orient (1,1,3,3), init_body_pose (1,23,3,3), init_betas (1,10)
    init_data = {
        "init_global_orient": torch.eye(3, device=device).reshape(1, 1, 3, 3),
        "init_body_pose": torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, 23, 1, 1),
        "init_betas": torch.zeros(1, 10, device=device),
    }

    # --- mirror the input building in scripts/profile_wham.py ---
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

    norm_kp2d, _ = keypoints_normalizer(
        kp2d_t[..., :-1].clone(), res, intrinsics, 224, 224, bbox_t
    )
    norm_kp2d = norm_kp2d.unsqueeze(0).to(device)

    with torch.inference_mode():
        init_output = smpl.get_output(
            global_orient=init_data["init_global_orient"],
            body_pose=init_data["init_body_pose"],
            betas=init_data["init_betas"],
            pose2rot=False,
            return_full_pose=True,
        )
    init_kp3d = root_centering(init_output.joints[:, :17], "coco")
    init_kp = torch.cat(
        (init_kp3d.reshape(1, -1), norm_kp2d[0, 0].reshape(1, -1)), dim=-1
    ).unsqueeze(0).to(device)
    init_smpl = pt_transforms.matrix_to_rotation_6d(init_output.full_pose).unsqueeze(0).to(device)
    init_root = pt_transforms.matrix_to_rotation_6d(init_output.global_orient).to(device)
    inits = (init_kp, init_smpl)

    cam_angvel = torch.zeros((1, F, 6), device=device)
    bbox_b = bbox_t.unsqueeze(0).to(device)

    def run_network():
        with torch.inference_mode():
            return network(
                norm_kp2d, inits, feature_t,
                mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                return_y_up=True, cam_intrinsics=intrinsics_batch,
                bbox=bbox_b, res=res_batch, refine_traj=False, states=None,
            )

    # warm up
    run_network()

    print("[deep] profiling with with_stack=True (legacy autograd profiler) ...")
    with torch.autograd.profiler.profile(use_cuda=True, with_stack=True) as prof:
        run_network()

    def walk_syncs(events):
        for e in events:
            key = e.key.split("::")[-1]
            if key in ("_local_scalar_dense", "item") and getattr(e, "self_cuda_time_total", 0.0) > 0:
                print(f"=== {e.key}  count={e.count}  cuda={e.self_cuda_time_total/1000:.3f} ms ===")
                for fr in getattr(e, "stack", None) or []:
                    if isinstance(fr, (tuple, list)):
                        fn = str(fr[0]).replace(ROOT + os.sep, "")
                        print(f"     {fn}:{fr[1]}  {fr[2]}")
                    else:
                        print(f"     {fr}")
                print()
            walk_syncs(getattr(e, "cpu_children", []))

    print("\n[deep] FULL stacks for remaining sync ops (_local_scalar_dense / item):\n")
    walk_syncs(prof.function_events)

    targets = {"_local_scalar_dense", "item", "copy_", "to", "_to_copy", "_cudnn_rnn"}
    print("\n[deep] call sites for the sync/copy ops (top-6 stack frames each):\n")
    for e in prof.key_averages(group_by_stack_n=6):
        key = e.key.split("::")[-1]
        if key in targets:
            cuda_ms = getattr(e, "self_cuda_time_total", getattr(e, "self_device_time_total", 0.0)) / 1000.0
            print(f"=== {e.key}  count={e.count}  cuda={cuda_ms:.2f} ms ===")
            st = getattr(e, "stack", None) or []
            for fr in st:
                if isinstance(fr, (tuple, list)):
                    fn = str(fr[0])
                    rel = fn.replace(ROOT + os.sep, "")
                    print(f"     {rel}:{fr[1]}  {fr[2]}")
                else:
                    print(f"     {fr}")
            print()


if __name__ == "__main__":
    main()

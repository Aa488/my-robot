"""Benchmark torch.compile on the WHAM network.

Mirrors the real streaming pipeline (network.half() + smpl.float() + fp16 autocast)
and answers: (1) does torch.compile help the WHAM forward, (2) where does the time
go (LSTM vs SMPL LBS).

Run:  python scripts/bench_wham_compile.py
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


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return a.mean(), a.min(), a.max()


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(ROOT, "configs", "yamls", "demo.yaml"))
    device = cfg.DEVICE.lower()
    print(f"[bench] device={device} cuda={torch.cuda.is_available()} torch={torch.__version__}")

    print("[bench] loading SMPL + WHAM network ...")
    smpl = build_body_model("cpu")
    network = build_network(cfg, smpl)
    network.eval()

    # --- mirror the pipeline dtype setup: RNN/LSTM half, SMPL LBS float32 ---
    network = network.half()
    smpl = smpl.float()  # same object as network.smpl

    keypoints_normalizer = Normalizer(cfg)
    F = 16

    feature_t = torch.randn(1, F, 1024, device=device).half()

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

    def run(net):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return net(
                norm_kp2d, inits, feature_t,
                mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                return_y_up=True, cam_intrinsics=intrinsics_batch,
                bbox=bbox_b, res=res_batch, refine_traj=False, states=None,
            )

    # --- baseline eager ---
    print("[bench] warming up eager ...")
    run(network)
    m = timeit(lambda: run(network))
    print(f"[eager]      WHAM network (fp16) : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}")
    eager_out = run(network)
    eager_verts = eager_out["verts_cam"].detach().clone()

    # --- torch.compile: full network, default mode ---
    for mode in ("default", "reduce-overhead"):
        try:
            print(f"[bench] torch.compile(network, mode={mode!r}) ...")
            net_c = torch.compile(network, mode=mode)
            run(net_c)  # trigger compile
            m = timeit(lambda: run(net_c))
            out_c = run(net_c)
            diff = (out_c["verts_cam"].detach() - eager_verts).abs().max().item()
            print(f"[compile/{mode}] WHAM network : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}  "
                  f"max|verts diff|={diff:.3e}")
        except Exception as e:
            print(f"[compile/{mode}] FAILED: {type(e).__name__}: {e}")

    # --- torch.compile: SMPL LBS layer only (in-place replacement) ---
    try:
        print("[bench] torch.compile(network.smpl, mode='default') ...")
        network.smpl = torch.compile(network.smpl, mode="default")
        run(network)  # trigger compile of the smpl submodule
        m = timeit(lambda: run(network))
        out_c = run(network)
        diff = (out_c["verts_cam"].detach() - eager_verts).abs().max().item()
        print(f"[compile/smpl] WHAM network : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}  "
              f"max|verts diff|={diff:.3e}")
    except Exception as e:
        print(f"[compile/smpl] FAILED: {type(e).__name__}: {e}")

    # --- non-Triton backends (Windows: Inductor needs Triton, which is absent) ---
    for backend in ("aot_eager", "cudagraphs"):
        try:
            print(f"[bench] torch.compile(network, backend={backend!r}) ...")
            net_c = torch.compile(network, backend=backend)
            run(net_c)  # trigger compile
            m = timeit(lambda: run(net_c))
            out_c = run(net_c)
            diff = (out_c["verts_cam"].detach() - eager_verts).abs().max().item()
            print(f"[compile/{backend}] WHAM network : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}  "
                  f"max|verts diff|={diff:.3e}")
        except Exception as e:
            print(f"[compile/{backend}] FAILED: {type(e).__name__}: {e}")

    # --- profiler breakdown: where does the time go? ---
    print("\n[profiler] WHAM network top ops by CUDA time:")
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            run(network)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    except Exception as e:
        print(f"[profiler] failed: {e}")


if __name__ == "__main__":
    main()

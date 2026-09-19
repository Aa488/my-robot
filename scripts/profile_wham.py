"""Profile the WHAM pipeline's heavy stages on synthetic input.

Answers: where does the ~190ms actually go?
  1. extractor.model(norm_img, encode=True)   -> ViT-H backbone + 1 token
  2. extractor.model(norm_img, encode=False)  -> ViT-H + full SMPL head (HMR2 single-frame)
  3. network(...)                             -> WHAM LSTM temporal net + SMPL forward

Run:  python scripts/profile_wham.py
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
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.preproc.backbone.utils import process_image
from lib.utils.kp_utils import root_centering
from lib.data.utils.normalizer import Normalizer
from lib.utils.imutils import compute_cam_intrinsics
import lib.utils.transforms as pt_transforms


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return a.mean(), a.min(), a.max()


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(ROOT, "configs", "yamls", "demo.yaml"))
    device = cfg.DEVICE.lower()
    print(f"[profile] device={device} cuda={torch.cuda.is_available()}")

    print("[profile] loading HMR2 extractor (ViT-H) ...")
    extractor = FeatureExtractor(device, cfg.FLIP_EVAL)
    print("[profile] loading SMPL + WHAM network ...")
    smpl = build_body_model("cpu")
    network = build_network(cfg, smpl)
    network.eval()

    keypoints_normalizer = Normalizer(cfg)

    # --- synthetic image ---
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    norm_img_np, _ = process_image(img, [128, 128], 1.0, 256, 256)
    norm_img = torch.from_numpy(norm_img_np).unsqueeze(0).to(device)

    # --- Stage A: extractor encode=True (ViT-H backbone + token) ---
    def run_encode():
        with torch.inference_mode():
            return extractor.model(norm_img, encode=True)

    m = timeit(run_encode)
    print(f"[A] extractor encode=True  (ViT-H + token)     : mean={m[0]:7.1f} ms  "
          f"min={m[1]:.1f} max={m[2]:.1f}")

    # --- Stage B: extractor encode=False (ViT-H + full SMPL head) ---
    def run_decode():
        with torch.inference_mode():
            return extractor.model(norm_img, encode=False)

    m = timeit(run_decode)
    print(f"[B] extractor encode=False (ViT-H + SMPL head)  : mean={m[0]:7.1f} ms  "
          f"min={m[1]:.1f} max={m[2]:.1f}")

    # --- build network inputs (mirror wham_thread + build_wham_inits) ---
    F = 16  # stream_seq_len
    with torch.inference_mode():
        feature = extractor.model(norm_img, encode=True).squeeze(0).detach()
    feature_t = torch.stack([feature] * F, dim=0).unsqueeze(0).to(device)

    dummy_tracking = {0: {}}
    with torch.inference_mode():
        extractor.predict_init(norm_img, dummy_tracking, 0, flip_eval=False)
    init_data = dummy_tracking[0]

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
            global_orient=init_data["init_global_orient"].float().to(device),
            body_pose=init_data["init_body_pose"].float().to(device),
            betas=init_data["init_betas"].float().to(device),
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

    m = timeit(run_network)
    print(f"[C] WHAM network (LSTM + SMPL, seq=16)         : mean={m[0]:7.1f} ms  "
          f"min={m[1]:.1f} max={m[2]:.1f}")

    # --- torch.profiler breakdown of the ViT extractor ---
    print("\n[profiler] ViT extractor (encode=True) top ops by CUDA time:")
    try:
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            with torch.inference_mode():
                extractor.model(norm_img, encode=True)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    except Exception as e:
        print(f"[profiler] ViT profile failed: {e}")

    # --- torch.profiler breakdown of the WHAM network ---
    print("\n[profiler] WHAM network top ops by CUDA time:")
    try:
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            with torch.inference_mode():
                network(
                    norm_kp2d, inits, feature_t,
                    mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                    return_y_up=True, cam_intrinsics=intrinsics_batch,
                    bbox=bbox_b, res=res_batch, refine_traj=False, states=None,
                )
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    except Exception as e:
        print(f"[profiler] network profile failed: {e}")

    # =====================================================================
    # [NEW] ① detector (YOLOv8n + ViTPose-base) timing
    # =====================================================================
    try:
        from lib.models.preproc.detector import DetectionModel
        from mmpose.apis import inference_top_down_pose_model

        pose_model_cfg = os.path.join(
            ROOT, "third-party", "ViTPose", "configs", "body",
            "2d_kpt_sview_rgb_img", "topdown_heatmap", "coco",
            "ViTPose_base_coco_256x192.py",
        )
        pose_model_ckpt = os.path.join(ROOT, "checkpoints", "vitpose_base_coco_aic_mpii.pth")
        bbox_model_ckpt = os.path.join(ROOT, "checkpoints", "yolov8n.pt")
        print("[profile] loading detector (YOLOv8n + ViTPose-base) ...")
        detector = DetectionModel(
            device, pose_model_cfg=pose_model_cfg,
            pose_model_ckpt=pose_model_ckpt, bbox_model_ckpt=bbox_model_ckpt,
        )

        frame_bgr = np.random.randint(0, 255, (640, 512, 3), dtype=np.uint8)

        def run_yolo():
            return detector.bbox_model.predict(
                frame_bgr, device=device, classes=0, conf=0.5,
                save=False, verbose=False,
            )[0].boxes.xyxy.detach().cpu().numpy()

        def run_vitpose():
            return inference_top_down_pose_model(
                detector.pose_model, frame_bgr,
                person_results=[{"bbox": [100.0, 80.0, 400.0, 480.0]}],
                format="xyxy", return_heatmap=False, outputs=None,
            )

        m = timeit(run_yolo)
        print(f"[D1] YOLOv8n bbox detect     : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}")
        m = timeit(run_vitpose)
        print(f"[D2] ViTPose-base keypoints : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}")
    except Exception as e:
        print(f"[D] detector profile failed: {e}")

    # =====================================================================
    # [NEW] ② FP16 / AMP (autocast) speedup
    # =====================================================================
    try:
        def run_encode_amp():
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                return extractor.model(norm_img, encode=True)

        def run_network_amp():
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                return network(
                    norm_kp2d, inits, feature_t, mask=mask, init_root=init_root,
                    cam_angvel=cam_angvel, return_y_up=True,
                    cam_intrinsics=intrinsics_batch, bbox=bbox_b, res=res_batch,
                    refine_traj=False, states=None,
                )

        m = timeit(run_encode_amp)
        print(f"[A-amp] extractor encode=True FP16 : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}")
        m = timeit(run_network_amp)
        print(f"[C-amp] WHAM network FP16           : mean={m[0]:7.1f} ms  min={m[1]:.1f} max={m[2]:.1f}")
    except Exception as e:
        print(f"[FP16] profile failed: {e}")

    # =====================================================================
    # [NEW] ③ deep profiler: where do copy_ / .item() come from?
    # =====================================================================
    try:
        from torch.profiler import profile, ProfilerActivity
        from collections import Counter

        def _frame_of(f):
            if f is None:
                return None
            if isinstance(f, (tuple, list)) and len(f) >= 3:
                return (str(f[0]), f[1], str(f[2]))
            fn = getattr(f, "filename", None)
            if fn:
                return (str(fn), getattr(f, "line", None), getattr(f, "name", None))
            return None

        with profile(activities=[ProfilerActivity.CUDA], with_stack=True) as prof:
            with torch.inference_mode():
                network(
                    norm_kp2d, inits, feature_t, mask=mask, init_root=init_root,
                    cam_angvel=cam_angvel, return_y_up=True,
                    cam_intrinsics=intrinsics_batch, bbox=bbox_b, res=res_batch,
                    refine_traj=False, states=None,
                )

        targets = {"aten::copy_", "aten::_local_scalar_dense", "aten::item"}
        root_norm = ROOT.replace("\\", "/")
        hit = Counter()
        for evt in prof.events():
            if evt.name not in targets:
                continue
            repo = None
            for f in (evt.stack or []):
                fr = _frame_of(f)
                if fr and root_norm in fr[0].replace("\\", "/") and "site-packages" not in fr[0]:
                    repo = fr
                    break
            hit[repo or ("(no repo frame)",)] += 1
        print("[deep] copy_/item() call sites (by # calls):")
        for fr, c in hit.most_common(12):
            if isinstance(fr, tuple) and len(fr) == 3 and fr[0] != "(no repo frame)":
                rel = str(fr[0]).replace(ROOT + os.sep, "")
                print(f"    {c:4d}x  {rel}:{fr[1]}  {fr[2]}")
            else:
                print(f"    {c:4d}x  {fr}")
    except Exception as e:
        print(f"[deep] profiler failed: {e}")


if __name__ == "__main__":
    main()

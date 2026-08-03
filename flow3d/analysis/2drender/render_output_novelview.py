#!/usr/bin/env python3
"""
Render a trained DAVIS/DiVa-360 scene from nearby DiVa-360 camera viewpoints.

Motivation
----------
Training only ever supervises one physical camera (e.g. cam00), so the model
has no ground truth for any other viewpoint. This script picks a handful of
*other* DiVa-360 rig cameras that are angularly close to the training camera
(so the extrapolation stays modest) and renders the trained scene as if seen
from each of them.

Coordinate-frame note
----------------------
DiVa-360's `transforms_*.json` camera poses live in their own real-world,
metric coordinate frame, using the OpenGL/NeRF local camera-axis convention
(camera looks down -Z, Y is up). The trained scene instead lives in
mega-sam's SLAM coordinate frame (arbitrary origin/scale, further normalized
by the dataset), using the OpenCV local camera-axis convention (camera looks
down +Z, Y is down). The two frames are NOT related by a simple transform, so
DiVa-360 camera poses cannot be plugged into the renderer directly.

Instead, this script:
  1. Takes the *relative* rotation from the reference camera (cam00) to each
     candidate camera, in DiVa-360's own frame. A relative rotation expressed
     in a camera's local axes is coordinate-frame independent (up to the
     fixed OpenGL<->OpenCV local-axis flip, which is conjugated in), so it
     can be transplanted onto the reference camera's *trained* pose.
  2. Keeps the novel camera at the same distance from the scene as the
     reference camera (mega-sam's scale is unknown, so DiVa-360's absolute
     translations aren't usable) and orbits it around the foreground
     centroid using that transplanted rotation.

This gives a physically-reasonable novel view for small angular offsets from
the reference camera; it is not a substitute for real multi-view supervision.

Example
-------
    python flow3d/analysis/2drender/render_output_novelview.py \\
        --ckpt outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3/checkpoints/last.ckpt \\
        --seq_name dog_cam00 \\
        --diva_dir data/DiVa360/processed_data/dog
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v2 as iio
import numpy as np
import torch
import yaml

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.renderer import Renderer

# Fixed local-axis flip between the OpenGL/NeRF convention (DiVa-360
# transforms.json: camera looks down -Z, Y-up) and the OpenCV convention
# (mega-sam: camera looks down +Z, Y-down).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt", type=Path,
        default=Path("outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3/checkpoints/last.ckpt"),
        help="Path to a trained SceneModel checkpoint (last.ckpt).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--seq_name", type=str, default="dog_cam00")
    parser.add_argument(
        "--diva_dir", type=Path, default=Path("data/DiVa360/processed_data/dog"),
        help="DiVa-360 sequence dir containing transforms_{train,test,val}.json.",
    )
    parser.add_argument("--ref_cam", type=str, default="cam00", help="The camera used for training.")
    parser.add_argument(
        "--max_angle_deg", type=float, default=40.0,
        help="Discard DiVa-360 cameras farther than this angle from --ref_cam.",
    )
    parser.add_argument("--num_cams", type=int, default=6, help="Max number of novel viewpoints to render.")
    parser.add_argument(
        "--frames", type=str, default=None,
        help='Comma list of training frame indices, or "all" (default: middle frame only).',
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def load_diva_cam_poses(diva_dir: Path) -> dict[str, np.ndarray]:
    """Load DiVa-360 camera-to-world matrices (OpenGL/NeRF axis convention)."""
    cams: dict[str, np.ndarray] = {}
    for name in ("transforms_train.json", "transforms_test.json", "transforms_val.json"):
        path = diva_dir / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for frame in data.get("frames", []):
            parts = frame["file_path"].split("/")
            if len(parts) < 2:
                continue
            cam = parts[1]
            cams.setdefault(cam, np.array(frame["transform_matrix"], dtype=np.float64))
    return cams


def select_neighbor_cams(
    cams: dict[str, np.ndarray], ref_cam: str, max_angle_deg: float, num_cams: int
) -> list[tuple[str, float]]:
    """Rank DiVa-360 cameras by angular distance (about the rig center) from ref_cam."""
    if ref_cam not in cams:
        raise ValueError(f"Reference camera {ref_cam!r} not found in DiVa-360 calibration.")

    names = sorted(cams.keys(), key=lambda n: int(n[3:]))
    positions = np.stack([cams[n][:3, 3] for n in names])
    center = positions.mean(axis=0)
    dirs = positions - center
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    ref_dir = dirs[names.index(ref_cam)]
    angles = np.degrees(np.arccos(np.clip(dirs @ ref_dir, -1.0, 1.0)))

    candidates = [
        (name, float(angle))
        for name, angle in zip(names, angles)
        if name != ref_cam and angle <= max_angle_deg
    ]
    candidates.sort(key=lambda item: item[1])
    return candidates[:num_cams]


def relative_rotation_cv(r_ref_gl: np.ndarray, r_cam_gl: np.ndarray) -> np.ndarray:
    """Relative rotation ref->cam (DiVa-360, OpenGL axes) re-expressed in OpenCV axes."""
    rel_gl = r_ref_gl.T @ r_cam_gl
    return _GL_TO_CV @ rel_gl @ _GL_TO_CV


def parse_frames(text: str, total_frames: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total_frames))
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("No frame was selected.")
    invalid = [x for x in values if x < 0 or x >= total_frames]
    if invalid:
        raise ValueError(f"Invalid frames {invalid}; total frame count is {total_frames}.")
    return values


def load_dataset(config_path: Path, seq_name: str) -> CasualDataset:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(
        root_dir=scene_cfg["data_dir"], seq_name=seq_name, **scene_cfg.get("data", {}),
    )
    return CasualDataset(**asdict(data_cfg))


def main() -> None:
    args = build_parser().parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.diva_dir.is_dir():
        raise FileNotFoundError(f"DiVa-360 sequence dir not found: {args.diva_dir}")

    out_dir = args.out_dir if args.out_dir is not None else args.ckpt.parents[1] / "novel_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    diva_cams = load_diva_cam_poses(args.diva_dir)
    neighbors = select_neighbor_cams(diva_cams, args.ref_cam, args.max_angle_deg, args.num_cams)
    if not neighbors:
        raise RuntimeError(f"No DiVa-360 cameras found within {args.max_angle_deg} deg of {args.ref_cam}.")
    print(f"Selected {len(neighbors)} novel viewpoint(s) near {args.ref_cam}:")
    for name, angle in neighbors:
        print(f"  {name}: {angle:.1f} deg")

    dataset = load_dataset(args.config, args.seq_name)
    frames = (
        parse_frames(args.frames, dataset.num_frames)
        if args.frames is not None
        else [dataset.num_frames // 2]
    )
    print(f"Rendering {len(frames)} frame(s): {frames}")

    r_ref_diva = diva_cams[args.ref_cam][:3, :3]
    summary = []

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(str(args.ckpt), device, work_dir=str(out_dir), port=None)
        model = renderer.model
        model.eval()

        for frame_idx in frames:
            w2c_ref = dataset.w2cs[frame_idx].to(device)
            K = dataset.Ks[frame_idx].to(device)
            img = dataset.get_image(frame_idx)
            H, W = img.shape[:2]
            img_wh = (W, H)

            c2w_ref = torch.linalg.inv(w2c_ref)
            r_ref_train = c2w_ref[:3, :3]
            c_ref_train = c2w_ref[:3, 3]

            # Scene pivot: foreground centroid at this frame, in the trained model's frame.
            fg_means = model.compute_poses_fg(torch.tensor([frame_idx], device=device))[0]
            pivot = fg_means[:, 0, :].mean(dim=0)
            radius = torch.linalg.norm(c_ref_train - pivot)

            ref_out = model.render(frame_idx, w2c_ref[None], K[None], img_wh, use_learned_poses=False)
            ref_img = (ref_out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            iio.imwrite(out_dir / f"{args.ref_cam}_frame{frame_idx:04d}_ref.png", ref_img)

            for name, angle in neighbors:
                r_cam_diva = diva_cams[name][:3, :3]
                r_rel_cv = torch.from_numpy(relative_rotation_cv(r_ref_diva, r_cam_diva)).to(
                    device=device, dtype=r_ref_train.dtype
                )

                r_novel = r_ref_train @ r_rel_cv
                forward = r_novel[:, 2]
                forward = forward / torch.linalg.norm(forward)
                c_novel = pivot - radius * forward

                c2w_novel = torch.eye(4, device=device, dtype=r_ref_train.dtype)
                c2w_novel[:3, :3] = r_novel
                c2w_novel[:3, 3] = c_novel
                w2c_novel = torch.linalg.inv(c2w_novel)

                out = model.render(frame_idx, w2c_novel[None], K[None], img_wh, use_learned_poses=False)
                img_out = (out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                fname = f"{name}_angle{angle:04.1f}_frame{frame_idx:04d}.png"
                iio.imwrite(out_dir / fname, img_out)
                summary.append({"cam": name, "angle_deg": angle, "frame_idx": frame_idx, "file": fname})
                print(f"  saved {fname}")

    (out_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "ref_cam": args.ref_cam,
                "frames": frames,
                "max_angle_deg": args.max_angle_deg,
                "neighbors": summary,
            },
            indent=2,
        )
    )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

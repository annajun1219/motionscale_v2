#!/usr/bin/env python3
"""
Render a trained DAVIS/megasam-format scene from a small grid of novel
viewpoints orbited around a single representative training camera.

Motivation
----------
render_output_novelview.py's synthetic views are anchored to DyCheck's
`camera/*.json` + `scene.json` + `extra.json` (real side cameras and a
bundle-adjusted scene pivot/up vector) -- metadata that only exists for
DyCheck-derived sequences (spin/spaceout/teddy). A plain DAVIS/megasam
sequence like "camel" has none of that, so there is no GT novel view and no
DyCheck scene geometry to anchor synthetic views to.

Instead, this script:
  1. Picks ONE representative training camera pose (default: the middle
     training frame's megasam camera, overridable via --ref_frame) as "the"
     viewpoint to evaluate novel views against.
  2. Orbits that single fixed camera around the foreground centroid (computed
     once, at --ref_frame) by every (yaw, pitch) combination in the requested
     grid -- default yaw in {-20,-10,0,10,20} deg, pitch in {-10,0,10} deg,
     i.e. 15 novel camera poses total. Each novel pose keeps the same
     distance to the pivot as the reference camera (mega-sam's scale is
     arbitrary, but preserving distance keeps foreshortening comparable) and
     reuses the reference camera's own up direction, so yaw=0/pitch=0
     reproduces the reference camera exactly.
  3. Since the camera for a given (yaw, pitch) is fixed in world space (not
     re-derived per frame), rendering it across the requested training frames
     produces a short "bullet time" video of the reconstructed motion seen
     from that one fixed novel angle -- the most direct way to sanity-check
     reconstruction quality when there is no ground-truth novel view to
     compare against.

Example
-------
    python flow3d/analysis/2drender/render_output_novelview2.py \\
        --ckpt outputs/davis/camel/2026_07_24_15_16_40__camel_simple_contact_ft50/checkpoints/last.ckpt \\
        --config configs/davis/default.yaml \\
        --seq_name camel
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v2 as iio
import imageio.v3 as iio3
import numpy as np
import torch
import yaml

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.renderer import Renderer
from flow3d.vis.utils import make_video_divisble

EPS = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="Path to a trained SceneModel checkpoint (last.ckpt).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument(
        "--ref_frame", type=int, default=None,
        help="Training frame index whose camera is the representative viewpoint "
             "novel views orbit around. Default: the middle training frame.",
    )
    parser.add_argument(
        "--yaw-deg", "--yaw_deg", dest="yaw_deg", type=str, default="-20,-10,0,10,20",
    )
    parser.add_argument(
        "--pitch-deg", "--pitch_deg", dest="pitch_deg", type=str, default="-10,0,10",
    )
    parser.add_argument(
        "--orbit-radius-scale", "--orbit_radius_scale", dest="orbit_radius_scale",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--frames", type=str, default="all",
        help='Comma list of training frame indices to render through each novel '
             'camera, or "all" (default: every training frame, so the per-view '
             "videos cover the full sequence).",
    )
    parser.add_argument(
        "--image_stride", type=int, default=10,
        help="Also dump a standalone PNG every N frames (e.g. frame0000, frame0010, "
             "frame0020, ...). Does not affect which frames go into the videos.",
    )
    parser.add_argument(
        "--no_video", action="store_true",
        help="Skip stitching per-view videos; only dump the strided PNGs.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frame rate for the per-view output videos.")
    parser.add_argument("--image-width", "--image_width", dest="image_width", type=int, default=None)
    parser.add_argument("--image-height", "--image_height", dest="image_height", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one angle is required.")
    return values


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


def angle_tag(value: float) -> str:
    return (
        f"{value:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    )


def camera_center_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return -(rotation.transpose(0, 1) @ translation)


def look_at_w2c(
    camera_center: torch.Tensor, target: torch.Tensor, up_hint: torch.Tensor
) -> torch.Tensor:
    """OpenCV camera axes (x=right, y=down, z=forward)."""
    forward = target - camera_center
    forward = forward / forward.norm().clamp_min(EPS)

    up = up_hint / up_hint.norm().clamp_min(EPS)
    right = torch.cross(forward, up, dim=0)
    if right.norm() < 1e-5:
        right = torch.tensor([1.0, 0.0, 0.0], dtype=target.dtype, device=target.device)
    right = right / right.norm().clamp_min(EPS)
    down = torch.cross(forward, right, dim=0)
    down = down / down.norm().clamp_min(EPS)

    w2c = torch.eye(4, dtype=target.dtype, device=target.device)
    w2c[:3, :3] = torch.stack([right, down, forward], dim=0)
    w2c[:3, 3] = -(w2c[:3, :3] @ camera_center)
    return w2c


def orbit_camera(
    reference_w2c: torch.Tensor,
    target: torch.Tensor,
    yaw_deg: float,
    pitch_deg: float,
    radius_scale: float,
) -> torch.Tensor:
    """Orbit the reference camera around `target` by (yaw, pitch), measured in
    the reference camera's own local right/up axes -- yaw=0/pitch=0 reproduces
    the reference camera exactly."""
    camera_center = camera_center_from_w2c(reference_w2c)
    offset = camera_center - target

    radius = offset.norm().clamp_min(EPS) * radius_scale
    offset = offset / offset.norm().clamp_min(EPS)

    reference_up = -reference_w2c[:3, :3][1]
    reference_up = reference_up / reference_up.norm().clamp_min(EPS)

    right = torch.cross(reference_up, offset, dim=0)
    if right.norm() < 1e-5:
        right = torch.tensor([1.0, 0.0, 0.0], dtype=target.dtype, device=target.device)
    right = right / right.norm().clamp_min(EPS)

    up = torch.cross(offset, right, dim=0)
    up = up / up.norm().clamp_min(EPS)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    direction = (
        math.cos(pitch) * math.cos(yaw) * offset
        + math.cos(pitch) * math.sin(yaw) * right
        + math.sin(pitch) * up
    )
    direction = direction / direction.norm().clamp_min(EPS)

    novel_center = target + radius * direction
    return look_at_w2c(novel_center, target, reference_up)


def main() -> None:
    args = build_parser().parse_args()

    if args.orbit_radius_scale <= 0:
        raise ValueError("--orbit-radius-scale must be > 0")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    out_dir = args.out_dir if args.out_dir is not None else args.ckpt.parents[1] / "novel_views2"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    yaw_values = parse_float_list(args.yaw_deg)
    pitch_values = parse_float_list(args.pitch_deg)

    dataset = load_dataset(args.config, args.seq_name)
    frames = parse_frames(args.frames, dataset.num_frames)
    image_frames = {f for f in frames if f % args.image_stride == 0}

    ref_frame = args.ref_frame if args.ref_frame is not None else dataset.num_frames // 2
    if not (0 <= ref_frame < dataset.num_frames):
        raise ValueError(
            f"--ref_frame {ref_frame} out of range [0, {dataset.num_frames})."
        )
    print(f"Representative camera: training frame {ref_frame} (of {dataset.num_frames})")
    print(f"Rendering {len(frames)} frame(s) through each of "
          f"{len(yaw_values) * len(pitch_values)} novel views "
          f"(yaw={yaw_values}, pitch={pitch_values})")

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(str(args.ckpt), device, work_dir=str(out_dir), port=None)
        model = renderer.model
        model.eval()

        w2c_ref = dataset.w2cs[ref_frame].to(device)
        K = dataset.Ks[ref_frame].to(device)
        img = dataset.get_image(ref_frame)
        H, W = img.shape[:2]
        if args.image_width is not None or args.image_height is not None:
            width = args.image_width or W
            height = args.image_height or H
            K = K.clone()
            K[0, :] *= float(width) / float(W)
            K[1, :] *= float(height) / float(H)
            W, H = width, height
        img_wh = (W, H)

        # Scene pivot: foreground centroid at the representative frame, fixed
        # for every novel view (the camera stays put in world space; only the
        # scene animates as `frames` are rendered through it).
        fg_means = model.compute_poses_fg(torch.tensor([ref_frame], device=device))[0]
        pivot = fg_means[:, 0, :].mean(dim=0)

        # Ground-truth-free "reference" render: the representative camera as-is.
        ref_out = model.render(ref_frame, w2c_ref[None], K[None], img_wh, use_learned_poses=False)
        ref_img = (ref_out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        iio.imwrite(out_dir / f"reference_frame{ref_frame:04d}.png", ref_img)

        novel_views = {}
        for pitch in pitch_values:
            for yaw in yaw_values:
                name = f"yaw{angle_tag(yaw)}_pitch{angle_tag(pitch)}"
                novel_views[name] = {
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "w2c": orbit_camera(w2c_ref, pivot, yaw, pitch, args.orbit_radius_scale),
                }
        print(f"Selected {len(novel_views)} novel viewpoint(s):")
        for name, v in novel_views.items():
            print(f"  {name}: yaw={v['yaw_deg']:+.1f} deg, pitch={v['pitch_deg']:+.1f} deg")

        summary = []
        video_buffers: dict[str, list[np.ndarray]] = {name: [] for name in novel_views}

        for frame_idx in frames:
            for name, v in novel_views.items():
                out = model.render(frame_idx, v["w2c"][None], K[None], img_wh, use_learned_poses=False)
                img_out = (out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

                if not args.no_video:
                    video_buffers[name].append(img_out)

                if frame_idx in image_frames:
                    view_dir = out_dir / name
                    view_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"frame{frame_idx:04d}.png"
                    iio.imwrite(view_dir / fname, img_out)
                    summary.append({
                        "view": name,
                        "frame_idx": frame_idx,
                        "yaw_deg": v["yaw_deg"],
                        "pitch_deg": v["pitch_deg"],
                        "file": f"{name}/{fname}",
                    })
            if frame_idx in image_frames:
                print(f"  saved frame {frame_idx:04d} ({len(novel_views)} views)")

        if not args.no_video:
            for name, frames_out in video_buffers.items():
                video = make_video_divisble(np.stack(frames_out, axis=0))
                video_path = out_dir / f"{name}.mp4"
                iio3.imwrite(video_path, video, fps=args.fps)
                print(f"  saved {video_path} ({len(frames_out)} frames @ {args.fps}fps)")

    (out_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "ref_frame": ref_frame,
                "frames": frames,
                "image_frames": sorted(image_frames),
                "video": (not args.no_video),
                "fps": args.fps,
                "views": summary,
            },
            indent=2,
        )
    )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

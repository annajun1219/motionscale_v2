#!/usr/bin/env python3
"""
Render a trained scene from novel viewpoints, isolating the *coarse* vs
*full* (coarse+fine-blended) motion of a small set of user-chosen clusters
(default: 4, 21) -- a visual counterpart to flow3d/analysis/loss_joint.py /
loss_joint_gnn_only.py's numeric seam-gap measurements.

Reuses the exact camera generation (orbit_camera/look_at_w2c) from
flow3d/analysis/2drender/render_ouput_novelview2.py, so a given (yaw, pitch)
here is directly comparable to that script's novel_views2/yaw..._pitch...
outputs.

For each requested (frame, yaw, pitch), produces three images:
  1. full.png           -- the normal, unmodified render (coarse+fine for
                            every cluster; identical to what
                            render_ouput_novelview2.py already produces).
  2. coarse_only.png     -- the SAME render, except the target clusters'
                            Gaussians use ONLY their coarse rigid transform
                            (model.compute_transforms_coarse) instead of the
                            normal coarse+fine blend -- everything else
                            (other clusters, background) is untouched. Shows
                            what those clusters' motion looks like with fine
                            motion removed.
  3. overlay.png         -- the full render as background, with the target
                            clusters' Gaussians projected twice: their
                            ACTUAL (full, what's really rendered) positions
                            in one color, and their COARSE-ONLY positions in
                            another -- directly visualizing, on the real
                            photo, how far and in which direction fine
                            motion displaces each cluster's points relative
                            to its own rigid skeleton.

Example
-------
    python flow3d/analysis/2drender/coarse_vs_fine_clusters.py \\
        --ckpt outputs/davis/camel/2026_08_27_08_43_37__gnn_correction_loss_joint/checkpoints/last.ckpt \\
        --config configs/davis/default.yaml \\
        --seq_name camel \\
        --clusters 4,21 \\
        --yaw-deg 20 --pitch-deg -10 \\
        --frames 0,16,35,45,60
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v2 as iio
import numpy as np
import roma
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.renderer import Renderer

EPS = 1e-8

# Per-cluster hue (bright = actual/full position, dark = coarse-only
# position of the SAME cluster's SAME Gaussians) -- lets the overlay show
# both effects at once: how far each cluster's fine motion pulls it from
# its own rigid skeleton (bright vs dark, same hue), AND whether the two
# clusters visually separate from each other (hue vs hue).
CLUSTER_HUES = [
    ((255, 140, 0), (110, 60, 0)),    # orange (full) / dark orange (coarse)
    ((0, 140, 255), (0, 60, 110)),    # blue (full) / dark blue (coarse)
    ((60, 220, 60), (20, 90, 20)),    # green (full) / dark green (coarse)
    ((220, 60, 220), (90, 20, 90)),   # magenta (full) / dark magenta (coarse)
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument(
        "--clusters", type=str, default="4,21",
        help="Comma list of raw cluster ids (model.fg.get_cluster_ids() space) to isolate.",
    )
    parser.add_argument(
        "--ref_frame", type=int, default=None,
        help="Training frame whose camera the novel view orbits around. Default: middle frame "
             "(same default as render_ouput_novelview2.py).",
    )
    parser.add_argument("--yaw-deg", "--yaw_deg", dest="yaw_deg", type=float, default=20.0)
    parser.add_argument("--pitch-deg", "--pitch_deg", dest="pitch_deg", type=float, default=-10.0)
    parser.add_argument("--orbit-radius-scale", "--orbit_radius_scale", dest="orbit_radius_scale", type=float, default=1.0)
    parser.add_argument(
        "--frames", type=str, default="0,10,20,30,40,50,60,70,80",
        help='Comma list of training frame indices, or "all".',
    )
    parser.add_argument(
        "--overlay-max-points", "--overlay_max_points", dest="overlay_max_points", type=int, default=1500,
        help="Max Gaussians per target cluster drawn in the overlay (deterministic subsample).",
    )
    parser.add_argument("--marker-radius", "--marker_radius", dest="marker_radius", type=int, default=2)
    parser.add_argument("--image-width", "--image_width", dest="image_width", type=int, default=None)
    parser.add_argument("--image-height", "--image_height", dest="image_height", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def parse_frames(text: str, total_frames: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total_frames))
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    invalid = [x for x in values if x < 0 or x >= total_frames]
    if invalid:
        raise ValueError(f"Invalid frames {invalid}; total frame count is {total_frames}.")
    return values


def load_dataset(config_path: Path, seq_name: str) -> CasualDataset:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(root_dir=scene_cfg["data_dir"], seq_name=seq_name, **scene_cfg.get("data", {}))
    return CasualDataset(**asdict(data_cfg))


def camera_center_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return -(rotation.transpose(0, 1) @ translation)


def look_at_w2c(camera_center: torch.Tensor, target: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    """OpenCV camera axes (x=right, y=down, z=forward). Identical to
    render_ouput_novelview2.py's helper of the same name."""
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
    reference_w2c: torch.Tensor, target: torch.Tensor, yaw_deg: float, pitch_deg: float, radius_scale: float,
) -> torch.Tensor:
    """Identical to render_ouput_novelview2.py's helper of the same name --
    duplicated (not imported) so this script has no import-time dependency
    on that script's argparse/module-level setup."""
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


def angle_tag(value: float) -> str:
    return f"{value:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "d")


def compute_poses_fg_coarse_only(model: Any, ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same math as SceneModel.compute_poses_fg, but using
    model.compute_transforms_coarse (rigid per-cluster transform only) in
    place of model.compute_transforms (coarse+fine blended) -- i.e. what the
    foreground would look like with fine motion entirely removed.

    :returns: means (G, B, 3), quats (G, B, 4).
    """
    means = model.fg.params["means"]
    quats = model.fg.get_quats()
    transfms = model.compute_transforms_coarse(ts)  # (G, B, 3, 4)
    means_out = torch.einsum("pnij,pj->pni", transfms, F.pad(means, (0, 1), value=1.0))
    quats_out = roma.quat_xyzw_to_wxyz(
        roma.quat_product(
            roma.rotmat_to_unitquat(transfms[..., :3, :3]),
            roma.quat_wxyz_to_xyzw(quats[:, None]),
        )
    )
    quats_out = F.normalize(quats_out, p=2, dim=-1)
    return means_out, quats_out


def build_coarse_only_override(
    model: Any, t: int, target_cluster_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-scene means/quats (background + shadow + foreground, all normal
    coarse+fine motion) EXCEPT the foreground Gaussians in
    target_cluster_mask, which use coarse-only motion instead.

    :returns: means (N_all, 3), quats (N_all, 4) -- single-frame, ready for
        model.render(..., means=..., quats=...).
    """
    device = model.fg.params["means"].device
    ts = torch.tensor([t], device=device)

    full_means, full_quats = model.compute_poses_all(ts)  # (N_all, 1, 3), (N_all, 1, 4)
    coarse_means, coarse_quats = compute_poses_fg_coarse_only(model, ts)  # (G_fg, 1, 3), (G_fg, 1, 4)

    mixed_means = full_means[:, 0].clone()
    mixed_quats = full_quats[:, 0].clone()
    num_fg = model.num_fg_gaussians
    mixed_means[:num_fg][target_cluster_mask] = coarse_means[:, 0][target_cluster_mask]
    mixed_quats[:num_fg][target_cluster_mask] = coarse_quats[:, 0][target_cluster_mask]
    return mixed_means, mixed_quats


def project_world_points(
    points: torch.Tensor, w2c: torch.Tensor, intrinsic: torch.Tensor, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    camera_points = (w2c @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    projected = (intrinsic @ camera_points.T).T
    pixels = projected[:, :2] / depth[:, None].clamp_min(1e-8)
    valid = (depth > 1e-8) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels.detach().cpu().numpy(), valid.detach().cpu().numpy()


def draw_points(draw: ImageDraw.ImageDraw, pixels: np.ndarray, valid: np.ndarray, color: tuple, radius: int) -> None:
    for x, y in pixels[valid]:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def main() -> None:
    args = build_parser().parse_args()
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    target_clusters = parse_int_list(args.clusters)
    out_dir = args.out_dir if args.out_dir is not None else args.ckpt.parents[1] / "coarse_vs_fine_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = load_dataset(args.config, args.seq_name)
    frames = parse_frames(args.frames, dataset.num_frames)

    ref_frame = args.ref_frame if args.ref_frame is not None else dataset.num_frames // 2
    if not (0 <= ref_frame < dataset.num_frames):
        raise ValueError(f"--ref_frame {ref_frame} out of range [0, {dataset.num_frames}).")

    view_name = f"yaw{angle_tag(args.yaw_deg)}_pitch{angle_tag(args.pitch_deg)}"
    print(f"Target clusters: {target_clusters}")
    print(f"Representative camera: training frame {ref_frame} (of {dataset.num_frames})")
    print(f"Novel view: {view_name}  (radius_scale={args.orbit_radius_scale})")
    print(f"Frames: {frames}")

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(str(args.ckpt), device, work_dir=str(out_dir), port=None)
        model = renderer.model
        model.eval()

        if len(target_clusters) > len(CLUSTER_HUES):
            raise ValueError(
                f"Only {len(CLUSTER_HUES)} distinct overlay hues are defined; "
                f"got {len(target_clusters)} target clusters."
            )

        cluster_ids = model.fg.get_cluster_ids().reshape(-1).long().to(device)
        target_mask = torch.isin(cluster_ids, torch.tensor(target_clusters, device=device, dtype=torch.long))
        num_target = int(target_mask.sum())
        if num_target == 0:
            raise RuntimeError(f"No foreground Gaussian belongs to clusters {target_clusters}.")
        print(f"Foreground Gaussians in target clusters: {num_target}")

        # per-cluster deterministic subsample for the overlay (drawing every
        # point can be dense enough to obscure the underlying render, and a
        # combined subsample could let one large cluster crowd out a smaller
        # one -- so subsample each cluster's own points independently)
        rng = np.random.default_rng(args.seed)
        overlay_idx_by_cluster: dict[int, torch.Tensor] = {}
        for cid in target_clusters:
            idx = torch.where(cluster_ids == cid)[0]
            if idx.numel() == 0:
                raise RuntimeError(f"No foreground Gaussian belongs to cluster {cid}.")
            if idx.numel() > args.overlay_max_points:
                chosen = np.sort(rng.choice(idx.numel(), size=args.overlay_max_points, replace=False))
                idx = idx[torch.as_tensor(chosen, device=device)]
            overlay_idx_by_cluster[cid] = idx
            print(f"  cluster {cid}: {int((cluster_ids == cid).sum())} Gaussians, "
                  f"{idx.numel()} drawn in overlay")

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

        fg_means_ref = model.compute_poses_fg(torch.tensor([ref_frame], device=device))[0]
        pivot = fg_means_ref[:, 0, :].mean(dim=0)
        w2c_novel = orbit_camera(w2c_ref, pivot, args.yaw_deg, args.pitch_deg, args.orbit_radius_scale)

        view_dir = out_dir / view_name
        full_dir = view_dir / "full"
        coarse_dir = view_dir / "coarse_only"
        overlay_dir = view_dir / "overlay"
        for d in (full_dir, coarse_dir, overlay_dir):
            d.mkdir(parents=True, exist_ok=True)

        for frame_idx in frames:
            ts = torch.tensor([frame_idx], device=device)

            # 1. full (normal) render
            full_out = model.render(frame_idx, w2c_novel[None], K[None], img_wh, use_learned_poses=False)
            full_img = (full_out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            iio.imwrite(full_dir / f"frame{frame_idx:04d}.png", full_img)

            # 2. coarse-only-for-target-clusters render
            mixed_means, mixed_quats = build_coarse_only_override(model, frame_idx, target_mask)
            coarse_out = model.render(
                frame_idx, w2c_novel[None], K[None], img_wh,
                means=mixed_means, quats=mixed_quats, use_learned_poses=False,
            )
            coarse_img = (coarse_out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            iio.imwrite(coarse_dir / f"frame{frame_idx:04d}.png", coarse_img)

            # 3. overlay: full render as background, per-cluster point sets
            # projected twice each (bright=actual, dark=coarse-only of the
            # SAME Gaussians) so both "how far does fine motion pull this
            # cluster from its own skeleton" and "do the two clusters
            # visually separate from each other" are visible at once.
            full_means, _ = model.compute_poses_fg(ts)  # (G_fg, 1, 3), actual (coarse+fine) positions
            coarse_means, _ = compute_poses_fg_coarse_only(model, ts)  # (G_fg, 1, 3), coarse-only positions

            overlay_img = Image.fromarray(full_img)
            draw = ImageDraw.Draw(overlay_img)
            legend_y = 10
            for cid, (color_full, color_coarse) in zip(target_clusters, CLUSTER_HUES):
                idx = overlay_idx_by_cluster[cid]
                pixels_full, valid_full = project_world_points(full_means[:, 0][idx], w2c_novel, K, W, H)
                pixels_coarse, valid_coarse = project_world_points(coarse_means[:, 0][idx], w2c_novel, K, W, H)
                draw_points(draw, pixels_coarse, valid_coarse, color_coarse, args.marker_radius)
                draw_points(draw, pixels_full, valid_full, color_full, args.marker_radius)

                r = args.marker_radius
                draw.ellipse((10, legend_y, 10 + 2 * r, legend_y + 2 * r), fill=color_full)
                draw.text((20 + 2 * r, legend_y - 4), f"cluster {cid} actual", fill=(255, 255, 255))
                legend_y += 18
                draw.ellipse((10, legend_y, 10 + 2 * r, legend_y + 2 * r), fill=color_coarse)
                draw.text((20 + 2 * r, legend_y - 4), f"cluster {cid} coarse-only", fill=(255, 255, 255))
                legend_y += 22
            overlay_img.save(overlay_dir / f"frame{frame_idx:04d}.png")

            print(f"  frame {frame_idx:04d}: saved full/coarse_only/overlay")

    print(f"Done. Outputs in {view_dir}")


if __name__ == "__main__":
    main()

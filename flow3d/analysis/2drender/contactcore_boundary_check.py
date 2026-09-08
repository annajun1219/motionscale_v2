#!/usr/bin/env python3
"""
Visualize a cluster pair's contact core vs. boundary patch (minus core), read
straight from a build_cluster_graph_mesh.py `boundary_patch.pt`, rendered
from one fixed novel view and compared across one or more checkpoints.

Only the two requested clusters are drawn -- everything else in the scene is
left as the plain rendered RGB background (unlike cluster_pairs.py /
novelview2_cluster_pairs.py, which color every cluster).

boundary_patch.pt schema (see build_cluster_graph_mesh.py's
build_contact_core_and_patch / write_boundary_patch_pt)
----------------------------------------------------------
{"pairs": [{cluster_a, cluster_b, global_indices_a, global_indices_b,
            weight_a, weight_b, local_scale_a, local_scale_b}, ...], "meta": {...}}

Each pair's global_indices_{a,b} are the geodesically-dilated boundary patch
(3x the contact core's own radius) for that side, indexed into the model's
foreground Gaussians (`model.fg.params["means"]`) -- the same global-index
space cluster_pairs.py's ClusterInfo.global_indices uses, and stable across
any checkpoint fine-tuned onward from the same base (no Gaussians added,
removed, or reordered), so a boundary_patch.pt built once (e.g. from a
warmup checkpoint) can be applied to later checkpoints unchanged. weight_{a,b}
is a smoothstep falloff from the touching seam and is exactly 1.0 at the
contact core, < 1.0 everywhere else in the patch -- that's what separates
"core" from "patch minus core" below.

Coloring
--------
    cluster-a contact core              -> red
    cluster-a boundary patch minus core -> yellow
    cluster-b contact core              -> blue
    cluster-b boundary patch minus core -> purple

Example
-------
    python flow3d/analysis/2drender/contactcore_boundary_check.py \\
        --run baseline1500_samecluster=outputs/davis/camel/2026_09_01_02_39_28__baseline1500_samecluster/checkpoints/last.ckpt \\
        --run gnn=outputs/davis/camel/2026_09_01_03_58_01__gnn/checkpoints/last.ckpt \\
        --boundary-patch-path outputs/davis/camel/2026_08_31_16_18_46__warmup_1500epoch/analysis/cluster_graph_mesh/boundary_patch.pt \\
        --config configs/davis/default.yaml --seq_name camel \\
        --cluster-a 12 --cluster-b 19 \\
        --frames 60,70,80 --yaw-deg 20 --pitch-deg -10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from flow3d.analysis.cluster_pairs import (
    _get_dynamic_fg_means,
    _load_overlay_font,
    _project_world_points,
)
from flow3d.analysis.build_cluster_graph import load_model_and_clusters
from render_output_novelview2 import angle_tag, load_dataset, orbit_camera, parse_frames

CORE_WEIGHT_THRESHOLD = 0.999  # weight_{a,b} == 1.0 exactly at the contact core
COLOR_A_CORE = (220, 30, 30)      # red
COLOR_A_PATCH = (240, 200, 30)    # yellow
COLOR_B_CORE = (30, 60, 220)      # blue
COLOR_B_PATCH = (150, 60, 200)    # purple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=CKPT_PATH",
        help="A checkpoint to render, as LABEL=path/to/last.ckpt. Repeat for "
             "multiple checkpoints (rendered as separate rows in the grid).",
    )
    parser.add_argument(
        "--boundary-patch-path", "--boundary_patch_path", dest="boundary_patch_path",
        type=Path, required=True,
        help="A build_cluster_graph_mesh.py boundary_patch.pt (e.g. "
             "<warmup_work_dir>/analysis/cluster_graph_mesh/boundary_patch.pt).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument("--cluster-a", "--cluster_a", dest="cluster_a", type=int, default=12)
    parser.add_argument("--cluster-b", "--cluster_b", dest="cluster_b", type=int, default=19)
    parser.add_argument(
        "--ref_frame", type=int, default=None,
        help="Training frame whose camera the fixed novel view orbits around. "
             "Default: the middle training frame.",
    )
    parser.add_argument("--yaw-deg", "--yaw_deg", dest="yaw_deg", type=float, default=20.0)
    parser.add_argument("--pitch-deg", "--pitch_deg", dest="pitch_deg", type=float, default=-10.0)
    parser.add_argument(
        "--orbit-radius-scale", "--orbit_radius_scale", dest="orbit_radius_scale",
        type=float, default=1.0,
    )
    parser.add_argument("--frames", type=str, default="60,70,80")
    parser.add_argument("--image-width", "--image_width", dest="image_width", type=int, default=None)
    parser.add_argument("--image-height", "--image_height", dest="image_height", type=int, default=None)
    parser.add_argument(
        "--min-cluster-size", "--min_cluster_size", dest="min_cluster_size",
        type=int, default=20,
    )
    parser.add_argument(
        "--point-radius", "--point_radius", dest="point_radius", type=int, default=5,
        help="Pixel radius of each drawn Gaussian dot.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def parse_runs(run_args: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for entry in run_args:
        if "=" not in entry:
            raise ValueError(f"--run must be LABEL=CKPT_PATH, got: {entry!r}")
        label, ckpt_str = entry.split("=", 1)
        label = label.strip()
        ckpt = Path(ckpt_str.strip()).expanduser().resolve()
        if not label:
            raise ValueError(f"--run has an empty label: {entry!r}")
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        runs.append((label, ckpt))
    return runs


def find_pair(boundary_patch: dict, cluster_a: int, cluster_b: int) -> dict:
    for pair in boundary_patch["pairs"]:
        if {pair["cluster_a"], pair["cluster_b"]} == {cluster_a, cluster_b}:
            return pair
    raise ValueError(
        f"No pair for clusters {{{cluster_a}, {cluster_b}}} in the given "
        f"boundary_patch.pt (available pairs: "
        f"{[(p['cluster_a'], p['cluster_b']) for p in boundary_patch['pairs']]})."
    )


def split_core_and_patch(
    pair: dict, cluster_a: int, cluster_b: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (core_a, patch_minus_core_a, core_b, patch_minus_core_b) as
    global fg-Gaussian index tensors."""
    if pair["cluster_a"] == cluster_a:
        idx_a, w_a = pair["global_indices_a"], pair["weight_a"]
        idx_b, w_b = pair["global_indices_b"], pair["weight_b"]
    else:
        idx_a, w_a = pair["global_indices_b"], pair["weight_b"]
        idx_b, w_b = pair["global_indices_a"], pair["weight_a"]

    core_a_mask = w_a >= CORE_WEIGHT_THRESHOLD
    core_b_mask = w_b >= CORE_WEIGHT_THRESHOLD
    return (
        idx_a[core_a_mask], idx_a[~core_a_mask],
        idx_b[core_b_mask], idx_b[~core_b_mask],
    )


def draw_points(
    draw: ImageDraw.ImageDraw,
    points_xy: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> int:
    count = 0
    for x, y in points_xy[valid]:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color, outline=(0, 0, 0), width=1,
        )
        count += 1
    return count


def main() -> None:
    args = build_parser().parse_args()

    if args.orbit_radius_scale <= 0:
        raise ValueError("--orbit-radius-scale must be > 0")
    if args.cluster_a == args.cluster_b:
        raise ValueError("--cluster-a and --cluster-b must differ.")
    runs = parse_runs(args.run)
    if not args.boundary_patch_path.is_file():
        raise FileNotFoundError(f"boundary_patch.pt not found: {args.boundary_patch_path}")

    out_dir = (
        args.out_dir if args.out_dir is not None
        else runs[0][1].parents[1] / "contactcore_boundary_check"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_patch = torch.load(args.boundary_patch_path, map_location="cpu")
    pair = find_pair(boundary_patch, args.cluster_a, args.cluster_b)
    core_a, patch_a, core_b, patch_b = split_core_and_patch(pair, args.cluster_a, args.cluster_b)
    print(
        f"cluster {args.cluster_a}: {core_a.numel()} core + {patch_a.numel()} patch-minus-core points\n"
        f"cluster {args.cluster_b}: {core_b.numel()} core + {patch_b.numel()} patch-minus-core points"
    )

    dataset = load_dataset(args.config, args.seq_name)
    frames = parse_frames(args.frames, dataset.num_frames)
    ref_frame = args.ref_frame if args.ref_frame is not None else dataset.num_frames // 2
    if not (0 <= ref_frame < dataset.num_frames):
        raise ValueError(f"--ref_frame {ref_frame} out of range [0, {dataset.num_frames}).")

    view_name = f"yaw{angle_tag(args.yaw_deg)}_pitch{angle_tag(args.pitch_deg)}"
    font = _load_overlay_font(16)

    images: dict[tuple[str, int], Image.Image] = {}
    point_counts: dict[str, dict] = {}

    for label, ckpt in runs:
        print(f"=== {label} ({ckpt}) ===")
        model, _clusters, _filtered = load_model_and_clusters(
            work_dir=out_dir, ckpt=ckpt, device_name=args.device,
            min_cluster_size=args.min_cluster_size,
        )
        device = model.fg.params["means"].device
        idx_a_core = core_a.to(device)
        idx_a_patch = patch_a.to(device)
        idx_b_core = core_b.to(device)
        idx_b_patch = patch_b.to(device)

        with torch.no_grad():
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

            fg_means = model.compute_poses_fg(torch.tensor([ref_frame], device=device))[0]
            pivot = fg_means[:, 0, :].mean(dim=0)
            w2c = orbit_camera(w2c_ref, pivot, args.yaw_deg, args.pitch_deg, args.orbit_radius_scale)

            for frame_idx in frames:
                render_output = model.render(
                    frame_idx, w2c[None], K[None], img_wh,
                    return_depth=True, use_learned_poses=False,
                )
                rgb = render_output["img"][0].detach().float().cpu().numpy()
                if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                image = Image.fromarray(rgb_uint8)
                draw = ImageDraw.Draw(image)

                dynamic_means = _get_dynamic_fg_means(model, frame_idx)
                drawn = {}
                for name, idx, color in (
                    (f"cluster{args.cluster_a}_patch", idx_a_patch, COLOR_A_PATCH),
                    (f"cluster{args.cluster_a}_core", idx_a_core, COLOR_A_CORE),
                    (f"cluster{args.cluster_b}_patch", idx_b_patch, COLOR_B_PATCH),
                    (f"cluster{args.cluster_b}_core", idx_b_core, COLOR_B_CORE),
                ):
                    if idx.numel() == 0:
                        drawn[name] = 0
                        continue
                    pts = dynamic_means[idx]
                    pixels, valid = _project_world_points(pts, w2c, K, W, H)
                    drawn[name] = draw_points(draw, pixels, valid, color, args.point_radius)

                title = (
                    f"{label}  frame {frame_idx}  "
                    f"cluster{args.cluster_a}/{args.cluster_b} core+patch  "
                    f"bg={view_name}"
                )
                draw.rectangle((0, 0, W, 22), fill=(0, 0, 0))
                draw.text((4, 3), title, fill=(255, 255, 255), font=font)

                image.save(out_dir / f"{label}_frame{frame_idx:04d}.png")
                images[(label, frame_idx)] = image
                point_counts[f"{label}_frame{frame_idx:04d}"] = drawn

        del model
        torch.cuda.empty_cache()

    # ---- assemble comparison grid ----
    n_rows, n_cols = len(runs), len(frames)
    cell_w_in, cell_h_in = 3.4, 3.4 * (480 / 854)
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(cell_w_in * n_cols, cell_h_in * n_rows + 0.6),
    )
    for r, (label, _) in enumerate(runs):
        for c, frame_idx in enumerate(frames):
            ax = axes[r][c]
            ax.imshow(images[(label, frame_idx)])
            ax.axis("off")
            if r == 0:
                ax.set_title(f"frame {frame_idx}", fontsize=11)
        axes[r][0].text(
            -0.06, 0.5, label, transform=axes[r][0].transAxes,
            rotation=90, va="center", ha="right", fontsize=11, fontweight="bold",
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"cluster {args.cluster_a} core",
                   markerfacecolor=tuple(v / 255 for v in COLOR_A_CORE), markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label=f"cluster {args.cluster_a} patch\\core",
                   markerfacecolor=tuple(v / 255 for v in COLOR_A_PATCH), markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label=f"cluster {args.cluster_b} core",
                   markerfacecolor=tuple(v / 255 for v in COLOR_B_CORE), markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label=f"cluster {args.cluster_b} patch\\core",
                   markerfacecolor=tuple(v / 255 for v in COLOR_B_PATCH), markersize=10),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        f"contact core vs. boundary patch -- cluster {args.cluster_a}/{args.cluster_b} -- bg={view_name}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.03, 0.05, 1, 0.95])
    grid_path = out_dir / f"contactcore_boundary_grid_cluster{args.cluster_a}_{args.cluster_b}.png"
    fig.savefig(grid_path, dpi=150)
    print(f"Saved {grid_path}")

    (out_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "boundary_patch_path": str(args.boundary_patch_path),
                "cluster_a": args.cluster_a,
                "cluster_b": args.cluster_b,
                "core_weight_threshold": CORE_WEIGHT_THRESHOLD,
                "ref_frame": ref_frame,
                "view": view_name,
                "yaw_deg": args.yaw_deg,
                "pitch_deg": args.pitch_deg,
                "frames": frames,
                "runs": [{"label": label, "ckpt": str(ckpt)} for label, ckpt in runs],
                "point_counts": point_counts,
                "grid": str(grid_path),
            },
            indent=2,
        )
    )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

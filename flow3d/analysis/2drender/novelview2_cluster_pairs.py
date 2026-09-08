#!/usr/bin/env python3
"""
Cluster-overlay videos for each of render_output_novelview2.py's fixed novel
viewpoints.

Motivation
----------
render_output_novelview2.py orbits one representative training camera by a
(yaw, pitch) grid to sanity-check reconstruction quality from angles that
have no ground-truth image to compare against. cluster_pairs.py separately
renders a `clusters_3d_rotation.mp4` that overlays each Gaussian cluster's
canonical points (colored dots) and cluster-ID label on top of the rendered
RGB, following the *training* cameras frame by frame.

This script combines the two: for every fixed novel camera in
render_output_novelview2.py's (yaw, pitch) grid, it produces a
cluster-overlay video analogous to `clusters_3d_rotation.mp4`, but seen from
that one fixed novel angle across the requested training frames -- so you
can see how the clusters (and their boundaries/labels) look from off-axis
viewpoints, not just from the real training trajectory.

Each novel camera is fixed in world space (not re-derived per frame), so its
video is a "bullet time" of the reconstructed motion, with per-Gaussian
cluster membership drawn on top, seen from that one novel angle.

Example
-------
    python flow3d/analysis/2drender/novelview2_cluster_pairs.py \\
        --ckpt outputs/davis/camel/2026_07_24_15_16_40__camel_simple_contact_ft50/checkpoints/last.ckpt \\
        --config configs/davis/default.yaml \\
        --seq_name camel
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

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow3d.analysis.cluster_pairs import ClusterInfo, _render_cluster_overlay_frame
from flow3d.analysis.build_cluster_graph import load_model_and_clusters

from render_output_novelview2 import (
    angle_tag,
    load_dataset,
    orbit_camera,
    parse_float_list,
    parse_frames,
)


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
        "--yaw-deg", "--yaw_deg", dest="yaw_deg", type=str,
        default="-30,-25,-20,-10,0,10,20,25,30",
    )
    parser.add_argument(
        "--pitch-deg", "--pitch_deg", dest="pitch_deg", type=str,
        default="-15,-10,0,10,15",
    )
    parser.add_argument(
        "--orbit-radius-scale", "--orbit_radius_scale", dest="orbit_radius_scale",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--frames", type=str, default="all",
        help='Comma list of training frame indices to render through each novel '
             'camera, or "all" (default: every training frame).',
    )
    parser.add_argument(
        "--image_stride", type=int, default=10,
        help="Also dump a standalone cluster-overlay PNG every N frames. Does not "
             "affect which frames go into the videos.",
    )
    parser.add_argument(
        "--no_video", action="store_true",
        help="Skip stitching per-view videos; only dump the strided PNGs.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frame rate for the per-view output videos.")
    parser.add_argument("--image-width", "--image_width", dest="image_width", type=int, default=None)
    parser.add_argument("--image-height", "--image_height", dest="image_height", type=int, default=None)
    parser.add_argument(
        "--min-cluster-size", "--min_cluster_size", dest="min_cluster_size",
        type=int, default=20, help="Clusters with fewer Gaussians than this are dropped.",
    )
    parser.add_argument(
        "--max-points-per-cluster", "--max_points_per_cluster", dest="max_points_per_cluster",
        type=int, default=2000,
        help="Max Gaussian points drawn per cluster in the overlay. Set <= 0 to draw all.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.orbit_radius_scale <= 0:
        raise ValueError("--orbit-radius-scale must be > 0")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    out_dir = args.out_dir if args.out_dir is not None else args.ckpt.parents[1] / "novelview2_cluster_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    model, clusters, filtered_ids = load_model_and_clusters(
        work_dir=out_dir,
        ckpt=args.ckpt,
        device_name=args.device,
        min_cluster_size=args.min_cluster_size,
    )
    device = model.fg.params["means"].device
    print(f"Valid clusters: {len(clusters)} (filtered {len(filtered_ids)} below "
          f"--min-cluster-size={args.min_cluster_size})")
    print(f"Representative camera: training frame {ref_frame} (of {dataset.num_frames})")
    print(f"Rendering {len(frames)} frame(s) through each of "
          f"{len(yaw_values) * len(pitch_values)} novel views "
          f"(yaw={yaw_values}, pitch={pitch_values})")

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

        # Scene pivot: foreground centroid at the representative frame, fixed
        # for every novel view (the camera stays put in world space; only the
        # scene animates as `frames` are rendered through it). Mirrors
        # render_output_novelview2.py's pivot exactly, so the two scripts'
        # novel cameras line up.
        fg_means = model.compute_poses_fg(torch.tensor([ref_frame], device=device))[0]
        pivot = fg_means[:, 0, :].mean(dim=0)

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

        # Same color-by-cluster-id assignment as cluster_pairs.py's
        # clusters_3d_rotation.mp4, so cluster colors match across scripts.
        cmap = plt.get_cmap("tab20")
        color_by_id = {
            cluster.cluster_id: cmap(index % cmap.N)
            for index, cluster in enumerate(clusters)
        }

        rng = np.random.default_rng(0)
        sampled_local_indices_by_id: dict[int, torch.Tensor] = {}
        for cluster in clusters:
            count = cluster.size
            if args.max_points_per_cluster > 0 and count > args.max_points_per_cluster:
                chosen = np.sort(
                    rng.choice(count, size=args.max_points_per_cluster, replace=False)
                )
            else:
                chosen = np.arange(count, dtype=np.int64)
            sampled_local_indices_by_id[cluster.cluster_id] = torch.as_tensor(
                chosen, dtype=torch.long, device=cluster.global_indices.device,
            )

        summary = []
        for view_index, (name, v) in enumerate(novel_views.items(), start=1):
            writer = None
            if not args.no_video:
                video_path = out_dir / f"{name}_clusters_3d.mp4"
                writer = imageio.get_writer(video_path, fps=args.fps)

            for frame_idx in frames:
                frame = _render_cluster_overlay_frame(
                    model=model,
                    clusters=clusters,
                    color_by_id=color_by_id,
                    sampled_local_indices_by_id=sampled_local_indices_by_id,
                    frame_index=frame_idx,
                    w2c=v["w2c"],
                    intrinsic=K,
                    image_size=img_wh,
                )

                if writer is not None:
                    writer.append_data(frame)

                if frame_idx in image_frames:
                    view_dir = out_dir / name
                    view_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"frame{frame_idx:04d}.png"
                    imageio.imwrite(view_dir / fname, frame)
                    summary.append({
                        "view": name,
                        "frame_idx": frame_idx,
                        "yaw_deg": v["yaw_deg"],
                        "pitch_deg": v["pitch_deg"],
                        "file": f"{name}/{fname}",
                    })

            if writer is not None:
                writer.close()
                print(f"  [{view_index:03d}/{len(novel_views):03d}] saved {video_path} "
                      f"({len(frames)} frames @ {args.fps}fps)")
            else:
                print(f"  [{view_index:03d}/{len(novel_views):03d}] {name} done "
                      f"({len(frames)} frames, no video)")

    (out_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "ref_frame": ref_frame,
                "frames": frames,
                "image_frames": sorted(image_frames),
                "video": (not args.no_video),
                "fps": args.fps,
                "min_cluster_size": args.min_cluster_size,
                "valid_cluster_ids": [cluster.cluster_id for cluster in clusters],
                "filtered_cluster_ids": filtered_ids,
                "views": summary,
            },
            indent=2,
        )
    )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

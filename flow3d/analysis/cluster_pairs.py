#!/usr/bin/env python3
"""
Recovered MotionScale cluster-boundary relation generator.

Purpose
-------
Reconstructs the missing upstream script that creates:

    <work-dir>/analysis/all_cluster_boundary_relations/fixed_boundary_indices.pt

The reconstruction is based on the surviving analysis_report.json and the
saved fixed_boundary_indices.pt format.

Recovered default rules
-----------------------
1. Raw cluster IDs:
       model.fg.get_cluster_ids()

2. Valid cluster:
       number of Gaussians >= 20

3. Candidate cluster pairs:
       canonical cluster-center kNN, k=8
       Pair set is symmetrized and deduplicated.

4. Contact threshold:
       median within-cluster nearest-neighbour spacing * 3.0

5. Retained adjacency:
       in both directions, at least 20 Gaussian points have nearest-neighbour
       distance <= contact threshold.

6. Fixed boundary:
       for each retained pair and each cluster, select the nearest
       clamp(round(cluster_size * 0.1), 20, 300) Gaussians to the other
       cluster, with preference for points inside the contact threshold.

7. Fixed canonical correspondence pairs:
       union of canonical A->B and B->A nearest-neighbour correspondences
       restricted to the saved boundary sets.

The script also writes concise diagnostic CSV/JSON files. It does not attempt
to reproduce every historical diagnostic metric exactly; the critical output
format used by downstream scripts is preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from flow3d.renderer import Renderer


EPS = 1e-12


@dataclass
class ClusterInfo:
    cluster_id: int
    global_indices: torch.Tensor
    canonical_points: torch.Tensor
    canonical_center: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.global_indices.numel())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover cluster adjacency and fixed canonical boundary indices "
            "for MotionScale."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--center-neighbor-k", type=int, default=8)
    parser.add_argument("--max-center-distance", type=float, default=None)

    parser.add_argument(
        "--contact-spacing-multiplier",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--contact-distance",
        type=float,
        default=None,
        help=(
            "Explicit contact threshold. If omitted, use median sampled "
            "within-cluster NN spacing times --contact-spacing-multiplier."
        ),
    )
    parser.add_argument(
        "--spacing-sample-limit",
        type=int,
        default=20000,
        help="Maximum total points sampled for spacing estimation.",
    )
    parser.add_argument(
        "--min-close-gaussians-each-direction",
        type=int,
        default=20,
    )

    parser.add_argument("--boundary-fraction", type=float, default=0.10)
    parser.add_argument("--boundary-min-gaussians", type=int, default=20)
    parser.add_argument("--boundary-max-gaussians", type=int, default=300)
    parser.add_argument(
        "--boundary-max-distance",
        type=float,
        default=None,
        help=(
            "Optional boundary distance ceiling. Default: resolved contact "
            "distance."
        ),
    )

    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help=(
            'Optional raw pair subset, e.g. "0-15,4-9". '
            'Use "all" to test every valid cluster pair. '
            "Default: canonical-center kNN candidates."
        ),
    )
    parser.add_argument(
        "--visualize-max-points-per-cluster",
        type=int,
        default=2000,
        help=(
            "Maximum Gaussian points drawn per cluster in diagnostic figures. "
            "Set <= 0 to draw every point."
        ),
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable cluster visualization output.",
    )
    parser.add_argument(
        "--no-cluster-video",
        action="store_true",
        help=(
            "Skip the rotating 3D cluster video. If visualization is "
            "enabled, the static PNG is still produced."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=12,
        help="Frames per second for the rotating cluster video.",
    )
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=120,
        help="Number of frames in the rotating cluster video.",
    )
    parser.add_argument(
        "--video-elevation",
        type=float,
        default=20.0,
        help="Fixed elevation angle in degrees for the rotating video.",
    )
    parser.add_argument(
        "--video-start-azim",
        type=float,
        default=30.0,
        help="Starting azimuth angle in degrees for the rotating video.",
    )
    parser.add_argument(
        "--video-end-azim",
        type=float,
        default=390.0,
        help="Ending azimuth angle in degrees for the rotating video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing fixed_boundary_indices.pt.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.min_cluster_size < 1:
        raise ValueError("--min-cluster-size must be >= 1")
    if args.center_neighbor_k < 1:
        raise ValueError("--center-neighbor-k must be >= 1")
    if args.contact_spacing_multiplier <= 0:
        raise ValueError("--contact-spacing-multiplier must be > 0")
    if args.min_close_gaussians_each_direction < 1:
        raise ValueError(
            "--min-close-gaussians-each-direction must be >= 1"
        )
    if not 0 < args.boundary_fraction <= 1:
        raise ValueError("--boundary-fraction must be in (0, 1]")
    if args.boundary_min_gaussians < 1:
        raise ValueError("--boundary-min-gaussians must be >= 1")
    if args.boundary_max_gaussians < args.boundary_min_gaussians:
        raise ValueError(
            "--boundary-max-gaussians must be >= --boundary-min-gaussians"
        )
    if args.spacing_sample_limit < 1:
        raise ValueError("--spacing-sample-limit must be >= 1")
    if args.visualize_max_points_per_cluster == 0:
        raise ValueError(
            "--visualize-max-points-per-cluster must be positive or negative "
            "to request all points."
        )
    if args.video_fps < 1:
        raise ValueError("--video-fps must be >= 1")
    if args.video_num_frames < 2:
        raise ValueError("--video-num-frames must be >= 2")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _sample_points_for_visualization(
    points: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    chosen = rng.choice(len(points), size=max_points, replace=False)
    return points[chosen]


def _set_axes_equal_3d(ax: Any, all_points: np.ndarray) -> None:
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float((maximum - minimum).max()) * 0.5, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _figure_to_rgb_array(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, :3])


def _prepare_cluster_visual_data(
    clusters: list[ClusterInfo],
    max_points_per_cluster: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, Any], np.ndarray]:
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("tab20")
    cluster_ids = [cluster.cluster_id for cluster in clusters]
    color_by_id = {
        cluster_id: cmap(index % cmap.N)
        for index, cluster_id in enumerate(cluster_ids)
    }

    sampled_by_id: dict[int, np.ndarray] = {}
    center_by_id: dict[int, np.ndarray] = {}
    all_sampled: list[np.ndarray] = []

    for cluster in clusters:
        points = cluster.canonical_points.detach().float().cpu().numpy()
        sampled = _sample_points_for_visualization(
            points,
            max_points_per_cluster,
            rng,
        )
        sampled_by_id[cluster.cluster_id] = sampled
        center = cluster.canonical_center.detach().float().cpu().numpy()
        center_by_id[cluster.cluster_id] = center
        all_sampled.append(sampled)

    all_points = np.concatenate(all_sampled, axis=0)
    return sampled_by_id, center_by_id, color_by_id, all_points


def _draw_cluster_scene(
    ax: Any,
    clusters: list[ClusterInfo],
    sampled_by_id: dict[int, np.ndarray],
    center_by_id: dict[int, np.ndarray],
    color_by_id: dict[int, Any],
    all_points: np.ndarray,
) -> None:
    for cluster in clusters:
        cluster_id = cluster.cluster_id
        points = sampled_by_id[cluster_id]
        color = color_by_id[cluster_id]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2.0,
            alpha=0.65,
            color=color,
            rasterized=True,
        )
        center = center_by_id[cluster_id]
        ax.text(
            center[0],
            center[1],
            center[2],
            str(cluster_id),
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                boxstyle="round,pad=0.15",
            ),
        )
    _set_axes_equal_3d(ax, all_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Canonical Gaussian clusters ({len(clusters)} valid clusters)"
    )



def _save_multiview_cluster_png(
    output_path: Path,
    clusters: list[ClusterInfo],
    sampled_by_id: dict[int, np.ndarray],
    center_by_id: dict[int, np.ndarray],
    color_by_id: dict[int, Any],
    all_points: np.ndarray,
) -> None:
    views = [
        (20, 35, "Perspective"),
        (20, 125, "Left / Back"),
        (20, 215, "Back"),
        (20, 305, "Right / Front"),
        (65, 35, "Top-front"),
        (65, 215, "Top-back"),
    ]

    fig = plt.figure(figsize=(18, 11))
    for subplot_index, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 3, subplot_index, projection="3d")
        _draw_cluster_scene(
            ax=ax,
            clusters=clusters,
            sampled_by_id=sampled_by_id,
            center_by_id=center_by_id,
            color_by_id=color_by_id,
            all_points=all_points,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
    fig.suptitle(
        f"Canonical Gaussian clusters ({len(clusters)} valid clusters)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _get_camera_w2cs(model: Any) -> torch.Tensor:
    camera_poses = getattr(model, "camera_poses", None)
    get_camera_matrix = (
        getattr(camera_poses, "get_camera_matrix", None)
        if camera_poses is not None
        else None
    )
    if callable(get_camera_matrix):
        matrices = get_camera_matrix()
        if isinstance(matrices, torch.Tensor):
            return matrices

    matrices = getattr(model, "w2cs", None)
    if isinstance(matrices, torch.Tensor):
        return matrices

    raise RuntimeError("Could not obtain per-frame camera matrices.")


def _get_dynamic_fg_means(model: Any, frame_index: int) -> torch.Tensor:
    device = model.fg.params["means"].device
    frame_tensor = torch.tensor([frame_index], device=device, dtype=torch.long)

    if hasattr(model, "compute_poses_fg"):
        means, _ = model.compute_poses_fg(frame_tensor)
        if means.ndim == 3:
            return means[:, 0]
        return means

    means, _ = model.compute_poses_all(frame_tensor)
    means = means[: model.num_fg_gaussians]
    if means.ndim == 3:
        return means[:, 0]
    return means


def _project_world_points(
    points: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    ones = torch.ones(
        (points.shape[0], 1),
        dtype=points.dtype,
        device=points.device,
    )
    homogeneous = torch.cat([points, ones], dim=-1)
    camera_points = (w2c @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    projected = (intrinsic @ camera_points.T).T
    pixels = projected[:, :2] / depth[:, None].clamp_min(1e-8)
    valid = (
        (depth > 1e-8)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return (
        pixels.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
    )


def _load_overlay_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _render_cluster_overlay_frame(
    model: Any,
    clusters: list[ClusterInfo],
    color_by_id: dict[int, Any],
    sampled_local_indices_by_id: dict[int, torch.Tensor],
    frame_index: int,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    render_output = model.render(
        frame_index,
        w2c[None],
        intrinsic[None],
        image_size,
        return_depth=True,
        use_learned_poses=False,
    )
    rgb = render_output["img"][0].detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(image)
    font = _load_overlay_font(14)
    dynamic_means = _get_dynamic_fg_means(model, frame_index)

    for cluster in clusters:
        cluster_id = cluster.cluster_id
        color_float = color_by_id[cluster_id][:3]
        color = tuple(int(round(value * 255)) for value in color_float)

        sampled_local = sampled_local_indices_by_id[cluster_id]
        sampled_global = cluster.global_indices[sampled_local]
        sampled_points = dynamic_means[sampled_global]
        pixels, valid = _project_world_points(
            sampled_points,
            w2c,
            intrinsic,
            width,
            height,
        )
        for x, y in pixels[valid]:
            radius = 2
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
            )

        cluster_points = dynamic_means[cluster.global_indices]
        center = cluster_points.mean(dim=0, keepdim=True)
        center_pixel, center_valid = _project_world_points(
            center,
            w2c,
            intrinsic,
            width,
            height,
        )
        if center_valid[0]:
            x, y = center_pixel[0]
            text = str(cluster_id)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except Exception:
                text_width, text_height = 12, 12
            left = x - text_width / 2 - 2
            top = y - text_height / 2 - 2
            right = x + text_width / 2 + 2
            bottom = y + text_height / 2 + 2
            draw.rounded_rectangle(
                (left, top, right, bottom),
                radius=3,
                fill=(255, 255, 255),
            )
            draw.text(
                (x - text_width / 2, y - text_height / 2),
                text,
                fill=(0, 0, 0),
                font=font,
            )

    return np.asarray(image)


def save_cluster_visualizations(
    model: Any,
    clusters: list[ClusterInfo],
    output_dir: Path,
    max_points_per_cluster: int,
    make_video: bool,
    video_fps: int,
    video_num_frames: int,
    video_elevation: float,
    video_start_azim: float,
    video_end_azim: float,
) -> list[str]:
    """Save the original static 3D figure and a temporal camera-view video."""
    del video_elevation, video_start_azim, video_end_azim

    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    sampled_by_id, center_by_id, color_by_id, all_points = (
        _prepare_cluster_visual_data(
            clusters=clusters,
            max_points_per_cluster=max_points_per_cluster,
        )
    )

    saved_files: list[str] = []

    # Save a multi-view static PNG so clusters hidden in a single angle
    # remain visible from additional viewpoints.
    path = visualization_dir / "clusters_3d.png"
    _save_multiview_cluster_png(
        output_path=path,
        clusters=clusters,
        sampled_by_id=sampled_by_id,
        center_by_id=center_by_id,
        color_by_id=color_by_id,
        all_points=all_points,
    )
    saved_files.append(str(path))

    if make_video:
        rng = np.random.default_rng(0)
        sampled_local_indices_by_id: dict[int, torch.Tensor] = {}
        for cluster in clusters:
            count = cluster.size
            if max_points_per_cluster > 0 and count > max_points_per_cluster:
                chosen = np.sort(
                    rng.choice(
                        count,
                        size=max_points_per_cluster,
                        replace=False,
                    )
                )
            else:
                chosen = np.arange(count, dtype=np.int64)
            sampled_local_indices_by_id[cluster.cluster_id] = torch.as_tensor(
                chosen,
                dtype=torch.long,
                device=cluster.global_indices.device,
            )

        total_frames = int(model.num_frames)
        output_frame_count = min(video_num_frames, total_frames)
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            num=output_frame_count,
            dtype=np.int64,
        )
        frame_indices = np.unique(frame_indices)

        w2cs = _get_camera_w2cs(model).to(model.fg.params["means"].device)
        intrinsics = model.Ks.to(model.fg.params["means"].device)

        principal_x = float(intrinsics[0, 0, 2].item())
        principal_y = float(intrinsics[0, 1, 2].item())
        width = max(int(round(principal_x * 2.0)), 2)
        height = max(int(round(principal_y * 2.0)), 2)
        image_size = (width, height)

        mp4_path = visualization_dir / "clusters_3d_rotation.mp4"
        try:
            with imageio.get_writer(mp4_path, fps=video_fps) as writer:
                for output_index, frame_index in enumerate(frame_indices, start=1):
                    frame = _render_cluster_overlay_frame(
                        model=model,
                        clusters=clusters,
                        color_by_id=color_by_id,
                        sampled_local_indices_by_id=sampled_local_indices_by_id,
                        frame_index=int(frame_index),
                        w2c=w2cs[int(frame_index)],
                        intrinsic=intrinsics[int(frame_index)],
                        image_size=image_size,
                    )
                    writer.append_data(frame)
                    print(
                        f"[Cluster video {output_index:03d}/{len(frame_indices):03d}] "
                        f"frame={int(frame_index):04d}"
                    )
            saved_files.append(str(mp4_path))
        except Exception as exc:
            print(f"[Warning] MP4 writing failed: {mp4_path} ({exc})")
            raise

    return saved_files

def parse_manual_pairs(
    text: str,
    valid_ids: list[int],
) -> list[tuple[int, int]]:
    text = text.strip().lower()
    valid_set = set(valid_ids)

    if text == "all":
        return [
            (a, b)
            for i, a in enumerate(valid_ids)
            for b in valid_ids[i + 1 :]
        ]

    pairs: set[tuple[int, int]] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue

        parts = token.replace(":", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair token: {token}")

        a, b = int(parts[0]), int(parts[1])
        if a == b:
            raise ValueError(f"Self pair is invalid: {token}")
        if a not in valid_set or b not in valid_set:
            raise ValueError(
                f"Pair {a}-{b} contains a filtered or absent cluster ID."
            )
        pairs.add((min(a, b), max(a, b)))

    if not pairs:
        raise ValueError("No valid manual pair was supplied.")
    return sorted(pairs)


def build_candidate_pairs(
    clusters: list[ClusterInfo],
    neighbor_k: int,
    max_center_distance: float | None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float]]:
    centers = torch.stack([cluster.canonical_center for cluster in clusters])
    distances = torch.cdist(centers, centers)
    k = min(neighbor_k, len(clusters) - 1)
    neighbor_indices = distances.topk(k + 1, largest=False).indices[:, 1:]

    pairs: set[tuple[int, int]] = set()
    distance_by_pair: dict[tuple[int, int], float] = {}

    for src_local, neighbors in enumerate(neighbor_indices):
        for dst_tensor in neighbors:
            dst_local = int(dst_tensor.item())
            distance = float(distances[src_local, dst_local].item())

            if (
                max_center_distance is not None
                and distance > max_center_distance
            ):
                continue

            a = clusters[src_local].cluster_id
            b = clusters[dst_local].cluster_id
            pair = (min(a, b), max(a, b))
            pairs.add(pair)

            previous = distance_by_pair.get(pair)
            if previous is None or distance < previous:
                distance_by_pair[pair] = distance

    return sorted(pairs), distance_by_pair


def estimate_within_cluster_spacing(
    clusters: list[ClusterInfo],
    sample_limit: int,
) -> tuple[float, int]:
    total_points = sum(cluster.size for cluster in clusters)
    if total_points == 0:
        raise RuntimeError("No valid cluster points for spacing estimation.")

    rng = np.random.default_rng(0)
    per_cluster_budget = max(
        2,
        int(math.ceil(sample_limit / max(len(clusters), 1))),
    )

    nearest_distances: list[np.ndarray] = []
    sampled_count = 0

    for cluster in clusters:
        points = cluster.canonical_points.detach().float().cpu().numpy()
        if len(points) < 2:
            continue

        if len(points) > per_cluster_budget:
            chosen = rng.choice(
                len(points),
                size=per_cluster_budget,
                replace=False,
            )
            query_points = points[chosen]
        else:
            query_points = points

        tree = cKDTree(points)
        distances, _ = tree.query(query_points, k=2)
        second = np.asarray(distances[:, 1], dtype=np.float64)
        second = second[np.isfinite(second) & (second > 0)]

        if second.size:
            nearest_distances.append(second)
            sampled_count += int(second.size)

    if not nearest_distances:
        raise RuntimeError(
            "Could not estimate within-cluster nearest-neighbour spacing."
        )

    values = np.concatenate(nearest_distances)
    return float(np.median(values)), sampled_count


def directed_nn(
    source: np.ndarray,
    destination: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(destination)
    distance, index = tree.query(source, k=1)
    return (
        np.asarray(distance, dtype=np.float64),
        np.asarray(index, dtype=np.int64),
    )


def select_boundary_local_indices(
    distances_to_other: np.ndarray,
    cluster_size: int,
    fraction: float,
    minimum: int,
    maximum: int,
    max_distance: float,
) -> np.ndarray:
    target_count = int(round(cluster_size * fraction))
    target_count = max(minimum, target_count)
    target_count = min(maximum, target_count)
    target_count = min(cluster_size, target_count)

    order = np.argsort(distances_to_other, kind="stable")
    inside = order[distances_to_other[order] <= max_distance]

    if len(inside) >= target_count:
        selected = inside[:target_count]
    else:
        # Preserve the recovered fixed boundary size rule even when fewer
        # points fall inside the threshold by filling from the nearest points.
        selected = order[:target_count]

    return np.sort(selected.astype(np.int64))


def build_fixed_correspondences(
    boundary_points_a: np.ndarray,
    boundary_points_b: np.ndarray,
    boundary_global_a: np.ndarray,
    boundary_global_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distance_ab, index_ab = directed_nn(
        boundary_points_a,
        boundary_points_b,
    )
    distance_ba, index_ba = directed_nn(
        boundary_points_b,
        boundary_points_a,
    )

    del distance_ab, distance_ba

    pair_a = np.concatenate(
        [
            boundary_global_a,
            boundary_global_a[index_ba],
        ]
    )
    pair_b = np.concatenate(
        [
            boundary_global_b[index_ab],
            boundary_global_b,
        ]
    )

    # Deduplicate exact index correspondences while preserving order.
    seen: set[tuple[int, int]] = set()
    unique_a: list[int] = []
    unique_b: list[int] = []

    for a, b in zip(pair_a.tolist(), pair_b.tolist()):
        key = (int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        unique_a.append(key[0])
        unique_b.append(key[1])

    return (
        np.asarray(unique_a, dtype=np.int64),
        np.asarray(unique_b, dtype=np.int64),
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    work_dir = args.work_dir.expanduser().resolve()
    checkpoint = (
        args.ckpt.expanduser().resolve()
        if args.ckpt is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "analysis" / "all_cluster_boundary_relations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pt = output_dir / "fixed_boundary_indices.pt"
    if output_pt.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_pt} already exists. Back it up and pass --overwrite "
            "only when replacement is intended."
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(
            str(checkpoint),
            device,
            work_dir=str(work_dir),
            port=None,
        )
        model = renderer.model
        model.eval()

        canonical_means = model.fg.params["means"]
        cluster_ids = (
            model.fg.get_cluster_ids()
            .reshape(-1)
            .long()
            .to(canonical_means.device)
        )

        if canonical_means.shape[0] != cluster_ids.numel():
            raise RuntimeError(
                "Foreground Gaussian count differs from cluster-ID count."
            )

        raw_ids = sorted(
            int(value)
            for value in torch.unique(cluster_ids).tolist()
        )

        clusters: list[ClusterInfo] = []
        filtered_ids: list[int] = []

        for cluster_id in raw_ids:
            global_indices = torch.where(cluster_ids == cluster_id)[0]
            if global_indices.numel() < args.min_cluster_size:
                filtered_ids.append(cluster_id)
                continue

            points = canonical_means[global_indices]
            clusters.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    global_indices=global_indices,
                    canonical_points=points,
                    canonical_center=points.mean(dim=0),
                )
            )

    if len(clusters) < 2:
        raise RuntimeError(
            "Too few valid clusters. Lower --min-cluster-size."
        )

    cluster_by_id = {
        cluster.cluster_id: cluster
        for cluster in clusters
    }
    valid_ids = sorted(cluster_by_id)

    if args.pairs is not None:
        candidate_pairs = parse_manual_pairs(args.pairs, valid_ids)
        center_distance_by_pair = {}
        for a, b in candidate_pairs:
            center_distance_by_pair[(a, b)] = float(
                torch.linalg.norm(
                    cluster_by_id[a].canonical_center
                    - cluster_by_id[b].canonical_center
                ).item()
            )
        candidate_mode = (
            "all_valid_pairs"
            if args.pairs.strip().lower() == "all"
            else "manual_pairs"
        )
    else:
        candidate_pairs, center_distance_by_pair = build_candidate_pairs(
            clusters=clusters,
            neighbor_k=args.center_neighbor_k,
            max_center_distance=args.max_center_distance,
        )
        candidate_mode = "canonical_center_knn"

    within_spacing, spacing_sample_count = (
        estimate_within_cluster_spacing(
            clusters,
            sample_limit=args.spacing_sample_limit,
        )
    )
    contact_distance = (
        float(args.contact_distance)
        if args.contact_distance is not None
        else (
            within_spacing
            * float(args.contact_spacing_multiplier)
        )
    )
    boundary_max_distance = (
        float(args.boundary_max_distance)
        if args.boundary_max_distance is not None
        else contact_distance
    )

    adjacent_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    saved_pairs: dict[str, dict[str, Any]] = {}
    pair_summaries: list[dict[str, Any]] = []

    print(f"Valid clusters: {len(valid_ids)}")
    print(f"Candidate pairs: {len(candidate_pairs)}")
    print(f"Within-cluster median spacing: {within_spacing:.9f}")
    print(f"Contact distance: {contact_distance:.9f}")

    for pair_index, (cluster_a, cluster_b) in enumerate(
        candidate_pairs,
        start=1,
    ):
        info_a = cluster_by_id[cluster_a]
        info_b = cluster_by_id[cluster_b]

        points_a = (
            info_a.canonical_points.detach().float().cpu().numpy()
        )
        points_b = (
            info_b.canonical_points.detach().float().cpu().numpy()
        )

        distance_ab, index_ab = directed_nn(points_a, points_b)
        distance_ba, index_ba = directed_nn(points_b, points_a)

        close_mask_a = distance_ab <= contact_distance
        close_mask_b = distance_ba <= contact_distance
        close_count_a = int(close_mask_a.sum())
        close_count_b = int(close_mask_b.sum())

        retained = (
            close_count_a
            >= args.min_close_gaussians_each_direction
            and close_count_b
            >= args.min_close_gaussians_each_direction
        )

        common_row = {
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "cluster_size_a": info_a.size,
            "cluster_size_b": info_b.size,
            "canonical_center_distance": center_distance_by_pair[
                (cluster_a, cluster_b)
            ],
            "contact_distance": contact_distance,
            "close_count_a": close_count_a,
            "close_count_b": close_count_b,
            "close_fraction_a": close_count_a / info_a.size,
            "close_fraction_b": close_count_b / info_b.size,
            "median_nn_a_to_b": float(np.median(distance_ab)),
            "median_nn_b_to_a": float(np.median(distance_ba)),
            "p90_nn_a_to_b": float(np.quantile(distance_ab, 0.9)),
            "p90_nn_b_to_a": float(np.quantile(distance_ba, 0.9)),
        }

        if not retained:
            reasons: list[str] = []
            if (
                close_count_a
                < args.min_close_gaussians_each_direction
            ):
                reasons.append("insufficient_close_gaussians_a_to_b")
            if (
                close_count_b
                < args.min_close_gaussians_each_direction
            ):
                reasons.append("insufficient_close_gaussians_b_to_a")

            rejected_rows.append(
                {
                    **common_row,
                    "rejection_reason": ";".join(reasons),
                }
            )
            continue

        local_boundary_a = select_boundary_local_indices(
            distances_to_other=distance_ab,
            cluster_size=info_a.size,
            fraction=args.boundary_fraction,
            minimum=args.boundary_min_gaussians,
            maximum=args.boundary_max_gaussians,
            max_distance=boundary_max_distance,
        )
        local_boundary_b = select_boundary_local_indices(
            distances_to_other=distance_ba,
            cluster_size=info_b.size,
            fraction=args.boundary_fraction,
            minimum=args.boundary_min_gaussians,
            maximum=args.boundary_max_gaussians,
            max_distance=boundary_max_distance,
        )

        boundary_global_a = (
            info_a.global_indices[
                torch.as_tensor(
                    local_boundary_a,
                    device=info_a.global_indices.device,
                )
            ]
            .detach()
            .cpu()
            .long()
        )
        boundary_global_b = (
            info_b.global_indices[
                torch.as_tensor(
                    local_boundary_b,
                    device=info_b.global_indices.device,
                )
            ]
            .detach()
            .cpu()
            .long()
        )

        fixed_pair_a, fixed_pair_b = build_fixed_correspondences(
            boundary_points_a=points_a[local_boundary_a],
            boundary_points_b=points_b[local_boundary_b],
            boundary_global_a=boundary_global_a.numpy(),
            boundary_global_b=boundary_global_b.numpy(),
        )

        boundary_distance_ab, _ = directed_nn(
            points_a[local_boundary_a],
            points_b[local_boundary_b],
        )
        boundary_distance_ba, _ = directed_nn(
            points_b[local_boundary_b],
            points_a[local_boundary_a],
        )
        boundary_balanced = np.concatenate(
            [boundary_distance_ab, boundary_distance_ba]
        )

        pair_key = f"{cluster_a}_{cluster_b}"
        saved_pairs[pair_key] = {
            "cluster_a": int(cluster_a),
            "cluster_b": int(cluster_b),
            "boundary_global_indices_a": boundary_global_a,
            "boundary_global_indices_b": boundary_global_b,
            "fixed_pair_global_indices_a": torch.from_numpy(
                fixed_pair_a
            ).long(),
            "fixed_pair_global_indices_b": torch.from_numpy(
                fixed_pair_b
            ).long(),
        }

        summary_row = {
            **common_row,
            "num_boundary_a": int(boundary_global_a.numel()),
            "num_boundary_b": int(boundary_global_b.numel()),
            "num_fixed_pairs": int(len(fixed_pair_a)),
            "canonical_boundary_median": float(
                np.median(boundary_balanced)
            ),
            "canonical_boundary_p90": float(
                np.quantile(boundary_balanced, 0.9)
            ),
        }
        adjacent_rows.append(summary_row)
        pair_summaries.append(summary_row)

        print(
            f"[{pair_index:03d}/{len(candidate_pairs):03d}] "
            f"retain {cluster_a}-{cluster_b}: "
            f"close=({close_count_a},{close_count_b}), "
            f"boundary=({len(local_boundary_a)},"
            f"{len(local_boundary_b)}), "
            f"fixed_pairs={len(fixed_pair_a)}"
        )

    visualization_files: list[str] = []
    if not args.no_visualization:
        visualization_files = save_cluster_visualizations(
            model=model,
            clusters=clusters,
            output_dir=output_dir,
            max_points_per_cluster=args.visualize_max_points_per_cluster,
            make_video=not args.no_cluster_video,
            video_fps=args.video_fps,
            video_num_frames=args.video_num_frames,
            video_elevation=args.video_elevation,
            video_start_azim=args.video_start_azim,
            video_end_azim=args.video_end_azim,
        )

    torch.save(saved_pairs, output_pt)

    write_csv(output_dir / "adjacent_pairs.csv", adjacent_rows)
    write_csv(output_dir / "rejected_pairs.csv", rejected_rows)
    write_csv(output_dir / "pair_summary.csv", pair_summaries)

    report = {
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "cluster_id_source": "model.fg.get_cluster_ids()",
        "canonical_position_source": "model.fg.params['means']",
        "valid_cluster_count": len(valid_ids),
        "valid_cluster_ids": valid_ids,
        "filtered_cluster_ids": filtered_ids,
        "candidate_pair_count": len(candidate_pairs),
        "retained_edge_count": len(saved_pairs),
        "rejected_pair_count": len(rejected_rows),
        "candidate_pair_mode": candidate_mode,
        "adjacency": {
            "fixed_in_canonical_space": True,
            "center_neighbor_k": (
                args.center_neighbor_k
                if args.pairs is None
                else None
            ),
            "max_center_distance": args.max_center_distance,
            "contact_distance": contact_distance,
            "contact_distance_source": (
                "explicit"
                if args.contact_distance is not None
                else "within_cluster_nn_median"
            ),
            "within_cluster_median_spacing": within_spacing,
            "spacing_sample_count": spacing_sample_count,
            "contact_spacing_multiplier": (
                args.contact_spacing_multiplier
            ),
            "min_close_gaussians_each_direction": (
                args.min_close_gaussians_each_direction
            ),
        },
        "boundary": {
            "indices_fixed_in_canonical_space": True,
            "fraction": args.boundary_fraction,
            "min_gaussians": args.boundary_min_gaussians,
            "max_gaussians": args.boundary_max_gaussians,
            "max_distance": boundary_max_distance,
            "max_distance_source": (
                "explicit"
                if args.boundary_max_distance is not None
                else "contact_distance"
            ),
            "fixed_pair_definition": (
                "Union of canonical A-to-B and B-to-A nearest-neighbour "
                "index correspondences restricted to saved boundary sets."
            ),
        },
        "pair_summaries": pair_summaries,
        "visualization": {
            "enabled": not args.no_visualization,
            "max_points_per_cluster": (
                args.visualize_max_points_per_cluster
            ),
            "files": visualization_files,
            "cluster_video_enabled": (
                (not args.no_visualization) and (not args.no_cluster_video)
            ),
            "video_fps": args.video_fps,
            "video_num_frames": args.video_num_frames,
            "video_elevation": args.video_elevation,
            "video_start_azim": args.video_start_azim,
            "video_end_azim": args.video_end_azim,
            "description": (
                "Canonical-space Gaussian clusters are color-coded by raw "
                "cluster ID and cluster centers are labeled. A static 3D "
                "PNG is saved, and an optional rotating 3D video is also "
                "written to help inspect cluster separation from multiple "
                "viewpoints."
            ),
        },
        "compatibility_note": (
            "fixed_boundary_indices.pt preserves the fields required by "
            "novel_view_structure_analysis.py and related downstream tools."
        ),
        "recovery_note": (
            "This is a functional reconstruction based on surviving outputs; "
            "some historical diagnostic CSV metrics are not reproduced."
        ),
    }

    with (output_dir / "analysis_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("Recovered boundary-relation generation complete")
    print(f"Valid clusters : {len(valid_ids)}")
    print(f"Candidates     : {len(candidate_pairs)}")
    print(f"Retained       : {len(saved_pairs)}")
    print(f"Rejected       : {len(rejected_rows)}")
    print(f"Output PT      : {output_pt}")
    if visualization_files:
        print(
            "Visualizations : "
            f"{output_dir / 'visualizations'}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
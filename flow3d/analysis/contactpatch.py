#!/usr/bin/env python3
"""
Select broad canonical contact patches for retained MotionScale cluster pairs.

Expected input
--------------
A pair file produced by the cluster-boundary relation script:

    <work-dir>/analysis/all_cluster_boundary_relations/fixed_boundary_indices.pt

Each entry must contain at least:

    cluster_a
    cluster_b
    boundary_global_indices_a
    boundary_global_indices_b

The boundary indices are treated only as contact seeds. This script expands
them into wider set-to-set contact patches. It does not preserve fixed
Gaussian-to-Gaussian correspondences.

Selection rule
--------------
For each retained cluster pair (A, B):

1. Load canonical foreground Gaussian positions and raw cluster IDs.
2. Use saved boundary Gaussians as contact seeds.
3. Estimate the canonical nearest-neighbour spacing of each cluster.
4. Expand each seed set inside its own cluster.
5. Keep expansion candidates that are:
      - near the seed region, and
      - still reasonably near the opposite cluster.
6. Enforce minimum/maximum patch sizes deterministically.
7. Build one shared PCA contact frame and fixed 2-D spatial-bin assignments.

Outputs
-------
    contact_patches.pt
    contact_patch_summary.csv
    analysis_report.json
    visualizations/pair_<A>_<B>_3d.png
    visualizations/pair_<A>_<B>_contact_plane.png

The saved patch indices are fixed Gaussian identities in canonical space.
During later training, nearest neighbours between the two patch sets should
be recomputed at each frame; fixed pair correspondences are intentionally not
saved or enforced here.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from flow3d.renderer import Renderer


EPS = 1e-12


@dataclass
class ClusterData:
    cluster_id: int
    global_indices: np.ndarray
    points: np.ndarray

    @property
    def size(self) -> int:
        return int(self.global_indices.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand retained MotionScale cluster boundaries into broad contact patches.",
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
    parser.add_argument(
        "--pair-file",
        type=Path,
        default=None,
        help=(
            "Input fixed_boundary_indices.pt. Default: "
            "<work-dir>/analysis/all_cluster_boundary_relations/"
            "fixed_boundary_indices.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <work-dir>/analysis/contact_patches",
    )

    parser.add_argument(
        "--patch-radius-multiplier",
        type=float,
        default=3.0,
        help=(
            "Same-cluster expansion radius = cluster median NN spacing times "
            "this value."
        ),
    )
    parser.add_argument(
        "--opposite-distance-multiplier",
        type=float,
        default=4.0,
        help=(
            "Candidate must also remain within this multiple of the balanced "
            "pair spacing from the opposite cluster."
        ),
    )
    parser.add_argument(
        "--minimum-opposite-band-multiplier",
        type=float,
        default=2.0,
        help=(
            "The opposite-cluster distance ceiling is never smaller than "
            "this multiple of the cluster spacing."
        ),
    )
    parser.add_argument(
        "--patch-min-gaussians",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--patch-max-gaussians",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--spacing-sample-limit",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--bins-u",
        type=int,
        default=4,
        help="Number of fixed spatial bins along the first contact-plane axis.",
    )
    parser.add_argument(
        "--bins-v",
        type=int,
        default=4,
        help="Number of fixed spatial bins along the second contact-plane axis.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help='Optional subset such as "1-28,41-42". Default: every saved pair.',
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Do not save per-pair contact-patch PNG visualizations.",
    )
    parser.add_argument(
        "--visualization-max-context-points",
        type=int,
        default=3000,
        help=(
            "Maximum number of non-patch context Gaussians drawn per cluster "
            "in each 3-D visualization."
        ),
    )
    parser.add_argument(
        "--visualization-dpi",
        type=int,
        default=180,
        help="DPI of saved visualization images.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.patch_radius_multiplier <= 0:
        raise ValueError("--patch-radius-multiplier must be > 0")
    if args.opposite_distance_multiplier <= 0:
        raise ValueError("--opposite-distance-multiplier must be > 0")
    if args.minimum_opposite_band_multiplier <= 0:
        raise ValueError("--minimum-opposite-band-multiplier must be > 0")
    if args.patch_min_gaussians < 1:
        raise ValueError("--patch-min-gaussians must be >= 1")
    if args.patch_max_gaussians < args.patch_min_gaussians:
        raise ValueError("--patch-max-gaussians must be >= --patch-min-gaussians")
    if args.spacing_sample_limit < 2:
        raise ValueError("--spacing-sample-limit must be >= 2")
    if args.bins_u < 1 or args.bins_v < 1:
        raise ValueError("--bins-u and --bins-v must be >= 1")
    if args.visualization_max_context_points < 0:
        raise ValueError("--visualization-max-context-points must be >= 0")
    if args.visualization_dpi < 1:
        raise ValueError("--visualization-dpi must be >= 1")


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_numpy_indices(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    return np.unique(array)


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


def parse_pair_subset(text: str | None) -> set[tuple[int, int]] | None:
    if text is None:
        return None

    result: set[tuple[int, int]] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.replace(":", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair token: {token}")
        a, b = int(parts[0]), int(parts[1])
        if a == b:
            raise ValueError(f"Self-pair is invalid: {token}")
        result.add((min(a, b), max(a, b)))

    if not result:
        raise ValueError("--pairs did not contain a valid pair.")
    return result


def median_nn_spacing(points: np.ndarray, sample_limit: int, seed: int) -> float:
    if len(points) < 2:
        raise ValueError("At least two points are required for spacing estimation.")

    rng = np.random.default_rng(seed)
    if len(points) > sample_limit:
        chosen = rng.choice(len(points), size=sample_limit, replace=False)
        query = points[chosen]
    else:
        query = points

    tree = cKDTree(points)
    distances, _ = tree.query(query, k=2)
    values = np.asarray(distances[:, 1], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]

    if values.size == 0:
        raise RuntimeError("Could not estimate positive nearest-neighbour spacing.")
    return float(np.median(values))


def directed_nn(
    source: np.ndarray,
    destination: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(destination)
    distances, indices = tree.query(source, k=1)
    return (
        np.asarray(distances, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def distance_to_seed_set(
    cluster_points: np.ndarray,
    seed_points: np.ndarray,
) -> np.ndarray:
    distances, _ = directed_nn(cluster_points, seed_points)
    return distances


def deterministic_farthest_point_subset(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    mandatory_indices: np.ndarray,
    target_count: int,
) -> np.ndarray:
    """
    Deterministic Euclidean FPS over candidate_indices.

    mandatory_indices are always kept first. If they already exceed the target,
    they are reduced by FPS as well.
    """
    candidates = np.unique(candidate_indices.astype(np.int64))
    mandatory = np.intersect1d(
        np.unique(mandatory_indices.astype(np.int64)),
        candidates,
        assume_unique=False,
    )

    if len(candidates) <= target_count:
        return np.sort(candidates)

    # If seeds alone exceed the budget, spread the retained seeds spatially.
    if len(mandatory) > target_count:
        pool = mandatory
        chosen = [int(pool[np.lexsort(points[pool].T[::-1])[0]])]
        min_dist2 = np.sum((points[pool] - points[chosen[0]]) ** 2, axis=1)
        while len(chosen) < target_count:
            next_local = int(np.argmax(min_dist2))
            next_idx = int(pool[next_local])
            chosen.append(next_idx)
            dist2 = np.sum((points[pool] - points[next_idx]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
        return np.sort(np.asarray(chosen, dtype=np.int64))

    selected = mandatory.tolist()

    remaining_mask = ~np.isin(candidates, mandatory)
    remaining = candidates[remaining_mask]

    if selected:
        selected_points = points[np.asarray(selected, dtype=np.int64)]
        diff = points[remaining, None, :] - selected_points[None, :, :]
        min_dist2 = np.min(np.sum(diff * diff, axis=-1), axis=1)
    else:
        first = int(candidates[np.lexsort(points[candidates].T[::-1])[0]])
        selected = [first]
        remaining = candidates[candidates != first]
        min_dist2 = np.sum((points[remaining] - points[first]) ** 2, axis=1)

    while len(selected) < target_count and len(remaining) > 0:
        next_local = int(np.argmax(min_dist2))
        next_idx = int(remaining[next_local])
        selected.append(next_idx)

        keep = np.ones(len(remaining), dtype=bool)
        keep[next_local] = False
        remaining = remaining[keep]
        min_dist2 = min_dist2[keep]

        if len(remaining) > 0:
            dist2 = np.sum((points[remaining] - points[next_idx]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)

    return np.sort(np.asarray(selected, dtype=np.int64))


def fill_to_minimum_by_score(
    selected: np.ndarray,
    score: np.ndarray,
    minimum: int,
    cluster_size: int,
) -> np.ndarray:
    selected = np.unique(selected.astype(np.int64))
    target = min(minimum, cluster_size)
    if len(selected) >= target:
        return selected

    order = np.argsort(score, kind="stable")
    return np.sort(np.unique(np.concatenate([selected, order[:target]])))


def select_one_side_patch(
    cluster_points: np.ndarray,
    opposite_points: np.ndarray,
    seed_local_indices: np.ndarray,
    patch_radius: float,
    opposite_band: float,
    minimum: int,
    maximum: int,
) -> tuple[np.ndarray, dict[str, float]]:
    seed_local_indices = np.unique(seed_local_indices.astype(np.int64))
    if seed_local_indices.size == 0:
        raise ValueError("A contact patch cannot be expanded from an empty seed set.")

    seed_points = cluster_points[seed_local_indices]
    distance_seed = distance_to_seed_set(cluster_points, seed_points)
    distance_other, _ = directed_nn(cluster_points, opposite_points)

    candidate_mask = (
        (distance_seed <= patch_radius)
        & (distance_other <= opposite_band)
    )
    candidate_mask[seed_local_indices] = True
    selected = np.flatnonzero(candidate_mask).astype(np.int64)

    # Low score means both close to the original junction and close to the
    # opposite cluster. This only acts as a fallback when the strict patch is
    # smaller than the requested minimum.
    score = (
        distance_seed / max(patch_radius, EPS)
        + distance_other / max(opposite_band, EPS)
    )
    selected = fill_to_minimum_by_score(
        selected=selected,
        score=score,
        minimum=minimum,
        cluster_size=len(cluster_points),
    )

    if len(selected) > maximum:
        selected = deterministic_farthest_point_subset(
            points=cluster_points,
            candidate_indices=selected,
            mandatory_indices=seed_local_indices,
            target_count=maximum,
        )

    metrics = {
        "seed_count": float(len(seed_local_indices)),
        "selected_count": float(len(selected)),
        "selected_median_distance_to_seed": float(
            np.median(distance_seed[selected])
        ),
        "selected_p90_distance_to_seed": float(
            np.quantile(distance_seed[selected], 0.9)
        ),
        "selected_median_distance_to_other": float(
            np.median(distance_other[selected])
        ),
        "selected_p90_distance_to_other": float(
            np.quantile(distance_other[selected], 0.9)
        ),
        "selected_fraction_inside_strict_rule": float(
            np.mean(candidate_mask[selected])
        ),
    }
    return selected, metrics


def orient_contact_frame(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return origin [3] and right-handed axes [3, 3].

    axes[0], axes[1] span the estimated contact plane.
    axes[2] points approximately from patch A toward patch B.
    """
    all_points = np.concatenate([points_a, points_b], axis=0)
    origin = np.mean(all_points, axis=0)

    centered = all_points - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    centroid_direction = np.mean(points_b, axis=0) - np.mean(points_a, axis=0)
    normal_norm = float(np.linalg.norm(centroid_direction))

    if normal_norm > EPS:
        normal = centroid_direction / normal_norm
    else:
        normal = vh[-1]
        normal /= max(float(np.linalg.norm(normal)), EPS)

    # Choose the strongest PCA direction that is not parallel to the normal.
    tangent_u = vh[0] - np.dot(vh[0], normal) * normal
    if np.linalg.norm(tangent_u) <= EPS:
        tangent_u = vh[1] - np.dot(vh[1], normal) * normal
    tangent_u /= max(float(np.linalg.norm(tangent_u)), EPS)

    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= max(float(np.linalg.norm(tangent_v)), EPS)

    # Re-orthogonalize to limit numerical drift.
    tangent_u = np.cross(tangent_v, normal)
    tangent_u /= max(float(np.linalg.norm(tangent_u)), EPS)

    axes = np.stack([tangent_u, tangent_v, normal], axis=0)
    return origin.astype(np.float32), axes.astype(np.float32)


def fixed_bin_assignment(
    points_a: np.ndarray,
    points_b: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    bins_u: int,
    bins_v: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_points = np.concatenate([points_a, points_b], axis=0)
    uv = (all_points - origin[None, :]) @ axes[:2].T

    uv_min = np.min(uv, axis=0)
    uv_max = np.max(uv, axis=0)
    span = np.maximum(uv_max - uv_min, 1e-8)

    normalized = (uv - uv_min[None, :]) / span[None, :]
    bin_u = np.minimum(
        (normalized[:, 0] * bins_u).astype(np.int64),
        bins_u - 1,
    )
    bin_v = np.minimum(
        (normalized[:, 1] * bins_v).astype(np.int64),
        bins_v - 1,
    )
    bin_ids = bin_v * bins_u + bin_u

    count_a = len(points_a)
    bin_a = bin_ids[:count_a]
    bin_b = bin_ids[count_a:]

    num_bins = bins_u * bins_v
    counts_a = np.bincount(bin_a, minlength=num_bins)
    counts_b = np.bincount(bin_b, minlength=num_bins)

    metadata = {
        "bins_u": int(bins_u),
        "bins_v": int(bins_v),
        "uv_min": torch.from_numpy(uv_min.astype(np.float32)),
        "uv_max": torch.from_numpy(uv_max.astype(np.float32)),
        "bin_counts_a": torch.from_numpy(counts_a.astype(np.int64)),
        "bin_counts_b": torch.from_numpy(counts_b.astype(np.int64)),
        "joint_valid_bin_mask": torch.from_numpy(
            ((counts_a > 0) & (counts_b > 0))
        ),
    }
    return bin_a, bin_b, metadata


def map_global_to_local(
    cluster_global_indices: np.ndarray,
    requested_global_indices: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(global_index): local_index
        for local_index, global_index in enumerate(cluster_global_indices.tolist())
    }
    missing = [
        int(index)
        for index in requested_global_indices.tolist()
        if int(index) not in lookup
    ]
    if missing:
        preview = missing[:10]
        raise RuntimeError(
            "Saved boundary indices do not belong to the expected cluster. "
            f"Missing examples: {preview}"
        )
    return np.asarray(
        [lookup[int(index)] for index in requested_global_indices.tolist()],
        dtype=np.int64,
    )



def sample_context_indices(
    cluster_size: int,
    excluded_local_indices: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Deterministically sample non-patch Gaussian indices for visualization."""
    if maximum <= 0 or cluster_size <= 0:
        return np.empty(0, dtype=np.int64)

    all_indices = np.arange(cluster_size, dtype=np.int64)
    available = all_indices[
        ~np.isin(all_indices, np.unique(excluded_local_indices.astype(np.int64)))
    ]
    if len(available) <= maximum:
        return available

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(available, size=maximum, replace=False))


def set_axes_equal_3d(ax: Any, arrays: list[np.ndarray]) -> None:
    valid = [array for array in arrays if array.size > 0]
    if not valid:
        return

    points = np.concatenate(valid, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * float(np.max(maximum - minimum))
    radius = max(radius, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_contact_patch_visualizations(
    output_dir: Path,
    cluster_a: int,
    cluster_b: int,
    data_a: ClusterData,
    data_b: ClusterData,
    seed_local_a: np.ndarray,
    seed_local_b: np.ndarray,
    selected_local_a: np.ndarray,
    selected_local_b: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    bin_a: np.ndarray,
    bin_b: np.ndarray,
    bins_u: int,
    bins_v: int,
    max_context_points: int,
    dpi: int,
    seed: int,
) -> tuple[Path, Path]:
    """Save a canonical 3-D overview and a contact-plane 2-D view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_name = f"{cluster_a}_{cluster_b}"

    patch_points_a = data_a.points[selected_local_a]
    patch_points_b = data_b.points[selected_local_b]
    seed_points_a = data_a.points[seed_local_a]
    seed_points_b = data_b.points[seed_local_b]

    context_local_a = sample_context_indices(
        cluster_size=data_a.size,
        excluded_local_indices=selected_local_a,
        maximum=max_context_points,
        seed=seed + cluster_a * 1009 + cluster_b,
    )
    context_local_b = sample_context_indices(
        cluster_size=data_b.size,
        excluded_local_indices=selected_local_b,
        maximum=max_context_points,
        seed=seed + cluster_b * 1009 + cluster_a,
    )
    context_points_a = data_a.points[context_local_a]
    context_points_b = data_b.points[context_local_b]

    path_3d = output_dir / f"pair_{pair_name}_3d.png"
    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection="3d")

    if len(context_points_a):
        ax.scatter(
            context_points_a[:, 0],
            context_points_a[:, 1],
            context_points_a[:, 2],
            s=2,
            alpha=0.10,
            color="tab:blue",
            label=f"cluster {cluster_a} context",
        )
    if len(context_points_b):
        ax.scatter(
            context_points_b[:, 0],
            context_points_b[:, 1],
            context_points_b[:, 2],
            s=2,
            alpha=0.10,
            color="tab:orange",
            label=f"cluster {cluster_b} context",
        )

    ax.scatter(
        patch_points_a[:, 0],
        patch_points_a[:, 1],
        patch_points_a[:, 2],
        s=14,
        alpha=0.85,
        color="blue",
        label=f"patch A ({len(patch_points_a)})",
    )
    ax.scatter(
        patch_points_b[:, 0],
        patch_points_b[:, 1],
        patch_points_b[:, 2],
        s=14,
        alpha=0.85,
        color="red",
        label=f"patch B ({len(patch_points_b)})",
    )
    ax.scatter(
        seed_points_a[:, 0],
        seed_points_a[:, 1],
        seed_points_a[:, 2],
        s=45,
        marker="x",
        linewidths=1.5,
        color="cyan",
        label=f"boundary seed A ({len(seed_points_a)})",
    )
    ax.scatter(
        seed_points_b[:, 0],
        seed_points_b[:, 1],
        seed_points_b[:, 2],
        s=45,
        marker="x",
        linewidths=1.5,
        color="magenta",
        label=f"boundary seed B ({len(seed_points_b)})",
    )

    axis_scale = max(
        float(np.linalg.norm(np.ptp(np.concatenate([patch_points_a, patch_points_b]), axis=0))),
        1e-6,
    ) * 0.20
    axis_colors = ("green", "purple", "black")
    axis_names = ("u", "v", "normal A→B")
    for axis_index in range(3):
        direction = axes[axis_index] * axis_scale
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=axis_colors[axis_index],
            linewidth=2,
        )
        endpoint = origin + direction
        ax.text(endpoint[0], endpoint[1], endpoint[2], axis_names[axis_index])

    set_axes_equal_3d(
        ax,
        [context_points_a, context_points_b, patch_points_a, patch_points_b],
    )
    ax.set_xlabel("canonical x")
    ax.set_ylabel("canonical y")
    ax.set_zlabel("canonical z")
    ax.set_title(f"Canonical contact patch: cluster {cluster_a}-{cluster_b}")
    ax.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path_3d, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    path_plane = output_dir / f"pair_{pair_name}_contact_plane.png"
    uv_a = (patch_points_a - origin[None, :]) @ axes[:2].T
    uv_b = (patch_points_b - origin[None, :]) @ axes[:2].T
    seed_uv_a = (seed_points_a - origin[None, :]) @ axes[:2].T
    seed_uv_b = (seed_points_b - origin[None, :]) @ axes[:2].T

    all_uv = np.concatenate([uv_a, uv_b], axis=0)
    uv_min = all_uv.min(axis=0)
    uv_max = all_uv.max(axis=0)

    figure, ax = plt.subplots(figsize=(9, 8))
    scatter_a = ax.scatter(
        uv_a[:, 0],
        uv_a[:, 1],
        c=bin_a,
        cmap="Blues",
        s=24,
        alpha=0.85,
        marker="o",
        label=f"patch A: cluster {cluster_a}",
    )
    scatter_b = ax.scatter(
        uv_b[:, 0],
        uv_b[:, 1],
        c=bin_b,
        cmap="Reds",
        s=24,
        alpha=0.85,
        marker="^",
        label=f"patch B: cluster {cluster_b}",
    )
    ax.scatter(
        seed_uv_a[:, 0],
        seed_uv_a[:, 1],
        s=70,
        marker="x",
        linewidths=1.8,
        color="cyan",
        label="boundary seeds A",
    )
    ax.scatter(
        seed_uv_b[:, 0],
        seed_uv_b[:, 1],
        s=70,
        marker="x",
        linewidths=1.8,
        color="magenta",
        label="boundary seeds B",
    )

    for index in range(1, bins_u):
        value = uv_min[0] + (uv_max[0] - uv_min[0]) * index / bins_u
        ax.axvline(value, color="gray", linewidth=0.8, alpha=0.55)
    for index in range(1, bins_v):
        value = uv_min[1] + (uv_max[1] - uv_min[1]) * index / bins_v
        ax.axhline(value, color="gray", linewidth=0.8, alpha=0.55)

    ax.set_xlabel("contact-plane u")
    ax.set_ylabel("contact-plane v")
    ax.set_title(
        f"Contact-plane projection and fixed bins: cluster {cluster_a}-{cluster_b}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    figure.colorbar(scatter_a, ax=ax, fraction=0.045, pad=0.02, label="A bin id")
    figure.colorbar(scatter_b, ax=ax, fraction=0.045, pad=0.08, label="B bin id")
    figure.tight_layout()
    figure.savefig(path_plane, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return path_3d, path_plane

def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    work_dir = args.work_dir.expanduser().resolve()
    checkpoint = (
        args.ckpt.expanduser().resolve()
        if args.ckpt is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )
    pair_file = (
        args.pair_file.expanduser().resolve()
        if args.pair_file is not None
        else (
            work_dir
            / "analysis"
            / "all_cluster_boundary_relations"
            / "fixed_boundary_indices.pt"
        )
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "analysis" / "contact_patches"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    if not args.no_visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    output_pt = output_dir / "contact_patches.pt"
    if output_pt.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_pt} already exists. Pass --overwrite only when replacement "
            "is intended."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not pair_file.is_file():
        raise FileNotFoundError(f"Pair file not found: {pair_file}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    saved_pairs = torch_load_cpu(pair_file)
    if not isinstance(saved_pairs, dict):
        raise TypeError("The pair file must contain a dictionary keyed by pair.")

    requested_pairs = parse_pair_subset(args.pairs)

    with torch.inference_mode():
        renderer = Renderer.init_from_checkpoint(
            str(checkpoint),
            device,
            work_dir=str(work_dir),
            port=None,
        )
        model = renderer.model
        model.eval()

        canonical_means_t = model.fg.params["means"].detach()
        cluster_ids_t = (
            model.fg.get_cluster_ids()
            .reshape(-1)
            .long()
            .to(canonical_means_t.device)
        )

        if canonical_means_t.shape[0] != cluster_ids_t.numel():
            raise RuntimeError(
                "Foreground Gaussian count differs from cluster-ID count."
            )

        canonical_means = (
            canonical_means_t.float().cpu().numpy().astype(np.float64)
        )
        cluster_ids = cluster_ids_t.cpu().numpy().astype(np.int64)

    needed_cluster_ids: set[int] = set()
    normalized_entries: list[tuple[str, dict[str, Any], int, int]] = []

    for original_key, entry in saved_pairs.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Pair entry {original_key!r} is not a dictionary.")

        cluster_a = int(entry["cluster_a"])
        cluster_b = int(entry["cluster_b"])
        canonical_pair = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))

        if requested_pairs is not None and canonical_pair not in requested_pairs:
            continue

        normalized_entries.append(
            (str(original_key), entry, cluster_a, cluster_b)
        )
        needed_cluster_ids.update([cluster_a, cluster_b])

    if not normalized_entries:
        raise RuntimeError("No saved pair matched the requested subset.")

    clusters: dict[int, ClusterData] = {}
    for cluster_id in sorted(needed_cluster_ids):
        global_indices = np.flatnonzero(cluster_ids == cluster_id).astype(np.int64)
        if global_indices.size == 0:
            raise RuntimeError(
                f"Saved pair references absent cluster ID {cluster_id}."
            )
        clusters[cluster_id] = ClusterData(
            cluster_id=cluster_id,
            global_indices=global_indices,
            points=canonical_means[global_indices],
        )

    spacing_by_cluster: dict[int, float] = {}
    for cluster_id, cluster in clusters.items():
        spacing_by_cluster[cluster_id] = median_nn_spacing(
            cluster.points,
            sample_limit=args.spacing_sample_limit,
            seed=args.seed + cluster_id,
        )

    output_pairs: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for pair_index, (original_key, entry, cluster_a, cluster_b) in enumerate(
        normalized_entries,
        start=1,
    ):
        data_a = clusters[cluster_a]
        data_b = clusters[cluster_b]

        seed_global_a = as_numpy_indices(entry["boundary_global_indices_a"])
        seed_global_b = as_numpy_indices(entry["boundary_global_indices_b"])
        seed_local_a = map_global_to_local(data_a.global_indices, seed_global_a)
        seed_local_b = map_global_to_local(data_b.global_indices, seed_global_b)

        spacing_a = spacing_by_cluster[cluster_a]
        spacing_b = spacing_by_cluster[cluster_b]
        balanced_spacing = math.sqrt(max(spacing_a * spacing_b, EPS))

        # Estimate how wide the existing canonical junction already is.
        seed_distance_ab, _ = directed_nn(
            data_a.points[seed_local_a],
            data_b.points,
        )
        seed_distance_ba, _ = directed_nn(
            data_b.points[seed_local_b],
            data_a.points,
        )
        balanced_seed_distance = float(
            np.median(np.concatenate([seed_distance_ab, seed_distance_ba]))
        )

        patch_radius_a = spacing_a * args.patch_radius_multiplier
        patch_radius_b = spacing_b * args.patch_radius_multiplier
        opposite_band = max(
            balanced_seed_distance * args.opposite_distance_multiplier,
            balanced_spacing * args.minimum_opposite_band_multiplier,
        )

        selected_local_a, metrics_a = select_one_side_patch(
            cluster_points=data_a.points,
            opposite_points=data_b.points,
            seed_local_indices=seed_local_a,
            patch_radius=patch_radius_a,
            opposite_band=opposite_band,
            minimum=args.patch_min_gaussians,
            maximum=args.patch_max_gaussians,
        )
        selected_local_b, metrics_b = select_one_side_patch(
            cluster_points=data_b.points,
            opposite_points=data_a.points,
            seed_local_indices=seed_local_b,
            patch_radius=patch_radius_b,
            opposite_band=opposite_band,
            minimum=args.patch_min_gaussians,
            maximum=args.patch_max_gaussians,
        )

        patch_global_a = data_a.global_indices[selected_local_a]
        patch_global_b = data_b.global_indices[selected_local_b]
        patch_points_a = data_a.points[selected_local_a]
        patch_points_b = data_b.points[selected_local_b]

        origin, axes = orient_contact_frame(
            patch_points_a,
            patch_points_b,
        )
        bin_a, bin_b, bin_metadata = fixed_bin_assignment(
            points_a=patch_points_a,
            points_b=patch_points_b,
            origin=origin,
            axes=axes,
            bins_u=args.bins_u,
            bins_v=args.bins_v,
        )

        patch_distance_ab, _ = directed_nn(patch_points_a, patch_points_b)
        patch_distance_ba, _ = directed_nn(patch_points_b, patch_points_a)
        balanced_patch_distance = np.concatenate(
            [patch_distance_ab, patch_distance_ba]
        )

        visualization_3d = ""
        visualization_contact_plane = ""
        if not args.no_visualizations:
            path_3d, path_plane = save_contact_patch_visualizations(
                output_dir=visualization_dir,
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                data_a=data_a,
                data_b=data_b,
                seed_local_a=seed_local_a,
                seed_local_b=seed_local_b,
                selected_local_a=selected_local_a,
                selected_local_b=selected_local_b,
                origin=origin,
                axes=axes,
                bin_a=bin_a,
                bin_b=bin_b,
                bins_u=args.bins_u,
                bins_v=args.bins_v,
                max_context_points=args.visualization_max_context_points,
                dpi=args.visualization_dpi,
                seed=args.seed,
            )
            visualization_3d = str(path_3d)
            visualization_contact_plane = str(path_plane)

        pair_key = f"{cluster_a}_{cluster_b}"
        output_pairs[pair_key] = {
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "source_pair_key": original_key,
            "seed_boundary_global_indices_a": torch.from_numpy(
                seed_global_a
            ).long(),
            "seed_boundary_global_indices_b": torch.from_numpy(
                seed_global_b
            ).long(),
            "contact_patch_global_indices_a": torch.from_numpy(
                patch_global_a
            ).long(),
            "contact_patch_global_indices_b": torch.from_numpy(
                patch_global_b
            ).long(),
            "contact_patch_bin_ids_a": torch.from_numpy(bin_a).long(),
            "contact_patch_bin_ids_b": torch.from_numpy(bin_b).long(),
            "canonical_contact_origin": torch.from_numpy(origin),
            "canonical_contact_axes": torch.from_numpy(axes),
            "cluster_spacing_a": float(spacing_a),
            "cluster_spacing_b": float(spacing_b),
            "patch_radius_a": float(patch_radius_a),
            "patch_radius_b": float(patch_radius_b),
            "opposite_distance_band": float(opposite_band),
            "canonical_patch_median_nn": float(
                np.median(balanced_patch_distance)
            ),
            "canonical_patch_p90_nn": float(
                np.quantile(balanced_patch_distance, 0.9)
            ),
            "binning": bin_metadata,
            "visualization_3d": visualization_3d,
            "visualization_contact_plane": visualization_contact_plane,
        }

        row = {
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "cluster_size_a": data_a.size,
            "cluster_size_b": data_b.size,
            "seed_count_a": len(seed_global_a),
            "seed_count_b": len(seed_global_b),
            "patch_count_a": len(patch_global_a),
            "patch_count_b": len(patch_global_b),
            "cluster_spacing_a": spacing_a,
            "cluster_spacing_b": spacing_b,
            "balanced_seed_median_nn": balanced_seed_distance,
            "patch_radius_a": patch_radius_a,
            "patch_radius_b": patch_radius_b,
            "opposite_distance_band": opposite_band,
            "canonical_patch_median_nn": float(
                np.median(balanced_patch_distance)
            ),
            "canonical_patch_p90_nn": float(
                np.quantile(balanced_patch_distance, 0.9)
            ),
            "strict_fraction_a": metrics_a[
                "selected_fraction_inside_strict_rule"
            ],
            "strict_fraction_b": metrics_b[
                "selected_fraction_inside_strict_rule"
            ],
            "joint_valid_bins": int(
                bin_metadata["joint_valid_bin_mask"].sum().item()
            ),
            "total_bins": int(args.bins_u * args.bins_v),
            "visualization_3d": visualization_3d,
            "visualization_contact_plane": visualization_contact_plane,
        }
        summary_rows.append(row)

        print(
            f"[{pair_index:03d}/{len(normalized_entries):03d}] "
            f"{cluster_a}-{cluster_b}: "
            f"seed=({len(seed_global_a)},{len(seed_global_b)}) -> "
            f"patch=({len(patch_global_a)},{len(patch_global_b)}), "
            f"median_nn={row['canonical_patch_median_nn']:.6f}, "
            f"valid_bins={row['joint_valid_bins']}/{row['total_bins']}"
        )

    payload = {
        "format_version": 1,
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint),
        "source_pair_file": str(pair_file),
        "selection_space": "canonical_foreground_gaussian_means",
        "pair_correspondence_policy": (
            "No fixed A-B correspondence. Recompute patch-to-patch nearest "
            "neighbours at each frame."
        ),
        "config": {
            "patch_radius_multiplier": args.patch_radius_multiplier,
            "opposite_distance_multiplier": args.opposite_distance_multiplier,
            "minimum_opposite_band_multiplier": (
                args.minimum_opposite_band_multiplier
            ),
            "patch_min_gaussians": args.patch_min_gaussians,
            "patch_max_gaussians": args.patch_max_gaussians,
            "spacing_sample_limit": args.spacing_sample_limit,
            "bins_u": args.bins_u,
            "bins_v": args.bins_v,
            "seed": args.seed,
            "visualizations_enabled": not args.no_visualizations,
            "visualization_max_context_points": (
                args.visualization_max_context_points
            ),
            "visualization_dpi": args.visualization_dpi,
        },
        "pairs": output_pairs,
    }
    torch.save(payload, output_pt)
    write_csv(output_dir / "contact_patch_summary.csv", summary_rows)

    report = {
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint),
        "source_pair_file": str(pair_file),
        "output_file": str(output_pt),
        "pair_count": len(output_pairs),
        "selection": {
            "seed_source": (
                "boundary_global_indices_a/b from fixed_boundary_indices.pt"
            ),
            "patch_definition": (
                "Same-cluster expansion around saved boundary seeds, filtered "
                "by distance to the opposite cluster."
            ),
            "fixed_pair_correspondences": False,
            "canonical_patch_indices_fixed": True,
            "dynamic_training_policy": (
                "At frame t, transform the saved patch Gaussians with the "
                "current MotionScale motion and recompute set-to-set nearest "
                "neighbours."
            ),
        },
        "config": payload["config"],
        "summary": summary_rows,
    }
    with (output_dir / "analysis_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 76)
    print("Contact-patch selection complete")
    print(f"Pairs       : {len(output_pairs)}")
    print(f"Output PT   : {output_pt}")
    print(f"Summary CSV : {output_dir / 'contact_patch_summary.csv'}")
    if not args.no_visualizations:
        print(f"Images      : {visualization_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
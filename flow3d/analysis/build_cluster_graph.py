#!/usr/bin/env python3
"""
Build the offline cluster-connectivity graph for one trained checkpoint.

This is the I/O layer around flow3d/analysis/cluster_graph.py's pure graph
builder: it loads the checkpoint, gathers per-frame Gaussian positions and
cluster assignments, estimates the contact-distance threshold (reusing
cluster_pairs.py's within-cluster spacing estimate), runs
cluster_graph.build_cluster_graph, logs validation info, and writes:

    <work-dir>/analysis/cluster_graph/edges.pt
    <work-dir>/analysis/cluster_graph/edges_kept.csv
    <work-dir>/analysis/cluster_graph/edges_cut.csv
    <work-dir>/analysis/cluster_graph/report.json
    <work-dir>/analysis/cluster_graph/graph_edges.png       (unless --no-visualization)
    <work-dir>/analysis/cluster_graph/graph_edges_2d.png    (unless --no-visualization)

edges.pt is the format consumed by trainer.py's update_rigidity_weights and
by the graph GNN (edge_index + per-edge stats). The PNG visualizations
(render_3d_multiview / render_2d_overlay, below) live in this file rather
than a separate module -- they're only ever called from here, right after
the graph is built, using the already-loaded model/clusters.

Example
-------
    python flow3d/analysis/build_cluster_graph.py \\
        --work-dir outputs/davis/spin/2026_08_03_08_41_11__spin_run1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from flow3d.renderer import Renderer
from flow3d.analysis.cluster_pairs import (
    ClusterInfo,
    estimate_within_cluster_spacing,
    write_csv,
    _get_camera_w2cs,
    _get_dynamic_fg_means,
    _load_overlay_font,
    _prepare_cluster_visual_data,
    _project_world_points,
    _set_axes_equal_3d,
)
from flow3d.analysis.cluster_graph import (
    ClusterGraphConfig,
    ClusterGraphResult,
    ClusterPairEdge,
    build_cluster_graph,
)

KEPT_COLOR = (0.10, 0.55, 0.15)
CUT_COLOR = (0.80, 0.10, 0.10)
KEPT_COLOR_2D = (25, 150, 45)
CUT_COLOR_2D = (210, 40, 40)

VIEWS = [
    (20, 35, "Perspective"),
    (20, 125, "Left / Back"),
    (20, 215, "Back"),
    (20, 305, "Right / Front"),
    (65, 35, "Top-front"),
    (65, 215, "Top-back"),
]


def _edge_linewidth(gap_summary: float, contact_distance: float, keep_multiplier: float) -> float:
    """Thicker/darker for smaller gap_summary (tighter contact)."""
    if contact_distance <= 0:
        return 2.0
    ratio = float(np.clip(gap_summary / (contact_distance * max(keep_multiplier, 1e-6)), 0.0, 1.0))
    return float(np.interp(ratio, [0.0, 1.0], [3.2, 1.0]))


def _draw_edges_3d(
    ax: Any,
    center_by_id: dict[int, np.ndarray],
    kept_edges: list[dict],
    cut_edges: list[dict],
    contact_distance: float,
    keep_multiplier: float,
) -> None:
    for e in cut_edges:
        ca = center_by_id.get(e["cluster_a"])
        cb = center_by_id.get(e["cluster_b"])
        if ca is None or cb is None:
            continue
        ax.plot(
            [ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]],
            linestyle=(0, (4, 3)), color=CUT_COLOR, alpha=0.35, linewidth=1.2, zorder=1,
        )
    for e in kept_edges:
        ca = center_by_id.get(e["cluster_a"])
        cb = center_by_id.get(e["cluster_b"])
        if ca is None or cb is None:
            continue
        lw = _edge_linewidth(e["gap_summary"], contact_distance, keep_multiplier)
        alpha = float(np.clip(np.interp(lw, [1.0, 3.2], [0.55, 0.95]), 0.5, 1.0))
        ax.plot(
            [ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]],
            linestyle="-", color=KEPT_COLOR, alpha=alpha, linewidth=lw, zorder=2,
        )


def _draw_graph_scene(
    ax: Any,
    clusters: list[ClusterInfo],
    sampled_by_id: dict[int, np.ndarray],
    center_by_id: dict[int, np.ndarray],
    color_by_id: dict[int, Any],
    all_points: np.ndarray,
    kept_edges: list[dict],
    cut_edges: list[dict],
    contact_distance: float,
    keep_multiplier: float,
) -> None:
    for cluster in clusters:
        cid = cluster.cluster_id
        points = sampled_by_id[cid]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=1.5, alpha=0.12, color=color_by_id[cid], rasterized=True, zorder=0,
        )

    _draw_edges_3d(ax, center_by_id, kept_edges, cut_edges, contact_distance, keep_multiplier)

    for cluster in clusters:
        cid = cluster.cluster_id
        center = center_by_id[cid]
        ax.scatter([center[0]], [center[1]], [center[2]], s=18, color="black", zorder=3)
        ax.text(
            center[0], center[1], center[2], str(cid),
            fontsize=8, fontweight="bold", color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="round,pad=0.15"),
            zorder=4,
        )

    _set_axes_equal_3d(ax, all_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def render_3d_multiview(
    clusters: list[ClusterInfo],
    kept_edges: list[dict],
    cut_edges: list[dict],
    contact_distance: float,
    keep_multiplier: float,
    output_path: Path,
    max_points_per_cluster: int = 2000,
) -> Path:
    sampled_by_id, center_by_id, color_by_id, all_points = _prepare_cluster_visual_data(
        clusters=clusters, max_points_per_cluster=max_points_per_cluster,
    )

    fig = plt.figure(figsize=(18, 11))
    for subplot_index, (elev, azim, title) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(2, 3, subplot_index, projection="3d")
        _draw_graph_scene(
            ax=ax,
            clusters=clusters,
            sampled_by_id=sampled_by_id,
            center_by_id=center_by_id,
            color_by_id=color_by_id,
            all_points=all_points,
            kept_edges=kept_edges,
            cut_edges=cut_edges,
            contact_distance=contact_distance,
            keep_multiplier=keep_multiplier,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)

    fig.suptitle(
        f"Cluster connectivity graph -- kept edges: {len(kept_edges)}, "
        f"cut candidates: {len(cut_edges)}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    xy0: tuple[float, float],
    xy1: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 2,
    dash_len: float = 6.0,
    gap_len: float = 5.0,
) -> None:
    x0, y0 = xy0
    x1, y1 = xy1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-6:
        return
    direction = ((x1 - x0) / length, (y1 - y0) / length)
    pos = 0.0
    draw_on = True
    while pos < length:
        seg_len = dash_len if draw_on else gap_len
        end = min(pos + seg_len, length)
        if draw_on:
            sx, sy = x0 + direction[0] * pos, y0 + direction[1] * pos
            ex, ey = x0 + direction[0] * end, y0 + direction[1] * end
            draw.line((sx, sy, ex, ey), fill=fill, width=width)
        pos = end
        draw_on = not draw_on


def _render_graph_overlay_frame(
    model: Any,
    clusters: list[ClusterInfo],
    color_by_id: dict[int, Any],
    frame_index: int,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    kept_edges: list[dict],
    cut_edges: list[dict],
    contact_distance: float,
    keep_multiplier: float,
) -> np.ndarray:
    width, height = image_size
    render_output = model.render(
        frame_index, w2c[None], intrinsic[None], image_size,
        return_depth=True, use_learned_poses=False,
    )
    rgb = render_output["img"][0].detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(image)
    font = _load_overlay_font(16)
    dynamic_means = _get_dynamic_fg_means(model, frame_index)

    center_pixel_by_id: dict[int, tuple[float, float]] = {}
    for cluster in clusters:
        cluster_points = dynamic_means[cluster.global_indices]
        center = cluster_points.mean(dim=0, keepdim=True)
        center_pixel, center_valid = _project_world_points(center, w2c, intrinsic, width, height)
        if center_valid[0]:
            center_pixel_by_id[cluster.cluster_id] = (
                float(center_pixel[0, 0]), float(center_pixel[0, 1])
            )

    for e in cut_edges:
        pa = center_pixel_by_id.get(e["cluster_a"])
        pb = center_pixel_by_id.get(e["cluster_b"])
        if pa is None or pb is None:
            continue
        _dashed_line(draw, pa, pb, fill=CUT_COLOR_2D, width=2)

    for e in kept_edges:
        pa = center_pixel_by_id.get(e["cluster_a"])
        pb = center_pixel_by_id.get(e["cluster_b"])
        if pa is None or pb is None:
            continue
        lw = _edge_linewidth(e["gap_summary"], contact_distance, keep_multiplier)
        width_px = max(2, int(round(lw * 1.6)))
        draw.line((pa[0], pa[1], pb[0], pb[1]), fill=KEPT_COLOR_2D, width=width_px)

    for cluster in clusters:
        cid = cluster.cluster_id
        pixel = center_pixel_by_id.get(cid)
        if pixel is None:
            continue
        color_float = color_by_id[cid][:3]
        color = tuple(int(round(v * 255)) for v in color_float)
        x, y = pixel
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0))

        text = str(cid)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_width, text_height = 12, 12
        left = x - text_width / 2 - 3
        bottom = y - radius - 4
        top = bottom - text_height - 4
        right = x + text_width / 2 + 3
        draw.rounded_rectangle((left, top, right, bottom), radius=3, fill=(255, 255, 255))
        draw.text((x - text_width / 2, top + 2), text, fill=(0, 0, 0), font=font)

    return np.asarray(image)


def render_2d_overlay(
    model: Any,
    clusters: list[ClusterInfo],
    kept_edges: list[dict],
    cut_edges: list[dict],
    contact_distance: float,
    keep_multiplier: float,
    output_path: Path,
    frame_index: int = 0,
) -> Path:
    _, _, color_by_id, _ = _prepare_cluster_visual_data(clusters=clusters, max_points_per_cluster=1)

    device = model.fg.params["means"].device
    w2cs = _get_camera_w2cs(model).to(device)
    intrinsics = model.Ks.to(device)
    frame_index = int(np.clip(frame_index, 0, w2cs.shape[0] - 1))

    principal_x = float(intrinsics[frame_index, 0, 2].item())
    principal_y = float(intrinsics[frame_index, 1, 2].item())
    width = max(int(round(principal_x * 2.0)), 2)
    height = max(int(round(principal_y * 2.0)), 2)

    with torch.no_grad():
        frame = _render_graph_overlay_frame(
            model=model,
            clusters=clusters,
            color_by_id=color_by_id,
            frame_index=frame_index,
            w2c=w2cs[frame_index],
            intrinsic=intrinsics[frame_index],
            image_size=(width, height),
            kept_edges=kept_edges,
            cut_edges=cut_edges,
            contact_distance=contact_distance,
            keep_multiplier=keep_multiplier,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--min-cluster-size", type=int, default=20)

    parser.add_argument("--contact-spacing-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--contact-distance",
        type=float,
        default=None,
        help="Explicit contact threshold. Default: median within-cluster NN "
        "spacing * --contact-spacing-multiplier.",
    )
    parser.add_argument("--spacing-sample-limit", type=int, default=20000)

    parser.add_argument("--candidate-radius-multiplier", type=float, default=30.0)
    parser.add_argument(
        "--center-gate-max-median-p10-ratio",
        type=float,
        default=ClusterGraphConfig().center_gate_max_median_to_p10_ratio,
        help="Sanity gate: a pair is cut outright if its cluster CENTERS' "
        "median distance across the whole sequence exceeds this many times "
        "their 10th-percentile distance, independent of the boundary-gap "
        "check. Relative rather than an absolute distance, so it doesn't "
        "need recalibrating per checkpoint/scene scale. Compares against "
        "the p10 distance rather than the single closest-approach frame -- "
        "a one-frame min lets an articulated real neighbor's centroids swing "
        "close together at one arbitrary pose purely by chance and wrongly "
        "inflate the ratio, even while its boundary stays touching every "
        "frame; p10 is the closest-approach *regime*, not one lucky frame. "
        "A pair that stays close throughout (small ratio) passes through to "
        "the gap check; a pair whose closest-approach regime is rare "
        "relative to its typical separation (large ratio) is cut here -- "
        "indistinguishable from the min-over-many-noisy-candidate-pairs "
        "artifact (a large candidate-point pool always contains SOME near "
        "pair by chance) this gate exists to catch.",
    )
    parser.add_argument("--keep-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--gap-smoothing-window",
        type=int,
        default=5,
        help="Frames per confidence-weighted smoothing window applied to the "
        "per-frame gap sequence before taking its max (the keep/cut summary "
        "statistic) -- every frame contributes, weighted by its own "
        "confidence, none are dropped. 1 disables windowed smoothing "
        "(falls back to the raw, unweighted max). Higher = more tolerant of "
        "transient noise/occlusion, at the cost of blurring genuinely short "
        "separations.",
    )
    parser.add_argument(
        "--gap-confidence-threshold",
        type=float,
        default=0.5,
        help="Diagnostic-only cutoff: frames with confidence_t below this "
        "are reported via the gap_low_confidence_frac column, but no longer "
        "dropped from the gap computation (that hard gate silently discarded "
        "exactly the frames that would show a real separation whenever those "
        "frames also happened to be low-confidence). Confidence now only "
        "downweights a frame within --gap-smoothing-window or feeds this "
        "diagnostic. No effect when confidences_all_frames is None (learned "
        "position source).",
    )
    parser.add_argument(
        "--gap-skip-first-n-frames",
        type=int,
        default=0,
        help="Drop the first N frames from candidate/gap computation entirely "
        "(cluster centers, boundary selection, and the per-frame gap "
        "sequence all skip them, for both position sources) -- for a clip "
        "whose earliest frames have unreliable raw-track/depth lifting (e.g. "
        "a tracking-window warm-up artifact) that reads as a false "
        "separation between parts that are actually touching there. "
        "Confirmed by visual inspection of the affected frames, not applied "
        "blindly.",
    )

    parser.add_argument("--boundary-fraction", type=float, default=0.10)
    parser.add_argument(
        "--boundary-min-gaussians",
        type=int,
        default=20,
        help="Size of the per-cluster candidate-point pool each frame's "
        "cKDTree nearest-neighbor query picks from -- not a fixed pairing. "
        "Kept generous so a joint rotation that changes which points are "
        "actually closest doesn't fall outside the pool. Raise (up to a "
        "cluster's full size) if edges still look biased by rotation.",
    )
    parser.add_argument("--boundary-max-gaussians", type=int, default=150)
    parser.add_argument(
        "--boundary-max-distance",
        type=float,
        default=None,
        help="Default: contact_distance (same convention as cluster_pairs.py).",
    )

    parser.add_argument(
        "--gate-cluster-id",
        type=int,
        default=33,
        help="Cluster id to run the self-contact validation gate on.",
    )
    parser.add_argument("--gate-expect-neighbor", type=int, default=28)
    parser.add_argument("--gate-expect-absent", type=str, default="7,38")

    parser.add_argument(
        "--position-source",
        type=str,
        choices=["learned", "raw_tracks"],
        default="learned",
        help="learned: model.compute_poses_fg (the fitted motion-basis "
        "trajectory). raw_tracks: re-associate each cluster Gaussian with a "
        "freshly-loaded raw 2D-lifted track (pre-motion-basis) via nearest "
        "canonical-position match, and use that track's own trajectory "
        "instead. Useful to tell a real self-contact apart from a "
        "learned-motion artifact (e.g. an undertrained basis letting a "
        "cluster's points drift/swap onto the wrong limb) corrupting the "
        "graph -- if a cluster's neighbors change a lot under raw_tracks, "
        "the 'learned' graph was likely reflecting a training artifact, not "
        "physical self-contact.",
    )
    parser.add_argument(
        "--raw-track-num-query-frames",
        type=int,
        default=20,
        help="Number of evenly-spaced query frames to pool raw tracks from "
        "for --position-source raw_tracks. A single query frame's track "
        "count can be well under the total cluster-Gaussian count, forcing "
        "unrelated clusters' boundary points to collide onto the same "
        "track and read as spurious always-touching contact -- pooling "
        "several query frames spreads that risk thin.",
    )
    parser.add_argument("--raw-track-samples-per-query", type=int, default=20000)
    parser.add_argument(
        "--raw-track-max-match-distance",
        type=float,
        default=None,
        help="Warn if a cluster Gaussian's nearest raw track is farther than "
        "this. Default: contact_distance * 5.",
    )

    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--max-points-per-cluster", type=int, default=2000)
    parser.add_argument("--frame-index-2d", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.min_cluster_size < 1:
        raise ValueError("--min-cluster-size must be >= 1")
    if args.contact_spacing_multiplier <= 0:
        raise ValueError("--contact-spacing-multiplier must be > 0")
    if args.spacing_sample_limit < 1:
        raise ValueError("--spacing-sample-limit must be >= 1")
    if args.candidate_radius_multiplier <= 0:
        raise ValueError("--candidate-radius-multiplier must be > 0")
    if args.center_gate_max_median_p10_ratio <= 0:
        raise ValueError("--center-gate-max-median-p10-ratio must be > 0")
    if args.keep_multiplier <= 0:
        raise ValueError("--keep-multiplier must be > 0")
    if args.gap_smoothing_window < 1:
        raise ValueError("--gap-smoothing-window must be >= 1")
    if not 0 <= args.gap_confidence_threshold <= 1:
        raise ValueError("--gap-confidence-threshold must be in [0, 1]")
    if args.gap_skip_first_n_frames < 0:
        raise ValueError("--gap-skip-first-n-frames must be >= 0")
    if not 0 < args.boundary_fraction <= 1:
        raise ValueError("--boundary-fraction must be in (0, 1]")
    if args.boundary_min_gaussians < 1:
        raise ValueError("--boundary-min-gaussians must be >= 1")
    if args.boundary_max_gaussians < args.boundary_min_gaussians:
        raise ValueError("--boundary-max-gaussians must be >= --boundary-min-gaussians")


def load_model_and_clusters(
    work_dir: Path,
    ckpt: Path | None,
    device_name: str,
    min_cluster_size: int,
) -> tuple[Any, list[ClusterInfo], list[int]]:
    """
    Load a checkpoint and split its foreground Gaussians into per-cluster
    ClusterInfo entries (mirrors cluster_pairs.py's main()), filtering out
    clusters smaller than min_cluster_size.

    :return: (model, clusters, filtered_cluster_ids)
    """
    checkpoint = (
        ckpt.expanduser().resolve()
        if ckpt is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(
            str(checkpoint), device, work_dir=str(work_dir), port=None
        )
        model = renderer.model
        model.eval()

        canonical_means = model.fg.params["means"]
        cluster_ids = model.fg.get_cluster_ids().reshape(-1).long().to(canonical_means.device)
        if canonical_means.shape[0] != cluster_ids.numel():
            raise RuntimeError("Foreground Gaussian count differs from cluster-ID count.")

        raw_ids = sorted(int(v) for v in torch.unique(cluster_ids).tolist())
        clusters: list[ClusterInfo] = []
        filtered_ids: list[int] = []
        for cluster_id in raw_ids:
            global_indices = torch.where(cluster_ids == cluster_id)[0]
            if global_indices.numel() < min_cluster_size:
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
        raise RuntimeError("Too few valid clusters. Lower --min-cluster-size.")

    return model, clusters, filtered_ids


def _greedy_unique_match(
    query_points: np.ndarray, pool_points: np.ndarray, k: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nearest-neighbor match query_points -> pool_points, but greedily enforce
    a *unique* pool assignment (no two query points share a pool point).

    Plain 1-NN matching lets many query points collapse onto the same pool
    point whenever the pool is sparser than the query set -- for the
    raw-tracks diagnostic that shows up as two different clusters' boundary
    points matching the identical raw track, i.e. gap_summary == 0.0 for every
    frame by construction, not because they're actually touching. Greedily
    claiming each query point's best *available* candidate among its k
    nearest removes that artifact at the source. A query point whose k
    nearest candidates are all already claimed is left UNMATCHED rather than
    falling back to a duplicate: forcing a match there is exactly what
    creates the exact-position collision this function exists to prevent
    (confirmed cause of the cluster-33 false edges -- 13/1760 and 1/1861
    Gaussians shared a bit-identical raw-track trajectory with another
    cluster). The caller drops unmatched points from the graph entirely.

    :return: (match_dist, match_idx, matched), each (len(query_points),).
        match_dist/match_idx are only meaningful where matched[i] is True.
    """
    tree = cKDTree(pool_points)
    k_eff = min(k, len(pool_points))
    dists, idxs = tree.query(query_points, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    order = np.argsort(dists[:, 0])
    claimed = np.zeros(len(pool_points), dtype=bool)
    match_idx = np.full(len(query_points), -1, dtype=np.int64)
    match_dist = np.full(len(query_points), np.inf, dtype=np.float64)
    matched = np.zeros(len(query_points), dtype=bool)

    for qi in order:
        for rank in range(k_eff):
            pi = int(idxs[qi, rank])
            if not claimed[pi]:
                claimed[pi] = True
                match_idx[qi] = pi
                match_dist[qi] = dists[qi, rank]
                matched[qi] = True
                break

    num_unmatched = int((~matched).sum())
    if num_unmatched:
        print(
            f"[raw_tracks] {num_unmatched}/{len(query_points)} Gaussians had all "
            f"{k_eff} nearest raw-track candidates already claimed by another Gaussian; "
            f"EXCLUDED from the graph (no forced duplicate match) -- raise k or the "
            f"track pool size if this count is large"
        )

    return match_dist, match_idx, matched


def load_raw_track_positions(
    work_dir: Path,
    clusters: list[ClusterInfo],
    num_query_frames: int = 20,
    num_samples_per_query: int = 20000,
    max_match_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], int]:
    """
    Per-frame positions (and per-frame confidences) from raw (pre-motion-basis)
    2D-lifted tracks, as an alternative to model.compute_poses_fg's *learned*
    motion. Diagnostic tool for telling a real self-contact apart from a
    learned-motion artifact: if the connectivity graph looks very different
    here vs. "learned", the "learned" graph was likely reflecting an
    undertrained/noisy motion basis (e.g. a cluster's points drifting onto
    the wrong limb) rather than actual physical contact.

    The returned confidences (raw cotracker per-point, per-frame confidence)
    feed cluster_graph's confidence-weighted gap smoothing, so an
    occluded/unreliable frame for a specific boundary point can't spike that
    edge's gap_summary on its own.

    There is no way to recover the *exact* raw track each fg Gaussian was
    initialized from (dataset.get_tracks_3d subsamples with un-seeded
    randomness), so instead this loads a dense, fresh pool of raw tracks --
    pooled from num_query_frames different query frames, not just one, since
    a single query frame's track count can be well under the total number of
    cluster Gaussians (observed: ~9.7k tracks vs ~47k Gaussians on one
    checkpoint), which forces unrelated clusters' boundary points to
    nearest-match the *same* track and reads as spurious always-touching
    (gap_summary == 0.0) contact. Pooling frames spreads that collision risk
    thin; see the printed match-count log, and note the pairs that were
    literally gap_summary == 0.0 in the per-edge log -- those are always exact
    match-collisions, not measured contact, no matter how the pooling is
    sized.

    :param clusters: only clusters actually needed (already min-size-filtered).
        Only model.fg (canonical means + cluster ids) is used here, both
        frame-count-independent, so this works against a checkpoint that
        hasn't been frame-propagated at all yet (e.g. straight out of init) --
        raw tracks come from the dataset's own cached track files, not from
        the model's motion bases, so the graph can cover the *full* video
        even when the model currently only spans its init window.
    :return: (positions_all_frames, confidences_all_frames,
        global_indices_by_cluster, num_frames). Indices are LOCAL to the
        returned (N, T, ...) arrays (N = sum of cluster sizes), not the
        original fg Gaussian ids -- these arrays only cover the clusters
        passed in, unlike the full-G array compute_poses_fg produces.
        num_frames is dataset.num_frames (the full video length), returned so
        the caller can log/record it instead of assuming it up front.
    """
    from flow3d.data.casual_dataset import CasualDataset
    from flow3d.data.utils import get_tracks_3d_for_query_frame

    cfg = yaml.safe_load((work_dir / "cfg.yaml").read_text())
    data_cfg = dict(cfg["data"])
    # Force cache reuse so the scene scale/transform matches what this
    # checkpoint was actually trained with, instead of a fresh (slightly
    # different, due to internal RNG) recomputation.
    data_cfg["load_from_cache"] = True

    dataset = CasualDataset(**data_cfg)
    num_frames = dataset.num_frames

    target_idcs = list(range(num_frames))
    masks = torch.stack([dataset.get_mask(i) for i in target_idcs], dim=0)
    depth_masks = torch.stack([dataset.get_depth_mask(i) for i in target_idcs], dim=0)
    fg_masks = (masks == 1).float() * (depth_masks == 1).float()
    depths = torch.stack([dataset.get_depth(i) for i in target_idcs], dim=0)
    inv_Ks = torch.linalg.inv(dataset.Ks[target_idcs])
    c2ws = torch.linalg.inv(dataset.w2cs[target_idcs])

    query_frames = sorted(
        set(np.linspace(0, num_frames - 1, num=min(num_query_frames, num_frames), dtype=int).tolist())
    )
    xyz_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    for q in query_frames:
        tracks_2d = dataset.load_target_tracks(q, target_idcs)  # (Nq, T, 4)
        if tracks_2d.shape[0] > num_samples_per_query:
            sel = np.random.choice(tracks_2d.shape[0], num_samples_per_query, replace=False)
            tracks_2d = tracks_2d[sel]
        query_img = dataset.get_image(q)
        tidx = target_idcs.index(q)
        xyz_q, _colors_q, _visibles_q, _invisibles_q, confidences_q, _depths_q = (
            get_tracks_3d_for_query_frame(
                tidx, query_img, tracks_2d, depths, fg_masks, inv_Ks, c2ws,
                track_type=dataset.track_2d_type,
            )
        )
        xyz_chunks.append(xyz_q.detach().float().cpu().numpy())
        confidence_chunks.append(confidences_q.detach().float().cpu().numpy())

    xyz = np.concatenate(xyz_chunks, axis=0)  # (N_pool, T, 3); frame axis is target_idcs = [0, num_frames)
    pool_confidences = np.concatenate(confidence_chunks, axis=0)  # (N_pool, T)
    cano_xyz = xyz[:, 0]

    all_canonical = (
        torch.cat([c.canonical_points for c in clusters], dim=0).detach().float().cpu().numpy()
    )
    match_dist, match_idx, matched = _greedy_unique_match(all_canonical, cano_xyz, k=12)

    if max_match_distance is not None:
        bad = matched & (match_dist > max_match_distance)
        if bad.any():
            print(
                f"[raw_tracks] warning: {int(bad.sum())}/{int(matched.sum())} matched cluster "
                f"Gaussians matched a raw track >{max_match_distance:.6f} away (unreliable match, "
                f"used anyway -- nearest available)"
            )

    # Points with no free pool track to claim are dropped entirely (see
    # _greedy_unique_match) rather than forced onto a duplicate -- so
    # match_idx is now injective on the kept subset and no two Gaussians can
    # end up with an exact-identical trajectory (the cluster-33 false-edge
    # cause) by construction.
    kept_idx = match_idx[matched]
    positions_all_frames = xyz[kept_idx]  # (num_matched, T, 3)
    confidences_all_frames = pool_confidences[kept_idx]  # (num_matched, T)
    print(
        f"[raw_tracks] pooled {cano_xyz.shape[0]} raw tracks from {len(query_frames)} query "
        f"frames {query_frames}; matched {int(matched.sum())}/{len(all_canonical)} cluster "
        f"Gaussians to distinct tracks (dropped {int((~matched).sum())} with no free track to "
        f"claim; median match distance {float(np.median(match_dist[matched])):.6f}, "
        f"mean track confidence {float(confidences_all_frames.mean()):.3f})"
    )

    global_indices_by_cluster: dict[int, np.ndarray] = {}
    offset = 0
    cluster_start = 0
    for cluster in clusters:
        n = cluster.size
        cluster_matched = matched[cluster_start : cluster_start + n]
        num_kept = int(cluster_matched.sum())
        if num_kept == 0:
            raise RuntimeError(
                f"cluster {cluster.cluster_id}: all {n} Gaussians were dropped for lacking a "
                "free raw-track match -- raise --raw-track-num-query-frames / "
                "--raw-track-samples-per-query to grow the track pool."
            )
        global_indices_by_cluster[cluster.cluster_id] = np.arange(offset, offset + num_kept, dtype=np.int64)
        offset += num_kept
        cluster_start += n

    return positions_all_frames, confidences_all_frames, global_indices_by_cluster, num_frames


def _edge_to_dict(e: ClusterPairEdge, frame_offset: int = 0) -> dict[str, Any]:
    return {
        "cluster_a": int(e.cluster_a),
        "cluster_b": int(e.cluster_b),
        "kept": bool(e.kept),
        "reason": e.reason,
        # frame_t_star is an index into the (possibly --gap-skip-first-n-frames
        # trimmed) sequence build_cluster_graph() saw -- add the offset back so
        # it reads as the original video's frame number everywhere it's reported.
        "frame_t_star": int(e.frame_t_star) + frame_offset,
        "center_distance_t_star": float(e.center_distance_t_star),
        "center_distance_median": float(e.center_distance_median),
        "center_distance_p10": float(e.center_distance_p10),
        "gap_summary": float(e.gap_summary),
        "gap_median": float(e.gap_median),
        "gap_min": float(e.gap_min),
        "gap_max": float(e.gap_max),
        "gap_mean_confidence": float(e.gap_mean_confidence),
        "gap_low_confidence_frac": float(e.gap_low_confidence_frac),
        "num_boundary_a": int(e.num_boundary_a),
        "num_boundary_b": int(e.num_boundary_b),
        "boundary_global_indices_a": torch.from_numpy(e.boundary_global_indices_a).long(),
        "boundary_global_indices_b": torch.from_numpy(e.boundary_global_indices_b).long(),
        "contact_distance": float(e.contact_distance),
        "threshold": float(e.threshold),
    }


def _edge_to_csv_row(e: ClusterPairEdge, frame_offset: int = 0) -> dict[str, Any]:
    row = _edge_to_dict(e, frame_offset)
    row.pop("boundary_global_indices_a")
    row.pop("boundary_global_indices_b")
    return row


def log_summary(result: ClusterGraphResult, scene_scale: float) -> None:
    kept, cut = result.kept_edges, result.cut_edges
    print(f"Candidate pairs : {result.candidate_pair_count}")
    print(f"Kept edges      : {len(kept)}")
    print(f"Cut edges       : {len(cut)}")
    print(
        f"Contact distance: {result.contact_distance:.6f} "
        f"(scene_scale={scene_scale:.6f}, "
        f"contact/scene_scale={result.contact_distance / max(scene_scale, 1e-12):.6f})"
    )
    print(
        f"Keep threshold  : {result.contact_distance * result.config.keep_multiplier:.6f} "
        f"(keep_multiplier={result.config.keep_multiplier})"
    )


def log_gap_histogram(result: ClusterGraphResult, num_bins: int = 10) -> None:
    gaps = np.asarray([e.gap_summary for e in result.edges], dtype=np.float64)
    if gaps.size == 0:
        print("[gap histogram] no candidate edges.")
        return
    ratios = gaps / max(result.contact_distance, 1e-12)
    counts, bin_edges = np.histogram(ratios, bins=num_bins)
    print(
        f"[gap histogram] gap_summary / contact_distance "
        f"(keep cutoff at {result.config.keep_multiplier:.2f}):"
    )
    for count, lo, hi in zip(counts, bin_edges[:-1], bin_edges[1:]):
        print(f"  [{lo:6.2f}, {hi:6.2f}) {count:4d} {'#' * int(count)}")


def log_edge_reasons(result: ClusterGraphResult, frame_offset: int = 0) -> None:
    for e in sorted(result.edges, key=lambda e: (e.cluster_a, e.cluster_b)):
        tag = "KEEP" if e.kept else "CUT "
        print(
            f"[{tag}] {e.cluster_a}-{e.cluster_b} gap_summary={e.gap_summary:.6f} "
            f"threshold={e.threshold:.6f} t*={e.frame_t_star + frame_offset} reason={e.reason}"
        )


def log_neighbor_lists(result: ClusterGraphResult) -> None:
    neighbors = result.neighbors()
    for cid in result.cluster_ids:
        print(f"[neighbors] cluster {cid}: {sorted(neighbors.get(cid, []))}")


def log_gate_check(
    result: ClusterGraphResult,
    target_id: int,
    expect_neighbor: int,
    expect_absent: list[int],
) -> None:
    neighbors = result.neighbors()
    if target_id not in neighbors:
        print(f"[gate] cluster {target_id} is not a valid cluster here; skipping gate check.")
        return
    ns = sorted(neighbors[target_id])
    ok_present = expect_neighbor in ns
    ok_absent = all(x not in ns for x in expect_absent)
    status = "PASS" if (ok_present and ok_absent) else "FAIL"
    print(f"[gate] cluster {target_id} neighbors: {ns}")
    print(
        f"[gate] expect {expect_neighbor} in neighbors: {ok_present}; "
        f"expect {expect_absent} absent: {ok_absent} -> {status}"
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    work_dir = args.work_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "analysis" / "cluster_graph"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_pt = output_dir / "edges.pt"
    if edges_pt.exists() and not args.overwrite:
        raise FileExistsError(f"{edges_pt} already exists. Pass --overwrite to replace it.")

    model, clusters, filtered_ids = load_model_and_clusters(
        work_dir=work_dir,
        ckpt=args.ckpt,
        device_name=args.device,
        min_cluster_size=args.min_cluster_size,
    )
    cluster_by_id = {c.cluster_id: c for c in clusters}
    valid_ids = sorted(cluster_by_id)
    device = model.fg.params["means"].device
    scene_scale = float(model.fg.scene_scale.item())

    within_spacing, spacing_sample_count = estimate_within_cluster_spacing(
        clusters, sample_limit=args.spacing_sample_limit
    )
    contact_distance = (
        float(args.contact_distance)
        if args.contact_distance is not None
        else within_spacing * float(args.contact_spacing_multiplier)
    )

    if args.position_source == "raw_tracks":
        max_match_distance = (
            args.raw_track_max_match_distance
            if args.raw_track_max_match_distance is not None
            else contact_distance * 5.0
        )
        # Raw tracks come from the dataset's own cached track files, not the
        # model's motion bases -- num_frames is the full video length
        # (dataset.num_frames), independent of how far this checkpoint has
        # been frame-propagated. This can be less than model.num_frames
        # (a not-yet-propagated checkpoint) without any problem.
        positions_all_frames, confidences_all_frames, global_indices_by_cluster, num_frames = (
            load_raw_track_positions(
                work_dir=work_dir,
                clusters=clusters,
                num_query_frames=args.raw_track_num_query_frames,
                num_samples_per_query=args.raw_track_samples_per_query,
                max_match_distance=max_match_distance,
            )
        )
    else:
        num_frames = model.num_frames
        with torch.no_grad():
            means_all, _ = model.compute_poses_fg(torch.arange(num_frames, device=device))
        positions_all_frames = means_all.detach().float().cpu().numpy()  # (G, T, 3)
        # No natural per-frame confidence for a learned motion basis.
        confidences_all_frames = None
        global_indices_by_cluster = {
            cid: cluster_by_id[cid].global_indices.detach().cpu().numpy().astype(np.int64)
            for cid in valid_ids
        }

    skip = args.gap_skip_first_n_frames
    if skip > 0:
        if skip >= num_frames:
            raise ValueError(
                f"--gap-skip-first-n-frames ({skip}) must be < num_frames ({num_frames})"
            )
        print(f"[gap] skipping frames 0..{skip - 1}; {num_frames - skip}/{num_frames} frames remain")
        positions_all_frames = positions_all_frames[:, skip:, :]
        if confidences_all_frames is not None:
            confidences_all_frames = confidences_all_frames[:, skip:]
        num_frames -= skip

    config = ClusterGraphConfig(
        candidate_radius_multiplier=args.candidate_radius_multiplier,
        center_gate_max_median_to_p10_ratio=args.center_gate_max_median_p10_ratio,
        keep_multiplier=args.keep_multiplier,
        gap_smoothing_window=args.gap_smoothing_window,
        gap_confidence_threshold=args.gap_confidence_threshold,
        boundary_fraction=args.boundary_fraction,
        boundary_min_gaussians=args.boundary_min_gaussians,
        boundary_max_gaussians=args.boundary_max_gaussians,
        boundary_max_distance=args.boundary_max_distance,
    )

    print(f"Valid clusters  : {len(valid_ids)}")
    print(f"Frames          : {num_frames}")
    print(f"Position source : {args.position_source}")
    print(f"Within-cluster median spacing: {within_spacing:.9f} (n={spacing_sample_count})")

    result = build_cluster_graph(
        cluster_ids=valid_ids,
        global_indices_by_cluster=global_indices_by_cluster,
        positions_all_frames=positions_all_frames,
        contact_distance=contact_distance,
        config=config,
        confidences_all_frames=confidences_all_frames,
    )

    print()
    log_summary(result, scene_scale)
    print()
    log_gap_histogram(result)
    print()
    log_edge_reasons(result, frame_offset=skip)
    print()
    log_neighbor_lists(result)
    print()
    gate_expect_absent = [
        int(v.strip()) for v in args.gate_expect_absent.split(",") if v.strip()
    ]
    log_gate_check(result, args.gate_cluster_id, args.gate_expect_neighbor, gate_expect_absent)

    edges_kept_dicts = [_edge_to_dict(e, frame_offset=skip) for e in result.kept_edges]
    edges_cut_dicts = [_edge_to_dict(e, frame_offset=skip) for e in result.cut_edges]

    payload = {
        "edge_index": torch.from_numpy(result.edge_index()).long(),
        "edges_kept": edges_kept_dicts,
        "edges_cut": edges_cut_dicts,
        "cluster_ids": valid_ids,
        "meta": {
            "work_dir": str(work_dir),
            "checkpoint": str(
                args.ckpt.expanduser().resolve()
                if args.ckpt is not None
                else work_dir / "checkpoints" / "last.ckpt"
            ),
            "num_frames": int(num_frames),
            "gap_skip_first_n_frames": skip,
            "position_source": args.position_source,
            "scene_scale": scene_scale,
            "contact_distance": result.contact_distance,
            "contact_distance_source": (
                "explicit" if args.contact_distance is not None else "within_cluster_nn_median"
            ),
            "within_cluster_median_spacing": within_spacing,
            "spacing_sample_count": spacing_sample_count,
            "candidate_pair_count": result.candidate_pair_count,
            "kept_edge_count": len(edges_kept_dicts),
            "cut_edge_count": len(edges_cut_dicts),
            "valid_cluster_ids": valid_ids,
            "filtered_cluster_ids": filtered_ids,
            "config": {**asdict(config), "min_cluster_size": args.min_cluster_size},
        },
    }
    torch.save(payload, edges_pt)

    write_csv(output_dir / "edges_kept.csv", [_edge_to_csv_row(e, frame_offset=skip) for e in result.kept_edges])
    write_csv(output_dir / "edges_cut.csv", [_edge_to_csv_row(e, frame_offset=skip) for e in result.cut_edges])

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {**payload["meta"], "gate_check": {
                "cluster_id": args.gate_cluster_id,
                "expect_neighbor": args.gate_expect_neighbor,
                "expect_absent": gate_expect_absent,
                "actual_neighbors": sorted(result.neighbors().get(args.gate_cluster_id, [])),
            }},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    visualization_files: list[str] = []
    if not args.no_visualization:
        path_3d = render_3d_multiview(
            clusters=clusters,
            kept_edges=edges_kept_dicts,
            cut_edges=edges_cut_dicts,
            contact_distance=result.contact_distance,
            keep_multiplier=config.keep_multiplier,
            output_path=output_dir / "graph_edges.png",
            max_points_per_cluster=args.max_points_per_cluster,
        )
        visualization_files.append(str(path_3d))
        print(f"[visualization] saved 3D multiview: {path_3d}")

        path_2d = render_2d_overlay(
            model=model,
            clusters=clusters,
            kept_edges=edges_kept_dicts,
            cut_edges=edges_cut_dicts,
            contact_distance=result.contact_distance,
            keep_multiplier=config.keep_multiplier,
            output_path=output_dir / "graph_edges_2d.png",
            frame_index=args.frame_index_2d,
        )
        visualization_files.append(str(path_2d))
        print(f"[visualization] saved 2D overlay: {path_2d}")

    print()
    print("=" * 72)
    print("Cluster connectivity graph build complete")
    print(f"Valid clusters : {len(valid_ids)}")
    print(f"Candidates     : {result.candidate_pair_count}")
    print(f"Kept edges     : {len(edges_kept_dicts)}")
    print(f"Cut edges      : {len(edges_cut_dicts)}")
    print(f"Output PT      : {edges_pt}")
    if visualization_files:
        print("Visualizations :")
        for path in visualization_files:
            print(f"  {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

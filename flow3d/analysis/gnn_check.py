#!/usr/bin/env python3
"""
Quantify what the graph-coupling GNN (flow3d/graph_coupling.py /
flow3d/graph_coupling_relative.py) actually changes on a trained checkpoint.

Motivation
----------
GT-based renders/metrics only show the *net* effect of a change, mixed in
with everything else the model does. This script isolates the GNN's own
contribution directly, without touching GT:

1. Correction magnitude per cluster per frame: |omega| (rotation angle, via
   the so(3) exponential map used to compose the correction) and |delta_t|
   (translation), read straight off GraphCorrectedScalableMotionBases /
   RelativeGraphCorrectedScalableMotionBases.last_correction.
2. Direct effect on Gaussian positions: for the same frames/coefs/cluster_ids,
   the transform is computed twice -- once through the graph-corrected
   subclass (as used in training/rendering) and once through the *parent*
   ScalableMotionBases.compute_transforms called on the same instance (same
   coarse/fine params, but the override -- and hence the GNN -- never runs).
   The two are otherwise identical inputs, so their per-point world-space
   position difference is exactly what the GNN moved, in world units, with
   no GT involved.
3. Which clusters the GNN actually touches: ranks clusters by mean
   correction size and by mean point displacement, so e.g. a hand cluster's
   coupling to its forearm cluster (the motivating case in
   flow3d/graph_coupling.py) can be checked directly instead of inferred
   from a GT metric moving.

Outputs (under <output-dir>, default <work-dir>/analysis/gnn_check)
---------------------------------------------------------------------
    report.json                 -- summary stats + per-cluster ranking
    per_frame.csv                -- correction / displacement stats per frame
    per_cluster.csv               -- correction / displacement stats per cluster
    correction_over_time.png     -- |omega|, |delta_t| per cluster vs frame
    point_displacement_over_time.png -- with-GNN vs no-GNN position diff vs frame
    cluster_ranking.png          -- top clusters by correction size
    point_displacement_map.png   -- canonical Gaussians colored by displacement,
                                     with the fixed graph edges drawn for context

Example
-------
    python flow3d/analysis/gnn_check.py \\
        --work-dir outputs/davis/spin/2026_08_03_08_41_11__spin_run1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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

from flow3d.analysis.cluster_pairs import _set_axes_equal_3d
from flow3d.params import ScalableMotionBases
from flow3d.renderer import Renderer

MAP_VIEWS = [(20, 35, "Perspective"), (20, 125, "Left / Back")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="Evaluate every Nth frame (default: all frames).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap the number of evaluated frames after striding (default: no cap).",
    )
    parser.add_argument(
        "--point-subsample", type=int, default=20000,
        help="Max fg Gaussians used for the point-displacement pass "
        "(deterministic random subsample; the whole scene is used if smaller).",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Clusters highlighted in plots/report.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _resolve_edges(gnn: Any) -> list[tuple[int, int]]:
    """Undirected (a, b) edges (a < b) this GNN instance actually message-passes
    over at inference, read from its own fixed-topology buffer -- not
    re-derived from edges.pt, so this reflects what the checkpoint really uses."""
    if hasattr(gnn, "edge_index_dir"):  # RelativeClusterGraphGNN
        ei = gnn.edge_index_dir.cpu().numpy()
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(ei[0], ei[1])}
        return sorted(pairs)

    adj = gnn.adj_norm.cpu().numpy()  # ClusterGraphGNN: (C, C), self-loops on diagonal
    num_clusters = adj.shape[0]
    pairs = set()
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if adj[i, j] > 0 or adj[j, i] > 0:
                pairs.add((i, j))
    return sorted(pairs)


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def _plot_correction_over_time(
    frame_ids: np.ndarray,
    rot_angle_deg: np.ndarray,  # (C, T)
    trans_mag: np.ndarray,  # (C, T)
    top_clusters: list[int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("tab10")

    for ax, data, ylabel, title in (
        (axes[0], rot_angle_deg, "rotation correction (deg)", "|omega| per cluster"),
        (axes[1], trans_mag, "translation correction (world units)", "|delta_t| per cluster"),
    ):
        for cluster_id in range(data.shape[0]):
            if cluster_id in top_clusters:
                continue
            ax.plot(frame_ids, data[cluster_id], color="gray", alpha=0.2, linewidth=0.8)
        for rank, cluster_id in enumerate(top_clusters):
            ax.plot(
                frame_ids, data[cluster_id],
                color=cmap(rank % cmap.N), linewidth=1.8, label=f"cluster {cluster_id}",
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_point_displacement_over_time(
    frame_ids: np.ndarray,
    point_disp: np.ndarray,  # (G, T)
    output_path: Path,
) -> None:
    mean_t = point_disp.mean(axis=0)
    median_t = np.median(point_disp, axis=0)
    p90_t = np.percentile(point_disp, 90, axis=0)
    max_t = point_disp.max(axis=0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame_ids, mean_t, label="mean", linewidth=1.8)
    ax.plot(frame_ids, median_t, label="median", linewidth=1.4, linestyle="--")
    ax.plot(frame_ids, p90_t, label="p90", linewidth=1.4, linestyle=":")
    ax.plot(frame_ids, max_t, label="max", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("frame")
    ax.set_ylabel("|position(with GNN) - position(GNN off)|  (world units)")
    ax.set_title("Direct effect of the GNN correction on Gaussian positions")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_ranking(
    per_cluster: list[dict],
    top_k: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title, xlabel in (
        (axes[0], "mean_rot_angle_deg", "Top clusters by rotation correction", "mean |omega| (deg)"),
        (axes[1], "mean_point_displacement", "Top clusters by point displacement", "mean world-unit displacement"),
    ):
        ranked = sorted(per_cluster, key=lambda r: r[key], reverse=True)[:top_k]
        ids = [f"cluster {r['cluster_id']}" for r in ranked][::-1]
        values = [r[key] for r in ranked][::-1]
        ax.barh(ids, values, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_displacement_map(
    canonical_points: np.ndarray,  # (N, 3), subsampled fg means
    point_disp_mean: np.ndarray,  # (N,), mean over evaluated frames
    cluster_centers: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    output_path: Path,
) -> None:
    color = np.log1p(point_disp_mean)

    fig = plt.figure(figsize=(14, 6.5))
    for subplot_index, (elev, azim, title) in enumerate(MAP_VIEWS, start=1):
        ax = fig.add_subplot(1, len(MAP_VIEWS), subplot_index, projection="3d")
        scatter = ax.scatter(
            canonical_points[:, 0], canonical_points[:, 1], canonical_points[:, 2],
            c=color, cmap="viridis", s=2.5, alpha=0.6, rasterized=True,
        )
        for a, b in edges:
            if a not in cluster_centers or b not in cluster_centers:
                continue
            ca, cb = cluster_centers[a], cluster_centers[b]
            ax.plot([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]], color="black", alpha=0.5, linewidth=1.2)
        for cluster_id, center in cluster_centers.items():
            ax.scatter([center[0]], [center[1]], [center[2]], s=14, color="red", zorder=3)
            ax.text(center[0], center[1], center[2], str(cluster_id), fontsize=7, color="black")
        _set_axes_equal_3d(ax, canonical_points)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)

    fig.suptitle("Mean point displacement from GNN correction (log1p scale, red=cluster center)", fontsize=12)
    fig.colorbar(scatter, ax=fig.axes, shrink=0.6, label="log1p(world-unit displacement)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    ckpt = (args.ckpt if args.ckpt is not None else work_dir / "checkpoints" / "last.ckpt").expanduser().resolve()
    output_dir = (
        args.output_dir if args.output_dir is not None else work_dir / "analysis" / "gnn_check"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(str(ckpt), device, work_dir=str(work_dir), port=None)
        model = renderer.model
        model.eval()

        motion_bases = model.motion_bases
        gnn = getattr(motion_bases, "gnn", None)
        if gnn is None:
            raise RuntimeError(
                f"{ckpt} has no motion_bases.gnn -- it wasn't trained with "
                "--enable_graph_coupling, so there's no GNN correction to check."
            )
        variant = "relative" if hasattr(gnn, "edge_index_dir") else "absolute"
        num_clusters = motion_bases.num_clusters
        num_frames_total = motion_bases.num_frames
        edges = _resolve_edges(gnn)
        print(
            f"[gnn_check] variant={variant}  num_clusters={num_clusters}  "
            f"num_frames={num_frames_total}  edges={len(edges)}"
        )

        frame_ids_t = torch.arange(0, num_frames_total, args.frame_stride, device=device, dtype=torch.long)
        if args.max_frames is not None:
            frame_ids_t = frame_ids_t[: args.max_frames]
        num_frames_eval = int(frame_ids_t.numel())
        print(f"[gnn_check] evaluating {num_frames_eval} frames (stride={args.frame_stride})")

        # --- 1. per-cluster correction magnitude (omega, delta_t) ---
        all_cluster_ids = torch.arange(num_clusters, device=device)
        motion_bases.compute_transforms_coarse(frame_ids_t, all_cluster_ids)  # populates last_correction
        omega = motion_bases.last_correction["omega"]  # (C, T, 3)
        delta_t = motion_bases.last_correction["delta_t"]  # (C, T, 3)
        rot_angle_deg = (omega.norm(dim=-1) * (180.0 / np.pi)).cpu().numpy()  # (C, T)
        trans_mag = delta_t.norm(dim=-1).cpu().numpy()  # (C, T)

        # --- 2. direct effect on Gaussian positions: with GNN vs GNN bypassed ---
        coefs_all = model.fg.get_coefs()  # (G, F)
        cluster_ids_pts_all = model.fg.get_cluster_ids()  # (G,)
        fg_means_all = model.fg.params["means"]  # (G, 3)

        num_fg = fg_means_all.shape[0]
        if num_fg > args.point_subsample:
            perm = torch.randperm(num_fg, generator=torch.Generator().manual_seed(0))[: args.point_subsample]
            perm = perm.to(device)
        else:
            perm = torch.arange(num_fg, device=device)

        coefs = coefs_all[perm]
        cluster_ids_pts = cluster_ids_pts_all[perm]
        fg_means = fg_means_all[perm]
        fg_means_h = torch.cat([fg_means, torch.ones(fg_means.shape[0], 1, device=device)], dim=-1)

        transfms_with = motion_bases.compute_transforms(frame_ids_t, coefs, cluster_ids_pts)  # (G, T, 3, 4)
        transfms_without = ScalableMotionBases.compute_transforms(
            motion_bases, frame_ids_t, coefs, cluster_ids_pts
        )  # (G, T, 3, 4); bypasses the GNN override entirely

        means_with = torch.einsum("gtij,gj->gti", transfms_with, fg_means_h)
        means_without = torch.einsum("gtij,gj->gti", transfms_without, fg_means_h)
        point_disp = (means_with - means_without).norm(dim=-1).cpu().numpy()  # (G, T)

        scene_extent = float((fg_means_all.max(dim=0).values - fg_means_all.min(dim=0).values).norm().item())
        typical_motion = (
            (means_without - means_without[:, :1]).norm(dim=-1).amax(dim=1).median().item()
            if num_frames_eval > 1
            else float("nan")
        )

        cluster_centers = {
            int(cid): fg_means_all[cluster_ids_pts_all == cid].mean(dim=0).cpu().numpy()
            for cid in range(num_clusters)
            if (cluster_ids_pts_all == cid).any()
        }

        canonical_points_np = fg_means.cpu().numpy()
        cluster_ids_pts_np = cluster_ids_pts.cpu().numpy()

    frame_ids_np = frame_ids_t.cpu().numpy()

    # --- per-cluster aggregation ---
    per_cluster: list[dict] = []
    for cluster_id in range(num_clusters):
        mask = cluster_ids_pts_np == cluster_id
        num_points = int(mask.sum())
        entry = {
            "cluster_id": cluster_id,
            "num_points": num_points,
            "mean_rot_angle_deg": float(rot_angle_deg[cluster_id].mean()),
            "max_rot_angle_deg": float(rot_angle_deg[cluster_id].max()),
            "mean_trans_mag": float(trans_mag[cluster_id].mean()),
            "max_trans_mag": float(trans_mag[cluster_id].max()),
            "mean_point_displacement": float(point_disp[mask].mean()) if num_points else 0.0,
            "max_point_displacement": float(point_disp[mask].max()) if num_points else 0.0,
        }
        per_cluster.append(entry)

    top_by_rotation = sorted(per_cluster, key=lambda r: r["mean_rot_angle_deg"], reverse=True)[: args.top_k]
    top_by_translation = sorted(per_cluster, key=lambda r: r["mean_trans_mag"], reverse=True)[: args.top_k]
    top_by_displacement = sorted(per_cluster, key=lambda r: r["mean_point_displacement"], reverse=True)[: args.top_k]
    top_cluster_ids = sorted({r["cluster_id"] for r in top_by_rotation} | {r["cluster_id"] for r in top_by_translation})

    # --- per-frame aggregation ---
    per_frame_rows = []
    for t_idx, frame_id in enumerate(frame_ids_np):
        per_frame_rows.append(
            {
                "frame": int(frame_id),
                "mean_rot_angle_deg": float(rot_angle_deg[:, t_idx].mean()),
                "max_rot_angle_deg": float(rot_angle_deg[:, t_idx].max()),
                "mean_trans_mag": float(trans_mag[:, t_idx].mean()),
                "max_trans_mag": float(trans_mag[:, t_idx].max()),
                "mean_point_disp": float(point_disp[:, t_idx].mean()),
                "median_point_disp": float(np.median(point_disp[:, t_idx])),
                "p90_point_disp": float(np.percentile(point_disp[:, t_idx], 90)),
                "max_point_disp": float(point_disp[:, t_idx].max()),
            }
        )

    overall = {
        "rot_angle_deg": _stats(rot_angle_deg),
        "trans_mag": _stats(trans_mag),
        "point_displacement": _stats(point_disp),
        "scene_extent": scene_extent,
        "mean_point_displacement_relative_to_scene_extent_pct": (
            100.0 * float(point_disp.mean()) / scene_extent if scene_extent > 0 else float("nan")
        ),
        "typical_frame_to_frame_motion_range": typical_motion,
        "mean_point_displacement_relative_to_typical_motion_pct": (
            100.0 * float(point_disp.mean()) / typical_motion
            if typical_motion and typical_motion > 0
            else float("nan")
        ),
    }

    report = {
        "checkpoint": str(ckpt),
        "variant": variant,
        "num_clusters": num_clusters,
        "num_frames_total": num_frames_total,
        "num_frames_evaluated": num_frames_eval,
        "frame_stride": args.frame_stride,
        "num_points_evaluated": int(fg_means.shape[0]),
        "num_edges": len(edges),
        "edges": [list(e) for e in edges],
        "overall": overall,
        "top_clusters_by_rotation_correction": top_by_rotation,
        "top_clusters_by_translation_correction": top_by_translation,
        "top_clusters_by_point_displacement": top_by_displacement,
        "per_cluster": per_cluster,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    with (output_dir / "per_frame.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    with (output_dir / "per_cluster.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_cluster[0].keys()))
        writer.writeheader()
        writer.writerows(per_cluster)

    print(f"[gnn_check] wrote report.json / per_frame.csv / per_cluster.csv to {output_dir}")
    print(
        f"[gnn_check] mean point displacement = {overall['point_displacement']['mean']:.4g} "
        f"world units ({overall['mean_point_displacement_relative_to_scene_extent_pct']:.3g}% of scene extent, "
        f"{overall['mean_point_displacement_relative_to_typical_motion_pct']:.3g}% of typical frame-to-frame motion range)"
    )
    print(f"[gnn_check] top clusters by rotation correction: {[r['cluster_id'] for r in top_by_rotation]}")
    print(f"[gnn_check] top clusters by point displacement: {[r['cluster_id'] for r in top_by_displacement]}")

    if not args.no_plots:
        _plot_correction_over_time(
            frame_ids_np, rot_angle_deg, trans_mag, top_cluster_ids, output_dir / "correction_over_time.png"
        )
        _plot_point_displacement_over_time(frame_ids_np, point_disp, output_dir / "point_displacement_over_time.png")
        _plot_cluster_ranking(per_cluster, args.top_k, output_dir / "cluster_ranking.png")
        _plot_displacement_map(
            canonical_points_np,
            point_disp.mean(axis=1),
            cluster_centers,
            edges,
            output_dir / "point_displacement_map.png",
        )
        print(f"[gnn_check] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Edge-centric structural checks for the graph-coupling GNN (flow3d/graph_coupling.py /
flow3d/graph_coupling_relative.py), complementing gnn_check.py.

Motivation
----------
gnn_check.py quantifies the GNN's per-CLUSTER correction magnitude and its net
effect on point positions, but it never looks at whether the GNN is doing its
actual job: keeping the cluster PAIRS it message-passes over structurally
together. A cluster can move a lot (large correction, large point
displacement) while still opening up a gap at its boundary with a neighbor,
or two neighbors can each get a large *individually* reasonable correction
that nonetheless pulls them apart. Rendering/GT metrics only show the net
mix of everything the model does, so this script checks the graph edges
directly, without touching GT:

1. Cluster-pair boundary gap (with GNN vs GNN bypassed): for every edge the
   GNN itself actually message-passes over (read from the checkpoint's own
   edge_index_dir/adj_norm buffer, not re-derived), use edges.pt's reference
   frame/config to reselect FIXED boundary Gaussian indices on the current
   checkpoint and recompute the per-frame nearest-neighbor boundary gap
   (flow3d/analysis/cluster_graph.py's compute_all_frames_gap) twice -- once
   through the graph-corrected transforms (with GNN) and once through the
   *parent* ScalableMotionBases.compute_transforms on the same instance (GNN
   bypassed, same coarse/fine params otherwise). The two use the identical
   boundary points and frames, so their difference is exactly what the GNN
   did to that edge's gap, frame by frame -- including whether an edge that
   would have exceeded the graph-build keep threshold without the GNN stays
   under it with the GNN.
2. Edge-wise relative correction: for the same edges, per-frame (omega,
   delta_t) is read directly off motion_bases.last_correction for both
   endpoints and compared: translation correction difference
   |delta_t_a - delta_t_b| (world units) and rotation correction difference,
   the geodesic angle between exp(omega_a) and exp(omega_b) (degrees). A
   small, correlated correction across an edge preserves that pair's
   relative pose; a large, uncorrelated one pulls it apart -- checkable per
   edge, per frame, with no GT involved.

Outputs (under <output-dir>, default <work-dir>/analysis/gnn_check2)
----------------------------------------------------------------------
    report.json                        -- summary stats + per-edge ranking
    per_edge.csv                       -- gap / relative-motion stats per edge
    per_edge_frame.csv                 -- gap_with/gap_without/trans_diff/rot_diff per edge per frame
    gap_comparison_over_time.png       -- boundary gap, with vs without GNN, top edges
    relative_motion_over_time.png      -- |delta_t_a-delta_t_b| and rotation diff vs frame
    edge_ranking.png                   -- top edges by gap improvement / relative motion
    edge_relative_vs_gap_scatter.png   -- relative-correction diff vs gap improvement, per edge

Example
-------
    python flow3d/analysis/gnn_check2.py \\
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
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.analysis.cluster_graph import (
    compute_all_frames_gap,
    select_boundary_at_frame,
    smooth_and_summarize_gap,
)
from flow3d.analysis.gnn_check import _resolve_edges, _stats
from flow3d.graph_coupling import so3_exp_map
from flow3d.params import ScalableMotionBases
from flow3d.renderer import Renderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--edges-pt", type=Path, default=None,
        help="Default: <work-dir>/analysis/cluster_graph/edges.pt, falling back to "
        "cfg.yaml's graph_coupling_path (the same file the GNN's edge_index was built from).",
    )
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
    parser.add_argument("--top-k", type=int, default=8, help="Edges highlighted in plots/report.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _load_edge_boundary_map(edges_pt: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """(a, b) -> the edges.pt record (kept or cut) for that pair, plus payload['meta']."""
    if not edges_pt.is_file():
        raise FileNotFoundError(
            f"edges.pt not found: {edges_pt}. Run build_cluster_graph.py first, or pass --edges-pt."
        )
    payload = torch.load(edges_pt, map_location="cpu", weights_only=False)
    boundary_map: dict[tuple[int, int], dict[str, Any]] = {}
    # Kept edges take priority; cut edges are a fallback so a resolved GNN
    # edge whose current edges.pt classifies it differently (rebuilt with
    # different knobs since training) still gets its boundary indices.
    for e in payload.get("edges_cut", []):
        key = (min(int(e["cluster_a"]), int(e["cluster_b"])), max(int(e["cluster_a"]), int(e["cluster_b"])))
        boundary_map[key] = e
    for e in payload.get("edges_kept", []):
        key = (min(int(e["cluster_a"]), int(e["cluster_b"])), max(int(e["cluster_a"]), int(e["cluster_b"])))
        boundary_map[key] = e
    return boundary_map, payload.get("meta", {})


def _plot_gap_comparison(
    frame_ids: np.ndarray,
    per_edge: list[dict[str, Any]],
    top_edges: list[tuple[int, int]],
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("tab10")
    by_pair = {(r["cluster_a"], r["cluster_b"]): r for r in per_edge}

    for rank, pair in enumerate(top_edges):
        r = by_pair[pair]
        color = cmap(rank % cmap.N)
        ax.plot(frame_ids, r["gap_t_with"], color=color, linewidth=1.8, label=f"{pair[0]}-{pair[1]}")
        ax.plot(frame_ids, r["gap_t_without"], color=color, linewidth=1.2, linestyle="--", alpha=0.7)

    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2, label="keep threshold")
    ax.set_xlabel("frame")
    ax.set_ylabel("boundary gap (world units)")
    ax.set_title("Cluster-pair boundary gap -- solid = with GNN, dashed = GNN bypassed")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_relative_motion(
    frame_ids: np.ndarray,
    per_edge: list[dict[str, Any]],
    top_edges: list[tuple[int, int]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("tab10")
    by_pair = {(r["cluster_a"], r["cluster_b"]): r for r in per_edge}
    top_set = set(top_edges)

    for ax, key, ylabel, title in (
        (axes[0], "rot_diff_t", "rotation correction diff (deg)", "|exp(omega_a) vs exp(omega_b)| per edge"),
        (axes[1], "trans_diff_t", "translation correction diff (world units)", "|delta_t_a - delta_t_b| per edge"),
    ):
        for r in per_edge:
            pair = (r["cluster_a"], r["cluster_b"])
            if pair in top_set:
                continue
            ax.plot(frame_ids, r[key], color="gray", alpha=0.2, linewidth=0.8)
        for rank, pair in enumerate(top_edges):
            r = by_pair[pair]
            ax.plot(
                frame_ids, r[key],
                color=cmap(rank % cmap.N), linewidth=1.8, label=f"{pair[0]}-{pair[1]}",
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_edge_ranking(per_edge: list[dict[str, Any]], top_k: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ranked = sorted(per_edge, key=lambda r: r["gap_reduction_mean"], reverse=True)
    shown = (ranked[:top_k] + ranked[-top_k:]) if len(ranked) > 2 * top_k else ranked
    shown = sorted(shown, key=lambda r: r["gap_reduction_mean"])
    labels = [f"{r['cluster_a']}-{r['cluster_b']}" for r in shown]
    values = [r["gap_reduction_mean"] for r in shown]
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in values]
    axes[0].barh(labels, values, color=colors)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Gap reduction (without - with GNN)\ngreen = GNN helps, red = GNN hurts")
    axes[0].set_xlabel("mean gap reduction (world units)")
    axes[0].grid(alpha=0.25, axis="x")

    for ax, key, title, xlabel in (
        (axes[1], "trans_diff_mean", "Top edges by translation relative motion", "mean |delta_t_a-delta_t_b|"),
        (axes[2], "rot_diff_deg_mean", "Top edges by rotation relative motion", "mean rotation diff (deg)"),
    ):
        ranked_k = sorted(per_edge, key=lambda r: r[key], reverse=True)[:top_k]
        ids = [f"{r['cluster_a']}-{r['cluster_b']}" for r in ranked_k][::-1]
        vals = [r[key] for r in ranked_k][::-1]
        ax.barh(ids, vals, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_relative_vs_gap_scatter(per_edge: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, xlabel in (
        (axes[0], "trans_diff_mean", "mean translation correction diff (world units)"),
        (axes[1], "rot_diff_deg_mean", "mean rotation correction diff (deg)"),
    ):
        x = [r[key] for r in per_edge]
        y = [r["gap_reduction_mean"] for r in per_edge]
        ax.scatter(x, y, s=28, color="steelblue", alpha=0.8)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        for r in per_edge:
            ax.annotate(
                f"{r['cluster_a']}-{r['cluster_b']}", (r[key], r["gap_reduction_mean"]),
                fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean gap reduction (without - with GNN)")
        ax.grid(alpha=0.25)

    fig.suptitle("Does a smaller cross-edge correction diff track a smaller boundary gap?", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    ckpt = (args.ckpt if args.ckpt is not None else work_dir / "checkpoints" / "last.ckpt").expanduser().resolve()
    if args.edges_pt is not None:
        edges_pt = args.edges_pt.expanduser().resolve()
    else:
        edges_pt = (work_dir / "analysis" / "cluster_graph" / "edges.pt").resolve()
        if not edges_pt.is_file():
            cfg_path = work_dir / "cfg.yaml"
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
            configured_edges = cfg.get("graph_coupling_path") if isinstance(cfg, dict) else None
            if configured_edges:
                configured_path = Path(configured_edges).expanduser()
                edges_pt = (
                    configured_path if configured_path.is_absolute()
                    else _REPO_ROOT / configured_path
                ).resolve()
    output_dir = (
        args.output_dir if args.output_dir is not None else work_dir / "analysis" / "gnn_check2"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    boundary_map, edges_meta = _load_edge_boundary_map(edges_pt)
    gap_smoothing_window = int(edges_meta.get("config", {}).get("gap_smoothing_window", 5))

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(str(ckpt), device, work_dir=str(work_dir), port=None)
        model = renderer.model
        model.eval()

        motion_bases = model.motion_bases
        gnn = getattr(motion_bases, "gnn", None)
        if gnn is None:
            raise RuntimeError(
                f"{ckpt} has no motion_bases.gnn -- it wasn't trained with "
                "--enable_graph_coupling, so there's no GNN edge structure to check."
            )
        variant = "relative" if hasattr(gnn, "edge_index_dir") else "absolute"
        num_clusters = motion_bases.num_clusters
        num_frames_total = motion_bases.num_frames
        resolved_edges = _resolve_edges(gnn)

        matched: list[tuple[int, int, dict[str, Any]]] = []
        unmatched: list[tuple[int, int]] = []
        for a, b in resolved_edges:
            rec = boundary_map.get((a, b))
            if rec is None:
                unmatched.append((a, b))
            else:
                matched.append((a, b, rec))

        print(
            f"[gnn_check2] variant={variant}  num_clusters={num_clusters}  "
            f"num_frames={num_frames_total}  gnn_edges={len(resolved_edges)}  "
            f"matched_to_edges_pt={len(matched)}  unmatched={len(unmatched)}"
        )
        if unmatched:
            print(f"[gnn_check2] WARNING: no boundary indices in {edges_pt} for edges: {unmatched}")
        if not matched:
            raise RuntimeError(
                f"None of the GNN's {len(resolved_edges)} edges were found in {edges_pt}. "
                "Pass --edges-pt to point at the edges.pt this checkpoint's edge_index was built from."
            )

        frame_ids_t = torch.arange(0, num_frames_total, args.frame_stride, device=device, dtype=torch.long)
        if args.max_frames is not None:
            frame_ids_t = frame_ids_t[: args.max_frames]
        num_frames_eval = int(frame_ids_t.numel())
        print(f"[gnn_check2] evaluating {num_frames_eval} frames (stride={args.frame_stride})")

        # --- correction (omega, delta_t) for every cluster, same frames ---
        all_cluster_ids = torch.arange(num_clusters, device=device)
        motion_bases.compute_transforms_coarse(frame_ids_t, all_cluster_ids)  # populates last_correction
        omega = motion_bases.last_correction["omega"]  # (C, T, 3)
        delta_t = motion_bases.last_correction["delta_t"]  # (C, T, 3)

        a_ids = torch.tensor([a for a, _, _ in matched], device=device, dtype=torch.long)
        b_ids = torch.tensor([b for _, b, _ in matched], device=device, dtype=torch.long)

        R_all = so3_exp_map(omega)  # (C, T, 3, 3)
        R_a = R_all[a_ids]  # (E, T, 3, 3)
        R_b = R_all[b_ids]
        R_rel = torch.einsum("etij,etjk->etik", R_a.transpose(-1, -2), R_b)
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # (E, T)
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        rot_diff_deg = (torch.arccos(cos_theta) * (180.0 / np.pi)).cpu().numpy()  # (E, T)
        trans_diff = (delta_t[a_ids] - delta_t[b_ids]).norm(dim=-1).cpu().numpy()  # (E, T)

        # --- boundary gap, with GNN vs GNN bypassed, for every matched edge ---
        # edges.pt may have been built with position_source=raw_tracks. In that
        # case its boundary_global_indices_* are indices into the temporary raw
        # track array, not model.fg Gaussian ids. Even model-based indices are
        # not stable after training-time pruning/densification. Keep the graph
        # topology, reference frames, thresholds, and boundary-selection config
        # from edges.pt, but reselect boundary identities on this checkpoint.
        coefs_all = model.fg.get_coefs()
        cluster_ids_pts_all = model.fg.get_cluster_ids()
        means_all = model.fg.params["means"]
        means_all_h = torch.cat(
            [means_all, torch.ones(means_all.shape[0], 1, device=device)], dim=-1
        )

        requested_ref_frames = [int(rec.get("frame_t_star", 0)) for _, _, rec in matched]
        clamped_ref_frames = [min(max(frame, 0), num_frames_total - 1) for frame in requested_ref_frames]
        if requested_ref_frames != clamped_ref_frames:
            print(
                "[gnn_check2] WARNING: some edges.pt reference frames are outside the "
                "checkpoint frame range; clamping them to the nearest valid frame"
            )
        ref_frame_ids = sorted(set(clamped_ref_frames))
        ref_frame_ids_t = torch.tensor(ref_frame_ids, device=device, dtype=torch.long)
        ref_frame_to_local = {frame: i for i, frame in enumerate(ref_frame_ids)}

        transfms_ref_without = ScalableMotionBases.compute_transforms(
            motion_bases, ref_frame_ids_t, coefs_all, cluster_ids_pts_all
        )
        positions_ref_without = torch.einsum(
            "ntij,nj->nti", transfms_ref_without, means_all_h
        ).cpu().numpy()
        del transfms_ref_without

        cluster_ids_pts_all_np = cluster_ids_pts_all.cpu().numpy()
        global_indices_by_cluster = {
            cluster_id: np.flatnonzero(cluster_ids_pts_all_np == cluster_id).astype(np.int64)
            for cluster_id in range(num_clusters)
        }
        boundary_cfg = edges_meta.get("config", {})
        boundary_fraction = float(boundary_cfg.get("boundary_fraction", 0.10))
        boundary_min = int(boundary_cfg.get("boundary_min_gaussians", 20))
        boundary_max = int(boundary_cfg.get("boundary_max_gaussians", 150))
        boundary_max_distance_cfg = boundary_cfg.get("boundary_max_distance")
        boundary_max_distance_multiplier = float(
            boundary_cfg.get("boundary_max_distance_multiplier", 1.0)
        )

        current_boundaries: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
        for (a, b, rec), ref_frame in zip(matched, clamped_ref_frames):
            global_a = global_indices_by_cluster[a]
            global_b = global_indices_by_cluster[b]
            if not len(global_a) or not len(global_b):
                raise RuntimeError(
                    f"GNN edge {a}-{b} refers to a cluster with no foreground Gaussians "
                    "in the current checkpoint."
                )
            boundary_max_distance = (
                float(boundary_max_distance_cfg)
                if boundary_max_distance_cfg is not None
                else float(rec["contact_distance"]) * boundary_max_distance_multiplier
            )
            boundary_a, boundary_b = select_boundary_at_frame(
                global_a,
                global_b,
                positions_ref_without,
                ref_frame_to_local[ref_frame],
                boundary_fraction,
                boundary_min,
                boundary_max,
                boundary_max_distance,
            )
            current_boundaries[(a, b)] = (boundary_a, boundary_b, ref_frame)

        needed_indices = torch.unique(
            torch.cat(
                [torch.from_numpy(current_boundaries[(a, b)][0]) for a, b, _ in matched]
                + [torch.from_numpy(current_boundaries[(a, b)][1]) for a, b, _ in matched]
            )
        ).to(device)
        needed_indices_np = needed_indices.cpu().numpy()

        coefs = coefs_all[needed_indices]
        cluster_ids_pts = cluster_ids_pts_all[needed_indices]
        means = means_all[needed_indices]
        means_h = torch.cat([means, torch.ones(means.shape[0], 1, device=device)], dim=-1)

        transfms_with = motion_bases.compute_transforms(frame_ids_t, coefs, cluster_ids_pts)  # (N, T, 3, 4)
        transfms_without = ScalableMotionBases.compute_transforms(
            motion_bases, frame_ids_t, coefs, cluster_ids_pts
        )  # (N, T, 3, 4); bypasses the GNN override entirely

        positions_with = torch.einsum("ntij,nj->nti", transfms_with, means_h).cpu().numpy()  # (N, T, 3)
        positions_without = torch.einsum("ntij,nj->nti", transfms_without, means_h).cpu().numpy()

    frame_ids_np = frame_ids_t.cpu().numpy()

    per_edge: list[dict[str, Any]] = []
    per_edge_frame_rows: list[dict[str, Any]] = []
    for edge_idx, (a, b, rec) in enumerate(matched):
        boundary_a, boundary_b, boundary_ref_frame = current_boundaries[(a, b)]
        boundary_a_local = np.searchsorted(needed_indices_np, boundary_a)
        boundary_b_local = np.searchsorted(needed_indices_np, boundary_b)

        gap_t_with, _ = compute_all_frames_gap(boundary_a_local, boundary_b_local, positions_with)
        gap_t_without, _ = compute_all_frames_gap(boundary_a_local, boundary_b_local, positions_without)

        threshold = float(rec["threshold"])
        confidence_ones = np.ones(num_frames_eval, dtype=np.float64)
        gap_summary_with = smooth_and_summarize_gap(gap_t_with, confidence_ones, gap_smoothing_window)
        gap_summary_without = smooth_and_summarize_gap(gap_t_without, confidence_ones, gap_smoothing_window)

        trans_diff_t = trans_diff[edge_idx]
        rot_diff_deg_t = rot_diff_deg[edge_idx]
        gap_reduction_t = gap_t_without - gap_t_with

        entry = {
            "cluster_a": a,
            "cluster_b": b,
            "num_boundary_a": int(len(boundary_a)),
            "num_boundary_b": int(len(boundary_b)),
            "boundary_reference_frame": boundary_ref_frame,
            "contact_distance": float(rec["contact_distance"]),
            "threshold": threshold,
            "gap_with_mean": float(gap_t_with.mean()),
            "gap_with_max": float(gap_t_with.max()),
            "gap_without_mean": float(gap_t_without.mean()),
            "gap_without_max": float(gap_t_without.max()),
            "gap_summary_with": gap_summary_with,
            "gap_summary_without": gap_summary_without,
            "kept_with_gnn": gap_summary_with < threshold,
            "kept_without_gnn": gap_summary_without < threshold,
            "gap_reduction_mean": float(gap_reduction_t.mean()),
            "gap_reduction_min": float(gap_reduction_t.min()),
            "gap_reduction_max": float(gap_reduction_t.max()),
            "trans_diff_mean": float(trans_diff_t.mean()),
            "trans_diff_max": float(trans_diff_t.max()),
            "rot_diff_deg_mean": float(rot_diff_deg_t.mean()),
            "rot_diff_deg_max": float(rot_diff_deg_t.max()),
            "gap_t_with": gap_t_with,
            "gap_t_without": gap_t_without,
            "trans_diff_t": trans_diff_t,
            "rot_diff_t": rot_diff_deg_t,
        }
        per_edge.append(entry)

        for t_idx, frame_id in enumerate(frame_ids_np):
            per_edge_frame_rows.append(
                {
                    "cluster_a": a,
                    "cluster_b": b,
                    "frame": int(frame_id),
                    "gap_with": float(gap_t_with[t_idx]),
                    "gap_without": float(gap_t_without[t_idx]),
                    "gap_reduction": float(gap_reduction_t[t_idx]),
                    "trans_diff": float(trans_diff_t[t_idx]),
                    "rot_diff_deg": float(rot_diff_deg_t[t_idx]),
                }
            )

    rescued = [r for r in per_edge if (not r["kept_without_gnn"]) and r["kept_with_gnn"]]
    broken = [r for r in per_edge if r["kept_without_gnn"] and (not r["kept_with_gnn"])]

    top_by_gap_help = sorted(per_edge, key=lambda r: r["gap_reduction_mean"], reverse=True)[: args.top_k]
    top_by_gap_hurt = sorted(per_edge, key=lambda r: r["gap_reduction_mean"])[: args.top_k]
    top_by_trans_diff = sorted(per_edge, key=lambda r: r["trans_diff_mean"], reverse=True)[: args.top_k]
    top_by_rot_diff = sorted(per_edge, key=lambda r: r["rot_diff_deg_mean"], reverse=True)[: args.top_k]

    top_gap_pairs = sorted({(r["cluster_a"], r["cluster_b"]) for r in top_by_gap_help + top_by_gap_hurt})
    top_relmotion_pairs = sorted({(r["cluster_a"], r["cluster_b"]) for r in top_by_trans_diff + top_by_rot_diff})

    overall = {
        "gap_with": _stats(np.concatenate([r["gap_t_with"] for r in per_edge])),
        "gap_without": _stats(np.concatenate([r["gap_t_without"] for r in per_edge])),
        "gap_reduction": _stats(np.concatenate([r["gap_t_without"] - r["gap_t_with"] for r in per_edge])),
        "trans_diff": _stats(np.concatenate([r["trans_diff_t"] for r in per_edge])),
        "rot_diff_deg": _stats(np.concatenate([r["rot_diff_t"] for r in per_edge])),
        "num_edges_rescued_by_gnn": len(rescued),  # would exceed keep threshold without GNN, doesn't with
        "num_edges_broken_by_gnn": len(broken),  # under keep threshold without GNN, exceeds it with
    }

    def _strip(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if not k.startswith("gap_t_") and k not in ("trans_diff_t", "rot_diff_t")}

    report = {
        "checkpoint": str(ckpt),
        "edges_pt": str(edges_pt),
        "variant": variant,
        "num_clusters": num_clusters,
        "num_frames_total": num_frames_total,
        "num_frames_evaluated": num_frames_eval,
        "frame_stride": args.frame_stride,
        "num_gnn_edges": len(resolved_edges),
        "num_matched_edges": len(matched),
        "unmatched_gnn_edges": [list(e) for e in unmatched],
        "gap_smoothing_window": gap_smoothing_window,
        "overall": overall,
        "edges_rescued_by_gnn": [_strip(r) for r in rescued],
        "edges_broken_by_gnn": [_strip(r) for r in broken],
        "top_edges_by_gap_improvement": [_strip(r) for r in top_by_gap_help],
        "top_edges_by_gap_worsened": [_strip(r) for r in top_by_gap_hurt],
        "top_edges_by_translation_relative_motion": [_strip(r) for r in top_by_trans_diff],
        "top_edges_by_rotation_relative_motion": [_strip(r) for r in top_by_rot_diff],
        "per_edge": [_strip(r) for r in per_edge],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    with (output_dir / "per_edge.csv").open("w", newline="") as f:
        rows = [_strip(r) for r in per_edge]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "per_edge_frame.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_edge_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_edge_frame_rows)

    def _pair_labels(rows: list[dict[str, Any]]) -> list[str]:
        return [f"{r['cluster_a']}-{r['cluster_b']}" for r in rows]

    print(f"[gnn_check2] wrote report.json / per_edge.csv / per_edge_frame.csv to {output_dir}")
    print(
        f"[gnn_check2] mean gap reduction = {overall['gap_reduction']['mean']:.4g} world units "
        f"(positive = GNN shrinks the boundary gap)"
    )
    print(f"[gnn_check2] edges rescued by GNN (would exceed keep threshold without it): {_pair_labels(rescued)}")
    if broken:
        print(
            f"[gnn_check2] edges BROKEN by GNN (were under keep threshold without it, aren't with it): "
            f"{_pair_labels(broken)}"
        )
    print(f"[gnn_check2] top edges by translation relative motion: {_pair_labels(top_by_trans_diff)}")
    print(f"[gnn_check2] top edges by rotation relative motion: {_pair_labels(top_by_rot_diff)}")

    if not args.no_plots:
        reference_threshold = float(per_edge[0]["threshold"]) if per_edge else 0.0
        _plot_gap_comparison(frame_ids_np, per_edge, top_gap_pairs, reference_threshold, output_dir / "gap_comparison_over_time.png")
        _plot_relative_motion(frame_ids_np, per_edge, top_relmotion_pairs, output_dir / "relative_motion_over_time.png")
        _plot_edge_ranking(per_edge, args.top_k, output_dir / "edge_ranking.png")
        _plot_relative_vs_gap_scatter(per_edge, output_dir / "edge_relative_vs_gap_scatter.png")
        print(f"[gnn_check2] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()

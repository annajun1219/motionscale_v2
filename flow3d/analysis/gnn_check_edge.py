#!/usr/bin/env python3
"""
flow3d/analysis/gnn_check_edge.py

Quantify + visualize what the edge-boundary correction GNN
(flow3d/graph_relative_edge.py's EdgeBoundaryGraphCorrectedScalableMotionBases)
actually does on a trained checkpoint -- the per-edge sibling of
flow3d/analysis/gnn_check.py (which is for the per-cluster omega/delta_t
correction and doesn't understand this variant's API at all: it reads
motion_bases.last_correction["omega"/"delta_t"], which this class never sets).

Answers two questions directly, without touching GT:
1. Did the GNN actually fire? (correction magnitude m_e(t) per edge, over time)
2. What did it do to the geometry? (per-edge boundary gap before vs after the
   correction, using the exact same falloff-weighted-mean-position math as the
   training loss -- flow3d/graph_relative_edge.py's
   compute_boundary_gap_distances -- plus a 3D before/after picture of the
   actual boundary-adjacent Gaussians for the most active edges.)

Outputs (under <output-dir>, default <work-dir>/analysis/gnn_check_edge)
--------------------------------------------------------------------------
    report.json                 -- summary stats + per-edge ranking
    per_edge.csv                 -- per-edge magnitude/gap stats
    per_frame.csv                 -- per-frame aggregate magnitude/gap stats
    magnitude_over_time.png      -- |m_e| per edge vs frame
    gap_before_after.png         -- per-edge bar chart: canonical / before / after gap
    boundary_3d_top_edges.png    -- 3D before(hollow)->after(solid) scatter with
                                     displacement arrows, for the most active edges,
                                     each at its own peak-correction frame

Example
-------
    python flow3d/analysis/gnn_check_edge.py \\
        --work-dir outputs/davis/camel/2026_08_29_06_10_04__gnn_correction_edge
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.analysis.cluster_pairs import _set_axes_equal_3d
from flow3d.graph_relative_edge import (
    EdgeBoundaryGraphCorrectedScalableMotionBases,
    compute_boundary_gap_distances,
)
from flow3d.params import ScalableMotionBases
from flow3d.renderer import Renderer

BEFORE_COLOR = "#999999"
SIDE_COLORS = ("#1f77b4", "#ff7f0e")  # side a, side b


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
    parser.add_argument("--top-k", type=int, default=6, help="Edges highlighted in plots/report.")
    parser.add_argument(
        "--max-points-per-side", type=int, default=250,
        help="Max falloff points per side drawn in the 3D before/after plot (deterministic subsample).",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def _plot_magnitude_over_time(
    frame_ids: np.ndarray,
    magnitude: np.ndarray,  # (E, T)
    top_edges: list[int],
    edge_labels: dict[int, str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")

    for edge_id in range(magnitude.shape[0]):
        if edge_id in top_edges:
            continue
        ax.plot(frame_ids, magnitude[edge_id], color="gray", alpha=0.15, linewidth=0.7)
    for rank, edge_id in enumerate(top_edges):
        ax.plot(
            frame_ids, magnitude[edge_id],
            color=cmap(rank % cmap.N), linewidth=1.8, label=edge_labels[edge_id],
        )
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("frame")
    ax.set_ylabel("correction magnitude m_e  (world units, +closes the gap)")
    ax.set_title("Edge-boundary correction magnitude per edge over time")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_gap_before_after(
    per_edge: list[dict],
    top_k: int,
    output_path: Path,
) -> None:
    ranked = sorted(per_edge, key=lambda r: r["mean_gap_closed"], reverse=True)[:top_k]
    ranked = ranked[::-1]  # largest at top of barh
    labels = [r["label"] for r in ranked]
    canonical = [r["canonical_distance"] for r in ranked]
    before = [r["mean_dist_before"] for r in ranked]
    after = [r["mean_dist_after"] for r in ranked]

    y = np.arange(len(ranked))
    h = 0.25
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(ranked) + 1.0)))
    ax.barh(y - h, canonical, height=h, color="#2ca02c", label="canonical (rest-pose) distance")
    ax.barh(y, before, height=h, color=BEFORE_COLOR, label="actual gap, correction OFF")
    ax.barh(y + h, after, height=h, color="#d62728", label="actual gap, correction ON")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean boundary-Gaussian gap over evaluated frames (world units)")
    ax.set_title("Per-edge boundary gap: canonical vs. actual (correction off/on)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_boundary_3d(
    motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases,
    top_edges: list[int],
    peak_frame_idx: dict[int, int],  # edge_id -> index into frame_ids_t of its peak-|magnitude| frame
    frame_ids_t: torch.Tensor,
    edge_labels: dict[int, str],
    max_points_per_side: int,
    output_path: Path,
) -> None:
    mb = motion_bases
    device = mb.falloff_global_idx.device

    n_cols = min(3, max(1, len(top_edges)))
    n_rows = int(np.ceil(len(top_edges) / n_cols))
    fig = plt.figure(figsize=(5.5 * n_cols, 5.0 * n_rows))

    for panel_idx, edge_id in enumerate(top_edges, start=1):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx, projection="3d")

        row_mask = mb.falloff_edge_id == edge_id
        rows = row_mask.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            ax.set_title(f"{edge_labels[edge_id]} (no falloff rows)")
            continue

        t_idx = peak_frame_idx[edge_id]
        t = frame_ids_t[t_idx : t_idx + 1]  # (1,)

        sign = mb.falloff_sign[rows]
        weight = mb.falloff_weight[rows]
        # Same snapshot _falloff_side_means/_edge_features_and_direction use --
        # no separate canonical_means/coefs_all needed (module docstring).
        row_coefs = mb.falloff_coefs[rows]
        row_cluster_id = torch.where(sign > 0, mb.edge_cluster_a[edge_id], mb.edge_cluster_b[edge_id])

        base_transforms = ScalableMotionBases.compute_transforms(
            mb, t, row_coefs, row_cluster_id
        ).detach()  # (R, 1, 3, 4)
        homog = torch.cat(
            [mb.falloff_canonical_mean[rows], torch.ones(rows.shape[0], 1, device=device)], dim=-1
        )
        before = torch.einsum("rij,rj->ri", base_transforms[:, 0], homog)  # (R, 3)

        magnitude, direction, _, _ = mb._edge_features_and_direction(t, detach_base=True)
        m = magnitude[edge_id, 0]
        d = direction[edge_id, 0]
        displacement = 0.5 * sign[:, None] * m * d[None, :] * weight[:, None]  # (R, 3)
        after = before + displacement

        before_np = before.cpu().numpy()
        after_np = after.cpu().numpy()
        sign_np = sign.cpu().numpy()

        weight_np = weight.cpu().numpy()
        for side_val, color in ((1.0, SIDE_COLORS[0]), (-1.0, SIDE_COLORS[1])):
            side_rows = np.nonzero(sign_np == side_val)[0]
            if side_rows.size == 0:
                continue
            if side_rows.size > max_points_per_side:
                # Prefer the highest-falloff-weight (closest-to-the-seam) points --
                # the ones the correction actually pulls the hardest -- rather than
                # a uniform sample that would dilute the before->after motion with
                # far, barely-moved points and blow out the axis scale.
                order = np.argsort(-weight_np[side_rows])
                side_rows = side_rows[order[:max_points_per_side]]

            ax.scatter(
                before_np[side_rows, 0], before_np[side_rows, 1], before_np[side_rows, 2],
                s=8, facecolors="none", edgecolors=color, alpha=0.5, linewidths=0.6,
                label=("side a, before" if side_val > 0 else "side b, before"),
            )
            ax.scatter(
                after_np[side_rows, 0], after_np[side_rows, 1], after_np[side_rows, 2],
                s=10, color=color, alpha=0.9,
                label=("side a, after" if side_val > 0 else "side b, after"),
            )
            for r in side_rows:
                ax.plot(
                    [before_np[r, 0], after_np[r, 0]],
                    [before_np[r, 1], after_np[r, 1]],
                    [before_np[r, 2], after_np[r, 2]],
                    color=color, alpha=0.35, linewidth=0.7,
                )

        all_pts = np.concatenate([before_np, after_np], axis=0)
        _set_axes_equal_3d(ax, all_pts)
        ax.view_init(elev=20, azim=35)
        ax.set_title(
            f"{edge_labels[edge_id]}  (frame {int(frame_ids_t[t_idx].item())}, "
            f"m={m.item():+.4g})",
            fontsize=10,
        )
        if panel_idx == 1:
            ax.legend(loc="upper left", fontsize=6)

    fig.suptitle(
        "Boundary Gaussians before (hollow) -> after (solid) the edge correction, "
        "at each edge's peak-correction frame",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    ckpt = (args.ckpt if args.ckpt is not None else work_dir / "checkpoints" / "last.ckpt").expanduser().resolve()
    output_dir = (
        args.output_dir if args.output_dir is not None else work_dir / "analysis" / "gnn_check_edge"
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
        renderer = Renderer.init_from_checkpoint(str(ckpt), device, work_dir=str(output_dir), port=None)
        model = renderer.model
        model.eval()

        motion_bases = model.motion_bases
        if not isinstance(motion_bases, EdgeBoundaryGraphCorrectedScalableMotionBases):
            raise RuntimeError(
                f"{ckpt} wasn't trained with --gnn_variant=relative_edge_boundary "
                f"(motion_bases is {type(motion_bases).__name__}) -- use "
                "flow3d/analysis/gnn_check.py for the per-cluster omega/delta_t variant instead."
            )

        num_edges = motion_bases.num_edges
        num_frames_total = motion_bases.num_frames
        edge_cluster_a = motion_bases.edge_cluster_a.cpu().numpy()
        edge_cluster_b = motion_bases.edge_cluster_b.cpu().numpy()
        canonical_distance = motion_bases.canonical_distance.cpu().numpy()
        max_displacement = motion_bases.max_displacement.cpu().numpy()  # (E,)
        edge_labels = {
            e: f"edge {e} ({int(edge_cluster_a[e])}-{int(edge_cluster_b[e])})" for e in range(num_edges)
        }
        print(f"[gnn_check_edge] num_edges={num_edges}  num_frames={num_frames_total}")

        frame_ids_t = torch.arange(0, num_frames_total, args.frame_stride, device=device, dtype=torch.long)
        if args.max_frames is not None:
            frame_ids_t = frame_ids_t[: args.max_frames]
        num_frames_eval = int(frame_ids_t.numel())
        print(f"[gnn_check_edge] evaluating {num_frames_eval} frames (stride={args.frame_stride})")

        result = compute_boundary_gap_distances(motion_bases, frame_ids_t)
        if result is None:
            raise RuntimeError(
                f"{ckpt}'s motion_bases has zero falloff rows for every edge -- "
                "nothing for the correction to touch (falloff_radius too small, or "
                "refresh_boundary_falloff never ran after a density-control step that "
                "emptied every edge's neighborhood)."
            )

        magnitude = result["magnitude"].cpu().numpy()  # (E, T)
        alpha = result["alpha"].cpu().numpy()  # (E, T)
        dist_before = result["dist_before"].cpu().numpy()  # (E, T)
        dist_after = result["dist_after"].cpu().numpy()  # (E, T)
        has_both_sides = result["has_both_sides"].cpu().numpy()  # (E,)

        num_side_a = np.array(
            [int((motion_bases.falloff_edge_id[motion_bases.falloff_sign > 0] == e).sum().item()) for e in range(num_edges)]
        )
        num_side_b = np.array(
            [int((motion_bases.falloff_edge_id[motion_bases.falloff_sign < 0] == e).sum().item()) for e in range(num_edges)]
        )

    frame_ids_np = frame_ids_t.cpu().numpy()
    abs_magnitude = np.abs(magnitude)

    # --- per-edge aggregation ---
    per_edge: list[dict] = []
    for e in range(num_edges):
        valid = has_both_sides[e]
        gap_closed = dist_before[e] - dist_after[e] if valid else np.zeros(num_frames_eval)
        entry = {
            "edge_id": e,
            "label": edge_labels[e],
            "cluster_a": int(edge_cluster_a[e]),
            "cluster_b": int(edge_cluster_b[e]),
            "canonical_distance": float(canonical_distance[e]),
            "max_displacement": float(max_displacement[e]),
            "num_points_side_a": int(num_side_a[e]),
            "num_points_side_b": int(num_side_b[e]),
            "valid_all_frames": bool(valid) if isinstance(valid, (bool, np.bool_)) else bool(np.all(valid)),
            "mean_abs_magnitude": float(abs_magnitude[e].mean()),
            "max_abs_magnitude": float(abs_magnitude[e].max()),
            "mean_alpha": float(alpha[e].mean()),
            "max_alpha": float(alpha[e].max()),
            "mean_dist_before": float(dist_before[e].mean()) if valid else float("nan"),
            "mean_dist_after": float(dist_after[e].mean()) if valid else float("nan"),
            "mean_gap_closed": float(gap_closed.mean()) if valid else 0.0,
            "mean_gap_closed_pct": (
                float(100.0 * gap_closed.mean() / dist_before[e].mean())
                if valid and dist_before[e].mean() > 1e-8
                else float("nan")
            ),
        }
        per_edge.append(entry)

    top_edges = [
        r["edge_id"] for r in sorted(per_edge, key=lambda r: r["max_abs_magnitude"], reverse=True)[: args.top_k]
    ]
    peak_frame_idx = {e: int(np.argmax(abs_magnitude[e])) for e in top_edges}

    # --- per-frame aggregation ---
    per_frame_rows = []
    for t_idx, frame_id in enumerate(frame_ids_np):
        valid_mask = has_both_sides
        per_frame_rows.append(
            {
                "frame": int(frame_id),
                "mean_abs_magnitude": float(abs_magnitude[:, t_idx].mean()),
                "max_abs_magnitude": float(abs_magnitude[:, t_idx].max()),
                "mean_dist_before": float(dist_before[valid_mask, t_idx].mean()) if valid_mask.any() else float("nan"),
                "mean_dist_after": float(dist_after[valid_mask, t_idx].mean()) if valid_mask.any() else float("nan"),
            }
        )

    overall = {
        "magnitude_abs": _stats(abs_magnitude),
        "gap_before": _stats(dist_before[has_both_sides]) if has_both_sides.any() else None,
        "gap_after": _stats(dist_after[has_both_sides]) if has_both_sides.any() else None,
        "num_edges_with_zero_correction_ever": int(np.sum(abs_magnitude.max(axis=1) < 1e-6)),
        "num_edges_missing_one_side_some_frame": int(num_edges - int(has_both_sides.sum())),
    }

    report = {
        "checkpoint": str(ckpt),
        "num_edges": num_edges,
        "num_frames_total": num_frames_total,
        "num_frames_evaluated": num_frames_eval,
        "frame_stride": args.frame_stride,
        "overall": overall,
        "top_edges_by_max_magnitude": [edge_labels[e] for e in top_edges],
        "per_edge": per_edge,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    with (output_dir / "per_frame.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    with (output_dir / "per_edge.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_edge[0].keys()))
        writer.writeheader()
        writer.writerows(per_edge)

    print(f"[gnn_check_edge] wrote report.json / per_frame.csv / per_edge.csv to {output_dir}")
    print(
        f"[gnn_check_edge] |magnitude|: mean={overall['magnitude_abs']['mean']:.4g} "
        f"max={overall['magnitude_abs']['max']:.4g} world units "
        f"({overall['num_edges_with_zero_correction_ever']}/{num_edges} edges never moved off ~0)"
    )
    if overall["gap_before"] is not None:
        print(
            f"[gnn_check_edge] boundary gap: before mean={overall['gap_before']['mean']:.4g}, "
            f"after mean={overall['gap_after']['mean']:.4g} world units"
        )
    print(f"[gnn_check_edge] top edges by max |magnitude|: {[edge_labels[e] for e in top_edges]}")

    if not args.no_plots:
        _plot_magnitude_over_time(
            frame_ids_np, magnitude, top_edges, edge_labels, output_dir / "magnitude_over_time.png"
        )
        _plot_gap_before_after(per_edge, args.top_k, output_dir / "gap_before_after.png")
        with torch.no_grad():
            _plot_boundary_3d(
                motion_bases,
                top_edges,
                peak_frame_idx,
                frame_ids_t,
                edge_labels,
                args.max_points_per_side,
                output_dir / "boundary_3d_top_edges.png",
            )
        print(f"[gnn_check_edge] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()

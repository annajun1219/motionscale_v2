#!/usr/bin/env python3
"""
Pure raw-mesh-contact cluster graph builder, a stricter/simpler variant of
flow3d/analysis/build_cluster_graph_mesh.py (read but NOT modified by this
module -- see that file's own docstring for the full mesh-reconstruction
rationale, which this module reuses unchanged).

Motivation
----------
build_cluster_graph_mesh.py's final edge decision is a patch-LOCALIZED score
(score_frame_against_patches): cross-cluster mesh edges are only counted if
both endpoints already lie inside a boundary Gaussian patch built from
earlier multi-frame seed voting, and the edge survives if that score's
across-frame persistence beats tau_p. It also produces a contact core +
geodesically-dilated boundary patch (a set of real Gaussians near the
touching surface) for downstream GNN correction-range use.

This module drops ALL of that and answers only one question -- "are these
two clusters' surfaces touching in this frame's mesh, and does that persist
across enough frames" -- directly from the mesh itself, with no boundary
Gaussians, no patches, no score, no tau_s/tau_p anywhere:

    For a candidate pair, group this frame's cross-cluster mesh edges into
    components where two edges are connected iff they share a mesh vertex
    (pure graph adjacency over the mesh -- not proximity, not a patch
    membership test). Take the largest such component's edge count
    (largest_component_edge_count). The frame is "connected" for that pair
    iff that count is >= --min-seam-edges. A pair's final edge is kept iff
    at least --min-connected-frames of the sampled frames were judged
    connected this way.

No Gaussian is ever singled out as a "seed", no contact core or boundary
patch is built, and nothing is produced for GNN correction-range use --
edges.pt here carries pure cluster-graph topology (cluster_a, cluster_b,
kept/cut, connected-frame bookkeeping), nothing else.

Algorithm
---------
1. Cluster ids: the checkpoint's fixed model.fg.get_cluster_ids(), filtered
   by --min-cluster-size, via build_cluster_graph.py's
   load_model_and_clusters (reused directly, same as
   build_cluster_graph_mesh.py).
2. Single pass: for each sampled frame t in range(0, num_frames,
   --frame-interval), render this frame's own visible-surface mesh
   (build_cluster_graph_mesh.py's render_frame_mesh, reused unchanged --
   alpha/depth-consistent grid triangulation, no Poisson, no implicit-
   function bridging). For every cross-cluster pair present this frame:
   group cross edges by shared mesh vertex into connected components,
   record largest_component_edge_count, and mark the frame "connected" for
   that pair iff it's >= --min-seam-edges.
3. Final edge decision (the ONLY thing that decides kept/cut):
   kept(a, b) = (# connected frames) >= --min-connected-frames. Default 15,
   i.e. the 15-of-20 ratio this module was designed around; scale
   proportionally if your sampled frame count differs a lot from 20.

Output
------
Writes, under <output-dir> (default: <work-dir>/analysis/cluster_graph_mesh_only):

    edges.pt                 -- {"edge_index", "edges_kept", "edges_cut",
                                 "cluster_ids", "meta"}, same top-level
                                 shape as build_cluster_graph.py /
                                 build_cluster_graph_mesh.py's edges.pt, so
                                 anything that only needs cluster-graph
                                 topology (e.g.
                                 flow3d/graph_coupling.py's
                                 build_edge_index_from_edges_pt) still
                                 works unchanged. Each edges_kept/edges_cut
                                 entry: cluster_a, cluster_b, kept, reason,
                                 persistence (= num_connected_frames /
                                 num_sampled_frames), num_connected_frames,
                                 num_sampled_frames, connected_frame_indices,
                                 max/mean_largest_component_edge_count. NO
                                 boundary_global_indices_a/b or any other
                                 correction-range field -- this module does
                                 not build boundary Gaussians, so it is not
                                 usable by flow3d/graph_relative_edge.py's
                                 load_edge_boundary_sets (that needs a
                                 correction-range-producing builder, e.g.
                                 build_cluster_graph_mesh.py).
    edges_kept.csv / edges_cut.csv
                              -- one row per candidate pair: final kept/cut
                                 + reason, connected-frame count,
                                 max/mean largest_component_edge_count.
    frame_pair_states.csv    -- one row per (candidate pair, sampled frame):
                                 largest_component_edge_count and whether
                                 that frame was connected.
    report.json               -- {"meta": {...run config...}, "pairs": [...
                                 per-pair final temporal judgment...]}.
    summary.txt
    meshes/frame_XXXX.ply, frame_XXXX.png
    colored_meshes/frame_XXXX.ply, frame_XXXX.png
    figures/connectivity_heatmap.png
    figures/final_edge_graph.png
    figures/graph_edges_2d.png / graph_edges_2d.mp4  -- unless
                                 --no-visualization / --no-video.

Example
-------
    python flow3d/analysis/build_cluster_graph_mesh_only.py \\
        --work-dir outputs/davis/spaceout/2026_08_11_12_41_36__warmup_initframe120_60epoch
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from flow3d.analysis.cluster_pairs import _get_camera_w2cs, write_csv
from flow3d.analysis.build_cluster_graph import load_model_and_clusters
from flow3d.analysis.build_cluster_graph_mesh import (
    build_cluster_colormap,
    load_old_graph_edge_pairs,
    render_2d_overlay,
    render_2d_overlay_video,
    render_final_edge_graph,
    render_frame_mesh,
    save_frame_mesh_outputs,
    save_persistence_heatmap,
)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MeshOnlyConfig:
    tau_o: float
    mask_alpha_threshold: float
    depth_jump_ratio: float
    frame_interval: int
    min_cluster_size: int
    min_seam_edges: int
    min_connected_frames: int
    # connected-frame-fraction equivalent of min_connected_frames, i.e.
    # min_connected_frames / num_sampled_frames -- computed once
    # num_sampled_frames is known, purely so this module can reuse
    # build_cluster_graph_mesh.py's render_final_edge_graph/
    # save_persistence_heatmap/render_2d_overlay(_video) unchanged (they
    # take a plain tau_p-shaped float for linewidth/heatmap scaling). NEVER
    # used for the edge keep/cut decision itself -- that decision applies
    # min_connected_frames to a raw connected-frame COUNT (see module
    # docstring step 3).
    tau_p: float


@dataclass
class MeshOnlyClusterPairEdge:
    cluster_a: int
    cluster_b: int
    kept: bool
    reason: str
    persistence: float  # num_connected_frames / num_sampled_frames
    num_connected_frames: int
    num_sampled_frames: int
    connected_frame_indices: list[int]
    max_largest_component_edge_count: int
    mean_largest_component_edge_count: float


@dataclass
class MeshOnlyClusterGraphResult:
    cluster_ids: list[int]
    edges: list[MeshOnlyClusterPairEdge]
    candidate_pair_count: int
    num_sampled_frames: int
    config: MeshOnlyConfig

    @property
    def kept_edges(self) -> list[MeshOnlyClusterPairEdge]:
        return [e for e in self.edges if e.kept]

    @property
    def cut_edges(self) -> list[MeshOnlyClusterPairEdge]:
        return [e for e in self.edges if not e.kept]

    def edge_index(self) -> np.ndarray:
        """Symmetric (2, 2*num_kept) edge index: both (a,b) and (b,a)."""
        kept = self.kept_edges
        if not kept:
            return np.zeros((2, 0), dtype=np.int64)
        a = np.asarray([e.cluster_a for e in kept], dtype=np.int64)
        b = np.asarray([e.cluster_b for e in kept], dtype=np.int64)
        src = np.concatenate([a, b])
        dst = np.concatenate([b, a])
        return np.stack([src, dst], axis=0)

    def neighbors(self) -> dict[int, list[int]]:
        adjacency: dict[int, list[int]] = {cid: [] for cid in self.cluster_ids}
        for e in self.kept_edges:
            adjacency.setdefault(e.cluster_a, []).append(e.cluster_b)
            adjacency.setdefault(e.cluster_b, []).append(e.cluster_a)
        return adjacency


@dataclass
class FramePairStat:
    largest_component_edge_count: int
    is_connected: bool


# ---------------------------------------------------------------------------
# Largest shared-vertex-connected cross-edge component
# ---------------------------------------------------------------------------


def largest_cross_component_edges(cross_edges: np.ndarray) -> int:
    """
    `cross_edges`: (E, 2) mesh-vertex-index pairs -- one candidate pair's
    cross-cluster mesh edges in a single frame. Two edges are grouped into
    the same component iff they share a mesh vertex (pure graph adjacency
    over the mesh itself, not spatial proximity). Returns the edge count of
    the largest such component (0 if `cross_edges` is empty).
    """
    if cross_edges.shape[0] == 0:
        return 0

    unique_vertices, inverse = np.unique(cross_edges.reshape(-1), return_inverse=True)
    local = inverse.reshape(-1, 2)
    n = unique_vertices.shape[0]
    ones = np.ones(local.shape[0], dtype=np.float64)
    graph = coo_matrix((ones, (local[:, 0], local[:, 1])), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)

    edge_labels = labels[local[:, 0]]  # == labels[local[:, 1]], connected by this very edge
    counts = np.bincount(edge_labels, minlength=labels.max() + 1 if labels.size else 0)
    return int(counts.max())


def collect_frame_pair_stats(
    vertex_cluster_ids: np.ndarray,
    mesh_edges: np.ndarray,
    valid_ids: list[int],
    min_seam_edges: int,
) -> dict[tuple[int, int], FramePairStat]:
    """
    For every cross-cluster pair present in this frame's mesh: this pair's
    largest_component_edge_count (see largest_cross_component_edges) and
    whether that makes the frame "connected" for this pair
    (>= min_seam_edges). Keyed (cid_lo, cid_hi) with cid_lo < cid_hi.
    """
    va = vertex_cluster_ids[mesh_edges[:, 0]]
    vb = vertex_cluster_ids[mesh_edges[:, 1]]
    cross_mask = va != vb
    stats: dict[tuple[int, int], FramePairStat] = {}
    if not bool(cross_mask.any()):
        return stats

    cross_edge_rows = np.nonzero(cross_mask)[0]
    cross_a_cid = va[cross_mask]
    cross_b_cid = vb[cross_mask]
    key_lo = np.minimum(cross_a_cid, cross_b_cid)
    key_hi = np.maximum(cross_a_cid, cross_b_cid)
    multiplier = int(max(valid_ids, default=0)) + 1
    combined_key = key_lo.astype(np.int64) * multiplier + key_hi.astype(np.int64)
    unique_keys, inverse = np.unique(combined_key, return_inverse=True)

    for k_index, key in enumerate(unique_keys.tolist()):
        cid_lo = key // multiplier
        cid_hi = key % multiplier
        rows_for_pair = cross_edge_rows[inverse == k_index]
        pair_cross_edges = mesh_edges[rows_for_pair]

        count = largest_cross_component_edges(pair_cross_edges)
        stats[(cid_lo, cid_hi)] = FramePairStat(
            largest_component_edge_count=count,
            is_connected=count >= min_seam_edges,
        )

    return stats


# ---------------------------------------------------------------------------
# edges.pt / CSV / report.json payloads
# ---------------------------------------------------------------------------


def _edge_to_dict(e: MeshOnlyClusterPairEdge) -> dict[str, Any]:
    return {
        "cluster_a": int(e.cluster_a),
        "cluster_b": int(e.cluster_b),
        "kept": bool(e.kept),
        "reason": e.reason,
        "persistence": float(e.persistence),
        "num_connected_frames": int(e.num_connected_frames),
        "num_sampled_frames": int(e.num_sampled_frames),
        "connected_frame_indices": list(e.connected_frame_indices),
        "max_largest_component_edge_count": int(e.max_largest_component_edge_count),
        "mean_largest_component_edge_count": float(e.mean_largest_component_edge_count),
    }


def _edge_to_csv_row(e: MeshOnlyClusterPairEdge) -> dict[str, Any]:
    row = _edge_to_dict(e)
    row["connected_frame_indices"] = " ".join(str(f) for f in e.connected_frame_indices)
    return row


def write_summary(
    path: Path,
    result: MeshOnlyClusterGraphResult,
    old_graph_path: Path,
    old_pairs: set[tuple[int, int]] | None,
    num_valid_clusters: int,
) -> None:
    cfg = result.config
    lines = [
        "Raw-mesh-contact cluster graph (edges only, no boundary Gaussians) -- summary",
        "=" * 78,
        f"valid clusters  : {num_valid_clusters}",
        f"sampled frames  : {result.num_sampled_frames}",
        f"frame_interval={cfg.frame_interval} tau_o={cfg.tau_o} "
        f"mask_alpha_threshold={cfg.mask_alpha_threshold} depth_jump_ratio={cfg.depth_jump_ratio} "
        f"min_cluster_size={cfg.min_cluster_size}",
        f"min_seam_edges={cfg.min_seam_edges} min_connected_frames={cfg.min_connected_frames}",
        f"candidate pairs : {result.candidate_pair_count}",
        f"kept edges      : {len(result.kept_edges)}",
        f"cut edges       : {len(result.cut_edges)}",
        "",
        "Kept edges (cluster_a-cluster_b  connected/sampled  max/mean largest_component_edge_count):",
    ]
    for e in sorted(result.kept_edges, key=lambda e: -e.num_connected_frames):
        lines.append(
            f"  {e.cluster_a:3d}-{e.cluster_b:<3d} "
            f"connected={e.num_connected_frames}/{e.num_sampled_frames} "
            f"max_seam={e.max_largest_component_edge_count} mean_seam={e.mean_largest_component_edge_count:.2f}"
        )
    lines.append("")
    lines.append("Cut candidate edges (reason, connected/sampled, max/mean seam):")
    for e in sorted(result.cut_edges, key=lambda e: -e.num_connected_frames):
        lines.append(
            f"  {e.cluster_a:3d}-{e.cluster_b:<3d} reason={e.reason} "
            f"connected={e.num_connected_frames}/{e.num_sampled_frames} "
            f"max_seam={e.max_largest_component_edge_count} mean_seam={e.mean_largest_component_edge_count:.2f}"
        )
    lines.append("")

    if old_pairs is None:
        lines.append(f"No comparison graph found at {old_graph_path} -- skipping comparison.")
    else:
        new_pairs = {(e.cluster_a, e.cluster_b) for e in result.kept_edges}
        common = sorted(new_pairs & old_pairs)
        added = sorted(new_pairs - old_pairs)
        removed = sorted(old_pairs - new_pairs)
        lines.append(f"Comparison with {old_graph_path}:")
        lines.append(f"  old kept edges         : {len(old_pairs)}")
        lines.append(f"  new kept edges         : {len(new_pairs)}")
        lines.append(f"  common                 : {len(common)} -> {common}")
        lines.append(f"  added (mesh-only)      : {len(added)} -> {added}")
        lines.append(f"  removed (old only)     : {len(removed)} -> {removed}")

    path.write_text("\n".join(str(l) for l in lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--old-graph-path",
        type=Path,
        default=None,
        help="An edges.pt (build_cluster_graph.py or build_cluster_graph_mesh.py) to compare "
        "against in summary.txt. Default: <work-dir>/analysis/cluster_graph_mesh/edges.pt "
        "(the patch-score-based mesh graph this module supersedes) if it exists.",
    )

    parser.add_argument("--min-cluster-size", type=int, default=20)

    parser.add_argument(
        "--tau-o", type=float, default=0.1,
        help="Opacity threshold: Gaussians with activated opacity <= this are excluded from "
        "the render used to build the mesh (same scale/convention as optim.cull_opacity_threshold).",
    )
    parser.add_argument(
        "--mask-alpha-threshold", type=float, default=0.5,
        help="A rendered pixel is treated as foreground (and back-projected into the mesh) "
        "only if its accumulated alpha exceeds this.",
    )
    parser.add_argument(
        "--depth-jump-ratio", type=float, default=0.05,
        help="A grid triangle is dropped if any pair of its 3 corner pixels' depths differs "
        "by more than this fraction of their own depth -- the guardrail that keeps a "
        "self-occlusion silhouette (e.g. an arm in front of the torso) from being meshed as "
        "one continuous surface.",
    )
    parser.add_argument(
        "--min-seam-edges", type=int, default=8,
        help="A candidate pair's frame is judged 'connected' (raw mesh contact) iff the "
        "largest shared-mesh-vertex-connected component among that frame's cross-cluster mesh "
        "edges has at least this many edges. This -- and ONLY this, across "
        "--min-connected-frames sampled frames -- decides the final edge; no boundary Gaussian "
        "or score is involved (see module docstring).",
    )
    parser.add_argument(
        "--min-connected-frames", type=int, default=15,
        help="Final edge-keep threshold: a candidate pair is kept iff at least this many "
        "SAMPLED frames were judged 'connected' (see --min-seam-edges). Default 15 assumes "
        "the common case of ~20 sampled frames (a 15/20 ratio); scale proportionally if your "
        "--frame-interval produces a very different sampled-frame count.",
    )

    parser.add_argument(
        "--frame-interval", type=int, default=1,
        help="Sample every Nth frame (0, N, 2N, ...) for mesh reconstruction.",
    )

    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--max-points-per-cluster", type=int, default=2000)
    parser.add_argument(
        "--frame-index-2d", type=int, default=0,
        help="Frame to render graph_edges_2d.png on (same convention as build_cluster_graph.py).",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip the 2D edge-overlay mp4 (the PNG is still saved)."
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-frame-stride", type=int, default=1,
        help="Render every Nth frame for the edge-overlay video (>1 speeds up rendering for long sequences).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.min_cluster_size < 1:
        raise ValueError("--min-cluster-size must be >= 1")
    if not 0.0 <= args.tau_o < 1.0:
        raise ValueError("--tau-o must be in [0, 1)")
    if not 0.0 < args.mask_alpha_threshold < 1.0:
        raise ValueError("--mask-alpha-threshold must be in (0, 1)")
    if args.depth_jump_ratio <= 0.0:
        raise ValueError("--depth-jump-ratio must be > 0")
    if args.min_seam_edges < 1:
        raise ValueError("--min-seam-edges must be >= 1")
    if args.min_connected_frames < 1:
        raise ValueError("--min-connected-frames must be >= 1")
    if args.frame_interval < 1:
        raise ValueError("--frame-interval must be >= 1")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be >= 1")
    if args.video_frame_stride < 1:
        raise ValueError("--video-frame-stride must be >= 1")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    work_dir = args.work_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "analysis" / "cluster_graph_mesh_only"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    colored_meshes_dir = output_dir / "colored_meshes"
    figures_dir = output_dir / "figures"

    edges_pt = output_dir / "edges.pt"
    if edges_pt.exists() and not args.overwrite:
        raise FileExistsError(f"{edges_pt} already exists. Pass --overwrite to replace it.")

    model, clusters, filtered_ids = load_model_and_clusters(
        work_dir=work_dir,
        ckpt=args.ckpt,
        device_name=args.device,
        min_cluster_size=args.min_cluster_size,
    )
    valid_ids = sorted(c.cluster_id for c in clusters)
    color_by_id = build_cluster_colormap(valid_ids)

    device = model.fg.params["means"].device
    w2cs = _get_camera_w2cs(model).to(device)
    intrinsics = model.Ks.to(device)
    principal_x = float(intrinsics[0, 0, 2].item())
    principal_y = float(intrinsics[0, 1, 2].item())
    image_size = (max(int(round(principal_x * 2.0)), 2), max(int(round(principal_y * 2.0)), 2))

    num_frames = model.num_frames
    sampled_frames = list(range(0, num_frames, args.frame_interval))
    num_sampled_frames = len(sampled_frames)
    print(f"Valid clusters  : {len(valid_ids)}")
    print(f"Frames          : {num_frames} (sampling {num_sampled_frames} at interval {args.frame_interval})")

    cfg = MeshOnlyConfig(
        tau_o=args.tau_o,
        mask_alpha_threshold=args.mask_alpha_threshold,
        depth_jump_ratio=args.depth_jump_ratio,
        frame_interval=args.frame_interval,
        min_cluster_size=args.min_cluster_size,
        min_seam_edges=args.min_seam_edges,
        min_connected_frames=args.min_connected_frames,
        tau_p=args.min_connected_frames / max(num_sampled_frames, 1),
    )

    # ---- Single pass: per-frame mesh -> raw cross-cluster connectivity ----
    pair_frame_counts: dict[tuple[int, int], dict[int, int]] = {}
    connected_frames_by_pair: dict[tuple[int, int], list[int]] = {}

    for i, frame_index in enumerate(sampled_frames, start=1):
        frame_data = render_frame_mesh(
            model, valid_ids, frame_index, w2cs[frame_index], intrinsics[frame_index], image_size, cfg
        )

        if not args.no_visualization:
            save_frame_mesh_outputs(
                frame_data.vertices, frame_data.triangles, frame_data.vertex_cluster_ids, color_by_id,
                frame_index, meshes_dir, colored_meshes_dir,
            )

        stats = collect_frame_pair_stats(
            frame_data.vertex_cluster_ids, frame_data.mesh_edges, valid_ids, cfg.min_seam_edges
        )
        num_connected_this_frame = 0
        for pair, stat in stats.items():
            pair_frame_counts.setdefault(pair, {})[frame_index] = stat.largest_component_edge_count
            if stat.is_connected:
                num_connected_this_frame += 1
                connected_frames_by_pair.setdefault(pair, []).append(frame_index)

        print(
            f"[{i:03d}/{num_sampled_frames:03d}] t={frame_index:04d} "
            f"verts={frame_data.vertices.shape[0]} cross_pairs={len(stats)} connected_pairs={num_connected_this_frame}"
        )

    candidate_pairs = sorted(pair_frame_counts.keys())
    print()
    print(f"Candidate pairs (>= 1 frame of raw cross-cluster mesh contact) : {len(candidate_pairs)}")

    # ---- Final edge decision: raw connected-frame count only ----
    frame_state_rows: list[dict[str, Any]] = []
    edges: list[MeshOnlyClusterPairEdge] = []
    for pair in candidate_pairs:
        cluster_a, cluster_b = pair
        counts = pair_frame_counts[pair]
        connected_set = set(connected_frames_by_pair.get(pair, []))
        connected = sorted(connected_set)
        num_connected = len(connected)
        kept = num_connected >= cfg.min_connected_frames
        reason = "kept_connected_frames" if kept else "cut_insufficient_connected_frames"
        count_values = list(counts.values())

        for frame_index in sampled_frames:
            frame_state_rows.append(
                {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "frame_index": frame_index,
                    "largest_component_edge_count": counts.get(frame_index, 0),
                    "is_connected": frame_index in connected_set,
                }
            )

        edges.append(
            MeshOnlyClusterPairEdge(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                kept=kept,
                reason=reason,
                persistence=num_connected / max(num_sampled_frames, 1),
                num_connected_frames=num_connected,
                num_sampled_frames=num_sampled_frames,
                connected_frame_indices=connected,
                max_largest_component_edge_count=max(count_values) if count_values else 0,
                mean_largest_component_edge_count=float(np.mean(count_values)) if count_values else 0.0,
            )
        )

    result = MeshOnlyClusterGraphResult(
        cluster_ids=valid_ids,
        edges=edges,
        candidate_pair_count=len(edges),
        num_sampled_frames=num_sampled_frames,
        config=cfg,
    )
    print()
    print(f"Candidate pairs : {result.candidate_pair_count}")
    print(f"Kept edges      : {len(result.kept_edges)}")
    print(f"Cut edges       : {len(result.cut_edges)}")
    for e in sorted(result.edges, key=lambda e: (e.cluster_a, e.cluster_b)):
        tag = "KEEP" if e.kept else "CUT "
        print(
            f"[{tag}] {e.cluster_a}-{e.cluster_b} reason={e.reason} "
            f"connected={e.num_connected_frames}/{e.num_sampled_frames} "
            f"max_seam={e.max_largest_component_edge_count} mean_seam={e.mean_largest_component_edge_count:.2f}"
        )

    old_graph_path = (
        args.old_graph_path.expanduser().resolve()
        if args.old_graph_path is not None
        else work_dir / "analysis" / "cluster_graph_mesh" / "edges.pt"
    )
    old_pairs = load_old_graph_edge_pairs(old_graph_path)

    edges_kept_dicts = [_edge_to_dict(e) for e in result.kept_edges]
    edges_cut_dicts = [_edge_to_dict(e) for e in result.cut_edges]

    meta = {
        "method": "mesh_only_raw_contact",
        "work_dir": str(work_dir),
        "checkpoint": str(
            args.ckpt.expanduser().resolve()
            if args.ckpt is not None
            else work_dir / "checkpoints" / "last.ckpt"
        ),
        "num_frames": int(num_frames),
        "sampled_frame_indices": sampled_frames,
        "tau_o": cfg.tau_o,
        "mask_alpha_threshold": cfg.mask_alpha_threshold,
        "depth_jump_ratio": cfg.depth_jump_ratio,
        "min_seam_edges": cfg.min_seam_edges,
        "min_connected_frames": cfg.min_connected_frames,
        "min_cluster_size": cfg.min_cluster_size,
        "candidate_pair_count": result.candidate_pair_count,
        "kept_edge_count": len(edges_kept_dicts),
        "cut_edge_count": len(edges_cut_dicts),
        "valid_cluster_ids": valid_ids,
        "filtered_cluster_ids": filtered_ids,
        "old_graph_path": str(old_graph_path),
        "old_graph_found": old_pairs is not None,
    }

    payload = {
        "edge_index": torch.from_numpy(result.edge_index()).long(),
        "edges_kept": edges_kept_dicts,
        "edges_cut": edges_cut_dicts,
        "cluster_ids": valid_ids,
        "meta": meta,
    }
    torch.save(payload, edges_pt)
    write_csv(output_dir / "edges_kept.csv", [_edge_to_csv_row(e) for e in result.kept_edges])
    write_csv(output_dir / "edges_cut.csv", [_edge_to_csv_row(e) for e in result.cut_edges])
    write_csv(output_dir / "frame_pair_states.csv", frame_state_rows)

    report = {
        "meta": meta,
        "pairs": [
            {
                "cluster_a": e.cluster_a,
                "cluster_b": e.cluster_b,
                "kept": e.kept,
                "reason": e.reason,
                "num_connected_frames": e.num_connected_frames,
                "num_sampled_frames": e.num_sampled_frames,
                "max_largest_component_edge_count": e.max_largest_component_edge_count,
                "mean_largest_component_edge_count": e.mean_largest_component_edge_count,
            }
            for e in sorted(result.edges, key=lambda e: (e.cluster_a, e.cluster_b))
        ],
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    write_summary(output_dir / "summary.txt", result, old_graph_path, old_pairs, len(valid_ids))

    visualization_files: list[str] = []
    if not args.no_visualization:
        save_persistence_heatmap(result, valid_ids, figures_dir / "connectivity_heatmap.png")
        render_final_edge_graph(
            clusters, result, color_by_id, figures_dir / "final_edge_graph.png",
            max_points_per_cluster=args.max_points_per_cluster,
        )

        path_2d = render_2d_overlay(
            model=model,
            clusters=clusters,
            color_by_id=color_by_id,
            kept_edges=edges_kept_dicts,
            cut_edges=edges_cut_dicts,
            tau_p=cfg.tau_p,
            output_path=figures_dir / "graph_edges_2d.png",
            frame_index=args.frame_index_2d,
        )
        visualization_files.append(str(path_2d))
        print(f"[visualization] saved 2D overlay: {path_2d}")

        if not args.no_video:
            path_video = render_2d_overlay_video(
                model=model,
                clusters=clusters,
                color_by_id=color_by_id,
                kept_edges=edges_kept_dicts,
                cut_edges=edges_cut_dicts,
                tau_p=cfg.tau_p,
                output_path=figures_dir / "graph_edges_2d.mp4",
                fps=args.video_fps,
                frame_stride=args.video_frame_stride,
            )
            visualization_files.append(str(path_video))
            print(f"[visualization] saved 2D overlay video: {path_video}")

    print()
    print("=" * 78)
    print("Mesh-only raw-contact cluster graph build complete")
    print(f"Output PT : {edges_pt}")
    print(f"Frame-pair states CSV : {output_dir / 'frame_pair_states.csv'}")
    print(f"Report    : {output_dir / 'report.json'}")
    print(f"Summary   : {output_dir / 'summary.txt'}")
    if visualization_files:
        print("Visualizations :")
        for path in visualization_files:
            print(f"  {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()

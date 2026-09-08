#!/usr/bin/env python3
"""
Mesh-boundary-persistence cluster graph builder (DreaMo-style surface
adjacency), an alternative to flow3d/analysis/build_cluster_graph.py's
distance-based candidate-pair graph.

Motivation
----------
build_cluster_graph.py decides adjacency purely from *point-to-point*
distance between each cluster's designated boundary Gaussians. That signal
degrades when the candidate-pair search radius or the boundary-Gaussian
selection picks up a handful of stray/noisy points that happen to sit close
to an unrelated cluster (see build_cluster_graph.py's own gate-check, which
is a self-test for exactly this failure mode). Two clusters that are merely
*near* each other in canonical space can look connected even with no shared
physical surface between them.

This module used to answer that with a Poisson-reconstructed surface over
the raw 3D Gaussian point cloud, but Poisson is a *global* implicit-function
solve: it happily bridges two point clusters that are merely close in space
(closer than its own effective smoothing radius) with a continuous sheet of
triangles, even when nothing is actually there -- confirmed on a real
checkpoint (visibly malformed, self-intersecting meshes; spurious edges like
1-2/1-7/7-27 where the real surfaces don't touch). No amount of tau_s/tau_p
tuning fixes that, because the mesh itself is wrong before any threshold is
even applied.

Instead, this module builds the mesh the way a depth camera would: at each
sampled frame, render this specific frame's own foreground alpha mask and
depth from its actual camera pose, back-project only the mask-covered
pixels to 3D, and triangulate the regular pixel grid directly -- a triangle
only exists where three adjacent pixels are all foreground AND agree in
depth. A self-occlusion silhouette (say, a forearm in front of the torso)
has a large depth jump between the arm's near pixels and the torso's far
pixels right at the boundary, so the grid triangulation simply never
connects them -- there is no implicit function to be fooled into bridging
the gap. Each mesh vertex is labeled with whichever cluster the renderer's
own alpha-compositing says actually painted that pixel (see
_build_cluster_onehot below), not a post-hoc nearest-neighbor guess. Cross-
cluster mesh edges and persistence are otherwise exactly as before.

No motion/velocity signal is used anywhere in this module -- persistence is
purely "was cluster_a's surface touching cluster_b's surface in this
sampled frame", independent of how either cluster moves.

Algorithm
---------
1. Cluster ids are the checkpoint's fixed ``model.fg.get_cluster_ids()`` --
   never recomputed here, only filtered by --min-cluster-size (same
   convention as build_cluster_graph.py's load_model_and_clusters, reused
   directly).
2. Pass 1 (seed collection): for each sampled frame t in
   range(0, num_frames, --frame-interval), render this frame's own alpha,
   depth and per-cluster one-hot color (colors_override) restricted to
   Gaussians in a valid cluster with opacity > --tau-o, mask to pixels with
   alpha > --mask-alpha-threshold, back-project to 3D (unproject_depth_map),
   and triangulate the pixel grid (build_visible_surface_mesh) exactly as
   before -- a triangle only survives if all 3 corner pixels are foreground
   AND no pair of their depths differs by more than --depth-jump-ratio of
   their own depth. For every cross-cluster mesh edge (endpoints with
   different rendered-argmax cluster labels), the boundary vertices on each
   side contribute their nearest 1..--seed-knn-k same-cluster real
   Gaussians as that frame's "seed" set for that side (collect_frame_seeds).
   Across all sampled frames, a per-Gaussian vote count accumulates
   per (pair, side).
3. Between passes (contact core -> boundary patch, build_contact_core_and_
   patch): a pair needs contact in >= --min-contact-frames frames to be
   considered at all. A Gaussian becomes part of the "contact core" once its
   vote fraction (of that pair's own contact-frame count) reaches
   --seed-vote-min-fraction -- multi-frame agreement instead of trusting any
   single frame's noise. The core is then geodesically dilated along a
   same-cluster Gaussian graph (mutual-kNN over canonical positions, long
   edges pruned relative to each Gaussian's own local spacing so the
   dilation can never leak onto a different, unconnected part of the
   surface) out to a radius 3x the core's own characteristic radius
   (computed per connected component, since a core split across disconnected
   surface parts has no single well-defined radius) -- this is the
   "boundary patch". A pair with an empty core on either side is dropped.
4. Pass 2 (score + visibility): the per-frame cross-cluster score is now
   localized to the boundary patch on BOTH sides -- the numerator only
   counts cross-cluster mesh edges whose two endpoints are each a patch
   member on their own side, and the denominator is each side's own
   patch-restricted incident-edge count (not the whole cluster's mesh-edge
   budget, which is what made the old score's denominator close to the
   cluster's total surface area regardless of how localized the actual
   contact was). Separately, per-Gaussian visibility of the (small) contact
   core -- not the wider patch -- is judged this frame via a depth test
   against the frame's own rendered depth map (is this Gaussian's own
   camera-space depth close to what was actually rendered at its pixel) AND
   a check that the pixel's rendered-argmax cluster owner really is this
   side's cluster (a depth match alone can't distinguish this Gaussian from
   a different, similarly-deep cluster that actually painted that pixel --
   exactly the ambiguity at a contact seam). A frame counts as "known" for a
   pair only if both sides' contact cores are each >= --visibility-fraction-
   threshold visible; otherwise the frame is "unknown" and excluded from the
   persistence denominator entirely (score is still recorded for the
   summary, connectivity is not judged).
5. persistence(a, b) = (# known frames scored connected) / (# known frames),
   with a floor: a pair needs >= --min-known-frames known frames or it is
   cut outright (an occluded-the-whole-sequence or barely-observed pair
   should never be vacuously "kept"). final_edge(a, b) = persistence > tau_p.

Output
------
Writes, under <output-dir> (default: <work-dir>/analysis/cluster_graph_mesh):

    edges.pt              -- same schema build_cluster_graph.py's edges.pt
                              uses (edge_index, edges_kept, edges_cut,
                              cluster_ids, meta), consumed by the same GNN /
                              loss code. boundary_global_indices_a/b (the
                              field those consumers read) is populated with
                              this pair's boundary_patch_global_indices_a/b.
                              The OLD distance-based builder and its output
                              directory are untouched.
    contact_core.pt        -- {"pairs": [{cluster_a, cluster_b,
                              global_indices_a, global_indices_b}, ...]} --
                              the small, stable, multi-frame-voted touching
                              region, for every candidate pair that survived
                              --min-contact-frames and produced a non-empty
                              core (not just kept edges).
    boundary_patch.pt      -- same shape as contact_core.pt, but the
                              geodesically-dilated (3x) patch used as the
                              score's numerator/denominator region.
    edges_kept.csv / edges_cut.csv
    report.json
    summary.txt            -- kept/cut edge list (with cut reasons and
                               known/unknown frame counts) + comparison
                               against the old distance-based graph, if one
                               exists at <work-dir>/analysis/cluster_graph/
                               edges.pt (override with --old-graph-path).
    meshes/frame_XXXX.ply, meshes/frame_XXXX.png
                            -- uncolored per-frame mesh.
    colored_meshes/frame_XXXX.ply, colored_meshes/frame_XXXX.png
                            -- same mesh, vertices colored by cluster id
                               (one fixed color per cluster id across every
                               frame).
    figures/score_matrices/frame_XXXX.npy, frame_XXXX.csv
                            -- per-frame normalized cross-cluster score
                               matrix (npy, valid_ids x valid_ids) and a
                               sparse listing of its nonzero pairs (csv).
    figures/contact_patches/pair_A_B.png
                            -- one figure per candidate pair with a stable
                               core: this pair's contact core (red) and
                               boundary patch (orange) projected onto the
                               actual rendered RGB frame at a sampled frame
                               where the pair was connected (or otherwise
                               visible), not an abstract point cloud.
    figures/persistence_heatmap.png
    figures/final_edge_graph.png
    figures/graph_edges_2d.png       -- kept/cut edges overlaid on one
                                         rendered frame (same style as
                                         build_cluster_graph.py's own
                                         graph_edges_2d.png), unless
                                         --no-visualization.
    figures/graph_edges_2d.mp4       -- the same overlay across every
                                         rendered frame, unless
                                         --no-visualization or --no-video.

Example
-------
    python flow3d/analysis/build_cluster_graph_mesh.py \\
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

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from flow3d.analysis.cluster_pairs import (
    ClusterInfo,
    _get_camera_w2cs,
    _get_dynamic_fg_means,
    _load_overlay_font,
    _project_world_points,
    _sample_points_for_visualization,
    _set_axes_equal_3d,
    write_csv,
)
from flow3d.analysis.build_cluster_graph import (
    CUT_COLOR_2D,
    KEPT_COLOR_2D,
    _dashed_line,
    load_model_and_clusters,
)

# Cap on triangles actually drawn per PNG (the saved .ply keeps all of
# them) -- matplotlib's Poly3DCollection gets slow well before Open3D does.
_MAX_RENDER_FACES = 20000
_MESH_VIEW = (20, 35)  # (elev, azim), matches build_cluster_graph.py's "Perspective" view
# A visible-surface mesh is inherently a one-sided shell (only the
# camera-facing pixels of this one frame exist, no back side) -- expected,
# not a bug, when eyeballing meshes/*.png.

# Boundary patch radius, as a multiple of the contact core's own
# characteristic geodesic radius (see build_contact_core_and_patch). Not a
# CLI knob -- "1.3x the touching region" is the algorithm's definition, not a
# threshold meant to be swept per-checkpoint like tau_s/tau_p.
_PATCH_EXPANSION_RATIO = 0.4

# Temporal contact-stability thresholds (see _has_temporal_contact_stability):
# a candidate pair needs either this many CONSECUTIVE contact frames, or at
# least _CONTACT_WINDOW_MIN_COUNT hits within any _CONTACT_WINDOW_SIZE-frame
# window, to survive as a real (not scattered-noise) contact. Not CLI knobs,
# same rationale as _PATCH_EXPANSION_RATIO.
_MIN_CONSECUTIVE_CONTACT_FRAMES = 3
_CONTACT_WINDOW_SIZE = 5
_CONTACT_WINDOW_MIN_COUNT = 3


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MeshGraphConfig:
    tau_o: float
    mask_alpha_threshold: float
    depth_jump_ratio: float
    tau_s: float
    tau_p: float
    frame_interval: int
    min_cluster_size: int
    seed_knn_k: int
    seed_vote_min_fraction: float
    graph_knn_k: int
    graph_edge_length_ratio: float
    visibility_depth_tol: float
    visibility_fraction_threshold: float
    min_contact_frames: int
    min_known_frames: int


@dataclass
class MeshClusterPairEdge:
    cluster_a: int
    cluster_b: int
    kept: bool
    reason: str
    persistence: float
    num_connected_frames: int
    num_known_frames: int
    num_unknown_frames: int
    num_sampled_frames: int
    mean_score: float  # mean/max are over sampled frames with a nonzero
    max_score: float  # patch score, NOT necessarily every known frame.
    connected_frame_indices: list[int]
    unknown_frame_indices: list[int]
    num_boundary_a: int
    num_boundary_b: int
    boundary_global_indices_a: np.ndarray  # == boundary_patch_global_indices_a
    boundary_global_indices_b: np.ndarray  # (kept for downstream compatibility)
    contact_core_global_indices_a: np.ndarray
    contact_core_global_indices_b: np.ndarray
    boundary_patch_global_indices_a: np.ndarray
    boundary_patch_global_indices_b: np.ndarray
    boundary_patch_weight_a: np.ndarray  # aligned with boundary_patch_global_indices_a
    boundary_patch_weight_b: np.ndarray
    boundary_patch_local_scale_a: np.ndarray  # aligned with boundary_patch_global_indices_a
    boundary_patch_local_scale_b: np.ndarray
    contact_reference_distance: float  # median weighted-patch-centroid distance over CONNECTED frames


@dataclass
class MeshClusterGraphResult:
    cluster_ids: list[int]
    edges: list[MeshClusterPairEdge]
    candidate_pair_count: int
    num_sampled_frames: int
    config: MeshGraphConfig

    @property
    def kept_edges(self) -> list[MeshClusterPairEdge]:
        return [e for e in self.edges if e.kept]

    @property
    def cut_edges(self) -> list[MeshClusterPairEdge]:
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
class FrameRenderData:
    """Everything one sampled frame's render+mesh pass produces. Shared by
    both pipeline passes: pass 1 (collect_frame_seeds) only reads the mesh
    fields; pass 2 (score_frame_against_patches,
    compute_contact_core_visibility) additionally needs the full (H, W)
    label grid and every fg Gaussian's posed position this frame. Rebuilt
    fresh each pass (an accepted tradeoff -- see module docstring -- so
    per-frame depth/alpha/label maps are never all held in memory at once
    across the whole sequence)."""

    frame_index: int
    vertices: np.ndarray  # (V, 3) world-space
    triangles: np.ndarray  # (F, 3) vertex-local indices into vertices
    vertex_cluster_ids: np.ndarray  # (V,) each vertex's rendered-argmax cluster id
    mesh_edges: np.ndarray  # (M, 2) unique undirected mesh edges
    vertex_cluster_ids_grid: np.ndarray  # (H, W) full rendered-argmax label grid (pre-mask)
    alpha_map: np.ndarray  # (H, W)
    depth_map: np.ndarray  # (H, W)
    gaussian_positions: np.ndarray  # (G, 3) posed, valid-cluster+opacity filtered
    gaussian_cluster_ids: np.ndarray  # (G,)
    gaussian_global_indices: np.ndarray  # (G,) into the full canonical fg array
    posed_means_all_fg: np.ndarray  # (N_fg, 3) posed, EVERY fg Gaussian, unfiltered
    w2c: torch.Tensor
    intrinsic: torch.Tensor


# ---------------------------------------------------------------------------
# Cluster coloring (fixed across frames)
# ---------------------------------------------------------------------------


def _id_to_pos_lookup(sorted_ids: list[int]) -> np.ndarray:
    """Dense cluster-id -> compact-position lookup array (id_to_pos_arr[cid]
    == its index in sorted_ids). Cluster ids are small non-negative ints in
    practice, so this dense array is both simpler and far faster than a
    python-dict + np.vectorize for the (V, k) - sized lookups below."""
    max_id = max(sorted_ids)
    lookup = np.full(max_id + 1, -1, dtype=np.int64)
    for i, cid in enumerate(sorted_ids):
        lookup[cid] = i
    return lookup


def build_cluster_colormap(valid_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    """One fixed RGBA color per cluster id, same convention (tab20, indexed
    by sorted-id enumeration order) as cluster_pairs.py's
    _prepare_cluster_visual_data, so a cluster's color is identical here and
    in the old graph's visualizations."""
    cmap = plt.get_cmap("tab20")
    return {cid: cmap(i % cmap.N) for i, cid in enumerate(sorted(valid_ids))}


# ---------------------------------------------------------------------------
# Per-frame Gaussian snapshot
# ---------------------------------------------------------------------------


def load_frame_snapshot(
    model: Any,
    valid_ids: list[int],
    frame_index: int,
    tau_o: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return: (positions (N,3) float64, cluster_ids (N,) int64, global_indices
        (N,) int64 into the full canonical foreground array), restricted to
        Gaussians in a valid (>= --min-cluster-size) cluster with opacity >
        tau_o.
    """
    device = model.fg.params["means"].device
    with torch.no_grad():
        means, _ = model.compute_poses_fg(torch.tensor([frame_index], device=device))
    positions = means[:, 0, :].detach().double().cpu().numpy()  # (G, 3)
    opacities = model.fg.get_opacities().reshape(-1).detach().float().cpu().numpy()  # (G,)
    cluster_ids_all = model.fg.get_cluster_ids().reshape(-1).long().cpu().numpy()  # (G,)

    valid_mask = np.isin(cluster_ids_all, np.asarray(valid_ids, dtype=np.int64))
    keep_mask = valid_mask & (opacities > tau_o)
    global_indices = np.nonzero(keep_mask)[0].astype(np.int64)
    return positions[keep_mask], cluster_ids_all[keep_mask], global_indices


# ---------------------------------------------------------------------------
# Visible-surface mesh from rendered depth (pure numpy/torch + Open3D for I/O)
# ---------------------------------------------------------------------------


def _build_cluster_onehot(cluster_ids_all: torch.Tensor, valid_ids: list[int]) -> torch.Tensor:
    """(num_fg_gaussians, C) one-hot "color": row g is all-zero unless
    Gaussian g belongs to a valid cluster, in which case exactly one entry
    (that cluster's compact position in sorted(valid_ids)) is 1. Fed to
    model.render as colors_override so the renderer's own alpha compositing
    tells us, per pixel, how much of its accumulated opacity came from each
    cluster -- an exact, occlusion-aware answer to "which Gaussian actually
    painted this pixel", unlike a post-hoc nearest-3D-point guess."""
    device = cluster_ids_all.device
    sorted_ids = sorted(valid_ids)
    onehot = torch.zeros(cluster_ids_all.shape[0], len(sorted_ids), device=device)
    for pos, cid in enumerate(sorted_ids):
        onehot[cluster_ids_all == cid, pos] = 1.0
    return onehot


def unproject_depth_map(
    depth_map: np.ndarray,
    intrinsic: torch.Tensor,
    w2c: torch.Tensor,
) -> np.ndarray:
    """(H, W) camera-space depth -> (H, W, 3) world-space points (garbage,
    but never read, outside the caller's own valid-pixel mask)."""
    height, width = depth_map.shape
    device = intrinsic.device
    inv_k = torch.linalg.inv(intrinsic.double())
    c2w = torch.linalg.inv(w2c.double())

    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels_h = torch.from_numpy(np.stack([us, vs, np.ones_like(us)], axis=-1)).to(
        device=device, dtype=torch.float64
    )  # (H, W, 3)
    rays = torch.einsum("ij,hwj->hwi", inv_k, pixels_h)  # (H, W, 3)
    depth_t = torch.from_numpy(depth_map).to(device=device, dtype=torch.float64)
    camera_points = rays * depth_t[..., None]  # (H, W, 3)
    homogeneous = torch.cat([camera_points, torch.ones_like(camera_points[..., :1])], dim=-1)
    world_points = torch.einsum("ij,hwj->hwi", c2w, homogeneous)[..., :3]
    return world_points.detach().cpu().numpy()


def build_visible_surface_mesh(
    valid_mask: np.ndarray,
    depth_map: np.ndarray,
    world_points: np.ndarray,
    depth_jump_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate the regular pixel grid directly: each 2x2 pixel block
    contributes up to two triangles (top-left, bottom-right split), kept
    only if all 3 corners are inside valid_mask (so a triangle can never
    cross outside the foreground silhouette) AND no pair of the 3 corners'
    depths differs by more than depth_jump_ratio of their own depth (so a
    self-occlusion silhouette -- e.g. an arm in front of the torso -- is
    never bridged into one continuous surface, unlike Poisson reconstruction
    which has no such guardrail).

    :param valid_mask: (H, W) bool, this frame's foreground alpha mask.
    :param depth_map: (H, W) float, this frame's rendered depth.
    :param world_points: (H, W, 3) float, unproject_depth_map's output.
    :param depth_jump_ratio: relative depth-discontinuity cutoff.
    :return: vertices (V, 3) float64, triangles (F, 3) int64 (vertex-local,
        compact indices into `vertices`).
    """
    height, width = valid_mask.shape
    vertex_id = np.full((height, width), -1, dtype=np.int64)
    valid_flat_idx = np.nonzero(valid_mask.reshape(-1))[0]
    vertex_id.reshape(-1)[valid_flat_idx] = np.arange(valid_flat_idx.shape[0])
    vertices = world_points.reshape(-1, 3)[valid_flat_idx]

    r, c = np.meshgrid(np.arange(height - 1), np.arange(width - 1), indexing="ij")
    r, c = r.reshape(-1), c.reshape(-1)

    v00, v01 = vertex_id[r, c], vertex_id[r, c + 1]
    v10, v11 = vertex_id[r + 1, c], vertex_id[r + 1, c + 1]
    d00, d01 = depth_map[r, c], depth_map[r, c + 1]
    d10, d11 = depth_map[r + 1, c], depth_map[r + 1, c + 1]

    def _depth_ok(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        return np.abs(da - db) <= depth_jump_ratio * np.maximum(np.maximum(da, db), 1e-8)

    triangle_chunks = []
    for va, vb, vc, da, db, dc in (
        (v00, v01, v10, d00, d01, d10),
        (v01, v11, v10, d01, d11, d10),
    ):
        ok = (
            (va >= 0) & (vb >= 0) & (vc >= 0)
            & _depth_ok(da, db) & _depth_ok(db, dc) & _depth_ok(da, dc)
        )
        if bool(ok.any()):
            triangle_chunks.append(np.stack([va[ok], vb[ok], vc[ok]], axis=1))

    triangles = (
        np.concatenate(triangle_chunks, axis=0) if triangle_chunks else np.zeros((0, 3), dtype=np.int64)
    )
    return vertices, triangles


def mesh_edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """(F, 3) triangle vertex indices -> (M, 2) unique undirected edges."""
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


# ---------------------------------------------------------------------------
# Same-cluster Gaussian graph + geodesic contact core / boundary patch
# ---------------------------------------------------------------------------


def build_mutual_knn_graph(
    positions: np.ndarray, graph_knn_k: int, edge_length_ratio: float
) -> tuple[csr_matrix, np.ndarray]:
    """
    Same-cluster Gaussian graph over `positions` (this cluster's own
    canonical-space points, one graph node per point): mutual-kNN with
    long-edge pruning relative to each node's OWN local spacing, so
    geodesic dilation downstream can never leak onto a different,
    spatially-close-but-unconnected part of the surface -- the same failure
    mode this whole module exists to avoid in the mesh itself (see module
    docstring), now guarded in the Gaussian graph too.

    :return: (weighted sparse adjacency [Euclidean edge weights, symmetric],
        per-node local_scale [median distance to its own k nearest
        neighbors, excluding itself -- used both for pruning here and as
        the degenerate-core-radius fallback in build_contact_core_and_patch]).
    """
    n = positions.shape[0]
    if n == 0:
        return csr_matrix((0, 0)), np.zeros(0, dtype=np.float64)

    tree = cKDTree(positions)
    k_query = min(graph_knn_k + 1, n)
    dists, idxs = tree.query(positions, k=k_query)
    if k_query == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    # Column 0 is each point's self-match (distance 0); the real neighbors
    # are columns 1..k_query-1.
    neighbor_dists = dists[:, 1:]
    neighbor_idxs = idxs[:, 1:]

    if neighbor_dists.shape[1] == 0:
        return csr_matrix((n, n)), np.zeros(n, dtype=np.float64)

    # The MEDIAN (not the farthest, dists[:, -1]) of a node's own neighbor
    # distances: every candidate kNN edge is by construction no longer than
    # dists[:, -1], so comparing an edge against a multiple of its own
    # farthest-neighbor distance could never prune anything. The median is
    # a genuinely "typical" local spacing an edge can meaningfully exceed.
    local_scale = np.median(neighbor_dists, axis=1)

    src = np.repeat(np.arange(n), neighbor_idxs.shape[1])
    dst = neighbor_idxs.reshape(-1)
    candidate = coo_matrix(
        (np.ones(src.shape[0], dtype=np.float64), (src, dst)), shape=(n, n)
    ).tocsr()
    mutual = candidate.multiply(candidate.T).tocsr()  # nonzero only where both directions agree
    mutual.eliminate_zeros()
    mi, mj = mutual.nonzero()
    if mi.size == 0:
        return csr_matrix((n, n)), local_scale

    edge_len = np.linalg.norm(positions[mi] - positions[mj], axis=1)
    keep = edge_len <= edge_length_ratio * np.maximum(local_scale[mi], local_scale[mj])
    mi, mj, edge_len = mi[keep], mj[keep], edge_len[keep]

    graph = coo_matrix((edge_len, (mi, mj)), shape=(n, n)).tocsr()
    return graph, local_scale


def build_contact_core_and_patch(
    graph: csr_matrix,
    local_scale: np.ndarray,
    canonical_positions: np.ndarray,
    global_indices: np.ndarray,
    core_node_mask: np.ndarray,
    patch_expansion_ratio: float = _PATCH_EXPANSION_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Geodesically dilate `core_node_mask` (already vote-thresholded, in this
    cluster's compact node ordering matching `graph`/`global_indices`) along
    `graph` into a boundary patch `patch_expansion_ratio`x the core's own
    characteristic radius. Computed PER CONNECTED COMPONENT of `graph`, so a
    core split across disconnected parts of the surface doesn't produce an
    undefined cross-component "radius" (a single medoid can't reach a core
    member in a different component -- infinite distance).

    Each patch member also gets a geodesic falloff WEIGHT (1.0 exactly at the
    core, smoothly down to 0.0 at the patch's own outer radius via a cubic
    smoothstep of dist_to_core/target_radius) and its own `local_scale`
    (from `build_mutual_knn_graph`) carried along -- both consumed downstream
    by flow3d/graph_relative_edge.py, which snaps a live (post-densify/cull)
    Gaussian onto its nearest frozen patch-reference point instead of
    recomputing geodesic distance itself.

    :return: (contact_core_global_indices, boundary_patch_global_indices,
        boundary_patch_weight, boundary_patch_local_scale) -- the last two
        aligned 1:1 with boundary_patch_global_indices.
    """
    n = graph.shape[0]
    core_positions = np.nonzero(core_node_mask)[0]
    if core_positions.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty, empty, empty_f, empty_f

    _, labels = connected_components(graph, directed=False)

    target_radius_by_component: dict[int, float] = {}
    for comp_id in np.unique(labels[core_positions]).tolist():
        comp_core = core_positions[labels[core_positions] == comp_id]
        core_radius = 0.0
        if comp_core.size > 1:
            centroid = canonical_positions[comp_core].mean(axis=0)
            medoid = int(
                comp_core[np.argmin(np.linalg.norm(canonical_positions[comp_core] - centroid, axis=1))]
            )
            dist_from_medoid = dijkstra(graph, directed=False, indices=[medoid])[0]
            finite = dist_from_medoid[comp_core]
            finite = finite[np.isfinite(finite)]
            core_radius = float(finite.mean()) if finite.size > 0 else 0.0
        if core_radius <= 0.0:
            comp_local_scale = local_scale[labels == comp_id]
            core_radius = float(np.median(comp_local_scale)) if comp_local_scale.size > 0 else 0.0
        target_radius_by_component[comp_id] = patch_expansion_ratio * core_radius

    dist_to_core = dijkstra(graph, directed=False, indices=core_positions, min_only=True)

    weight_all = np.zeros(n, dtype=np.float64)
    patch_mask = core_node_mask.copy()
    for comp_id, target_radius in target_radius_by_component.items():
        comp_node_mask = labels == comp_id
        comp_patch_mask = comp_node_mask & (dist_to_core <= target_radius)
        patch_mask |= comp_patch_mask
        t = np.clip(dist_to_core[comp_patch_mask] / max(target_radius, 1e-8), 0.0, 1.0)
        weight_all[comp_patch_mask] = 1.0 - (3 * t**2 - 2 * t**3)
    weight_all[core_node_mask] = 1.0  # exact at the core (dist_to_core=0 already gives this)

    return (
        global_indices[core_node_mask],
        global_indices[patch_mask],
        weight_all[patch_mask],
        local_scale[patch_mask],
    )


def _cluster_graph_for(
    cache: dict[int, tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, int]]],
    cluster: ClusterInfo,
    cfg: MeshGraphConfig,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, int]]:
    """Lazily build (and cache) one cluster's own mutual-kNN graph -- a
    cluster can appear in multiple candidate pairs (once per neighbor), and
    the graph only depends on the cluster's own canonical points."""
    if cluster.cluster_id not in cache:
        positions = cluster.canonical_points.detach().double().cpu().numpy()
        global_indices = cluster.global_indices.detach().cpu().numpy().astype(np.int64)
        graph, local_scale = build_mutual_knn_graph(
            positions, cfg.graph_knn_k, cfg.graph_edge_length_ratio
        )
        global_to_local = {int(g): i for i, g in enumerate(global_indices.tolist())}
        cache[cluster.cluster_id] = (graph, local_scale, global_indices, global_to_local)
    return cache[cluster.cluster_id]


def build_contact_core_and_patch_for_side(
    graph_cache: dict[int, tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, int]]],
    cluster: ClusterInfo,
    votes: dict[int, int],
    num_candidate_frames: int,
    cfg: MeshGraphConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One side of one candidate pair: threshold this side's seed votes into
    a stable contact-core node mask, then dilate. `votes` maps a global
    Gaussian index (within this cluster) to how many of the pair's
    `num_candidate_frames` contact frames selected it as a seed.

    :return: (core_global_indices, patch_global_indices, patch_weight,
        patch_local_scale) -- see build_contact_core_and_patch.
    """
    graph, local_scale, global_indices, global_to_local = _cluster_graph_for(graph_cache, cluster, cfg)
    core_node_mask = np.zeros(global_indices.shape[0], dtype=bool)
    for g, count in votes.items():
        if count / num_candidate_frames >= cfg.seed_vote_min_fraction:
            local_pos = global_to_local.get(int(g))
            if local_pos is not None:
                core_node_mask[local_pos] = True
    if not core_node_mask.any():
        empty = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty, empty, empty_f, empty_f

    canonical_positions = cluster.canonical_points.detach().double().cpu().numpy()
    return build_contact_core_and_patch(
        graph, local_scale, canonical_positions, global_indices, core_node_mask
    )


def _has_temporal_contact_stability(contact_frame_indices: list[int]) -> bool:
    """True iff a candidate pair's raw per-frame contact hits (Pass 1's
    cross-cluster mesh-edge detection, now over every frame -- see module
    docstring) show real temporal coherence: either
    `_MIN_CONSECUTIVE_CONTACT_FRAMES` frames in a row, or at least
    `_CONTACT_WINDOW_MIN_COUNT` hits within any `_CONTACT_WINDOW_SIZE`-frame
    window -- rather than a handful of scattered single-frame noise hits
    that happen to clear --min-contact-frames on raw count alone."""
    if not contact_frame_indices:
        return False
    frame_set = set(contact_frame_indices)
    frames = sorted(frame_set)
    if any(
        all((f + k) in frame_set for k in range(_MIN_CONSECUTIVE_CONTACT_FRAMES))
        for f in frames
    ):
        return True
    return any(
        sum(1 for k in range(start, start + _CONTACT_WINDOW_SIZE) if k in frame_set)
        >= _CONTACT_WINDOW_MIN_COUNT
        for start in frames
    )


# ---------------------------------------------------------------------------
# Pass 1: per-frame mesh + cross-cluster seed collection
# ---------------------------------------------------------------------------


def collect_frame_seeds(
    vertices: np.ndarray,
    vertex_cluster_ids: np.ndarray,
    mesh_edges: np.ndarray,
    gaussian_positions: np.ndarray,
    gaussian_cluster_ids: np.ndarray,
    gaussian_global_indices: np.ndarray,
    valid_ids: list[int],
    seed_knn_k: int,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """
    For each cross-cluster pair present in this frame's mesh, this frame's
    seed Gaussians on each side: the nearest `seed_knn_k` (1..3) same-
    cluster real Gaussians to that side's boundary mesh vertices -- the raw
    material multi-frame voting turns into a stable contact core (see
    module docstring). Keyed (cid_lo, cid_hi) with cid_lo < cid_hi; values
    are (seed_global_indices_lo, seed_global_indices_hi).
    """
    va = vertex_cluster_ids[mesh_edges[:, 0]]
    vb = vertex_cluster_ids[mesh_edges[:, 1]]
    cross_mask = va != vb
    seeds: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    if not bool(cross_mask.any()):
        return seeds

    trees: dict[int, tuple[cKDTree, np.ndarray]] = {}

    def _tree_for(cid: int) -> tuple[cKDTree, np.ndarray]:
        if cid not in trees:
            cluster_mask = gaussian_cluster_ids == cid
            trees[cid] = (
                cKDTree(gaussian_positions[cluster_mask]),
                gaussian_global_indices[cluster_mask],
            )
        return trees[cid]

    def _seed_gaussians(vertex_indices: np.ndarray, cid: int) -> np.ndarray:
        if vertex_indices.size == 0:
            return np.zeros(0, dtype=np.int64)
        tree, cluster_global_indices = _tree_for(cid)
        k_eff = min(seed_knn_k, cluster_global_indices.shape[0])
        if k_eff == 0:
            return np.zeros(0, dtype=np.int64)
        _, idxs = tree.query(vertices[vertex_indices], k=k_eff)
        if k_eff == 1:
            idxs = idxs[:, None]
        return np.unique(cluster_global_indices[idxs.reshape(-1)])

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
        edge_rows_for_pair = cross_edge_rows[inverse == k_index]
        edge_endpoints = mesh_edges[edge_rows_for_pair]  # (n, 2)
        endpoint_cids = vertex_cluster_ids[edge_endpoints]  # (n, 2)

        side_lo_vertices = np.unique(edge_endpoints[endpoint_cids == cid_lo])
        side_hi_vertices = np.unique(edge_endpoints[endpoint_cids == cid_hi])

        seeds[(cid_lo, cid_hi)] = (
            _seed_gaussians(side_lo_vertices, cid_lo),
            _seed_gaussians(side_hi_vertices, cid_hi),
        )

    return seeds


def render_frame_mesh(
    model: Any,
    valid_ids: list[int],
    frame_index: int,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    cfg: MeshGraphConfig,
) -> FrameRenderData:
    """Build one sampled frame's visible-surface mesh directly from that
    frame's own rendered alpha mask + depth (see module docstring). Shared
    by both pipeline passes -- see FrameRenderData."""
    device = model.fg.params["means"].device
    width, height = image_size

    cluster_ids_all = model.fg.get_cluster_ids().reshape(-1).long().to(device)
    opacities_all = model.fg.get_opacities().reshape(-1).to(device)
    onehot = _build_cluster_onehot(cluster_ids_all, valid_ids)
    in_valid_cluster = onehot.sum(dim=1) > 0
    filter_mask = in_valid_cluster & (opacities_all > cfg.tau_o)

    with torch.no_grad():
        posed_means_all, _ = model.compute_poses_fg(torch.tensor([frame_index], device=device))
        posed_means_all = posed_means_all[:, 0, :]  # (N_fg, 3)
        render_out = model.render(
            frame_index, w2c[None], intrinsic[None], (width, height),
            return_color=True, return_depth=True, fg_only=True,
            filter_mask=filter_mask, colors_override=onehot, bg_color=0.0,
            use_learned_poses=False,
        )
    cluster_weights = render_out["img"][0].detach().float().cpu().numpy()  # (H, W, C)
    depth_map = render_out["depth"][0, ..., 0].detach().double().cpu().numpy()  # (H, W)
    alpha_map = render_out["acc"][0, ..., 0].detach().float().cpu().numpy()  # (H, W)

    valid_mask = (alpha_map > cfg.mask_alpha_threshold) & (depth_map > 1e-6)
    sorted_ids = sorted(valid_ids)
    vertex_cluster_ids_grid = np.asarray(sorted_ids, dtype=np.int64)[cluster_weights.argmax(axis=-1)]

    world_points = unproject_depth_map(depth_map, intrinsic, w2c)  # (H, W, 3)
    vertices, triangles = build_visible_surface_mesh(valid_mask, depth_map, world_points, cfg.depth_jump_ratio)
    vertex_cluster_ids = vertex_cluster_ids_grid.reshape(-1)[valid_mask.reshape(-1)]
    mesh_edges = mesh_edges_from_triangles(triangles)

    positions, gaussian_cluster_ids, global_indices = load_frame_snapshot(
        model, valid_ids, frame_index, cfg.tau_o
    )

    return FrameRenderData(
        frame_index=frame_index,
        vertices=vertices,
        triangles=triangles,
        vertex_cluster_ids=vertex_cluster_ids,
        mesh_edges=mesh_edges,
        vertex_cluster_ids_grid=vertex_cluster_ids_grid,
        alpha_map=alpha_map,
        depth_map=depth_map,
        gaussian_positions=positions,
        gaussian_cluster_ids=gaussian_cluster_ids,
        gaussian_global_indices=global_indices,
        posed_means_all_fg=posed_means_all.detach().double().cpu().numpy(),
        w2c=w2c,
        intrinsic=intrinsic,
    )


# ---------------------------------------------------------------------------
# Pass 2: patch-localized score + contact-core visibility
# ---------------------------------------------------------------------------


def score_frame_against_patches(
    frame_data: FrameRenderData,
    boundary_patch_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[int, int], float]:
    """
    Per-frame score for every candidate pair that has a boundary patch,
    localized to the patch on BOTH the numerator (cross edges) and
    denominator (incident-edge budget): the numerator only counts cross-
    cluster mesh edges whose two endpoints are each a patch member on their
    own side, and the denominator is each side's own patch-restricted
    incident-edge count -- see module docstring. Pairs with zero patch
    presence this frame (no incident edges AND no cross edges) are omitted
    from the returned dict.
    """
    vertices = frame_data.vertices
    vertex_cluster_ids = frame_data.vertex_cluster_ids
    mesh_edges = frame_data.mesh_edges
    gaussian_positions = frame_data.gaussian_positions
    gaussian_cluster_ids = frame_data.gaussian_cluster_ids
    gaussian_global_indices = frame_data.gaussian_global_indices

    trees: dict[int, tuple[cKDTree, np.ndarray]] = {}

    def _tree_for(cid: int) -> tuple[cKDTree, np.ndarray]:
        if cid not in trees:
            cluster_mask = gaussian_cluster_ids == cid
            trees[cid] = (
                cKDTree(gaussian_positions[cluster_mask]),
                gaussian_global_indices[cluster_mask],
            )
        return trees[cid]

    def _patch_membership(cid: int, patch_global_indices: np.ndarray) -> np.ndarray:
        """(V,) bool -- True at vertices labeled `cid` whose nearest real
        Gaussian (of cluster `cid`) is inside `patch_global_indices`."""
        membership = np.zeros(vertices.shape[0], dtype=bool)
        vertex_mask = vertex_cluster_ids == cid
        if not bool(vertex_mask.any()) or patch_global_indices.size == 0:
            return membership
        tree, cluster_global_indices = _tree_for(cid)
        if cluster_global_indices.shape[0] == 0:
            return membership
        _, idx = tree.query(vertices[vertex_mask], k=1)
        nearest_global = cluster_global_indices[idx]
        membership[vertex_mask] = np.isin(nearest_global, patch_global_indices)
        return membership

    va = vertex_cluster_ids[mesh_edges[:, 0]]
    vb = vertex_cluster_ids[mesh_edges[:, 1]]

    scores: dict[tuple[int, int], float] = {}
    for (cid_lo, cid_hi), (patch_lo, patch_hi) in boundary_patch_by_pair.items():
        member_lo = _patch_membership(cid_lo, patch_lo)
        member_hi = _patch_membership(cid_hi, patch_hi)
        endpoint0_member_lo = member_lo[mesh_edges[:, 0]]
        endpoint1_member_lo = member_lo[mesh_edges[:, 1]]
        endpoint0_member_hi = member_hi[mesh_edges[:, 0]]
        endpoint1_member_hi = member_hi[mesh_edges[:, 1]]

        incident_lo = int(
            ((va == cid_lo) & endpoint0_member_lo).sum()
            + ((vb == cid_lo) & endpoint1_member_lo).sum()
        )
        incident_hi = int(
            ((va == cid_hi) & endpoint0_member_hi).sum()
            + ((vb == cid_hi) & endpoint1_member_hi).sum()
        )

        cross_01 = (va == cid_lo) & (vb == cid_hi) & endpoint0_member_lo & endpoint1_member_hi
        cross_10 = (va == cid_hi) & (vb == cid_lo) & endpoint0_member_hi & endpoint1_member_lo
        cross_count = int(cross_01.sum() + cross_10.sum())

        if incident_lo == 0 and incident_hi == 0 and cross_count == 0:
            continue
        denom = max(min(incident_lo, incident_hi), 1)
        scores[(cid_lo, cid_hi)] = cross_count / denom

    return scores


def _scores_dict_to_matrix(scores: dict[tuple[int, int], float], valid_ids: list[int]) -> np.ndarray:
    sorted_ids = sorted(valid_ids)
    id_to_pos = {cid: i for i, cid in enumerate(sorted_ids)}
    n = len(sorted_ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    for (cid_a, cid_b), score in scores.items():
        pa, pb = id_to_pos[cid_a], id_to_pos[cid_b]
        matrix[pa, pb] = score
        matrix[pb, pa] = score
    return matrix


def _core_side_visible_fraction(
    frame_data: FrameRenderData,
    core_global_indices: np.ndarray,
    side_cluster_id: int,
    cfg: MeshGraphConfig,
    width: int,
    height: int,
) -> float:
    if core_global_indices.size == 0:
        return 0.0
    device = frame_data.w2c.device
    points = torch.from_numpy(frame_data.posed_means_all_fg[core_global_indices]).to(
        device=device, dtype=torch.float64
    )
    w2c64 = frame_data.w2c.to(dtype=torch.float64)
    intrinsic64 = frame_data.intrinsic.to(dtype=torch.float64)
    pixels, in_bounds = _project_world_points(points, w2c64, intrinsic64, width, height)

    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    camera_points = (w2c64 @ homogeneous.T).T[:, :3]
    z_g = camera_points[:, 2].detach().cpu().numpy()

    px = np.clip(np.floor(pixels[:, 0]).astype(np.int64), 0, width - 1)
    py = np.clip(np.floor(pixels[:, 1]).astype(np.int64), 0, height - 1)

    rendered_depth = frame_data.depth_map[py, px]
    rendered_alpha = frame_data.alpha_map[py, px]
    rendered_label = frame_data.vertex_cluster_ids_grid[py, px]

    visible = (
        in_bounds
        & (rendered_alpha > cfg.mask_alpha_threshold)
        & (np.abs(z_g - rendered_depth) <= cfg.visibility_depth_tol * np.maximum(rendered_depth, 1e-8))
        & (rendered_label == side_cluster_id)
    )
    return float(visible.sum()) / float(core_global_indices.shape[0])


def compute_contact_core_visibility(
    frame_data: FrameRenderData,
    contact_core_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    cfg: MeshGraphConfig,
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Per pair, per side: fraction of that side's CONTACT-CORE Gaussians (not
    the wider boundary patch -- the core is the actual touching region, so
    that's what determines whether this frame's evidence can be trusted;
    see module docstring) that are visible this frame. A core Gaussian
    counts as visible only if it is both depth-consistent with, AND the
    rendered dominant-cluster owner of, the pixel it projects to -- a pure
    depth match can't tell "this Gaussian" from "a different, similarly-deep
    cluster painted this pixel", which matters most right at a contact seam.
    """
    height, width = frame_data.alpha_map.shape
    result: dict[tuple[int, int], tuple[float, float]] = {}
    for (cid_lo, cid_hi), (core_lo, core_hi) in contact_core_by_pair.items():
        frac_lo = _core_side_visible_fraction(frame_data, core_lo, cid_lo, cfg, width, height)
        frac_hi = _core_side_visible_fraction(frame_data, core_hi, cid_hi, cfg, width, height)
        result[(cid_lo, cid_hi)] = (frac_lo, frac_hi)
    return result


# ---------------------------------------------------------------------------
# Cross-frame aggregation (persistence)
# ---------------------------------------------------------------------------


def _empty_edge(cluster_a: int, cluster_b: int, reason: str, num_sampled_frames: int) -> MeshClusterPairEdge:
    empty = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    return MeshClusterPairEdge(
        cluster_a=cluster_a, cluster_b=cluster_b, kept=False, reason=reason,
        persistence=0.0, num_connected_frames=0, num_known_frames=0, num_unknown_frames=0,
        num_sampled_frames=num_sampled_frames, mean_score=0.0, max_score=0.0,
        connected_frame_indices=[], unknown_frame_indices=[],
        num_boundary_a=0, num_boundary_b=0,
        boundary_global_indices_a=empty, boundary_global_indices_b=empty,
        contact_core_global_indices_a=empty, contact_core_global_indices_b=empty,
        boundary_patch_global_indices_a=empty, boundary_patch_global_indices_b=empty,
        boundary_patch_weight_a=empty_f, boundary_patch_weight_b=empty_f,
        boundary_patch_local_scale_a=empty_f, boundary_patch_local_scale_b=empty_f,
        contact_reference_distance=0.0,
    )


def build_mesh_cluster_graph(
    contact_core_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    boundary_patch_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    boundary_patch_weight_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    boundary_patch_local_scale_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    contact_ref_distances_by_pair: dict[tuple[int, int], list[float]],
    scores_by_pair: dict[tuple[int, int], list[float]],
    connected_frames_by_pair: dict[tuple[int, int], list[int]],
    known_frames_by_pair: dict[tuple[int, int], list[int]],
    unknown_frames_by_pair: dict[tuple[int, int], list[int]],
    dropped_reasons: dict[tuple[int, int], str],
    valid_ids: list[int],
    num_sampled_frames: int,
    cfg: MeshGraphConfig,
) -> MeshClusterGraphResult:
    edges: list[MeshClusterPairEdge] = []

    for (cluster_a, cluster_b), reason in dropped_reasons.items():
        edges.append(_empty_edge(cluster_a, cluster_b, reason, num_sampled_frames))

    for (cluster_a, cluster_b), (core_a, core_b) in contact_core_by_pair.items():
        patch_a, patch_b = boundary_patch_by_pair[(cluster_a, cluster_b)]
        weight_a, weight_b = boundary_patch_weight_by_pair[(cluster_a, cluster_b)]
        local_scale_a, local_scale_b = boundary_patch_local_scale_by_pair[(cluster_a, cluster_b)]
        scores = scores_by_pair.get((cluster_a, cluster_b), [])
        connected = connected_frames_by_pair.get((cluster_a, cluster_b), [])
        known = known_frames_by_pair.get((cluster_a, cluster_b), [])
        unknown = unknown_frames_by_pair.get((cluster_a, cluster_b), [])
        ref_distances = contact_ref_distances_by_pair.get((cluster_a, cluster_b), [])
        num_known = len(known)
        persistence = len(connected) / max(num_known, 1)

        if num_known < cfg.min_known_frames:
            kept = False
            reason = "cut_insufficient_known_frames"
        else:
            kept = persistence > cfg.tau_p
            reason = "kept_persistence" if kept else "cut_persistence_below_threshold"

        edges.append(
            MeshClusterPairEdge(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                kept=kept,
                reason=reason,
                persistence=persistence,
                num_connected_frames=len(connected),
                num_known_frames=num_known,
                num_unknown_frames=len(unknown),
                num_sampled_frames=num_sampled_frames,
                mean_score=float(np.mean(scores)) if scores else 0.0,
                max_score=float(np.max(scores)) if scores else 0.0,
                connected_frame_indices=sorted(connected),
                unknown_frame_indices=sorted(unknown),
                num_boundary_a=int(patch_a.shape[0]),
                num_boundary_b=int(patch_b.shape[0]),
                boundary_global_indices_a=patch_a,
                boundary_global_indices_b=patch_b,
                contact_core_global_indices_a=core_a,
                contact_core_global_indices_b=core_b,
                boundary_patch_global_indices_a=patch_a,
                boundary_patch_global_indices_b=patch_b,
                boundary_patch_weight_a=weight_a,
                boundary_patch_weight_b=weight_b,
                boundary_patch_local_scale_a=local_scale_a,
                boundary_patch_local_scale_b=local_scale_b,
                contact_reference_distance=float(np.median(ref_distances)) if ref_distances else 0.0,
            )
        )

    return MeshClusterGraphResult(
        cluster_ids=sorted(valid_ids),
        edges=edges,
        candidate_pair_count=len(edges),
        num_sampled_frames=num_sampled_frames,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Old-graph comparison
# ---------------------------------------------------------------------------


def load_old_graph_edge_pairs(path: Path) -> set[tuple[int, int]] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pairs: set[tuple[int, int]] = set()
    for e in payload.get("edges_kept", []):
        a, b = int(e["cluster_a"]), int(e["cluster_b"])
        pairs.add((min(a, b), max(a, b)))
    return pairs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def render_mesh_png(
    vertices: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    face_colors: np.ndarray | None = None,
) -> None:
    rng = np.random.default_rng(0)
    if triangles.shape[0] > _MAX_RENDER_FACES:
        selection = np.sort(rng.choice(triangles.shape[0], size=_MAX_RENDER_FACES, replace=False))
        triangles_draw = triangles[selection]
        colors_draw = face_colors[selection] if face_colors is not None else None
    else:
        triangles_draw = triangles
        colors_draw = face_colors

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(vertices[triangles_draw], linewidths=0.0)
    poly.set_facecolor(colors_draw if colors_draw is not None else (0.6, 0.6, 0.6, 1.0))
    ax.add_collection3d(poly)
    _set_axes_equal_3d(ax, vertices)
    ax.view_init(elev=_MESH_VIEW[0], azim=_MESH_VIEW[1])
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def face_cluster_ids_from_vertices(vertex_cluster_ids: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Majority vote of each triangle's 3 vertex cluster ids (ties/all-
    distinct fall back to the first vertex, an arbitrary but harmless choice
    since that only happens exactly at 3-cluster boundary triangles)."""
    c0 = vertex_cluster_ids[triangles[:, 0]]
    c1 = vertex_cluster_ids[triangles[:, 1]]
    c2 = vertex_cluster_ids[triangles[:, 2]]
    face_cluster_ids = c0.copy()
    agree_bc = c1 == c2
    face_cluster_ids[agree_bc] = c1[agree_bc]
    return face_cluster_ids


def save_frame_mesh_outputs(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_cluster_ids: np.ndarray,
    color_by_id: dict[int, tuple[float, float, float, float]],
    frame_index: int,
    meshes_dir: Path,
    colored_meshes_dir: Path,
) -> None:
    stem = f"frame_{frame_index:04d}"

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    meshes_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(meshes_dir / f"{stem}.ply"), mesh)
    render_mesh_png(vertices, triangles, meshes_dir / f"{stem}.png", face_colors=None)

    colored_mesh = o3d.geometry.TriangleMesh(mesh)
    vertex_colors = np.asarray([color_by_id[cid][:3] for cid in vertex_cluster_ids])
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    colored_meshes_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(colored_meshes_dir / f"{stem}.ply"), colored_mesh)

    face_cluster_ids = face_cluster_ids_from_vertices(vertex_cluster_ids, triangles)
    face_colors = np.asarray([color_by_id[cid] for cid in face_cluster_ids])
    render_mesh_png(vertices, triangles, colored_meshes_dir / f"{stem}.png", face_colors=face_colors)


def save_frame_score_matrix(
    score_matrix: np.ndarray,
    valid_ids: list[int],
    frame_index: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_index:04d}"
    np.save(output_dir / f"{stem}.npy", score_matrix)

    sorted_ids = sorted(valid_ids)
    rows = []
    pos_i, pos_j = np.nonzero(np.triu(score_matrix, k=1))
    for pi, pj in zip(pos_i.tolist(), pos_j.tolist()):
        rows.append(
            {
                "cluster_a": sorted_ids[pi],
                "cluster_b": sorted_ids[pj],
                "score": float(score_matrix[pi, pj]),
            }
        )
    write_csv(output_dir / f"{stem}.csv", rows)


def save_persistence_heatmap(
    result: MeshClusterGraphResult,
    valid_ids: list[int],
    output_path: Path,
) -> None:
    sorted_ids = sorted(valid_ids)
    id_to_pos = {cid: i for i, cid in enumerate(sorted_ids)}
    num_clusters = len(sorted_ids)
    matrix = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    for e in result.edges:
        pa, pb = id_to_pos[e.cluster_a], id_to_pos[e.cluster_b]
        matrix[pa, pb] = e.persistence
        matrix[pb, pa] = e.persistence

    fig, ax = plt.subplots(figsize=(max(6, num_clusters * 0.35), max(5, num_clusters * 0.35)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(num_clusters))
    ax.set_yticks(range(num_clusters))
    ax.set_xticklabels(sorted_ids, fontsize=6, rotation=90)
    ax.set_yticklabels(sorted_ids, fontsize=6)
    ax.set_title(f"Cluster-pair persistence (tau_p={result.config.tau_p})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="persistence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_final_edge_graph(
    clusters: list[ClusterInfo],
    result: MeshClusterGraphResult,
    color_by_id: dict[int, tuple[float, float, float, float]],
    output_path: Path,
    max_points_per_cluster: int = 2000,
) -> None:
    rng = np.random.default_rng(0)
    sampled_by_id: dict[int, np.ndarray] = {}
    center_by_id: dict[int, np.ndarray] = {}
    all_points: list[np.ndarray] = []
    for cluster in clusters:
        points = cluster.canonical_points.detach().float().cpu().numpy()
        sampled = _sample_points_for_visualization(points, max_points_per_cluster, rng)
        sampled_by_id[cluster.cluster_id] = sampled
        center_by_id[cluster.cluster_id] = cluster.canonical_center.detach().float().cpu().numpy()
        all_points.append(sampled)
    all_points_arr = np.concatenate(all_points, axis=0)

    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_subplot(111, projection="3d")
    for cluster in clusters:
        cid = cluster.cluster_id
        pts = sampled_by_id[cid]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, alpha=0.12, color=color_by_id[cid], rasterized=True)

    for e in result.cut_edges:
        ca, cb = center_by_id.get(e.cluster_a), center_by_id.get(e.cluster_b)
        if ca is None or cb is None:
            continue
        ax.plot([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]], linestyle=(0, (4, 3)), color=(0.8, 0.1, 0.1), alpha=0.3, linewidth=1.0)
    for e in result.kept_edges:
        ca, cb = center_by_id.get(e.cluster_a), center_by_id.get(e.cluster_b)
        if ca is None or cb is None:
            continue
        linewidth = float(np.interp(e.persistence, [result.config.tau_p, 1.0], [1.0, 3.5]))
        ax.plot([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]], color=(0.10, 0.55, 0.15), alpha=0.9, linewidth=linewidth)

    for cluster in clusters:
        cid = cluster.cluster_id
        center = center_by_id[cid]
        ax.scatter([center[0]], [center[1]], [center[2]], s=18, color="black")
        ax.text(center[0], center[1], center[2], str(cid), fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="round,pad=0.15"))

    _set_axes_equal_3d(ax, all_points_arr)
    ax.set_title(f"Mesh-boundary-persistence graph -- kept {len(result.kept_edges)}, cut {len(result.cut_edges)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _representative_frame_for_pair(
    pair: tuple[int, int],
    connected_frames_by_pair: dict[tuple[int, int], list[int]],
    known_frames_by_pair: dict[tuple[int, int], list[int]],
    sampled_frames: list[int],
    default_frame_index: int,
) -> int:
    """Which sampled frame to render as this pair's contact-patch PNG
    background: prefer a frame the pair actually scored connected in, else
    any frame it was known (visible) in, else fall back to the same default
    frame graph_edges_2d.png uses -- always something that exists in
    `sampled_frames`."""
    for frames in (connected_frames_by_pair.get(pair, []), known_frames_by_pair.get(pair, [])):
        if frames:
            return frames[0]
    return default_frame_index if default_frame_index in sampled_frames else sampled_frames[0]


def render_contact_patch_png(
    model: Any,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    frame_index: int,
    cluster_a: ClusterInfo,
    cluster_b: ClusterInfo,
    core_a: np.ndarray,
    core_b: np.ndarray,
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    output_path: Path,
) -> None:
    """This pair's contact core (red) and boundary patch (orange, core
    excluded) projected onto the ACTUAL rendered RGB frame at
    `frame_index` -- lets a user see, against the real geometry rather than
    an abstract point cloud, whether the geodesic dilation stayed on the
    touching surface or leaked."""
    width, height = image_size
    with torch.no_grad():
        render_output = model.render(
            frame_index, w2c[None], intrinsic[None], image_size,
            use_learned_poses=False,
        )
    rgb = render_output["img"][0].detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8).convert("RGBA")
    draw = ImageDraw.Draw(image)

    dynamic_means = _get_dynamic_fg_means(model, frame_index)

    def _draw_points(global_indices: np.ndarray, color: tuple[int, int, int, int], radius: int) -> None:
        if global_indices.size == 0:
            return
        points = dynamic_means[global_indices]
        pixels, valid = _project_world_points(points, w2c, intrinsic, width, height)
        for x, y in pixels[valid]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    for core, patch in ((core_a, patch_a), (core_b, patch_b)):
        _draw_points(np.setdiff1d(patch, core), (255, 140, 0, 190), 3)
    for core in (core_a, core_b):
        _draw_points(core, (220, 0, 0, 230), 4)

    font = _load_overlay_font(16)
    label = f"cluster {cluster_a.cluster_id}-{cluster_b.cluster_id}  (frame {frame_index})"
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        label_w, label_h = 10 * len(label), 16
    draw.rectangle((6, 6, 6 + label_w + 8, 6 + label_h + 8), fill=(255, 255, 255, 200))
    draw.text((10, 10), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def _edge_linewidth_persistence(persistence: float, tau_p: float) -> float:
    """Thicker for HIGHER persistence (more reliably-touching pair) --
    opposite convention from build_cluster_graph.py's gap-based
    _edge_linewidth (there, smaller gap = thicker), since here bigger
    persistence = better instead of smaller gap = better."""
    ratio = float(np.clip((persistence - tau_p) / max(1.0 - tau_p, 1e-6), 0.0, 1.0))
    return float(np.interp(ratio, [0.0, 1.0], [1.0, 3.2]))


def _render_graph_overlay_frame_mesh(
    model: Any,
    clusters: list[ClusterInfo],
    color_by_id: dict[int, Any],
    frame_index: int,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    kept_edges: list[dict],
    cut_edges: list[dict],
    tau_p: float,
) -> np.ndarray:
    """Same rendered-frame + fixed kept/cut edge overlay as
    build_cluster_graph.py's _render_graph_overlay_frame, just keyed by
    persistence (this module's own edge dicts) instead of gap_summary."""
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
        lw = _edge_linewidth_persistence(e["persistence"], tau_p)
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
    color_by_id: dict[int, Any],
    kept_edges: list[dict],
    cut_edges: list[dict],
    tau_p: float,
    output_path: Path,
    frame_index: int = 0,
) -> Path:
    device = model.fg.params["means"].device
    w2cs = _get_camera_w2cs(model).to(device)
    intrinsics = model.Ks.to(device)
    frame_index = int(np.clip(frame_index, 0, w2cs.shape[0] - 1))

    principal_x = float(intrinsics[frame_index, 0, 2].item())
    principal_y = float(intrinsics[frame_index, 1, 2].item())
    width = max(int(round(principal_x * 2.0)), 2)
    height = max(int(round(principal_y * 2.0)), 2)

    with torch.no_grad():
        frame = _render_graph_overlay_frame_mesh(
            model=model,
            clusters=clusters,
            color_by_id=color_by_id,
            frame_index=frame_index,
            w2c=w2cs[frame_index],
            intrinsic=intrinsics[frame_index],
            image_size=(width, height),
            kept_edges=kept_edges,
            cut_edges=cut_edges,
            tau_p=tau_p,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)
    return output_path


def render_2d_overlay_video(
    model: Any,
    clusters: list[ClusterInfo],
    color_by_id: dict[int, Any],
    kept_edges: list[dict],
    cut_edges: list[dict],
    tau_p: float,
    output_path: Path,
    fps: int = 10,
    frame_stride: int = 1,
) -> Path:
    """Same overlay as render_2d_overlay, but across every rendered frame of
    the sequence instead of a single one (build_cluster_graph.py's own
    render_2d_overlay_video, adapted to persistence-keyed edge dicts)."""
    device = model.fg.params["means"].device
    w2cs = _get_camera_w2cs(model).to(device)
    intrinsics = model.Ks.to(device)
    total_frames = w2cs.shape[0]
    frame_indices = list(range(0, total_frames, max(frame_stride, 1)))

    principal_x = float(intrinsics[0, 0, 2].item())
    principal_y = float(intrinsics[0, 1, 2].item())
    width = max(int(round(principal_x * 2.0)), 2)
    height = max(int(round(principal_y * 2.0)), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), imageio.get_writer(output_path, fps=fps) as writer:
        for output_index, frame_index in enumerate(frame_indices, start=1):
            frame = _render_graph_overlay_frame_mesh(
                model=model,
                clusters=clusters,
                color_by_id=color_by_id,
                frame_index=frame_index,
                w2c=w2cs[frame_index],
                intrinsic=intrinsics[frame_index],
                image_size=(width, height),
                kept_edges=kept_edges,
                cut_edges=cut_edges,
                tau_p=tau_p,
            )
            writer.append_data(frame)
            print(f"[graph video {output_index:03d}/{len(frame_indices):03d}] frame={frame_index:04d}")

    return output_path


def write_summary(
    path: Path,
    result: MeshClusterGraphResult,
    old_graph_path: Path,
    old_pairs: set[tuple[int, int]] | None,
    num_valid_clusters: int,
) -> None:
    cfg = result.config
    lines = [
        "Mesh-boundary-persistence cluster graph -- summary",
        "=" * 72,
        f"valid clusters  : {num_valid_clusters}",
        f"sampled frames  : {result.num_sampled_frames}",
        f"frame_interval={cfg.frame_interval} tau_o={cfg.tau_o} "
        f"mask_alpha_threshold={cfg.mask_alpha_threshold} depth_jump_ratio={cfg.depth_jump_ratio} "
        f"tau_s={cfg.tau_s} tau_p={cfg.tau_p} min_cluster_size={cfg.min_cluster_size}",
        f"seed_knn_k={cfg.seed_knn_k} seed_vote_min_fraction={cfg.seed_vote_min_fraction} "
        f"graph_knn_k={cfg.graph_knn_k} graph_edge_length_ratio={cfg.graph_edge_length_ratio}",
        f"visibility_depth_tol={cfg.visibility_depth_tol} "
        f"visibility_fraction_threshold={cfg.visibility_fraction_threshold} "
        f"min_contact_frames={cfg.min_contact_frames} min_known_frames={cfg.min_known_frames}",
        f"candidate pairs : {result.candidate_pair_count}",
        f"kept edges      : {len(result.kept_edges)}",
        f"cut edges       : {len(result.cut_edges)}",
        "",
        "Kept edges (cluster_a-cluster_b  persistence  known/unknown  mean_score  max_score  "
        "boundary_a/b  contact_ref_dist  frames):",
    ]
    for e in sorted(result.kept_edges, key=lambda e: -e.persistence):
        lines.append(
            f"  {e.cluster_a:3d}-{e.cluster_b:<3d} persistence={e.persistence:.3f} "
            f"known={e.num_known_frames} unknown={e.num_unknown_frames} "
            f"mean_score={e.mean_score:.4f} max_score={e.max_score:.4f} "
            f"boundary=({e.num_boundary_a},{e.num_boundary_b}) "
            f"contact_ref_dist={e.contact_reference_distance:.4f} frames={e.connected_frame_indices}"
        )
    lines.append("")
    lines.append("Cut candidate edges (reason, persistence, known/unknown frames):")
    for e in sorted(result.cut_edges, key=lambda e: -e.persistence):
        lines.append(
            f"  {e.cluster_a:3d}-{e.cluster_b:<3d} reason={e.reason} persistence={e.persistence:.3f} "
            f"known={e.num_known_frames} unknown={e.num_unknown_frames} "
            f"mean_score={e.mean_score:.4f} max_score={e.max_score:.4f} frames={e.connected_frame_indices}"
        )
    lines.append("")

    if old_pairs is None:
        lines.append(f"No old distance-based graph found at {old_graph_path} -- skipping comparison.")
    else:
        new_pairs = {(e.cluster_a, e.cluster_b) for e in result.kept_edges}
        common = sorted(new_pairs & old_pairs)
        added = sorted(new_pairs - old_pairs)
        removed = sorted(old_pairs - new_pairs)
        lines.append(f"Comparison with old distance-based graph ({old_graph_path}):")
        lines.append(f"  old kept edges        : {len(old_pairs)}")
        lines.append(f"  new kept edges        : {len(new_pairs)}")
        lines.append(f"  common                : {len(common)} -> {common}")
        lines.append(f"  added (mesh only)     : {len(added)} -> {added}")
        lines.append(f"  removed (distance only): {len(removed)} -> {removed}")

    path.write_text("\n".join(str(l) for l in lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# edges.pt / contact_core.pt / boundary_patch.pt payloads
# ---------------------------------------------------------------------------


def _edge_to_dict(e: MeshClusterPairEdge) -> dict[str, Any]:
    return {
        "cluster_a": int(e.cluster_a),
        "cluster_b": int(e.cluster_b),
        "kept": bool(e.kept),
        "reason": e.reason,
        "persistence": float(e.persistence),
        "num_connected_frames": int(e.num_connected_frames),
        "num_known_frames": int(e.num_known_frames),
        "num_unknown_frames": int(e.num_unknown_frames),
        "num_sampled_frames": int(e.num_sampled_frames),
        "mean_score": float(e.mean_score),
        "max_score": float(e.max_score),
        "connected_frame_indices": list(e.connected_frame_indices),
        "unknown_frame_indices": list(e.unknown_frame_indices),
        "num_boundary_a": int(e.num_boundary_a),
        "num_boundary_b": int(e.num_boundary_b),
        "boundary_global_indices_a": torch.from_numpy(e.boundary_global_indices_a).long(),
        "boundary_global_indices_b": torch.from_numpy(e.boundary_global_indices_b).long(),
        "contact_core_global_indices_a": torch.from_numpy(e.contact_core_global_indices_a).long(),
        "contact_core_global_indices_b": torch.from_numpy(e.contact_core_global_indices_b).long(),
        "boundary_patch_global_indices_a": torch.from_numpy(e.boundary_patch_global_indices_a).long(),
        "boundary_patch_global_indices_b": torch.from_numpy(e.boundary_patch_global_indices_b).long(),
        "boundary_patch_weight_a": torch.from_numpy(e.boundary_patch_weight_a).float(),
        "boundary_patch_weight_b": torch.from_numpy(e.boundary_patch_weight_b).float(),
        "boundary_patch_local_scale_a": torch.from_numpy(e.boundary_patch_local_scale_a).float(),
        "boundary_patch_local_scale_b": torch.from_numpy(e.boundary_patch_local_scale_b).float(),
        "contact_reference_distance": float(e.contact_reference_distance),
    }


def _edge_to_csv_row(e: MeshClusterPairEdge) -> dict[str, Any]:
    row = _edge_to_dict(e)
    for key in (
        "boundary_global_indices_a",
        "boundary_global_indices_b",
        "contact_core_global_indices_a",
        "contact_core_global_indices_b",
        "boundary_patch_global_indices_a",
        "boundary_patch_global_indices_b",
        "boundary_patch_weight_a",
        "boundary_patch_weight_b",
        "boundary_patch_local_scale_a",
        "boundary_patch_local_scale_b",
    ):
        row.pop(key)
    row["connected_frame_indices"] = " ".join(str(f) for f in e.connected_frame_indices)
    row["unknown_frame_indices"] = " ".join(str(f) for f in e.unknown_frame_indices)
    return row


def build_pairs_payload(
    data_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    meta: dict[str, Any],
    weight_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
    local_scale_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    pairs = []
    for (cluster_a, cluster_b), (indices_a, indices_b) in sorted(data_by_pair.items()):
        entry = {
            "cluster_a": int(cluster_a),
            "cluster_b": int(cluster_b),
            "global_indices_a": torch.from_numpy(indices_a).long(),
            "global_indices_b": torch.from_numpy(indices_b).long(),
        }
        if weight_by_pair is not None:
            weight_a, weight_b = weight_by_pair[(cluster_a, cluster_b)]
            entry["weight_a"] = torch.from_numpy(weight_a).float()
            entry["weight_b"] = torch.from_numpy(weight_b).float()
        if local_scale_by_pair is not None:
            local_scale_a, local_scale_b = local_scale_by_pair[(cluster_a, cluster_b)]
            entry["local_scale_a"] = torch.from_numpy(local_scale_a).float()
            entry["local_scale_b"] = torch.from_numpy(local_scale_b).float()
        pairs.append(entry)
    return {"pairs": pairs, "meta": meta}


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
        help="build_cluster_graph.py's edges.pt to compare against in summary.txt. "
        "Default: <work-dir>/analysis/cluster_graph/edges.pt if it exists.",
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
        "--seed-knn-k", type=int, default=3,
        help="k (1-3) nearest real Gaussians of a cross-cluster boundary vertex's own assigned "
        "cluster to treat as this frame's contact 'seeds' for that side -- the raw material of "
        "the multi-frame seed-voting step (see module docstring).",
    )
    parser.add_argument(
        "--seed-vote-min-fraction", type=float, default=0.3,
        help="A Gaussian joins the stable 'contact core' once it was picked as a seed in at "
        "least this fraction of the frames where its pair had any cross-cluster contact.",
    )
    parser.add_argument(
        "--graph-knn-k", type=int, default=8,
        help="k for the same-cluster Gaussian mutual-kNN graph that geodesic dilation walks "
        "over (candidate k before the mutual-edge filter).",
    )
    parser.add_argument(
        "--graph-edge-length-ratio", type=float, default=3.0,
        help="An edge in the same-cluster Gaussian graph is pruned if its length exceeds this "
        "multiple of the larger endpoint's own local neighbor spacing -- keeps dilation from "
        "leaking onto a different, unconnected part of the surface. Reasonable range 2.0-3.0.",
    )
    parser.add_argument(
        "--visibility-depth-tol", type=float, default=0.05,
        help="A contact-core Gaussian is depth-consistent with its rendered pixel if "
        "|its own camera-space depth - the pixel's rendered depth| is within this fraction of "
        "the rendered depth (same relative-tolerance convention as --depth-jump-ratio).",
    )
    parser.add_argument(
        "--visibility-fraction-threshold", type=float, default=0.2,
        help="A frame is 'known' for a pair only if at least this fraction of BOTH sides' "
        "contact-core Gaussians are visible this frame; otherwise the frame is 'unknown' and "
        "excluded from the persistence denominator.",
    )
    parser.add_argument(
        "--min-contact-frames", type=int, default=3,
        help="A pair needs cross-cluster mesh contact in at least this many sampled frames "
        "before a contact core is even attempted -- stops a pair glimpsed in one or two frames "
        "from producing a 'stable' core. Applied alongside (not instead of) a fixed temporal-"
        "stability check (_has_temporal_contact_stability): the raw count alone isn't enough "
        "once every frame is sampled by default, since scattered single-frame noise can clear "
        "a low count without ever being a real, temporally-coherent contact.",
    )
    parser.add_argument(
        "--min-known-frames", type=int, default=3,
        help="A pair needs at least this many KNOWN (visible, judged) frames to be eligible to "
        "be kept -- also the floor that rejects a pair that was occluded the whole sequence.",
    )
    parser.add_argument(
        "--tau-s", type=float, default=0.01,
        help="Per-frame cross-cluster boundary score threshold (fraction of the smaller side's "
        "own PATCH-restricted mesh-edge budget spent bordering the other cluster -- both the "
        "numerator and denominator changed scale from a whole-cluster-area normalization to a "
        "boundary-patch-local one; retune from a real checkpoint's figures/score_matrices before "
        "trusting the default).",
    )
    parser.add_argument(
        "--tau-p", type=float, default=0.95,
        help="Persistence threshold: fraction of KNOWN sampled frames (unknown/occluded frames "
        "excluded from the denominator) a pair must be connected in (score > tau_s) to be kept "
        "as a final edge.",
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
    if not 1 <= args.seed_knn_k <= 3:
        raise ValueError("--seed-knn-k must be in [1, 3]")
    if not 0.0 < args.seed_vote_min_fraction <= 1.0:
        raise ValueError("--seed-vote-min-fraction must be in (0, 1]")
    if args.graph_knn_k < 1:
        raise ValueError("--graph-knn-k must be >= 1")
    if args.graph_edge_length_ratio <= 0.0:
        raise ValueError("--graph-edge-length-ratio must be > 0")
    if args.visibility_depth_tol <= 0.0:
        raise ValueError("--visibility-depth-tol must be > 0")
    if not 0.0 < args.visibility_fraction_threshold <= 1.0:
        raise ValueError("--visibility-fraction-threshold must be in (0, 1]")
    if args.min_contact_frames < 1:
        raise ValueError("--min-contact-frames must be >= 1")
    if args.min_known_frames < 1:
        raise ValueError("--min-known-frames must be >= 1")
    if args.tau_s <= 0.0:
        raise ValueError("--tau-s must be > 0")
    if not 0.0 < args.tau_p < 1.0:
        raise ValueError("--tau-p must be in (0, 1)")
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
        else work_dir / "analysis" / "cluster_graph_mesh"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    colored_meshes_dir = output_dir / "colored_meshes"
    figures_dir = output_dir / "figures"
    score_matrices_dir = figures_dir / "score_matrices"
    contact_patches_dir = figures_dir / "contact_patches"

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
    clusters_by_id = {c.cluster_id: c for c in clusters}
    color_by_id = build_cluster_colormap(valid_ids)

    cfg = MeshGraphConfig(
        tau_o=args.tau_o,
        mask_alpha_threshold=args.mask_alpha_threshold,
        depth_jump_ratio=args.depth_jump_ratio,
        tau_s=args.tau_s,
        tau_p=args.tau_p,
        frame_interval=args.frame_interval,
        min_cluster_size=args.min_cluster_size,
        seed_knn_k=args.seed_knn_k,
        seed_vote_min_fraction=args.seed_vote_min_fraction,
        graph_knn_k=args.graph_knn_k,
        graph_edge_length_ratio=args.graph_edge_length_ratio,
        visibility_depth_tol=args.visibility_depth_tol,
        visibility_fraction_threshold=args.visibility_fraction_threshold,
        min_contact_frames=args.min_contact_frames,
        min_known_frames=args.min_known_frames,
    )

    device = model.fg.params["means"].device
    w2cs = _get_camera_w2cs(model).to(device)
    intrinsics = model.Ks.to(device)
    principal_x = float(intrinsics[0, 0, 2].item())
    principal_y = float(intrinsics[0, 1, 2].item())
    image_size = (max(int(round(principal_x * 2.0)), 2), max(int(round(principal_y * 2.0)), 2))

    num_frames = model.num_frames
    sampled_frames = list(range(0, num_frames, cfg.frame_interval))
    print(f"Valid clusters  : {len(valid_ids)}")
    print(f"Frames          : {num_frames} (sampling {len(sampled_frames)} at interval {cfg.frame_interval})")

    # ---- Pass 1: per-frame mesh + cross-cluster seed collection ----
    vote_counts: dict[tuple[int, int], tuple[dict[int, int], dict[int, int]]] = {}
    contact_frame_indices_by_pair: dict[tuple[int, int], list[int]] = {}
    for i, frame_index in enumerate(sampled_frames, start=1):
        frame_data = render_frame_mesh(
            model, valid_ids, frame_index, w2cs[frame_index], intrinsics[frame_index], image_size, cfg
        )
        seeds = collect_frame_seeds(
            frame_data.vertices, frame_data.vertex_cluster_ids, frame_data.mesh_edges,
            frame_data.gaussian_positions, frame_data.gaussian_cluster_ids, frame_data.gaussian_global_indices,
            valid_ids, cfg.seed_knn_k,
        )
        for pair, (seed_lo, seed_hi) in seeds.items():
            contact_frame_indices_by_pair.setdefault(pair, []).append(frame_index)
            votes_lo, votes_hi = vote_counts.setdefault(pair, ({}, {}))
            for g in seed_lo.tolist():
                votes_lo[g] = votes_lo.get(g, 0) + 1
            for g in seed_hi.tolist():
                votes_hi[g] = votes_hi.get(g, 0) + 1
        print(
            f"[pass1 seeds {i:03d}/{len(sampled_frames):03d}] t={frame_index:04d} "
            f"verts={frame_data.vertices.shape[0]} cross_pairs={len(seeds)}"
        )

    # ---- Between passes: contact core + geodesic boundary patch ----
    dropped_reasons: dict[tuple[int, int], str] = {}
    contact_core_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    boundary_patch_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    boundary_patch_weight_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    boundary_patch_local_scale_by_pair: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    graph_cache: dict[int, tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, int]]] = {}

    for pair, frame_indices in contact_frame_indices_by_pair.items():
        n_candidate = len(frame_indices)
        if n_candidate < cfg.min_contact_frames:
            dropped_reasons[pair] = "cut_insufficient_contact_frames"
            continue
        if not _has_temporal_contact_stability(frame_indices):
            dropped_reasons[pair] = "cut_no_temporal_contact_stability"
            continue
        cluster_lo, cluster_hi = pair
        votes_lo, votes_hi = vote_counts[pair]
        core_lo, patch_lo, weight_lo, local_scale_lo = build_contact_core_and_patch_for_side(
            graph_cache, clusters_by_id[cluster_lo], votes_lo, n_candidate, cfg
        )
        core_hi, patch_hi, weight_hi, local_scale_hi = build_contact_core_and_patch_for_side(
            graph_cache, clusters_by_id[cluster_hi], votes_hi, n_candidate, cfg
        )
        if core_lo.size == 0 or core_hi.size == 0:
            dropped_reasons[pair] = "cut_no_stable_contact_core"
            continue
        contact_core_by_pair[pair] = (core_lo, core_hi)
        boundary_patch_by_pair[pair] = (patch_lo, patch_hi)
        boundary_patch_weight_by_pair[pair] = (weight_lo, weight_hi)
        boundary_patch_local_scale_by_pair[pair] = (local_scale_lo, local_scale_hi)

    print()
    print(f"Candidate pairs (>= min-contact-frames, temporally stable) : {len(contact_core_by_pair) + sum(1 for r in dropped_reasons.values() if r == 'cut_no_stable_contact_core')}")
    print(f"Pairs with a stable contact core                            : {len(contact_core_by_pair)}")

    # ---- Pass 2: patch-localized score + contact-core visibility ----
    scores_by_pair: dict[tuple[int, int], list[float]] = {}
    connected_frames_by_pair: dict[tuple[int, int], list[int]] = {}
    known_frames_by_pair: dict[tuple[int, int], list[int]] = {}
    unknown_frames_by_pair: dict[tuple[int, int], list[int]] = {}
    contact_ref_distances_by_pair: dict[tuple[int, int], list[float]] = {}

    for i, frame_index in enumerate(sampled_frames, start=1):
        frame_data = render_frame_mesh(
            model, valid_ids, frame_index, w2cs[frame_index], intrinsics[frame_index], image_size, cfg
        )

        if not args.no_visualization:
            save_frame_mesh_outputs(
                frame_data.vertices, frame_data.triangles, frame_data.vertex_cluster_ids, color_by_id,
                frame_index, meshes_dir, colored_meshes_dir,
            )

        frame_scores = score_frame_against_patches(frame_data, boundary_patch_by_pair)
        frame_visibility = compute_contact_core_visibility(frame_data, contact_core_by_pair, cfg)

        if not args.no_visualization:
            score_matrix = _scores_dict_to_matrix(frame_scores, valid_ids)
            save_frame_score_matrix(score_matrix, valid_ids, frame_index, score_matrices_dir)

        for pair in contact_core_by_pair.keys():
            score = frame_scores.get(pair, 0.0)
            frac_lo, frac_hi = frame_visibility.get(pair, (0.0, 0.0))
            known = (
                frac_lo >= cfg.visibility_fraction_threshold
                and frac_hi >= cfg.visibility_fraction_threshold
            )
            scores_by_pair.setdefault(pair, []).append(score)
            if known:
                known_frames_by_pair.setdefault(pair, []).append(frame_index)
                if score > cfg.tau_s:
                    connected_frames_by_pair.setdefault(pair, []).append(frame_index)
                    patch_a, patch_b = boundary_patch_by_pair[pair]
                    weight_a, weight_b = boundary_patch_weight_by_pair[pair]
                    pos_a = frame_data.posed_means_all_fg[patch_a]
                    pos_b = frame_data.posed_means_all_fg[patch_b]
                    centroid_a = (pos_a * weight_a[:, None]).sum(axis=0) / max(float(weight_a.sum()), 1e-8)
                    centroid_b = (pos_b * weight_b[:, None]).sum(axis=0) / max(float(weight_b.sum()), 1e-8)
                    contact_ref_distances_by_pair.setdefault(pair, []).append(
                        float(np.linalg.norm(centroid_a - centroid_b))
                    )
            else:
                unknown_frames_by_pair.setdefault(pair, []).append(frame_index)

        print(
            f"[pass2 score {i:03d}/{len(sampled_frames):03d}] t={frame_index:04d} "
            f"pairs_scored={len(frame_scores)}"
        )

    result = build_mesh_cluster_graph(
        contact_core_by_pair, boundary_patch_by_pair,
        boundary_patch_weight_by_pair, boundary_patch_local_scale_by_pair,
        contact_ref_distances_by_pair, scores_by_pair,
        connected_frames_by_pair, known_frames_by_pair, unknown_frames_by_pair,
        dropped_reasons, valid_ids, len(sampled_frames), cfg,
    )
    print()
    print(f"Candidate pairs : {result.candidate_pair_count}")
    print(f"Kept edges      : {len(result.kept_edges)}")
    print(f"Cut edges       : {len(result.cut_edges)}")
    for e in sorted(result.edges, key=lambda e: (e.cluster_a, e.cluster_b)):
        tag = "KEEP" if e.kept else "CUT "
        print(
            f"[{tag}] {e.cluster_a}-{e.cluster_b} reason={e.reason} "
            f"persistence={e.persistence:.3f} mean_score={e.mean_score:.4f} "
            f"known={e.num_known_frames} unknown={e.num_unknown_frames}"
        )

    old_graph_path = (
        args.old_graph_path.expanduser().resolve()
        if args.old_graph_path is not None
        else work_dir / "analysis" / "cluster_graph" / "edges.pt"
    )
    old_pairs = load_old_graph_edge_pairs(old_graph_path)

    edges_kept_dicts = [_edge_to_dict(e) for e in result.kept_edges]
    edges_cut_dicts = [_edge_to_dict(e) for e in result.cut_edges]

    payload = {
        "edge_index": torch.from_numpy(result.edge_index()).long(),
        "edges_kept": edges_kept_dicts,
        "edges_cut": edges_cut_dicts,
        "cluster_ids": valid_ids,
        "meta": {
            "method": "mesh_boundary_persistence",
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
            "seed_knn_k": cfg.seed_knn_k,
            "seed_vote_min_fraction": cfg.seed_vote_min_fraction,
            "graph_knn_k": cfg.graph_knn_k,
            "graph_edge_length_ratio": cfg.graph_edge_length_ratio,
            "patch_expansion_ratio": _PATCH_EXPANSION_RATIO,
            "visibility_depth_tol": cfg.visibility_depth_tol,
            "visibility_fraction_threshold": cfg.visibility_fraction_threshold,
            "min_contact_frames": cfg.min_contact_frames,
            "min_known_frames": cfg.min_known_frames,
            "tau_s": cfg.tau_s,
            "tau_p": cfg.tau_p,
            "min_cluster_size": cfg.min_cluster_size,
            "candidate_pair_count": result.candidate_pair_count,
            "kept_edge_count": len(edges_kept_dicts),
            "cut_edge_count": len(edges_cut_dicts),
            "valid_cluster_ids": valid_ids,
            "filtered_cluster_ids": filtered_ids,
            "old_graph_path": str(old_graph_path),
            "old_graph_found": old_pairs is not None,
        },
    }
    torch.save(payload, edges_pt)
    write_csv(output_dir / "edges_kept.csv", [_edge_to_csv_row(e) for e in result.kept_edges])
    write_csv(output_dir / "edges_cut.csv", [_edge_to_csv_row(e) for e in result.cut_edges])

    pairs_meta = {
        "work_dir": str(work_dir),
        "num_pairs": len(contact_core_by_pair),
        "seed_vote_min_fraction": cfg.seed_vote_min_fraction,
        "patch_expansion_ratio": _PATCH_EXPANSION_RATIO,
    }
    torch.save(build_pairs_payload(contact_core_by_pair, pairs_meta), output_dir / "contact_core.pt")
    torch.save(
        build_pairs_payload(
            boundary_patch_by_pair, pairs_meta,
            weight_by_pair=boundary_patch_weight_by_pair,
            local_scale_by_pair=boundary_patch_local_scale_by_pair,
        ),
        output_dir / "boundary_patch.pt",
    )

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["meta"], handle, indent=2, ensure_ascii=False)

    write_summary(output_dir / "summary.txt", result, old_graph_path, old_pairs, len(valid_ids))

    visualization_files: list[str] = []
    if not args.no_visualization:
        save_persistence_heatmap(result, valid_ids, figures_dir / "persistence_heatmap.png")
        render_final_edge_graph(
            clusters, result, color_by_id, figures_dir / "final_edge_graph.png",
            max_points_per_cluster=args.max_points_per_cluster,
        )

        for (cluster_a, cluster_b), (core_a, core_b) in contact_core_by_pair.items():
            patch_a, patch_b = boundary_patch_by_pair[(cluster_a, cluster_b)]
            pair_frame_index = _representative_frame_for_pair(
                (cluster_a, cluster_b), connected_frames_by_pair, known_frames_by_pair,
                sampled_frames, args.frame_index_2d,
            )
            render_contact_patch_png(
                model, w2cs[pair_frame_index], intrinsics[pair_frame_index], image_size, pair_frame_index,
                clusters_by_id[cluster_a], clusters_by_id[cluster_b],
                core_a, core_b, patch_a, patch_b,
                contact_patches_dir / f"pair_{cluster_a}_{cluster_b}.png",
            )
        if contact_core_by_pair:
            print(f"[visualization] saved {len(contact_core_by_pair)} contact-patch figures under {contact_patches_dir}")

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
    print("=" * 72)
    print("Mesh-boundary-persistence cluster graph build complete")
    print(f"Output PT : {edges_pt}")
    print(f"Contact core PT   : {output_dir / 'contact_core.pt'}")
    print(f"Boundary patch PT : {output_dir / 'boundary_patch.pt'}")
    print(f"Summary   : {output_dir / 'summary.txt'}")
    if visualization_files:
        print("Visualizations :")
        for path in visualization_files:
            print(f"  {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
Pure, testable builder for the offline cluster-connectivity graph.

Problem
-------
The canonical-space adjacency in cluster_pairs.py connects two clusters
whenever their boundary Gaussians are close *in canonical pose*. That is
wrong for self-contact poses (a hand resting on a knee, clasped hands): the
canonical pose can have those parts touching even though they move
independently everywhere else, so canonical-only adjacency creates a false
edge (hand<->hand, hand<->torso).

This module fixes that by looking across *all* frames instead of just
canonical: two clusters keep an edge only if their boundary Gaussians stay
close in every frame. If the gap opens up in even one frame, the edge is cut.

Algorithm (see build_cluster_graph)
------------------------------------
1. Cheap candidate prefilter: two clusters are a candidate pair if their
   center-to-center distance drops below a generous radius in *some* frame.
   This also gives us t*, the frame where the centers are closest.
2. At t*, pick each cluster's boundary Gaussians (the ones nearest the other
   cluster, reusing cluster_pairs.select_boundary_local_indices) -- but
   *generously*, not just the handful closest at t*. This fixes *which*
   Gaussians are candidates for the boundary; their identity doesn't change
   across frames, only their position does. (Picking too few here would
   reintroduce the bias this module exists to avoid: if a joint rotates, the
   specific points closest at t* can drift apart from each other while the
   parts stay genuinely touching through some *other* pair of points -- a
   fixed, narrow boundary would miss that pair and read a spurious gap.)
3. For every frame, rebuild a cKDTree from cluster B's candidate points at
   that frame and query cluster A's candidate points against it -- the
   nearest-neighbor *pairing* is recomputed fresh each frame, not carried
   over from t*, so it finds whichever pair is actually closest at that
   moment. Confidence-weighted-smooth that per-frame sequence (a small
   temporal window, weighted by per-point confidence when the position
   source provides one) and summarize with its max -- smoothing first so a
   single noisy/occluded frame can't spike the max on its own, while still
   using the true max (not a percentile) so a sustained separation isn't
   averaged away.
4. Keep the edge iff that (smoothed) max gap stays under contact_distance *
   keep_multiplier for the whole sequence; otherwise cut it.

No I/O, no torch model, no matplotlib -- everything here is plain numpy/scipy
(and the two reused numpy helpers from cluster_pairs.py) so it can be unit
tested with synthetic arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from flow3d.analysis.cluster_pairs import directed_nn, select_boundary_local_indices


@dataclass
class ClusterGraphConfig:
    # Candidate prefilter: pair is a candidate if center distance drops below
    # contact_distance * candidate_radius_multiplier in some frame.
    # contact_distance is a *point-density* scale (within-cluster Gaussian NN
    # spacing), not a body scale, so this ratio needs to be large -- on a
    # calibration checkpoint, genuinely-adjacent cluster centers sat 8x-20x
    # contact_distance apart. Default errs generous so real edges aren't
    # dropped before the (cheap) gap check gets a chance to prune them.
    candidate_radius_multiplier: float = 30.0
    # Kept iff the smoothed, confidence-weighted max all-frames gap <
    # contact_distance * keep_multiplier.
    keep_multiplier: float = 1.5
    # Temporal window (frames) for confidence-weighted smoothing of the
    # per-frame gap sequence before taking its max; see smooth_and_summarize_gap.
    # 1 disables smoothing (still confidence-weighted: zero-confidence frames
    # are excluded from the max where any confident frame exists).
    gap_smoothing_window: int = 5
    # Boundary-point *candidate set* selection at the closest frame (reuses
    # cluster_pairs.py). Deliberately generous -- this is no longer "the
    # boundary pair", just the pool each frame's cKDTree nearest-neighbor
    # query picks from, so it needs to stay wide enough to still contain
    # whichever points are actually closest after a joint rotates. Set high
    # (or equal to a cluster's full size) to search the whole cluster instead
    # of a boundary subsample, at higher per-frame cKDTree cost.
    boundary_fraction: float = 0.10
    boundary_min_gaussians: int = 20
    boundary_max_gaussians: int = 150
    # Distance ceiling used when picking boundary points; defaults to
    # contact_distance * boundary_max_distance_multiplier unless an explicit
    # boundary_max_distance is given.
    boundary_max_distance_multiplier: float = 1.0
    boundary_max_distance: float | None = None


@dataclass
class ClusterPairEdge:
    cluster_a: int
    cluster_b: int
    kept: bool
    reason: str
    frame_t_star: int
    center_distance_t_star: float
    gap_summary: float
    gap_median: float
    gap_min: float
    gap_max: float
    gap_mean_confidence: float
    num_boundary_a: int
    num_boundary_b: int
    boundary_global_indices_a: np.ndarray
    boundary_global_indices_b: np.ndarray
    contact_distance: float
    threshold: float


@dataclass
class ClusterGraphResult:
    cluster_ids: list[int]
    edges: list[ClusterPairEdge]
    candidate_pair_count: int
    contact_distance: float
    config: ClusterGraphConfig = field(default_factory=ClusterGraphConfig)

    @property
    def kept_edges(self) -> list[ClusterPairEdge]:
        return [e for e in self.edges if e.kept]

    @property
    def cut_edges(self) -> list[ClusterPairEdge]:
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


def compute_cluster_center_trajectories(
    cluster_ids: list[int],
    global_indices_by_cluster: dict[int, np.ndarray],
    positions_all_frames: np.ndarray,
) -> np.ndarray:
    """
    :param positions_all_frames: (G, T, 3) per-Gaussian, per-frame positions.
    :return: (C, T, 3) per-cluster center trajectories, C == len(cluster_ids).
    """
    return np.stack(
        [
            positions_all_frames[global_indices_by_cluster[cid]].mean(axis=0)
            for cid in cluster_ids
        ],
        axis=0,
    )


def find_candidate_pairs(
    cluster_ids: list[int],
    centers_ts: np.ndarray,
    radius: float,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, float]]]:
    """
    Cheap prefilter over cluster *centers* only (no boundary/gap work yet): a
    pair is a candidate if its center-to-center distance drops below `radius`
    in at least one frame.

    :param centers_ts: (C, T, 3).
    :return: (candidate_pairs, info) where info[(a, b)] = (t_star, min_dist),
        t_star the frame of closest approach and min_dist the center distance
        there -- reused as the boundary-selection frame downstream.
    """
    diff = centers_ts[:, None, :, :] - centers_ts[None, :, :, :]  # (C, C, T, 3)
    dist_t = np.linalg.norm(diff, axis=-1)  # (C, C, T)
    t_star = np.argmin(dist_t, axis=-1)  # (C, C)
    min_dist = np.take_along_axis(dist_t, t_star[..., None], axis=-1)[..., 0]  # (C, C)

    pairs: list[tuple[int, int]] = []
    info: dict[tuple[int, int], tuple[int, float]] = {}
    C = len(cluster_ids)
    for i in range(C):
        for j in range(i + 1, C):
            if min_dist[i, j] < radius:
                a, b = cluster_ids[i], cluster_ids[j]
                pairs.append((a, b))
                info[(a, b)] = (int(t_star[i, j]), float(min_dist[i, j]))
    return pairs, info


def select_boundary_at_frame(
    global_indices_a: np.ndarray,
    global_indices_b: np.ndarray,
    positions_all_frames: np.ndarray,
    frame_index: int,
    boundary_fraction: float,
    boundary_min: int,
    boundary_max: int,
    boundary_max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick each cluster's boundary-candidate Gaussians (nearest the other
    cluster) at a fixed frame, and return their *global* indices. Only the
    *identities* are fixed across frames -- which pair among them is closest
    is re-derived every frame in compute_all_frames_gap, not carried over
    from this frame. Kept generous (see ClusterGraphConfig.boundary_max_gaussians)
    so a rotation that shifts which points are closest still finds them here.
    """
    points_a = positions_all_frames[global_indices_a, frame_index]  # (Ga, 3)
    points_b = positions_all_frames[global_indices_b, frame_index]  # (Gb, 3)

    distance_ab, _ = directed_nn(points_a, points_b)
    distance_ba, _ = directed_nn(points_b, points_a)

    local_a = select_boundary_local_indices(
        distances_to_other=distance_ab,
        cluster_size=len(global_indices_a),
        fraction=boundary_fraction,
        minimum=boundary_min,
        maximum=boundary_max,
        max_distance=boundary_max_distance,
    )
    local_b = select_boundary_local_indices(
        distances_to_other=distance_ba,
        cluster_size=len(global_indices_b),
        fraction=boundary_fraction,
        minimum=boundary_min,
        maximum=boundary_max,
        max_distance=boundary_max_distance,
    )
    return global_indices_a[local_a], global_indices_b[local_b]


def compute_all_frames_gap(
    boundary_global_indices_a: np.ndarray,
    boundary_global_indices_b: np.ndarray,
    positions_all_frames: np.ndarray,
    confidences_all_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame nearest-neighbor gap between two (generous) candidate point
    sets -- recomputed fresh via cKDTree at *every* frame, not carried over
    from whichever pair was closest at a single reference frame.

    A fixed reference-frame pairing biases the gap upward: if a joint
    rotates (a wrist bends), the specific points closest at the reference
    frame drift apart from each other even while the parts stay genuinely
    touching through some other pair -- inflating the measured gap for a
    part that never actually separated. Rebuilding the nearest-neighbor
    query every frame from the full candidate set finds whichever pair is
    truly closest *at that frame*, so a real, sustained separation still
    reads as separated, but a rotation that keeps some pair of points
    touching no longer reads as one.

    :param confidences_all_frames: (G, T) per-Gaussian, per-frame confidence
        in [0, 1] (e.g. raw-track confidence), aligned with positions_all_frames.
        None (e.g. positions from a learned motion basis, which has no
        natural per-frame confidence) is treated as uniform confidence 1.0.
    :return: (gap_t, confidence_t), both (T,). confidence_t is the weaker
        (min) of the two points that achieved that frame's minimum gap.
    """
    pts_a = positions_all_frames[boundary_global_indices_a]  # (m, T, 3)
    pts_b = positions_all_frames[boundary_global_indices_b]  # (n, T, 3)
    T = pts_a.shape[1]

    gap_t = np.empty(T, dtype=np.float64)
    confidence_t = np.empty(T, dtype=np.float64)
    for t in range(T):
        tree = cKDTree(pts_b[:, t, :])
        dist, idx = tree.query(pts_a[:, t, :], k=1)
        a_star = int(np.argmin(dist))
        b_star = int(idx[a_star])
        gap_t[t] = dist[a_star]

        if confidences_all_frames is None:
            confidence_t[t] = 1.0
        else:
            conf_a = confidences_all_frames[boundary_global_indices_a[a_star], t]
            conf_b = confidences_all_frames[boundary_global_indices_b[b_star], t]
            confidence_t[t] = min(float(conf_a), float(conf_b))

    return gap_t, confidence_t


def smooth_and_summarize_gap(
    gap_t: np.ndarray,
    confidence_t: np.ndarray,
    window: int = 5,
) -> float:
    """
    Confidence-weighted temporal smoothing of the per-frame gap sequence,
    then take the max of the smoothed sequence as the summary statistic.

    A plain max over raw per-frame gaps is fragile: one occluded/jittery
    frame can spike the gap and cut an edge that's actually attached the
    whole sequence. Smoothing first (weighted by confidence, so untrustworthy
    frames are downweighted rather than allowed to dominate their window)
    tempers isolated spikes while still letting a *sustained* separation --
    several consecutive frames genuinely apart -- carry through to the max,
    unlike a percentile which would just discount it as an outlier.

    :param window: frames per smoothing window (odd covers symmetrically).
        <= 1 skips windowed smoothing but still applies confidence gating:
        zero-confidence frames are excluded from the max whenever at least
        one confident frame exists.
    """
    T = len(gap_t)
    if T == 0:
        return 0.0

    if window <= 1 or T <= 1:
        trustworthy = confidence_t > 0
        return float(gap_t[trustworthy].max()) if trustworthy.any() else float(gap_t.max())

    half = window // 2
    smoothed = np.empty(T, dtype=np.float64)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        w = confidence_t[lo:hi]
        g = gap_t[lo:hi]
        wsum = float(w.sum())
        smoothed[t] = float((g * w).sum() / wsum) if wsum > 0 else float(g.mean())

    return float(smoothed.max())


def build_cluster_graph(
    cluster_ids: list[int],
    global_indices_by_cluster: dict[int, np.ndarray],
    positions_all_frames: np.ndarray,
    contact_distance: float,
    config: ClusterGraphConfig | None = None,
    confidences_all_frames: np.ndarray | None = None,
) -> ClusterGraphResult:
    """
    Build the offline cluster-connectivity graph.

    :param cluster_ids: valid cluster ids.
    :param global_indices_by_cluster: cluster_id -> (Gc,) int64 indices into
        the first axis of positions_all_frames.
    :param positions_all_frames: (G, T, 3) per-Gaussian, per-frame positions
        (e.g. from model.compute_poses_fg(torch.arange(num_frames))).
    :param contact_distance: canonical within-cluster spacing-derived contact
        threshold (see cluster_pairs.estimate_within_cluster_spacing).
    :param config: thresholds/knobs; see ClusterGraphConfig.
    :param confidences_all_frames: (G, T) per-Gaussian, per-frame confidence
        in [0, 1], aligned with positions_all_frames. None (e.g. positions
        from a learned motion basis) is treated as uniform confidence 1.0 --
        the gap summary is then plain confidence-*un*weighted smoothing.
    """
    config = config or ClusterGraphConfig()

    boundary_max_distance = (
        config.boundary_max_distance
        if config.boundary_max_distance is not None
        else contact_distance * config.boundary_max_distance_multiplier
    )
    threshold = contact_distance * config.keep_multiplier
    radius = contact_distance * config.candidate_radius_multiplier

    centers_ts = compute_cluster_center_trajectories(
        cluster_ids, global_indices_by_cluster, positions_all_frames
    )
    candidate_pairs, candidate_info = find_candidate_pairs(cluster_ids, centers_ts, radius)

    edges: list[ClusterPairEdge] = []
    for a, b in candidate_pairs:
        t_star, center_distance = candidate_info[(a, b)]
        global_a = global_indices_by_cluster[a]
        global_b = global_indices_by_cluster[b]

        boundary_a, boundary_b = select_boundary_at_frame(
            global_a,
            global_b,
            positions_all_frames,
            t_star,
            boundary_fraction=config.boundary_fraction,
            boundary_min=config.boundary_min_gaussians,
            boundary_max=config.boundary_max_gaussians,
            boundary_max_distance=boundary_max_distance,
        )

        gap_t, confidence_t = compute_all_frames_gap(
            boundary_a, boundary_b, positions_all_frames, confidences_all_frames,
        )
        gap_summary = smooth_and_summarize_gap(gap_t, confidence_t, config.gap_smoothing_window)
        kept = gap_summary < threshold

        edges.append(
            ClusterPairEdge(
                cluster_a=a,
                cluster_b=b,
                kept=kept,
                reason="kept_all_frames_contact" if kept else "cut_gap_exceeds_threshold",
                frame_t_star=t_star,
                center_distance_t_star=center_distance,
                gap_summary=gap_summary,
                gap_median=float(np.median(gap_t)),
                gap_min=float(gap_t.min()),
                gap_max=float(gap_t.max()),
                gap_mean_confidence=float(confidence_t.mean()),
                num_boundary_a=len(boundary_a),
                num_boundary_b=len(boundary_b),
                boundary_global_indices_a=boundary_a,
                boundary_global_indices_b=boundary_b,
                contact_distance=contact_distance,
                threshold=threshold,
            )
        )

    return ClusterGraphResult(
        cluster_ids=list(cluster_ids),
        edges=edges,
        candidate_pair_count=len(candidate_pairs),
        contact_distance=contact_distance,
        config=config,
    )

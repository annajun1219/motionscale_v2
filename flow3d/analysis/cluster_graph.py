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
    # Sanity gate, independent of the boundary-gap check below: a pair is cut
    # outright if its cluster CENTERS' median distance across the *whole*
    # sequence is more than this many times their 10th-percentile (not
    # single-frame min) distance. Relative (not an absolute distance), so it
    # doesn't need recalibrating per checkpoint/scene scale.
    #
    # Originally compared against the single closest-approach FRAME (min),
    # but that's a one-frame statistic and articulated real neighbors (a
    # joint that keeps rotating, e.g. hand<->forearm) can have their
    # *centroids* swing close together at one arbitrary pose purely by
    # chance, even while the boundary stays touching every frame -- that
    # single lucky frame made the ratio blow up and cut pairs that were
    # genuinely, consistently attached (confirmed: gap_summary far under
    # threshold at pairs the min-based ratio cut). p10 is the closest-
    # approach *regime* instead of one frame, so it isn't hijacked by a
    # single outlier pose. A pair that stays close throughout has median
    # close to its p10, so a small ratio: kept (subject to the gap check
    # below). A pair whose median is many times its p10 spends most of the
    # clip far from where it typically gets closest -- two unrelated parts
    # that swung within the generous candidate radius only occasionally --
    # and is cut outright here, before the (more expensive) boundary-gap
    # check ever runs on it.
    center_gate_max_median_to_p10_ratio: float = 5.0
    # Absolute-distance sanity gate, independent of the ratio gate above: a
    # pair is cut outright if its cluster CENTERS' median distance across the
    # whole sequence exceeds this many world units, regardless of how that
    # median compares to the pair's own p10 (a pair can have a "tight" ratio
    # while still sitting nowhere near contact in absolute terms, e.g. two
    # centers that are always ~2 units apart but only ever vary a little).
    center_gate_max_median_absolute_distance: float = 0.4
    # Kept iff the smoothed, confidence-weighted max all-frames gap <
    # contact_distance * keep_multiplier.
    keep_multiplier: float = 1.5
    # Per-frame gap statistic passed to compute_all_frames_gap; see its
    # gap_percentile docstring. 0 (default) = plain single-nearest-pair gap.
    gap_percentile: float = 0.0
    # Temporal window (frames) for confidence-weighted smoothing of the
    # per-frame gap sequence before taking its max; see smooth_and_summarize_gap.
    # 1 disables smoothing (still confidence-weighted: zero-confidence frames
    # are excluded from the max where any confident frame exists).
    gap_smoothing_window: int = 5
    # Diagnostic-only cutoff: frames with confidence_t below this are
    # reported via ClusterPairEdge.gap_low_confidence_frac, but no longer
    # dropped from the gap computation itself (see smooth_and_summarize_gap).
    # A hard drop used to sit here, on the same "don't trust it at all below
    # this" philosophy as flow3d/data/utils.py's parse_cotracker3_track_info
    # visibility*confidence>0.5 gate -- removed because it silently discarded
    # exactly the frames that would show a real separation whenever those
    # frames also happened to be low-confidence (confirmed: a pair with a
    # genuine 0.19 separation read as always-touching because every one of
    # its high-gap frames had confidence == 0.0 and got dropped before the
    # max was ever computed). Confidence now only downweights a frame in the
    # smoothing average (see smooth_and_summarize_gap) or feeds this
    # diagnostic -- it can no longer make a frame's gap invisible.
    gap_confidence_threshold: float = 0.5
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
    center_distance_median: float
    center_distance_p10: float
    gap_summary: float
    gap_median: float
    gap_min: float
    gap_max: float
    gap_mean_confidence: float
    gap_low_confidence_frac: float
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
    gap_percentile: float = 0.0,
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
    :param gap_percentile: 0 (default) uses the single closest pair each
        frame (the plain nearest-neighbor gap). >0 instead takes that
        percentile (e.g. 10 for "p10") of the combined a->b and b->a
        nearest-neighbor distance distribution -- robust to one outlier
        near-touching point (e.g. a stray boundary Gaussian or a raw-track
        mismatch) creating a spuriously tight single-pair gap when the two
        clusters' surfaces aren't actually broadly close. Confidence is
        read off the same point that achieved the reported percentile
        distance, not averaged separately.
    :return: (gap_t, confidence_t), both (T,). confidence_t is the weaker
        (min) of the two points that achieved that frame's reported gap.
    """
    pts_a = positions_all_frames[boundary_global_indices_a]  # (m, T, 3)
    pts_b = positions_all_frames[boundary_global_indices_b]  # (n, T, 3)
    T = pts_a.shape[1]

    gap_t = np.empty(T, dtype=np.float64)
    confidence_t = np.empty(T, dtype=np.float64)
    for t in range(T):
        a_t = pts_a[:, t, :]
        b_t = pts_b[:, t, :]

        dist_ab, idx_ab = cKDTree(b_t).query(a_t, k=1)  # (m,): a -> nearest b
        dist_ba, idx_ba = cKDTree(a_t).query(b_t, k=1)  # (n,): b -> nearest a

        if confidences_all_frames is None:
            conf_a_own = np.ones(len(boundary_global_indices_a))
            conf_b_own = np.ones(len(boundary_global_indices_b))
        else:
            conf_a_own = confidences_all_frames[boundary_global_indices_a, t]
            conf_b_own = confidences_all_frames[boundary_global_indices_b, t]
        # confidence of each nearest-neighbor pair = weaker of its two points.
        conf_ab = np.minimum(conf_a_own, conf_b_own[idx_ab])
        conf_ba = np.minimum(conf_b_own, conf_a_own[idx_ba])

        combined_dist = np.concatenate([dist_ab, dist_ba])
        combined_conf = np.concatenate([conf_ab, conf_ba])

        if gap_percentile <= 0.0:
            k = int(np.argmin(combined_dist))
        else:
            # Nearest-rank (not interpolated) so the reported gap is an
            # actual measured distance, with a real point pair -- and
            # therefore a real confidence -- behind it.
            rank = int(round((gap_percentile / 100.0) * (len(combined_dist) - 1)))
            k = int(np.argsort(combined_dist)[rank])

        gap_t[t] = combined_dist[k]
        confidence_t[t] = combined_conf[k]

    return gap_t, confidence_t


def smooth_and_summarize_gap(
    gap_t: np.ndarray,
    confidence_t: np.ndarray,
    window: int = 5,
    confidence_floor: float = 1e-3,
) -> float:
    """
    Confidence-*weighted* temporal smoothing of the per-frame gap sequence,
    then take the max of the smoothed sequence as the summary statistic.
    Every frame contributes -- none are dropped.

    A plain max over raw per-frame gaps is fragile: one occluded/jittery
    frame can spike the gap and cut an edge that's actually attached the
    whole sequence. This used to hard-drop any frame below a confidence
    threshold before smoothing, but that silently discarded exactly the
    frames that would reveal a real separation whenever those frames also
    happened to be low-confidence -- e.g. an occlusion moment where the true
    gap is large but the raw track lost the point, so every frame proving
    the separation got vetoed and the pair read as always-touching. Instead,
    each frame's confidence is used as its smoothing *weight*: a frame with
    near-zero confidence still counts, just barely, so a sustained low-
    confidence separation still drags the smoothed value up rather than
    vanishing outright; confidence_floor keeps an all-zero-confidence window
    from a divide-by-zero rather than making it disappear.

    :param window: frames per smoothing window (odd covers symmetrically),
        confidence-weighted within each window. <= 1 skips windowed
        smoothing and returns the (unweighted) raw max instead.
    """
    T = len(gap_t)
    if T == 0:
        return 0.0

    if window <= 1 or T <= 1:
        return float(gap_t.max())

    weights = confidence_t + confidence_floor
    half = window // 2
    smoothed: list[float] = []
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        g = gap_t[lo:hi]
        w = weights[lo:hi]
        smoothed.append(float(np.average(g, weights=w)))

    return float(max(smoothed))


def build_cluster_graph(
    cluster_ids: list[int],
    global_indices_by_cluster: dict[int, np.ndarray],
    positions_all_frames: np.ndarray,
    contact_distance: float,
    config: ClusterGraphConfig | None = None,
    confidences_all_frames: np.ndarray | None = None,
    force_include_pairs: set[tuple[int, int]] | None = None,
    force_exclude_pairs: set[tuple[int, int]] | None = None,
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
    :param force_include_pairs: Canonical ``(min_id, max_id)`` pairs that are
        evaluated and kept regardless of the automatic gates.
    :param force_exclude_pairs: Canonical pairs that are evaluated and cut
        regardless of the automatic gates.
    """
    config = config or ClusterGraphConfig()
    force_include_pairs = force_include_pairs or set()
    force_exclude_pairs = force_exclude_pairs or set()
    overlap = force_include_pairs & force_exclude_pairs
    if overlap:
        raise ValueError(f"Pairs cannot be both force-included and force-excluded: {sorted(overlap)}")

    valid_ids = set(cluster_ids)
    override_ids = {cid for pair in force_include_pairs | force_exclude_pairs for cid in pair}
    unknown_ids = override_ids - valid_ids
    if unknown_ids:
        raise ValueError(f"Override pairs contain invalid cluster ids: {sorted(unknown_ids)}")

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
    cluster_row = {cid: i for i, cid in enumerate(cluster_ids)}
    candidate_pairs, candidate_info = find_candidate_pairs(cluster_ids, centers_ts, radius)
    # Explicit overrides are evaluated even when the automatic center-radius
    # prefilter would not have selected them.
    candidate_pairs = sorted(set(candidate_pairs) | force_include_pairs | force_exclude_pairs)

    edges: list[ClusterPairEdge] = []
    for a, b in candidate_pairs:
        if (a, b) in candidate_info:
            t_star, center_distance = candidate_info[(a, b)]
        else:
            center_distances = np.linalg.norm(
                centers_ts[cluster_row[a]] - centers_ts[cluster_row[b]], axis=-1
            )
            t_star = int(np.argmin(center_distances))
            center_distance = float(center_distances[t_star])
        global_a = global_indices_by_cluster[a]
        global_b = global_indices_by_cluster[b]

        center_distance_all_frames = np.linalg.norm(
            centers_ts[cluster_row[a]] - centers_ts[cluster_row[b]], axis=-1
        )
        center_distance_median = float(np.median(center_distance_all_frames))
        center_distance_p10 = float(np.percentile(center_distance_all_frames, 10))

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
            gap_percentile=config.gap_percentile,
        )
        gap_summary = smooth_and_summarize_gap(
            gap_t, confidence_t, config.gap_smoothing_window,
        )
        gap_low_confidence_frac = float(
            np.mean(confidence_t < config.gap_confidence_threshold)
        )

        # Safety net: a real (even always-touching) pair still has 143 frames
        # of independent floating-point jitter, so gap_t reading EXACTLY 0.0
        # in every single frame is not physically plausible -- it's the
        # signature of two different clusters' boundary Gaussians having
        # been matched to the identical position source (e.g. a raw-track
        # pooling collision), not measured contact. Cut unconditionally,
        # regardless of gap_summary/threshold.
        suspected_collision = bool(np.all(gap_t == 0.0))

        # Sanity gates, checked in order of cheapest/most fundamental
        # implausibility first:
        # 1) Absolute gate: the cluster CENTERS' median distance across the
        #    whole sequence must not exceed center_gate_max_median_absolute_distance
        #    world units, regardless of the ratio gate below -- catches pairs
        #    that are simply never close in absolute terms.
        # 2) Ratio gate: that same median must not be more than
        #    center_gate_max_median_to_p10_ratio times their 10th-percentile
        #    distance. A large ratio means this pair's closest-approach
        #    regime is rare relative to how far apart they usually are --
        #    indistinguishable here from the min-over-many-noisy-candidate-
        #    pairs artifact this gate exists to catch (see
        #    ClusterGraphConfig.center_gate_max_median_to_p10_ratio). p10 <= 0
        #    (degenerate exact-coincidence for >=10% of frames) is treated as
        #    an infinite ratio, i.e. always cut here.
        center_distance_median_to_p10_ratio = (
            center_distance_median / center_distance_p10 if center_distance_p10 > 0 else float("inf")
        )
        if center_distance_median > config.center_gate_max_median_absolute_distance:
            kept = False
            reason = "cut_center_median_absolute_distance_too_high"
        elif center_distance_median_to_p10_ratio > config.center_gate_max_median_to_p10_ratio:
            kept = False
            reason = "cut_center_median_p10_ratio_too_high"
        elif suspected_collision:
            kept = False
            reason = "cut_suspected_position_collision"
        else:
            kept = gap_summary < threshold
            reason = "kept_all_frames_contact" if kept else "cut_gap_exceeds_threshold"

        if (a, b) in force_include_pairs:
            kept = True
            reason = "kept_manual_override"
        elif (a, b) in force_exclude_pairs:
            kept = False
            reason = "cut_manual_override"

        edges.append(
            ClusterPairEdge(
                cluster_a=a,
                cluster_b=b,
                kept=kept,
                reason=reason,
                frame_t_star=t_star,
                center_distance_t_star=center_distance,
                center_distance_median=center_distance_median,
                center_distance_p10=center_distance_p10,
                gap_summary=gap_summary,
                gap_median=float(np.median(gap_t)),
                gap_min=float(gap_t.min()),
                gap_max=float(gap_t.max()),
                gap_mean_confidence=float(confidence_t.mean()),
                gap_low_confidence_frac=gap_low_confidence_frac,
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

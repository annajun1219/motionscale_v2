"""
flow3d/analysis/loss.py

Auxiliary losses for the coarse-transform GNN correction (omega, delta_t)
produced by any *GraphCorrectedScalableMotionBases (flow3d/graph_coupling.py
and its "relative"/"relative_velocity_*"/"relative_*_attention*" variants,
including flow3d/graph_relative_linear_attention.py). Kept in this separate
file on purpose: node/edge features, message passing, and the correction API
are those files' concern; how the correction is regularized during training
is this file's concern. flow3d/trainer.py imports these functions and calls
them with the correction pulled from `motion_bases.last_correction` and the
GNN's own registered `edge_index_dir` buffer (`motion_bases.gnn.edge_index_dir`)
-- no changes to graph_relative_linear_attention.py (or any other graph_*.py
variant) are needed for that, since `last_correction` / `gnn.edge_index_dir`
are already part of every *GraphCorrectedScalableMotionBases's public API
(see graph_relative_linear_attention.py's own module docstring).

All three losses are exactly 0.0 while the GNN head is still zero-initialized
(the correction itself is exactly 0 then), so they don't perturb the
zero-init-equivalence property documented in
flow3d/graph_relative_linear_attention.py -- these losses only start
influencing the parameters once ordinary training has moved the GNN head off
zero and the correction becomes nonzero.

1. gnn_correction_magnitude_loss
   Plain L2 penalty on (omega, delta_t). The correction is meant to be a
   *small* nudge on top of the coarse rigid-body fit (keeping e.g. a hand
   cluster coordinated with its own forearm), not a free-floating residual
   that can drift arbitrarily far from the transform it's supposed to be
   correcting.

2. gnn_correction_smoothness_loss
   Second-order ("acceleration") penalty on the correction evaluated across
   a (t-1, t, t+1) frame triplet -- the same central-difference math as
   flow3d/loss_utils.py's compute_se3_smoothness_loss/compute_accel_loss,
   applied to the correction instead of the raw motion-basis params. The
   correction has its own independent per-frame GNN forward pass, so even a
   smooth underlying coarse motion doesn't automatically make the
   correction smooth frame-to-frame without this term.

3. gnn_correction_edge_consistency_loss
   Direction-only, hinge-margin penalty between the (omega, delta_t) of
   cluster pairs connected in the *fixed* graph topology (the same
   edge_index_dir the message-passing layer itself uses). It only fires
   when two connected clusters' corrections point in genuinely *opposite*
   directions (cosine similarity below -cos_margin); a joint where e.g. a
   forearm and hand correction differ substantially but aren't a direct
   sign flip incurs zero loss, so ordinary articulation is not constrained.
   Cosine similarity (not raw dot product) makes this a direction-only
   constraint, independent of correction magnitude (magnitude is already
   regularized separately by gnn_correction_magnitude_loss above), and
   F.cosine_similarity is numerically inert at zero (near-zero vectors ->
   near-zero similarity -> no spurious gradient near GNN zero-init).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "gnn_correction_magnitude_loss",
    "gnn_correction_smoothness_loss",
    "gnn_correction_edge_consistency_loss",
    "local_gnn_anchor_loss",
]


def gnn_correction_magnitude_loss(
    omega: torch.Tensor,
    delta_t: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 1.0,
) -> torch.Tensor:
    """
    L2 magnitude regularizer on the GNN's (omega, delta_t) correction.

    :param omega: (..., 3) so(3) correction (so3_exp_map input).
    :param delta_t: (..., 3) translation correction, same batch shape as omega.
    :return: scalar.
    """
    rot_loss = omega.pow(2).sum(dim=-1).mean()
    transl_loss = delta_t.pow(2).sum(dim=-1).mean()
    return weight_rot * rot_loss + weight_transl * transl_loss


def _second_order_diff_norm(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: (..., 3, D), axis=-2 indexing a (t-1, t, t+1) triplet.
    :return: scalar mean central-difference ("acceleration") norm.
    """
    accel = 2 * x[..., 1, :] - x[..., 0, :] - x[..., 2, :]
    return accel.norm(dim=-1).mean()


def gnn_correction_smoothness_loss(
    omega_triplet: torch.Tensor,
    delta_t_triplet: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
) -> torch.Tensor:
    """
    Second-order ("acceleration") temporal smoothness on the GNN correction,
    evaluated at consecutive frames (t-1, t, t+1). Mirrors
    flow3d/loss_utils.py's compute_se3_smoothness_loss/compute_accel_loss,
    applied to the correction (omega, delta_t) rather than the raw motion
    basis params.

    :param omega_triplet: (..., 3, 3), axis=-2 is the (t-1, t, t+1) triplet.
    :param delta_t_triplet: (..., 3, 3), same layout as omega_triplet.
    :return: scalar.
    """
    r_accel_loss = _second_order_diff_norm(omega_triplet)
    t_accel_loss = _second_order_diff_norm(delta_t_triplet)
    return weight_rot * r_accel_loss + weight_transl * t_accel_loss


def gnn_correction_edge_consistency_loss(
    omega: torch.Tensor,
    delta_t: torch.Tensor,
    edge_index_dir: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 1.0,
    cos_margin: float = 0.5,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weak edge-consistency loss: penalizes connected clusters' corrections
    only when they point in genuinely *opposite* directions (cosine
    similarity < -cos_margin). Differing-but-not-opposing corrections incur
    no loss, so ordinary articulation at a joint is unaffected.

    :param omega: (C, B, 3).
    :param delta_t: (C, B, 3), same batch shape as omega.
    :param edge_index_dir: (2, E) [src, dst] pairs over the *same* fixed
        graph topology the GNN's message passing uses (e.g.
        motion_bases.gnn.edge_index_dir). The loss is symmetric in src/dst,
        so a directed edge list with both (i, j) and (j, i) present just
        double-counts each undirected pair -- harmless, only rescales the
        loss by a constant factor.
    :param cos_margin: only pairs with cosine similarity below -cos_margin
        incur loss. cos_margin=0.5 means only genuinely opposing (more than
        ~120 degrees apart) corrections are penalized.
    :param edge_weight: (E, B) or None. None (default -- every variant other
        than flow3d/graph_relative_linear_attention_frame.py takes this path)
        reproduces the previous behavior exactly: every edge weighted
        equally, plain mean. When given (E, B) -- e.g. the frame variant's
        `motion_bases.last_correction["edge_gate_dir"]`, this batch's
        per-directed-edge CONNECTED/DISCONNECTED/UNKNOWN gate -- each edge's
        (squared) hinge term is weighted by it and the result is normalized
        by `sum(edge_weight)` (a weighted mean, NOT a count of "active"
        edges): a DISCONNECTED edge (weight exactly 0) contributes exactly 0
        to both the numerator and denominator, and an UNKNOWN edge with e.g.
        weight 0.3 contributes exactly 0.3x to both -- a continuous weighting,
        not a hard include/exclude split.
    :return: scalar; exactly 0.0 (no graph edges) if edge_index_dir is empty.
    """
    if edge_index_dir.numel() == 0:
        return omega.new_zeros(())

    src, dst = edge_index_dir[0], edge_index_dir[1]

    cos_omega = F.cosine_similarity(omega[src], omega[dst], dim=-1)  # (E, B)
    cos_delta_t = F.cosine_similarity(delta_t[src], delta_t[dst], dim=-1)  # (E, B)

    hinge_omega = F.relu(-cos_omega - cos_margin)
    hinge_delta_t = F.relu(-cos_delta_t - cos_margin)

    if edge_weight is None:
        rot_term = hinge_omega.pow(2).mean()
        transl_term = hinge_delta_t.pow(2).mean()
    else:
        denom = edge_weight.sum().clamp_min(1e-6)
        rot_term = (hinge_omega.pow(2) * edge_weight).sum() / denom
        transl_term = (hinge_delta_t.pow(2) * edge_weight).sum() / denom

    return weight_rot * rot_term + weight_transl * transl_term


def local_gnn_anchor_loss(
    corrected: torch.Tensor,
    predicted: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Huber anchor loss for flow3d/graph_relative_local_attention.py's
    LocalRelativeAttentionGNN: pulls each target node's ACTUAL corrected
    center (mean of its member Gaussians' real post-correction positions --
    the top-k RBF blend + pivoted rotation already applied, not a node-level
    shortcut) toward the rigid-consistency position its valid (message_source)
    neighbors predict for it. Robust (Huber, not L2) since a neighbor's
    rigid-consistency prediction can be a poor match at a genuinely
    non-rigid joint -- such (target, frame) pairs shouldn't get an
    unbounded gradient.

    Already-flattened to the (target, frame) pairs that actually have >=1
    valid source this step -- the caller (the wrapper's forward pass) is
    responsible for that selection; pairs with 0 sources have no prediction
    to anchor to and are skipped before ever reaching this function.

    :param corrected: (M, 3) each target's real corrected center this step.
    :param predicted: (M, 3) same (target, frame) pairs' source-predicted
        position (mean over valid sources of `p_j^t + R_j^t @ d_ij^0`).
    :return: scalar; `corrected.new_zeros(())` if M == 0.
    """
    if corrected.shape[0] == 0:
        return corrected.new_zeros(())
    dist = (corrected - predicted).norm(dim=-1)  # (M,)
    return F.huber_loss(dist, torch.zeros_like(dist), delta=beta, reduction="mean")


if __name__ == "__main__":
    # Sanity checks:
    #   1. all three losses are exactly 0 at zero correction (zero-init equivalence).
    #   2. magnitude loss grows with |omega|/|delta_t|.
    #   3. smoothness loss is 0 for a constant-velocity (linear-in-time) correction,
    #      nonzero for a jerky one.
    #   4. edge-consistency loss is 0 for aligned/orthogonal/mildly-opposing
    #      (within cos_margin) neighbor corrections, positive once they truly oppose.
    torch.manual_seed(0)
    C, B = 6, 4

    # --- 1. zero-init equivalence ---
    zero_omega = torch.zeros(C, B, 3)
    zero_delta_t = torch.zeros(C, B, 3)
    zero_triplet = torch.zeros(C, B, 3, 3)
    edge_index_dir = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)

    mag0 = gnn_correction_magnitude_loss(zero_omega, zero_delta_t)
    smooth0 = gnn_correction_smoothness_loss(zero_triplet, zero_triplet)
    edge0 = gnn_correction_edge_consistency_loss(zero_omega, zero_delta_t, edge_index_dir)
    print(f"[zero-init] magnitude={mag0.item():.3e} smoothness={smooth0.item():.3e} edge={edge0.item():.3e}")
    assert mag0.item() == 0.0 and smooth0.item() == 0.0 and edge0.item() == 0.0

    # --- 2. magnitude grows with correction size ---
    small = gnn_correction_magnitude_loss(0.01 * torch.randn(C, B, 3), 0.01 * torch.randn(C, B, 3))
    large = gnn_correction_magnitude_loss(1.0 * torch.randn(C, B, 3), 1.0 * torch.randn(C, B, 3))
    print(f"[magnitude] small={small.item():.3e} large={large.item():.3e}")
    assert large.item() > small.item()

    # --- 3. smoothness: linear-in-time -> 0, jerky -> > 0 ---
    t = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3, 1)
    velocity = torch.randn(C, B, 1, 3)
    linear = (velocity * t).expand(C, B, 3, 3).clone()  # x(t) = v * t, exactly linear
    smooth_linear = gnn_correction_smoothness_loss(linear, linear)
    jerky = linear.clone()
    jerky[:, :, 1] += torch.randn(C, B, 3)  # perturb the middle (t=0) frame only
    smooth_jerky = gnn_correction_smoothness_loss(jerky, jerky)
    print(f"[smoothness] linear={smooth_linear.item():.3e} jerky={smooth_jerky.item():.3e}")
    assert smooth_linear.item() < 1e-5
    assert smooth_jerky.item() > smooth_linear.item()

    # --- 4. edge consistency: aligned/orthogonal/mild-opposition -> 0, strong opposition -> > 0 ---
    aligned = torch.ones(C, B, 3)
    edge_aligned = gnn_correction_edge_consistency_loss(aligned, aligned, edge_index_dir)

    orthogonal = torch.zeros(C, B, 3)
    orthogonal[0::2] = torch.tensor([1.0, 0.0, 0.0])
    orthogonal[1::2] = torch.tensor([0.0, 1.0, 0.0])
    edge_orth = gnn_correction_edge_consistency_loss(orthogonal, orthogonal, edge_index_dir)

    opposing = torch.zeros(C, B, 3)
    opposing[0::2] = torch.tensor([1.0, 0.0, 0.0])
    opposing[1::2] = torch.tensor([-1.0, 0.0, 0.0])  # cos = -1, well past any reasonable margin
    edge_opp = gnn_correction_edge_consistency_loss(opposing, opposing, edge_index_dir)

    print(
        f"[edge-consistency] aligned={edge_aligned.item():.3e} "
        f"orthogonal={edge_orth.item():.3e} opposing={edge_opp.item():.3e}"
    )
    assert edge_aligned.item() == 0.0
    assert edge_orth.item() == 0.0
    assert edge_opp.item() > 0.0

    # empty graph -> exactly 0, no crash
    empty_edges = torch.zeros(2, 0, dtype=torch.long)
    edge_empty = gnn_correction_edge_consistency_loss(opposing, opposing, empty_edges)
    print(f"[edge-consistency] empty graph = {edge_empty.item():.3e}")
    assert edge_empty.item() == 0.0

    # --- 5. edge_weight (frame-variant gate weighting): backward-compatible
    #     default, all-zero excludes everything, and a partial (0/1) mask
    #     reproduces the plain mean restricted to the weight==1 subset exactly
    #     (since sum(hinge^2 * weight) / sum(weight) collapses to that subset's
    #     mean when weight is binary).
    E = edge_index_dir.shape[1]
    edge_weight_ones = torch.ones(E, B)
    edge_opp_weight_ones = gnn_correction_edge_consistency_loss(
        opposing, opposing, edge_index_dir, edge_weight=edge_weight_ones
    )
    print(f"[edge-consistency gate] weight=1 everywhere == unweighted: {edge_opp_weight_ones.item():.3e} vs {edge_opp.item():.3e}")
    assert abs(edge_opp_weight_ones.item() - edge_opp.item()) < 1e-6

    edge_weight_zeros = torch.zeros(E, B)
    edge_opp_weight_zeros = gnn_correction_edge_consistency_loss(
        opposing, opposing, edge_index_dir, edge_weight=edge_weight_zeros
    )
    print(f"[edge-consistency gate] weight=0 everywhere -> loss = {edge_opp_weight_zeros.item():.3e} (expect 0)")
    assert edge_opp_weight_zeros.item() == 0.0

    edge_weight_partial = torch.zeros(E, B)
    edge_weight_partial[0] = 1.0  # only the first (opposing) edge contributes
    edge_opp_weight_partial = gnn_correction_edge_consistency_loss(
        opposing, opposing, edge_index_dir, edge_weight=edge_weight_partial, cos_margin=0.5
    )
    src0, dst0 = edge_index_dir[0, 0], edge_index_dir[1, 0]
    cos0 = F.cosine_similarity(opposing[src0], opposing[dst0], dim=-1)
    # omega and delta_t are both `opposing` here, so the rot and transl terms
    # are identical and both count (weight_rot=weight_transl=1.0 default).
    expected_partial = 2.0 * F.relu(-cos0 - 0.5).pow(2).mean().item()
    print(
        f"[edge-consistency gate] weight on edge 0 only -> loss = "
        f"{edge_opp_weight_partial.item():.3e} vs expected {expected_partial:.3e}"
    )
    assert abs(edge_opp_weight_partial.item() - expected_partial) < 1e-6

    # --- 6. local_gnn_anchor_loss: 0 at perfect agreement, grows with distance,
    #     robust (sub-quadratic) for far outliers, empty input -> exactly 0 ---
    M = 5
    predicted = torch.randn(M, 3)
    anchor_zero = local_gnn_anchor_loss(predicted.clone(), predicted, beta=0.01)
    print(f"[anchor] exact match = {anchor_zero.item():.3e}")
    assert anchor_zero.item() == 0.0

    small_offset = predicted + 0.001  # well inside beta -> ~quadratic regime
    anchor_small = local_gnn_anchor_loss(small_offset, predicted, beta=0.01)
    large_offset = predicted + 10.0  # far outside beta -> linear regime
    anchor_large = local_gnn_anchor_loss(large_offset, predicted, beta=0.01)
    print(f"[anchor] small_offset={anchor_small.item():.3e} large_offset={anchor_large.item():.3e}")
    assert 0.0 < anchor_small.item() < anchor_large.item()
    # robustness: a 1000x larger offset should NOT cost ~1e6x more (that would be pure L2) --
    # Huber's linear-beyond-beta regime keeps the ratio close to the offset-distance ratio, not its square.
    dist_ratio = (large_offset - predicted).norm(dim=-1).mean() / (small_offset - predicted).norm(dim=-1).mean()
    loss_ratio = anchor_large.item() / anchor_small.item()
    print(f"[anchor] dist_ratio={dist_ratio.item():.1f} loss_ratio={loss_ratio:.1f} (robust: not dist_ratio^2)")
    assert loss_ratio < dist_ratio.item() ** 2 / 10

    anchor_empty = local_gnn_anchor_loss(torch.zeros(0, 3), torch.zeros(0, 3), beta=0.01)
    print(f"[anchor] empty input = {anchor_empty.item():.3e}")
    assert anchor_empty.item() == 0.0

    print("OK")

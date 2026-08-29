"""
flow3d/analysis/loss_joint_gnn_only.py

GNN-only joint anchor loss: same "keep the joint from opening" idea as
flow3d/analysis/loss_joint.py, but built to directly answer a question that
file's coarse-only version cannot: measured on a real trained checkpoint,
flow3d/analysis/loss_joint.py's loss converged its coarse-only distance to
~1x the canonical target while the TRUE (coarse+fine-blended, i.e. actually
rendered) boundary-Gaussian distance barely moved at all. The reason:
compute_transforms_coarse is only the rigid per-cluster skeleton; what's
rendered is compute_transforms (coarse composed with a per-Gaussian FINE
correction), and fine motion bases are entirely per-cluster with no
mechanism forcing two clusters' fine corrections to agree at a shared
boundary -- so a perfectly-aligned coarse skeleton can still render as an
open seam.

This file fixes that by changing what is measured, and isolating what is
allowed to fix it:

1. The per-frame distance is measured on the TRUE (coarse+fine-blended)
   boundary-Gaussian positions -- each side's anchor is the mean position of
   its own boundary_global_indices_a/b (see loss_joint.py's
   JointAnchorBoundarySets... actually JointAnchors comment) run through the
   *full* compute_transforms, not a single canonical point run through only
   compute_transforms_coarse. Fine motion is per-Gaussian, so (unlike the
   pure-rigid coarse case) mean-of-transformed-positions and
   transform-of-the-mean-position are no longer equivalent -- each boundary
   Gaussian has to be transformed individually and then averaged, hence
   JointAnchorBoundarySets keeps index sets instead of one pre-averaged
   canonical point.
2. Gradient from this loss reaches ONLY motion_bases.gnn's own parameters
   (the ones producing the (omega, delta_t) correction). The base coarse
   motion basis (rots/transls), cluster centers, fine motion bases
   (fine_rots/fine_transls), and per-Gaussian motion coefficients are all
   treated as fixed (detached) inputs -- see _gnn_only_compute_transforms.
   This is a deliberate probe, not an accident: it answers "how much of the
   true gap can the GNN's rigid-per-cluster correction alone close", with
   nothing else moving to help (or to obscure the answer by moving instead).

Known limitation, by design and not a bug: a rigid per-cluster correction
can only close the *coherent* (rigid-shiftable) component of a boundary
gap. Measured directly on this codebase's own trained checkpoints, that
coherent component is typically only ~15-35% of the total gap magnitude at
a pair's worst frame -- individual boundary points' nearest-neighbour
displacement vectors are largely uncorrelated with each other (mean cosine
similarity to their own average direction ~0.1-0.2, and some points'
displacements point in nearly the opposite direction, cosine down to
~-0.95). So this loss should meaningfully reduce the true rendered gap, but
is not expected to fully close it by itself; the remaining non-rigid
scatter lives in the fine motion bases. A fine-level treatment is
deliberately NOT included in this file -- a raw-displacement-variance
penalty was considered and rejected as the first step (harder to reason
about what it actually optimizes for and easy to fight the coarse-level fix
above); the safer next step, if the residual after this loss is still too
large, is a *distance* loss between actual boundary correspondence points
under the full transform (same shape as this file's own measurement, just
letting fine motion move too) rather than a variance-based penalty -- left
for a follow-up file once this one's isolated effect has been measured.

Pieces, matching loss_joint.py's split of I/O+averaging from per-step math:
1. build_joint_anchor_boundary_sets: I/O. Loads build_cluster_graph.py's
   edges.pt (the same file loss_joint.py's build_joint_anchors_from_edges
   reads) and keeps each kept edge's boundary_global_indices_a/b as index
   sets, plus the same canonical_distance definition as loss_joint.py's
   JointAnchors (mean-of-canonical-positions gap) -- the fixed Huber target
   doesn't change, only how the *current* distance is measured does.
2. _gnn_only_compute_transforms: the isolation mechanism. A line-for-line
   mirror of
   RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.compute_transforms's
   coarse-to-fine composition (flow3d/graph_relative_linear_attention.py),
   with the base coarse inputs, centers, and fine motion detached before the
   GNN forward pass and before fine blending.
3. transform_joint_anchor_boundary_sets_gnn_only: per-step glue -- runs (2)
   once over the union of every pair's boundary Gaussians, then
   scatter-means the resulting positions back into one (P, 2, B, 3) trajectory
   per pair/side, the same shape loss_joint.py's transform_joint_anchors
   returns, so the loss itself is unchanged: flow3d/analysis/loss_joint.py's
   own joint_anchor_loss (imported, not duplicated) is reused as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from flow3d.analysis.loss_joint import joint_anchor_loss
from flow3d.graph_coupling import compose_rotation, so3_exp_map
from flow3d.transforms import cont_6d_to_rmat

__all__ = [
    "JointAnchorBoundarySets",
    "build_joint_anchor_boundary_sets",
    "transform_joint_anchor_boundary_sets_gnn_only",
    "joint_anchor_loss",
]


@dataclass
class JointAnchorBoundarySets:
    """Per-pair boundary-Gaussian identity, kept as index sets rather than a
    single pre-averaged canonical point (unlike loss_joint.py's
    JointAnchors) -- fine motion blending is per-Gaussian, so each boundary
    Gaussian has to be transformed individually before averaging.

    :param cluster_ids: (P, 2) long. Same convention as JointAnchors:
        [:, 0] is cluster_a's raw id, [:, 1] is cluster_b's.
    :param global_indices_a / global_indices_b: length-P Python lists of
        long tensors (variable size per pair), each indexing into the
        canonical foreground Gaussian arrays (fg.params["means"],
        fg.get_cluster_ids(), fg.get_coefs()).
    :param canonical_distance: (P,) float. Same definition as
        loss_joint.py's JointAnchors.canonical_distance -- the fixed Huber
        target, unaffected by how the *current* distance is measured.
    """

    cluster_ids: torch.Tensor
    global_indices_a: list[torch.Tensor]
    global_indices_b: list[torch.Tensor]
    canonical_distance: torch.Tensor

    @property
    def num_pairs(self) -> int:
        return int(self.cluster_ids.shape[0])


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_long_indices(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


def build_joint_anchor_boundary_sets(
    edges_path: str | Path,
    canonical_means: torch.Tensor,
    device: torch.device | None = None,
) -> JointAnchorBoundarySets:
    """
    Build joint-anchor boundary index sets from
    flow3d/analysis/build_cluster_graph.py's edges.pt -- the same file
    flow3d/analysis/loss_joint.py's build_joint_anchors_from_edges reads
    (see that function's docstring for why edges.pt rather than
    cluster_pairs.py's fixed_boundary_indices.pt).

    :param edges_path: path to build_cluster_graph.py's edges.pt (a dict
        with an "edges_kept" list; each entry holds at least "cluster_a",
        "cluster_b", "boundary_global_indices_a", "boundary_global_indices_b").
    :param canonical_means: (G, 3) canonical foreground Gaussian means the
        saved indices index into (e.g. model.fg.params["means"].detach()).
    :param device: device for the returned tensors. Default:
        canonical_means's device.
    :return: JointAnchorBoundarySets, one row per kept edge with a non-empty
        boundary set on both sides.
    :raises RuntimeError: if no kept edge has a usable boundary set.
    """
    if device is None:
        device = canonical_means.device

    payload = _torch_load(edges_path)
    if not isinstance(payload, dict) or "edges_kept" not in payload:
        raise TypeError(
            f"{edges_path} must be a build_cluster_graph.py edges.pt "
            "(a dict with an 'edges_kept' list)."
        )

    means_cpu = canonical_means.detach().to("cpu")

    cluster_id_rows: list[tuple[int, int]] = []
    idx_a_rows: list[torch.Tensor] = []
    idx_b_rows: list[torch.Tensor] = []
    canonical_distance_rows: list[torch.Tensor] = []

    required_keys = (
        "cluster_a",
        "cluster_b",
        "boundary_global_indices_a",
        "boundary_global_indices_b",
    )
    for entry in payload["edges_kept"]:
        if not isinstance(entry, dict):
            raise TypeError(f"Malformed edges_kept entry in {edges_path}: {entry!r}")
        missing = [key for key in required_keys if key not in entry]
        if missing:
            label = f"{entry.get('cluster_a')}-{entry.get('cluster_b')}"
            raise KeyError(f"edges_kept entry {label!r} in {edges_path} is missing {missing}.")

        idx_a = _as_long_indices(entry["boundary_global_indices_a"])
        idx_b = _as_long_indices(entry["boundary_global_indices_b"])
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue

        anchor_a = means_cpu[idx_a].mean(dim=0)
        anchor_b = means_cpu[idx_b].mean(dim=0)

        cluster_id_rows.append((int(entry["cluster_a"]), int(entry["cluster_b"])))
        idx_a_rows.append(idx_a.to(device))
        idx_b_rows.append(idx_b.to(device))
        canonical_distance_rows.append((anchor_a - anchor_b).norm())

    if not cluster_id_rows:
        raise RuntimeError(
            f"No kept edge in {edges_path} had a usable boundary set on both "
            "sides -- cannot build any joint anchor."
        )

    return JointAnchorBoundarySets(
        cluster_ids=torch.tensor(cluster_id_rows, dtype=torch.long, device=device),
        global_indices_a=idx_a_rows,
        global_indices_b=idx_b_rows,
        canonical_distance=torch.stack(canonical_distance_rows).to(device),
    )


def _gnn_only_compute_transforms(
    motion_bases: Any,
    ts: torch.Tensor,
    coefs: torch.Tensor,
    cluster_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Line-for-line mirror of
    RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.compute_transforms's
    coarse-to-fine composition (flow3d/graph_relative_linear_attention.py),
    with one change: the coarse transform's inputs (base rots/transls,
    centers) and every fine-motion input (fine_rots, fine_transls) are
    detached before the GNN forward pass and before fine blending, so
    gradient can reach ONLY motion_bases.gnn's own parameters -- not the
    base coarse motion basis, not cluster centers, not fine motion. `coefs`
    is expected to already be detached by the caller (see
    build_joint_anchor_boundary_sets / trainer.py's call site) since it's a
    per-Gaussian *fine* quantity, same as canonical_means.

    The GNN itself is called on the FULL (C, B, ...) cluster set (never a
    subset restricted to the clusters a specific pair cares about) because
    its message passing needs every cluster's own (detached) features to
    route correctly through the fixed graph topology
    (motion_bases.gnn.edge_index_dir) -- slicing down would silently break
    cross-cluster attention for those clusters' real neighbors.

    :param motion_bases: a *GraphCorrectedScalableMotionBases (must expose
        .gnn -- this function has nothing meaningful to isolate for a plain
        ScalableMotionBases without a GNN correction).
    :param ts: (B,) frame indices.
    :param coefs: (G, F) per-Gaussian fine-basis blend weights (detached).
    :param cluster_ids: (G,) per-Gaussian raw cluster id.
    :return: (G, B, 3, 4) transforms.
    """
    mb = motion_bases
    if not hasattr(mb, "gnn"):
        raise TypeError(
            "motion_bases has no .gnn -- this loss only makes sense for a "
            "*GraphCorrectedScalableMotionBases (--enable_graph_coupling)."
        )

    coarse_rot_6d = mb.params["rots"][:, ts].detach()  # (C, B, 6)
    coarse_transl = mb.params["transls"][:, ts].detach()  # (C, B, 3)
    centers = mb.params["centers"].detach()  # (C, 3)
    ts_prev = (ts - 1).clamp(min=0)
    coarse_transl_prev = mb.params["transls"][:, ts_prev].detach()  # (C, B, 3)
    coarse_vel = coarse_transl - coarse_transl_prev  # detached input feature only

    omega, delta_t = mb.gnn(coarse_rot_6d, coarse_transl, centers, coarse_vel)  # (C,B,3) x2, trainable via mb.gnn params

    coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)  # (C, B, 3, 3), detached
    R_correction = so3_exp_map(omega)  # (C, B, 3, 3), trainable
    corrected_rotmats = compose_rotation(R_correction, coarse_rotmats)  # grad flows only via R_correction
    corrected_transl = coarse_transl + delta_t  # grad flows only via delta_t

    fine_transls = mb.params["fine_transls"][:, :, ts].detach()  # (C, F, B, 3)
    fine_rots = mb.params["fine_rots"][:, :, ts].detach()  # (C, F, B, 6)
    fine_rotmats = cont_6d_to_rmat(fine_rots)  # (C, F, B, 3, 3)

    C, F_dim, B, _ = fine_transls.shape
    total_rotmats = torch.einsum(
        "cbij,cfbjk->cfbik", corrected_rotmats, fine_rotmats
    )  # (C, F, B, 3, 3)
    total_6d = total_rotmats[..., :, :2].transpose(-1, -2).reshape(C, F_dim, B, 6)

    R_total_c = torch.einsum("cfbij,cj->cfbi", total_rotmats, centers)
    R_coarse_t_fine = torch.einsum("cbij,cfbj->cfbi", corrected_rotmats, fine_transls)
    total_transls = (
        -R_total_c + R_coarse_t_fine + corrected_transl[:, None] + centers[:, None, None]
    )

    transls_flat = total_transls.contiguous().view(C * F_dim, -1)
    rots_flat = total_6d.contiguous().view(C * F_dim, -1)

    G = cluster_ids.shape[0]
    base_offsets = torch.arange(F_dim, device=cluster_ids.device)
    bag_indices = (cluster_ids.unsqueeze(1) * F_dim) + base_offsets  # (G, F)

    transls = F.embedding_bag(
        weight=transls_flat, input=bag_indices, per_sample_weights=coefs, mode="sum",
    ).view(G, B, 3)
    rots_blended = F.embedding_bag(
        weight=rots_flat, input=bag_indices, per_sample_weights=coefs, mode="sum",
    ).view(G, B, 6)
    rotmats = cont_6d_to_rmat(rots_blended)

    return torch.cat([rotmats, transls[..., None]], dim=-1)  # (G, B, 3, 4)


def _apply_transform(transforms: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """transforms: (N, B, 3, 4), points: (N, 3) -> (N, B, 3)."""
    homog = F.pad(points, (0, 1), value=1.0)  # (N, 4)
    return torch.einsum("nbij,nj->nbi", transforms, homog)


def transform_joint_anchor_boundary_sets_gnn_only(
    motion_bases: Any,
    ts: torch.Tensor,
    boundary_sets: JointAnchorBoundarySets,
    canonical_means: torch.Tensor,
    coefs_all: torch.Tensor,
    cluster_ids_all: torch.Tensor,
) -> torch.Tensor:
    """
    For every pair, mean(TRUE position over that side's boundary Gaussians),
    with gradient reaching only motion_bases.gnn's parameters (see
    _gnn_only_compute_transforms). One shared _gnn_only_compute_transforms
    call over the union of every pair's/side's boundary Gaussians
    (deduplicated) -- the underlying (C, F, B) coarse-to-fine composition is
    identical regardless of which specific Gaussians are queried, only the
    final per-Gaussian embedding_bag blend depends on the query set, so
    there is no need to repeat the (shared, more expensive) composition
    per pair.

    :param motion_bases: a *GraphCorrectedScalableMotionBases.
    :param ts: (B,) frame indices.
    :param boundary_sets: JointAnchorBoundarySets with P pairs.
    :param canonical_means: (G_fg, 3) canonical foreground Gaussian means
        (e.g. model.fg.params["means"].detach() -- pass already-detached,
        same convention as build_joint_anchor_boundary_sets).
    :param coefs_all: (G_fg, F) per-Gaussian fine-basis blend weights (e.g.
        model.fg.get_coefs().detach()).
    :param cluster_ids_all: (G_fg,) per-Gaussian raw cluster id (e.g.
        model.fg.get_cluster_ids().reshape(-1).long()).
    :return: (P, 2, B, 3) transformed anchor positions; [:, 0] is
        cluster_ids[:, 0]'s anchor, [:, 1] is cluster_ids[:, 1]'s -- same
        shape/convention as loss_joint.py's transform_joint_anchors, so
        loss_joint.py's own joint_anchor_loss applies unchanged.
    """
    device = canonical_means.device
    all_indices = torch.cat(boundary_sets.global_indices_a + boundary_sets.global_indices_b)
    unique_indices, inverse = torch.unique(all_indices, return_inverse=True)

    transforms = _gnn_only_compute_transforms(
        motion_bases, ts, coefs_all[unique_indices], cluster_ids_all[unique_indices]
    )  # (U, B, 3, 4)
    positions = _apply_transform(transforms, canonical_means[unique_indices])  # (U, B, 3)
    per_entry_positions = positions[inverse]  # (len(all_indices), B, 3), un-deduplicated

    P = boundary_sets.num_pairs
    B = positions.shape[1]

    group_id_rows: list[torch.Tensor] = []
    for p in range(P):
        na = boundary_sets.global_indices_a[p].numel()
        nb = boundary_sets.global_indices_b[p].numel()
        group_id_rows.append(torch.full((na,), 2 * p, dtype=torch.long, device=device))
        group_id_rows.append(torch.full((nb,), 2 * p + 1, dtype=torch.long, device=device))
    group_ids = torch.cat(group_id_rows)  # (len(all_indices),)

    sums = positions.new_zeros(P * 2, B, 3)
    counts = positions.new_zeros(P * 2)
    sums.index_add_(0, group_ids, per_entry_positions)
    counts.index_add_(0, group_ids, torch.ones_like(group_ids, dtype=positions.dtype))

    means = sums / counts[:, None, None].clamp_min(1.0)
    return means.view(P, 2, B, 3)


if __name__ == "__main__":
    # Sanity checks:
    #   1. zero-init: the GNN-only transform reproduces EXACTLY the same
    #      positions as the official (non-isolated) compute_transforms,
    #      since zero-init means the composed correction is the identity.
    #   2. value match after the GNN moves off zero: the GNN-only transform
    #      still numerically matches the official compute_transforms (same
    #      forward VALUES), even though its gradient behavior differs --
    #      confirms the coarse-to-fine composition was reimplemented
    #      correctly.
    #   3. gradient isolation: after backward(), motion_bases.gnn's own
    #      params receive gradient; base coarse rots/transls/centers and
    #      fine_rots/fine_transls receive NONE.
    #   4. build_joint_anchor_boundary_sets round-trips a saved
    #      edges.pt-shaped file into the expected index sets / canonical
    #      distance.
    #   5. end-to-end: transform_joint_anchor_boundary_sets_gnn_only +
    #      loss_joint.joint_anchor_loss produce a finite, correctly-shaped
    #      scalar.
    import tempfile

    from flow3d.graph_relative_linear_attention import (
        RelativeVelLinearAttentionGraphCorrectedScalableMotionBases,
    )
    from flow3d.params import ScalableMotionBases

    torch.manual_seed(0)
    num_clusters, num_frames, num_fine, num_fg = 6, 8, 3, 40

    centers = torch.randn(num_clusters, 3)
    rots = torch.randn(num_clusters, num_frames, 6)
    transls = torch.randn(num_clusters, num_frames, 3) * 0.1
    fine_rots = torch.randn(num_clusters, num_fine, num_frames, 6)
    fine_transls = torch.randn(num_clusters, num_fine, num_frames, 3) * 0.01
    baseline = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    graph_bases = RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline, edge_index=edge_index, gnn_hidden_dim=32, gnn_num_layers=2, gnn_num_heads=4,
    )

    ts = torch.arange(num_frames)
    cluster_ids_all = torch.randint(0, num_clusters, (num_fg,))
    coefs_all = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)
    canonical_means = torch.randn(num_fg, 3)

    # --- 1. zero-init: GNN-only transform == official compute_transforms ---
    official = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all)
    gnn_only = _gnn_only_compute_transforms(graph_bases, ts, coefs_all, cluster_ids_all)
    max_diff_zero_init = (official - gnn_only).abs().max().item()
    print(f"[zero-init] max |official - gnn_only| = {max_diff_zero_init:.3e}")
    assert max_diff_zero_init < 1e-5

    # --- move the GNN's head off zero (same trick graph_relative_linear_attention.py's own test uses) ---
    graph_bases.zero_grad()
    loss0 = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all).pow(2).mean()
    loss0.backward()
    with torch.no_grad():
        for p in graph_bases.gnn.head.parameters():
            if p.grad is not None:
                p -= 0.5 * p.grad

    # --- 2. post-step: GNN-only transform still matches official VALUES ---
    official2 = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all)
    gnn_only2 = _gnn_only_compute_transforms(graph_bases, ts, coefs_all, cluster_ids_all)
    max_diff_post_step = (official2 - gnn_only2).abs().max().item()
    print(f"[post-step] max |official - gnn_only| = {max_diff_post_step:.3e}  "
          f"(both nonzero now: |official|_max={official2.abs().max().item():.3e})")
    assert max_diff_post_step < 1e-4
    assert official2.abs().max().item() > 1e-3, "the GNN correction should be nonzero after the step"

    # --- 3. gradient isolation ---
    graph_bases.zero_grad()
    out = _gnn_only_compute_transforms(graph_bases, ts, coefs_all, cluster_ids_all)
    isolation_loss = out.pow(2).mean()
    isolation_loss.backward()

    head_grad_norm = graph_bases.gnn.head.weight.grad.norm().item()
    print(f"[isolation] gnn.head.weight.grad norm = {head_grad_norm:.3e} (expect > 0)")
    assert head_grad_norm > 0.0, "GNN's own parameters must receive gradient from this loss"

    for name in ("rots", "transls", "centers", "fine_rots", "fine_transls"):
        grad = graph_bases.params[name].grad
        is_none_or_zero = grad is None or grad.abs().max().item() == 0.0
        print(f"[isolation] motion_bases.params[{name!r}].grad "
              f"{'is None' if grad is None else f'max={grad.abs().max().item():.3e}'} (expect None/0)")
        assert is_none_or_zero, f"params[{name!r}] must receive NO gradient from this loss"

    # --- 4. build_joint_anchor_boundary_sets round-trip ---
    boundary_a = torch.tensor([0, 1, 2], dtype=torch.long)
    boundary_b = torch.tensor([3, 4, 5], dtype=torch.long)
    edges_payload = {
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edges_kept": [
            {
                "cluster_a": 0,
                "cluster_b": 1,
                "reason": "kept_all_frames_contact",
                "boundary_global_indices_a": boundary_a,
                "boundary_global_indices_b": boundary_b,
            }
        ],
        "edges_cut": [],
        "cluster_ids": list(range(num_clusters)),
        "meta": {},
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(edges_payload, f.name)
        boundary_sets = build_joint_anchor_boundary_sets(f.name, canonical_means)

    expected_distance = (
        canonical_means[boundary_a].mean(dim=0) - canonical_means[boundary_b].mean(dim=0)
    ).norm()
    print(
        f"[build] num_pairs={boundary_sets.num_pairs} "
        f"dist_err={(boundary_sets.canonical_distance[0] - expected_distance).abs().item():.3e}"
    )
    assert boundary_sets.num_pairs == 1
    assert (boundary_sets.canonical_distance[0] - expected_distance).abs().item() < 1e-6

    # --- 5. end-to-end scalar ---
    transformed = transform_joint_anchor_boundary_sets_gnn_only(
        graph_bases, ts, boundary_sets, canonical_means, coefs_all, cluster_ids_all,
    )
    end_to_end_loss = joint_anchor_loss(transformed, boundary_sets.canonical_distance, huber_delta=0.075)
    print(f"[end-to-end] transformed.shape={tuple(transformed.shape)} loss={end_to_end_loss.item():.3e}")
    assert transformed.shape == (1, 2, num_frames, 3)
    assert torch.isfinite(end_to_end_loss)

    print("OK")

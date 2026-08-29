"""
flow3d/analysis/loss_joint.py

Joint anchor loss: fixes one 3D anchor point per side of each retained
cluster-pair "joint" (canonical mean of a boundary-Gaussian set on each
side), then, every frame, transforms each side's anchor with that cluster's
own per-frame coarse transform and measures the distance between the two
transformed anchors against the fixed canonical (pre-transform)
anchor-to-anchor distance with a Huber loss.

Which pairs get an anchor: edges.pt, not cluster_pairs.py
------------------------------------------------------------
flow3d/trainer.py builds its JointAnchors via build_joint_anchors_from_edges,
which reads flow3d/analysis/build_cluster_graph.py's edges.pt -- the SAME
file already required for --graph-coupling-path (the GNN's message-passing
topology, see flow3d/graph_relative_linear_attention.py) and optionally for
optim_cfg.rigidity_graph_path (rigidity_graph_type="cluster_graph_file").
Restricting the joint-anchor loss to exactly the edges that graph already
treats as connected -- rather than flow3d/analysis/cluster_pairs.py's
separately-computed fixed_boundary_indices.pt -- matters because the two
detectors can disagree: cluster_pairs.py's single-canonical-frame kNN
heuristic has been observed to attach a cluster to spatially-close-but-
physically-unrelated neighbors (e.g. two clusters that happen to touch only
in the rest pose), while build_cluster_graph.py's edges.pt is validated
across all frames (gap_median/gap_confidence, "kept_all_frames_contact") and
supports manual force_include_pairs/force_exclude_pairs overrides for
exactly this failure mode. Training on a false-positive pair doesn't just
waste weight on a wrong constraint -- it actively fights the correct
constraints on the same cluster (a cluster's coarse transform is one rigid
transform per frame; contradictory anchor pulls on it can prevent *any* of
its real joints from converging). build_joint_anchors (the older
cluster_pairs.py-based loader) is kept below for standalone/diagnostic use
and is not called by flow3d/trainer.py.

The coarse transform used is whatever motion_bases.compute_transforms_coarse
returns -- for any of flow3d/graph_coupling.py's or
flow3d/graph_relative_linear_attention.py's *GraphCorrectedScalableMotionBases
variants that already includes the per-frame GNN correction (composed the
same way as flow3d/analysis/loss.py's gnn_correction_* losses see it via
motion_bases.last_correction), and for plain flow3d/params.py's
ScalableMotionBases it's just the rigid coarse transform. No changes to
graph_relative_linear_attention.py (or any other graph_*.py variant) are
needed for that: compute_transforms_coarse(ts, cluster_ids) is already part
of every motion_bases class's public API.

Why this only penalizes seam-opening, not articulation
--------------------------------------------------------
Each side's anchor already sits at (approximately) the physical joint
location, since it's built by averaging the boundary Gaussians nearest the
other cluster. If cluster A and cluster B rotate relative to each other
about an axis passing through that shared location (ordinary articulation,
e.g. an elbow bending), each transformed anchor stays close to the shared
joint location regardless of how much rotation is applied -- see the
__main__ block below for the exact identity (anchor at the rotation center
-> transformed anchor == the center, independent of the rotation). The
distance between the two transformed anchors only grows when the two
clusters' transforms pull the two anchors apart in a way rotation-about-the-
joint doesn't explain, i.e. when the seam between them actually opens (e.g.
an unconstrained per-cluster GNN translation correction pulling one side
away from the other).

Three pieces, split the same way flow3d/analysis/loss.py splits its three
losses from trainer.py's motion_bases access:
1. build_joint_anchors_from_edges: I/O + averaging, edges.pt source. Loads
   build_cluster_graph.py's edges.pt and, for every kept edge, averages the
   canonical positions of its own boundary_global_indices_a/b (that script's
   multi-frame-validated boundary set) into one anchor point per side. This
   is what flow3d/trainer.py actually calls.
1b. build_joint_anchors: the older cluster_pairs.py-based loader, averaging
   fixed_pair_global_indices_a/b from fixed_boundary_indices.pt. Not used by
   flow3d/trainer.py; kept for standalone/diagnostic use.
   Both return fixed tensors meant to be cached once by the caller
   (flow3d/trainer.py) and reused every step -- the reference distance has
   to stay fixed for the Huber loss to have a stable target, so this is
   intentionally *not* recomputed from the live (possibly still-optimizing)
   means every step.
2. transform_joint_anchors + joint_anchor_loss: the per-step math. The
   former is a thin, motion_bases-class-agnostic wrapper around
   compute_transforms_coarse (kept out of trainer.py just to avoid
   repeating the homogeneous-coordinate einsum boilerplate at the call
   site); the latter is pure tensor math (no I/O, no motion_bases access),
   exactly like every function in loss.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "JointAnchors",
    "build_joint_anchors",
    "build_joint_anchors_from_edges",
    "transform_joint_anchors",
    "joint_anchor_loss",
]


@dataclass
class JointAnchors:
    """Fixed per-pair joint-anchor targets, built once by build_joint_anchors
    and cached by the caller (flow3d/trainer.py) for the rest of training.

    :param cluster_ids: (P, 2) long. [:, 0] is cluster_a's raw id, [:, 1] is
        cluster_b's, in the same indexing motion_bases uses (i.e.
        model.fg.get_cluster_ids() space).
    :param anchor_points: (P, 2, 3) float. Canonical (pre-transform) anchor
        position per side; anchor_points[:, 0] belongs to cluster_ids[:, 0]
        and anchor_points[:, 1] to cluster_ids[:, 1].
    :param canonical_distance: (P,) float. ||anchor_a - anchor_b|| in
        canonical space -- the fixed Huber-loss target.
    """

    cluster_ids: torch.Tensor
    anchor_points: torch.Tensor
    canonical_distance: torch.Tensor

    @property
    def num_pairs(self) -> int:
        return int(self.cluster_ids.shape[0])

    def to(self, device: torch.device) -> "JointAnchors":
        return JointAnchors(
            cluster_ids=self.cluster_ids.to(device),
            anchor_points=self.anchor_points.to(device),
            canonical_distance=self.canonical_distance.to(device),
        )


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_long_indices(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


def build_joint_anchors(
    pair_file: str | Path,
    canonical_means: torch.Tensor,
    device: torch.device | None = None,
) -> JointAnchors:
    """
    Build fixed joint-anchor targets from
    flow3d/analysis/cluster_pairs.py's fixed_boundary_indices.pt.

    :param pair_file: path to fixed_boundary_indices.pt: a dict keyed by
        "{cluster_a}_{cluster_b}", each entry holding at least "cluster_a",
        "cluster_b", "fixed_pair_global_indices_a",
        "fixed_pair_global_indices_b" (see cluster_pairs.py's
        build_fixed_correspondences).
    :param canonical_means: (G, 3) canonical foreground Gaussian means the
        saved indices index into (e.g. model.fg.params["means"].detach()).
        Passed in rather than re-derived so the caller controls exactly
        which (fixed) means snapshot the anchors are averaged from.
    :param device: device for the returned tensors. Default:
        canonical_means's device.
    :return: JointAnchors, one row per pair whose saved fixed_pair
        correspondence set is non-empty (a pair with no close-point
        correspondence saved is skipped -- there's nothing to average).
    :raises RuntimeError: if no pair in the file has a usable
        correspondence set.
    """
    if device is None:
        device = canonical_means.device

    payload = _torch_load(pair_file)
    if not isinstance(payload, dict):
        raise TypeError(
            f"{pair_file} must contain a dict keyed by pair (see cluster_pairs.py)."
        )

    means_cpu = canonical_means.detach().to("cpu")

    cluster_id_rows: list[tuple[int, int]] = []
    anchor_rows: list[torch.Tensor] = []

    required_keys = (
        "cluster_a",
        "cluster_b",
        "fixed_pair_global_indices_a",
        "fixed_pair_global_indices_b",
    )
    for pair_key, entry in payload.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Pair entry {pair_key!r} in {pair_file} is not a dict.")
        missing = [key for key in required_keys if key not in entry]
        if missing:
            raise KeyError(f"Pair entry {pair_key!r} in {pair_file} is missing {missing}.")

        idx_a = _as_long_indices(entry["fixed_pair_global_indices_a"])
        idx_b = _as_long_indices(entry["fixed_pair_global_indices_b"])
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue

        anchor_a = means_cpu[idx_a].mean(dim=0)
        anchor_b = means_cpu[idx_b].mean(dim=0)

        cluster_id_rows.append((int(entry["cluster_a"]), int(entry["cluster_b"])))
        anchor_rows.append(torch.stack([anchor_a, anchor_b], dim=0))

    if not cluster_id_rows:
        raise RuntimeError(
            f"No pair in {pair_file} had a usable fixed_pair correspondence -- "
            "cannot build any joint anchor."
        )

    return _finalize_joint_anchors(cluster_id_rows, anchor_rows, device)


def _finalize_joint_anchors(
    cluster_id_rows: list[tuple[int, int]],
    anchor_rows: list[torch.Tensor],
    device: torch.device,
) -> JointAnchors:
    cluster_ids = torch.tensor(cluster_id_rows, dtype=torch.long, device=device)
    anchor_points = torch.stack(anchor_rows, dim=0).to(device)  # (P, 2, 3)
    canonical_distance = (anchor_points[:, 0] - anchor_points[:, 1]).norm(dim=-1)  # (P,)
    return JointAnchors(
        cluster_ids=cluster_ids,
        anchor_points=anchor_points,
        canonical_distance=canonical_distance,
    )


def build_joint_anchors_from_edges(
    edges_path: str | Path,
    canonical_means: torch.Tensor,
    device: torch.device | None = None,
) -> JointAnchors:
    """
    Build fixed joint-anchor targets from
    flow3d/analysis/build_cluster_graph.py's edges.pt -- the SAME file
    already required for --graph-coupling-path (see this module's docstring
    for why using this file, rather than cluster_pairs.py's
    fixed_boundary_indices.pt, matters: it restricts the loss to exactly the
    edges the GNN's message passing already treats as connected, instead of
    an independently-computed adjacency detection that can disagree with it).

    edges.pt's edges_kept entries each already carry their own
    boundary_global_indices_a/b (build_cluster_graph.py's multi-frame,
    confidence-weighted gap analysis -- see that script's module docstring),
    so unlike build_joint_anchors there's no separate nearest-neighbour
    correspondence step: each side's anchor is simply the mean of its own
    boundary set.

    :param edges_path: path to build_cluster_graph.py's edges.pt (a dict with
        an "edges_kept" list; each entry holds at least "cluster_a",
        "cluster_b", "boundary_global_indices_a", "boundary_global_indices_b").
    :param canonical_means: (G, 3) canonical foreground Gaussian means the
        saved indices index into (e.g. model.fg.params["means"].detach()).
        edges.pt's boundary_global_indices_a/b are Gaussian indices into this
        same array regardless of build_cluster_graph.py's --position-source
        (it always resolves them back to model.fg.get_cluster_ids() space
        before saving -- see ClusterInfo.global_indices there).
    :param device: device for the returned tensors. Default:
        canonical_means's device.
    :return: JointAnchors, one row per kept edge with a non-empty boundary
        set on both sides (an edge with an empty side, though not expected
        in practice, is skipped rather than crashing -- nothing to average).
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
    anchor_rows: list[torch.Tensor] = []

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
        anchor_rows.append(torch.stack([anchor_a, anchor_b], dim=0))

    if not cluster_id_rows:
        raise RuntimeError(
            f"No kept edge in {edges_path} had a usable boundary set on both "
            "sides -- cannot build any joint anchor."
        )

    return _finalize_joint_anchors(cluster_id_rows, anchor_rows, device)


def transform_joint_anchors(
    motion_bases: Any,
    ts: torch.Tensor,
    anchors: JointAnchors,
) -> torch.Tensor:
    """
    Apply each side's own per-frame coarse transform to its anchor point.

    :param motion_bases: anything exposing
        compute_transforms_coarse(ts, cluster_ids) -> (N, B, 3, 4), i.e.
        flow3d/params.py's ScalableMotionBases or any
        *GraphCorrectedScalableMotionBases (e.g.
        flow3d/graph_relative_linear_attention.py's
        RelativeVelLinearAttentionGraphCorrectedScalableMotionBases). The
        GNN correction (if any) is composed internally by that call --
        nothing GNN-variant-specific happens here.
    :param ts: (B,) frame indices.
    :param anchors: JointAnchors with P pairs.
    :return: (P, 2, B, 3) transformed anchor positions; [:, 0] is
        cluster_ids[:, 0]'s anchor, [:, 1] is cluster_ids[:, 1]'s.
    """
    num_pairs = anchors.num_pairs
    cluster_ids_flat = anchors.cluster_ids.reshape(-1)  # (2P,): [a0, b0, a1, b1, ...]
    transforms = motion_bases.compute_transforms_coarse(ts, cluster_ids_flat)  # (2P, B, 3, 4)

    anchor_points_flat = anchors.anchor_points.reshape(-1, 3)  # (2P, 3)
    anchor_homog = F.pad(anchor_points_flat, (0, 1), value=1.0)  # (2P, 4)

    positions = torch.einsum("nbij,nj->nbi", transforms, anchor_homog)  # (2P, B, 3)
    num_frames = positions.shape[1]
    return positions.view(num_pairs, 2, num_frames, 3)


def joint_anchor_loss(
    transformed_anchors: torch.Tensor,
    canonical_distance: torch.Tensor,
    huber_delta: float = 0.01,
) -> torch.Tensor:
    """
    Huber loss between the per-frame transformed anchor-to-anchor distance
    and the fixed canonical anchor-to-anchor distance.

    :param transformed_anchors: (P, 2, B, 3), see transform_joint_anchors.
    :param canonical_distance: (P,), JointAnchors.canonical_distance.
    :param huber_delta: Huber transition point, in the same length units as
        canonical_distance (world/canonical scene units). Below this the
        penalty is quadratic; above it, linear -- so a few frames with a
        very open seam don't dominate the gradient relative to many frames
        with a small one.
    :return: scalar; exactly 0.0 if there are no pairs (P == 0).
    """
    if transformed_anchors.shape[0] == 0:
        return canonical_distance.new_zeros(())

    dist = (transformed_anchors[:, 0] - transformed_anchors[:, 1]).norm(dim=-1)  # (P, B)
    target = canonical_distance[:, None].expand_as(dist)
    return F.huber_loss(dist, target, delta=huber_delta)


if __name__ == "__main__":
    # Sanity checks:
    #   1. build_joint_anchors round-trips a saved fixed_boundary_indices.pt-shaped
    #      file into the expected anchor points / canonical distance.
    #   1b. build_joint_anchors_from_edges round-trips a saved edges.pt-shaped
    #      file (edges_kept list) the same way, ignoring cut edges.
    #   2. canonical (identity transform) -> loss == 0.
    #   3. pure articulation about a shared joint center -> loss == 0, for ANY
    #      relative rotation between the two clusters (the key claim of this file).
    #   4. an actual seam-opening translation -> loss > 0.
    #   5. joint_anchor_loss handles P == 0 gracefully (0.0, no crash).
    import tempfile

    from flow3d.params import ScalableMotionBases

    torch.manual_seed(0)

    # --- 1. build_joint_anchors round-trip ---
    num_fg = 20
    canonical_means = torch.randn(num_fg, 3)
    # pair (cluster 0, cluster 1): correspondence indices 0..2 (cluster 0 side)
    # and 3..5 (cluster 1 side), interpreted the same way
    # cluster_pairs.py's build_fixed_correspondences produces them (parallel arrays).
    fixed_pair_a = torch.tensor([0, 1, 2], dtype=torch.long)
    fixed_pair_b = torch.tensor([3, 4, 5], dtype=torch.long)
    saved_pairs = {
        "0_1": {
            "cluster_a": 0,
            "cluster_b": 1,
            "boundary_global_indices_a": fixed_pair_a,
            "boundary_global_indices_b": fixed_pair_b,
            "fixed_pair_global_indices_a": fixed_pair_a,
            "fixed_pair_global_indices_b": fixed_pair_b,
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(saved_pairs, f.name)
        anchors = build_joint_anchors(f.name, canonical_means)

    expected_anchor_a = canonical_means[fixed_pair_a].mean(dim=0)
    expected_anchor_b = canonical_means[fixed_pair_b].mean(dim=0)
    expected_distance = (expected_anchor_a - expected_anchor_b).norm()
    print(
        f"[build] num_pairs={anchors.num_pairs} "
        f"anchor_a_err={(anchors.anchor_points[0, 0] - expected_anchor_a).abs().max().item():.3e} "
        f"anchor_b_err={(anchors.anchor_points[0, 1] - expected_anchor_b).abs().max().item():.3e} "
        f"dist_err={(anchors.canonical_distance[0] - expected_distance).abs().item():.3e}"
    )
    assert anchors.num_pairs == 1
    assert (anchors.anchor_points[0, 0] - expected_anchor_a).abs().max().item() < 1e-6
    assert (anchors.anchor_points[0, 1] - expected_anchor_b).abs().max().item() < 1e-6
    assert (anchors.canonical_distance[0] - expected_distance).abs().item() < 1e-6

    # --- 1b. build_joint_anchors_from_edges round-trip, edges.pt shape ---
    # Mirrors build_cluster_graph.py's saved format: a dict with an
    # "edges_kept" list (plus edges_cut/cluster_ids/meta, irrelevant here).
    # A "kept_manual_override" edge (like the 3-4/4-21 force_include_pairs
    # case that motivated this loader) and a "cut" edge that must be ignored.
    boundary_a = torch.tensor([0, 1, 2], dtype=torch.long)
    boundary_b = torch.tensor([3, 4, 5], dtype=torch.long)
    edges_payload = {
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edges_kept": [
            {
                "cluster_a": 0,
                "cluster_b": 1,
                "reason": "kept_manual_override",
                "boundary_global_indices_a": boundary_a,
                "boundary_global_indices_b": boundary_b,
            }
        ],
        "edges_cut": [
            {"cluster_a": 0, "cluster_b": 2, "reason": "cut_center_median_absolute_distance_too_high"}
        ],
        "cluster_ids": [0, 1, 2],
        "meta": {"force_include_pairs": [[0, 1]], "force_exclude_pairs": [[0, 2]]},
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(edges_payload, f.name)
        edge_anchors = build_joint_anchors_from_edges(f.name, canonical_means)

    print(
        f"[build_from_edges] num_pairs={edge_anchors.num_pairs} "
        f"anchor_a_err={(edge_anchors.anchor_points[0, 0] - expected_anchor_a).abs().max().item():.3e} "
        f"anchor_b_err={(edge_anchors.anchor_points[0, 1] - expected_anchor_b).abs().max().item():.3e}"
    )
    assert edge_anchors.num_pairs == 1, "only the single edges_kept entry should produce an anchor"
    assert (edge_anchors.anchor_points[0, 0] - expected_anchor_a).abs().max().item() < 1e-6
    assert (edge_anchors.anchor_points[0, 1] - expected_anchor_b).abs().max().item() < 1e-6
    assert tuple(edge_anchors.cluster_ids[0].tolist()) == (0, 1)

    # --- toy 2-cluster ScalableMotionBases for the transform-level checks ---
    num_clusters, num_frames = 2, 4
    joint_center = torch.tensor([1.0, 2.0, 3.0])
    centers = joint_center[None, :].repeat(num_clusters, 1)  # both clusters pivot at the joint
    rots = torch.zeros(num_clusters, num_frames, 6)
    rots[..., 0] = 1.0
    rots[..., 4] = 1.0  # identity 6D rotation (first two columns of I_3)
    transls = torch.zeros(num_clusters, num_frames, 3)
    fine_rots = rots[:, None, :, :].clone()  # (C, 1, T, 6), unused by compute_transforms_coarse
    fine_transls = torch.zeros(num_clusters, 1, num_frames, 3)
    bases = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    ts = torch.arange(num_frames)

    # anchors sit exactly at the shared joint center on both sides
    anchors_at_joint = JointAnchors(
        cluster_ids=torch.tensor([[0, 1]], dtype=torch.long),
        anchor_points=torch.stack([joint_center, joint_center], dim=0)[None],  # (1, 2, 3)
        canonical_distance=torch.zeros(1),
    )

    # --- 2. identity transform -> loss == 0 ---
    transformed = transform_joint_anchors(bases, ts, anchors_at_joint)
    loss_identity = joint_anchor_loss(transformed, anchors_at_joint.canonical_distance)
    print(f"[identity] loss={loss_identity.item():.3e}")
    assert loss_identity.item() == 0.0

    # --- 3. pure articulation about the shared joint center -> loss == 0 ---
    # give cluster 0 and cluster 1 two DIFFERENT random rotations about the
    # same shared pivot (joint_center) -- an anchor sitting exactly at the
    # rotation center is a fixed point of R(.) + (center - R@center), so it
    # maps to itself regardless of R, for both clusters independently.
    from flow3d.transforms import rmat_to_cont_6d

    random_rotmats = torch.linalg.qr(torch.randn(num_clusters, num_frames, 3, 3))[0]
    # guarantee proper rotations (det == +1), not reflections
    det = torch.linalg.det(random_rotmats)
    random_rotmats[..., 0] *= det.sign()[..., None]
    articulated_rots = rmat_to_cont_6d(random_rotmats)
    bases_articulated = ScalableMotionBases(centers, articulated_rots, transls, fine_rots, fine_transls)

    transformed_art = transform_joint_anchors(bases_articulated, ts, anchors_at_joint)
    loss_articulated = joint_anchor_loss(transformed_art, anchors_at_joint.canonical_distance)
    print(
        f"[articulation] max |transformed - joint_center| = "
        f"{(transformed_art - joint_center).abs().max().item():.3e}  loss={loss_articulated.item():.3e}"
    )
    assert (transformed_art - joint_center).abs().max().item() < 1e-4
    assert loss_articulated.item() < 1e-8

    # --- 4. an actual seam-opening translation -> loss > 0 ---
    opened_transls = transls.clone()
    opened_transls[1] += 0.5  # cluster 1 translates away from cluster 0
    bases_opened = ScalableMotionBases(centers, rots, opened_transls, fine_rots, fine_transls)
    transformed_opened = transform_joint_anchors(bases_opened, ts, anchors_at_joint)
    loss_opened = joint_anchor_loss(transformed_opened, anchors_at_joint.canonical_distance, huber_delta=0.01)
    print(f"[seam-opening] loss={loss_opened.item():.3e} (expect > 0)")
    assert loss_opened.item() > 0.0

    # --- 5. P == 0 handled gracefully ---
    empty_transformed = torch.zeros(0, 2, num_frames, 3)
    empty_distance = torch.zeros(0)
    loss_empty = joint_anchor_loss(empty_transformed, empty_distance)
    print(f"[empty] loss={loss_empty.item():.3e} (expect 0)")
    assert loss_empty.item() == 0.0

    print("OK")

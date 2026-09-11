"""
Extract persistent, world-space 3D motion tracks for one DAVIS sequence.

Step 1 of the skeleton-extraction pipeline: turns raw CoTracker 2D tracks +
MegaSAM depth/camera into world-space 3D trajectories that keep the same
track id across every frame of the video. No trained checkpoint, Gaussian
pose, motion basis, or existing cluster id is used or produced here.
FPS/temporal subsampling, rigid-part merging, and skeleton node/edge
construction are later pipeline steps and are out of scope for this script.

Tracks are seeded from multiple query frames (e.g. 0, 10, 20, ...) rather
than a single one, so that points occluded at one query frame (legs, most
often) can still be captured from another. Per-query foreground tracks are
then merged across queries: two tracks from different queries are fused
into one only if they are a mutual nearest-neighbor pair whose 2D and 3D
trajectories stay close over their *entire* commonly-valid overlap (not
merely close at one shared frame) and that overlap is long enough to be
meaningful.

Output
------
outputs/davis/<seq-name>/skeleton/motion_skeleton/
    raw_tracks_3d.pt
    preview_first_frame<NNNNN>.png
    preview_mid_frame<NNNNN>.png
    preview_last_frame<NNNNN>.png
    trajectories_3d.png
    report.json

Example
-------
    python flow3d/skeleton/extract_motion_tracks.py --seq-name camel --overwrite
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # skeleton/ -> flow3d/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from matplotlib import colormaps
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.data.utils import (
    get_tracks_3d_for_query_frame,
    normalize_coords,
    parse_cotracker3_track_info,
    to_serializable,
)

# Not imported: flow3d.vis.utils (heavy nvdiffrast/viser module-level side effect for
# trivial helper functions; reimplemented locally below), flow3d.scene_model,
# flow3d.params, flow3d.renderer, flow3d.trainer, or any checkpoint/motion-basis/
# cluster-id code.

# Per-track fields carried through foreground-pool filtering and per-group fusion.
# ("positions_world" is deliberately excluded -- it is rebuilt post-fusion from the
# fused "positions_world_raw" + fused "valid_mask".)
POOL_FIELDS = [
    "positions_2d",
    "positions_world_raw",
    "valid_mask",
    "confidence",
    "sampled_depth",
    "in_bounds_mask",
    "depth_finite_mask",
    "depth_positive_mask",
    "fg_mask_valid",
    "depth_mask_valid",
    "visible_mask",
    "confidence_mask",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--query-frame-stride", type=int, default=10,
        help="Query frames = range(0, num_frames, stride).",
    )
    parser.add_argument(
        "--query-frames", type=str, default=None,
        help="Explicit comma-separated query frame list, e.g. '0,15,30,45'. "
             "Overrides --query-frame-stride.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--stable-ratio", type=float, default=0.5)
    parser.add_argument(
        "--min-valid-frames", type=int, default=15,
        help="A track is stable if valid_ratio >= --stable-ratio OR valid_count >= this "
             "(rescues frequently-occluded-but-long-duration tracks, e.g. legs).",
    )
    parser.add_argument(
        "--min-overlap-frames", type=int, default=5,
        help="Minimum commonly-valid frame count for a cross-query track pair to be "
             "considered for merging.",
    )
    parser.add_argument(
        "--merge-max-2d-px", type=float, default=3.0,
        help="Max 2D pixel distance over the entire common-valid overlap for a "
             "cross-query pair to be merge-eligible.",
    )
    parser.add_argument(
        "--merge-max-3d-dist", type=float, default=0.02,
        help="Max 3D distance over the entire common-valid overlap, in the dataset's "
             "scene-normalized world units (scene-scale dependent; inspect "
             "trajectories_3d.png's axis range to judge whether this needs retuning).",
    )
    parser.add_argument(
        "--merge-candidate-topk", type=int, default=8,
        help="Coarse candidate width per track for cross-query matching (performance knob).",
    )
    parser.add_argument(
        "--reproj-atol-px", type=float, default=1e-3,
        help="Camera-math round-trip sanity tolerance (NOT a geometry-quality metric).",
    )
    parser.add_argument(
        "--parity-atol", type=float, default=1e-3,
        help="World-unit tolerance vs. the get_tracks_3d_for_query_frame oracle.",
    )
    parser.add_argument("--num-preview-tracks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_output_dir(seq_name: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return _REPO_ROOT / "outputs" / "davis" / seq_name / "skeleton" / "motion_skeleton"


def resolve_config_path(config: Path) -> Path:
    return config if config.is_absolute() else (_REPO_ROOT / config).resolve()


def load_dataset(config_path: Path, seq_name: str) -> tuple[CasualDataset, dict]:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(
        root_dir=scene_cfg["data_dir"],
        seq_name=seq_name,
        # Reuse the cached scene_norm_dict (scale + transform baked into w2cs/depth)
        # instead of recomputing it from a nondeterministic random sample every run
        # and overwriting the shared cache file training depends on. Matches the
        # choice made by flow3d/analysis/build_cluster_graph.py's raw-track loader.
        load_from_cache=True,
        **scene_cfg.get("data", {}),
    )
    dataset = CasualDataset(**asdict(data_cfg))
    return dataset, scene_cfg


def generate_query_frames(num_frames: int, stride: int, explicit: str | None) -> list[int]:
    if explicit:
        frames = sorted({int(x) for x in explicit.split(",") if x.strip() != ""})
    else:
        frames = list(range(0, num_frames, stride))
    if not frames:
        raise ValueError("no query frames resolved")
    for q in frames:
        if not (0 <= q < num_frames):
            raise ValueError(f"query frame {q} out of range [0, {num_frames})")
    return frames


def compute_world_tracks(
    dataset: CasualDataset, query_frame: int, min_confidence: float, device: torch.device
) -> dict:
    T = dataset.num_frames
    H, W = dataset.get_image(0).shape[:2]
    target_idcs = list(range(T))

    # (N, T, 4): dim=1 default stacks per-frame (N, 4) arrays along a new time axis.
    raw, valid_visible_nt, _valid_invisible_nt, confidence_nt = dataset.load_target_tracks(
        query_frame, target_idcs, return_vis=True
    )
    positions_2d = raw[..., :2]  # (N, T, 2), CPU
    tracks_2d_tn = positions_2d.to(device).swapaxes(0, 1)  # (T, N, 2)
    valid_visible_tn = valid_visible_nt.to(device).swapaxes(0, 1)  # (T, N)
    confidence_tn = confidence_nt.to(device).swapaxes(0, 1)  # (T, N)

    depths = torch.stack([dataset.get_depth(i) for i in target_idcs], dim=0).to(device)  # (T, H, W)
    fg_masks = torch.stack([dataset.get_mask(i) for i in target_idcs], dim=0).to(device)  # (T, H, W)
    depth_masks = (
        torch.stack([dataset.get_depth_mask(i) for i in target_idcs], dim=0).float().to(device)
    )  # (T, H, W)

    Ks = dataset.Ks[target_idcs].to(device)  # (T, 3, 3)
    w2cs = dataset.w2cs[target_idcs].to(device)  # (T, 4, 4)
    inv_Ks = torch.linalg.inv(Ks)
    c2ws = torch.linalg.inv(w2cs)

    x, y = tracks_2d_tn[..., 0], tracks_2d_tn[..., 1]
    in_bounds_tn = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)  # (T, N)

    coords_norm = normalize_coords(tracks_2d_tn[:, None], H, W)  # (T, 1, N, 2)

    # Backprojection copied from flow3d.data.utils.get_tracks_3d_for_query_frame
    # (utils.py:150-167): bilinear depth sample, align_corners=True, border padding.
    sampled_depth_tn = F.grid_sample(
        depths[:, None], coords_norm, align_corners=True, padding_mode="border",
    )[:, 0, 0]  # (T, N)
    cam_pts_tn = (
        torch.einsum("tij,tnj->tni", inv_Ks, F.pad(tracks_2d_tn, (0, 1), value=1.0))
        * sampled_depth_tn[..., None]
    )
    world_pts_raw_tn = torch.einsum(
        "tij,tnj->tni", c2ws, F.pad(cam_pts_tn, (0, 1), value=1.0)
    )[..., :3]  # (T, N, 3), unfiltered

    depth_finite_tn = torch.isfinite(sampled_depth_tn)
    depth_positive_tn = sampled_depth_tn > 0

    # Nearest-neighbor mask lookup (deviates from get_tracks_3d_for_query_frame's
    # bilinear is_in_masks): a hard "which pixel does this track sit on" read,
    # rather than a boundary-blended interpolation. align_corners=True kept
    # identical to the reference to preserve the same pixel-grid convention.
    fg_mask_valid_tn = (
        F.grid_sample(fg_masks[:, None], coords_norm, mode="nearest", align_corners=True)[:, 0, 0] == 1
    )
    depth_mask_valid_tn = (
        F.grid_sample(depth_masks[:, None], coords_norm, mode="nearest", align_corners=True)[:, 0, 0]
        == 1
    )
    confidence_mask_tn = confidence_tn >= min_confidence

    valid_mask_tn = (
        in_bounds_tn
        & depth_finite_tn
        & depth_positive_tn
        & fg_mask_valid_tn
        & depth_mask_valid_tn
        & valid_visible_tn
        & confidence_mask_tn
    )

    world_pts_tn = world_pts_raw_tn.clone()
    world_pts_tn[~valid_mask_tn] = float("nan")

    def to_nt(t: torch.Tensor) -> torch.Tensor:
        return t.swapaxes(0, 1).cpu()

    return {
        "positions_world": to_nt(world_pts_tn),  # (N, T, 3), NaN where invalid
        "positions_world_raw": to_nt(world_pts_raw_tn),  # (N, T, 3), unfiltered (for parity check)
        "positions_2d": positions_2d,  # (N, T, 2), CPU
        "valid_mask": to_nt(valid_mask_tn),
        "confidence": confidence_nt,  # (N, T), CPU, raw
        "sampled_depth": to_nt(sampled_depth_tn),
        "in_bounds_mask": to_nt(in_bounds_tn),
        "depth_finite_mask": to_nt(depth_finite_tn),
        "depth_positive_mask": to_nt(depth_positive_tn),
        "fg_mask_valid": to_nt(fg_mask_valid_tn),
        "depth_mask_valid": to_nt(depth_mask_valid_tn),
        "visible_mask": valid_visible_nt,  # (N, T), CPU
        "confidence_mask": to_nt(confidence_mask_tn),
        "Ks": Ks.cpu(),
        "w2cs": w2cs.cpu(),
        # Kept only for that query's own parity check, not written to raw_tracks_3d.pt:
        "raw_tracks": raw,
        "depths": depths.cpu(),
        "fg_masks": fg_masks.cpu(),
        "depth_masks": depth_masks.cpu(),
    }


def filter_foreground_pool(tracks: dict, query_frame: int) -> dict:
    """The 'foreground track' pool for one query: rows valid (fg+depth+visible+confident)
    at their own query frame, mirroring get_tracks_3d_for_query_frame's own
    valid = is_in_masks[query_index] gate. Keeps the cross-query matching pool small."""
    fg_idx = torch.nonzero(tracks["valid_mask"][:, query_frame], as_tuple=False).squeeze(-1)
    pool = {k: tracks[k][fg_idx] for k in POOL_FIELDS}
    pool["query_frame"] = query_frame
    return pool


class _UnionFind:
    """Union-find where each root additionally tracks the set of query frames already
    present in its component. A union that would put two tracks from the SAME query
    into one component is rejected: two rows from one query are, by construction,
    two different pixel seeds -- transitively chaining them together (A matches B in
    query pair (0,10), B matches C in pair (10,20), C matches D in pair (0,20), where
    A and D both happen to come from query 0) would silently fuse distinct physical
    points into one track. Without this guard, dense regions (e.g. the torso) can
    produce runaway components spanning far more members than there are query frames."""

    def __init__(self, n: int, node_query: list[int]):
        self.parent = list(range(n))
        self.root_queries = [{node_query[i]} for i in range(n)]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def try_union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.root_queries[ra] & self.root_queries[rb]:
            return False
        self.parent[ra] = rb
        self.root_queries[rb] |= self.root_queries[ra]
        self.root_queries[ra] = None
        return True


def gather_candidates(
    pos_from: torch.Tensor, valid_from: torch.Tensor, world_from: torch.Tensor,
    pos_to: torch.Tensor, valid_to: torch.Tensor, world_to: torch.Tensor,
    ref_frame: int, topk: int, min_overlap_frames: int, max_2d_px: float, max_3d_dist: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarse-to-exact cross-query candidate generation. Stage 1 is a single-frame
    (Na, Nb) pixel-distance matrix at ref_frame (cheap even for Na, Nb ~ a few
    thousand); computing exact overlap/distance stats over all T frames for every
    (Na, Nb) pair is infeasible (Na*Nb*T blows up into terabytes), so only the
    stage-1 top-k candidates per row get the exact stage-2 treatment."""
    Na, Nb = pos_from.shape[0], pos_to.shape[0]
    k = min(topk, Nb)
    d0 = torch.cdist(pos_from[:, ref_frame], pos_to[:, ref_frame])  # (Na, Nb)
    topk_idx = d0.topk(k, dim=1, largest=False).indices  # (Na, k)

    g_valid = valid_to[topk_idx]  # (Na, k, T)
    g_pos = pos_to[topk_idx]  # (Na, k, T, 2)
    g_world = world_to[topk_idx]  # (Na, k, T, 3)

    common = valid_from[:, None, :] & g_valid  # (Na, k, T)
    overlap_count = common.sum(-1)  # (Na, k)

    d2d = (pos_from[:, None, :, :] - g_pos).norm(dim=-1)  # (Na, k, T)
    d3d = (world_from[:, None, :, :] - g_world).norm(dim=-1)  # (Na, k, T)

    mean_2d = (d2d * common).sum(-1) / overlap_count.clamp(min=1)
    max_2d = d2d.masked_fill(~common, -1.0).max(-1).values
    max_3d = d3d.masked_fill(~common, -1.0).max(-1).values

    eligible = (
        (overlap_count >= min_overlap_frames)
        & (max_2d >= 0)
        & (max_2d <= max_2d_px)
        & (max_3d <= max_3d_dist)
    )
    return topk_idx, eligible, mean_2d


def match_query_pair(
    pool_a: dict, pool_b: dict, topk: int, min_overlap_frames: int,
    max_2d_px: float, max_3d_dist: float, device: torch.device,
) -> list[tuple[int, int, float]]:
    """Mutual-nearest-neighbor pairs between two queries' foreground pools. Run in
    both directions (each using the *other* pool's own query frame as the reliable
    stage-1 reference) so a true match isn't missed purely because it fell outside
    one side's top-k coarse window."""
    Na, Nb = pool_a["positions_2d"].shape[0], pool_b["positions_2d"].shape[0]
    if Na == 0 or Nb == 0:
        return []

    pos_a = pool_a["positions_2d"].to(device)
    valid_a = pool_a["valid_mask"].to(device)
    world_a = pool_a["positions_world_raw"].to(device)
    pos_b = pool_b["positions_2d"].to(device)
    valid_b = pool_b["valid_mask"].to(device)
    world_b = pool_b["positions_world_raw"].to(device)

    idx_ab, elig_ab, score_ab = gather_candidates(
        pos_a, valid_a, world_a, pos_b, valid_b, world_b,
        pool_b["query_frame"], topk, min_overlap_frames, max_2d_px, max_3d_dist,
    )
    idx_ba, elig_ba, score_ba = gather_candidates(
        pos_b, valid_b, world_b, pos_a, valid_a, world_a,
        pool_a["query_frame"], topk, min_overlap_frames, max_2d_px, max_3d_dist,
    )

    inf = torch.tensor(float("inf"), device=device)
    best_b_for_a = torch.full((Na,), -1, dtype=torch.long, device=device)
    best_b_for_a_score = torch.full((Na,), float("inf"), device=device)
    for col in range(idx_ab.shape[1]):
        cand_j, elig, sc = idx_ab[:, col], elig_ab[:, col], score_ab[:, col]
        sc = torch.where(elig, sc, inf)
        better = sc < best_b_for_a_score
        best_b_for_a = torch.where(better, cand_j, best_b_for_a)
        best_b_for_a_score = torch.where(better, sc, best_b_for_a_score)

    best_a_for_b = torch.full((Nb,), -1, dtype=torch.long, device=device)
    best_a_for_b_score = torch.full((Nb,), float("inf"), device=device)
    for col in range(idx_ba.shape[1]):
        cand_i, elig, sc = idx_ba[:, col], elig_ba[:, col], score_ba[:, col]
        sc = torch.where(elig, sc, inf)
        better = sc < best_a_for_b_score
        best_a_for_b = torch.where(better, cand_i, best_a_for_b)
        best_a_for_b_score = torch.where(better, sc, best_a_for_b_score)

    has_partner = best_b_for_a >= 0
    partner = best_b_for_a.clamp(min=0)
    reciprocal = best_a_for_b[partner] == torch.arange(Na, device=device)
    mutual = has_partner & reciprocal

    a_idx = torch.nonzero(mutual, as_tuple=False).squeeze(-1)
    b_idx = best_b_for_a[a_idx]
    scores = best_b_for_a_score[a_idx]
    return list(zip(a_idx.tolist(), b_idx.tolist(), scores.tolist()))


def fuse_track_group(members: list[dict], T: int) -> dict:
    """Per frame: prefer a VALID member; among valid members pick the highest
    confidence; if no member is valid at that frame, still pick the highest-
    confidence member overall (a displayable, if invalid, value); ties broken by
    member order (== ascending query frame, since callers build `members` that way).
    valid_mask is the OR of every member's per-frame valid_mask."""
    M = len(members)
    valid_stack = torch.stack([m["valid_mask"] for m in members], dim=0)  # (M, T)
    conf_stack = torch.stack([m["confidence"] for m in members], dim=0)  # (M, T)

    BIG = 1e6
    tie_break = torch.arange(M, dtype=conf_stack.dtype)[:, None] * 1e-9
    score = conf_stack + valid_stack.float() * BIG - tie_break  # (M, T)
    best_idx = torch.argmax(score, dim=0)  # (T,)
    frame_idx = torch.arange(T)

    fused = {}
    for key in POOL_FIELDS:
        if key == "valid_mask":
            continue
        stacked = torch.stack([m[key] for m in members], dim=0)  # (M, T, ...)
        fused[key] = stacked[best_idx, frame_idx]
    fused["valid_mask"] = valid_stack.any(dim=0)
    return fused


def extract_and_merge_tracks(
    dataset: CasualDataset, query_frames: list[int], args: argparse.Namespace, device: torch.device
) -> tuple[dict, dict]:
    T = dataset.num_frames

    per_query_tracks: dict[int, dict] = {}
    pools: dict[int, dict] = {}
    parity_checks: dict[int, dict] = {}
    per_query_track_counts: dict[int, int] = {}

    for q in query_frames:
        tracks_q = compute_world_tracks(dataset, q, args.min_confidence, device)
        parity_checks[q] = run_parity_check(dataset, q, tracks_q, args.parity_atol)
        pool = filter_foreground_pool(tracks_q, q)
        pools[q] = pool
        per_query_tracks[q] = tracks_q
        per_query_track_counts[q] = int(pool["positions_2d"].shape[0])

    offsets: dict[int, int] = {}
    node_query: list[int] = []
    total_nodes = 0
    for q in query_frames:
        offsets[q] = total_nodes
        n_q = pools[q]["positions_2d"].shape[0]
        node_query.extend([q] * n_q)
        total_nodes += n_q
    uf = _UnionFind(total_nodes, node_query)

    # Collect mutual-NN edges from every query pair first, then apply them globally
    # in ascending-distance (best match first) order, skipping any edge that would
    # violate the one-track-per-query-per-component invariant (see _UnionFind).
    # Committing edges greedily by quality, rather than in arbitrary pair order,
    # means a strong match is never bumped by a weaker one that happens to be
    # processed first.
    all_edges: list[tuple[int, int, float]] = []
    for q_a, q_b in combinations(query_frames, 2):
        pairs = match_query_pair(
            pools[q_a], pools[q_b], args.merge_candidate_topk,
            args.min_overlap_frames, args.merge_max_2d_px, args.merge_max_3d_dist, device,
        )
        for i, j, score in pairs:
            all_edges.append((offsets[q_a] + i, offsets[q_b] + j, score))

    all_edges.sort(key=lambda e: e[2])
    for global_a, global_b, _score in all_edges:
        uf.try_union(global_a, global_b)

    groups: dict[int, list[tuple[int, int]]] = {}
    for q in query_frames:
        for local_i in range(pools[q]["positions_2d"].shape[0]):
            root = uf.find(offsets[q] + local_i)
            groups.setdefault(root, []).append((q, local_i))

    fused_list = []
    source_query_frames: list[list[int]] = []
    component_sizes = []
    for members in groups.values():
        members = sorted(members, key=lambda qi: qi[0])  # ascending query frame, for deterministic tie-break
        member_records = [{k: pools[q][k][i] for k in POOL_FIELDS} for (q, i) in members]
        fused_list.append(fuse_track_group(member_records, T))
        source_query_frames.append([q for q, _ in members])
        component_sizes.append(len(members))

    N_final = len(fused_list)
    fused = {
        key: torch.stack([f[key] for f in fused_list], dim=0) if N_final > 0
        else torch.empty((0, T) + ({"positions_2d": (2,), "positions_world_raw": (3,)}.get(key, ())))
        for key in POOL_FIELDS
    }
    fused["positions_world"] = fused["positions_world_raw"].clone()
    fused["positions_world"][~fused["valid_mask"]] = float("nan")
    fused["Ks"] = dataset.Ks.cpu()
    fused["w2cs"] = dataset.w2cs.cpu()

    coverage_per_frame = fused["valid_mask"].sum(dim=0).tolist() if N_final > 0 else [0] * T
    coverage_per_frame_query0 = pools[query_frames[0]]["valid_mask"].sum(dim=0).tolist()

    size_hist = {"1": 0, "2": 0, "3+": 0}
    for s in component_sizes:
        size_hist["3+" if s >= 3 else str(s)] += 1

    merge_info = {
        "query_frames": query_frames,
        "per_query_track_counts": {str(q): c for q, c in per_query_track_counts.items()},
        "total_tracks_before_merge": sum(per_query_track_counts.values()),
        "total_tracks_after_merge": N_final,
        "merge_component_size_histogram": size_hist,
        "coverage_per_frame": coverage_per_frame,
        "coverage_per_frame_query0_baseline": coverage_per_frame_query0,
        "parity_checks": {str(q): s for q, s in parity_checks.items()},
        "source_query_frames": source_query_frames,
    }
    return fused, merge_info


def compute_stable_mask(valid_mask: torch.Tensor, stable_ratio: float, min_valid_frames: int) -> torch.Tensor:
    valid_count = valid_mask.sum(dim=1)
    return (valid_count.float() / valid_mask.shape[1] >= stable_ratio) | (valid_count >= min_valid_frames)


def _project_2d_tracks(
    positions_world_tn: torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor
) -> torch.Tensor:
    """Local reimplementation of flow3d.vis.utils.project_2d_tracks (same math),
    to avoid that module's heavy nvdiffrast/viser import for this one function.
    positions_world_tn: (T, N, 3); Ks: (T, 3, 3); w2cs: (T, 4, 4) -> (T, N, 2)."""
    pts_cam = torch.einsum("tij,tnj->tni", w2cs, F.pad(positions_world_tn, (0, 1), value=1.0))[..., :3]
    pts_img = torch.einsum("tij,tnj->tni", Ks, pts_cam)
    return pts_img[..., :2] / torch.clamp(pts_img[..., 2:], min=1e-5)


def verify_reprojection(tracks: dict, atol_px: float) -> dict:
    """Backproject->reproject is a mathematical identity for ANY depth value as long
    as K/w2c are applied consistently in both directions, so a wrong sampled depth
    still reprojects onto the original pixel. This can only catch bugs in the camera
    math itself (wrong frame indexed, transposed matrix, wrong w2c/c2w direction) —
    never a depth/backprojection-quality problem. See run_parity_check for that."""
    positions_world_tn = tracks["positions_world_raw"].swapaxes(0, 1)  # (T, N, 3)
    reproj_tn = _project_2d_tracks(positions_world_tn, tracks["Ks"], tracks["w2cs"])
    reproj_nt = reproj_tn.swapaxes(0, 1)  # (N, T, 2)

    err = (reproj_nt - tracks["positions_2d"]).norm(dim=-1)  # (N, T)
    err_valid = err[tracks["in_bounds_mask"]]
    return {
        "num_evaluated": int(tracks["in_bounds_mask"].sum().item()),
        "mean_px": float(err_valid.mean().item()) if err_valid.numel() else float("nan"),
        "median_px": float(err_valid.median().item()) if err_valid.numel() else float("nan"),
        "p95_px": float(err_valid.quantile(0.95).item()) if err_valid.numel() else float("nan"),
        "max_px": float(err_valid.max().item()) if err_valid.numel() else float("nan"),
        "atol_px": atol_px,
        "passed": bool(err_valid.numel() == 0 or err_valid.max().item() <= atol_px),
        "note": (
            "Backproject->reproject is an identity for any depth; this checks only "
            "camera-math wiring (K/w2c indexing), not depth/geometry accuracy. "
            "See parity_check for the geometric-correctness test."
        ),
    }


def run_parity_check(dataset: CasualDataset, query_frame: int, tracks: dict, parity_atol: float) -> dict:
    """Cross-check our reimplemented backprojection against the trusted, training-used
    flow3d.data.utils.get_tracks_3d_for_query_frame oracle, on the subset of tracks it
    selects, comparing raw (pre-NaN) 3D positions -- the actual geometric-correctness
    test (unaffected by the nearest-vs-bilinear mask-sampling difference above, since
    that only changes which N-subset each side selects, not the 3D math itself)."""
    depths, fg_masks, depth_masks = tracks["depths"], tracks["fg_masks"], tracks["depth_masks"]
    raw = tracks["raw_tracks"]
    T, H, W = depths.shape

    combined_masks = (fg_masks == 1).float() * (depth_masks == 1).float()  # (T, H, W)
    inv_Ks = torch.linalg.inv(tracks["Ks"])
    c2ws = torch.linalg.inv(tracks["w2cs"])
    query_img = dataset.get_image(query_frame)

    oracle_tracks_3d, _colors, _vis, _invis, _conf, _depths = get_tracks_3d_for_query_frame(
        query_frame, query_img, raw, depths, combined_masks, inv_Ks, c2ws, track_type="cotracker3",
    )

    # Replicate the oracle's internal N-selection (utils.py:187-191) to map its
    # filtered output rows back to our original track_ids (order-preserving).
    tracks_2d_tn = raw[..., :2].swapaxes(0, 1)  # (T, N, 2)
    occs_tn, dists_tn = raw[..., 2].swapaxes(0, 1), raw[..., 3].swapaxes(0, 1)
    coords_norm = normalize_coords(tracks_2d_tn[:, None], H, W)
    is_in_masks_tn = F.grid_sample(combined_masks[:, None], coords_norm, align_corners=True)[:, 0, 0] == 1
    visibles_tn, _invisibles_tn, _confidences_tn = parse_cotracker3_track_info(occs_tn, dists_tn)
    visibles_tn = visibles_tn * is_in_masks_tn
    visible_counts = visibles_tn.sum(0)
    thresh = min(int(0.05 * T), visible_counts.float().quantile(0.1).item())
    valid_n = is_in_masks_tn[query_frame] & (visible_counts >= thresh)

    matched_idx = torch.nonzero(valid_n, as_tuple=False).squeeze(-1)
    ours = tracks["positions_world_raw"][matched_idx]  # (N1, T, 3)

    if ours.shape[0] != oracle_tracks_3d.shape[0]:
        raise RuntimeError(
            f"Parity check index-count mismatch for query {query_frame}: "
            f"{ours.shape[0]} (ours) vs {oracle_tracks_3d.shape[0]} (oracle) -- the "
            "N-selection replica diverged from get_tracks_3d_for_query_frame's internal logic."
        )

    diff = (ours - oracle_tracks_3d).abs()
    return {
        "num_matched_tracks": int(ours.shape[0]),
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else float("nan"),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else float("nan"),
        "atol": parity_atol,
        "passed": bool(diff.numel() == 0 or diff.max().item() <= parity_atol),
    }


def _draw_keypoints(img_bgr: np.ndarray, kps_xy: np.ndarray, colors_bgr: np.ndarray, radius: int = 3) -> np.ndarray:
    """Local reimplementation of flow3d.vis.utils.draw_keypoints_cv2 (same behavior),
    to avoid that module's heavy nvdiffrast/viser import for this one function."""
    out = img_bgr.copy()
    kps = kps_xy.round().astype(int)
    for i in range(len(kps)):
        color = tuple(int(c) for c in colors_bgr[i])
        cv2.circle(out, (int(kps[i, 0]), int(kps[i, 1])), radius, color, -1, cv2.LINE_AA)
    return out


def _select_preview_tracks(stable_track_mask: torch.Tensor, num_preview_tracks: int, seed: int) -> np.ndarray:
    stable_idx = torch.nonzero(stable_track_mask, as_tuple=False).squeeze(-1).numpy()
    rng = np.random.default_rng(seed)
    if len(stable_idx) > num_preview_tracks:
        sel = rng.choice(stable_idx, size=num_preview_tracks, replace=False)
    else:
        sel = stable_idx
    return np.sort(sel)


def save_track_previews(
    dataset: CasualDataset,
    tracks: dict,
    stable_track_mask: torch.Tensor,
    num_preview_tracks: int,
    seed: int,
    output_dir: Path,
) -> list[Path]:
    positions_2d = tracks["positions_2d"]  # (N, T, 2)
    T = positions_2d.shape[1]
    sel = _select_preview_tracks(stable_track_mask, num_preview_tracks, seed)

    cmap = colormaps.get_cmap("gist_rainbow")
    colors_rgb = np.asarray(cmap(np.linspace(0, 1, max(len(sel), 1)))[:, :3])
    colors_bgr = (colors_rgb[:, ::-1] * 255).astype(int)

    paths = []
    for label, t in [("first", 0), ("mid", T // 2), ("last", T - 1)]:
        img_rgb = (dataset.get_image(t).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        kps = positions_2d[sel, t].numpy()
        out = _draw_keypoints(img_bgr, kps, colors_bgr, radius=3)
        out_path = output_dir / f"preview_{label}_frame{dataset.frame_names[t]}.png"
        cv2.imwrite(str(out_path), out)
        paths.append(out_path)
    return paths


def save_3d_trajectory_plot(
    tracks: dict, stable_track_mask: torch.Tensor, num_preview_tracks: int, seed: int, output_dir: Path
) -> Path:
    positions_world = tracks["positions_world"]  # (N, T, 3), NaN where invalid
    sel = _select_preview_tracks(stable_track_mask, num_preview_tracks, seed)

    cmap = colormaps.get_cmap("gist_rainbow")
    colors = cmap(np.linspace(0, 1, max(len(sel), 1)))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, n in enumerate(sel):
        traj = positions_world[n].numpy()  # (T, 3); matplotlib breaks the line at NaN rows
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Stable track 3D trajectories (world space), n={len(sel)}")
    out_path = output_dir / "trajectories_3d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_report(
    args: argparse.Namespace,
    config_path: Path,
    dataset: CasualDataset,
    tracks: dict,
    stable_track_mask: torch.Tensor,
    reproj_stats: dict,
    merge_info: dict,
    output_paths: dict,
    elapsed_seconds: float,
) -> dict:
    valid_mask = tracks["valid_mask"]
    N, T = valid_mask.shape
    H, W = dataset.get_image(0).shape[:2]

    def fail_frac(mask: torch.Tensor) -> float:
        return float((~mask).float().mean().item()) if mask.numel() else float("nan")

    validity_breakdown = {
        "in_bounds_fail_frac": fail_frac(tracks["in_bounds_mask"]),
        "depth_finite_fail_frac": fail_frac(tracks["depth_finite_mask"]),
        "depth_positive_fail_frac": fail_frac(tracks["depth_positive_mask"]),
        "fg_mask_fail_frac": fail_frac(tracks["fg_mask_valid"]),
        "depth_mask_fail_frac": fail_frac(tracks["depth_mask_valid"]),
        "visible_fail_frac": fail_frac(tracks["visible_mask"]),
        "confidence_fail_frac": fail_frac(tracks["confidence_mask"]),
        "overall_valid_frac": float(valid_mask.float().mean().item()) if valid_mask.numel() else float("nan"),
    }

    parity_passed = all(s["passed"] for s in merge_info["parity_checks"].values())

    return {
        "seq_name": args.seq_name,
        "config_path": str(config_path),
        "output_dir": str(output_paths["output_dir"]),
        "query_frames": merge_info["query_frames"],
        "num_frames": T,
        "image_hw": [int(H), int(W)],
        "num_tracks_total": N,
        "num_tracks_stable": int(stable_track_mask.sum().item()),
        "per_query_track_counts": merge_info["per_query_track_counts"],
        "total_tracks_before_merge": merge_info["total_tracks_before_merge"],
        "total_tracks_after_merge": merge_info["total_tracks_after_merge"],
        "merge_component_size_histogram": merge_info["merge_component_size_histogram"],
        "coverage_per_frame": merge_info["coverage_per_frame"],
        "coverage_per_frame_query0_baseline": merge_info["coverage_per_frame_query0_baseline"],
        "thresholds": {
            "min_confidence": args.min_confidence,
            "stable_ratio": args.stable_ratio,
            "min_valid_frames": args.min_valid_frames,
            "min_overlap_frames": args.min_overlap_frames,
            "merge_max_2d_px": args.merge_max_2d_px,
            "merge_max_3d_dist": args.merge_max_3d_dist,
            "merge_candidate_topk": args.merge_candidate_topk,
            "reproj_atol_px": args.reproj_atol_px,
            "parity_atol": args.parity_atol,
        },
        "validity_breakdown": validity_breakdown,
        "reprojection_check": reproj_stats,
        "parity_checks": merge_info["parity_checks"],
        "parity_passed": parity_passed,
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    args = build_parser().parse_args()

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    output_dir = resolve_output_dir(args.seq_name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_tracks_path = output_dir / "raw_tracks_3d.pt"
    if raw_tracks_path.exists() and not args.overwrite:
        raise FileExistsError(f"{raw_tracks_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    config_path = resolve_config_path(args.config)
    dataset, _scene_cfg = load_dataset(config_path, args.seq_name)
    T = dataset.num_frames
    query_frames = generate_query_frames(T, args.query_frame_stride, args.query_frames)

    tracks, merge_info = extract_and_merge_tracks(dataset, query_frames, args, device)
    stable_track_mask = compute_stable_mask(tracks["valid_mask"], args.stable_ratio, args.min_valid_frames)

    reproj_stats = verify_reprojection(tracks, args.reproj_atol_px)
    if not reproj_stats["passed"]:
        print(
            f"[WARN] reprojection camera-math check exceeded tolerance: "
            f"max {reproj_stats['max_px']:.6g}px > {args.reproj_atol_px}px"
        )
    for q, stats in merge_info["parity_checks"].items():
        if not stats["passed"]:
            print(
                f"[WARN] parity check for query frame {q} exceeded tolerance: "
                f"max {stats['max_abs_diff']:.6g} > {args.parity_atol}"
            )

    N = tracks["valid_mask"].shape[0]
    payload = {
        "positions_world": tracks["positions_world"],
        "positions_2d": tracks["positions_2d"],
        "valid_mask": tracks["valid_mask"],
        "confidence": tracks["confidence"],
        "sampled_depth": tracks["sampled_depth"],
        "in_bounds_mask": tracks["in_bounds_mask"],
        "depth_finite_mask": tracks["depth_finite_mask"],
        "depth_positive_mask": tracks["depth_positive_mask"],
        "fg_mask_valid": tracks["fg_mask_valid"],
        "depth_mask_valid": tracks["depth_mask_valid"],
        "visible_mask": tracks["visible_mask"],
        "confidence_mask": tracks["confidence_mask"],
        "track_ids": torch.arange(N, dtype=torch.long),
        "frame_indices": torch.arange(T, dtype=torch.long),
        "stable_track_mask": stable_track_mask,
        "source_query_frames": merge_info["source_query_frames"],
        "Ks": tracks["Ks"],
        "w2cs": tracks["w2cs"],
        "thresholds": {
            "min_confidence": args.min_confidence,
            "stable_ratio": args.stable_ratio,
            "min_valid_frames": args.min_valid_frames,
            "query_frames": query_frames,
            "min_overlap_frames": args.min_overlap_frames,
            "merge_max_2d_px": args.merge_max_2d_px,
            "merge_max_3d_dist": args.merge_max_3d_dist,
            "merge_candidate_topk": args.merge_candidate_topk,
            "reproj_atol_px": args.reproj_atol_px,
            "parity_atol": args.parity_atol,
        },
        "meta": {
            "seq_name": args.seq_name,
            "config_path": str(config_path),
            "num_frames": T,
            "image_hw": list(dataset.get_image(0).shape[:2]),
        },
    }
    torch.save(payload, raw_tracks_path)

    preview_paths = save_track_previews(
        dataset, tracks, stable_track_mask, args.num_preview_tracks, args.seed, output_dir
    )
    trajectory_plot_path = save_3d_trajectory_plot(
        tracks, stable_track_mask, args.num_preview_tracks, args.seed, output_dir
    )

    output_paths = {
        "output_dir": output_dir,
        "raw_tracks_3d": raw_tracks_path,
        "trajectories_3d": trajectory_plot_path,
        **{p.stem: p for p in preview_paths},
    }
    elapsed_seconds = time.time() - start_time
    report = build_report(
        args, config_path, dataset, tracks, stable_track_mask,
        reproj_stats, merge_info, output_paths, elapsed_seconds,
    )
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2)

    print(
        f"[extract_motion_tracks] seq={args.seq_name} query_frames={query_frames} "
        f"before_merge={merge_info['total_tracks_before_merge']} "
        f"after_merge={merge_info['total_tracks_after_merge']} "
        f"N_stable={int(stable_track_mask.sum().item())} "
        f"reproj_max_px={reproj_stats['max_px']:.3g} "
        f"parity_passed={report['parity_passed']} output_dir={output_dir}"
    )

    if not reproj_stats["passed"]:
        raise RuntimeError(
            f"Reprojection camera-math check failed: max {reproj_stats['max_px']:.6g}px > "
            f"tol {args.reproj_atol_px}px (outputs were written to {output_dir} for debugging)"
        )
    if not report["parity_passed"]:
        raise RuntimeError(
            f"Parity check(s) vs get_tracks_3d_for_query_frame failed for one or more "
            f"query frames (see report.json parity_checks) "
            f"(outputs were written to {output_dir} for debugging)"
        )


if __name__ == "__main__":
    main()

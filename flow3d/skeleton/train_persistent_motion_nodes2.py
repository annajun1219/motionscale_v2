"""
Train a canonical-position-conditioned deformation field for persistent 3D
motion nodes -- variant with strict active-node gating on the 2D supervision.

Copy of train_persistent_motion_nodes.py (that script is left untouched),
defaulting to the motion_skeleton2/ raw tracks (extract_motion_tracks2.py's
complete-link + anchor/donor output) and writing to a separate
persistent_nodes2/ output directory. Model architecture, loss weights, and
node selection (coverage-aware FPS, ARAP graph, transport graph, canonical
position + deformation field) are all unchanged from the base script.

The one substantive change: every 2D-image-space supervision term (the
Chamfer term against the thinned SAM-mask skeleton, and the outside-mask
exclusion penalty) is now gated by an explicit ACTIVE mask equal to
fit_valid_mask at the sampled frames, instead of implicitly supervising every
node's projection regardless of whether that node's own 3D observation was
trusted. A node that is unobserved at a frame, or was excluded there as a
suspected identity switch, contributes to neither the Chamfer distance (in
either direction: node->skeleton nor skeleton->node) nor the mask-exclusion
average, and receives exactly zero gradient from either term at that frame --
verified numerically after training (see verify_inactive_node_isolation).
Frames where a sampled batch has zero active nodes are skipped safely rather
than computing a degenerate (empty-set) Chamfer term.

The neighbor-transport loss's gap definition is also switched from
source_valid to fit_valid_mask: an outlier-excluded observation is now
treated the same as a true observation gap for transport purposes (pulled
toward the trusted neighborhood's own Kabsch-predicted motion), whereas
before it was only unobserved (never-tracked) frames that counted as a gap.
Acceleration and ARAP remain unmasked, applying uniformly everywhere
(including gaps), which was already true in the base script.

Output
------
outputs/davis/<seq-name>/skeleton/persistent_nodes2/
    persistent_nodes.pt
    checkpoint.pt
    loss_log.json
    preview_first_frame<NNNNN>.png
    preview_mid_frame<NNNNN>.png
    preview_last_frame<NNNNN>.png
    persistent_nodes_overlay.mp4
    report.json

Example
-------
    python flow3d/skeleton/train_persistent_motion_nodes2.py --seq-name camel --overwrite
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # skeleton/ -> flow3d/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from matplotlib import colormaps
from skimage.morphology import skeletonize

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.data.utils import to_serializable

# Not imported: flow3d.scene_model, flow3d.params, flow3d.renderer, flow3d.trainer,
# any checkpoint/Gaussian/motion-basis code. build_motion_nodes.py's motion_nodes.pt
# and any pre-existing cluster ids are never read either -- only raw_tracks_3d.pt.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument(
        "--raw-tracks-path", type=Path, default=None,
        help="Override the input raw_tracks_3d.pt path. Defaults to "
             "outputs/davis/<seq-name>/skeleton/motion_skeleton2/raw_tracks_3d.pt "
             "(extract_motion_tracks2.py's output).",
    )
    parser.add_argument("--num-nodes", type=int, default=256)
    parser.add_argument("--min-valid-frames", type=int, default=20)
    parser.add_argument("--min-continuous-frames", type=int, default=10)
    parser.add_argument("--min-common-frames", type=int, default=10)
    parser.add_argument("--spatial-knn", type=int, default=8)
    parser.add_argument("--canonical-encoding-freqs", type=int, default=4)
    parser.add_argument("--transport-knn", type=int, default=8)
    parser.add_argument(
        "--transport-spatial-knn", type=int, default=16,
        help="Per-node spatial candidate width (nearest by dist_matrix's common-"
             "overlap median 3D distance, the same matrix the ARAP graph uses) that "
             "gates the transport graph's rigidity ranking. A node outside this "
             "spatial candidate set can never become a transport neighbor no matter "
             "how rigidly it happens to move (e.g. the opposite leg in a symmetric gait).",
    )
    parser.add_argument("--time-encoding-freqs", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-fit3d", type=float, default=1.0)
    parser.add_argument("--w-chamfer2d", type=float, default=0.1)
    parser.add_argument("--w-mask", type=float, default=0.1)
    parser.add_argument("--w-vel-match", type=float, default=0.05)
    parser.add_argument("--w-acc", type=float, default=0.01)
    parser.add_argument("--w-arap", type=float, default=0.1)
    parser.add_argument("--w-neighbor-transport", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--ramp-iters", type=int, default=1000)
    parser.add_argument("--num-2d-frames-per-iter", type=int, default=12)
    parser.add_argument("--max-skeleton-points-per-frame", type=int, default=256)
    parser.add_argument("--template-coverage-percentile", type=float, default=0.5)
    parser.add_argument(
        "--depth-jump-mad-multiplier", type=float, default=5.0,
        help="Robust (median+k*MAD) threshold multiplier for the momentary-depth-jump detector.",
    )
    parser.add_argument(
        "--switch-mad-multiplier", type=float, default=5.0,
        help="Per-node robust (median+k*MAD) threshold multiplier for the fixed, pre-training "
             "identity-switch outlier detector.",
    )
    parser.add_argument(
        "--inactive-perturb-magnitude", type=float, default=5.0,
        help="World-unit offset applied to inactive nodes' positions in the post-training "
             "isolation check (verify_inactive_node_isolation) -- deliberately far larger "
             "than any plausible scene motion, to prove Chamfer/mask loss and gradient are "
             "exactly invariant to it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_config_path(config: Path) -> Path:
    return config if config.is_absolute() else (_REPO_ROOT / config).resolve()


def load_dataset(config_path: Path, seq_name: str) -> tuple[CasualDataset, dict]:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(
        root_dir=scene_cfg["data_dir"], seq_name=seq_name,
        load_from_cache=True, **scene_cfg.get("data", {}),
    )
    dataset = CasualDataset(**asdict(data_cfg))
    return dataset, scene_cfg


# ---------------------------------------------------------------------------
# Robust masked statistics + track-distance + depth-jump cleaning, reimplemented
# locally (same approach validated in build_motion_nodes.py; this codebase's
# flow3d/skeleton/*.py scripts each own their copies -- no shared package).
# ---------------------------------------------------------------------------

def masked_row_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = torch.where(mask, values, torch.full_like(values, float("inf")))
    sorted_vals, _ = filled.sort(dim=1)
    counts = mask.sum(dim=1)
    lower_idx = ((counts - 1) // 2).clamp(min=0)
    upper_idx = (counts // 2).clamp(min=0)
    lower = sorted_vals.gather(1, lower_idx[:, None]).squeeze(1)
    upper = sorted_vals.gather(1, upper_idx[:, None]).squeeze(1)
    median = (lower + upper) / 2
    return torch.where(counts > 0, median, torch.full_like(median, float("nan")))


def masked_row_mad(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    med = masked_row_median(values, mask)
    dev = (values - med[:, None]).abs()
    return masked_row_median(dev, mask)


def masked_row_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    filled = torch.where(mask, values, torch.full_like(values, float("inf")))
    sorted_vals, _ = filled.sort(dim=1)
    counts = mask.sum(dim=1)
    pos = q * (counts.float() - 1).clamp(min=0)
    lower_idx = pos.floor().long().clamp(min=0)
    upper_idx = pos.ceil().long().clamp(min=0)
    frac = pos - lower_idx.float()
    lower = sorted_vals.gather(1, lower_idx[:, None]).squeeze(1)
    upper = sorted_vals.gather(1, upper_idx[:, None]).squeeze(1)
    result = lower + (upper - lower) * frac
    return torch.where(counts > 0, result, torch.full_like(result, float("nan")))


def compute_clean_valid_mask(
    depth: torch.Tensor, valid: torch.Tensor, mad_multiplier: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flags momentary single-frame depth-jump outliers (strictly-adjacent integer
    frames t-1,t,t+1, all three already-valid) and excludes only those frames --
    never the whole track. Same detector as build_motion_nodes.py."""
    diff = (depth[:, 1:] - depth[:, :-1]).abs()
    diff_valid = valid[:, 1:] & valid[:, :-1]
    scale = masked_row_median(diff, diff_valid) + mad_multiplier * masked_row_mad(diff, diff_valid)
    scale = torch.nan_to_num(scale, nan=float("inf"))

    d1, d2 = diff[:, :-1], diff[:, 1:]
    d3 = (depth[:, 2:] - depth[:, :-2]).abs()
    candidate_valid = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    s = scale[:, None]
    is_jump = candidate_valid & (d1 > s) & (d2 > s) & (d3 <= s)

    depth_jump_mask = torch.zeros_like(valid)
    depth_jump_mask[:, 1:-1] = is_jump
    return valid & ~depth_jump_mask, depth_jump_mask


def longest_contiguous_run(mask: torch.Tensor) -> torch.Tensor:
    """Longest run of consecutive True values per row, (N, T) -> (N,)."""
    N, T = mask.shape
    run = torch.zeros((N, T), dtype=torch.long)
    run[:, 0] = mask[:, 0].long()
    for t in range(1, T):
        run[:, t] = torch.where(mask[:, t], run[:, t - 1] + 1, torch.zeros_like(run[:, t - 1]))
    return run.max(dim=1).values


def track_distance_to_pool(
    pos_ref: torch.Tensor, valid_ref: torch.Tensor,
    pos_pool: torch.Tensor, valid_pool: torch.Tensor, min_common_frames: int,
) -> torch.Tensor:
    common = valid_ref[None, :] & valid_pool
    d = (pos_pool - pos_ref[None, :, :]).norm(dim=-1)
    dist = masked_row_median(d, common)
    counts = common.sum(dim=1)
    return torch.where(counts >= min_common_frames, dist, torch.full_like(dist, float("nan")))


def pairwise_track_distance(pos: torch.Tensor, valid: torch.Tensor, min_common_frames: int) -> torch.Tensor:
    M = pos.shape[0]
    dist = torch.full((M, M), float("nan"))
    for i in range(M):
        dist[i] = track_distance_to_pool(pos[i], valid[i], pos, valid, min_common_frames)
    dist.fill_diagonal_(0.0)
    return dist


def pairwise_rigidity_score(pos: torch.Tensor, valid: torch.Tensor, min_common_frames: int) -> torch.Tensor:
    """q90(|dist_t - median|) / median over commonly-observed frames -- smaller means
    the pair moves rigidly together (trustworthy same-limb neighbor), same metric
    build_motion_nodes.py used for edge rigidity. NaN below min_common_frames."""
    M, T, _ = pos.shape
    score = torch.full((M, M), float("nan"))
    for i in range(M):
        common = valid[i][None, :] & valid  # (M, T)
        d = (pos - pos[i][None, :, :]).norm(dim=-1)  # (M, T)
        med = masked_row_median(d, common)
        dev = (d - med[:, None]).abs()
        q90 = masked_row_quantile(dev, common, 0.9)
        counts = common.sum(dim=1)
        row = torch.where(counts >= min_common_frames, q90 / (med + 1e-6), torch.full_like(q90, float("nan")))
        score[i] = row
    score.fill_diagonal_(0.0)
    return score


def build_mutual_knn_graph(dist: torch.Tensor, k: int) -> list[tuple[int, int]]:
    M = dist.shape[0]
    k_eff = max(min(k, M - 1), 0)
    if k_eff == 0:
        return []
    ranking = torch.nan_to_num(dist, nan=float("inf")).clone()
    ranking.fill_diagonal_(float("inf"))
    knn_val, knn_idx = ranking.topk(k_eff, dim=1, largest=False)
    knn_mask = torch.zeros((M, M), dtype=torch.bool)
    row_idx = torch.arange(M).unsqueeze(1).expand(-1, k_eff)
    valid_entries = knn_val < float("inf")
    knn_mask[row_idx[valid_entries], knn_idx[valid_entries]] = True
    mutual = knn_mask & knn_mask.T
    ii, jj = torch.triu(mutual, diagonal=1).nonzero(as_tuple=True)
    return list(zip(ii.tolist(), jj.tolist()))


def spatial_candidate_mask(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Per-row top-k nearest neighbors by dist_matrix (the same common-overlap
    median 3D distance used for the ARAP graph), excluding self and any NaN
    (undefined/insufficient overlap) entry. (M,) rows independently -- NOT
    symmetrized here; build_mutual_knn_graph's own mutual-kNN step on the gated
    rigidity matrix is what ultimately requires both directions to agree."""
    M = dist_matrix.shape[0]
    k_eff = max(min(k, M - 1), 0)
    mask = torch.zeros((M, M), dtype=torch.bool)
    if k_eff == 0:
        return mask
    ranking = torch.nan_to_num(dist_matrix, nan=float("inf")).clone()
    ranking.fill_diagonal_(float("inf"))
    knn_val, knn_idx = ranking.topk(k_eff, dim=1, largest=False)
    row_idx = torch.arange(M).unsqueeze(1).expand(-1, k_eff)
    valid_entries = knn_val < float("inf")
    mask[row_idx[valid_entries], knn_idx[valid_entries]] = True
    return mask


def edges_to_padded_neighbors(edges: list[tuple[int, int]], M: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected edge list -> (neighbor_idx (M,K), neighbor_mask (M,K)), K = max degree
    (no edges dropped; K is whatever it needs to be, not clamped to the kNN parameter,
    since mutual-kNN symmetrization can give some nodes higher in-degree)."""
    neighbor_lists: list[list[int]] = [[] for _ in range(M)]
    for i, j in edges:
        neighbor_lists[i].append(j)
        neighbor_lists[j].append(i)
    K = max((len(lst) for lst in neighbor_lists), default=0)
    K = max(K, 1)
    neighbor_idx = torch.zeros(M, K, dtype=torch.long)
    neighbor_mask = torch.zeros(M, K, dtype=torch.bool)
    for m, lst in enumerate(neighbor_lists):
        for k, nb in enumerate(lst):
            neighbor_idx[m, k] = nb
            neighbor_mask[m, k] = True
    return neighbor_idx, neighbor_mask


# ---------------------------------------------------------------------------
# Coverage-aware FPS: selection score is (spatial distance to selected set) *
# (valid ratio), so a spatially-isolated but poorly-observed track never wins
# purely on distance; undefined distances (no overlap yet) never win either.
# No relaxation fallback: the candidate pool is never padded with tracks that
# fail the coverage/continuity filter just to reach --num-nodes.
# ---------------------------------------------------------------------------

def fps_select_coverage_aware(
    pos: torch.Tensor, valid: torch.Tensor, num_nodes: int, min_common_frames: int, device: torch.device
) -> list[int]:
    N = pos.shape[0]
    pos_d, valid_d = pos.to(device), valid.to(device)
    valid_ratio = valid_d.float().mean(dim=1)
    num_nodes = min(num_nodes, N)

    first = int(torch.argmax(valid_d.sum(dim=1)).item())
    selected = [first]
    min_dist = torch.full((N,), float("nan"), device=device)

    for _ in range(num_nodes - 1):
        last = selected[-1]
        d = track_distance_to_pool(pos_d[last], valid_d[last], pos_d, valid_d, min_common_frames)
        have_new = ~d.isnan()
        had_prior = ~min_dist.isnan()
        updated = torch.minimum(
            torch.nan_to_num(min_dist, nan=float("inf")), torch.nan_to_num(d, nan=float("inf"))
        )
        min_dist = torch.where(have_new & had_prior, updated, min_dist)
        min_dist = torch.where(have_new & ~had_prior, d, min_dist)

        selection_score = torch.where(
            min_dist.isnan(), torch.full_like(min_dist, float("-inf")), min_dist * valid_ratio
        )
        selection_score[selected] = float("-inf")
        selected.append(int(torch.argmax(selection_score).item()))
    return selected


def compute_initial_trajectory(source_pos: torch.Tensor, source_valid: torch.Tensor) -> torch.Tensor:
    """(T, M, 3): per node, linear interpolation across its valid observations, with
    nearest-observation (flat) hold before/after the first/last valid frame -- exactly
    np.interp's default out-of-range behavior. No longer the model's baseline (see
    the canonical-position model below); kept solely to give every node a robust,
    data-driven canonical position even when it isn't observed exactly at t_ref."""
    T, M, _ = source_pos.shape
    pos_np = source_pos.numpy()
    valid_np = source_valid.numpy()
    all_frames = np.arange(T)
    init = np.zeros((T, M, 3), dtype=np.float32)
    for n in range(M):
        idx = np.nonzero(valid_np[:, n])[0]
        for axis in range(3):
            init[:, n, axis] = np.interp(all_frames, idx, pos_np[idx, n, axis])
    return torch.from_numpy(init)


# ---------------------------------------------------------------------------
# Model: canonical position (fixed per node, NOT learned) + time encoding ->
# shared deformation field D. x_i(t) = c_i + D(c_i,t) - D(c_i,t_ref). Zero-init
# last layer of D means x_i(t) == c_i everywhere at iteration 0.
# ---------------------------------------------------------------------------

def time_encoding(T: int, num_freqs: int, device: torch.device) -> torch.Tensor:
    t_norm = torch.linspace(0, 1, T, device=device)
    freqs = (2.0 ** torch.arange(num_freqs, device=device)) * math.pi
    args = t_norm[:, None] * freqs[None, :]
    return torch.cat([t_norm[:, None], torch.sin(args), torch.cos(args)], dim=-1)  # (T, 2L+1)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, out_dim: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CanonicalPositionEncoding(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs

    def forward(self, c: torch.Tensor) -> torch.Tensor:  # (M, 3) -> (M, 3 + 6*num_freqs)
        freqs = (2.0 ** torch.arange(self.num_freqs, device=c.device)) * math.pi
        args = c[..., None] * freqs[None, None, :]  # (M, 3, L)
        return torch.cat([c, torch.sin(args).flatten(1), torch.cos(args).flatten(1)], dim=-1)


class PersistentNodeModel(nn.Module):
    def __init__(
        self, canonical_pos: torch.Tensor, t_ref: int, canonical_freqs: int,
        time_enc_dim: int, hidden_dim: int, num_layers: int,
    ):
        super().__init__()
        self.register_buffer("canonical_pos", canonical_pos)  # (M, 3), fixed -- NOT a parameter
        self.t_ref = t_ref
        self.pos_encoder = CanonicalPositionEncoding(canonical_freqs)
        enc_dim = 3 + 6 * canonical_freqs
        self.deform = ResidualMLP(enc_dim + time_enc_dim, hidden_dim, num_layers)

    def forward(self, time_enc: torch.Tensor) -> torch.Tensor:
        T = time_enc.shape[0]
        M = self.canonical_pos.shape[0]
        c_enc = self.pos_encoder(self.canonical_pos)  # (M, Ce)
        x_t = torch.cat(
            [c_enc[None].expand(T, -1, -1), time_enc[:, None].expand(-1, M, -1)], dim=-1
        ).reshape(T * M, -1)
        D_t = self.deform(x_t).reshape(T, M, 3)
        x_ref = torch.cat([c_enc, time_enc[self.t_ref : self.t_ref + 1].expand(M, -1)], dim=-1)
        D_ref = self.deform(x_ref).reshape(1, M, 3)
        return self.canonical_pos[None] + D_t - D_ref


# ---------------------------------------------------------------------------
# One reusable, masked, batched Kabsch-expected-position function. Used for:
# (a) the training-time neighbor-transport loss (current_pos = pred_pos,
#     differentiable, extra_valid_mask=None),
# (b) the fixed pre-training outlier mask (current_pos = source_pos_safe,
#     extra_valid_mask = source_valid gathered at the neighbor indices), and
# (c) post-training switch-score comparisons (current_pos = a full trajectory,
#     extra_valid_mask=None).
# Falls back to a translation-only (centroid-offset) estimate where fewer than
# 3 neighbors are usable, since a rotation isn't observable from < 3 points.
# ---------------------------------------------------------------------------

def neighbor_kabsch_expected(
    canonical_pos: torch.Tensor,          # (M, 3)
    neighbor_idx: torch.Tensor,            # (M, K)
    neighbor_struct_mask: torch.Tensor,    # (M, K)
    current_pos: torch.Tensor,             # (T, M, 3)
    extra_valid_mask: torch.Tensor | None = None,  # (T, M, K) or None
) -> tuple[torch.Tensor, torch.Tensor]:
    T = current_pos.shape[0]
    M, K = neighbor_idx.shape
    device = current_pos.device

    P_nb = canonical_pos[neighbor_idx]  # (M, K, 3)
    Q_nb = current_pos[:, neighbor_idx]  # (T, M, K, 3)

    if extra_valid_mask is None:
        mask = neighbor_struct_mask[None].expand(T, -1, -1).to(current_pos.dtype)
    else:
        mask = (neighbor_struct_mask[None] & extra_valid_mask).to(current_pos.dtype)

    count = mask.sum(-1)  # (T, M)
    count_safe = count.clamp(min=1)

    P_nb_b = P_nb[None].expand(T, -1, -1, -1)  # (T, M, K, 3)
    P_mean = (P_nb_b * mask[..., None]).sum(2) / count_safe[..., None]  # (T, M, 3)
    Q_mean = (Q_nb * mask[..., None]).sum(2) / count_safe[..., None]  # (T, M, 3)

    Pc = (P_nb_b - P_mean[:, :, None, :]) * mask[..., None]
    Qc = (Q_nb - Q_mean[:, :, None, :]) * mask[..., None]

    H = torch.einsum("tmki,tmkj->tmij", Pc, Qc)  # (T, M, 3, 3)
    U, _S, Vt = torch.linalg.svd(H)
    Ut, Vtt = U.transpose(-1, -2), Vt.transpose(-1, -2)
    d = torch.sign(torch.linalg.det(torch.matmul(Vtt, Ut)))
    D = torch.zeros(T, M, 3, 3, device=device, dtype=H.dtype)
    D[..., 0, 0] = 1.0
    D[..., 1, 1] = 1.0
    D[..., 2, 2] = d
    R = torch.matmul(torch.matmul(Vtt, D), Ut)  # (T, M, 3, 3)

    t_vec = Q_mean - torch.einsum("tmij,tmj->tmi", R, P_mean)
    kabsch_expected = torch.einsum("tmij,mj->tmi", R, canonical_pos) + t_vec
    fallback_expected = (canonical_pos[None] - P_mean) + Q_mean  # translation-only

    valid_enough = (count >= 3)[..., None]
    expected = torch.where(valid_enough, kabsch_expected, fallback_expected)
    return expected, count


def normalize_coords_local(coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0


def project_to_2d(pos_tn: torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor) -> torch.Tensor:
    """pos_tn: (T, N, 3); Ks: (T, 3, 3); w2cs: (T, 4, 4) -> (T, N, 2)."""
    pts_cam = torch.einsum("tij,tnj->tni", w2cs, F.pad(pos_tn, (0, 1), value=1.0))[..., :3]
    pts_img = torch.einsum("tij,tnj->tni", Ks, pts_cam)
    return pts_img[..., :2] / torch.clamp(pts_img[..., 2:], min=1e-5)


# ---------------------------------------------------------------------------
# 2D supervision precompute: SAM-mask skeleton (thinning) + an outside-mask
# distance field, both computed ONCE (skeletonization is CPU-bound), both
# normalized by the image diagonal so loss weights aren't resolution-dependent.
# ---------------------------------------------------------------------------

def precompute_2d_supervision(
    dataset: CasualDataset, T: int, max_skel_pts: int, seed: int
) -> tuple[list[torch.Tensor], torch.Tensor, float, int, int]:
    H, W = dataset.get_image(0).shape[:2]
    norm = math.sqrt(H ** 2 + W ** 2)
    rng = np.random.default_rng(seed)
    skeleton_pts: list[torch.Tensor] = []
    dist_fields = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        fg = dataset.get_fg_mask(t).numpy().astype(bool)  # (H, W)
        skel = skeletonize(fg)
        ys, xs = np.nonzero(skel)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        if len(pts) > max_skel_pts:
            sel = rng.choice(len(pts), size=max_skel_pts, replace=False)
            pts = pts[sel]
        skeleton_pts.append(torch.from_numpy(pts))
        dist = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        dist_fields[t] = dist / norm
    return skeleton_pts, torch.from_numpy(dist_fields), norm, H, W


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def fit3d_loss(pred_pos, source_pos_safe, fit_valid_mask, confidence) -> torch.Tensor:
    diff = F.smooth_l1_loss(pred_pos, source_pos_safe, reduction="none").sum(dim=-1)  # (T, M)
    weight = confidence * fit_valid_mask.float()
    denom = weight.sum().clamp(min=1e-8)
    return (weight * diff).sum() / denom


def acceleration_loss(pred_pos: torch.Tensor) -> torch.Tensor:
    acc = pred_pos[2:] - 2 * pred_pos[1:-1] + pred_pos[:-2]
    return (acc ** 2).sum(dim=-1).mean()


def velocity_match_loss(pred_pos, source_pos_safe, fit_valid_mask) -> torch.Tensor:
    pred_vel = pred_pos[1:] - pred_pos[:-1]
    source_vel = source_pos_safe[1:] - source_pos_safe[:-1]
    pair_valid = (fit_valid_mask[1:] & fit_valid_mask[:-1]).float()
    diff = ((pred_vel - source_vel) ** 2).sum(dim=-1)
    denom = pair_valid.sum().clamp(min=1e-8)
    return (pair_valid * diff).sum() / denom


def arap_loss(pred_pos, edges_i, edges_j, rest_dist) -> torch.Tensor:
    if edges_i.numel() == 0:
        return torch.zeros((), device=pred_pos.device)
    pi = pred_pos[:, edges_i]
    pj = pred_pos[:, edges_j]
    d = (pi - pj).norm(dim=-1)
    return ((d - rest_dist[None, :]) ** 2).mean()


def neighbor_transport_loss(
    pred_pos, fit_valid_mask, canonical_pos, transport_neighbor_idx, transport_neighbor_mask
) -> torch.Tensor:
    """`fit_valid_mask` (not raw source_valid) defines the gap: a node/frame excluded
    as a suspected identity switch is treated exactly like an unobserved frame here,
    so the neighbor-transport term pulls it toward the trusted neighborhood's own
    Kabsch-predicted motion instead of leaving it unconstrained.

    The Kabsch fit (SVD) is computed as a fixed target each iteration, not
    differentiated through: backpropagating through SVD is numerically unstable
    whenever singular values are near-degenerate (e.g. very early in training,
    when predictions are still close to canonical_pos for every node), which
    otherwise produces NaN gradients. Only pred_vel (the trainable side) carries
    gradient; expected_vel is a detached, per-iteration-recomputed target."""
    with torch.no_grad():
        expected, _count = neighbor_kabsch_expected(
            canonical_pos, transport_neighbor_idx, transport_neighbor_mask, pred_pos.detach()
        )
    pred_vel = pred_pos[1:] - pred_pos[:-1]
    expected_vel = expected[1:] - expected[:-1]
    gap = (~(fit_valid_mask[1:] & fit_valid_mask[:-1])).float()
    diff = ((pred_vel - expected_vel) ** 2).sum(-1)
    denom = gap.sum().clamp(min=1e-8)
    return (gap * diff).sum() / denom


def chamfer2d_loss(
    proj_xy: torch.Tensor, frame_indices: list[int], skeleton_pts_list: list[torch.Tensor],
    norm: float, active_mask: torch.Tensor,
) -> torch.Tensor:
    """`active_mask` (F, M) gates BOTH Chamfer directions: node->skeleton nearest
    distance is only computed and averaged over active nodes, and skeleton->node
    nearest distance only ever searches active nodes as candidates -- an inactive
    node's projected position is excluded from `nodes_t` entirely (not merely
    down-weighted), so it receives exactly zero gradient from this term. A frame
    with zero active nodes is skipped outright rather than computing a degenerate
    empty-set Chamfer distance."""
    total = proj_xy.new_zeros(())
    count = 0
    for i, t in enumerate(frame_indices):
        pts = skeleton_pts_list[t]
        if pts.numel() == 0:
            continue
        active_i = active_mask[i]  # (M,) bool
        if not bool(active_i.any()):
            continue
        nodes_t = proj_xy[i][active_i]
        d = torch.cdist(nodes_t, pts.to(nodes_t.device))
        node_to_skel = d.min(dim=1).values.mean()
        skel_to_node = d.min(dim=0).values.mean()
        total = total + (node_to_skel + skel_to_node) / norm
        count += 1
    return total / count if count > 0 else total


def mask_exclusion_loss(
    dist_fields: torch.Tensor, frame_indices: torch.Tensor, proj_xy: torch.Tensor, H: int, W: int,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Sampled distance is computed for every (frame, node) pair (grid_sample has no
    notion of skipping entries), but only active entries contribute to the mean: the
    inactive-entry terms are multiplied by an exact 0 weight, which -- since proj_xy
    is always finite (never NaN, unlike raw source positions) -- yields an exactly
    zero gradient contribution at those entries, not merely a downweighted one."""
    coords = normalize_coords_local(proj_xy, H, W)  # (F, M, 2)
    field = dist_fields[frame_indices][:, None]  # (F, 1, H, W)
    grid = coords[:, None, :, :]  # (F, 1, M, 2)
    sampled = F.grid_sample(field, grid, align_corners=True, padding_mode="border")[:, 0, 0, :]  # (F, M)
    weight = active_mask.to(sampled.dtype)
    denom = weight.sum().clamp(min=1e-8)
    return (sampled * weight).sum() / denom


# ---------------------------------------------------------------------------
# Fixed, pre-training, raw-observation-only identity-switch outlier mask.
# Never depends on pred_pos; never touches source_valid/raw_tracks_3d.pt.
# ---------------------------------------------------------------------------

def compute_fixed_outlier_mask(
    source_pos_safe: torch.Tensor, source_valid: torch.Tensor, canonical_pos: torch.Tensor,
    transport_neighbor_idx: torch.Tensor, transport_neighbor_mask: torch.Tensor, mad_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extra_valid = source_valid[:, transport_neighbor_idx]  # (T, M, K): neighbor observed at that frame
    expected_raw, valid_count = neighbor_kabsch_expected(
        canonical_pos, transport_neighbor_idx, transport_neighbor_mask, source_pos_safe, extra_valid_mask=extra_valid
    )
    residual = (source_pos_safe - expected_raw).norm(dim=-1)  # (T, M)
    eligible = (valid_count >= 3) & source_valid
    residual_masked = torch.where(eligible, residual, torch.full_like(residual, float("nan")))

    r = residual_masked.T  # (M, T)
    r_mask = ~r.isnan()
    node_threshold = masked_row_median(r, r_mask) + mad_multiplier * masked_row_mad(r, r_mask)
    node_threshold = torch.nan_to_num(node_threshold, nan=float("inf"))  # no eligible data -> never flagged

    outlier_mask = eligible & (residual > node_threshold[None, :])
    fit_valid_mask = source_valid & ~outlier_mask
    return fit_valid_mask, outlier_mask, node_threshold


def compute_switch_scores(
    pred_pos: torch.Tensor, canonical_pos: torch.Tensor,
    transport_neighbor_idx: torch.Tensor, transport_neighbor_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected, valid_count = neighbor_kabsch_expected(canonical_pos, transport_neighbor_idx, transport_neighbor_mask, pred_pos)
    deviation = (pred_pos - expected).norm(dim=-1)
    deviation = torch.where(valid_count >= 3, deviation, torch.zeros_like(deviation))
    switch_score_per_node = deviation.max(dim=0).values
    return switch_score_per_node, deviation


# ---------------------------------------------------------------------------
# Post-training sanity check: an inactive (fit_valid_mask == False) node's
# position must have zero effect -- in both value and gradient -- on the 2D
# supervision terms, since those terms are supposed to only ever look at
# active nodes.
# ---------------------------------------------------------------------------

def verify_inactive_node_isolation(
    pred_pos: torch.Tensor, fit_valid_mask: torch.Tensor,
    Ks: torch.Tensor, w2cs: torch.Tensor, skeleton_pts_list: list[torch.Tensor],
    dist_fields: torch.Tensor, norm: float, H: int, W: int, device: torch.device,
    perturb_magnitude: float,
) -> dict:
    T, M = fit_valid_mask.shape
    frame_idx = torch.arange(T, device=device)
    active_mask = fit_valid_mask.to(device)
    pred_pos_d = pred_pos.detach().to(device)

    base = pred_pos_d.clone().requires_grad_(True)
    proj_base = project_to_2d(base[frame_idx], Ks[frame_idx], w2cs[frame_idx])
    l_cham_base = chamfer2d_loss(proj_base, frame_idx.tolist(), skeleton_pts_list, norm, active_mask)
    l_mask_base = mask_exclusion_loss(dist_fields, frame_idx, proj_base, H, W, active_mask)
    (l_cham_base + l_mask_base).backward()
    grad_on_inactive = base.grad[~active_mask]
    grad_on_inactive_max_abs = float(grad_on_inactive.abs().max().item()) if grad_on_inactive.numel() else 0.0

    offset = torch.where(
        (~active_mask)[..., None].expand(-1, -1, 3),
        torch.full_like(pred_pos_d, perturb_magnitude),
        torch.zeros_like(pred_pos_d),
    )
    perturbed = (pred_pos_d.clone() + offset).requires_grad_(True)
    proj_pert = project_to_2d(perturbed[frame_idx], Ks[frame_idx], w2cs[frame_idx])
    l_cham_pert = chamfer2d_loss(proj_pert, frame_idx.tolist(), skeleton_pts_list, norm, active_mask)
    l_mask_pert = mask_exclusion_loss(dist_fields, frame_idx, proj_pert, H, W, active_mask)

    chamfer_diff = float((l_cham_pert - l_cham_base).abs().item())
    mask_diff = float((l_mask_pert - l_mask_base).abs().item())
    passed = chamfer_diff < 1e-6 and mask_diff < 1e-6 and grad_on_inactive_max_abs < 1e-8
    return {
        "num_inactive_node_frames_perturbed": int((~active_mask).sum().item()),
        "perturb_magnitude": perturb_magnitude,
        "chamfer_loss_base": float(l_cham_base.item()),
        "chamfer_loss_perturbed": float(l_cham_pert.item()),
        "chamfer_loss_abs_diff": chamfer_diff,
        "mask_loss_base": float(l_mask_base.item()),
        "mask_loss_perturbed": float(l_mask_pert.item()),
        "mask_loss_abs_diff": mask_diff,
        "grad_on_inactive_nodes_max_abs": grad_on_inactive_max_abs,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Synthetic regression test for the transport graph's spatial gate: two
# far-apart "limbs" are constructed so every CROSS-limb pair moves as a
# perfectly rigid unit (identical global motion added to both, offset by a
# large constant translation) while genuine SAME-limb pairs carry a little
# independent per-node jitter -- so cross-limb pairs score STRICTLY better
# (lower) on rigidity than the real same-limb neighbors do. Without the
# spatial gate, mutual-kNN would pick these as top rigidity matches purely
# because they are mathematically more "rigid". Confirms the gate blocks them.
# ---------------------------------------------------------------------------

def verify_spatial_gating_blocks_distant_rigid_pairs(
    transport_spatial_knn: int, transport_knn: int, min_common_frames: int,
) -> dict:
    gen = torch.Generator().manual_seed(0)
    T = 30
    # Each limb must have MORE nodes than transport_spatial_knn, or the spatial top-k
    # has no choice but to include cross-limb candidates regardless of the gate --
    # that would make the test vacuous, not a real check of the gating logic.
    nodes_per_limb = transport_spatial_knn + 6
    M = nodes_per_limb * 2

    t = torch.linspace(0, 1, T)
    global_motion = torch.stack([torch.sin(t * 3), torch.cos(t * 2), t], dim=-1) * 0.05  # (T, 3)

    base_a = torch.rand(nodes_per_limb, 3, generator=gen) * 0.02
    base_b = torch.rand(nodes_per_limb, 3, generator=gen) * 0.02 + torch.tensor([100.0, 0.0, 0.0])

    jitter_a = torch.randn(nodes_per_limb, T, 3, generator=gen) * 0.002  # independent per-node noise
    jitter_b = torch.randn(nodes_per_limb, T, 3, generator=gen) * 0.002

    # Cross-limb pairs share the IDENTICAL global_motion term -> constant relative
    # offset over time -> ~0 rigidity score, "better" than any real same-limb pair.
    pos_a = base_a[:, None, :] + global_motion[None, :, :] + jitter_a  # (nodes_per_limb, T, 3)
    pos_b = base_b[:, None, :] + global_motion[None, :, :] + jitter_b

    pos = torch.cat([pos_a, pos_b], dim=0)  # (M, T, 3)
    valid = torch.ones(M, T, dtype=torch.bool)

    dist_matrix = pairwise_track_distance(pos, valid, min_common_frames)
    rigidity_score = pairwise_rigidity_score(pos, valid, min_common_frames)

    limb_a_idx = set(range(nodes_per_limb))

    def is_cross(i: int, j: int) -> bool:
        return (i in limb_a_idx) != (j in limb_a_idx)

    cross_scores = [
        float(rigidity_score[i, j]) for i in range(M) for j in range(M) if i != j and is_cross(i, j)
    ]
    same_limb_scores = [
        float(rigidity_score[i, j]) for i in range(M) for j in range(M) if i != j and not is_cross(i, j)
    ]
    cross_lower = bool(max(cross_scores) < min(same_limb_scores)) if cross_scores and same_limb_scores else False

    ungated_edges = build_mutual_knn_graph(rigidity_score, transport_knn)
    ungated_cross_edges = [e for e in ungated_edges if is_cross(*e)]

    spatial_mask = spatial_candidate_mask(dist_matrix, transport_spatial_knn)
    gated_rigidity = torch.where(spatial_mask, rigidity_score, torch.full_like(rigidity_score, float("nan")))
    gated_edges = build_mutual_knn_graph(gated_rigidity, transport_knn)
    gated_cross_edges = [e for e in gated_edges if is_cross(*e)]

    # The test is only meaningful if, absent the gate, at least one cross-limb edge
    # WOULD have formed -- otherwise "0 cross edges with gating" proves nothing.
    passed = cross_lower and len(ungated_cross_edges) > 0 and len(gated_cross_edges) == 0
    return {
        "num_limb_nodes_each": nodes_per_limb,
        "limb_separation": 100.0,
        "cross_limb_rigidity_strictly_better_than_same_limb": cross_lower,
        "cross_limb_rigidity_score_max": max(cross_scores) if cross_scores else None,
        "same_limb_rigidity_score_min": min(same_limb_scores) if same_limb_scores else None,
        "cross_limb_edges_without_spatial_gating": len(ungated_cross_edges),
        "cross_limb_edges_with_spatial_gating": len(gated_cross_edges),
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Previews / video: one fixed color per global node id. Three marker states:
# filled circle = observed and used for fitting; drawMarker cross = observed
# but excluded as a suspected identity switch; hollow ring = not observed
# (MLP-completed).
# ---------------------------------------------------------------------------

def _node_color_palette(M: int) -> np.ndarray:
    cmap = colormaps.get_cmap("gist_rainbow")
    colors_rgb = np.asarray([cmap(i / max(M - 1, 1))[:3] for i in range(max(M, 1))])
    return (colors_rgb[:, ::-1] * 255).astype(int)  # BGR


def _render_persistent_overlay_frame(
    dataset: CasualDataset, t: int, proj2d_np: np.ndarray,
    observed_np: np.ndarray, fit_valid_np: np.ndarray, colors_bgr: np.ndarray,
) -> np.ndarray:
    img_rgb = (dataset.get_image(t).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    M = proj2d_np.shape[1]
    for n in range(M):
        p = tuple(proj2d_np[t, n].round().astype(int))
        color = tuple(int(c) for c in colors_bgr[n])
        if fit_valid_np[t, n]:
            cv2.circle(img_bgr, p, 4, color, -1, cv2.LINE_AA)
        elif observed_np[t, n]:
            cv2.drawMarker(img_bgr, p, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
        else:
            cv2.circle(img_bgr, p, 4, color, 2, cv2.LINE_AA)
    return img_bgr


def save_previews_and_video(
    dataset: CasualDataset, proj2d: torch.Tensor, observed_mask: torch.Tensor,
    fit_valid_mask: torch.Tensor, output_dir: Path, fps: int,
) -> tuple[list[Path], Path]:
    T, M = observed_mask.shape
    colors_bgr = _node_color_palette(M)
    proj2d_np = proj2d.numpy()
    observed_np = observed_mask.numpy()
    fit_valid_np = fit_valid_mask.numpy()

    preview_paths = []
    for label, t in [("first", 0), ("mid", T // 2), ("last", T - 1)]:
        img_bgr = _render_persistent_overlay_frame(dataset, t, proj2d_np, observed_np, fit_valid_np, colors_bgr)
        out_path = output_dir / f"preview_{label}_frame{dataset.frame_names[t]}.png"
        cv2.imwrite(str(out_path), img_bgr)
        preview_paths.append(out_path)

    video_path = output_dir / "persistent_nodes_overlay.mp4"
    with imageio.get_writer(video_path, fps=fps) as writer:
        for t in range(T):
            img_bgr = _render_persistent_overlay_frame(dataset, t, proj2d_np, observed_np, fit_valid_np, colors_bgr)
            writer.append_data(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return preview_paths, video_path


def select_template_frame(
    pred_pos: torch.Tensor, source_valid: torch.Tensor, coverage_percentile: float
) -> tuple[int, dict]:
    coverage = source_valid.sum(dim=1).float()  # (T,)
    thresh = torch.quantile(coverage, 1 - coverage_percentile)
    candidates = torch.nonzero(coverage >= thresh, as_tuple=False).squeeze(-1)
    mean_pose = pred_pos.mean(dim=0)  # (M, 3)
    deviation = (pred_pos - mean_pose[None]).norm(dim=-1).mean(dim=1)  # (T,)
    best = candidates[torch.argmin(deviation[candidates])]
    template_frame = int(best.item())
    info = {
        "coverage_at_template": int(coverage[template_frame].item()),
        "deviation_at_template": float(deviation[template_frame].item()),
        "coverage_threshold": float(thresh.item()),
    }
    return template_frame, info


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    config_path = resolve_config_path(args.config)
    if args.raw_tracks_path is not None:
        raw_tracks_path = (
            args.raw_tracks_path if args.raw_tracks_path.is_absolute()
            else (_REPO_ROOT / args.raw_tracks_path).resolve()
        )
    else:
        raw_tracks_path = (
            _REPO_ROOT / "outputs" / "davis" / args.seq_name / "skeleton" / "motion_skeleton2" / "raw_tracks_3d.pt"
        )
    if not raw_tracks_path.exists():
        raise FileNotFoundError(
            f"{raw_tracks_path} not found. Run "
            f"'python flow3d/skeleton/extract_motion_tracks2.py --seq-name {args.seq_name}' first."
        )

    output_dir = _REPO_ROOT / "outputs" / "davis" / args.seq_name / "skeleton" / "persistent_nodes2"
    output_dir.mkdir(parents=True, exist_ok=True)
    persistent_nodes_path = output_dir / "persistent_nodes.pt"
    if persistent_nodes_path.exists() and not args.overwrite:
        raise FileExistsError(f"{persistent_nodes_path} already exists. Pass --overwrite to replace it.")

    before_trajectory = None
    before_available = False
    if persistent_nodes_path.exists():
        try:
            before_payload = torch.load(persistent_nodes_path)
            before_trajectory = before_payload["trajectory"].clone()
            before_available = True
        except Exception as e:  # noqa: BLE001 -- comparison is a diagnostic, never fatal
            print(f"[WARN] could not load existing persistent_nodes.pt for before/after comparison: {e}")

    start_time = time.time()
    raw = torch.load(raw_tracks_path)
    dataset, _scene_cfg = load_dataset(config_path, args.seq_name)
    T = dataset.num_frames
    if raw["meta"]["num_frames"] != T:
        raise ValueError(f"Frame count mismatch: raw_tracks_3d.pt has {raw['meta']['num_frames']}, dataset has {T}.")

    all_pos = raw["positions_world"]
    all_depth = raw["sampled_depth"]
    all_valid = raw["valid_mask"]
    all_pos2d = raw["positions_2d"]
    all_confidence = raw["confidence"]
    all_track_ids = raw["track_ids"]

    clean_valid, depth_jump_mask = compute_clean_valid_mask(all_depth, all_valid, args.depth_jump_mad_multiplier)
    valid_counts = clean_valid.sum(dim=1)
    run_lengths = longest_contiguous_run(clean_valid)
    candidate_mask = (valid_counts >= args.min_valid_frames) & (run_lengths >= args.min_continuous_frames)
    candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    num_candidates = int(candidate_idx.numel())

    selected_local = fps_select_coverage_aware(
        all_pos[candidate_idx], clean_valid[candidate_idx], args.num_nodes, args.min_common_frames, device
    )
    selected_idx = candidate_idx[selected_local]
    M = len(selected_local)

    # Canonical (T, M, ...) order throughout, matching the model's own output order
    # and the persistent_nodes.pt schema.
    source_pos = all_pos[selected_idx].swapaxes(0, 1).contiguous()  # (T, M, 3), NaN at invalid
    source_valid = clean_valid[selected_idx].swapaxes(0, 1).contiguous()  # (T, M)
    source_confidence = all_confidence[selected_idx].swapaxes(0, 1).contiguous()  # (T, M)
    source_pos2d = all_pos2d[selected_idx].swapaxes(0, 1).contiguous()  # (T, M, 2)
    node_track_ids = all_track_ids[selected_idx].clone()  # (M,), fixed forever

    # NaN-free stand-in for source_pos: 0*NaN is still NaN, so every loss below
    # reads this instead of source_pos directly and relies on masking for the
    # forward VALUE selection (not post-hoc multiplication) to stay finite.
    source_pos_safe = torch.where(source_valid.unsqueeze(-1), source_pos, torch.zeros_like(source_pos))

    # --- Canonical position + reference frame (§1) ---
    t_ref = int(source_valid.sum(dim=1).argmax().item())
    initial_trajectory = compute_initial_trajectory(source_pos_safe, source_valid)  # (T, M, 3)
    canonical_pos = initial_trajectory[t_ref].clone()  # (M, 3), never a bare zero-fill

    # --- ARAP graph (unchanged from before) ---
    dist_matrix = pairwise_track_distance(
        source_pos.swapaxes(0, 1), source_valid.swapaxes(0, 1), args.min_common_frames
    )
    arap_edges = build_mutual_knn_graph(dist_matrix, args.spatial_knn)
    if arap_edges:
        edges_i = torch.tensor([e[0] for e in arap_edges], dtype=torch.long)
        edges_j = torch.tensor([e[1] for e in arap_edges], dtype=torch.long)
        rest_dist = dist_matrix[edges_i, edges_j]
    else:
        edges_i = torch.empty((0,), dtype=torch.long)
        edges_j = torch.empty((0,), dtype=torch.long)
        rest_dist = torch.empty((0,))

    # --- Dedicated transport graph: SPATIALLY-GATED, rigidity-ranked mutual-kNN,
    # independent of ARAP (§3). dist_matrix (just computed above for the ARAP graph)
    # first restricts each node to its --transport-spatial-knn nearest candidates by
    # common-overlap median 3D distance; only inside that candidate set is rigidity
    # then ranked. Without this gate, a spatially distant node that happens to move
    # in lockstep -- the opposite leg in a symmetric gait is the textbook case --
    # could still win purely on rigidity and become a transport neighbor.
    rigidity_score = pairwise_rigidity_score(
        source_pos.swapaxes(0, 1), source_valid.swapaxes(0, 1), args.min_common_frames
    )
    transport_spatial_mask = spatial_candidate_mask(dist_matrix, args.transport_spatial_knn)
    rigidity_score_gated = torch.where(
        transport_spatial_mask, rigidity_score, torch.full_like(rigidity_score, float("nan"))
    )
    transport_edges = build_mutual_knn_graph(rigidity_score_gated, args.transport_knn)
    transport_neighbor_idx, transport_neighbor_mask = edges_to_padded_neighbors(transport_edges, M)

    # --- Fixed, pre-training outlier mask (§6) -- computed once, from raw data only ---
    fit_valid_mask, outlier_mask, node_threshold = compute_fixed_outlier_mask(
        source_pos_safe, source_valid, canonical_pos, transport_neighbor_idx, transport_neighbor_mask,
        args.switch_mad_multiplier,
    )

    skeleton_pts_list, dist_fields, norm, H, W = precompute_2d_supervision(
        dataset, T, args.max_skeleton_points_per_frame, args.seed
    )

    # Move fixed, per-run tensors to device once.
    source_pos_safe_d = source_pos_safe.to(device)
    source_valid_d = source_valid.to(device)
    fit_valid_mask_d = fit_valid_mask.to(device)
    source_confidence_d = source_confidence.to(device)
    canonical_pos_d = canonical_pos.to(device)
    transport_neighbor_idx_d = transport_neighbor_idx.to(device)
    transport_neighbor_mask_d = transport_neighbor_mask.to(device)
    Ks_d = raw["Ks"].to(device)
    w2cs_d = raw["w2cs"].to(device)
    edges_i_d, edges_j_d, rest_dist_d = edges_i.to(device), edges_j.to(device), rest_dist.to(device)
    dist_fields_d = dist_fields.to(device)
    skeleton_pts_list_d = [p.to(device) for p in skeleton_pts_list]
    time_enc = time_encoding(T, args.time_encoding_freqs, device)

    model = PersistentNodeModel(
        canonical_pos_d, t_ref, args.canonical_encoding_freqs, time_enc.shape[1], args.hidden_dim, args.num_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_log = []
    log_every = max(args.num_iters // 100, 1)
    for it in range(args.num_iters):
        optimizer.zero_grad()
        pred_pos = model(time_enc)

        l_fit = fit3d_loss(pred_pos, source_pos_safe_d, fit_valid_mask_d, source_confidence_d)
        l_acc = acceleration_loss(pred_pos)
        l_vel = velocity_match_loss(pred_pos, source_pos_safe_d, fit_valid_mask_d)
        total = args.w_fit3d * l_fit + args.w_acc * l_acc + args.w_vel_match * l_vel

        entry = {
            "iter": it, "loss_fit3d": l_fit.item(), "loss_acc": l_acc.item(), "loss_vel_match": l_vel.item(),
            "w_chamfer2d": 0.0, "w_mask": 0.0, "w_arap": 0.0, "w_neighbor_transport": 0.0,
            "loss_chamfer2d": None, "loss_mask": None, "loss_arap": None, "loss_neighbor_transport": None,
            "num_2d_frames_with_active_nodes": None, "num_active_node_frame_samples": None,
        }

        if it >= args.warmup_iters:
            ramp = min(1.0, (it - args.warmup_iters) / max(args.ramp_iters, 1))
            w_cham, w_mask_eff = args.w_chamfer2d * ramp, args.w_mask * ramp
            w_arap_eff, w_transport_eff = args.w_arap * ramp, args.w_neighbor_transport * ramp

            frame_idx = torch.randperm(T, device=device)[: args.num_2d_frames_per_iter]
            proj = project_to_2d(pred_pos[frame_idx], Ks_d[frame_idx], w2cs_d[frame_idx])
            # Only nodes actually trusted for 3D fitting at these frames may drive 2D
            # supervision -- an unobserved or outlier-excluded node's projection must
            # never receive Chamfer/mask gradient.
            active_mask = fit_valid_mask_d[frame_idx]  # (F, M)
            l_cham = chamfer2d_loss(proj, frame_idx.tolist(), skeleton_pts_list_d, norm, active_mask)
            l_mask = mask_exclusion_loss(dist_fields_d, frame_idx, proj, H, W, active_mask)
            l_arap = arap_loss(pred_pos, edges_i_d, edges_j_d, rest_dist_d)
            l_transport = neighbor_transport_loss(
                pred_pos, fit_valid_mask_d, canonical_pos_d, transport_neighbor_idx_d, transport_neighbor_mask_d
            )

            total = total + w_cham * l_cham + w_mask_eff * l_mask + w_arap_eff * l_arap + w_transport_eff * l_transport
            entry.update({
                "loss_chamfer2d": l_cham.item(), "loss_mask": l_mask.item(),
                "loss_arap": l_arap.item(), "loss_neighbor_transport": l_transport.item(),
                "w_chamfer2d": w_cham, "w_mask": w_mask_eff, "w_arap": w_arap_eff, "w_neighbor_transport": w_transport_eff,
                "num_2d_frames_with_active_nodes": int(active_mask.any(dim=1).sum().item()),
                "num_active_node_frame_samples": int(active_mask.sum().item()),
            })

        entry["loss_total"] = total.item()
        total.backward()
        optimizer.step()

        if it % log_every == 0 or it == args.num_iters - 1:
            loss_log.append(entry)
            print(f"[train_persistent_motion_nodes2] it={it} loss={entry['loss_total']:.6f}")

    with torch.no_grad():
        final_pred = model(time_enc).cpu()

    template_frame, template_info = select_template_frame(
        final_pred, source_valid, args.template_coverage_percentile
    )

    with torch.no_grad():
        proj2d_full = project_to_2d(final_pred.to(device), Ks_d, w2cs_d).cpu()  # (T, M, 2)

    # --- Switch score (post-training) + before/after comparison (§7) ---
    switch_score_after, deviation_after = compute_switch_scores(
        final_pred, canonical_pos, transport_neighbor_idx, transport_neighbor_mask
    )
    flagged_after = deviation_after > node_threshold[None, :]
    switch_events_during_observation = int((flagged_after & source_valid).sum().item())
    switch_events_during_gap = int((flagged_after & ~source_valid).sum().item())

    if before_available and before_trajectory is not None and before_trajectory.shape == final_pred.shape:
        switch_score_before, _dev_before = compute_switch_scores(
            before_trajectory, canonical_pos, transport_neighbor_idx, transport_neighbor_mask
        )
        traj_diff = (final_pred - before_trajectory).norm(dim=-1)  # (T, M)
        eps = 1e-6
        improved = (switch_score_after < switch_score_before - eps).sum().item()
        regressed = (switch_score_after > switch_score_before + eps).sum().item()
        unchanged = M - improved - regressed
        identity_switch_comparison = {
            "before_available": True,
            "switch_events_during_observation": switch_events_during_observation,
            "switch_events_during_gap": switch_events_during_gap,
            "summary": {
                "num_improved": int(improved), "num_regressed": int(regressed), "num_unchanged": int(unchanged),
                "mean_switch_score_before": float(switch_score_before.mean().item()),
                "mean_switch_score_after": float(switch_score_after.mean().item()),
            },
            "nodes": [
                {
                    "node": n,
                    "switch_score_before": float(switch_score_before[n].item()),
                    "switch_score_after": float(switch_score_after[n].item()),
                    "traj_diff_mean": float(traj_diff[:, n].mean().item()),
                    "traj_diff_max": float(traj_diff[:, n].max().item()),
                }
                for n in range(M)
            ],
        }
    else:
        identity_switch_comparison = {
            "before_available": False,
            "reason": "no prior persistent_nodes.pt (matching shape) found to compare against",
            "switch_events_during_observation": switch_events_during_observation,
            "switch_events_during_gap": switch_events_during_gap,
        }

    # --- Active-node accounting for the 2D loss (§ active_mask) ---
    active_node_count_per_frame = fit_valid_mask.sum(dim=1)  # (T,)
    total_active_node_frames = int(fit_valid_mask.sum().item())
    total_node_frames = T * M
    two_d_loss_active_nodes = {
        "active_node_count_per_frame": active_node_count_per_frame.tolist(),
        "total_active_node_frames": total_active_node_frames,
        "total_node_frames": total_node_frames,
        "excluded_node_frames_2d_loss": total_node_frames - total_active_node_frames,
        "frames_with_zero_active_nodes": int((active_node_count_per_frame == 0).sum().item()),
    }

    # --- Post-training isolation check: inactive nodes must not affect Chamfer/mask ---
    inactive_isolation_check = verify_inactive_node_isolation(
        final_pred, fit_valid_mask, Ks_d, w2cs_d, skeleton_pts_list_d, dist_fields_d, norm, H, W, device,
        args.inactive_perturb_magnitude,
    )
    if not inactive_isolation_check["passed"]:
        print(f"[WARN] inactive-node isolation check FAILED: {inactive_isolation_check}")

    # --- Synthetic regression test: spatial gate must block distant-but-rigid pairs ---
    spatial_gating_test = verify_spatial_gating_blocks_distant_rigid_pairs(
        args.transport_spatial_knn, args.transport_knn, args.min_common_frames,
    )
    if not spatial_gating_test["passed"]:
        print(f"[WARN] transport spatial-gating synthetic test FAILED: {spatial_gating_test}")

    # --- Verification metrics ---
    finite_ok = bool(torch.isfinite(final_pred).all().item())
    unique_ids = torch.unique(node_track_ids)
    fixed_id_ok = bool(unique_ids.numel() == M)

    fit_err = (final_pred - source_pos_safe).norm(dim=-1)[fit_valid_mask]
    fit3d_error_stats = {
        "mean": float(fit_err.mean().item()) if fit_err.numel() else float("nan"),
        "median": float(fit_err.median().item()) if fit_err.numel() else float("nan"),
        "p95": float(fit_err.quantile(0.95).item()) if fit_err.numel() else float("nan"),
    }
    reproj_err = (proj2d_full - source_pos2d).norm(dim=-1)[fit_valid_mask]
    reproj_error_stats = {
        "mean_px": float(reproj_err.mean().item()) if reproj_err.numel() else float("nan"),
        "median_px": float(reproj_err.median().item()) if reproj_err.numel() else float("nan"),
        "p95_px": float(reproj_err.quantile(0.95).item()) if reproj_err.numel() else float("nan"),
    }
    if reproj_error_stats["mean_px"] > 50.0:
        print(f"[WARN] mean 2D reprojection-vs-source error is high: {reproj_error_stats['mean_px']:.2f}px")

    # --- Save persistent_nodes.pt ---
    thresholds = {
        "num_nodes": args.num_nodes, "min_valid_frames": args.min_valid_frames,
        "min_continuous_frames": args.min_continuous_frames, "min_common_frames": args.min_common_frames,
        "spatial_knn": args.spatial_knn, "canonical_encoding_freqs": args.canonical_encoding_freqs,
        "transport_knn": args.transport_knn, "transport_spatial_knn": args.transport_spatial_knn,
        "time_encoding_freqs": args.time_encoding_freqs, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "num_iters": args.num_iters, "lr": args.lr,
        "w_fit3d": args.w_fit3d, "w_chamfer2d": args.w_chamfer2d, "w_mask": args.w_mask,
        "w_vel_match": args.w_vel_match, "w_acc": args.w_acc, "w_arap": args.w_arap,
        "w_neighbor_transport": args.w_neighbor_transport,
        "warmup_iters": args.warmup_iters, "ramp_iters": args.ramp_iters,
        "num_2d_frames_per_iter": args.num_2d_frames_per_iter,
        "max_skeleton_points_per_frame": args.max_skeleton_points_per_frame,
        "template_coverage_percentile": args.template_coverage_percentile,
        "depth_jump_mad_multiplier": args.depth_jump_mad_multiplier,
        "switch_mad_multiplier": args.switch_mad_multiplier,
        "inactive_perturb_magnitude": args.inactive_perturb_magnitude,
    }
    payload = {
        "trajectory": final_pred,
        "template_frame": template_frame,
        "template_node": final_pred[template_frame].clone(),
        "source_track_ids": node_track_ids,
        "observed_mask": source_valid,
        "fit_valid_mask": fit_valid_mask,
        "confidence": source_confidence,
        "arap_edges": torch.stack([edges_i, edges_j], dim=1) if edges_i.numel() else torch.empty((0, 2), dtype=torch.long),
        "arap_rest_dist": rest_dist,
        "thresholds": thresholds,
        "meta": {
            "seq_name": args.seq_name, "config_path": str(config_path), "raw_tracks_path": str(raw_tracks_path),
            "num_frames": T, "num_nodes_requested": args.num_nodes, "num_nodes_actual": M,
        },
    }
    torch.save(payload, persistent_nodes_path)

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "canonical_pos": model.canonical_pos.cpu(),
        "t_ref": t_ref,
        "deform": model.deform.state_dict(),
        "transport_neighbor_idx": transport_neighbor_idx,
        "transport_neighbor_mask": transport_neighbor_mask,
        "seed": args.seed,
        "meta": payload["meta"],
    }, checkpoint_path)

    loss_log_path = output_dir / "loss_log.json"
    with loss_log_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(loss_log), f, indent=2)

    preview_paths, video_path = save_previews_and_video(
        dataset, proj2d_full, source_valid, fit_valid_mask, output_dir, fps=10
    )

    output_paths = {
        "output_dir": output_dir,
        "persistent_nodes": persistent_nodes_path,
        "checkpoint": checkpoint_path,
        "loss_log": loss_log_path,
        "persistent_nodes_overlay_video": video_path,
        **{p.stem: p for p in preview_paths},
    }
    elapsed_seconds = time.time() - start_time

    report = {
        "seq_name": args.seq_name,
        "config_path": str(config_path),
        "raw_tracks_path": str(raw_tracks_path),
        "output_dir": str(output_dir),
        "num_frames": T,
        "node_selection": {
            "num_source_tracks": int(all_pos.shape[0]),
            "num_candidates_after_filter": num_candidates,
            "num_nodes_requested": args.num_nodes,
            "num_nodes_actual": M,
        },
        "depth_jump_stats": {
            "total_frames_flagged": int(depth_jump_mask.sum().item()),
            "tracks_affected": int(depth_jump_mask.any(dim=1).sum().item()),
        },
        "canonical_reference": {"t_ref": t_ref, "template_frame": template_frame},
        "transport_graph": {
            "num_edges": len(transport_edges),
            "max_degree": int(transport_neighbor_mask.sum(dim=1).max().item()) if M else 0,
            "num_nodes_with_zero_neighbors": int((transport_neighbor_mask.sum(dim=1) == 0).sum().item()),
            "num_nodes_degree_lt_3": int((transport_neighbor_mask.sum(dim=1) < 3).sum().item()),
            "transport_spatial_knn": args.transport_spatial_knn,
            "degree_per_node": transport_neighbor_mask.sum(dim=1).tolist(),
            "neighbor_ids_per_node": [
                transport_neighbor_idx[m][transport_neighbor_mask[m]].tolist() for m in range(M)
            ],
            "edge_spatial_dist_median": (
                float(torch.tensor([dist_matrix[i, j] for i, j in transport_edges]).median().item())
                if transport_edges else float("nan")
            ),
            "edge_spatial_dist_max": (
                float(torch.tensor([dist_matrix[i, j] for i, j in transport_edges]).max().item())
                if transport_edges else float("nan")
            ),
        },
        "training_time_outlier_exclusion": {
            "count": int(outlier_mask.sum().item()),
            "num_nodes_affected": int(outlier_mask.any(dim=0).sum().item()),
            "switch_mad_multiplier": args.switch_mad_multiplier,
            "mean_node_threshold": float(node_threshold[torch.isfinite(node_threshold)].mean().item())
                if torch.isfinite(node_threshold).any() else float("nan"),
        },
        "two_d_loss_active_nodes": two_d_loss_active_nodes,
        "identity_switch_comparison": identity_switch_comparison,
        "thresholds": thresholds,
        "final_loss": loss_log[-1] if loss_log else {},
        "template_frame": {"frame": template_frame, **template_info},
        "verification": {
            "fixed_id_ok": fixed_id_ok,
            "finite_trajectory_ok": finite_ok,
            "fit3d_error": fit3d_error_stats,
            "reprojection_vs_source_error": reproj_error_stats,
            "inactive_node_isolation_check": inactive_isolation_check,
            "transport_spatial_gating_test": spatial_gating_test,
        },
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "elapsed_seconds": elapsed_seconds,
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2)

    print(
        f"[train_persistent_motion_nodes2] seq={args.seq_name} num_nodes={M} "
        f"template_frame={template_frame} t_ref={t_ref} fixed_id_ok={fixed_id_ok} finite_ok={finite_ok} "
        f"fit3d_mean_err={fit3d_error_stats['mean']:.6g} reproj_mean_px={reproj_error_stats['mean_px']:.3g} "
        f"switch_obs={switch_events_during_observation} switch_gap={switch_events_during_gap} "
        f"active_node_frames={total_active_node_frames}/{total_node_frames} "
        f"isolation_check_passed={inactive_isolation_check['passed']} "
        f"spatial_gating_test_passed={spatial_gating_test['passed']} "
        f"before_available={before_available} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

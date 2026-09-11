"""
Train a shared MLP deformation field for persistent 3D motion nodes.

Independent re-implementation of the general idea behind RigGS's stage-1 node
deformation + 2D skeleton supervision, adapted to this project's own data
(raw_tracks_3d.pt, CasualDataset) rather than ported from RigGS code. Only
raw_tracks_3d.pt (from extract_motion_tracks.py) is used as input;
build_motion_nodes.py's output and any pre-existing cluster ids are never
read. A coverage-aware FPS pass selects ~256 nodes from tracks with enough
valid AND contiguous observation; a fixed per-node embedding plus a time
encoding, through one shared MLP, predict a residual on top of a fixed,
data-driven initial (interpolated/held) trajectory -- so node identity can
never disappear or get reassigned, and the trajectory is finite everywhere
even where the source track had no observation. Supervision combines 3D
fitting to the source track (confidence-weighted), a 2D Chamfer loss against
a thinned SAM-mask skeleton, an outside-mask projection penalty, temporal
smoothness (acceleration + observed-velocity matching), and a local kNN ARAP
term. Rigid-part clustering, Kabsch, MST, joints, and any final skeleton are
explicitly out of scope for this script.

Output
------
outputs/davis/<seq-name>/skeleton/persistent_nodes/
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
    python flow3d/skeleton/train_persistent_motion_nodes.py --seq-name camel --overwrite
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
             "outputs/davis/<seq-name>/skeleton/motion_skeleton/raw_tracks_3d.pt.",
    )
    parser.add_argument("--num-nodes", type=int, default=256)
    parser.add_argument("--min-valid-frames", type=int, default=20)
    parser.add_argument("--min-continuous-frames", type=int, default=10)
    parser.add_argument("--min-common-frames", type=int, default=10)
    parser.add_argument("--spatial-knn", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=32)
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
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--ramp-iters", type=int, default=1000)
    parser.add_argument("--num-2d-frames-per-iter", type=int, default=12)
    parser.add_argument("--max-skeleton-points-per-frame", type=int, default=256)
    parser.add_argument("--template-coverage-percentile", type=float, default=0.5)
    parser.add_argument(
        "--depth-jump-mad-multiplier", type=float, default=5.0,
        help="Robust (median+k*MAD) threshold multiplier for the momentary-depth-jump detector.",
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
    """(T, M, 3) fixed baseline: per node, linear interpolation across its valid
    observations, with nearest-observation (flat) hold before/after the first/last
    valid frame -- exactly np.interp's default out-of-range behavior."""
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
# Model: fixed node embedding + time encoding -> shared MLP -> residual on top
# of the fixed initial_trajectory. Last layer zero-initialized so pred == the
# source-track-following baseline at iteration 0.
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


class PersistentNodeModel(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int, time_enc_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        self.residual_mlp = ResidualMLP(embed_dim + time_enc_dim, hidden_dim, num_layers)

    def forward(self, time_enc: torch.Tensor) -> torch.Tensor:
        T = time_enc.shape[0]
        M = self.node_embedding.num_embeddings
        emb = self.node_embedding.weight
        x = torch.cat(
            [emb[None, :, :].expand(T, -1, -1), time_enc[:, None, :].expand(-1, M, -1)], dim=-1
        ).reshape(T * M, -1)
        return self.residual_mlp(x).reshape(T, M, 3)


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

def fit3d_loss(pred_pos, source_pos_safe, source_valid, confidence) -> torch.Tensor:
    diff = F.smooth_l1_loss(pred_pos, source_pos_safe, reduction="none").sum(dim=-1)  # (T, M)
    weight = confidence * source_valid.float()
    denom = weight.sum().clamp(min=1e-8)
    return (weight * diff).sum() / denom


def acceleration_loss(pred_pos: torch.Tensor) -> torch.Tensor:
    acc = pred_pos[2:] - 2 * pred_pos[1:-1] + pred_pos[:-2]
    return (acc ** 2).sum(dim=-1).mean()


def velocity_match_loss(pred_pos, source_pos_safe, source_valid) -> torch.Tensor:
    pred_vel = pred_pos[1:] - pred_pos[:-1]
    source_vel = source_pos_safe[1:] - source_pos_safe[:-1]
    pair_valid = (source_valid[1:] & source_valid[:-1]).float()
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


def chamfer2d_loss(
    proj_xy: torch.Tensor, frame_indices: list[int], skeleton_pts_list: list[torch.Tensor], norm: float
) -> torch.Tensor:
    total = proj_xy.new_zeros(())
    count = 0
    for i, t in enumerate(frame_indices):
        pts = skeleton_pts_list[t]
        if pts.numel() == 0:
            continue
        nodes_t = proj_xy[i]
        d = torch.cdist(nodes_t, pts.to(nodes_t.device))
        node_to_skel = d.min(dim=1).values.mean()
        skel_to_node = d.min(dim=0).values.mean()
        total = total + (node_to_skel + skel_to_node) / norm
        count += 1
    return total / count if count > 0 else total


def mask_exclusion_loss(
    dist_fields: torch.Tensor, frame_indices: torch.Tensor, proj_xy: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    coords = normalize_coords_local(proj_xy, H, W)  # (F, M, 2)
    field = dist_fields[frame_indices][:, None]  # (F, 1, H, W)
    grid = coords[:, None, :, :]  # (F, 1, M, 2)
    sampled = F.grid_sample(field, grid, align_corners=True, padding_mode="border")[:, 0, 0, :]
    return sampled.mean()


# ---------------------------------------------------------------------------
# Previews / video: one fixed color per global node id; filled circle if
# observed at that frame, hollow ring if MLP-completed.
# ---------------------------------------------------------------------------

def _node_color_palette(M: int) -> np.ndarray:
    cmap = colormaps.get_cmap("gist_rainbow")
    colors_rgb = np.asarray([cmap(i / max(M - 1, 1))[:3] for i in range(max(M, 1))])
    return (colors_rgb[:, ::-1] * 255).astype(int)  # BGR


def _render_persistent_overlay_frame(
    dataset: CasualDataset, t: int, proj2d_np: np.ndarray, observed_np: np.ndarray, colors_bgr: np.ndarray
) -> np.ndarray:
    img_rgb = (dataset.get_image(t).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    M = proj2d_np.shape[1]
    for n in range(M):
        p = tuple(proj2d_np[t, n].round().astype(int))
        color = tuple(int(c) for c in colors_bgr[n])
        if observed_np[t, n]:
            cv2.circle(img_bgr, p, 4, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img_bgr, p, 4, color, 2, cv2.LINE_AA)
    return img_bgr


def save_previews_and_video(
    dataset: CasualDataset, proj2d: torch.Tensor, observed_mask: torch.Tensor, output_dir: Path, fps: int
) -> tuple[list[Path], Path]:
    T, M = observed_mask.shape
    colors_bgr = _node_color_palette(M)
    proj2d_np = proj2d.numpy()
    observed_np = observed_mask.numpy()

    preview_paths = []
    for label, t in [("first", 0), ("mid", T // 2), ("last", T - 1)]:
        img_bgr = _render_persistent_overlay_frame(dataset, t, proj2d_np, observed_np, colors_bgr)
        out_path = output_dir / f"preview_{label}_frame{dataset.frame_names[t]}.png"
        cv2.imwrite(str(out_path), img_bgr)
        preview_paths.append(out_path)

    video_path = output_dir / "persistent_nodes_overlay.mp4"
    with imageio.get_writer(video_path, fps=fps) as writer:
        for t in range(T):
            img_bgr = _render_persistent_overlay_frame(dataset, t, proj2d_np, observed_np, colors_bgr)
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
            _REPO_ROOT / "outputs" / "davis" / args.seq_name / "skeleton" / "motion_skeleton" / "raw_tracks_3d.pt"
        )
    if not raw_tracks_path.exists():
        raise FileNotFoundError(
            f"{raw_tracks_path} not found. Run "
            f"'python flow3d/skeleton/extract_motion_tracks.py --seq-name {args.seq_name}' first."
        )

    output_dir = _REPO_ROOT / "outputs" / "davis" / args.seq_name / "skeleton" / "persistent_nodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    persistent_nodes_path = output_dir / "persistent_nodes.pt"
    if persistent_nodes_path.exists() and not args.overwrite:
        raise FileExistsError(f"{persistent_nodes_path} already exists. Pass --overwrite to replace it.")

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

    initial_trajectory = compute_initial_trajectory(source_pos_safe, source_valid)

    skeleton_pts_list, dist_fields, norm, H, W = precompute_2d_supervision(
        dataset, T, args.max_skeleton_points_per_frame, args.seed
    )

    source_pos_safe_d = source_pos_safe.to(device)
    source_valid_d = source_valid.to(device)
    source_confidence_d = source_confidence.to(device)
    initial_trajectory_d = initial_trajectory.to(device)
    Ks_d = raw["Ks"].to(device)
    w2cs_d = raw["w2cs"].to(device)
    edges_i_d, edges_j_d, rest_dist_d = edges_i.to(device), edges_j.to(device), rest_dist.to(device)
    dist_fields_d = dist_fields.to(device)
    skeleton_pts_list_d = [p.to(device) for p in skeleton_pts_list]
    time_enc = time_encoding(T, args.time_encoding_freqs, device)

    model = PersistentNodeModel(M, args.embed_dim, time_enc.shape[1], args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_log = []
    log_every = max(args.num_iters // 100, 1)
    for it in range(args.num_iters):
        optimizer.zero_grad()
        delta = model(time_enc)
        pred_pos = initial_trajectory_d + delta

        l_fit = fit3d_loss(pred_pos, source_pos_safe_d, source_valid_d, source_confidence_d)
        l_acc = acceleration_loss(pred_pos)
        l_vel = velocity_match_loss(pred_pos, source_pos_safe_d, source_valid_d)
        total = args.w_fit3d * l_fit + args.w_acc * l_acc + args.w_vel_match * l_vel

        entry = {
            "iter": it, "loss_fit3d": l_fit.item(), "loss_acc": l_acc.item(), "loss_vel_match": l_vel.item(),
            "w_chamfer2d": 0.0, "w_mask": 0.0, "w_arap": 0.0,
            "loss_chamfer2d": None, "loss_mask": None, "loss_arap": None,
        }

        if it >= args.warmup_iters:
            ramp = min(1.0, (it - args.warmup_iters) / max(args.ramp_iters, 1))
            w_cham, w_mask_eff, w_arap_eff = args.w_chamfer2d * ramp, args.w_mask * ramp, args.w_arap * ramp

            frame_idx = torch.randperm(T, device=device)[: args.num_2d_frames_per_iter]
            proj = project_to_2d(pred_pos[frame_idx], Ks_d[frame_idx], w2cs_d[frame_idx])
            l_cham = chamfer2d_loss(proj, frame_idx.tolist(), skeleton_pts_list_d, norm)
            l_mask = mask_exclusion_loss(dist_fields_d, frame_idx, proj, H, W)
            l_arap = arap_loss(pred_pos, edges_i_d, edges_j_d, rest_dist_d)

            total = total + w_cham * l_cham + w_mask_eff * l_mask + w_arap_eff * l_arap
            entry.update({
                "loss_chamfer2d": l_cham.item(), "loss_mask": l_mask.item(), "loss_arap": l_arap.item(),
                "w_chamfer2d": w_cham, "w_mask": w_mask_eff, "w_arap": w_arap_eff,
            })

        entry["loss_total"] = total.item()
        total.backward()
        optimizer.step()

        if it % log_every == 0 or it == args.num_iters - 1:
            loss_log.append(entry)
            print(f"[train_persistent_motion_nodes] it={it} loss={entry['loss_total']:.6f}")

    with torch.no_grad():
        final_pred = (initial_trajectory_d + model(time_enc)).cpu()

    template_frame, template_info = select_template_frame(
        final_pred, source_valid, args.template_coverage_percentile
    )

    with torch.no_grad():
        proj2d_full = project_to_2d(final_pred.to(device), Ks_d, w2cs_d).cpu()

    finite_ok = bool(torch.isfinite(final_pred).all().item())
    unique_ids = torch.unique(node_track_ids)
    fixed_id_ok = bool(unique_ids.numel() == M)

    fit_err = (final_pred - source_pos_safe).norm(dim=-1)[source_valid]
    fit3d_error_stats = {
        "mean": float(fit_err.mean().item()) if fit_err.numel() else float("nan"),
        "median": float(fit_err.median().item()) if fit_err.numel() else float("nan"),
        "p95": float(fit_err.quantile(0.95).item()) if fit_err.numel() else float("nan"),
    }
    reproj_err = (proj2d_full - source_pos2d).norm(dim=-1)[source_valid]
    reproj_error_stats = {
        "mean_px": float(reproj_err.mean().item()) if reproj_err.numel() else float("nan"),
        "median_px": float(reproj_err.median().item()) if reproj_err.numel() else float("nan"),
        "p95_px": float(reproj_err.quantile(0.95).item()) if reproj_err.numel() else float("nan"),
    }
    if reproj_error_stats["mean_px"] > 50.0:
        print(f"[WARN] mean 2D reprojection-vs-source error is high: {reproj_error_stats['mean_px']:.2f}px")

    thresholds = {
        "num_nodes": args.num_nodes, "min_valid_frames": args.min_valid_frames,
        "min_continuous_frames": args.min_continuous_frames, "min_common_frames": args.min_common_frames,
        "spatial_knn": args.spatial_knn, "embed_dim": args.embed_dim,
        "time_encoding_freqs": args.time_encoding_freqs, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "num_iters": args.num_iters, "lr": args.lr,
        "w_fit3d": args.w_fit3d, "w_chamfer2d": args.w_chamfer2d, "w_mask": args.w_mask,
        "w_vel_match": args.w_vel_match, "w_acc": args.w_acc, "w_arap": args.w_arap,
        "warmup_iters": args.warmup_iters, "ramp_iters": args.ramp_iters,
        "num_2d_frames_per_iter": args.num_2d_frames_per_iter,
        "max_skeleton_points_per_frame": args.max_skeleton_points_per_frame,
        "template_coverage_percentile": args.template_coverage_percentile,
        "depth_jump_mad_multiplier": args.depth_jump_mad_multiplier,
    }
    payload = {
        "trajectory": final_pred,
        "template_frame": template_frame,
        "template_node": final_pred[template_frame].clone(),
        "source_track_ids": node_track_ids,
        "observed_mask": source_valid,
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
        "node_embedding": model.node_embedding.state_dict(),
        "residual_mlp": model.residual_mlp.state_dict(),
        "seed": args.seed,
        "meta": payload["meta"],
    }, checkpoint_path)

    loss_log_path = output_dir / "loss_log.json"
    with loss_log_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(loss_log), f, indent=2)

    preview_paths, video_path = save_previews_and_video(
        dataset, proj2d_full, source_valid, output_dir, fps=10
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
        "thresholds": thresholds,
        "final_loss": loss_log[-1] if loss_log else {},
        "template_frame": {"frame": template_frame, **template_info},
        "verification": {
            "fixed_id_ok": fixed_id_ok,
            "finite_trajectory_ok": finite_ok,
            "fit3d_error": fit3d_error_stats,
            "reprojection_vs_source_error": reproj_error_stats,
        },
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "elapsed_seconds": elapsed_seconds,
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2)

    print(
        f"[train_persistent_motion_nodes] seq={args.seq_name} num_nodes={M} "
        f"template_frame={template_frame} fixed_id_ok={fixed_id_ok} finite_ok={finite_ok} "
        f"fit3d_mean_err={fit3d_error_stats['mean']:.6g} reproj_mean_px={reproj_error_stats['mean_px']:.3g} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

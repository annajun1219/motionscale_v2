"""
flow3d/experiment/contactloss/frame_contact_loss.py

Frame/pair-gated contact loss.

simple_contact_loss.py는 등록된 모든 pair, 모든 frame에 항상 loss를
계산한다 (margin 안쪽이면 relu가 0을 만들지만, 계산 자체와 pair 평균에는
매 step 참여한다). 이 버전은 그 대신:

- pair gate:
    flow3d/analysis/contactpatch_distance.py가 만든
    contact_patch_pair_summary.csv의 flagged_frame_count가
    --min-flagged-frames 이상인 pair만 등록한다.
    (전체 시퀀스에서 거의 안 벌어지는 pair는 애초에 등록하지 않는다.)

- frame gate:
    등록된 pair라도, 같은 CSV의 flagged_frames 목록에 있는 frame에서만
    loss를 적용한다. flagged로 판정되지 않은 frame에서는 해당 pair의
    기여가 정확히 0이다 (거리 계산 자체를 건너뛴다).

flagged 판정 기준은 새로 만들지 않고 contactpatch_distance.py가 이미 계산한
결과를 그대로 재사용한다 (동일 기준 유지, 판정 로직 중복 방지).

입력
----
patch_path:
    <work-dir>/analysis/contact_patches/contact_patches.pt
    (simple_contact_loss.py와 동일 형식)

summary_csv_path:
    <work-dir>/analysis/contact_patch_distances/contact_patch_pair_summary.csv
    (contactpatch_distance.py 출력. pair_key, flagged_frame_count,
    flagged_frames 컬럼을 사용한다.)

사용 예
-------
    loss_module = build_frame_gated_contact_loss(
        patch_path=".../contact_patches.pt",
        summary_csv_path=".../contact_patch_pair_summary.csv",
        margin=0.02,
        device="cuda",
    )
    ...
    loss = loss_module(means_fg, frame_idx)   # frame_idx: 현재 frame 번호 (int)

run_contact_experiment.py의 install_contact_loss_patch에 연동하려면, 거기서
`contact_loss(means_fg[:, i])` 호출을 `contact_loss(means_fg[:, i],
frame_idx=int(batch["ts"][i]))`로 바꿔주면 된다 (frame_idx가 필수 인자로
추가된 것 외에는 SimpleContactLoss와 같은 방식으로 쓸 수 있다).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow3d.experiment.contactloss.simple_contact_loss import load_contact_patch_pairs


@dataclass(frozen=True)
class FrameGatedPair:
    cluster_a: int
    cluster_b: int
    patch_a_indices: Tensor
    patch_b_indices: Tensor
    flagged_frames: FrozenSet[int]

    @property
    def pair_name(self) -> str:
        return f"{self.cluster_a}-{self.cluster_b}"


def load_pair_frame_flags(
    summary_csv_path: str | Path,
) -> Dict[Tuple[int, int], FrozenSet[int]]:
    """
    contactpatch_distance.py의 contact_patch_pair_summary.csv를 읽어
    pair -> flagged frame 번호 집합을 만든다.
    """
    summary_csv_path = Path(summary_csv_path)
    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f"Pair summary CSV not found: {summary_csv_path}. "
            "Run flow3d/analysis/contactpatch_distance.py first."
        )

    flags: Dict[Tuple[int, int], FrozenSet[int]] = {}

    with summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cluster_a = int(row["cluster_a"])
            cluster_b = int(row["cluster_b"])
            raw_frames = (row.get("flagged_frames") or "").strip()

            frames = (
                frozenset(int(x) for x in raw_frames.split(",") if x.strip())
                if raw_frames
                else frozenset()
            )

            key = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
            flags[key] = frames

    if not flags:
        raise ValueError(f"No pair rows found in {summary_csv_path}")

    return flags


def load_frame_gated_pairs(
    patch_path: str | Path,
    summary_csv_path: str | Path,
    min_flagged_frames: int = 5,
) -> List[FrameGatedPair]:
    """
    contact_patches.pt의 pair 중, contactpatch_distance.py가 판정한
    flagged_frame_count가 min_flagged_frames 이상인 pair만 남기고,
    각 pair에 flagged frame 집합을 붙인다.
    """
    if min_flagged_frames < 1:
        raise ValueError("min_flagged_frames must be >= 1.")

    all_pairs = load_contact_patch_pairs(patch_path)
    frame_flags = load_pair_frame_flags(summary_csv_path)

    gated_pairs: List[FrameGatedPair] = []
    skipped: List[str] = []

    for pair in all_pairs:
        key = (min(pair.cluster_a, pair.cluster_b), max(pair.cluster_a, pair.cluster_b))
        flagged = frame_flags.get(key)

        if flagged is None:
            # summary CSV에 없는 pair(다른 candidate 집합으로 돌린 결과 등)는
            # gating 정보가 없으므로 안전하게 제외한다.
            skipped.append(f"{key[0]}-{key[1]} (no row in summary CSV)")
            continue

        if len(flagged) < min_flagged_frames:
            skipped.append(
                f"{key[0]}-{key[1]} (flagged_frame_count={len(flagged)} "
                f"< min_flagged_frames={min_flagged_frames})"
            )
            continue

        gated_pairs.append(
            FrameGatedPair(
                cluster_a=pair.cluster_a,
                cluster_b=pair.cluster_b,
                patch_a_indices=pair.patch_a_indices,
                patch_b_indices=pair.patch_b_indices,
                flagged_frames=flagged,
            )
        )

    if not gated_pairs:
        raise ValueError(
            "No pair passed the pair gate "
            f"(min_flagged_frames={min_flagged_frames}). Skipped: {skipped}"
        )

    print(f"[frame_contact_loss] registered {len(gated_pairs)} pair(s): "
          f"{[p.pair_name for p in gated_pairs]}")
    if skipped:
        print(f"[frame_contact_loss] skipped {len(skipped)} pair(s) (pair gate): {skipped}")

    return gated_pairs


class FrameContactLoss(nn.Module):
    """
    등록된 pair 중, 현재 frame이 그 pair의 flagged frame 집합에 속할 때만
    penalty를 계산하는 contact loss.

    frame이 gate를 통과하지 못하면 그 pair는 이번 step에서 정확히 기여 0
    이다 (거리 계산 자체를 건너뛴다). 등록된 pair가 전부 gate를 통과하지
    못하면 전체 loss는 0이다.

    pair별 loss 정의는 simple_contact_loss.SimpleContactLoss와 동일하다:
        0.5 * (
            mean(smooth_l1(relu(d_ab - margin)))
            + mean(smooth_l1(relu(d_ba - margin)))
        )
    """

    def __init__(
        self,
        gated_pairs: List[FrameGatedPair],
        margin: float,
        beta: float = 0.002,
        max_points_per_patch: int | None = None,
    ) -> None:
        super().__init__()

        if margin < 0:
            raise ValueError("margin must be >= 0.")
        if beta <= 0:
            raise ValueError("beta must be > 0.")
        if max_points_per_patch is not None and max_points_per_patch <= 0:
            raise ValueError("max_points_per_patch must be positive or None.")
        if not gated_pairs:
            raise ValueError("At least one frame-gated pair is required.")

        self.margin = float(margin)
        self.beta = float(beta)
        self.max_points_per_patch = max_points_per_patch

        self._pair_info: List[Tuple[int, int, str, str, FrozenSet[int]]] = []

        for pair_idx, pair in enumerate(gated_pairs):
            name_a = f"patch_a_indices_{pair_idx}"
            name_b = f"patch_b_indices_{pair_idx}"

            self.register_buffer(name_a, pair.patch_a_indices.clone().long(), persistent=True)
            self.register_buffer(name_b, pair.patch_b_indices.clone().long(), persistent=True)

            self._pair_info.append(
                (pair.cluster_a, pair.cluster_b, name_a, name_b, pair.flagged_frames)
            )

    def _subsample(self, points: Tensor) -> Tensor:
        cap = self.max_points_per_patch
        if cap is None or points.shape[0] <= cap:
            return points
        selected = torch.randperm(points.shape[0], device=points.device)[:cap]
        return points.index_select(0, selected)

    def _one_direction_loss(self, nearest_distances: Tensor) -> Tensor:
        excess = F.relu(nearest_distances - self.margin)
        return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=self.beta, reduction="mean")

    def active_pairs(self, frame_idx: int) -> List[str]:
        """이 frame에서 실제로 loss가 걸리는 pair 이름 목록."""
        return [f"{a}-{b}" for a, b, _, _, flagged in self._pair_info if frame_idx in flagged]

    def forward(
        self,
        means_fg: Tensor,
        frame_idx: int,
        return_stats: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        means_fg:
            [G_fg, 3], 현재 frame_idx에서 deformation이 적용된 foreground
            Gaussian 위치. detach하면 안 된다.
        frame_idx:
            현재 학습 frame 번호. contact_patch_pair_summary.csv의
            flagged_frames와 같은 인덱싱을 써야 한다.
        """
        if means_fg.ndim != 2 or means_fg.shape[-1] != 3:
            raise ValueError(f"means_fg must have shape [G_fg, 3], got {tuple(means_fg.shape)}")
        if not means_fg.is_floating_point():
            raise TypeError("means_fg must be a floating-point tensor.")

        num_fg = means_fg.shape[0]
        pair_losses: List[Tensor] = []
        stats: Dict[str, Tensor] = {}

        for cluster_a, cluster_b, name_a, name_b, flagged_frames in self._pair_info:
            pair_name = f"{cluster_a}-{cluster_b}"

            if frame_idx not in flagged_frames:
                if return_stats:
                    stats[f"{pair_name}/active"] = means_fg.new_zeros(())
                continue

            idx_a = getattr(self, name_a)
            idx_b = getattr(self, name_b)

            if idx_a.max().item() >= num_fg:
                raise IndexError(f"Pair {pair_name}: patch A index exceeds num_fg={num_fg}.")
            if idx_b.max().item() >= num_fg:
                raise IndexError(f"Pair {pair_name}: patch B index exceeds num_fg={num_fg}.")

            points_a = self._subsample(means_fg.index_select(0, idx_a))
            points_b = self._subsample(means_fg.index_select(0, idx_b))

            pairwise_dist = torch.cdist(points_a.float(), points_b.float(), p=2)
            nearest_ab = pairwise_dist.min(dim=1).values
            nearest_ba = pairwise_dist.min(dim=0).values

            loss_ab = self._one_direction_loss(nearest_ab)
            loss_ba = self._one_direction_loss(nearest_ba)
            pair_loss = 0.5 * (loss_ab + loss_ba)
            pair_losses.append(pair_loss)

            if return_stats:
                symmetric_nn = torch.cat([nearest_ab, nearest_ba], dim=0)
                stats[f"{pair_name}/active"] = means_fg.new_ones(())
                stats[f"{pair_name}/loss"] = pair_loss.detach()
                stats[f"{pair_name}/nn_mean"] = symmetric_nn.mean().detach()
                stats[f"{pair_name}/nn_max"] = symmetric_nn.max().detach()

        total_loss = torch.stack(pair_losses).mean() if pair_losses else means_fg.new_zeros(())

        if return_stats:
            stats["total"] = total_loss.detach()
            stats["num_active_pairs"] = means_fg.new_tensor(float(len(pair_losses)))
            return total_loss, stats

        return total_loss


def build_frame_gated_contact_loss(
    patch_path: str | Path,
    summary_csv_path: str | Path,
    margin: float,
    device: torch.device | str,
    beta: float = 0.002,
    min_flagged_frames: int = 5,
    max_points_per_patch: int | None = None,
) -> FrameContactLoss:
    gated_pairs = load_frame_gated_pairs(patch_path, summary_csv_path, min_flagged_frames)

    module = FrameContactLoss(
        gated_pairs=gated_pairs,
        margin=margin,
        beta=beta,
        max_points_per_patch=max_points_per_patch,
    )

    return module.to(torch.device(device))

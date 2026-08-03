"""
flow3d/experiment/contactloss/simple_contact_loss.py

단순 contact loss 전용.

- 모든 학습 frame에서 항상 적용
- 선택적 frame/pair gate 없음
- contact patch Gaussian index는 고정
- Gaussian 간 대응쌍은 고정하지 않음
- 현재 frame마다 A patch <-> B patch 최근접 이웃을 다시 계산
- 허용 거리 margin보다 멀어진 부분만 penalty

입력:
    means_fg: [G_fg, 3]
        현재 frame에서 deformation이 적용된 foreground Gaussian 위치

지원 patch 파일 형식(.pt)
-------------------------
실제 contact patch 선택 코드의 출력 형식:

{
    "pairs": {
        "14_22": {
            "cluster_a": 14,
            "cluster_b": 22,
            "contact_patch_global_indices_a": LongTensor([...]),
            "contact_patch_global_indices_b": LongTensor([...]),
            ...
        },
        ...
    }
}

이 파일에서 저장된 global index는 foreground Gaussian 배열
(model.fg.params["means"]) 기준 index이므로 means_fg에 그대로 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ContactPatchPair:
    cluster_a: int
    cluster_b: int
    patch_a_indices: Tensor
    patch_b_indices: Tensor

    @property
    def pair_name(self) -> str:
        return f"{self.cluster_a}-{self.cluster_b}"


def _to_long_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x.detach().to(dtype=torch.long, device="cpu").flatten()
    return torch.as_tensor(x, dtype=torch.long).flatten()


def _normalize_pair_entries(raw_pairs: Any) -> List[Mapping[str, Any]]:
    """
    `pairs`가 dict이든 list이든 pair entry list로 정규화한다.
    """
    if isinstance(raw_pairs, Mapping):
        entries = list(raw_pairs.values())
    elif isinstance(raw_pairs, Sequence) and not isinstance(
        raw_pairs, (str, bytes)
    ):
        entries = list(raw_pairs)
    else:
        raise TypeError(
            "'pairs' must be either a mapping or a sequence, "
            f"got {type(raw_pairs).__name__}."
        )

    for idx, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Pair entry {idx} must be a mapping, "
                f"got {type(item).__name__}."
            )

    return entries


def load_contact_patch_pairs(
    patch_path: str | Path,
) -> List[ContactPatchPair]:
    """
    contact_patches.pt를 읽는다.

    기본적으로 실제 contact patch 선택 코드의 출력 형식을 사용한다:

        raw["pairs"][pair_key][
            "contact_patch_global_indices_a"
        ]
        raw["pairs"][pair_key][
            "contact_patch_global_indices_b"
        ]

    이전 테스트 형식의 key도 호환 목적으로 지원한다:

        patch_a_indices
        patch_b_indices
    """
    patch_path = Path(patch_path)

    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    raw = torch.load(
        patch_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(raw, Mapping):
        raw_pairs = raw.get("pairs")
        if raw_pairs is None:
            raise KeyError(
                f"{patch_path} must contain a top-level 'pairs' key."
            )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        raw_pairs = raw
    else:
        raise TypeError(
            f"Unsupported patch file type: {type(raw).__name__}"
        )

    pair_entries = _normalize_pair_entries(raw_pairs)
    pairs: List[ContactPatchPair] = []

    for pair_idx, item in enumerate(pair_entries):
        cluster_a = int(item["cluster_a"])
        cluster_b = int(item["cluster_b"])

        indices_a = item.get(
            "contact_patch_global_indices_a",
            item.get("patch_a_indices"),
        )
        indices_b = item.get(
            "contact_patch_global_indices_b",
            item.get("patch_b_indices"),
        )

        if indices_a is None or indices_b is None:
            raise KeyError(
                f"Pair {cluster_a}-{cluster_b} must contain "
                "'contact_patch_global_indices_a/b'."
            )

        idx_a = _to_long_tensor(indices_a)
        idx_b = _to_long_tensor(indices_b)

        if idx_a.numel() == 0 or idx_b.numel() == 0:
            raise ValueError(
                f"Pair {cluster_a}-{cluster_b} has an empty patch."
            )

        if idx_a.min().item() < 0 or idx_b.min().item() < 0:
            raise ValueError(
                f"Pair {cluster_a}-{cluster_b} contains "
                "a negative Gaussian index."
            )

        pairs.append(
            ContactPatchPair(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                patch_a_indices=idx_a,
                patch_b_indices=idx_b,
            )
        )

    if not pairs:
        raise ValueError(f"No contact patch pairs found in {patch_path}")

    return pairs


class SimpleContactLoss(nn.Module):
    """
    모든 frame, 모든 등록 pair에 항상 적용되는 단순 contact loss.

    A -> B:
        A patch 각 Gaussian에서 B patch까지 최근접 거리 계산

    B -> A:
        B patch 각 Gaussian에서 A patch까지 최근접 거리 계산

    loss:
        0.5 * (
            mean(smooth_l1(relu(d_ab - margin)))
            + mean(smooth_l1(relu(d_ba - margin)))
        )

    margin 안쪽 거리는 penalty가 없다.
    """

    def __init__(
        self,
        patch_pairs: Sequence[ContactPatchPair],
        margin: float,
        beta: float = 0.002,
        max_points_per_patch: int | None = None,
    ) -> None:
        super().__init__()

        if margin < 0:
            raise ValueError("margin must be >= 0.")

        if beta <= 0:
            raise ValueError("beta must be > 0.")

        if (
            max_points_per_patch is not None
            and max_points_per_patch <= 0
        ):
            raise ValueError(
                "max_points_per_patch must be positive or None."
            )

        self.margin = float(margin)
        self.beta = float(beta)
        self.max_points_per_patch = max_points_per_patch

        self._pair_info: List[Tuple[int, int, str, str]] = []

        for pair_idx, pair in enumerate(patch_pairs):
            name_a = f"patch_a_indices_{pair_idx}"
            name_b = f"patch_b_indices_{pair_idx}"

            self.register_buffer(
                name_a,
                pair.patch_a_indices.clone().long(),
                persistent=True,
            )
            self.register_buffer(
                name_b,
                pair.patch_b_indices.clone().long(),
                persistent=True,
            )

            self._pair_info.append(
                (
                    pair.cluster_a,
                    pair.cluster_b,
                    name_a,
                    name_b,
                )
            )

        if not self._pair_info:
            raise ValueError(
                "At least one contact patch pair is required."
            )

    def _subsample(self, points: Tensor) -> Tensor:
        cap = self.max_points_per_patch

        if cap is None or points.shape[0] <= cap:
            return points

        selected = torch.randperm(
            points.shape[0],
            device=points.device,
        )[:cap]

        return points.index_select(0, selected)

    def _one_direction_loss(
        self,
        nearest_distances: Tensor,
    ) -> Tensor:
        excess = F.relu(nearest_distances - self.margin)

        return F.smooth_l1_loss(
            excess,
            torch.zeros_like(excess),
            beta=self.beta,
            reduction="mean",
        )

    def forward(
        self,
        means_fg: Tensor,
        return_stats: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        means_fg:
            [G_fg, 3]
            현재 frame에서 deformation이 적용된 foreground Gaussian 위치.
            detach하면 안 된다.
        """
        if means_fg.ndim != 2 or means_fg.shape[-1] != 3:
            raise ValueError(
                "means_fg must have shape [G_fg, 3], "
                f"got {tuple(means_fg.shape)}"
            )

        if not means_fg.is_floating_point():
            raise TypeError(
                "means_fg must be a floating-point tensor."
            )

        num_fg = means_fg.shape[0]
        pair_losses: List[Tensor] = []
        stats: Dict[str, Tensor] = {}

        for (
            cluster_a,
            cluster_b,
            name_a,
            name_b,
        ) in self._pair_info:
            idx_a = getattr(self, name_a)
            idx_b = getattr(self, name_b)

            if idx_a.max().item() >= num_fg:
                raise IndexError(
                    f"Pair {cluster_a}-{cluster_b}: "
                    f"patch A index exceeds num_fg={num_fg}."
                )

            if idx_b.max().item() >= num_fg:
                raise IndexError(
                    f"Pair {cluster_a}-{cluster_b}: "
                    f"patch B index exceeds num_fg={num_fg}."
                )

            points_a = means_fg.index_select(0, idx_a)
            points_b = means_fg.index_select(0, idx_b)

            points_a = self._subsample(points_a)
            points_b = self._subsample(points_b)

            # 현재 frame 위치에서 NN 대응을 매번 다시 계산한다.
            pairwise_dist = torch.cdist(
                points_a.float(),
                points_b.float(),
                p=2,
            )

            nearest_ab = pairwise_dist.min(dim=1).values
            nearest_ba = pairwise_dist.min(dim=0).values

            loss_ab = self._one_direction_loss(nearest_ab)
            loss_ba = self._one_direction_loss(nearest_ba)

            pair_loss = 0.5 * (loss_ab + loss_ba)
            pair_losses.append(pair_loss)

            if return_stats:
                pair_name = f"{cluster_a}-{cluster_b}"
                symmetric_nn = torch.cat(
                    [nearest_ab, nearest_ba],
                    dim=0,
                )

                stats[f"{pair_name}/loss"] = pair_loss.detach()
                stats[f"{pair_name}/nn_mean"] = (
                    symmetric_nn.mean().detach()
                )
                stats[f"{pair_name}/nn_median"] = (
                    symmetric_nn.median().detach()
                )
                stats[f"{pair_name}/nn_max"] = (
                    symmetric_nn.max().detach()
                )
                stats[f"{pair_name}/violating_ratio"] = (
                    (symmetric_nn > self.margin)
                    .float()
                    .mean()
                    .detach()
                )

        total_loss = torch.stack(pair_losses).mean()

        if return_stats:
            stats["total"] = total_loss.detach()
            return total_loss, stats

        return total_loss


def build_simple_contact_loss(
    patch_path: str | Path,
    margin: float,
    device: torch.device | str,
    beta: float = 0.002,
    max_points_per_patch: int | None = None,
) -> SimpleContactLoss:
    pairs = load_contact_patch_pairs(patch_path)

    module = SimpleContactLoss(
        patch_pairs=pairs,
        margin=margin,
        beta=beta,
        max_points_per_patch=max_points_per_patch,
    )

    return module.to(torch.device(device))
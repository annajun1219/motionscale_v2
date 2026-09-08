"""
flow3d/graph_relative_linear_attention_boundary.py

per-CLUSTER correction(flow3d/graph_relative_linear_attention.py)이 아니라
per-EDGE, per-SIDE boundary-patch correction을 내는 GNN 변형이다. 두 cluster가
맞닿은 boundary patch마다 서로 다른 (omega, delta_t)를 출력하므로, 같은
cluster가 여러 edge에 인접해도(예: cluster 19가 12/8/24와 각각 맞닿음) 그
edge마다 독립적인 correction을 가질 수 있다.

node encoder / relative edge feature / multi-head attention message passing
--------------------------------------------------------------------------
flow3d/graph_relative_linear_attention.py의 내부 구성요소(node encoder,
_RelativeVelLinearAttentionMessageLayer, 그 edge feature 계산식, dense-masked
multi-head attention)를 전부 그대로 import해서 재사용한다 -- 여기서 바뀌는
것은 message passing이 끝난 뒤의 "출력 head"뿐이다. 이 파일이 다른 GNN
variant 모듈에 의존하지 않는다는 원칙의 유일한 의도적 예외다: 사용자가
명시적으로 "기존 node encoder/relative edge feature/multi-head attention/
message passing 구조는 유지"를 요구했으므로, 그 내부 레이어 클래스
(_RelativeVelLinearAttentionMessageLayer 등, 밑줄로 시작하는 private 이름)를
직접 import한다.

message passing은 여전히 cluster graph 전체에서 이뤄진다 -- cluster 19의
hidden state는 19 자신의 움직임뿐 아니라 message passing을 통해 12/8/24 등
모든 이웃의 움직임을 반영한 contextual embedding이 된다 (기존과 동일).
바뀐 것은 그 다음이다: cluster-level 6D head 대신, message passing이 끝난
두 cluster의 hidden state(h_a, h_b) + 기존 relative edge feature +
boundary-patch feature를 결합하는 edge decoder가 undirected edge (a, b)마다
12차원(omega_a_from_b(3), delta_t_a_from_b(3), omega_b_from_a(3),
delta_t_b_from_a(3))을 출력한다. edge decoder의 마지막 Linear는 0으로
초기화되어 학습 시작 시 correction이 정확히 0이다 (zero-init 등가성).

edge topology의 단일 진실 공급원 (edges.pt)
--------------------------------------------
edge_id는 edges.pt(build_cluster_graph_mesh.py)의 "edges_kept" 리스트 순서로
직접 정의한다 -- 다른 GraphCorrectedScalableMotionBases variant들이 쓰는
build_edge_index_from_edges_pt는 내부적으로 정렬/재정렬을 할 수 있어(그
함수의 docstring 참고) 이 파일이 스스로 만드는 edge_id 순서와 어긋날 수
있으므로 사용하지 않는다. edge_cluster_a/edge_cluster_b(E,)를 edges_kept
순회로 직접 만들고, GNN 메시지패싱용 directed topology(edge_index_dir)도
바로 이 배열에서 파생시킨다 -- 그래서 "edge decoder가 보는 undirected edge
목록"과 "message passing이 쓰는 directed topology"가 항상 같은 edge_id
공간을 공유한다.

boundary_patch.pt는 GNN topology와 무관하게, 오직 "각 edge/side의 boundary
patch Gaussian이 누구고 weight가 얼마인지"만 공급한다. boundary_patch.pt에는
edges.pt의 edges_kept보다 더 많은(아직 tau_p/min_known_frames를 통과하지
못한 candidate) pair가 있을 수 있으므로, (cluster_a, cluster_b) unordered
pair로 edges_kept와 매칭해서 매칭되지 않는 pair는 버린다 (그 pair는 이
GNN의 message-passing 그래프에 아예 없는 edge이므로 correction을 만들 필요가
없다).

edge_cluster_a/edge_cluster_b 순회로부터 함께 읽는 값들
---------------------------------------------------------
edges.pt의 각 kept edge 항목에서 이 파일이 추가로 읽는 것:
  - connected_frame_indices / unknown_frame_indices -> per-frame CONNECTED=1/
    DISCONNECTED=0/UNKNOWN=보간 gate (correction_gate, connected_mask) --
    flow3d/graph_relative_edge.py's _build_edge_gate와 동일한 규칙을 이 파일
    안에 독립적으로 재구현한다 (그 함수를 직접 import하지 않는 이유는 모듈
    docstring 서두 참고 -- 여기서 import하는 것은 오직
    graph_relative_linear_attention.py의 attention 레이어뿐이다).
  - contact_reference_distance -> boundary gap loss의 목표 거리. 없으면 그
    edge는 reliable_edge_mask에서 제외된다(대체값을 쓰지 않음 -- gap loss가
    무의미한 목표로 학습을 왜곡하지 않도록).
  - persistence / num_known_frames -> reliable_edge_mask(에지 단위, 프레임과
    무관): gap loss는 CONNECTED라고 관찰된 모든 edge가 아니라, 이 값들이
    충분히 신뢰할 수 있는 edge에만 적용된다 (드물게만 관찰됐거나 지속성이
    낮은 edge의 노이즈 낀 거리 관측을 gap loss가 그대로 믿지 않도록).

canonical boundary reference (Stage 1, load_boundary_patch_reference) --
같은 (edge_id, side, global_index) 조합이 완전히 중복될 때만(정상적인
boundary_patch.pt에서는 사실상 일어나지 않는, 순수한 안전장치) weight를
최댓값으로 정리한다. 서로 다른 edge/side에 걸친 같은 Gaussian은 절대
합치거나 지우지 않는다 -- cluster 19의 19|12, 19|8, 19|24 patch가 각각
독립적인 row로 남는다.

Gaussian별 edge-side membership (Stage 2, assign_edge_memberships) --
"지금 존재하는" 각 Gaussian을 (cluster, edge, side)별로 그룹화된 canonical
reference cloud에 nearest-neighbor로 스냅한다(cutoff는 그 nearest point의
local_scale 기반, densify/cull에 무관하게 재계산 가능). 하나의 Gaussian이
여러 edge/side에 동시에 속할 수 있고, membership row는 (gaussian_idx,
edge_id, side, weight) 형태로 전부 보존된다(Gaussian당 스칼라 하나가 아님).
ASSIGNMENT(무거움, densify/cull에만 재계산: refresh_edge_membership)와
STATE(가벼움, 매 step: refresh_edge_state -- 각 row의 canonical mean/fine
coefs만 다시 gather, nearest-neighbor 재탐색 없음)로 분리한다 -- STATE가
필요한 이유는 patch pivot(centroid) 위치 자체를 매 step 계산해야 하기
때문이다(flow3d/graph_relative_edge.py의 falloff_canonical_mean/
falloff_coefs와 동일한 rationale).

compute_transforms: pivot = patch centroid, correction 결합 = 가중 tangent 평균
------------------------------------------------------------------------------
1. base_transforms = ScalableMotionBases.compute_transforms(...) -- correction
   없는 원본 coarse+fine transform.
2. omega_e, delta_t_e = self._compute_edge_correction(ts) -- (E, B, 2, 3)
   edge decoder raw 출력(clamp 적용됨). last_edge_correction에 저장(단, gap
   loss 계산용 detach_base=True 호출은 이 값을 절대 덮어쓰지 않는다 -- 아래
   "detach 분리" 참고).
3. 각 membership row m = (gaussian i, edge e, side s)마다:
     w_m = patch_weight_{i,m} * correction_gate_{e,t}   (공간 가중치 x 시간
       confidence -- DISCONNECTED면 정확히 0)
     omega_scaled_m = w_m * omega_{e,s}
     R_corr_m = Exp(omega_scaled_m)
     pivot_m = 그 edge/side patch의, correction 적용 *전* base transform으로
       이동한 현재 weighted centroid (cluster 전체 중심이 아니다)
     Δt_m = R_corr_m @ (t_base_i - pivot_m) + pivot_m + w_m*delta_t_{e,s}
            - t_base_i
   즉 이 row 하나만 적용됐다면 만들어졌을 translation 변화량이다 (mean을
   전혀 참조하지 않고 (R, t) 대수만으로 계산된다 -- baseline의 per-cluster
   correction이 coarse rotmat/transl에 직접 합성된 뒤 coarse-to-fine
   블렌딩된 것과 대수적으로 동일한 값이 나오는 것과 같은 유도).
4. Gaussian i에 닿는 모든 row를 결합한다(단순 합산도, w의 제곱으로 과도하게
   작아지지도 않는다):
     total_w_i = sum_m w_m
     a_i = max_m w_m
     combined_omega_i = a_i * (sum_m omega_scaled_m) / total_w_i
     combined_delta_t_i = a_i * (sum_m Δt_m) / total_w_i
   (omega_scaled_m/Δt_m는 이미 그 row 자신의 w_m이 적용된 값이므로, 위 식은
   "raw correction c_m의 weighted average"에 "가장 강하게 겹치는 membership의
   weight"를 곱하는 것과 같다: a_i * (sum_m w_m*c_m)/(sum_m w_m). membership이
   하나뿐이면 a_i == total_w_i == w_m이라 정확히 w_m*c_m으로 축소된다 -- 위
   3번 공식 그대로. 여러 membership이 겹쳐도 total_w_i로 나눈 평균이라 폭발적
   합산이 없고, a_i가 raw 값이 아니라 "가중치들의 최댓값"이라 w_m을 두 번
   곱하는 것(=w^2로 과도하게 작아짐)도 없다.) rotation은 여러 rotation
   matrix를 평균하지 않고 axis-angle tangent를 가중평균한 뒤 딱 한 번
   Exp()한다. translation은 서로 다른 pivot을 반영해 계산된 Δt_m을
   가중평균한다. total_w_i == 0(patch 밖 Gaussian)이면 둘 다 정확히 0 --
   baseline과 완전히 같다.
5. R_new = Exp(combined_omega_i) @ R_base_i, t_new = t_base_i +
   combined_delta_t_i.

compute_transforms_coarse는 여전히 오버라이드해서 원본(무보정) coarse-only
transform을 그대로 반환하되, GNN을 side effect로 1회 실행해
last_edge_correction을 이 ts 기준으로 갱신한다 (flow3d/trainer.py가
compute_transforms_coarse(ts) 호출 직후 correction을 읽는 기존 관례 유지).

detach 분리 (boundary gap loss)
--------------------------------
_compute_edge_correction(ts, detach_base=True)는 base coarse/fine motion과
patch pivot/거리/속도 feature 경로를 전부 detach한 뒤 GNN을 통과시켜, gap
loss의 gradient가 GNN(edge decoder + attention layer) 자신에게만 흐르고
base motion에는 흐르지 않게 한다. **이 detach 경로는 last_edge_correction을
절대 덮어쓰지 않는다** -- last_edge_correction은 오직 detach_base=False로
호출되는 주 경로(compute_transforms/compute_transforms_coarse)에서만
갱신되므로, gap loss 계산이 trainer의 다른 correction 관련 loss가 보는
"이번 배치의 실제 correction" 값을 오염시키지 않는다.

magnitude regularization은 이 raw (E, B, 2, 3) 값(patch weight 적용 *전*)에
직접 적용해야 한다 -- Gaussian별로 결합된 이후 값이 아니라, edge decoder가
실제로 예측한 크기 자체를 규제해야 patch weight가 작아 지금은 작아 보이는
correction이 나중에 커질 때 갑자기 튀는 것을 막는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from flow3d.graph_coupling import CENTER_DIM, ROT_DIM, TRANSL_DIM, compose_rotation, so3_exp_map
from flow3d.graph_relative_linear_attention import (
    EDGE_FEAT_DIM_VEL,
    NODE_FEAT_DIM_VEL,
    _build_directed_neighbor_edges,
    _compute_vel_scale,
    _RelativeVelLinearAttentionMessageLayer,
)
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "EdgePatchReference",
    "load_boundary_patch_reference",
    "assign_edge_memberships",
    "RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases",
    "compute_edge_patch_gap_loss",
]

# assign_edge_memberships: current-Gaussian <-> canonical reference snap
# cutoff, as a multiple of the nearest reference point's own local_scale
# (density-adaptive, mirrors flow3d/graph_relative_edge.py's
# _PATCH_SNAP_LOCAL_SCALE_RATIO).
_PATCH_SNAP_LOCAL_SCALE_RATIO = 2.0
_DEFAULT_SNAP_RADIUS = 0.05  # fallback cutoff when local_scale <= 0
_DEFAULT_MIN_WEIGHT = 1e-3
_DEFAULT_ASSIGN_CHUNK_SIZE = 8192  # see assign_edge_memberships docstring

_DEFAULT_MAX_OMEGA = 0.2  # radians, absolute rotation-correction cap
_DEFAULT_MAX_DISP_SCALE = 2.0  # multiplier on an edge's own local_scale median

EDGE_DECODER_OUT_DIM = 12  # omega_a(3) + delta_t_a(3) + omega_b(3) + delta_t_b(3)
# Edge-patch feature: pivot_a(3) + pivot_b(3) + rel_pos(3) + direction(3) +
# current_dist(1) + reference_dist(1) + dist_diff(1) + relative_velocity(3) +
# patch_size_a(1) + patch_size_b(1) + confidence(1) = 21
EDGE_PATCH_FEAT_DIM = 21


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_long(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


def _as_float(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.float32).reshape(-1)
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def _build_edge_gate(
    connected_frames: list[int] | None, unknown_frames: list[int] | None, num_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(connected_frame_indices, unknown_frame_indices) -> (correction_gate,
    connected_mask), both (num_frames,). Identical rule to
    flow3d/graph_relative_edge.py's _build_edge_gate (independently
    reimplemented here -- see module docstring): CONNECTED=1, DISCONNECTED=0,
    UNKNOWN interpolated only between observed knots; outside the observed
    range (or no knots at all) is conservatively 0."""
    if connected_frames is None or unknown_frames is None:
        return torch.ones(num_frames, dtype=torch.float32), torch.ones(num_frames, dtype=torch.bool)

    connected_set = {int(f) for f in connected_frames if 0 <= int(f) < num_frames}
    unknown_set = {int(f) for f in unknown_frames if 0 <= int(f) < num_frames}
    disconnected_set = set(range(num_frames)) - connected_set - unknown_set

    knot_frames = sorted(connected_set) + sorted(disconnected_set)
    knot_values = [1.0] * len(connected_set) + [0.0] * len(disconnected_set)
    if knot_frames:
        order = sorted(range(len(knot_frames)), key=lambda i: knot_frames[i])
        knot_frames_arr = [knot_frames[i] for i in order]
        knot_values_arr = [knot_values[i] for i in order]
        gate_np = np.interp(np.arange(num_frames), knot_frames_arr, knot_values_arr).astype(np.float32)
        all_frames = np.arange(num_frames)
        outside = (all_frames < knot_frames_arr[0]) | (all_frames > knot_frames_arr[-1])
        gate_np[outside] = 0.0
    else:
        gate_np = np.zeros(num_frames, dtype=np.float32)

    connected_mask_np = np.zeros(num_frames, dtype=bool)
    if connected_set:
        connected_mask_np[sorted(connected_set)] = True

    return torch.from_numpy(gate_np), torch.from_numpy(connected_mask_np)


def _load_edge_topology_from_edges_pt(
    edges_path: str | Path,
    num_frames: int,
    gap_loss_min_persistence: float,
    gap_loss_min_known_frames: int,
) -> dict[str, torch.Tensor]:
    """edges.pt(build_cluster_graph_mesh.py)의 "edges_kept" 리스트를 순회하며
    edge_id(=순회 순서)를 정의하고, GNN topology와 gap-loss 관련 정보를
    함께 만든다 (모듈 docstring의 "edge topology의 단일 진실 공급원" 참고).

    :return: dict with edge_cluster_a/edge_cluster_b (E,) long,
        correction_gate/connected_mask (E, num_frames), contact_reference_distance
        (E,) float (NaN이면 그 edge는 reference distance가 없다는 뜻),
        reliable_edge_mask (E,) bool.
    """
    payload = _torch_load(edges_path)
    if not isinstance(payload, dict) or "edges_kept" not in payload:
        raise TypeError(f"{edges_path} must be a build_cluster_graph.py-style edges.pt (a dict with 'edges_kept').")

    edges_kept = payload["edges_kept"]
    if not edges_kept:
        raise RuntimeError(f"{edges_path} has no edges_kept -- cannot build any edge topology.")

    cluster_a_rows, cluster_b_rows = [], []
    gate_rows, connected_mask_rows = [], []
    ref_dist_rows, reliable_rows = [], []

    for entry in edges_kept:
        if not isinstance(entry, dict) or "cluster_a" not in entry or "cluster_b" not in entry:
            raise KeyError(f"Malformed edges_kept entry in {edges_path}: {entry!r}")
        cluster_a_rows.append(int(entry["cluster_a"]))
        cluster_b_rows.append(int(entry["cluster_b"]))

        gate, connected_mask = _build_edge_gate(
            entry.get("connected_frame_indices"), entry.get("unknown_frame_indices"), num_frames
        )
        gate_rows.append(gate)
        connected_mask_rows.append(connected_mask)

        ref_dist = entry.get("contact_reference_distance")
        persistence = float(entry.get("persistence", 1.0))
        num_known = int(entry.get("num_known_frames", num_frames))
        is_reliable = (
            ref_dist is not None
            and persistence >= gap_loss_min_persistence
            and num_known >= gap_loss_min_known_frames
        )
        ref_dist_rows.append(float(ref_dist) if ref_dist is not None else float("nan"))
        reliable_rows.append(is_reliable)

    return {
        "edge_cluster_a": torch.tensor(cluster_a_rows, dtype=torch.long),
        "edge_cluster_b": torch.tensor(cluster_b_rows, dtype=torch.long),
        "correction_gate": torch.stack(gate_rows, dim=0),
        "connected_mask": torch.stack(connected_mask_rows, dim=0),
        "contact_reference_distance": torch.tensor(ref_dist_rows, dtype=torch.float32),
        "reliable_edge_mask": torch.tensor(reliable_rows, dtype=torch.bool),
    }


@dataclass
class EdgePatchReference:
    """boundary_patch.pt에서 읽은, edge_id별로 구분이 유지되는 canonical
    boundary reference cloud (모듈 docstring의 Stage 1). 같은 (edge_id, side,
    global_index)가 완전히 중복될 때만 weight가 최댓값으로 정리된다 -- 서로
    다른 edge/side에 걸친 같은 Gaussian은 절대 합쳐지지 않는다.

    :param ref_edge_id: (P,) long, edges.pt의 edges_kept 순회 순서로 정의된
        edge id.
    :param ref_side: (P,) long, 0 = edge_cluster_a 쪽, 1 = edge_cluster_b 쪽.
    :param ref_cluster_id: (P,) long, 그 point 자신이 속한 cluster id.
    :param points: (P, 3) 생성 시점 canonical 위치 스냅샷.
    :param weight: (P,) patch weight.
    :param local_scale: (P,) 로컬 Gaussian 밀도 스케일 (snap cutoff에 쓰임).
    """

    ref_edge_id: torch.Tensor
    ref_side: torch.Tensor
    ref_cluster_id: torch.Tensor
    points: torch.Tensor
    weight: torch.Tensor
    local_scale: torch.Tensor

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


def _dedup_rows_by_max_weight(
    composite_key: torch.Tensor, weight: torch.Tensor, *aux: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """composite_key가 같은 행이 여럿이면 weight가 가장 큰 행 하나만 남긴다
    (합산 금지). 두 번의 stable argsort로 벡터화 (weight 내림차순 정렬 ->
    그 순서를 유지한 채 composite_key로 재정렬하면 같은 key의 행들이 서로
    인접하면서 weight 내림차순을 유지 -- 각 그룹의 첫 행이 최댓값).
    :return: (weight_dedup, *aux_dedup), 같은 순서.
    """
    weight_order = torch.argsort(weight, descending=True, stable=True)
    key_by_weight = composite_key[weight_order]
    key_order = torch.argsort(key_by_weight, stable=True)
    sorted_keys = key_by_weight[key_order]
    final_order = weight_order[key_order]

    is_first = torch.ones_like(sorted_keys, dtype=torch.bool)
    if sorted_keys.numel() > 1:
        is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep_rows = final_order[is_first]

    return (weight[keep_rows], *(a[keep_rows] for a in aux))


def load_boundary_patch_reference(
    boundary_patch_path: str | Path,
    canonical_means: torch.Tensor,
    edge_cluster_a: torch.Tensor,
    edge_cluster_b: torch.Tensor,
    device: torch.device | None = None,
) -> EdgePatchReference:
    """build_cluster_graph_mesh.py의 boundary_patch.pt를 읽어 EdgePatchReference
    를 만든다. edge_id는 (edge_cluster_a, edge_cluster_b)로 이미 정의된
    edges.pt 기반 topology(모듈 docstring 참고)에 (cluster_a, cluster_b)
    unordered pair로 매칭한다 -- 매칭되지 않는 pair(아직 kept edge가 아닌
    candidate)는 버린다.

    :param canonical_means: (G, 3) 저장된 global index들이 가리키는, 이 함수를
        호출하는 시점의 canonical foreground Gaussian means.
    :param edge_cluster_a/edge_cluster_b: (E,) long, edges.pt 기반 edge topology.
    """
    if device is None:
        device = canonical_means.device

    payload = _torch_load(boundary_patch_path)
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise TypeError(
            f"{boundary_patch_path} must be a build_cluster_graph_mesh.py "
            "boundary_patch.pt (a dict with a 'pairs' list)."
        )

    means_cpu = canonical_means.detach().to("cpu")
    num_edges = edge_cluster_a.shape[0]
    edge_key_to_id = {
        frozenset((int(edge_cluster_a[e]), int(edge_cluster_b[e]))): e for e in range(num_edges)
    }

    edge_id_rows, side_rows, cluster_rows = [], [], []
    idx_rows, weight_rows, local_scale_rows = [], [], []

    for entry in payload["pairs"]:
        if not isinstance(entry, dict):
            raise TypeError(f"Malformed pairs entry in {boundary_patch_path}: {entry!r}")
        pair_a, pair_b = int(entry["cluster_a"]), int(entry["cluster_b"])
        e = edge_key_to_id.get(frozenset((pair_a, pair_b)))
        if e is None:
            continue  # candidate pair not in the kept edges.pt topology

        topo_a = int(edge_cluster_a[e])
        # side 0 always means "our topology's edge_cluster_a side" -- swap
        # the pair's a/b data if boundary_patch.pt happened to store them the
        # other way around relative to edges.pt's own (cluster_a, cluster_b).
        side_of_pair_a = 0 if pair_a == topo_a else 1
        side_of_pair_b = 1 - side_of_pair_a
        cluster_of_side = (int(edge_cluster_a[e]), int(edge_cluster_b[e]))

        idx_a = _as_long(entry["global_indices_a"])
        w_a = _as_float(entry["weight_a"])
        ls_a = _as_float(entry["local_scale_a"]) if "local_scale_a" in entry else torch.zeros_like(w_a)
        idx_b = _as_long(entry["global_indices_b"])
        w_b = _as_float(entry["weight_b"])
        ls_b = _as_float(entry["local_scale_b"]) if "local_scale_b" in entry else torch.zeros_like(w_b)

        if idx_a.numel() > 0:
            idx_rows.append(idx_a); weight_rows.append(w_a); local_scale_rows.append(ls_a)
            edge_id_rows.append(torch.full((idx_a.shape[0],), e, dtype=torch.long))
            side_rows.append(torch.full((idx_a.shape[0],), side_of_pair_a, dtype=torch.long))
            cluster_rows.append(torch.full((idx_a.shape[0],), cluster_of_side[side_of_pair_a], dtype=torch.long))
        if idx_b.numel() > 0:
            idx_rows.append(idx_b); weight_rows.append(w_b); local_scale_rows.append(ls_b)
            edge_id_rows.append(torch.full((idx_b.shape[0],), e, dtype=torch.long))
            side_rows.append(torch.full((idx_b.shape[0],), side_of_pair_b, dtype=torch.long))
            cluster_rows.append(torch.full((idx_b.shape[0],), cluster_of_side[side_of_pair_b], dtype=torch.long))

    if not idx_rows:
        raise RuntimeError(
            f"No pair in {boundary_patch_path} matched a kept edge from the edges.pt topology."
        )

    global_idx = torch.cat(idx_rows)
    weight = torch.cat(weight_rows)
    local_scale = torch.cat(local_scale_rows)
    edge_id = torch.cat(edge_id_rows)
    side = torch.cat(side_rows)
    cluster_id = torch.cat(cluster_rows)

    # Dedup only within the exact same (edge_id, side, global_idx) -- see
    # module docstring. BIG safely exceeds any realistic Gaussian count.
    BIG = 1 << 32
    composite_key = (edge_id * 2 + side) * BIG + global_idx
    weight, local_scale, edge_id, side, cluster_id, global_idx = _dedup_rows_by_max_weight(
        composite_key, weight, local_scale, edge_id, side, cluster_id, global_idx
    )

    return EdgePatchReference(
        ref_edge_id=edge_id.to(device),
        ref_side=side.to(device),
        ref_cluster_id=cluster_id.to(device),
        points=means_cpu[global_idx].float().to(device),
        weight=weight.to(device),
        local_scale=local_scale.to(device),
    )


@torch.no_grad()
def assign_edge_memberships(
    canonical_means: torch.Tensor,
    cluster_ids_all: torch.Tensor,
    patch_ref: EdgePatchReference,
    snap_radius: float = _DEFAULT_SNAP_RADIUS,
    min_weight: float = _DEFAULT_MIN_WEIGHT,
    chunk_size: int = _DEFAULT_ASSIGN_CHUNK_SIZE,
) -> dict[str, torch.Tensor]:
    """"지금 존재하는" 각 Gaussian이 어느 edge의 어느 side patch에 속하는지,
    ragged row 형태로 만든다 (모듈 docstring의 Stage 2) -- Gaussian 하나가
    여러 edge/side에 동시에 속할 수 있으므로 스칼라 weight 하나가 아니라
    row 목록을 반환한다.

    (edge_id, side)별로 그룹화해서, 같은 그룹에 속한 reference point 중
    **가장 가까운 하나**의 weight를 가져온다(합산도, cutoff 안 최댓값도
    아니다 -- nearest-neighbor). cutoff는 그 nearest point의 local_scale의
    _PATCH_SNAP_LOCAL_SCALE_RATIO배(local_scale <= 0이면 snap_radius로
    대체). Gaussian이 수십만 개까지 늘어날 수 있어 현재-Gaussian 축은
    chunk_size 단위로 나눠 cdist를 호출한다(결과는 완전히 동일, 메모리만 줄어듦).

    :return: dict with row_gaussian_idx/row_edge_id/row_side/row_cluster_id
        (M,) long, row_weight (M,) float. M개 행 각각이 하나의
        (Gaussian, edge, side) membership이다.
    """
    device = canonical_means.device
    cluster_ids_flat = cluster_ids_all.reshape(-1)

    gaussian_rows, edge_rows, side_rows, cluster_rows, weight_rows = [], [], [], [], []

    if patch_ref.num_points > 0:
        ref_points = patch_ref.points.to(device)
        ref_weight = patch_ref.weight.to(device)
        ref_local_scale = patch_ref.local_scale.to(device)
        ref_edge_id = patch_ref.ref_edge_id.to(device)
        ref_side = patch_ref.ref_side.to(device)
        ref_cluster_id = patch_ref.ref_cluster_id.to(device)

        group_keys = torch.unique(torch.stack([ref_edge_id, ref_side], dim=1), dim=0)
        for e_t, s_t in group_keys.tolist():
            ref_mask = (ref_edge_id == e_t) & (ref_side == s_t)
            cid = int(ref_cluster_id[ref_mask][0].item())
            cur_idx = (cluster_ids_flat == cid).nonzero(as_tuple=True)[0]
            if cur_idx.numel() == 0:
                continue

            r_points = ref_points[ref_mask]
            r_weight = ref_weight[ref_mask]
            r_local_scale = ref_local_scale[ref_mask]
            r_cutoff = torch.where(
                r_local_scale > 0,
                _PATCH_SNAP_LOCAL_SCALE_RATIO * r_local_scale,
                torch.full_like(r_local_scale, snap_radius),
            )

            for chunk_start in range(0, cur_idx.shape[0], chunk_size):
                idx_chunk = cur_idx[chunk_start : chunk_start + chunk_size]
                dist = torch.cdist(canonical_means[idx_chunk], r_points)  # (n_chunk, n_ref)
                nearest_dist, nearest_pos = dist.min(dim=1)
                weight = r_weight[nearest_pos]
                cutoff = r_cutoff[nearest_pos]
                keep = (nearest_dist <= cutoff) & (weight > min_weight)
                if bool(keep.any()):
                    kept_idx = idx_chunk[keep]
                    n_keep = kept_idx.shape[0]
                    gaussian_rows.append(kept_idx)
                    edge_rows.append(torch.full((n_keep,), e_t, dtype=torch.long, device=device))
                    side_rows.append(torch.full((n_keep,), s_t, dtype=torch.long, device=device))
                    cluster_rows.append(torch.full((n_keep,), cid, dtype=torch.long, device=device))
                    weight_rows.append(weight[keep])

    if not gaussian_rows:
        return {
            "row_gaussian_idx": torch.empty(0, dtype=torch.long, device=device),
            "row_edge_id": torch.empty(0, dtype=torch.long, device=device),
            "row_side": torch.empty(0, dtype=torch.long, device=device),
            "row_cluster_id": torch.empty(0, dtype=torch.long, device=device),
            "row_weight": torch.empty(0, dtype=torch.float32, device=device),
        }

    return {
        "row_gaussian_idx": torch.cat(gaussian_rows),
        "row_edge_id": torch.cat(edge_rows),
        "row_side": torch.cat(side_rows),
        "row_cluster_id": torch.cat(cluster_rows),
        "row_weight": torch.cat(weight_rows),
    }


def _clamp_vector_magnitude(vec: torch.Tensor, max_mag: torch.Tensor | float) -> torch.Tensor:
    """direction-preserving smooth clamp: |vec| -> max_mag*tanh(|vec|/max_mag),
    identity-like for small |vec| (zero-init equivalence: vec==0 -> exactly
    0), saturating smoothly toward max_mag for large |vec| (always
    differentiable, unlike a hard clamp).

    :param vec: (..., 3).
    :param max_mag: scalar or broadcastable to vec's leading dims (unsqueezed
        to align with vec's last dim internally by the caller).
    """
    norm = vec.norm(dim=-1, keepdim=True)
    norm_safe = norm.clamp_min(1e-8)
    max_mag_safe = max_mag if isinstance(max_mag, float) else max_mag.clamp_min(1e-8)
    scale = max_mag_safe * torch.tanh(norm_safe / max_mag_safe) / norm_safe
    return vec * scale


def _apply_transform(transforms: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """transforms (M, B, 3, 4), means (M, 3) -> posed positions (M, B, 3)."""
    R = transforms[..., :3]
    t = transforms[..., 3]
    return torch.einsum("mbij,mj->mbi", R, means) + t


def _compute_weight_sum_by_side(
    row_edge_id: torch.Tensor, row_side: torch.Tensor, row_weight: torch.Tensor, num_edges: int
) -> torch.Tensor:
    """(E, 2) sum of membership weight per (edge, side) -- side 0 = a, side 1
    = b. Pure ASSIGNMENT-level index_add_ (no ts/position dependence), so it
    is reused both by _edge_side_pivots's patch-size feature and by
    from_scalable_motion_bases's one-time patch-size normalization stats
    (see RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases's
    patch_size_mean/patch_size_std)."""
    weight_sums = row_weight.new_zeros(num_edges * 2)
    if row_edge_id.numel() > 0:
        group_id = row_edge_id * 2 + row_side
        weight_sums.index_add_(0, group_id, row_weight)
    return torch.stack([weight_sums[0::2], weight_sums[1::2]], dim=-1)  # (E, 2)


class _EdgePatchDecoderGNN(nn.Module):
    """flow3d/graph_relative_linear_attention.py's node encoder + relative
    edge feature + multi-head attention message-passing layer, unchanged (see
    module docstring), followed by an edge decoder instead of a per-cluster
    6D head. Message passing still runs over the FULL cluster graph -- a
    cluster's hidden state reflects every neighbor's motion, exactly as in
    the per-cluster baseline. Only the final readout changed: undirected edge
    (a, b) -> 12-dim (omega_a_from_b, delta_t_a_from_b, omega_b_from_a,
    delta_t_b_from_a).
    """

    def __init__(
        self,
        edge_cluster_a: torch.Tensor,
        edge_cluster_b: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vel_scale: float = 1.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        undirected_edge_index = torch.stack([edge_cluster_a, edge_cluster_b], dim=0)  # (2, E)
        self.register_buffer("undirected_edge_index", undirected_edge_index)
        edge_index_dir = _build_directed_neighbor_edges(undirected_edge_index, num_clusters)
        self.register_buffer("edge_index_dir", edge_index_dir)

        if not math.isfinite(vel_scale) or vel_scale <= 0.0:
            vel_scale = 1.0
        self.register_buffer("vel_scale", torch.tensor(float(vel_scale)))

        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM_VEL, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [_RelativeVelLinearAttentionMessageLayer(hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        # LayerNorm over the FULL decoder input -- some components (e.g. the
        # patch-size feature, already separately z-score normalized by the
        # outer class) can still differ in scale from the node-embedding/
        # edge-feature components; this keeps the first Linear's pre-
        # activation from being dominated by any one feature regardless.
        # LeakyReLU(0.1) instead of ReLU: a fully-negative pre-activation
        # (observed in practice once a large-magnitude, always-positive
        # feature -- log1p(patch weight sum) -- dominated the random-init
        # first layer) makes plain ReLU's gradient EXACTLY 0 everywhere,
        # permanently killing gradient to edge_decoder[0] and everything
        # upstream (encoder, attention layers). LeakyReLU still passes a
        # 0.1x gradient through in that regime, so the network can recover
        # instead of dying forever.
        decoder_in_dim = 2 * hidden_dim + EDGE_FEAT_DIM_VEL + EDGE_PATCH_FEAT_DIM
        self.edge_decoder = nn.Sequential(
            nn.LayerNorm(decoder_in_dim),
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, EDGE_DECODER_OUT_DIM),
        )
        nn.init.kaiming_uniform_(self.edge_decoder[1].weight, a=0.1, nonlinearity="leaky_relu")
        nn.init.zeros_(self.edge_decoder[1].bias)
        # Only the LAST Linear stays zero-init: correction must still be
        # exactly 0 at step 0 (zero-init equivalence with plain MotionScale),
        # regardless of how the hidden layer is initialized/activated.
        nn.init.zeros_(self.edge_decoder[-1].weight)
        nn.init.zeros_(self.edge_decoder[-1].bias)

    def _compute_directed_edge_features(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        vel_scaled: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """e_{dst<-src} = [t_src - t_dst, c_src - c_dst, rot6d(R_dst^T @ R_src),
        v_src - v_dst], identical formula to the baseline's per-directed-edge
        feature (graph_relative_linear_attention.py's _compute_edge_features),
        evaluated for an arbitrary (src, dst) pair list."""
        B = coarse_transl.shape[1]
        if src.numel() == 0:
            return coarse_rot_6d.new_zeros(0, B, EDGE_FEAT_DIM_VEL)

        rel_transl = coarse_transl[src] - coarse_transl[dst]
        rel_center = (centers[src] - centers[dst])[:, None, :].expand(-1, B, -1)

        coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)
        R_dst = coarse_rotmats[dst]
        R_src = coarse_rotmats[src]
        rel_rot6d = rmat_to_cont_6d(R_dst.transpose(-1, -2) @ R_src)

        rel_vel = vel_scaled[src] - vel_scaled[dst]

        return torch.cat([rel_transl, rel_center, rel_rot6d, rel_vel], dim=-1)

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        coarse_vel: torch.Tensor,
        edge_patch_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        coarse_rot_6d/coarse_transl/centers/coarse_vel: same as the baseline
        per-cluster GNN (C, B, *).
        edge_patch_feat: (E, B, EDGE_PATCH_FEAT_DIM), row-aligned with
            self.undirected_edge_index's columns.
        returns: omega (E, B, 2, 3), delta_t (E, B, 2, 3) -- side 0 = "a"
            (undirected_edge_index[0]), side 1 = "b" (undirected_edge_index[1]).
        """
        C, B, _ = coarse_rot_6d.shape
        center_feat = centers[:, None, :].expand(C, B, CENTER_DIM)
        vel_scaled = coarse_vel / self.vel_scale
        node_feat = torch.cat([coarse_rot_6d, coarse_transl, center_feat, vel_scaled], dim=-1)

        h = self.encoder(node_feat)
        edge_feat_dir = self._compute_directed_edge_features(
            coarse_rot_6d, coarse_transl, centers, vel_scaled,
            self.edge_index_dir[0], self.edge_index_dir[1],
        )
        for layer in self.layers:
            h = layer(h, self.edge_index_dir, edge_feat_dir)

        a, b = self.undirected_edge_index[0], self.undirected_edge_index[1]
        h_a, h_b = h[a], h[b]
        edge_feat_ab = self._compute_directed_edge_features(
            coarse_rot_6d, coarse_transl, centers, vel_scaled, b, a
        )  # dst=a, src=b -- an arbitrary but fixed convention for the undirected edge feature

        decoder_in = torch.cat([h_a, h_b, edge_feat_ab, edge_patch_feat], dim=-1)
        raw = self.edge_decoder(decoder_in)  # (E, B, 12)
        omega_a, delta_t_a, omega_b, delta_t_b = raw.split([3, 3, 3, 3], dim=-1)
        omega = torch.stack([omega_a, omega_b], dim=2)  # (E, B, 2, 3)
        delta_t = torch.stack([delta_t_a, delta_t_b], dim=2)  # (E, B, 2, 3)
        return omega, delta_t


class RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + per-EDGE, per-SIDE boundary-patch correction.
    GNN message passing (node encoder/relative edge feature/multi-head
    attention) is identical to flow3d/graph_relative_linear_attention.py; the
    difference is the readout (edge decoder instead of a per-cluster head)
    and how the correction is applied (per membership row, pivoted at the
    patch centroid, combined via a weighted tangent-space average -- see
    module docstring).

    subset-invariant하지 않다 (다른 boundary-style variant와 동일한 계약):
    membership이 Gaussian identity에 매이므로 compute_transforms는 cluster_ids
    /coefs가 정확히 전체 canonical foreground 배열일 때만 global_indices
    생략을 허용한다.

    densify/cull로 foreground Gaussian 개수/순서가 바뀌면
    refresh_edge_membership(canonical_means, cluster_ids_all, coefs_all)를
    호출해야 한다(무거움, nearest-neighbor 재탐색). 그 외 매 학습 step에는
    더 가벼운 refresh_edge_state(canonical_means, coefs_all)만 불러 patch
    pivot 계산에 쓰이는 row별 위치 스냅샷만 갱신하면 된다.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        rots: torch.Tensor,
        transls: torch.Tensor,
        fine_rots: torch.Tensor,
        fine_transls: torch.Tensor,
        edge_cluster_a: torch.Tensor,
        edge_cluster_b: torch.Tensor,
        correction_gate: torch.Tensor,
        connected_mask: torch.Tensor,
        contact_reference_distance: torch.Tensor,
        reliable_edge_mask: torch.Tensor,
        max_displacement: torch.Tensor,
        patch_ref_points: torch.Tensor,
        patch_ref_weight: torch.Tensor,
        patch_ref_local_scale: torch.Tensor,
        patch_ref_edge_id: torch.Tensor,
        patch_ref_side: torch.Tensor,
        patch_ref_cluster_id: torch.Tensor,
        row_gaussian_idx: torch.Tensor,
        row_edge_id: torch.Tensor,
        row_side: torch.Tensor,
        row_cluster_id: torch.Tensor,
        row_weight: torch.Tensor,
        row_canonical_mean: torch.Tensor,
        row_coefs: torch.Tensor,
        patch_size_mean: torch.Tensor,
        patch_size_std: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        snap_radius: float = _DEFAULT_SNAP_RADIUS,
        max_omega: float = _DEFAULT_MAX_OMEGA,
    ):
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        vel_scale = _compute_vel_scale(transls)
        self.gnn = _EdgePatchDecoderGNN(
            edge_cluster_a=edge_cluster_a,
            edge_cluster_b=edge_cluster_b,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            vel_scale=vel_scale,
        )

        # Fixed ("edge topology + gap-loss info"): from edges.pt, never
        # touched by refresh_edge_membership/refresh_edge_state.
        self.register_buffer("edge_cluster_a", edge_cluster_a.clone().long())
        self.register_buffer("edge_cluster_b", edge_cluster_b.clone().long())
        self.register_buffer("correction_gate", correction_gate.clone().float())
        self.register_buffer("connected_mask", connected_mask.clone().bool())
        self.register_buffer("contact_reference_distance", contact_reference_distance.clone().float())
        self.register_buffer("reliable_edge_mask", reliable_edge_mask.clone().bool())
        self.register_buffer("max_displacement", max_displacement.clone().float())
        self.register_buffer("snap_radius", torch.tensor(float(snap_radius)))
        self.register_buffer("max_omega", torch.tensor(float(max_omega)))
        # Fixed patch-size (log1p(weight_sum_by_side)) z-score normalization
        # stats, computed ONCE over the whole training edge set (see
        # from_scalable_motion_bases) and reused unchanged thereafter --
        # including after checkpoint restore (never recomputed by
        # refresh_edge_membership/refresh_edge_state) -- so the decoder
        # always sees the same normalization it was trained with.
        self.register_buffer("patch_size_mean", patch_size_mean.clone().float())
        self.register_buffer("patch_size_std", patch_size_std.clone().float())

        # Fixed ("boundary의 의미", canonical reference cloud): from
        # boundary_patch.pt, never touched by refresh_*.
        self.register_buffer("patch_ref_points", patch_ref_points.clone().float())
        self.register_buffer("patch_ref_weight", patch_ref_weight.clone().float())
        self.register_buffer("patch_ref_local_scale", patch_ref_local_scale.clone().float())
        self.register_buffer("patch_ref_edge_id", patch_ref_edge_id.clone().long())
        self.register_buffer("patch_ref_side", patch_ref_side.clone().long())
        self.register_buffer("patch_ref_cluster_id", patch_ref_cluster_id.clone().long())

        # Live ASSIGNMENT ("지금 어떤 Gaussian이 어떤 edge/side에 속하는지"):
        # refresh_edge_membership이 densify/cull 때만 통째로 재계산.
        self.register_buffer("row_gaussian_idx", row_gaussian_idx.clone().long())
        self.register_buffer("row_edge_id", row_edge_id.clone().long())
        self.register_buffer("row_side", row_side.clone().long())
        self.register_buffer("row_cluster_id", row_cluster_id.clone().long())
        self.register_buffer("row_weight", row_weight.clone().float())
        # Placeholder -- from_scalable_motion_bases/init_from_state_dict set
        # this to the real value right after construction via
        # _with_num_fg_gaussians (num_fg_gaussians isn't derivable from any
        # __init__ arg alone: row_canonical_mean's length is M, the number of
        # membership rows, not G, the number of live Gaussians).
        self.register_buffer("num_fg_gaussians", torch.tensor(0, dtype=torch.long))

        # Live STATE (row별 실제 위치 스냅샷, refresh_edge_state가 매 step
        # 재조회 -- nearest-neighbor 재탐색 없음).
        self.register_buffer("row_canonical_mean", row_canonical_mean.clone().float())
        self.register_buffer("row_coefs", row_coefs.clone().float())

        self._last_edge_correction: dict[str, torch.Tensor] | None = None

    @property
    def num_edges(self) -> int:
        return int(self.edge_cluster_a.shape[0])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        edges_path: str | Path,
        boundary_patch_path: str | Path,
        canonical_means: torch.Tensor,
        cluster_ids_all: torch.Tensor,
        coefs_all: torch.Tensor,
        total_num_frames: int,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        snap_radius: float = _DEFAULT_SNAP_RADIUS,
        max_omega: float = _DEFAULT_MAX_OMEGA,
        max_disp_scale: float = _DEFAULT_MAX_DISP_SCALE,
        gap_loss_min_persistence: float = 0.5,
        gap_loss_min_known_frames: int = 1,
    ) -> "RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases":
        """:param edge_index: 다른 *GraphCorrectedScalableMotionBases와
            call-site를 맞추기 위해 받지만 사용하지 않는다(EdgeBoundaryGraph
            CorrectedScalableMotionBases와 동일한 패턴) -- edge topology는
            edges_path에서 이 클래스가 직접, 자신만의 순서로 다시 읽는다
            (모듈 docstring 참고).
        :param edges_path: build_cluster_graph_mesh.py's edges.pt 경로 (GNN
            topology + per-edge gate/reference distance/persistence).
        :param boundary_patch_path: 같은 빌드의 boundary_patch.pt 경로 (patch
            Gaussian/weight/local_scale).
        :param total_num_frames: correction_gate/connected_mask의 시간축
            크기 -- 데이터셋 전체 프레임 수여야 한다(bases.num_frames가 아님).
            bases.num_frames는 attach 시점의 --num_init_frames일 뿐이라, 이후
            incremental propagation으로 프레임이 늘어나면 그보다 큰 ts가
            correction_gate를 인덱싱하게 된다. edges.pt의 각 edge는 이미
            connected_frame_indices/unknown_frame_indices로 전체 타임라인을
            커버하므로 여기서 num_frames만 전체 길이로 주면, 나중에
            rots/fine_rots가 자라도 gate가 out-of-bounds가 되지 않는다.
            rots/transls/fine_rots/fine_transls 등 coarse/fine motion
            파라미터 자체의 현재 프레임 수는 이 값과 무관하게 bases.num_frames를
            그대로 따른다(아래에서 bases.params를 그대로 clone).
        """
        del edge_index
        topo = _load_edge_topology_from_edges_pt(
            edges_path, total_num_frames, gap_loss_min_persistence, gap_loss_min_known_frames
        )
        assert topo["correction_gate"].shape[1] == total_num_frames, (
            f"correction_gate time axis ({topo['correction_gate'].shape[1]}) != "
            f"total_num_frames ({total_num_frames})"
        )
        assert topo["connected_mask"].shape[1] == total_num_frames, (
            f"connected_mask time axis ({topo['connected_mask'].shape[1]}) != "
            f"total_num_frames ({total_num_frames})"
        )
        patch_ref = load_boundary_patch_reference(
            boundary_patch_path, canonical_means, topo["edge_cluster_a"], topo["edge_cluster_b"]
        )
        membership = assign_edge_memberships(canonical_means, cluster_ids_all, patch_ref, snap_radius=snap_radius)

        max_displacement = _compute_edge_max_displacement(
            patch_ref, topo["edge_cluster_a"].shape[0], snap_radius, max_disp_scale
        )

        # One-time patch-size normalization stats over the whole training
        # edge set (see __init__'s patch_size_mean/patch_size_std docstring).
        weight_sum_by_side_init = _compute_weight_sum_by_side(
            membership["row_edge_id"], membership["row_side"], membership["row_weight"],
            topo["edge_cluster_a"].shape[0],
        )
        patch_size_init = torch.log1p(weight_sum_by_side_init)
        patch_size_mean = patch_size_init.mean()
        patch_size_std = patch_size_init.std()

        p = bases.params
        return cls(
            centers=p["centers"].detach().clone(),
            rots=p["rots"].detach().clone(),
            transls=p["transls"].detach().clone(),
            fine_rots=p["fine_rots"].detach().clone(),
            fine_transls=p["fine_transls"].detach().clone(),
            edge_cluster_a=topo["edge_cluster_a"],
            edge_cluster_b=topo["edge_cluster_b"],
            correction_gate=topo["correction_gate"],
            connected_mask=topo["connected_mask"],
            contact_reference_distance=topo["contact_reference_distance"],
            reliable_edge_mask=topo["reliable_edge_mask"],
            max_displacement=max_displacement,
            patch_ref_points=patch_ref.points,
            patch_ref_weight=patch_ref.weight,
            patch_ref_local_scale=patch_ref.local_scale,
            patch_ref_edge_id=patch_ref.ref_edge_id,
            patch_ref_side=patch_ref.ref_side,
            patch_ref_cluster_id=patch_ref.ref_cluster_id,
            row_gaussian_idx=membership["row_gaussian_idx"],
            row_edge_id=membership["row_edge_id"],
            row_side=membership["row_side"],
            row_cluster_id=membership["row_cluster_id"],
            row_weight=membership["row_weight"],
            row_canonical_mean=canonical_means[membership["row_gaussian_idx"]],
            row_coefs=coefs_all[membership["row_gaussian_idx"]],
            patch_size_mean=patch_size_mean,
            patch_size_std=patch_size_std,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_num_heads=gnn_num_heads,
            snap_radius=snap_radius,
            max_omega=max_omega,
        )._with_num_fg_gaussians(canonical_means.shape[0])

    def _with_num_fg_gaussians(self, n: int) -> "RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases":
        self.num_fg_gaussians = torch.tensor(int(n), dtype=torch.long, device=self.num_fg_gaussians.device)
        return self

    @staticmethod
    def has_gnn_state(state_dict: dict, prefix: str) -> bool:
        """이 클래스에만 있는 top-level "row_weight" buffer를 마커로 쓴다--
        EdgeBoundaryGraphCorrectedScalableMotionBases도 "patch_ref_points"라는
        이름의 buffer를 갖고 있어 그 이름은 마커로 쓸 수 없다."""
        return f"{prefix}row_weight" in state_dict

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases":
        """체크포인트만으로 통째로 복원한다. edge topology/gap-loss info/
        canonical boundary reference/live membership+state가 모두 buffer로
        저장돼 있으므로 edges_path/boundary_patch_path 없이도 정확히
        복원된다 (last_edge_correction만 예외 -- 매 forward에서 새로 채워지는
        진단용 값이라 buffer로 저장하지 않는다)."""
        base = ScalableMotionBases.init_from_state_dict(state_dict, prefix=f"{prefix}params.")

        gnn_prefix = f"{prefix}gnn."
        gnn_keys = [key for key in state_dict if key.startswith(gnn_prefix)]
        if not gnn_keys:
            raise KeyError(f"No '{gnn_prefix}*' keys found in state_dict.")

        required_keys = (
            f"{prefix}edge_cluster_a", f"{prefix}edge_cluster_b",
            f"{prefix}correction_gate", f"{prefix}connected_mask",
            f"{prefix}contact_reference_distance", f"{prefix}reliable_edge_mask",
            f"{prefix}max_displacement",
            f"{prefix}patch_ref_points", f"{prefix}patch_ref_weight",
            f"{prefix}patch_ref_local_scale", f"{prefix}patch_ref_edge_id",
            f"{prefix}patch_ref_side", f"{prefix}patch_ref_cluster_id",
            f"{prefix}row_gaussian_idx", f"{prefix}row_edge_id", f"{prefix}row_side",
            f"{prefix}row_cluster_id", f"{prefix}row_weight",
            f"{prefix}row_canonical_mean", f"{prefix}row_coefs",
            f"{prefix}snap_radius", f"{prefix}max_omega",
            f"{prefix}patch_size_mean", f"{prefix}patch_size_std",
        )
        missing = [key for key in required_keys if key not in state_dict]
        if missing:
            raise KeyError(f"state_dict is missing {missing} -- not an edge-patch boundary checkpoint.")

        hidden_dim = state_dict[f"{gnn_prefix}encoder.0.weight"].shape[0]
        layers_prefix = f"{gnn_prefix}layers."
        layer_indices = {
            int(key[len(layers_prefix):].split(".", 1)[0])
            for key in gnn_keys
            if key.startswith(layers_prefix)
        }
        num_layers = max(layer_indices) + 1
        num_heads = state_dict[f"{layers_prefix}0.attn"].shape[0]

        graph_bases = cls(
            centers=base.params["centers"].detach().clone(),
            rots=base.params["rots"].detach().clone(),
            transls=base.params["transls"].detach().clone(),
            fine_rots=base.params["fine_rots"].detach().clone(),
            fine_transls=base.params["fine_transls"].detach().clone(),
            edge_cluster_a=state_dict[f"{prefix}edge_cluster_a"],
            edge_cluster_b=state_dict[f"{prefix}edge_cluster_b"],
            correction_gate=state_dict[f"{prefix}correction_gate"],
            connected_mask=state_dict[f"{prefix}connected_mask"],
            contact_reference_distance=state_dict[f"{prefix}contact_reference_distance"],
            reliable_edge_mask=state_dict[f"{prefix}reliable_edge_mask"],
            max_displacement=state_dict[f"{prefix}max_displacement"],
            patch_ref_points=state_dict[f"{prefix}patch_ref_points"],
            patch_ref_weight=state_dict[f"{prefix}patch_ref_weight"],
            patch_ref_local_scale=state_dict[f"{prefix}patch_ref_local_scale"],
            patch_ref_edge_id=state_dict[f"{prefix}patch_ref_edge_id"],
            patch_ref_side=state_dict[f"{prefix}patch_ref_side"],
            patch_ref_cluster_id=state_dict[f"{prefix}patch_ref_cluster_id"],
            row_gaussian_idx=state_dict[f"{prefix}row_gaussian_idx"],
            row_edge_id=state_dict[f"{prefix}row_edge_id"],
            row_side=state_dict[f"{prefix}row_side"],
            row_cluster_id=state_dict[f"{prefix}row_cluster_id"],
            row_weight=state_dict[f"{prefix}row_weight"],
            row_canonical_mean=state_dict[f"{prefix}row_canonical_mean"],
            row_coefs=state_dict[f"{prefix}row_coefs"],
            patch_size_mean=state_dict[f"{prefix}patch_size_mean"],
            patch_size_std=state_dict[f"{prefix}patch_size_std"],
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
            gnn_num_heads=num_heads,
            snap_radius=float(state_dict[f"{prefix}snap_radius"].item()),
            max_omega=float(state_dict[f"{prefix}max_omega"].item()),
        )._with_num_fg_gaussians(int(state_dict.get(f"{prefix}num_fg_gaussians", torch.tensor(0)).item()))

        gnn_state = {
            key[len(gnn_prefix):]: value for key, value in state_dict.items() if key.startswith(gnn_prefix)
        }
        graph_bases.gnn.load_state_dict(gnn_state, strict=True)

        return graph_bases

    @staticmethod
    def regenerate_stale_gate_state_dict(
        state_dict: dict[str, torch.Tensor],
        edges_path: str | Path,
        total_num_frames: int,
        gap_loss_min_persistence: float,
        gap_loss_min_known_frames: int,
        prefix: str = "motion_bases.",
    ) -> dict[str, torch.Tensor] | None:
        """오래된 체크포인트 resume 지원: 이 클래스가 attach될 당시
        bases.num_frames(=--num_init_frames)로 correction_gate/connected_mask
        의 시간축을 잘라서 저장한 체크포인트는, 이후 incremental frame
        propagation으로 rots/fine_rots가 그보다 더 많은 프레임으로 자라면
        correction_gate[:, ts]가 out-of-bounds로 CUDA device-side assert를
        낸다 (ts가 attach 당시의 num_frames를 넘어서므로).

        edges.pt의 각 kept edge는 애초에 전체 타임라인의
        connected_frame_indices/unknown_frame_indices를 갖고 있으므로,
        correction_gate/connected_mask를 edges_path에서 total_num_frames
        길이로 다시 만들어서 state_dict 안의 두 buffer만 교체하면 된다 (edge
        topology/patch reference/GNN weight/optimizer state 등 나머지는 전혀
        건드리지 않는다).

        :return: 교체된 {"{prefix}correction_gate": ..., "{prefix}connected_mask": ...}
            두 키만 담은 dict, 또는 이미 total_num_frames와 일치해서 손댈
            필요가 없으면 None.
        """
        gate_key = f"{prefix}correction_gate"
        mask_key = f"{prefix}connected_mask"
        if gate_key not in state_dict:
            return None
        if int(state_dict[gate_key].shape[1]) == total_num_frames:
            return None

        topo = _load_edge_topology_from_edges_pt(
            edges_path, total_num_frames, gap_loss_min_persistence, gap_loss_min_known_frames
        )
        assert topo["correction_gate"].shape[1] == total_num_frames, (
            f"correction_gate time axis ({topo['correction_gate'].shape[1]}) != "
            f"total_num_frames ({total_num_frames})"
        )
        assert topo["connected_mask"].shape[1] == total_num_frames, (
            f"connected_mask time axis ({topo['connected_mask'].shape[1]}) != "
            f"total_num_frames ({total_num_frames})"
        )

        existing_a = state_dict[f"{prefix}edge_cluster_a"].cpu()
        existing_b = state_dict[f"{prefix}edge_cluster_b"].cpu()
        assert torch.equal(topo["edge_cluster_a"], existing_a) and torch.equal(
            topo["edge_cluster_b"], existing_b
        ), (
            f"{edges_path}'s edge topology no longer matches the checkpoint's -- "
            "can't safely regenerate correction_gate/connected_mask (edge "
            "order/count changed)."
        )

        device = state_dict[gate_key].device
        return {
            gate_key: topo["correction_gate"].to(device),
            mask_key: topo["connected_mask"].to(device),
        }

    # ------------------------------------------------------------------
    # Refresh (densify/cull)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def refresh_edge_membership(
        self, canonical_means: torch.Tensor, cluster_ids_all: torch.Tensor, coefs_all: torch.Tensor
    ) -> None:
        """EXPENSIVE: densify/cull 이후 ASSIGNMENT(어떤 Gaussian이 어떤
        edge/side에 속하는지, weight 포함 -- nearest-neighbor 재탐색)와
        num_fg_gaussians를 다시 계산한다. edge topology/patch reference
        cloud("boundary의 의미")는 건드리지 않는다. 끝에서
        refresh_edge_state도 호출한다."""
        patch_ref = EdgePatchReference(
            ref_edge_id=self.patch_ref_edge_id,
            ref_side=self.patch_ref_side,
            ref_cluster_id=self.patch_ref_cluster_id,
            points=self.patch_ref_points,
            weight=self.patch_ref_weight,
            local_scale=self.patch_ref_local_scale,
        )
        membership = assign_edge_memberships(
            canonical_means, cluster_ids_all, patch_ref, snap_radius=float(self.snap_radius.item())
        )
        device = self.row_gaussian_idx.device
        self.row_gaussian_idx = membership["row_gaussian_idx"].to(device)
        self.row_edge_id = membership["row_edge_id"].to(device)
        self.row_side = membership["row_side"].to(device)
        self.row_cluster_id = membership["row_cluster_id"].to(device)
        self.row_weight = membership["row_weight"].to(device)
        self.num_fg_gaussians = torch.tensor(int(canonical_means.shape[0]), dtype=torch.long, device=device)
        self.refresh_edge_state(canonical_means, coefs_all)

    @torch.no_grad()
    def refresh_edge_state(self, canonical_means: torch.Tensor, coefs_all: torch.Tensor) -> None:
        """CHEAP: ASSIGNMENT(row_gaussian_idx 등)는 그대로 두고, 그 rows가
        가리키는 Gaussian들의 CURRENT canonical_means/coefs_all 값만 다시
        gather한다 (nearest-neighbor 탐색 없음, 매 학습 step 호출 가능)."""
        idx = self.row_gaussian_idx
        if idx.numel() == 0:
            return
        device = idx.device
        self.row_canonical_mean = canonical_means[idx].detach().float().to(device)
        self.row_coefs = coefs_all[idx].detach().float().to(device)

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    @property
    def last_edge_correction(self) -> dict[str, torch.Tensor] | None:
        """가장 최근 detach_base=False 호출(compute_transforms/
        compute_transforms_coarse)에서 나온 raw (patch-weight 적용 전)
        {"omega": (E,B,2,3), "delta_t": (E,B,2,3), "gate": (E,B),
        "connected_mask": (E,B)}. detach_base=True(gap loss 전용) 호출은 이
        값을 절대 덮어쓰지 않는다."""
        return self._last_edge_correction

    def _edge_side_pivots(
        self, ts: torch.Tensor, detach: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """현재(correction 적용 전) base transform으로 이동한, edge/side별
        weighted patch centroid. :return: pivot_a, pivot_b (E,B,3),
        has_a, has_b (E,) bool, weight_sum_by_side (E,2) (patch 크기 feature용).
        """
        E = self.num_edges
        B = ts.shape[0]
        idx = self.row_gaussian_idx
        if idx.numel() == 0:
            zeros = ts.new_zeros(E, B, 3, dtype=torch.float32)
            zero_e = torch.zeros(E, dtype=torch.bool, device=idx.device)
            return zeros, zeros.clone(), zero_e, zero_e.clone(), ts.new_zeros(E, 2, dtype=torch.float32)

        row_transforms = ScalableMotionBases.compute_transforms(
            self, ts, self.row_coefs, self.row_cluster_id
        )  # (M, B, 3, 4)
        if detach:
            row_transforms = row_transforms.detach()
        row_positions = _apply_transform(row_transforms, self.row_canonical_mean)  # (M, B, 3)

        group_id = self.row_edge_id * 2 + self.row_side  # (M,)
        weight_sums = row_positions.new_zeros(E * 2)
        pos_sums = row_positions.new_zeros(E * 2, B, 3)
        weight_sums.index_add_(0, group_id, self.row_weight)
        pos_sums.index_add_(0, group_id, row_positions * self.row_weight[:, None, None])
        centroids = pos_sums / weight_sums[:, None, None].clamp_min(1e-8)

        pivot_a, pivot_b = centroids[0::2], centroids[1::2]
        has_a, has_b = weight_sums[0::2] > 0, weight_sums[1::2] > 0
        weight_sum_by_side = torch.stack([weight_sums[0::2], weight_sums[1::2]], dim=-1)  # (E, 2)
        return pivot_a, pivot_b, has_a, has_b, weight_sum_by_side

    def _compute_edge_correction(
        self, ts: torch.Tensor, detach_base: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GNN을 1회 실행해 edge-side-level (omega, delta_t), 이미
        _clamp_vector_magnitude로 clamp된 상태로 반환한다 ((E,B,2,3) 각각).

        :param detach_base: True면 base coarse/fine motion과 patch pivot/거리
            /속도 feature 경로를 전부 detach한다(boundary gap loss 전용 --
            gradient가 self.gnn 파라미터로만 흐르게 함). 이 경우
            last_edge_correction을 갱신하지 않는다(모듈 docstring의 "detach
            분리" 참고).
        """
        coarse_rot_6d = self.params["rots"][:, ts]
        coarse_transl = self.params["transls"][:, ts]
        centers = self.params["centers"]
        if detach_base:
            coarse_rot_6d = coarse_rot_6d.detach()
            coarse_transl = coarse_transl.detach()
            centers = centers.detach()

        ts_prev = (ts - 1).clamp(min=0)
        coarse_transl_prev = self.params["transls"][:, ts_prev]
        if detach_base:
            coarse_transl_prev = coarse_transl_prev.detach()
        coarse_vel = coarse_transl - coarse_transl_prev

        pivot_a, pivot_b, has_a, has_b, weight_sum_by_side = self._edge_side_pivots(ts, detach=detach_base)
        with torch.no_grad():
            pivot_a_prev, pivot_b_prev, _, _, _ = self._edge_side_pivots(ts_prev, detach=True)

        rel_pos = pivot_b - pivot_a  # (E, B, 3)
        current_dist = rel_pos.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (E, B, 1)
        direction = rel_pos / current_dist
        reference_dist = self.contact_reference_distance.nan_to_num(nan=0.0)[:, None, None].expand(
            -1, ts.shape[0], 1
        )
        dist_diff = current_dist - reference_dist
        rel_vel = (pivot_b - pivot_b_prev) - (pivot_a - pivot_a_prev)  # (E, B, 3)
        # z-score normalize against the FIXED training-time statistics
        # (patch_size_mean/patch_size_std, computed once in
        # from_scalable_motion_bases and reused as-is after checkpoint
        # restore -- see module docstring) rather than dividing by an ad hoc
        # constant. Unnormalized log1p(weight_sum) ranges ~3-6 in practice
        # while every other edge_patch_feat component is O(0.01-1); feeding
        # that in directly saturated the decoder's first Linear+ReLU
        # negative for every edge and permanently killed its gradient (see
        # module docstring / _EdgePatchDecoderGNN's LeakyReLU comment).
        patch_size_raw = torch.log1p(weight_sum_by_side)  # (E, 2)
        patch_size_norm = (patch_size_raw - self.patch_size_mean) / (self.patch_size_std + 1e-6)
        patch_size = patch_size_norm[:, None, :].expand(-1, ts.shape[0], -1)  # (E, B, 2)
        confidence = self.correction_gate[:, ts][..., None]  # (E, B, 1)

        edge_patch_feat = torch.cat(
            [pivot_a, pivot_b, rel_pos, direction, current_dist, reference_dist, dist_diff,
             rel_vel, patch_size, confidence],
            dim=-1,
        )  # (E, B, EDGE_PATCH_FEAT_DIM)

        omega_raw, delta_t_raw = self.gnn(coarse_rot_6d, coarse_transl, centers, coarse_vel, edge_patch_feat)
        omega = _clamp_vector_magnitude(omega_raw, self.max_omega)
        delta_t = _clamp_vector_magnitude(delta_t_raw, self.max_displacement[:, None, None, None])

        if not detach_base:
            self._last_edge_correction = {
                "omega": omega,
                "delta_t": delta_t,
                "gate": self.correction_gate[:, ts],
                "connected_mask": self.connected_mask[:, ts],
            }
        return omega, delta_t

    def compute_transforms_coarse(
        self, ts: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """부모의 coarse-only transform을 correction 없이 그대로 반환하되,
        GNN을 side effect로 실행해 last_edge_correction을 이 ts 기준으로
        갱신한다 (flow3d/trainer.py의 기존 관례 유지 -- 이전 per-cluster
        variant와 동일한 이유)."""
        self._compute_edge_correction(ts, detach_base=False)
        return ScalableMotionBases.compute_transforms_coarse(self, ts, cluster_ids)

    def compute_transforms(
        self,
        ts: torch.Tensor,
        coefs: torch.Tensor,
        cluster_ids: torch.Tensor,
        global_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """:param global_indices: (G,) long, optional -- 생략하면 전체
        canonical foreground 배열(순서 그대로)이라고 가정한다. 자세한 계약은
        모듈 docstring 및 다른 boundary-style variant와 동일.
        returns transforms (G, B, 3, 4)
        """
        base_transforms = ScalableMotionBases.compute_transforms(self, ts, coefs, cluster_ids)

        G = cluster_ids.shape[0]
        num_fg = int(self.num_fg_gaussians.item())
        device = cluster_ids.device
        if global_indices is None:
            if G != num_fg:
                raise ValueError(
                    "RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases."
                    "compute_transforms was called without global_indices and the query "
                    f"({G} rows) isn't the full canonical foreground array (expected {num_fg})."
                )
            global_indices = torch.arange(G, device=device)
        elif global_indices.shape[0] != G:
            raise ValueError(
                f"global_indices has {global_indices.shape[0]} entries but the query has {G} rows."
            )

        R_base = base_transforms[..., :3]
        t_base = base_transforms[..., 3]
        B = ts.shape[0]

        omega_e, delta_t_e = self._compute_edge_correction(ts, detach_base=False)  # (E, B, 2, 3) each
        pivot_a, pivot_b, _, _, _ = self._edge_side_pivots(ts, detach=False)
        pivot_by_side = torch.stack([pivot_a, pivot_b], dim=1)  # (E, 2, B, 3)

        if self.row_gaussian_idx.numel() == 0:
            return base_transforms

        inverse_map = torch.full((num_fg,), -1, dtype=torch.long, device=device)
        inverse_map[global_indices] = torch.arange(G, device=device)
        query_pos = inverse_map[self.row_gaussian_idx]  # (M,)
        in_query = query_pos >= 0
        if not bool(in_query.any()):
            return base_transforms

        row_query_idx = query_pos[in_query]  # (M',)
        row_edge_id_q = self.row_edge_id[in_query]
        row_side_q = self.row_side[in_query]
        row_weight_q = self.row_weight[in_query]
        M = row_query_idx.shape[0]

        gate_q = self.correction_gate[row_edge_id_q][:, ts]  # (M', B)
        combined_w = row_weight_q[:, None] * gate_q  # (M', B)

        omega_gathered = omega_e[row_edge_id_q, :, row_side_q]  # (M', B, 3)
        delta_t_gathered = delta_t_e[row_edge_id_q, :, row_side_q]  # (M', B, 3)
        pivot_gathered = pivot_by_side[row_edge_id_q, row_side_q]  # (M', B, 3)

        omega_scaled = combined_w[..., None] * omega_gathered  # (M', B, 3)
        R_corr_row = so3_exp_map(omega_scaled)  # (M', B, 3, 3)

        t_base_row = t_base[row_query_idx]  # (M', B, 3)
        delta_t_row_term = combined_w[..., None] * delta_t_gathered  # (M', B, 3)
        row_delta_t = (
            torch.einsum("mbij,mbj->mbi", R_corr_row, t_base_row - pivot_gathered)
            + pivot_gathered + delta_t_row_term - t_base_row
        )  # (M', B, 3), "Δt_m"

        total_w = t_base.new_zeros(G, B)
        total_w.index_add_(0, row_query_idx, combined_w)

        # max_m(w_m): the strongest membership sets the overall scale, instead
        # of an extra per-term w_m factor (which would square the weighting
        # and shrink multi-membership corrections far more than intended --
        # see module docstring's combination formula).
        max_w = t_base.new_zeros(G, B)
        row_index_expanded = row_query_idx[:, None].expand(-1, B)
        max_w.scatter_reduce_(0, row_index_expanded, combined_w, reduce="amax", include_self=True)

        omega_numerator = t_base.new_zeros(G, B, 3)
        omega_numerator.index_add_(0, row_query_idx, omega_scaled)

        delta_t_numerator = t_base.new_zeros(G, B, 3)
        delta_t_numerator.index_add_(0, row_query_idx, row_delta_t)

        total_w_safe = total_w.clamp_min(1e-8)[..., None]
        # a_i * mean_m(c_m) = max_m(w_m) * (sum_m w_m*c_m) / (sum_m w_m):
        # single membership reduces to exactly w*c (max_w == total_w there);
        # multiple memberships average without a naive sum-explosion and
        # without the w^2 over-shrinkage.
        combined_omega = max_w[..., None] * omega_numerator / total_w_safe
        combined_delta_t = max_w[..., None] * delta_t_numerator / total_w_safe

        R_new = compose_rotation(so3_exp_map(combined_omega), R_base)
        t_new = t_base + combined_delta_t

        return torch.cat([R_new, t_new.unsqueeze(-1)], dim=-1)


def _compute_edge_max_displacement(
    patch_ref: EdgePatchReference, num_edges: int, snap_radius: float, max_disp_scale: float
) -> torch.Tensor:
    """edge마다, correction magnitude의 절대 상한을 그 edge의 boundary patch
    (양쪽 side 합쳐) local Gaussian spacing의 median에서 유도한다 (flow3d/
    graph_relative_edge.py's _compute_edge_max_displacement와 동일한 설계)."""
    max_disp = torch.full((num_edges,), snap_radius * max_disp_scale, dtype=torch.float32)
    if patch_ref.num_points == 0:
        return max_disp
    for e in range(num_edges):
        local_scale_e = patch_ref.local_scale[patch_ref.ref_edge_id == e]
        local_scale_e = local_scale_e[local_scale_e > 0]
        if local_scale_e.numel() > 0:
            max_disp[e] = local_scale_e.median() * max_disp_scale
    return max_disp


def compute_edge_patch_gap_loss(
    motion_bases: RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases,
    ts: torch.Tensor,
    tolerance: float = 0.01,
) -> torch.Tensor:
    """opt-in 진단/보조 loss (flow3d/trainer.py's w_edge_boundary_gap, 기본값
    0.0). CONNECTED로 관찰됐고(connected_mask, 하드 마스크) reliable_edge_mask
    가 True인(신뢰할 수 있는) (edge, frame)에서만, correction 적용 *후* 실제
    양쪽 patch centroid 거리가 (contact_reference_distance + tolerance)보다
    벌어지면 hinge loss를 준다.

    gradient는 오직 motion_bases.gnn 자신에게만 흐른다 --
    _compute_edge_correction(detach_base=True)/_edge_side_pivots(detach=True)
    를 통해 base coarse/fine motion과 patch pivot feature 경로를 전부
    detach했기 때문이다. 이 detach 경로는 motion_bases.last_edge_correction을
    절대 덮어쓰지 않는다 (모듈 docstring 참고).

    centroid 자신은 자기 자신을 pivot으로 회전해도 움직이지 않으므로
    (R_corr @ (pivot - pivot) == 0), correction 적용 후 centroid는 단순히
    pivot + delta_t다 -- rotation은 이 loss에 기여하지 않는다(centroid 자체의
    이동에는 translation만 영향을 준다).

    :return: scalar; reliable + CONNECTED인 (edge, frame)이 하나도 없으면 0.0.
    """
    omega_e, delta_t_e = motion_bases._compute_edge_correction(ts, detach_base=True)
    pivot_a, pivot_b, has_a, has_b, _ = motion_bases._edge_side_pivots(ts, detach=True)

    pivot_a_after = pivot_a + delta_t_e[:, :, 0, :]
    pivot_b_after = pivot_b + delta_t_e[:, :, 1, :]
    dist_after = (pivot_b_after - pivot_a_after).norm(dim=-1)  # (E, B)

    connected_mask = motion_bases.connected_mask[:, ts]  # (E, B), hard
    include = has_a[:, None] & has_b[:, None] & connected_mask & motion_bases.reliable_edge_mask[:, None]
    if not bool(include.any()):
        return motion_bases.contact_reference_distance.new_zeros(())

    reference = motion_bases.contact_reference_distance[:, None]  # (E, 1)
    gap_excess = (dist_after - (reference + tolerance)).clamp_min(0.0)
    return (gap_excess.pow(2) * include).sum() / include.sum().clamp_min(1)


if __name__ == "__main__":
    # Verification:
    #   1. zero-init equivalence: step 0 == plain ScalableMotionBases exactly.
    #   2. interior Gaussians (no membership row anywhere) stay EXACTLY equal
    #      to the plain baseline even after the head is moved off zero.
    #   3. same cluster, different edge patch -> different correction
    #      (cluster 0's edge0-only vs edge1-only Gaussians).
    #   4. pivot correctness: a single-membership Gaussian's corrected
    #      transform matches a hand-recomputed formula using the patch
    #      CENTROID as pivot, and does NOT match if the cluster center is
    #      used instead (negative control).
    #   5. multi-membership combination is NOT naive summation: a Gaussian in
    #      two edges' patches matches the weighted-tangent-average formula,
    #      not omega_1+omega_2.
    #   6. densify/cull: refresh_edge_membership correctly reassigns
    #      membership via nearest-neighbor snap; an edge with no matching
    #      boundary_patch.pt pair degrades to zero rows without crashing.
    #   7. checkpoint round-trip.
    #   8. has_gnn_state marker doesn't collide with EdgeBoundary's
    #      "patch_ref_points" (which this class also happens to have).
    #   9. detach_base=True (gap-loss path) does NOT overwrite last_edge_correction.
    import tempfile
    import os

    import torch as T

    T.manual_seed(0)
    num_clusters, num_frames, num_fine, num_fg = 4, 8, 3, 20

    centers = T.randn(num_clusters, 3)
    rots = T.randn(num_clusters, num_frames, 6)
    transls = T.randn(num_clusters, num_frames, 3) * 0.1
    fine_rots = T.randn(num_clusters, num_fine, num_frames, 6)
    fine_transls = T.randn(num_clusters, num_fine, num_frames, 3) * 0.01
    baseline = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    # Synthetic foreground array: clusters 0/1/2/3 with 5 Gaussians each,
    # spaced 1.0 apart along x (see graph_relative_linear_attention_boundary's
    # earlier revisions for why -- keeps nearest-neighbor snap unambiguous).
    cluster_ids_all = T.repeat_interleave(T.arange(num_clusters), num_fg // num_clusters)
    canonical_means = T.zeros(num_fg, 3)
    canonical_means[:, 0] = T.arange(num_fg, dtype=T.float32)
    coefs_all = T.softmax(T.randn(num_fg, num_fine), dim=-1)

    # edges.pt: 3 kept edges. edge2 (cluster1-cluster3) deliberately has NO
    # matching boundary_patch.pt pair below (tests graceful degradation).
    def _edge_entry(ca, cb, ref_dist):
        return {
            "cluster_a": ca, "cluster_b": cb,
            "connected_frame_indices": list(range(num_frames)),
            "unknown_frame_indices": [],
            "contact_reference_distance": ref_dist,
            "persistence": 1.0,
            "num_known_frames": num_frames,
        }

    edges_payload = {
        "edges_kept": [
            _edge_entry(0, 1, 1.0),  # edge_id 0
            _edge_entry(0, 2, 2.0),  # edge_id 1
            _edge_entry(1, 3, 3.0),  # edge_id 2 (no patch data)
        ]
    }

    # boundary_patch.pt: cluster-0 Gaussian 0 belongs to BOTH edge0 and edge1
    # (multi-membership test subject). Gaussian 1 is edge0-only, Gaussian 2 is
    # edge1-only (same-cluster, different-edge-patch comparison target).
    # Gaussians 3, 4 (cluster 0), 6-9 (cluster 1), 11-14 (cluster 2), and all
    # of cluster 3 (15-19, edge2 has no patch) are interior.
    boundary_patch_payload = {
        "pairs": [
            {
                "cluster_a": 0, "cluster_b": 1,
                "global_indices_a": T.tensor([0, 1]), "global_indices_b": T.tensor([5]),
                "weight_a": T.tensor([0.9, 0.8]), "weight_b": T.tensor([0.6]),
                "local_scale_a": T.tensor([0.02, 0.02]), "local_scale_b": T.tensor([0.02]),
            },
            {
                "cluster_a": 0, "cluster_b": 2,
                "global_indices_a": T.tensor([0, 2]), "global_indices_b": T.tensor([10]),
                "weight_a": T.tensor([0.3, 0.7]), "weight_b": T.tensor([0.5]),
                "local_scale_a": T.tensor([0.02, 0.02]), "local_scale_b": T.tensor([0.02]),
            },
            {
                # candidate-only pair, no matching kept edge -- must be dropped.
                "cluster_a": 2, "cluster_b": 3,
                "global_indices_a": T.tensor([11]), "global_indices_b": T.tensor([15]),
                "weight_a": T.tensor([0.9]), "weight_b": T.tensor([0.9]),
                "local_scale_a": T.tensor([0.02]), "local_scale_b": T.tensor([0.02]),
            },
        ],
        "meta": {"num_pairs": 3},
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        edges_path = f.name
    T.save(edges_payload, edges_path)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        boundary_patch_path = f.name
    T.save(boundary_patch_payload, boundary_patch_path)

    graph_bases = RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline,
        edge_index=T.zeros(2, 0, dtype=T.long),  # unused, call-site parity only
        edges_path=edges_path,
        boundary_patch_path=boundary_patch_path,
        canonical_means=canonical_means,
        cluster_ids_all=cluster_ids_all,
        coefs_all=coefs_all,
        total_num_frames=num_frames,
        gnn_hidden_dim=32,
        gnn_num_layers=2,
        gnn_num_heads=4,
    )
    print(f"[topology] num_edges={graph_bases.num_edges} (expect 3, incl. edge2 with no patch)")
    assert graph_bases.num_edges == 3
    edge2_rows = (graph_bases.row_edge_id == 2).sum().item()
    print(f"[topology] edge2 (no matching boundary_patch pair) membership rows = {edge2_rows} (expect 0)")
    assert edge2_rows == 0

    ts = T.arange(num_frames)

    # --- 1. zero-init equivalence ---
    ref = baseline.compute_transforms(ts, coefs_all, cluster_ids_all)
    out = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all)
    max_diff = (ref - out).abs().max().item()
    print(f"[zero-init] max |baseline - edge_patch_graph_corrected| = {max_diff:.3e}")
    assert max_diff < 1e-5

    last0 = graph_bases.last_edge_correction
    assert last0["omega"].abs().max().item() == 0.0 and last0["delta_t"].abs().max().item() == 0.0

    # --- at zero-init, gradient is starved everywhere upstream of the last
    # (still-zero) Linear -- same expected "head hasn't moved yet" behavior
    # as the baseline per-cluster GNN, not a sign of anything broken.
    hidden_grad_at_init = graph_bases.gnn.edge_decoder[1].weight.grad
    attn_grad_at_init = graph_bases.gnn.layers[0].attn.grad
    print(
        f"[grad] at zero-init: edge_decoder[1] grad={None if hidden_grad_at_init is None else hidden_grad_at_init.norm().item()} "
        f"attn grad={None if attn_grad_at_init is None else attn_grad_at_init.norm().item()} (both expect None/0)"
    )
    assert hidden_grad_at_init is None or hidden_grad_at_init.norm().item() == 0.0
    assert attn_grad_at_init is None or attn_grad_at_init.norm().item() == 0.0

    # --- move the head off zero ---
    graph_bases.zero_grad()
    loss = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all).pow(2).mean()
    loss.backward()
    with T.no_grad():
        for p in graph_bases.gnn.edge_decoder[-1].parameters():
            if p.grad is not None:
                p -= 5.0 * p.grad

    # --- 0b. once the head is off zero, gradient must reach the FIRST
    # Linear and the attention layer too -- the dead-ReLU bug this LayerNorm
    # + LeakyReLU fix addresses would keep these at exactly 0 forever even
    # after the head moves, since a saturated ReLU's backward pass is an
    # exact zero regardless of downstream gradient magnitude.
    graph_bases.zero_grad()
    loss2 = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all).pow(2).mean()
    loss2.backward()
    hidden_grad = graph_bases.gnn.edge_decoder[1].weight.grad
    attn_grad = graph_bases.gnn.layers[0].attn.grad
    print(
        f"[grad] post-step: edge_decoder[1] grad norm={hidden_grad.norm().item():.3e} "
        f"attn grad norm={attn_grad.norm().item():.3e} (both expect > 0)"
    )
    assert hidden_grad.norm().item() > 0.0, "gradient must reach the first Linear once the head is off zero"
    assert attn_grad.norm().item() > 0.0, "gradient must reach the attention layer once the head is off zero"

    out2 = graph_bases.compute_transforms(ts, coefs_all, cluster_ids_all)

    # --- 2. interior Gaussians must stay EXACTLY the baseline ---
    interior_ids = [3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    interior_diff = (out2[interior_ids] - ref[interior_ids]).abs().max().item()
    print(f"[interior] max |out - baseline| after head off zero = {interior_diff:.3e} (expect 0)")
    assert interior_diff < 1e-6

    # --- 3. same cluster (0), different edge patch -> different correction ---
    diff1 = (out2[1] - ref[1])  # cluster0, edge0-only
    diff2 = (out2[2] - ref[2])  # cluster0, edge1-only
    assert diff1.abs().max().item() > 1e-6 and diff2.abs().max().item() > 1e-6
    cross_edge_gap = (diff1 - diff2).abs().max().item()
    print(f"[same-cluster] |edge0-only correction - edge1-only correction| = {cross_edge_gap:.3e} (expect > 0)")
    assert cross_edge_gap > 1e-6, "different edge patches on the same cluster must not collapse to the same correction"

    # --- 4. pivot correctness: hand-recompute Gaussian 1 (edge0-only, side a) ---
    last = graph_bases.last_edge_correction
    pivot_a, pivot_b, has_a, has_b, _ = graph_bases._edge_side_pivots(ts, detach=False)
    base_transforms = ScalableMotionBases.compute_transforms(graph_bases, ts, coefs_all, cluster_ids_all)
    R_base1, t_base1 = base_transforms[1][..., :3], base_transforms[1][..., 3]

    w1 = 0.8 * graph_bases.correction_gate[0, ts]  # edge0, gate (all-connected -> 1.0)
    omega1 = w1[:, None] * last["omega"][0, :, 0, :]  # edge0, side a
    delta_t1 = w1[:, None] * last["delta_t"][0, :, 0, :]
    R_corr1 = so3_exp_map(omega1)

    patch_centroid_pivot = pivot_a[0]  # edge0's side-a patch centroid
    manual_t1_patch = torch.einsum("bij,bj->bi", R_corr1, t_base1 - patch_centroid_pivot) + patch_centroid_pivot + delta_t1
    manual_R1 = R_corr1 @ R_base1
    patch_pivot_diff = (manual_t1_patch - out2[1][..., 3]).abs().max().item()
    patch_pivot_rot_diff = (manual_R1 - out2[1][..., :3]).abs().max().item()
    print(f"[pivot] hand-recomputed (patch centroid pivot) vs actual: t diff={patch_pivot_diff:.3e} R diff={patch_pivot_rot_diff:.3e}")
    assert patch_pivot_diff < 1e-4 and patch_pivot_rot_diff < 1e-4

    cluster_center_pivot = graph_bases.params["centers"][0] + graph_bases.params["transls"][0, ts]
    center_pivot_diff_check = (patch_centroid_pivot - cluster_center_pivot).abs().max().item()
    print(f"[pivot] |patch centroid - cluster center| = {center_pivot_diff_check:.3e} (expect > 0, they must differ)")
    assert center_pivot_diff_check > 1e-4
    manual_t1_center = torch.einsum("bij,bj->bi", R_corr1, t_base1 - cluster_center_pivot) + cluster_center_pivot + delta_t1
    center_pivot_diff = (manual_t1_center - out2[1][..., 3]).abs().max().item()
    print(f"[pivot] hand-recomputed (WRONG: cluster-center pivot) vs actual t diff = {center_pivot_diff:.3e} (expect > 0)")
    assert center_pivot_diff > 1e-4, "cluster-center pivot must NOT match -- only the patch centroid should"

    # --- 5. multi-membership: Gaussian 0 (edge0 side a w=0.9, edge1 side a w=0.3) ---
    R_base0, t_base0 = base_transforms[0][..., :3], base_transforms[0][..., 3]
    gate0 = graph_bases.correction_gate[0, ts]
    gate1 = graph_bases.correction_gate[1, ts]
    w_e0 = 0.9 * gate0
    w_e1 = 0.3 * gate1
    omega_e0 = last["omega"][0, :, 0, :]
    omega_e1 = last["omega"][1, :, 0, :]
    delta_t_e0 = last["delta_t"][0, :, 0, :]
    delta_t_e1 = last["delta_t"][1, :, 0, :]

    omega_scaled_e0 = w_e0[:, None] * omega_e0
    omega_scaled_e1 = w_e1[:, None] * omega_e1
    R_corr_e0 = so3_exp_map(omega_scaled_e0)
    R_corr_e1 = so3_exp_map(omega_scaled_e1)
    pivot_e0 = pivot_a[0]
    pivot_e1 = pivot_a[1]
    dt_e0 = (
        torch.einsum("bij,bj->bi", R_corr_e0, t_base0 - pivot_e0) + pivot_e0 + w_e0[:, None] * delta_t_e0 - t_base0
    )
    dt_e1 = (
        torch.einsum("bij,bj->bi", R_corr_e1, t_base0 - pivot_e1) + pivot_e1 + w_e1[:, None] * delta_t_e1 - t_base0
    )
    total_w0 = w_e0 + w_e1
    max_w0 = torch.maximum(w_e0, w_e1)  # a_i = max_m(w_m)
    combined_omega0 = max_w0[:, None] * (omega_scaled_e0 + omega_scaled_e1) / total_w0[:, None]
    combined_dt0 = max_w0[:, None] * (dt_e0 + dt_e1) / total_w0[:, None]
    manual_R0 = so3_exp_map(combined_omega0) @ R_base0
    manual_t0 = t_base0 + combined_dt0

    multi_t_diff = (manual_t0 - out2[0][..., 3]).abs().max().item()
    multi_R_diff = (manual_R0 - out2[0][..., :3]).abs().max().item()
    print(f"[multi-membership] hand-recomputed max-weight x weighted-average combine vs actual: t diff={multi_t_diff:.3e} R diff={multi_R_diff:.3e}")
    assert multi_t_diff < 1e-4 and multi_R_diff < 1e-4

    naive_sum_omega = omega_scaled_e0 + omega_scaled_e1  # what naive summation (forbidden) would give
    naive_vs_combined = (naive_sum_omega - combined_omega0).abs().max().item()
    print(f"[multi-membership] |naive sum - weighted combine| = {naive_vs_combined:.3e} (expect > 0, must not be naive sum)")
    assert naive_vs_combined > 1e-6

    old_w_squared_omega = (
        w_e0[:, None] * omega_scaled_e0 + w_e1[:, None] * omega_scaled_e1
    ) / total_w0[:, None]  # the previous (over-shrinking) w^2 formula, now fixed
    w_squared_vs_combined = (old_w_squared_omega - combined_omega0).abs().max().item()
    print(f"[multi-membership] |old w^2 formula - fixed max-weight formula| = {w_squared_vs_combined:.3e} (expect > 0)")
    assert w_squared_vs_combined > 1e-6

    # --- 9. detach_base=True must NOT overwrite last_edge_correction ---
    last_before_gap = {k: v.clone() for k, v in graph_bases.last_edge_correction.items()}
    _ = graph_bases._compute_edge_correction(ts, detach_base=True)
    last_after_gap = graph_bases.last_edge_correction
    unchanged = all(torch.equal(last_before_gap[k], last_after_gap[k]) for k in last_before_gap)
    print(f"[detach] last_edge_correction unchanged after a detach_base=True call = {unchanged} (expect True)")
    assert unchanged, "the detached gap-loss forward pass must not clobber last_edge_correction"

    # --- 6. refresh after simulated densify + cull ---
    dup_pos = canonical_means[1:2] + 1e-4  # duplicate of Gaussian 1 (edge0-only)
    means_densified = T.cat([canonical_means, dup_pos], dim=0)
    cluster_ids_densified = T.cat([cluster_ids_all, cluster_ids_all[1:2]], dim=0)
    coefs_densified = T.cat([coefs_all, coefs_all[1:2]], dim=0)
    graph_bases.refresh_edge_membership(means_densified, cluster_ids_densified, coefs_densified)
    new_idx = means_densified.shape[0] - 1
    dup_rows = (graph_bases.row_gaussian_idx == new_idx).nonzero(as_tuple=True)[0]
    print(f"[refresh] duplicate of Gaussian 1 got {dup_rows.numel()} membership row(s) (expect 1, edge0/side a)")
    assert dup_rows.numel() == 1
    assert graph_bases.row_edge_id[dup_rows[0]].item() == 0 and graph_bases.row_side[dup_rows[0]].item() == 0
    assert abs(graph_bases.row_weight[dup_rows[0]].item() - 0.8) < 1e-5

    keep_mask = T.ones(means_densified.shape[0], dtype=T.bool)
    keep_mask[3] = False  # cull an interior Gaussian
    means_culled = means_densified[keep_mask]
    cluster_ids_culled = cluster_ids_densified[keep_mask]
    coefs_culled = coefs_densified[keep_mask]
    graph_bases.refresh_edge_membership(means_culled, cluster_ids_culled, coefs_culled)
    print(f"[refresh] after cull: num_fg_gaussians={int(graph_bases.num_fg_gaussians)} (expect {means_culled.shape[0]})")
    assert int(graph_bases.num_fg_gaussians) == means_culled.shape[0]
    _ = graph_bases.compute_transforms(ts, coefs_culled, cluster_ids_culled)  # must not raise
    print("[refresh] compute_transforms after densify/cull OK")

    # --- 7. save -> init_from_state_dict round-trip ---
    full_state_dict = {f"motion_bases.{k}": v for k, v in graph_bases.state_dict().items()}
    assert "motion_bases.last_edge_correction" not in full_state_dict  # never a buffer
    restored = RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.init_from_state_dict(
        full_state_dict, prefix="motion_bases."
    )
    out_restored = restored.compute_transforms(ts, coefs_culled, cluster_ids_culled)
    out_original = graph_bases.compute_transforms(ts, coefs_culled, cluster_ids_culled)
    max_diff_roundtrip = (out_restored - out_original).abs().max().item()
    print(f"[round-trip] max |original - restored| = {max_diff_roundtrip:.3e}")
    assert max_diff_roundtrip < 1e-6

    # --- 8. has_gnn_state must not collide with EdgeBoundary's patch_ref_points ---
    assert RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.has_gnn_state(
        full_state_dict, "motion_bases."
    )
    fake_edge_boundary_like = {
        "motion_bases.patch_ref_points": T.zeros(3),
        "motion_bases.edge_cluster_a": T.zeros(2),
        # deliberately no "row_weight" key
    }
    falsely_matched = RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.has_gnn_state(
        fake_edge_boundary_like, "motion_bases."
    )
    print(f"[marker] has_gnn_state on an EdgeBoundary-shaped dict = {falsely_matched} (expect False)")
    assert not falsely_matched

    # --- 10. compute_edge_patch_gap_loss: callable, detach-separated, and
    # does not disturb last_edge_correction either ---
    last_before = {k: v.clone() for k, v in graph_bases.last_edge_correction.items()}
    gap_loss = compute_edge_patch_gap_loss(graph_bases, ts)
    print(f"[gap-loss] value = {gap_loss.item():.3e}, requires_grad = {gap_loss.requires_grad}")
    assert gap_loss.requires_grad
    unchanged_after_gap = all(torch.equal(last_before[k], graph_bases.last_edge_correction[k]) for k in last_before)
    assert unchanged_after_gap
    graph_bases.zero_grad()
    gap_loss.backward()
    base_motion_grad = graph_bases.params["rots"].grad
    edge_decoder_grad = graph_bases.gnn.edge_decoder[-1].weight.grad
    print(
        f"[gap-loss] base motion grad = {None if base_motion_grad is None else base_motion_grad.abs().max().item()} "
        f"(expect None -- detached), edge_decoder grad norm = {edge_decoder_grad.norm().item():.3e} (expect > 0)"
    )
    assert base_motion_grad is None, "gap loss must not reach base coarse/fine motion params"
    assert edge_decoder_grad.norm().item() > 0.0

    # --- 11. regenerate_stale_gate_state_dict: a checkpoint attached at a
    # shorter num_frames (simulating --num_init_frames < dataset length) gets
    # its correction_gate/connected_mask regenerated to the full length, and
    # matches what a from-scratch attach at the full length would produce;
    # already-current gates are left alone (returns None). ---
    stale_num_frames = num_frames - 3
    stale_topo = _load_edge_topology_from_edges_pt(edges_path, stale_num_frames, 0.5, 1)
    stale_state_dict = dict(full_state_dict)
    stale_state_dict["motion_bases.correction_gate"] = stale_topo["correction_gate"]
    stale_state_dict["motion_bases.connected_mask"] = stale_topo["connected_mask"]

    no_op = RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.regenerate_stale_gate_state_dict(
        full_state_dict, edges_path, num_frames, 0.5, 1
    )
    print(f"[regen] already-current gate -> {no_op} (expect None)")
    assert no_op is None

    regenerated = RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.regenerate_stale_gate_state_dict(
        stale_state_dict, edges_path, num_frames, 0.5, 1
    )
    assert regenerated is not None
    print(
        f"[regen] stale ({stale_num_frames}-frame) gate -> regenerated shape "
        f"{tuple(regenerated['motion_bases.correction_gate'].shape)} (expect (3, {num_frames}))"
    )
    assert regenerated["motion_bases.correction_gate"].shape == (3, num_frames)
    assert regenerated["motion_bases.connected_mask"].shape == (3, num_frames)
    assert torch.equal(regenerated["motion_bases.correction_gate"], full_state_dict["motion_bases.correction_gate"])
    assert torch.equal(regenerated["motion_bases.connected_mask"], full_state_dict["motion_bases.connected_mask"])

    os.unlink(edges_path)
    os.unlink(boundary_patch_path)

    print("OK")

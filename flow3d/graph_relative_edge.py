"""
flow3d/graph_relative_edge.py

Per-edge boundary correction: cluster 전체를 회전/이동시키는 기존
*GraphCorrectedScalableMotionBases (flow3d/graph_coupling.py,
flow3d/graph_relative_linear_attention.py) 대신, cluster graph의 각 edge(=한
쌍의 인접 cluster)마다 아주 작은 "이음매(boundary)" correction만 출력한다.

왜 per-cluster correction이 문제인가
--------------------------------------
기존 GNN은 cluster c 전체에 적용되는 단일 강체 보정 (omega_c, delta_t_c)를
예측한다. 이 보정은 c에 속한 모든 Gaussian에 동일하게 적용되므로, 한쪽 경계의
벌어짐을 닫으려고 예측한 correction이 반대쪽(먼) 실루엣/깊이까지 함께 밀어
버릴 수 있다 (예: 손목 경계를 닫으려는 보정이 손가락 끝 실루엣까지 통째로
옮겨버림).

이 파일의 접근
----------------
1. GNN은 cluster별이 아니라 cluster graph의 각 (무방향) edge (a, b)마다
   "이번 스텝에 이 edge의 max_displacement 중 몇 %를 당길지"
   alpha_{ab}(t) in [0, 1]만 예측한다 (회전 없음, 절대 크기 자체가 아니라
   비율).
2. 방향은 GNN이 학습하지 않는다 -- _falloff_side_means가 "지금 boundary에
   속한 Gaussian"의 실제(coarse+fine blended) falloff-weighted 평균 위치
   (mean_a, mean_b)에서 기하학적으로 계산하고 항상 detach한다:
     dir_{ab}(t) = normalize(mean_b(t) - mean_a(t))
   이 값은 compute_boundary_gap_distances(진단용)가 재는 방향과 **정의상
   완전히 동일한 계산**이다 (같은 _falloff_side_means 호출). GNN 입력으로는
   ||mean_b - mean_a||(dist_before)와 canonical_distance를 이 edge의
   max_displacement로 나눈 무차원 스칼라만 준다 (전역 좌표 방향 정보는 GNN에
   주지 않는다 -- 방향은 항상 위 식으로 기하학적으로 고정되므로). 예전 버전은
   correction magnitude를 gap_error = relu(dist_before -
   contact_reference_distance - gap_tolerance)에 곱해서, "같은 재구성 자신의
   CONNECTED 프레임 median"인 contact_reference_distance가 이미 그 edge의
   상시 misalignment를 "정상"으로 흡수해버린 edge(예: 두 클러스터가 처음부터
   끝까지 거의 일정하게 벌어져 있는 경우)에서는 gap_error가 모든 프레임에서
   정확히 0이 되어 render loss의 gradient가 alpha까지 전혀 도달하지 못하는
   문제가 실측됐다 (flow3d/analysis/gnn_check_edge.py, edge 15(4-21) 등).
   alpha는 gap_error가 아니라 render loss(RGB/depth/mask/track, 아래 3번)로
   직접 학습된다.
3. correction magnitude m_{ab}(t) = gate_{ab}(t) * max_displacement_{ab} *
   alpha_{ab}(t)이고, world-space additive translation으로 최종
   (coarse+fine blended) 위치의 translation 성분에만 더해진다 (회전은 절대
   건드리지 않는다):
     Gaussian i (side a, edge (a,b)) -> position_i += +0.5 * m_{ab} * dir_{ab} * w_i
     Gaussian i (side b, edge (a,b)) -> position_i += -0.5 * m_{ab} * dir_{ab} * w_i
   양쪽을 반씩 움직여 이음매를 좁힌다. alpha in [0,1]이므로 m_{ab}는 항상
   [0, gate * max_displacement] 안에 있다 -- sigmoid라 부호가 뒤집혀 벌리는
   방향으로 움직일 수는 없지만(pull-only), gap_error 기반 설계와 달리
   "실제 gap 크기를 절대 못 넘는다"는 보장은 없다 (대신 max_displacement
   자체를 boundary 국소 Gaussian spacing 기준으로 작게 잡아 과도한 이동을
   막는다 -- EdgeBoundaryGraphCorrectedScalableMotionBases 참고).
   compute_boundary_gap_loss/contact_reference_distance 기반 hinge loss는
   학습 경로에서는 기본적으로 꺼져 있다(w_boundary_gap 기본값 0) -- 진단용
   함수(compute_boundary_gap_distances 등)는 그대로 남아있다.
4. w_i는 경계에서 멀어질수록 부드럽게 0으로 줄어드는 falloff 가중치
   (Gaussian-RBF: exp(-(d_i / falloff_radius)^2), d_i = Gaussian i에서 그
   edge의 *반대편* cluster까지의 최근접-이웃 거리)이다. 이 거리 기준은 새로
   지어낸 것이 아니라 flow3d/analysis/cluster_graph.py가 애초에
   boundary_global_indices_a/b를 고를 때 쓰는 것과 정확히 같다
   (directed_nn/select_boundary_local_indices -- "반대편 cluster까지 가장
   가까운 점들"을 hard top-k로 뽑는다); 여기서는 그 hard cutoff를 부드러운
   RBF로 바꿨을 뿐이다.

"boundary의 의미"(고정) vs. "boundary에 속한 Gaussian"(densify/cull에 따라 갱신)
------------------------------------------------------------------------------
edges.pt(build_cluster_graph.py)가 저장한 boundary_global_indices_a/b는 그
파일을 만들 때의 한 특정 foreground Gaussian 배열 스냅샷을 가리키는 index다.
학습 중에는 adaptive density control(densify/cull, flow3d/trainer.py의
_densify_control_step/_cull_control_step)이 --optim.no-enable-bases-control
여부와 무관하게 계속 foreground Gaussian을 split/dup/cull하며 배열을 통째로
재구성한다(flow3d/params.py's GaussianParams.densify_params/cull_params --
concat/mask로 새 배열을 만들 뿐, 옛 index를 보존하지 않는다). 즉 index 기반
boundary 정의는 학습이 진행되면 곧바로 stale해진다.

그래서 이 파일은 "boundary가 무엇을 의미하는지"와 "지금 어떤 Gaussian이
거기에 속하는지"를 분리한다:
  - 고정(edges.pt에서 한 번 읽고 그 뒤로 절대 바뀌지 않음): edge topology
    (edge_cluster_a/b) 및 각 side의 anchor 위치(anchor_a/b_canonical -- 그
    edges.pt가 저장한 boundary Gaussian 집합의 canonical 평균 위치, 3D 점
    하나일 뿐 index가 아니다). canonical_distance(=||anchor_a - anchor_b||,
    허용 거리의 기준값)와 correction의 방향(direction, 아래 참고)만 이 고정
    anchor에서 유도되고, 그 둘 다 "대략 어디"만 알면 되는 단일 대표점 용도라
    densify/cull에 영향받지 않는다.
  - 라이브(densify/cull이 foreground Gaussian 개수/순서를 바꿀 때마다
    refresh_boundary_falloff로 다시 계산됨): 어떤 (지금 존재하는) Gaussian이
    그 edge의 falloff weight를 받는지(falloff_global_idx/edge_id/sign/
    weight). 위 4번처럼 "두 cluster의 CURRENT canonical 위치"만으로 순수하게
    정의되므로(저장된 boundary index는 전혀 쓰지 않는다), 그 시점의
    canonical_means/cluster_ids만 있으면 옛 상태 없이도 처음부터 다시 계산할
    수 있다. 새로 split/dup된 Gaussian은 자기 위치에 맞는 weight를 자동으로
    받고, culled된 Gaussian은 그냥 다음 refresh에서 사라진다.
5. alpha_{ab}는 head bias를 큰 음수(EdgeBoundaryGNN._INIT_LOGIT_BIAS=-4.5)로
   초기화하므로 학습 시작 시점에는 alpha가 0에 매우 가깝다 (sigmoid는 정확히
   0을 낼 수 없으므로, 기존 *GraphCorrectedScalableMotionBases들의 "정확히
   0" zero-init 등가성과 달리 "거의 0"이다). head weight는 정확히 0이 아니라
   작은 랜덤값으로 초기화한다(std=0.01) -- weight가 정확히 0이면
   d(logit)/d(edge_feat) = 0이라 encoder/message-passing 레이어에 처음부터
   gradient가 전혀 안 흘러(오직 bias만 학습되는 시작 구간이 생김) 이번
   수정의 목적(render loss가 GNN 전체에 직접 도달)과 어긋난다.
6. correction_gate(edge별, per-frame, 0~1)는 CONNECTED/DISCONNECTED로 관찰된
   프레임 사이의 UNKNOWN 구간만 선형 보간한다 -- 한 번도 CONNECTED/
   DISCONNECTED로 관찰되지 못한 채 시퀀스 맨 앞/뒤까지 이어지는 UNKNOWN
   구간(관찰 범위 바깥)은 보간할 두 매듭이 없으므로 그 바깥값을 그대로
   유지(np.interp 기본 동작)하는 대신 0(DISCONNECTED 취급)으로 둔다 --
   `_build_edge_gate` 참고.

Loss는 이 파일이 아니라 flow3d/trainer.py가 부른다 (다른 GNN correction
variant와 동일한 분리: 이 파일은 correction의 정의/적용, loss는 이 파일이
제공하는 boundary_magnitude_* 함수를 trainer가 호출). correction magnitude는
gap_error 기반 hinge loss가 아니라 render loss(RGB/depth/mask/track)로 직접
학습된다 (2번 참고) -- compute_boundary_gap_loss는 학습 경로에서는 기본적으로
꺼져 있고(trainer.py's w_boundary_gap 기본값 0.0), 켜고 싶으면 opt-in loss로,
아니면 compute_boundary_gap_distances를 통해 진단용으로 쓸 수 있다:
  - motion_bases._falloff_side_means(위 2번과 완전히 동일한 함수, 항상
    detach)로 각 side의 현재(실제 coarse+fine blended) 위치를 얻는다.
  - 그 거리가 (contact_reference_distance + motion_bases.gap_tolerance)보다
    벌어질 때만 (hinge) loss를 준다 -- 허용 범위 안에서는 정확히 0.
boundary_magnitude_reg_loss/boundary_magnitude_smoothness_loss(이 둘은 학습
경로에 그대로 남아 있다)는 correction 크기 자체와 그 시간 변화(가속도)를
제한한다.

compute_transforms의 global_indices -- subset query 지원
-----------------------------------------------------------
per-cluster correction과 달리 이 correction은 개별 Gaussian identity에 따라
달라지므로 (falloff weight가 Gaussian마다 다르다), 어떤 subset을 질의했는지가
아니라 "정확히 어떤 Gaussian인지"를 알아야 한다. cluster_ids/coefs 값만으로는
"어느 cluster인지"까지만 알 수 있지 "정확히 어떤 canonical Gaussian인지"는 알
수 없으므로, compute_transforms(ts, coefs, cluster_ids, global_indices=...)는
그 identity를 별도 인자로 받는다. flow3d/scene_model.py's
SceneModel.compute_transforms(ts, inds)가 자신이 받은 inds(없으면
arange(전체))를 그대로 넘겨주므로, flow3d/trainer.py가 쓰는 "항상 전체 배열"
호출(ts_neighbors/ts, inds=None)은 물론이고 flow3d/renderer.py's
Renderer.__init__(track 시각화용으로 10개 Gaussian만 inds=torch.arange(10)로
질의)처럼 임의의 subset/순서로 질의해도 정확히 동작한다 (해당 Gaussian이
어떤 edge의 falloff row에도 안 걸리면 단순히 correction 없이 base position
그대로 나온다). global_indices를 생략하면 query가 정확히 전체 canonical
foreground 배열(순서 그대로)이라고 가정하고, 아니면 ValueError를 던진다.
num_fg_gaussians는 고정 값이 아니라 refresh_boundary_falloff가 호출될 때마다
갱신되는 버퍼이므로, densify/cull 직후에도(refresh가 호출된 다음이라면) 이
검사는 여전히 정확하다.

refresh_boundary_falloff(ASSIGNMENT) vs. refresh_falloff_state(STATE)
-----------------------------------------------------------------------
falloff row가 "어떤 Gaussian인지"(ASSIGNMENT: falloff_global_idx/edge_id/
sign/weight)와 "그 Gaussian이 지금 어디 있는지"(STATE: falloff_canonical_mean/
falloff_coefs, means/motion_coefs 자체를 gather한 스냅샷)는 갱신 빈도가
다르다:
  - ASSIGNMENT는 densify/cull이 foreground Gaussian 개수/순서를 바꿀 때만
    바뀐다 -- flow3d/trainer.py는 매 control_step()(densify/cull/
    bases-control이 실제로 개수를 바꿀 수 있는 지점) 끝에서
    motion_bases.num_fg_gaussians를 model.fg.num_gaussians와 비교해 다르면
    refresh_boundary_falloff를 호출한다. 이 메서드는 비싼 nearest-neighbor
    탐색(_compute_falloff_rows)을 포함한다.
  - STATE(means/motion_coefs 값 자체)는 densify/cull과 무관하게 매 학습
    step의 gradient로 계속 움직인다. flow3d/trainer.py는 (ASSIGNMENT가
    바뀌었는지와 무관하게) 매 step, forward/loss 계산 직전에
    refresh_falloff_state를 불러 이 스냅샷을 최신화한다 -- 순수 인덱싱이라
    (falloff row 수에 선형) 저렴하다. 이걸 게을리하면 GNN이 보는
    dist_norm/direction과 실제 렌더링에 쓰이는 위치가 다시 서서히 어긋난다.
refresh_boundary_falloff는 ASSIGNMENT를 다시 계산한 뒤 내부에서
refresh_falloff_state도 호출하므로, 트레이너는 둘 다 신경 쓸 필요 없이
"개수가 바뀌었으면 refresh_boundary_falloff, 그 외 매 step은
refresh_falloff_state"만 부르면 된다.

num_fg_gaussians는 고정 값이 아니라 refresh_boundary_falloff가 호출될 때마다
갱신되는 버퍼이므로, densify/cull 직후에도(refresh가 호출된 다음이라면)
compute_transforms의 전체-배열 크기 검사는 여전히 정확하다.

이 파일에서 제공하는 것
------------------------
- EdgeBoundarySets / load_edge_boundary_sets: build_cluster_graph.py의
  edges.pt를 읽어 edge별 (cluster_a, cluster_b, canonical anchor)를 만든다
  (anchor는 그 파일이 저장한 boundary Gaussian 집합의 canonical 평균 위치일
  뿐이므로, 만든 뒤에는 index를 더 이상 갖고 있지 않는다).
- EdgeBoundaryGNN: cluster graph 위 mean-aggregation node encoding +
  edge-level MLP head로 edge별 alpha_{ab}(t) in [0,1] (이 edge의
  max_displacement 중 당길 비율)를 예측하는 작은 GNN.
- EdgeBoundaryGraphCorrectedScalableMotionBases: ScalableMotionBases를
  상속해 compute_transforms에서만 (falloff-weighted, translation-only) 보정을
  끼워 넣는 드롭인 대체. compute_transforms_coarse는 건드리지 않는다(상속
  그대로). refresh_boundary_falloff(canonical_means, cluster_ids_all, coefs_all)
  로 densify/cull 이후 falloff row ASSIGNMENT를, refresh_falloff_state
  (canonical_means, coefs_all)로 (더 자주) STATE 스냅샷만 다시 계산한다.
- boundary_magnitude_reg_loss/boundary_magnitude_smoothness_loss: flow3d/
  trainer.py가 학습 경로에서 부르는 loss 함수 (correction magnitude 자체는
  render loss로 직접 학습되므로, 이 둘은 그 크기/시간 smoothness만 규제).
  compute_boundary_gap_distances/compute_boundary_gap_loss는 진단용 및
  opt-in loss(기본 꺼짐)로 남아 있다.
- load_edge_frame_gates: edges.pt에서 (edge_index, correction_gate)만 읽는
  경량 로더 (anchor/patch reference cloud 등 boundary 전용 정보는 만들지
  않는다). 다른 GNN variant가 CONNECTED/UNKNOWN/DISCONNECTED per-frame gate
  로직만 재사용하고 싶을 때 쓴다 (예:
  flow3d/graph_relative_linear_attention_frame.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.graph_coupling import CENTER_DIM, NODE_FEAT_DIM, ROT_DIM, TRANSL_DIM
from flow3d.params import ScalableMotionBases

__all__ = [
    "EdgeBoundarySets",
    "load_edge_boundary_sets",
    "load_edge_frame_gates",
    "EdgeBoundaryGNN",
    "EdgeBoundaryGraphCorrectedScalableMotionBases",
    "compute_boundary_gap_distances",
    "compute_boundary_gap_loss",
    "boundary_magnitude_reg_loss",
    "boundary_magnitude_smoothness_loss",
]

EDGE_MLP_EXTRA_DIM = TRANSL_DIM + CENTER_DIM + 2  # rel_transl(3) + rel_center(3) + dist_norm(1) + canonical_distance_norm(1)
DEFAULT_FALLOFF_MIN_WEIGHT = 1e-3
# 현재(post-densify/cull) Gaussian을 가장 가까운 frozen patch-reference point에
# "snap"할 때 쓰는 cutoff -- 그 reference point 자신의 local_scale(그 point가
# build_cluster_graph_mesh.py에서 만들어질 때의 로컬 밀도)의 몇 배까지 허용할지.
# 고정 falloff_radius 대신 밀도-적응형 cutoff를 쓰는 이유는 _compute_falloff_rows
# 참고.
_PATCH_SNAP_LOCAL_SCALE_RATIO = 2.0


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_long_indices(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


def _as_float_or_default(value: Any, default_length: int, default_fill: float) -> torch.Tensor:
    """`value` (a saved tensor/array, or None for a legacy edges.pt missing
    this field) -> a (default_length,) float32 CPU tensor, filled with
    `default_fill` when `value` is None."""
    if value is None:
        return torch.full((default_length,), default_fill, dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.float32).reshape(-1)
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


@dataclass
class EdgeBoundarySets:
    """edges.pt(build_cluster_graph_mesh.py, 또는 구형 build_cluster_graph.py)에서
    읽은, edge별 "boundary의 의미" (고정): topology, canonical anchor 위치,
    patch reference cloud(geodesic weight/local_scale 포함), per-frame
    correction gate/connected mask, contact_reference_distance. 만든 시점의
    Gaussian index는 이 값들을 계산하는 데만 잠깐 쓰이고 저장되지 않는다 --
    학습 중 densify/cull로 그 index들이 stale해져도 이 구조체 자체는 영향받지
    않는다 (모듈 docstring 참고).

    :param cluster_a / cluster_b: (E,) long, 무방향 edge의 두 cluster id.
    :param anchor_a_canonical / anchor_b_canonical: (E, 3), 각 side의 anchor
        위치 (contact core의 canonical 평균; contact core 정보가 없는 구형
        edges.pt는 boundary set 평균으로 대체).
    :param contact_reference_distance: (E,), boundary-gap loss의 허용 거리
        기준값 -- CONNECTED로 관찰된 실제(posed) 프레임들에서 양쪽 weighted
        patch centroid 사이 거리의 median. 없으면 canonical_distance로 대체.
    :param patch_ref_points/edge_id/sign/weight/local_scale: patch reference
        cloud, ragged (모든 kept edge의 양쪽 boundary patch를 이어붙인 것).
        sign: +1 = side a, -1 = side b (falloff_sign과 같은 convention).
    :param correction_gate: (E, num_frames), 0~1 연속값. CONNECTED=1,
        DISCONNECTED=0, UNKNOWN/미평가 구간은 선형 보간.
    :param connected_mask: (E, num_frames), bool. 정확히 그 edge의
        connected_frame_indices에 있는 프레임만 True.
    """

    cluster_a: torch.Tensor
    cluster_b: torch.Tensor
    anchor_a_canonical: torch.Tensor
    anchor_b_canonical: torch.Tensor
    contact_reference_distance: torch.Tensor
    patch_ref_points: torch.Tensor
    patch_ref_edge_id: torch.Tensor
    patch_ref_sign: torch.Tensor
    patch_ref_weight: torch.Tensor
    patch_ref_local_scale: torch.Tensor
    correction_gate: torch.Tensor
    connected_mask: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.cluster_a.shape[0])

    @property
    def canonical_distance(self) -> torch.Tensor:
        """(E,) ||anchor_a - anchor_b|| in canonical space -- GNN feature로만 쓰인다
        (boundary-gap loss의 허용 거리 기준은 contact_reference_distance)."""
        return (self.anchor_a_canonical - self.anchor_b_canonical).norm(dim=-1)


def _build_edge_gate(
    connected_frames: list[int], unknown_frames: list[int], num_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    한 edge의 CONNECTED/DISCONNECTED/UNKNOWN 프레임 목록(build_cluster_graph_
    mesh.py, 이제 매 프레임 판정됨 -- 모듈 docstring 참고)으로부터
    (correction_gate, connected_mask) (둘 다 (num_frames,))를 만든다.

    - disconnected = 전체 프레임 중 connected/unknown 어디에도 없는 프레임.
    - correction_gate: connected=1.0, disconnected=0.0인 매듭점들 사이(오직
      그 사이만)를 np.interp로 선형 보간한다. 매듭 구간 밖(맨 앞/뒤 -- 즉 한
      번도 CONNECTED/DISCONNECTED로 관찰되지 못한 채 시퀀스 끝까지 이어지는
      UNKNOWN)은 np.interp의 기본 동작(가장 가까운 매듭 값으로 flat하게 유지)
      대신 0(DISCONNECTED 취급)으로 둔다 -- 관찰 범위 밖까지 correction을
      계속 켜 두지 않기 위함. CONNECTED/DISCONNECTED가 단 한 번도 없어서
      매듭이 하나도 없으면(모든 프레임이 UNKNOWN) 전부 0으로 둔다 -- 접촉
      여부를 한 번도 관찰하지 못한 edge를 기본값 1(항상 correction 켜짐)로
      취급하지 않는다.
    - connected_mask: 정확히 connected_frames에 있는 프레임만 True (하드
      마스크 -- boundary-gap loss 전용, gate와는 별개).
    """
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
        # np.interp extrapolates FLAT outside [min(knot), max(knot)] -- correct
        # for a frame that IS a knot (every CONNECTED/DISCONNECTED frame is
        # trivially inside that range), but wrong for an UNKNOWN frame that
        # falls before the first or after the last observed knot: there is no
        # bracketing observation to interpolate between, so np.interp would
        # otherwise hold the correction on (or off) forever past the edge of
        # what was ever actually observed. Treat those as conservatively
        # DISCONNECTED (gate=0) instead.
        all_frames = np.arange(num_frames)
        outside_observed_range = (all_frames < knot_frames_arr[0]) | (all_frames > knot_frames_arr[-1])
        gate_np[outside_observed_range] = 0.0
    else:
        # No CONNECTED/DISCONNECTED observation exists at all for this edge
        # (every frame is UNKNOWN) -- there is nothing to interpolate between,
        # so don't default to "always on" (gate=1 everywhere); treat it as
        # never-trusted instead (gate=0 everywhere).
        gate_np = np.zeros(num_frames, dtype=np.float32)

    connected_mask_np = np.zeros(num_frames, dtype=bool)
    if connected_set:
        connected_mask_np[sorted(connected_set)] = True

    return torch.from_numpy(gate_np), torch.from_numpy(connected_mask_np)


def load_edge_boundary_sets(
    edges_path: str | Path,
    canonical_means: torch.Tensor,
    num_frames: int,
    device: torch.device | None = None,
) -> EdgeBoundarySets:
    """
    build_cluster_graph_mesh.py(권장) 또는 구형 build_cluster_graph.py의
    edges.pt를 읽어 EdgeBoundarySets를 만든다.

    :param edges_path: edges.pt 경로. "edges_kept" 리스트를 가진 dict여야 하며,
        각 원소는 최소 cluster_a, cluster_b, boundary_global_indices_a,
        boundary_global_indices_b를 가져야 한다. contact_core_global_indices_a/b,
        boundary_patch_weight_a/b, boundary_patch_local_scale_a/b,
        contact_reference_distance, connected_frame_indices,
        unknown_frame_indices는 없으면 각각 문서화된 legacy fallback으로
        대체된다 (구형 edges.pt도 오늘과 동일하게 동작).
    :param canonical_means: (G, 3) 저장된 index들이 가리키는, 이 함수를 호출하는
        시점의 canonical foreground Gaussian means (예:
        model.fg.params["means"].detach()) -- 이 함수 안에서만 쓰이고, 이후로는
        참조하지 않는다 (patch reference cloud는 이 시점의 위치를 frozen
        snapshot으로 저장한다).
    :param num_frames: 이 학습 실행의 전체 프레임 수 (예: bases.num_frames) --
        correction_gate/connected_mask의 크기를 결정한다.
    :param device: 반환 텐서들의 device. 기본값: canonical_means.device.
    :raises RuntimeError: 양쪽 다 non-empty인 boundary set을 가진 edge가 하나도
        없을 때.
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

    cluster_a_rows: list[int] = []
    cluster_b_rows: list[int] = []
    anchor_a_rows: list[torch.Tensor] = []
    anchor_b_rows: list[torch.Tensor] = []
    contact_ref_dist_rows: list[float] = []
    patch_points_rows: list[torch.Tensor] = []
    patch_edge_id_rows: list[torch.Tensor] = []
    patch_sign_rows: list[torch.Tensor] = []
    patch_weight_rows: list[torch.Tensor] = []
    patch_local_scale_rows: list[torch.Tensor] = []
    gate_rows: list[torch.Tensor] = []
    connected_mask_rows: list[torch.Tensor] = []

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

        core_a_raw = entry.get("contact_core_global_indices_a")
        core_b_raw = entry.get("contact_core_global_indices_b")
        core_idx_a = _as_long_indices(core_a_raw) if core_a_raw is not None else torch.zeros(0, dtype=torch.long)
        core_idx_b = _as_long_indices(core_b_raw) if core_b_raw is not None else torch.zeros(0, dtype=torch.long)
        if core_idx_a.numel() == 0:
            core_idx_a = idx_a
        if core_idx_b.numel() == 0:
            core_idx_b = idx_b

        anchor_a = means_cpu[core_idx_a].mean(dim=0)
        anchor_b = means_cpu[core_idx_b].mean(dim=0)
        anchor_a_rows.append(anchor_a)
        anchor_b_rows.append(anchor_b)

        contact_ref_dist = entry.get("contact_reference_distance")
        contact_ref_dist_rows.append(
            float(contact_ref_dist) if contact_ref_dist is not None
            else float((anchor_a - anchor_b).norm())
        )

        e = len(cluster_a_rows)
        cluster_a_rows.append(int(entry["cluster_a"]))
        cluster_b_rows.append(int(entry["cluster_b"]))

        # local_scale default fill is a negative sentinel (real local_scale is
        # always > 0) -- _compute_falloff_rows treats any ref point with
        # local_scale <= 0 as "unavailable" and falls back to the flat
        # falloff_radius cutoff for that point, so legacy edges.pt entries
        # missing this field degrade gracefully per-point, not per-edge.
        weight_a = _as_float_or_default(entry.get("boundary_patch_weight_a"), idx_a.shape[0], 1.0)
        weight_b = _as_float_or_default(entry.get("boundary_patch_weight_b"), idx_b.shape[0], 1.0)
        local_scale_a = _as_float_or_default(entry.get("boundary_patch_local_scale_a"), idx_a.shape[0], -1.0)
        local_scale_b = _as_float_or_default(entry.get("boundary_patch_local_scale_b"), idx_b.shape[0], -1.0)

        patch_points_rows.append(means_cpu[idx_a])
        patch_edge_id_rows.append(torch.full((idx_a.shape[0],), e, dtype=torch.long))
        patch_sign_rows.append(torch.full((idx_a.shape[0],), 1.0, dtype=torch.float32))
        patch_weight_rows.append(weight_a)
        patch_local_scale_rows.append(local_scale_a)

        patch_points_rows.append(means_cpu[idx_b])
        patch_edge_id_rows.append(torch.full((idx_b.shape[0],), e, dtype=torch.long))
        patch_sign_rows.append(torch.full((idx_b.shape[0],), -1.0, dtype=torch.float32))
        patch_weight_rows.append(weight_b)
        patch_local_scale_rows.append(local_scale_b)

        connected_frames = entry.get("connected_frame_indices")
        unknown_frames = entry.get("unknown_frame_indices")
        if connected_frames is not None and unknown_frames is not None:
            gate, connected_mask = _build_edge_gate(connected_frames, unknown_frames, num_frames)
        else:
            gate = torch.ones(num_frames, dtype=torch.float32)
            connected_mask = torch.ones(num_frames, dtype=torch.bool)
        gate_rows.append(gate)
        connected_mask_rows.append(connected_mask)

    if not cluster_a_rows:
        raise RuntimeError(
            f"No kept edge in {edges_path} had a usable boundary set on both "
            "sides -- cannot build any edge boundary correction."
        )

    return EdgeBoundarySets(
        cluster_a=torch.tensor(cluster_a_rows, dtype=torch.long, device=device),
        cluster_b=torch.tensor(cluster_b_rows, dtype=torch.long, device=device),
        anchor_a_canonical=torch.stack(anchor_a_rows, dim=0).to(device),
        anchor_b_canonical=torch.stack(anchor_b_rows, dim=0).to(device),
        contact_reference_distance=torch.tensor(contact_ref_dist_rows, dtype=torch.float32, device=device),
        patch_ref_points=torch.cat(patch_points_rows, dim=0).float().to(device),
        patch_ref_edge_id=torch.cat(patch_edge_id_rows, dim=0).to(device),
        patch_ref_sign=torch.cat(patch_sign_rows, dim=0).to(device),
        patch_ref_weight=torch.cat(patch_weight_rows, dim=0).to(device),
        patch_ref_local_scale=torch.cat(patch_local_scale_rows, dim=0).to(device),
        correction_gate=torch.stack(gate_rows, dim=0).to(device),
        connected_mask=torch.stack(connected_mask_rows, dim=0).to(device),
    )


def load_edge_frame_gates(
    edges_path: str | Path,
    num_clusters: int,
    num_frames: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """edges.pt를 읽어 (edge_index, correction_gate)를 함께 반환하는 경량
    로더 -- anchor/patch reference cloud/contact_reference_distance 등
    boundary-only 정보는 전혀 만들지 않는다 (다른 GNN variant, 예:
    flow3d/graph_relative_linear_attention_frame.py가 CONNECTED/UNKNOWN/
    DISCONNECTED per-frame gate만 재사용하기 위한 용도).

    edge_index와 correction_gate는 **같은 함수 안에서 같은 edges_kept 순회로
    함께** 만들어지므로 행 순서가 항상 정확히 대응한다 -- 호출하는 쪽은 이
    반환값 쌍을 topology/gate의 단일 기준(single source of truth)으로 쓰고,
    별도로 만든 edge_index(예: flow3d/graph_coupling.py's
    build_edge_index_from_edges_pt)와 행 순서를 맞추려 하면 안 된다:
    build_edge_index_from_edges_pt는 내부적으로 정렬/재정렬을 할 수 있어 같은
    edges.pt를 읽어도 이 함수의 행 순서와 다를 수 있다 (topology 집합 자체는
    항상 같다 -- 순서만 다를 수 있다는 뜻).

    :param edges_path: build_cluster_graph.py's edges.pt 경로. "edges_kept"
        리스트를 가진 dict여야 하며, 각 원소는 최소 cluster_a, cluster_b를
        가져야 한다 (connected_frame_indices/unknown_frame_indices는 없으면
        legacy fallback으로 대체된다, 아래 참고).
    :param num_clusters: 이 edge_index의 cluster id가 속해야 하는 범위
        (0..num_clusters-1).
    :param num_frames: 이 학습 실행의 전체 프레임 수.
    :param device: 반환 텐서들의 device. 기본값: CPU.
    :return: edge_index (2, E) long [cluster_a; cluster_b],
        correction_gate (E, num_frames) float -- _build_edge_gate와 동일한
        규칙(CONNECTED=1, DISCONNECTED=0, UNKNOWN은 관측된 매듭 사이만
        선형보간, 관측 범위 밖은 0). legacy edges.pt(connected_frame_indices/
        unknown_frame_indices 없음)는 전부 1.0(항상 connected)으로 채운다 --
        load_edge_boundary_sets의 legacy fallback과 동일한 규칙.
    :raises TypeError: edges_path가 build_cluster_graph.py 포맷이 아니면.
    :raises KeyError: 어떤 edges_kept 원소에 cluster_a/cluster_b가 없으면.
    :raises ValueError: cluster_a/cluster_b가 num_clusters 범위를 벗어나면.
    :raises RuntimeError: edges_kept가 비어 있으면.
    """
    payload = _torch_load(edges_path)
    if not isinstance(payload, dict) or "edges_kept" not in payload:
        raise TypeError(
            f"{edges_path} must be a build_cluster_graph.py edges.pt "
            "(a dict with an 'edges_kept' list)."
        )
    edges_kept = payload["edges_kept"]
    if not edges_kept:
        raise RuntimeError(f"{edges_path} has no edges_kept -- cannot build any edge gate.")

    cluster_a_rows: list[int] = []
    cluster_b_rows: list[int] = []
    gate_rows: list[torch.Tensor] = []
    for entry in edges_kept:
        if not isinstance(entry, dict):
            raise TypeError(f"Malformed edges_kept entry in {edges_path}: {entry!r}")
        if "cluster_a" not in entry or "cluster_b" not in entry:
            label = f"{entry.get('cluster_a')}-{entry.get('cluster_b')}"
            raise KeyError(f"edges_kept entry {label!r} in {edges_path} is missing cluster_a/cluster_b.")

        cluster_a = int(entry["cluster_a"])
        cluster_b = int(entry["cluster_b"])
        if not (0 <= cluster_a < num_clusters) or not (0 <= cluster_b < num_clusters):
            raise ValueError(
                f"{edges_path} references cluster ids ({cluster_a}, {cluster_b}), "
                f"but num_clusters={num_clusters}."
            )
        cluster_a_rows.append(cluster_a)
        cluster_b_rows.append(cluster_b)

        connected_frames = entry.get("connected_frame_indices")
        unknown_frames = entry.get("unknown_frame_indices")
        if connected_frames is not None and unknown_frames is not None:
            gate, _connected_mask = _build_edge_gate(connected_frames, unknown_frames, num_frames)
        else:
            gate = torch.ones(num_frames, dtype=torch.float32)
        gate_rows.append(gate)

    edge_index = torch.stack(
        [
            torch.tensor(cluster_a_rows, dtype=torch.long),
            torch.tensor(cluster_b_rows, dtype=torch.long),
        ],
        dim=0,
    ).to(device)
    correction_gate = torch.stack(gate_rows, dim=0).to(device)
    return edge_index, correction_gate


def _compute_falloff_rows(
    canonical_means: torch.Tensor,
    cluster_ids_all: torch.Tensor,
    edge_cluster_a: torch.Tensor,
    edge_cluster_b: torch.Tensor,
    patch_ref_points: torch.Tensor,
    patch_ref_edge_id: torch.Tensor,
    patch_ref_sign: torch.Tensor,
    patch_ref_weight: torch.Tensor,
    patch_ref_local_scale: torch.Tensor,
    falloff_radius: float,
    min_weight: float = DEFAULT_FALLOFF_MIN_WEIGHT,
) -> dict[str, torch.Tensor]:
    """
    edge/side마다, "그 side의 cluster에 속한 (지금 존재하는) Gaussian"을 그
    edge/side의 FROZEN patch reference cloud(build_cluster_graph_mesh.py가
    이미 계산해 저장한 진짜 geodesic falloff weight/local_scale을 가진,
    boundary patch의 canonical snapshot)에서 가장 가까운 점에 "snap"해서
    falloff weight를 얻는다 -- 반대편 cluster까지의 거리를 매번 다시 재는
    대신(과거 방식), 빌드 시점에 이미 계산된 진짜 geodesic weight를 그대로
    재사용한다 (모듈 docstring 참고). Snap은 nearest reference point 자신의
    local_scale의 `_PATCH_SNAP_LOCAL_SCALE_RATIO`배보다 멀면 일어나지 않는다
    (밀도-적응형 cutoff -- densify/cull 이후 Gaussian 밀도가 바뀌어도
    안정적, 고정 반경 하나보다 안전하다). local_scale이 없는(구형 edges.pt)
    reference point는 고정 falloff_radius를 cutoff로 대신 쓴다.

    :return: dict of 1-D tensors (모두 길이 M, falloff row 개수):
        "global_idx" (long, canonical_means/cluster_ids_all 기준),
        "edge_id" (long), "sign" (float, +1=side a / -1=side b),
        "weight" (float, (0, 1]). 아무 row도 없으면 모두 길이 0.
    """
    device = canonical_means.device
    cluster_ids_flat = cluster_ids_all.reshape(-1)
    num_edges = edge_cluster_a.shape[0]

    rows_idx: list[torch.Tensor] = []
    rows_edge: list[torch.Tensor] = []
    rows_sign: list[torch.Tensor] = []
    rows_weight: list[torch.Tensor] = []

    for e in range(num_edges):
        cluster_id_a = int(edge_cluster_a[e].item())
        cluster_id_b = int(edge_cluster_b[e].item())
        idx_a = (cluster_ids_flat == cluster_id_a).nonzero(as_tuple=True)[0]
        idx_b = (cluster_ids_flat == cluster_id_b).nonzero(as_tuple=True)[0]

        for idx_side, sign in ((idx_a, 1.0), (idx_b, -1.0)):
            if idx_side.numel() == 0:
                continue
            ref_mask = (patch_ref_edge_id == e) & (patch_ref_sign == sign)
            if not bool(ref_mask.any()):
                continue
            ref_points = patch_ref_points[ref_mask]
            ref_weight = patch_ref_weight[ref_mask]
            ref_local_scale = patch_ref_local_scale[ref_mask]

            dist = torch.cdist(canonical_means[idx_side], ref_points)  # (n_side, P)
            nearest_dist, nearest_pos = dist.min(dim=1)
            weight = ref_weight[nearest_pos]
            local_scale = ref_local_scale[nearest_pos]
            cutoff = torch.where(
                local_scale > 0,
                _PATCH_SNAP_LOCAL_SCALE_RATIO * local_scale,
                torch.full_like(local_scale, falloff_radius),
            )
            keep = (nearest_dist <= cutoff) & (weight > min_weight)
            if not bool(keep.any()):
                continue

            n_keep = int(keep.sum())
            rows_idx.append(idx_side[keep])
            rows_edge.append(torch.full((n_keep,), e, dtype=torch.long, device=device))
            rows_sign.append(torch.full((n_keep,), sign, dtype=torch.float32, device=device))
            rows_weight.append(weight[keep])

    if not rows_idx:
        return {
            "global_idx": torch.empty(0, dtype=torch.long, device=device),
            "edge_id": torch.empty(0, dtype=torch.long, device=device),
            "sign": torch.empty(0, dtype=torch.float32, device=device),
            "weight": torch.empty(0, dtype=torch.float32, device=device),
        }

    return {
        "global_idx": torch.cat(rows_idx),
        "edge_id": torch.cat(rows_edge),
        "sign": torch.cat(rows_sign),
        "weight": torch.cat(rows_weight),
    }


def _compute_edge_max_displacement(
    patch_ref_edge_id: torch.Tensor,
    patch_ref_local_scale: torch.Tensor,
    num_edges: int,
    falloff_radius: float,
    scale: float,
) -> torch.Tensor:
    """edge마다, correction magnitude의 절대 상한(max_displacement)을 그
    edge의 boundary patch reference cloud가 가진 국소 Gaussian spacing
    (patch_ref_local_scale, build_cluster_graph_mesh.py가 이미 계산해 저장한
    값)에서 유도한다 -- "boundary의 의미"(고정) 쪽이라 densify/cull에
    영향받지 않고, edges.pt를 만들 때 한 번만 계산하면 된다.

    gap_error 기반 설계와 달리 이 상한은 "지금 실제 gap 크기"가 아니라 이
    edge 근처 Gaussian들이 원래 얼마나 촘촘한지에서 나온다 -- 그래서 정적으로
    벌어져 있는 edge(현재 gap이 항상 크더라도)도 국소 spacing만큼씩은 당길 수
    있게 학습 가능하다. 대신 한 스텝에 국소 spacing의 `scale`배보다 크게
    당기지는 못하므로, render loss만으로 학습되는 correction이 한 번에
    비현실적으로 크게 움직이는 것을 막는 안전판 역할을 한다.

    :param patch_ref_edge_id: (P,) 각 patch reference point가 속한 edge id.
    :param patch_ref_local_scale: (P,) 각 patch reference point의 로컬 밀도
        스케일. 값이 <= 0이면(legacy edges.pt, local_scale 없음) "이 point는
        정보 없음"으로 취급.
    :param num_edges: E.
    :param falloff_radius: local_scale이 하나도 없는 edge에 대한 fallback.
    :param scale: local_scale(또는 fallback falloff_radius)에 곱하는 배수.
    :return: (E,) float32, 항상 > 0.
    """
    device = patch_ref_edge_id.device
    max_disp = torch.full((num_edges,), float(falloff_radius) * float(scale), dtype=torch.float32, device=device)
    for e in range(num_edges):
        local_scale_e = patch_ref_local_scale[patch_ref_edge_id == e]
        local_scale_e = local_scale_e[local_scale_e > 0]
        if local_scale_e.numel() > 0:
            max_disp[e] = local_scale_e.median() * float(scale)
    return max_disp


class _MeanAggLayer(nn.Module):
    """flow3d/graph_coupling.py's _MeanAggLayer와 비슷한 mean-aggregation
    message passing (1 layer, residual)이지만, adjacency가 고정이 아니라
    frame마다 바뀌는 (C, C, B) 텐서다 (EdgeBoundaryGNN._build_dynamic_adjacency
    참고 -- 비활성 edge가 이웃 edge의 node feature에까지 영향을 주지 않도록).
    cluster node hidden state를 만드는 데만 쓰이고, correction 자체는 이
    레이어가 아니라 EdgeBoundaryGNN의 edge_mlp가 출력한다.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """:param adj_norm: (C, C, B), row-normalized per frame."""
        msg = self.lin(h)  # (C, B, H)
        agg = torch.einsum("ijb,jbh->ibh", adj_norm, msg)
        return h + self.act(agg)


class EdgeBoundaryGNN(nn.Module):
    """cluster graph 위에서 mean-aggregation으로 node feature를 encode한 뒤,
    각 (무방향) edge마다 "이번 스텝에 이 edge의 max_displacement 중 몇 %를
    당길지" alpha_{ab}(t) in [0, 1]만 출력하는 작은 GNN. 방향은 여기서
    예측하지 않는다 (호출하는 쪽이 falloff-weighted patch centroid로
    기하학적으로 계산한다, _edge_features_and_direction 참고) -- 그래서 이
    GNN의 입력에는 전역 좌표계의 방향 정보(예: 3D 상대 위치 벡터)를 굳이 주지
    않는다. 대신 "지금 patch가 얼마나 떨어져 있는지"를 이 edge의
    max_displacement로 정규화한 무차원 스칼라(dist_norm)와, 참고용
    canonical_distance_norm(같은 정규화, "정답 목표"가 아니라 힌트일 뿐)만
    준다.

    alpha를 signed raw magnitude가 아니라 sigmoid로 [0, 1]에 bound된 "당길
    비율"로 두는 이유(호출하는 쪽이 magnitude = gate * max_displacement *
    alpha를 계산): (1) 항상 >= 0이라 부호가 뒤집혀 오히려 벌리는 방향으로
    움직일 수 없고(pull-only), (2) max_displacement(이 edge의 boundary 국소
    Gaussian spacing 기준으로 작게 잡힌 절대 상한, 아래
    EdgeBoundaryGraphCorrectedScalableMotionBases 참고)를 넘지 못해 한 스텝에
    과도하게 당겨지는 것을 막는다. 이전 설계는 magnitude를 gap_error =
    relu(dist_before - contact_reference_distance - gap_tolerance)에
    곱했는데, contact_reference_distance가 "같은 재구성 자신의 CONNECTED
    프레임 median"이라 정적으로(거의 항상 비슷하게) 벌어져 있는 edge에서는
    gap_error가 모든 프레임에서 정확히 0이 되어 render loss의 gradient가
    alpha까지 전혀 도달하지 못했다 (edge 15(4-21) 등, flow3d/analysis/
    gnn_check_edge.py로 실측). 이번 설계는 magnitude가 더 이상 gap_error에
    곱해지지 않으므로 이 문제가 없다 -- 대신 "실제 gap을 절대 못 넘는다"는
    기하학적 보장은 사라졌고, max_displacement를 작게 잡는 것과 render
    loss(RGB/depth/mask/track) 자체가 과도한 이동을 벌하는 것에 의존한다.

    edge_head(edge_mlp)의 마지막 Linear는 bias를 _INIT_LOGIT_BIAS(음수)로
    초기화한다: sigmoid(0)=0.5이므로 순수 zero-bias는 alpha=0.5, 즉 학습
    시작부터 매 edge의 max_displacement 절반을 즉시 당겨버리는 큰 correction이
    되어 버린다. bias를 음수로 시작하면 sigmoid(bias)가 0에 가까워져
    (_INIT_LOGIT_BIAS=-4.5 -> sigmoid(-4.5)=1.1e-2) 학습 시작 시점의
    correction이 (정확히 0은 아니지만) 작아진다 -- sigmoid는 정확히 0을 낼 수
    없으므로 이전 variant들의 "정확히 0" zero-init 등가성을 문자 그대로
    재현할 수는 없다. bias를 너무 음수로 두면(예: -8, sigmoid(-8)=3.4e-4) 그
    지점의 sigmoid gradient(sigmoid'(x)=sigmoid(x)(1-sigmoid(x)))도 같이
    작아져서 head가 사실상 거의 학습되지 않으므로, "거의 0에서 시작"과 "그래도
    학습 가능한 gradient" 사이 절충으로 -4.5를 쓴다.
    반면 weight는 정확히 0으로 초기화하지 않고 작은 랜덤값(std=0.01)으로
    초기화한다 -- weight가 정확히 0이면 d(logit)/d(edge_feat) = weight^T = 0
    이라, render loss의 gradient가 이 Linear 앞의 encoder/message-passing
    레이어(및 detach_base=False 경로의 base coarse motion)까지 전혀 못 흐르고
    오직 bias만 (자기 자신의 국소 gradient로) 학습되는 구간이 생긴다. 이번
    수정의 목적이 정확히 "render loss가 GNN 전체에 곧장 도달하게 하는 것"이라
    이 죽은 구간을 피하는 게 중요하다.
    """

    _INIT_LOGIT_BIAS = -4.5  # sigmoid(-4.5) ~= 1.1e-2 -- small enough to start "as good as zero", but sigmoid'(-4.5) ~= 1.09e-2 keeps the head's own gradient usable.
    _INIT_WEIGHT_STD = 0.01  # small, nonzero -- lets gradient reach the encoder/message-passing layers from step 0 instead of only the bias.

    def __init__(
        self,
        edge_cluster_a: torch.Tensor,
        edge_cluster_b: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.register_buffer("edge_cluster_a", edge_cluster_a.clone().long())
        self.register_buffer("edge_cluster_b", edge_cluster_b.clone().long())

        self.num_clusters = num_clusters
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList([_MeanAggLayer(hidden_dim) for _ in range(num_layers)])

        edge_feat_dim = 2 * hidden_dim + EDGE_MLP_EXTRA_DIM
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.edge_mlp[-1].weight, mean=0.0, std=self._INIT_WEIGHT_STD)
        nn.init.constant_(self.edge_mlp[-1].bias, self._INIT_LOGIT_BIAS)

    def _build_dynamic_adjacency(self, edge_gate: torch.Tensor) -> torch.Tensor:
        """(E, B) per-edge/frame correction gate -> (C, C, B) row-normalized
        adjacency: off-diagonal (a,b)/(b,a) entries equal to that edge's gate
        THIS frame, self-loop fixed at 1.0. An inactive (gate~0) edge this
        frame barely mixes its two clusters' node features together, instead
        of always fully mixing through a static graph regardless of whether
        the edge is currently trusted (module docstring, point 10)."""
        C = self.num_clusters
        B = edge_gate.shape[1]
        adj = edge_gate.new_zeros(C, C, B)
        a, b = self.edge_cluster_a, self.edge_cluster_b
        adj[a, b] = edge_gate
        adj[b, a] = edge_gate
        diag = torch.arange(C, device=edge_gate.device)
        adj[diag, diag] = 1.0
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)  # self-loop guarantees >= 1
        return adj / degree

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        dist_norm: torch.Tensor,
        canonical_distance_norm: torch.Tensor,
        edge_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param coarse_rot_6d: (C, B, 6)
        :param coarse_transl: (C, B, 3)
        :param centers: (C, 3)
        :param dist_norm: (E, B) dist_before / max_displacement -- the actual
            (coarse+fine blended, always detached by the caller) falloff-weighted
            patch-centroid distance for this edge, normalized by this edge's own
            max_displacement so the feature scale is comparable across edges.
        :param canonical_distance_norm: (E, B) canonical_distance / max_displacement,
            same normalization -- a REFERENCE feature only (the rest-pose anchor
            distance), never treated as the "correct" target distance.
        :param edge_gate: (E, B) this batch's per-frame correction gate -- drives the
            dynamic message-passing adjacency (see _build_dynamic_adjacency). The
            caller separately multiplies the returned alpha by this same gate.
        :return: alpha (E, B) in [0, 1] -- fraction of this edge's max_displacement
            to pull this step. The caller forms
            magnitude = edge_gate * max_displacement * alpha, which is always
            in [0, gate * max_displacement]: never negative (no sign flip,
            pull-only), and bounded by the fixed per-edge cap (not by the current
            gap size).
        """
        if coarse_rot_6d.shape[0] != self.num_clusters:
            raise ValueError(
                f"coarse_rot_6d has {coarse_rot_6d.shape[0]} clusters, expected {self.num_clusters}"
            )

        C, B, _ = coarse_rot_6d.shape
        center_feat = centers[:, None, :].expand(C, B, CENTER_DIM)
        node_feat = torch.cat([coarse_rot_6d, coarse_transl, center_feat], dim=-1)  # (C, B, 12)

        adj = self._build_dynamic_adjacency(edge_gate)
        h = self.encoder(node_feat)
        for layer in self.layers:
            h = layer(h, adj)

        a, b = self.edge_cluster_a, self.edge_cluster_b
        h_a, h_b = h[a], h[b]  # (E, B, H) each
        rel_transl = coarse_transl[b] - coarse_transl[a]  # (E, B, 3)
        rel_center = (centers[b] - centers[a])[:, None, :].expand(-1, B, -1)  # (E, B, 3)
        dist_feat = dist_norm[..., None]  # (E, B, 1)
        canonical_feat = canonical_distance_norm[..., None]  # (E, B, 1)

        edge_feat = torch.cat([h_a, h_b, rel_transl, rel_center, dist_feat, canonical_feat], dim=-1)
        logit = self.edge_mlp(edge_feat).squeeze(-1)  # (E, B)
        alpha = torch.sigmoid(logit)
        return alpha


class EdgeBoundaryGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + per-edge boundary-only translation correction.

    compute_transforms_coarse는 상속 그대로(오버라이드하지 않음) -- 이 클래스의
    correction은 "coarse skeleton"이 아니라 실제 렌더되는(coarse+fine blended)
    위치에만 적용된다. compute_transforms만 오버라이드해서, 부모의 결과에
    falloff-weighted translation-only displacement를 더한다.

    subset-invariant하지 않다: displacement가 개별 Gaussian identity(=falloff
    weight)에 의존하므로, compute_transforms(ts, coefs, cluster_ids)는
    cluster_ids가 정확히 전체 canonical foreground 배열(순서 그대로, 길이
    num_fg_gaussians)일 때만 지원된다 -- flow3d/trainer.py가 실제로 호출하는
    방식(inds=None)과 일치한다. 다른 subset으로 호출하면 ValueError.

    densify/cull로 foreground Gaussian 개수/순서가 바뀌면
    refresh_boundary_falloff(canonical_means, cluster_ids_all, coefs_all)를
    호출해야 한다 -- edge topology/anchor("boundary의 의미")는 그대로 두고,
    falloff row ASSIGNMENT("지금 boundary에 속한 Gaussian이 누구인지")와 그
    STATE 스냅샷(falloff_canonical_mean/falloff_coefs)을 함께 다시 계산한다.
    Gaussian 개수/순서가 그대로인 매 학습 step에는 더 가벼운
    refresh_falloff_state(canonical_means, coefs_all)만 불러 STATE 스냅샷을
    최신화하면 된다 (ASSIGNMENT는 그대로 두고 값만 다시 gather).
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
        anchor_a_canonical: torch.Tensor,
        anchor_b_canonical: torch.Tensor,
        contact_reference_distance: torch.Tensor,
        patch_ref_points: torch.Tensor,
        patch_ref_edge_id: torch.Tensor,
        patch_ref_sign: torch.Tensor,
        patch_ref_weight: torch.Tensor,
        patch_ref_local_scale: torch.Tensor,
        correction_gate: torch.Tensor,
        connected_mask: torch.Tensor,
        falloff_global_idx: torch.Tensor,
        falloff_edge_id: torch.Tensor,
        falloff_sign: torch.Tensor,
        falloff_weight: torch.Tensor,
        falloff_canonical_mean: torch.Tensor,
        falloff_coefs: torch.Tensor,
        max_displacement: torch.Tensor,
        num_fg_gaussians: int,
        falloff_radius: float,
        gap_tolerance: float,
        falloff_min_weight: float = DEFAULT_FALLOFF_MIN_WEIGHT,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 1,
    ):
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        num_clusters = self.num_clusters

        # Fixed ("boundary의 의미"): never touched by refresh_boundary_falloff.
        self.register_buffer("edge_cluster_a", edge_cluster_a.clone().long())
        self.register_buffer("edge_cluster_b", edge_cluster_b.clone().long())
        self.register_buffer("anchor_a_canonical", anchor_a_canonical.clone().float())
        self.register_buffer("anchor_b_canonical", anchor_b_canonical.clone().float())
        self.register_buffer("contact_reference_distance", contact_reference_distance.clone().float())
        self.register_buffer("patch_ref_points", patch_ref_points.clone().float())
        self.register_buffer("patch_ref_edge_id", patch_ref_edge_id.clone().long())
        self.register_buffer("patch_ref_sign", patch_ref_sign.clone().float())
        self.register_buffer("patch_ref_weight", patch_ref_weight.clone().float())
        self.register_buffer("patch_ref_local_scale", patch_ref_local_scale.clone().float())
        self.register_buffer("correction_gate", correction_gate.clone().float())
        self.register_buffer("connected_mask", connected_mask.clone().bool())
        self.register_buffer("falloff_radius", torch.tensor(float(falloff_radius)))
        self.register_buffer("falloff_min_weight", torch.tensor(float(falloff_min_weight)))
        # gap_tolerance/contact_reference_distance: no longer used to gate the
        # trained correction (see _edge_features_and_direction) -- kept only
        # for compute_boundary_gap_loss/compute_boundary_gap_distances, which
        # stay available for diagnostics (flow3d/analysis/gnn_check_edge.py)
        # and as an opt-in loss (trainer.py's w_boundary_gap, default 0.0).
        self.register_buffer("gap_tolerance", torch.tensor(float(gap_tolerance)))
        # Absolute cap on |correction| per edge, derived once (by the caller --
        # from_scalable_motion_bases/init_from_state_dict, see
        # _compute_edge_max_displacement) from this edge's boundary patch's own
        # local Gaussian spacing. Fixed ("boundary의 의미"): doesn't depend on
        # the current (live) gap, so a persistently-open edge can still be
        # corrected up to this cap even though gap_error-based gating would see
        # it as "always been this way, nothing to fix".
        self.register_buffer("max_displacement", max_displacement.clone().float())

        # Live ("지금 boundary에 속한 Gaussian"): replaced wholesale by
        # refresh_boundary_falloff whenever foreground Gaussian count/order changes.
        self.register_buffer("falloff_global_idx", falloff_global_idx.clone().long())
        self.register_buffer("falloff_edge_id", falloff_edge_id.clone().long())
        self.register_buffer("falloff_sign", falloff_sign.clone().float())
        self.register_buffer("falloff_weight", falloff_weight.clone().float())
        # Per-falloff-row canonical mean / fine-basis coefs, snapshotted as of
        # the last refresh_falloff_state (or refresh_boundary_falloff, which
        # calls it) call -- lets _falloff_side_means/_falloff_row_positions
        # recompute the ACTUAL (coarse+fine blended) current position of every
        # boundary-adjacent Gaussian without needing canonical_means/coefs_all
        # threaded through every compute_transforms call. See module docstring
        # ("refresh_boundary_falloff vs. refresh_falloff_state") -- the STATE
        # half (this buffer) is meant to be refreshed every training step
        # (cheap, no nearest-neighbor search), only the ASSIGNMENT half
        # (falloff_global_idx etc. above) is tied to densify/cull.
        self.register_buffer("falloff_canonical_mean", falloff_canonical_mean.clone().float())
        self.register_buffer("falloff_coefs", falloff_coefs.clone().float())
        self.register_buffer(
            "num_fg_gaussians", torch.tensor(int(num_fg_gaussians), dtype=torch.long)
        )

        self.gnn = EdgeBoundaryGNN(
            self.edge_cluster_a,
            self.edge_cluster_b,
            num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
        )
        self._last_boundary_correction: dict[str, torch.Tensor] | None = None

    @property
    def canonical_distance(self) -> torch.Tensor:
        """(E,) ||anchor_a - anchor_b|| in canonical space."""
        return (self.anchor_a_canonical - self.anchor_b_canonical).norm(dim=-1)

    @property
    def num_edges(self) -> int:
        return int(self.edge_cluster_a.shape[0])

    @property
    def last_boundary_correction(self) -> dict[str, torch.Tensor] | None:
        """가장 최근 compute_transforms 호출에서 나온
        {"magnitude": (E, B), "gate": (E, B), "alpha": (E, B)} (디버깅/로깅/loss용)."""
        return self._last_boundary_correction

    @torch.no_grad()
    def refresh_boundary_falloff(
        self, canonical_means: torch.Tensor, cluster_ids_all: torch.Tensor, coefs_all: torch.Tensor
    ) -> None:
        """EXPENSIVE: densify/cull(flow3d/trainer.py's _densify_control_step/
        _cull_control_step/_bases_control_step) 이후, foreground Gaussian
        배열이 바뀐 CURRENT canonical_means/cluster_ids_all로 falloff row
        ASSIGNMENT(어떤 Gaussian이 어떤 edge/side에 속하는지, 및 그 weight --
        nearest-neighbor 탐색이 포함된 _compute_falloff_rows)와
        num_fg_gaussians를 다시 계산해 덮어쓴다. edge_cluster_a/b와
        anchor_a/b_canonical("boundary의 의미")는 건드리지 않는다 -- 옛 index를
        전혀 참조하지 않으므로 densify/cull이 몇 번 일어났든 항상 처음부터
        다시 계산 가능하다.

        ASSIGNMENT(이 메서드) vs. STATE(refresh_falloff_state)를 분리한
        이유: falloff row가 "누구인지"는 densify/cull이 Gaussian 개수/순서를
        바꿀 때만 바뀌므로 비싼 nearest-neighbor 탐색은 그때만 하면 되지만,
        그 Gaussian들의 실제 means/motion_coefs 값 자체는 매 학습 step의
        gradient로 계속 움직인다 -- refresh_falloff_state가 그 가벼운 부분을
        맡는다. 이 메서드는 끝에서 refresh_falloff_state를 호출해 스냅샷도
        함께 최신화한다 (assignment가 바뀌었으니 당연히 다시 gather해야 한다).

        :param canonical_means: (G, 3) 현재 canonical foreground Gaussian means
            (예: model.fg.params["means"].detach()).
        :param cluster_ids_all: (G,) 현재 foreground Gaussian cluster id, 같은
            순서 (예: model.fg.get_cluster_ids().reshape(-1).long()).
        :param coefs_all: (G, F) 현재 foreground Gaussian fine-basis blend weight,
            같은 순서 (예: model.fg.get_coefs().detach()).
        """
        device = self.falloff_global_idx.device
        falloff = _compute_falloff_rows(
            canonical_means,
            cluster_ids_all,
            self.edge_cluster_a,
            self.edge_cluster_b,
            self.patch_ref_points,
            self.patch_ref_edge_id,
            self.patch_ref_sign,
            self.patch_ref_weight,
            self.patch_ref_local_scale,
            float(self.falloff_radius.item()),
            float(self.falloff_min_weight.item()),
        )
        self.falloff_global_idx = falloff["global_idx"].to(device)
        self.falloff_edge_id = falloff["edge_id"].to(device)
        self.falloff_sign = falloff["sign"].to(device)
        self.falloff_weight = falloff["weight"].to(device)
        self.num_fg_gaussians = torch.tensor(
            int(canonical_means.shape[0]), dtype=torch.long, device=device
        )
        self.refresh_falloff_state(canonical_means, coefs_all)

    @torch.no_grad()
    def refresh_falloff_state(self, canonical_means: torch.Tensor, coefs_all: torch.Tensor) -> None:
        """CHEAP: falloff_global_idx(누가 boundary에 속하는지 -- ASSIGNMENT,
        refresh_boundary_falloff가 densify/cull 때만 바꿈)는 그대로 두고,
        그 Gaussian들의 CURRENT canonical_means/coefs_all 값만 다시 gather해
        falloff_canonical_mean/falloff_coefs 스냅샷을 갱신한다. Nearest-
        neighbor 탐색(_compute_falloff_rows)이 전혀 없는 순수 인덱싱이라
        (falloff row 수 M에 선형) 매 학습 step, 또는 최소한 forward 직전에
        불러도 저렴하다 -- means/motion_coefs 자체는 densify/cull과 무관하게
        매 step 자기 gradient로 계속 움직이므로, 이 스냅샷을 오래(트레이너의
        control_step 주기만큼) 방치하면 GNN이 보는 gap_error/direction과 실제
        렌더링에 쓰이는 위치가 다시 서서히 어긋난다.

        :param canonical_means: (G, 3) 현재 canonical foreground Gaussian means.
        :param coefs_all: (G, F) 현재 foreground Gaussian fine-basis blend weight,
            같은 순서.
        """
        idx = self.falloff_global_idx
        if idx.numel() == 0:
            return
        device = idx.device
        self.falloff_canonical_mean = canonical_means[idx].detach().float().to(device)
        self.falloff_coefs = coefs_all[idx].detach().float().to(device)

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        edges_path: str | Path,
        canonical_means: torch.Tensor,
        cluster_ids_all: torch.Tensor,
        coefs_all: torch.Tensor,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 1,
        falloff_radius: float = 0.05,
        falloff_min_weight: float = DEFAULT_FALLOFF_MIN_WEIGHT,
        gap_tolerance: float = 0.02,
        correction_max_disp_scale: float = 2.0,
    ) -> "EdgeBoundaryGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 edge-boundary-corrected
        버전을 만든다. coarse/fine motion 파라미터 값은 그대로 복사되고,
        correction만 새로 추가된다 (거의 0으로 초기화됨 -- EdgeBoundaryGNN의
        _INIT_LOGIT_BIAS 참고, sigmoid라 정확히 0은 아니다).

        :param edge_index: 다른 *GraphCorrectedScalableMotionBases와
            call-site를 맞추기 위해 받지만 사용하지 않는다 -- edge topology는
            edges_path의 cluster_a/cluster_b에서 그대로 다시 읽는다 (같은
            파일이므로 edge_index와 항상 일치한다).
        :param edges_path: build_cluster_graph.py's edges.pt 경로.
        :param canonical_means: (G, 3) canonical foreground Gaussian means
            (예: fg_params.params["means"].detach()).
        :param cluster_ids_all: (G,) canonical foreground Gaussian cluster id
            (예: fg_params.get_cluster_ids().reshape(-1).long()).
        :param coefs_all: (G, F) canonical foreground Gaussian fine-basis blend
            weight, 같은 순서 (예: fg_params.get_coefs().detach()) -- falloff
            Gaussian의 실제(coarse+fine blended) 현재 위치를 재계산하는 데 쓰인다
            (refresh_boundary_falloff/_falloff_side_means 참고).
        :param falloff_radius: 구형 edges.pt(local_scale 없음)에 대한 legacy
            flat snap-cutoff (scene 단위). 새 edges.pt는 patch reference
            point마다 저장된 local_scale 기반 cutoff를 쓴다 (모듈 docstring,
            _compute_falloff_rows 참고).
        :param gap_tolerance: compute_boundary_gap_loss(진단/opt-in loss 전용,
            기본적으로 학습 경로에서는 꺼져 있음 -- trainer.py의 w_boundary_gap
            기본값 0.0)의 hinge threshold eps. 더 이상 GNN 입력/correction
            magnitude 계산에는 쓰이지 않는다 (_edge_features_and_direction
            참고).
        :param correction_max_disp_scale: correction magnitude의 절대 상한
            (max_displacement)을 만들 때 이 edge의 boundary patch 국소 Gaussian
            spacing(patch_ref_local_scale)에 곱하는 배수. 크게 잡을수록 한
            스텝에 더 많이 당길 수 있지만, render loss만으로 학습되는 correction
            이라 너무 크게 잡으면 과도한 이동의 안전판이 약해진다.
        """
        del edge_index
        num_frames = bases.num_frames
        boundary_sets = load_edge_boundary_sets(edges_path, canonical_means, num_frames)
        falloff = _compute_falloff_rows(
            canonical_means,
            cluster_ids_all,
            boundary_sets.cluster_a,
            boundary_sets.cluster_b,
            boundary_sets.patch_ref_points,
            boundary_sets.patch_ref_edge_id,
            boundary_sets.patch_ref_sign,
            boundary_sets.patch_ref_weight,
            boundary_sets.patch_ref_local_scale,
            falloff_radius,
            falloff_min_weight,
        )
        if falloff["global_idx"].numel() == 0:
            raise RuntimeError(
                f"No boundary falloff rows were produced from {edges_path} -- "
                f"falloff_radius ({falloff_radius}) may be too small relative to the scene scale."
            )

        max_displacement = _compute_edge_max_displacement(
            boundary_sets.patch_ref_edge_id,
            boundary_sets.patch_ref_local_scale,
            boundary_sets.num_edges,
            falloff_radius,
            correction_max_disp_scale,
        )

        p = bases.params
        return cls(
            centers=p["centers"].detach().clone(),
            rots=p["rots"].detach().clone(),
            transls=p["transls"].detach().clone(),
            fine_rots=p["fine_rots"].detach().clone(),
            fine_transls=p["fine_transls"].detach().clone(),
            edge_cluster_a=boundary_sets.cluster_a,
            edge_cluster_b=boundary_sets.cluster_b,
            anchor_a_canonical=boundary_sets.anchor_a_canonical,
            anchor_b_canonical=boundary_sets.anchor_b_canonical,
            contact_reference_distance=boundary_sets.contact_reference_distance,
            patch_ref_points=boundary_sets.patch_ref_points,
            patch_ref_edge_id=boundary_sets.patch_ref_edge_id,
            patch_ref_sign=boundary_sets.patch_ref_sign,
            patch_ref_weight=boundary_sets.patch_ref_weight,
            patch_ref_local_scale=boundary_sets.patch_ref_local_scale,
            correction_gate=boundary_sets.correction_gate,
            connected_mask=boundary_sets.connected_mask,
            falloff_global_idx=falloff["global_idx"],
            falloff_edge_id=falloff["edge_id"],
            falloff_sign=falloff["sign"],
            falloff_weight=falloff["weight"],
            falloff_canonical_mean=canonical_means[falloff["global_idx"]],
            falloff_coefs=coefs_all[falloff["global_idx"]],
            max_displacement=max_displacement,
            num_fg_gaussians=canonical_means.shape[0],
            falloff_radius=falloff_radius,
            gap_tolerance=gap_tolerance,
            falloff_min_weight=falloff_min_weight,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
        )

    @staticmethod
    def has_gnn_state(state_dict: dict, prefix: str) -> bool:
        """state_dict의 motion_bases가 EdgeBoundaryGraphCorrectedScalableMotionBases로
        저장된 것인지 확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은
        형태여야 한다 (params.가 아니라). edge_cluster_a는 이 클래스에만 있는
        top-level buffer라서, 다른 *GraphCorrectedScalableMotionBases의
        "gnn." state와 구별하는 마커로 쓴다."""
        return f"{prefix}edge_cluster_a" in state_dict

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "EdgeBoundaryGraphCorrectedScalableMotionBases":
        """체크포인트만으로 통째로 복원한다. edges_path/canonical_means 없이
        저장된 buffer들에서 shape을 읽어 재구성하고, load_state_dict가 실제
        값으로 덮어쓴다.

        :raises KeyError: patch_ref_points/correction_gate/connected_mask 중
            하나라도 없으면 -- 이 스키마 이전에 저장된 체크포인트는 opposite-
            cluster distance 기반 falloff와 per-frame gating 없이 학습됐으므로
            여기서 안전하게 복원할 방법이 없다 (조용히 degrade하는 대신 즉시
            실패). 이런 체크포인트는 재개(resume)가 아니라
            from_scalable_motion_bases로 새로 초기화해서 다시 학습해야 한다.
        """
        base = ScalableMotionBases.init_from_state_dict(state_dict, prefix=f"{prefix}params.")

        if f"{prefix}edge_cluster_a" not in state_dict:
            raise KeyError(f"No '{prefix}edge_cluster_a' buffer found in state_dict.")

        required_new_keys = (
            f"{prefix}patch_ref_points",
            f"{prefix}correction_gate",
            f"{prefix}connected_mask",
        )
        missing_new_keys = [key for key in required_new_keys if key not in state_dict]
        if missing_new_keys:
            raise KeyError(
                f"state_dict is missing {missing_new_keys} -- this checkpoint predates the "
                "patch-reference/frame-gating schema (contact core, boundary patch, per-frame "
                "CONNECTED/DISCONNECTED/UNKNOWN state). It cannot be resumed through "
                "init_from_state_dict; reinitialize the GNN via from_scalable_motion_bases "
                "against a rebuilt build_cluster_graph_mesh.py edges.pt instead."
            )

        # falloff_canonical_mean/falloff_coefs/gap_tolerance: added when the GNN
        # was switched from a coarse-only anchor direction/gap to the actual
        # falloff-weighted patch-centroid distance (_falloff_side_means) -- a
        # checkpoint trained before that change has no way to recover the
        # per-Gaussian fine-basis coefs it needs, same reasoning as above.
        required_patch_position_keys = (
            f"{prefix}falloff_canonical_mean",
            f"{prefix}falloff_coefs",
            f"{prefix}gap_tolerance",
        )
        missing_patch_position_keys = [key for key in required_patch_position_keys if key not in state_dict]
        if missing_patch_position_keys:
            raise KeyError(
                f"state_dict is missing {missing_patch_position_keys} -- this checkpoint predates the "
                "falloff-weighted patch-centroid gap/direction schema (the GNN used to see a "
                "coarse-only anchor direction that could disagree with what compute_boundary_gap_loss "
                "actually measures). It cannot be resumed through init_from_state_dict; reinitialize "
                "the GNN via from_scalable_motion_bases instead."
            )

        # max_displacement: added when correction magnitude stopped being capped
        # by gap_error (dist_before - contact_reference_distance - gap_tolerance)
        # and started being capped by a fixed per-edge absolute displacement
        # instead -- a checkpoint trained before that change has no such cap
        # stored anywhere recoverable.
        if f"{prefix}max_displacement" not in state_dict:
            raise KeyError(
                f"state_dict is missing '{prefix}max_displacement' -- this checkpoint predates the "
                "gap_error-free correction schema (magnitude = gate * max_displacement * sigmoid(...), "
                "no longer gated by contact_reference_distance). It cannot be resumed through "
                "init_from_state_dict; reinitialize the GNN via from_scalable_motion_bases instead."
            )

        gnn_prefix = f"{prefix}gnn."
        gnn_keys = [key for key in state_dict if key.startswith(gnn_prefix)]
        if not gnn_keys:
            raise KeyError(f"No '{gnn_prefix}*' keys found in state_dict.")

        hidden_dim = state_dict[f"{gnn_prefix}encoder.0.weight"].shape[0]
        layers_prefix = f"{gnn_prefix}layers."
        layer_indices = {
            int(key[len(layers_prefix):].split(".", 1)[0])
            for key in gnn_keys
            if key.startswith(layers_prefix)
        }
        num_layers = max(layer_indices) + 1

        edge_cluster_a = state_dict[f"{prefix}edge_cluster_a"]
        edge_cluster_b = state_dict[f"{prefix}edge_cluster_b"]
        anchor_a_canonical = state_dict[f"{prefix}anchor_a_canonical"]
        anchor_b_canonical = state_dict[f"{prefix}anchor_b_canonical"]
        contact_reference_distance = state_dict[f"{prefix}contact_reference_distance"]
        patch_ref_points = state_dict[f"{prefix}patch_ref_points"]
        patch_ref_edge_id = state_dict[f"{prefix}patch_ref_edge_id"]
        patch_ref_sign = state_dict[f"{prefix}patch_ref_sign"]
        patch_ref_weight = state_dict[f"{prefix}patch_ref_weight"]
        patch_ref_local_scale = state_dict[f"{prefix}patch_ref_local_scale"]
        correction_gate = state_dict[f"{prefix}correction_gate"]
        connected_mask = state_dict[f"{prefix}connected_mask"]
        falloff_global_idx = state_dict[f"{prefix}falloff_global_idx"]
        falloff_edge_id = state_dict[f"{prefix}falloff_edge_id"]
        falloff_sign = state_dict[f"{prefix}falloff_sign"]
        falloff_weight = state_dict[f"{prefix}falloff_weight"]
        falloff_canonical_mean = state_dict[f"{prefix}falloff_canonical_mean"]
        falloff_coefs = state_dict[f"{prefix}falloff_coefs"]
        max_displacement = state_dict[f"{prefix}max_displacement"]
        num_fg_gaussians = int(state_dict[f"{prefix}num_fg_gaussians"].item())
        falloff_radius = float(state_dict[f"{prefix}falloff_radius"].item())
        gap_tolerance = float(state_dict[f"{prefix}gap_tolerance"].item())
        falloff_min_weight = float(
            state_dict.get(f"{prefix}falloff_min_weight", torch.tensor(DEFAULT_FALLOFF_MIN_WEIGHT)).item()
        )

        graph_bases = cls(
            centers=base.params["centers"].detach().clone(),
            rots=base.params["rots"].detach().clone(),
            transls=base.params["transls"].detach().clone(),
            fine_rots=base.params["fine_rots"].detach().clone(),
            fine_transls=base.params["fine_transls"].detach().clone(),
            edge_cluster_a=edge_cluster_a,
            edge_cluster_b=edge_cluster_b,
            anchor_a_canonical=anchor_a_canonical,
            anchor_b_canonical=anchor_b_canonical,
            contact_reference_distance=contact_reference_distance,
            patch_ref_points=patch_ref_points,
            patch_ref_edge_id=patch_ref_edge_id,
            patch_ref_sign=patch_ref_sign,
            patch_ref_weight=patch_ref_weight,
            patch_ref_local_scale=patch_ref_local_scale,
            correction_gate=correction_gate,
            connected_mask=connected_mask,
            falloff_global_idx=falloff_global_idx,
            falloff_edge_id=falloff_edge_id,
            falloff_sign=falloff_sign,
            falloff_weight=falloff_weight,
            falloff_canonical_mean=falloff_canonical_mean,
            falloff_coefs=falloff_coefs,
            max_displacement=max_displacement,
            num_fg_gaussians=num_fg_gaussians,
            falloff_radius=falloff_radius,
            gap_tolerance=gap_tolerance,
            falloff_min_weight=falloff_min_weight,
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
        )

        own_state = {
            key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        graph_bases.load_state_dict(own_state, strict=True)
        return graph_bases

    def _falloff_row_positions(
        self, ts: torch.Tensor, detach: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """현재 살아있는 모든 falloff row(M = falloff_global_idx.numel())의
        실제(coarse+fine blended) 현재 위치를, compute_transforms가 렌더링에
        실제로 쓰는 것과 동일한 ScalableMotionBases.compute_transforms
        수식으로 계산한다 (correction 적용 *전*). _falloff_side_means(집계된
        edge/side별 평균만 필요할 때)와 compute_boundary_gap_distances(집계
        *전* row 단위 위치가 필요할 때 -- 아래 참고)가 공유하는 최하위 빌딩
        블록.

        :param detach: True면 transform 자체를 detach한다 (direction/dist_norm은
            학습 대상이 아닌 "지금 상태" 신호이므로 호출하는 쪽은 항상
            detach=True로 부른다).
        :return: positions (M, B, 3), group_ids (M,) long
            (= falloff_edge_id*2 + (0 if side a else 1), edge/side별 집계용),
            weight (M,) = falloff_weight.
        """
        cluster_id = torch.where(
            self.falloff_sign > 0,
            self.edge_cluster_a[self.falloff_edge_id],
            self.edge_cluster_b[self.falloff_edge_id],
        )  # (M,)
        base_transforms = ScalableMotionBases.compute_transforms(
            self, ts, self.falloff_coefs, cluster_id
        )  # (M, B, 3, 4)
        if detach:
            base_transforms = base_transforms.detach()
        positions = _apply_transform(base_transforms, self.falloff_canonical_mean)  # (M, B, 3)
        side = (self.falloff_sign < 0).long()  # 0 = side a, 1 = side b
        group_ids = self.falloff_edge_id * 2 + side  # (M,)
        return positions, group_ids, self.falloff_weight

    def _falloff_side_means(
        self, ts: torch.Tensor, detach: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """edge/side별 falloff-weighted 평균 위치 (correction 적용 *전*,
        _falloff_row_positions의 결과를 falloff_weight로 가중 평균).

        _edge_features_and_direction(GNN이 보는 gap/direction)과
        compute_boundary_gap_loss(실제 hinge loss)가 **_falloff_row_positions
        하나**를 공유하도록 만든 게 이 리팩터의 핵심이다 -- 이전에는 GNN 입력이
        coarse-only anchor rigid transform(_coarse_rigid_transform)에서, loss의
        dist_after 판정은 falloff-weighted 실제 위치에서 각각 따로 계산되어
        둘이 가리키는 "지금 gap"이 서로 다를 수 있었다 (당겨도 loss가 재는
        gap이 줄어든다는 보장이 없었다).

        correction 적용 *후*(dist_after)는 이 메서드로 계산하면 안 된다 --
        "평균이 정확히 0.5*m*dir만큼 이동한다"는 가정은 falloff_weight가
        Gaussian마다 다르면(가중 평균이라 비선형) 성립하지 않고, 한 Gaussian이
        여러 edge의 falloff row에 동시에 걸리면(그 Gaussian이 속한 cluster가
        여러 edge에 인접) 다른 edge의 correction까지 더해져야 한다 --
        compute_boundary_gap_distances가 이 두 가지를 반영해 별도로 계산한다.

        :param detach: _falloff_row_positions로 그대로 전달.
        :return: mean_a, mean_b (E, B, 3) falloff-weighted 평균 위치,
            has_both_sides (E,) bool -- 그 edge가 지금 양쪽 다 falloff row를
            갖는지 (한쪽이라도 비면 평균이 정의되지 않으므로 0으로 채워지고
            has_both_sides=False).
        """
        idx = self.falloff_global_idx  # (M,)
        E, B = self.num_edges, ts.shape[0]
        if idx.numel() == 0:
            zeros = ts.new_zeros(E, B, 3, dtype=torch.float32)
            return zeros, zeros.clone(), torch.zeros(E, dtype=torch.bool, device=idx.device)

        positions, group_ids, weight = self._falloff_row_positions(ts, detach=detach)

        weight_sums = positions.new_zeros(E * 2)
        pos_sums = positions.new_zeros(E * 2, B, 3)
        weight_sums.index_add_(0, group_ids, weight)
        pos_sums.index_add_(0, group_ids, positions * weight[:, None, None])
        means = pos_sums / weight_sums[:, None, None].clamp_min(1e-8)
        mean_a = means[0::2]  # (E, B, 3)
        mean_b = means[1::2]  # (E, B, 3)
        has_both_sides = (weight_sums[0::2] > 0) & (weight_sums[1::2] > 0)  # (E,)
        return mean_a, mean_b, has_both_sides

    def _edge_features_and_direction(
        self, ts: torch.Tensor, detach_base: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """edge별 correction magnitude(gradient 보유)와 방향(항상 detach), 이
        프레임들의 correction gate, 그리고 alpha 자체를 계산한다.

        direction은 _falloff_side_means의 falloff-weighted patch-centroid
        위치(=compute_boundary_gap_distances가 재는 것과 정확히 같은 정의)에서
        유도된다 -- coarse-only anchor rigid transform이 아니다 (위
        _falloff_side_means의 docstring 참고). magnitude = gate *
        max_displacement * alpha이므로 alpha in [0,1]인 한 항상 [0, gate *
        max_displacement] 안에 있다 -- sigmoid라 부호가 뒤집혀 벌리는 방향으로
        움직일 수는 없지만(pull-only), 이전의 gap_error 기반 설계와 달리
        "실제 gap을 절대 못 넘는다"는 기하학적 보장은 없다 (대신
        max_displacement 자체가 boundary 국소 spacing 기준으로 작게 잡혀
        있다). GNN 입력(dist_norm/canonical_distance_norm)은 이 edge의
        max_displacement로 정규화한 무차원 스칼라다 -- gap_error처럼 "이미
        허용범위 안이면 0"이 아니라 항상 nonzero라서, render loss의 gradient가
        alpha(및 그 앞의 encoder/message-passing 레이어)까지 이 값이 0이든
        아니든 상관없이 도달한다 (이게 이번 설계가 gap_error 기반 설계를
        대체한 핵심 이유 -- 모듈 docstring 참고).

        :param detach_base: True면 GNN에 들어가는 coarse rot/transl/centers
            (node feature) 입력 자체를 detach한다 (isolated loss 계산용 --
            gradient가 오직 self.gnn 자신의 파라미터로만 흐르게 함,
            flow3d/analysis/loss_joint_gnn_only.py와 동일한 격리 방식).
            False면(기본 forward 경로) 평소처럼 gradient가 base coarse motion
            에도 정상적으로 흐른다 (렌더링/track 등 다른 loss가 이미 그렇게
            쓰고 있으므로). direction 자체는 이 플래그와 무관하게 항상
            detach된 신호다 (_falloff_side_means(..., detach=True)).
        :return: magnitude (E, B) [gradient는 self.gnn 파라미터(및
            detach_base=False일 때 base motion)로 흐름], direction (E, B, 3)
            [항상 detached], edge_gate (E, B) [self.correction_gate를 ts로
            gather한 것, 항상 detached], alpha (E, B) in [0, 1] [magnitude와
            같은 gradient 경로].
        """
        mean_a, mean_b, has_both_sides = self._falloff_side_means(ts, detach=True)

        with torch.no_grad():
            gap_vec = mean_b - mean_a  # (E, B, 3)
            dist_before = gap_vec.norm(dim=-1).clamp_min(1e-8)  # (E, B)
            direction = gap_vec / dist_before[..., None]  # (E, B, 3)
            max_disp = self.max_displacement[:, None].clamp_min(1e-8)  # (E, 1)
            dist_norm = dist_before / max_disp  # (E, B)
            canonical_distance_norm = (self.canonical_distance[:, None] / max_disp).expand_as(dist_norm)  # (E, B)
            edge_gate = self.correction_gate[:, ts]  # (E, B)

        rots = self.params["rots"][:, ts]  # (C, B, 6)
        transls = self.params["transls"][:, ts]  # (C, B, 3)
        centers = self.params["centers"]  # (C, 3)
        if detach_base:
            rots = rots.detach()
            transls = transls.detach()
            centers = centers.detach()

        alpha = self.gnn(rots, transls, centers, dist_norm, canonical_distance_norm, edge_gate)  # (E, B) in [0, 1]
        magnitude = edge_gate * self.max_displacement[:, None] * alpha * has_both_sides[:, None].to(alpha.dtype)
        return magnitude, direction, edge_gate, alpha

    def compute_transforms(
        self,
        ts: torch.Tensor,
        coefs: torch.Tensor,
        cluster_ids: torch.Tensor,
        global_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """coarse-to-fine 결합은 ScalableMotionBases.compute_transforms와 완전히
        동일하다 (부모 그대로 호출) -- 바뀐 부분은 그 결과의 translation
        성분에 falloff-weighted per-edge boundary displacement를 더하는
        것뿐이다.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        :param global_indices: (G,) long, optional. cluster_ids[i]/coefs[i]가
            속한 canonical foreground Gaussian의 실제 global index --
            per-Gaussian falloff row(self.falloff_global_idx)를 이 query의
            어느 row에 더해야 하는지는 오직 이 값으로만 알 수 있다 (cluster_ids
            값만으로는 "어느 cluster인지"까지만 알 수 있지, "정확히 어떤
            Gaussian인지"는 알 수 없다 -- falloff weight는 Gaussian마다 다르다).
            flow3d/scene_model.py's SceneModel.compute_transforms(ts, inds)가
            그 inds(또는 inds=None이면 arange)를 그대로 넘겨준다 -- 예를 들어
            flow3d/renderer.py's Renderer.__init__이 track 시각화용으로 딱 10개
            Gaussian만 (inds=torch.arange(10)) 질의하는 경우처럼, "전체 배열"이
            아닌 임의의 subset/순서로 질의해도 정확히 동작한다.
            생략하면(None) query가 정확히 전체 canonical foreground 배열
            (순서 그대로, 길이 self.num_fg_gaussians)이라고 가정한다 -- 다를
            경우 ValueError.
        returns transforms (G, B, 3, 4)
        """
        base_transforms = ScalableMotionBases.compute_transforms(self, ts, coefs, cluster_ids)  # (G, B, 3, 4)

        G = cluster_ids.shape[0]
        num_fg = int(self.num_fg_gaussians)
        device = cluster_ids.device
        if global_indices is None:
            if G != num_fg:
                raise ValueError(
                    "EdgeBoundaryGraphCorrectedScalableMotionBases.compute_transforms was called "
                    f"without global_indices and the query ({G} rows) isn't the full canonical "
                    f"foreground array (expected {num_fg}) -- its per-Gaussian boundary correction "
                    "is keyed by global identity, unlike the per-cluster "
                    "*GraphCorrectedScalableMotionBases variants, which are subset-invariant. Pass "
                    "global_indices (flow3d/scene_model.py's SceneModel.compute_transforms already "
                    "does this), or call with the full array in order."
                )
            global_indices = torch.arange(G, device=device)
        elif global_indices.shape[0] != G:
            raise ValueError(
                f"global_indices has {global_indices.shape[0]} entries but the query has {G} rows."
            )

        magnitude, direction, edge_gate, alpha = self._edge_features_and_direction(ts, detach_base=False)
        self._last_boundary_correction = {"magnitude": magnitude, "gate": edge_gate, "alpha": alpha}

        B = ts.shape[0]
        displacement = base_transforms.new_zeros(G, B, 3)
        if self.falloff_global_idx.numel() > 0:
            # inverse_map[global gaussian index] -> position within this query
            # (-1 if that Gaussian wasn't queried at all).
            inverse_map = torch.full((num_fg,), -1, dtype=torch.long, device=device)
            inverse_map[global_indices] = torch.arange(G, device=device)
            query_pos = inverse_map[self.falloff_global_idx]  # (M,)
            in_query = query_pos >= 0
            if bool(in_query.any()):
                edge_id = self.falloff_edge_id[in_query]
                m_rows = magnitude[edge_id]  # (M', B)
                dir_rows = direction[edge_id]  # (M', B, 3)
                contribution = (
                    0.5
                    * self.falloff_sign[in_query, None, None]
                    * m_rows[:, :, None]
                    * dir_rows
                    * self.falloff_weight[in_query, None, None]
                )  # (M', B, 3)
                displacement.index_add_(0, query_pos[in_query], contribution)

        transforms = base_transforms.clone()
        transforms[..., 3] = transforms[..., 3] + displacement
        return transforms


def _apply_transform(transforms: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """transforms: (N, B, 3, 4), points: (N, 3) -> (N, B, 3)."""
    homog = F.pad(points, (0, 1), value=1.0)
    return torch.einsum("nbij,nj->nbi", transforms, homog)


def compute_boundary_gap_distances(
    motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases,
    ts: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """
    per-edge, per-frame 경계 거리를 correction 적용 전/후로 계산한다.
    compute_boundary_gap_loss와 flow3d/analysis/gnn_check_edge.py(시각화)
    양쪽에서 재사용하는 공용 계산.

    motion_bases._falloff_row_positions(항상 detach)를 그대로 재사용한다 --
    "현재 side의 위치"를 이 함수가 독립적으로 다시 계산하던 예전 버전과 달리,
    motion_bases._edge_features_and_direction(GNN이 실제로 보는 dist_norm/
    direction)이 재는 것과 **정의상 동일한** 값이라는 것이 보장된다 (모듈
    docstring 참고 -- GNN이 못 보는 gap을 "고쳤다"고 착각하는 것을 막는 게 이
    리팩터의 목적이다). canonical_means/coefs_all/cluster_ids_all을 외부에서
    받지 않는 이유도 같다: motion_bases.falloff_canonical_mean/falloff_coefs
    (refresh_boundary_falloff가 채움)가 이미 그 값들의 스냅샷이므로, 호출하는
    쪽이 매번 model.fg에서 다시 읽어 넘길 필요가 없다.

    dist_after는 "각 side의 falloff-weighted 평균이 정확히 0.5*m*dir만큼
    이동한다"고 가정하지 않는다 -- 그 가정은 두 가지 이유로 틀릴 수 있다:
    (1) falloff_weight가 row마다 다르면 가중 평균은 비선형이라 "평균의 이동량
    = 이동량의 평균"이 아니다. (2) 한 Gaussian이(자기 cluster가 여러 edge에
    인접해 있으면) 둘 이상 edge의 falloff row에 동시에 걸릴 수 있고, 그러면
    compute_transforms에서 그 Gaussian은 모든 관련 edge의 correction을 다
    더해서 받는다. 그래서 여기서는 compute_transforms와 완전히 동일한 방식
    (row별 displacement를 같은 global Gaussian index로 index_add_ 합산)으로
    correction *후* row 위치를 만든 다음, 그 위치로 각 side의 falloff-weighted
    평균을 다시 계산해서 dist_after를 구한다.

    :param motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases.
    :param ts: (B,) frame indices.
    :return: None이면 falloff row가 하나도 없음 (correction이 어디에도 적용될
        수 없는 상태). 아니면 dict with (모두 (E, B), has_both_sides만 (E,)):
        "dist_before" (correction 적용 전 실제 경계 거리),
        "dist_after" (correction 적용 후, 즉 실제로 렌더되는 경계 거리),
        "magnitude" (gradient 보유, detach_base=True로 계산됨),
        "alpha" (gradient 보유, magnitude와 같은 경로 -- [0,1], "왜 이 edge가
        안 고쳐졌는지" 진단용: alpha는 작은데 max_displacement 자체는 큰지,
        아니면 alpha 자체가 0에 가까운지 구분할 수 있다),
        "has_both_sides" (bool, 그 edge가 지금 양쪽 다 falloff row를 갖는지 --
        한쪽이라도 없으면 mean position이 정의되지 않으므로 나머지 값들은
        무의미하다).
    """
    mb = motion_bases
    if not hasattr(mb, "gnn") or not hasattr(mb, "edge_cluster_a"):
        raise TypeError(
            "motion_bases must be an EdgeBoundaryGraphCorrectedScalableMotionBases."
        )

    if mb.falloff_global_idx.numel() == 0:
        return None

    E = mb.num_edges
    B = ts.shape[0]

    positions, group_ids, weight = mb._falloff_row_positions(ts, detach=True)  # (M,B,3), (M,), (M,)

    weight_sums = positions.new_zeros(E * 2)
    pos_sums = positions.new_zeros(E * 2, B, 3)
    weight_sums.index_add_(0, group_ids, weight)
    pos_sums.index_add_(0, group_ids, positions * weight[:, None, None])
    means_before = pos_sums / weight_sums[:, None, None].clamp_min(1e-8)
    mean_a = means_before[0::2]  # (E, B, 3)
    mean_b = means_before[1::2]  # (E, B, 3)
    has_both_sides = (weight_sums[0::2] > 0) & (weight_sums[1::2] > 0)  # (E,)
    dist_before = (mean_b - mean_a).norm(dim=-1)  # (E, B)

    magnitude, direction, _, alpha = mb._edge_features_and_direction(ts, detach_base=True)  # (E,B), (E,B,3)

    # Actual per-row displacement (compute_transforms's own formula), then
    # summed per Gaussian EXACTLY like compute_transforms's index_add_ over
    # the full query -- a Gaussian that carries a falloff row for more than
    # one edge picks up every relevant edge's contribution here too, not just
    # this row's own edge.
    m_rows = magnitude[mb.falloff_edge_id]  # (M, B)
    dir_rows = direction[mb.falloff_edge_id]  # (M, B, 3)
    contribution = (
        0.5 * mb.falloff_sign[:, None, None] * m_rows[:, :, None] * dir_rows * weight[:, None, None]
    )  # (M, B, 3)
    num_fg = int(mb.num_fg_gaussians)
    per_gaussian_displacement = contribution.new_zeros(num_fg, B, 3)
    per_gaussian_displacement.index_add_(0, mb.falloff_global_idx, contribution)
    displacement_per_row = per_gaussian_displacement[mb.falloff_global_idx]  # (M, B, 3)

    positions_after = positions + displacement_per_row
    pos_sums_after = positions_after.new_zeros(E * 2, B, 3)
    pos_sums_after.index_add_(0, group_ids, positions_after * weight[:, None, None])
    means_after = pos_sums_after / weight_sums[:, None, None].clamp_min(1e-8)  # same weight_sums as before
    mean_a_after = means_after[0::2]
    mean_b_after = means_after[1::2]
    dist_after = (mean_b_after - mean_a_after).norm(dim=-1)  # (E, B)

    return {
        "dist_before": dist_before,
        "dist_after": dist_after,
        "magnitude": magnitude,
        "alpha": alpha,
        "has_both_sides": has_both_sides,
    }


def compute_boundary_gap_loss(
    motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases,
    ts: torch.Tensor,
) -> torch.Tensor:
    """
    진단/opt-in 전용 loss (trainer.py's w_boundary_gap, 기본값 0.0 -- 명시적으로
    켜지 않는 한 학습에 반영되지 않는다). correction magnitude 자체는 더 이상
    이 함수가 쓰는 contact_reference_distance/gap_tolerance에 의존하지 않는다
    (_edge_features_and_direction 참고, max_displacement 기반으로 바뀜) --
    그래서 정적으로 벌어진 edge에서 gap_error가 항상 0이라 gradient가 아예
    도달하지 못하던 문제가 이 함수와는 무관해졌다. 이 함수는 여전히 "실제
    (coarse+fine blended, base 입력은 detach) 경계 거리가
    (contact_reference_distance + motion_bases.gap_tolerance)보다 벌어질 때만"
    hinge loss를 주는 그대로 남아 있고, 필요하면 opt-in loss로 켜거나(그 경우
    correction이 이 hinge도 함께 만족하도록 추가 압력을 준다) 진단 용도로
    compute_boundary_gap_distances를 통해 계속 쓸 수 있다.
    canonical_distance(rest pose 거리)가 아니라 contact_reference_distance를
    기준으로 쓰는 이유: 손-무릎처럼 일시적으로만 닿는 pair는 canonical(rest)
    pose에서 서로 멀리 떨어져 있을 수 있어서, rest-pose 거리를 목표로 삼으면
    CONNECTED 프레임에서도 두 영역을 붙이려는 압력이 생기지 않는다 -- 대신
    실제로 닿았을 때 관찰된 거리(빌드 시점에 CONNECTED 프레임들의 weighted
    patch centroid 거리의 median으로 계산됨, load_edge_boundary_sets 참고)를
    쓴다.

    **Correction gate(연속값)와는 다른, 하드 마스크**: motion_bases.
    connected_mask로 정확히 CONNECTED로 관찰된 (edge, frame) 항목만 loss에
    포함시킨다 -- DISCONNECTED는 물론, 보간된 correction_gate가 0보다 큰
    UNKNOWN 구간도 (correction 자체는 걸려 있어도) 실제 접촉 여부를 알 수
    없으므로 완전히 제외한다. Gradient는 오직 motion_bases.gnn 자신의
    파라미터로만 흐른다 -- compute_boundary_gap_distances 참고.

    :param motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases.
    :param ts: (B,) frame indices.
    :return: scalar; falloff row가 하나도 없거나 CONNECTED로 관찰된 항목이 하나도 없으면 0.0.
    """
    result = compute_boundary_gap_distances(motion_bases, ts)
    if result is None:
        return motion_bases.contact_reference_distance.new_zeros(())

    connected_mask = motion_bases.connected_mask[:, ts]  # (E, B), hard -- not gate-weighted
    include = result["has_both_sides"][:, None] & connected_mask
    if not bool(include.any()):
        return motion_bases.contact_reference_distance.new_zeros(())

    allowed = motion_bases.contact_reference_distance[:, None] + motion_bases.gap_tolerance  # (E, 1)
    hinge = F.relu(result["dist_after"] - allowed)
    return hinge[include].pow(2).mean()


def boundary_magnitude_reg_loss(magnitude: torch.Tensor) -> torch.Tensor:
    """L2 magnitude regularizer on the per-edge correction scalar. :param magnitude: (E, B) or (E, ...)."""
    return magnitude.pow(2).mean()


def boundary_magnitude_smoothness_loss(magnitude_triplet: torch.Tensor) -> torch.Tensor:
    """
    Second-order ("acceleration") temporal smoothness on the correction
    magnitude, evaluated at consecutive frames (t-1, t, t+1). Same
    central-difference math as flow3d/analysis/loss.py's
    gnn_correction_smoothness_loss, but on a scalar instead of a 3-vector.

    :param magnitude_triplet: (E, B, 3), axis=-1 is the (t-1, t, t+1) triplet.
    :return: scalar.
    """
    accel = 2 * magnitude_triplet[..., 1] - magnitude_triplet[..., 0] - magnitude_triplet[..., 2]
    return accel.abs().mean()


if __name__ == "__main__":
    # Sanity checks (updated for the gap_error-free redesign -- correction
    # magnitude = gate * max_displacement * alpha, alpha = sigmoid(...) in
    # [0, 1], no longer gated by gap_error/contact_reference_distance):
    #   1. near-zero-init: EdgeBoundary-corrected ~= plain ScalableMotionBases
    #      (not bit-exact any more -- sigmoid can never hit exactly 0).
    #   2. compute_transforms rejects a subset query (only full array supported).
    #   3a. CRITICAL: a dummy render-style loss's gradient reaches the GNN's
    #      encoder even when the boundary is well within the OLD design's
    #      tolerance (gap_error would have been exactly 0, killing the
    #      gradient there) -- this is the exact bug this redesign fixes.
    #   3. forced-open gap + head nudged off its near-zero init: magnitude is
    #      nonzero, pull-only (>= 0), and bounded by gate * max_displacement
    #      (NOT by the current gap size any more -- that geometric guarantee
    #      is gone by design, see EdgeBoundaryGNN docstring), boundary-anchor-
    #      adjacent points move by close to +-0.5*m*dir, and displacement
    #      decays smoothly away from the boundary (falloff).
    #   4. compute_boundary_gap_loss (opt-in/diagnostic, unchanged function):
    #      0 within tolerance, > 0 once forced open; gradient reaches only
    #      motion_bases.gnn's own parameters.
    #   5. boundary_magnitude_reg_loss / boundary_magnitude_smoothness_loss basic behavior.
    #   6. save -> init_from_state_dict round-trip reproduces identical output.
    #   7. refresh_boundary_falloff after a simulated densify/cull (Gaussian count AND
    #      order both change) keeps the correction correctly localized, with no stale
    #      indices anywhere -- and compute_transforms's full-array check tracks the new count.
    #   8. new schema (contact core / patch weight / correction_gate / gap_tolerance
    #      sharing) PLUS _build_edge_gate's before-first/after-last-observation ->
    #      0 fix (not held flat at the nearest knot forever) PLUS max_displacement
    #      old-checkpoint KeyError guard.
    import tempfile

    torch.manual_seed(0)
    num_clusters, num_frames, num_fine = 3, 6, 2

    # Two clusters (0, 1) share a boundary band; cluster 2 is unrelated/far away.
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    rots = torch.zeros(num_clusters, num_frames, 6)
    rots[..., 0] = 1.0
    rots[..., 4] = 1.0  # identity 6D rotation
    transls = torch.zeros(num_clusters, num_frames, 3)
    fine_rots = rots[:, None].repeat(1, num_fine, 1, 1)
    fine_transls = torch.zeros(num_clusters, num_fine, num_frames, 3)
    baseline = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    # Canonical Gaussians: cluster 0 spans x in [-1, 0.5] (boundary band near x=0.5),
    # cluster 1 spans x in [0.5, 2] (boundary band near x=0.5), cluster 2 is isolated.
    n0, n1, n2 = 40, 40, 10
    x0 = torch.linspace(-1.0, 0.5, n0)
    x1 = torch.linspace(0.5, 2.0, n1)
    means0 = torch.stack([x0, torch.zeros(n0), torch.zeros(n0)], dim=-1)
    means1 = torch.stack([x1, torch.zeros(n1), torch.zeros(n1)], dim=-1)
    means2 = torch.tensor([10.0, 10.0, 10.0]) + 0.01 * torch.randn(n2, 3)
    canonical_means = torch.cat([means0, means1, means2], dim=0)
    cluster_ids_all = torch.cat(
        [torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long), torch.full((n2,), 2, dtype=torch.long)]
    )
    num_fg = canonical_means.shape[0]

    boundary_a = torch.arange(n0 - 3, n0)  # last 3 points of cluster 0 (x close to 0.5)
    boundary_b = torch.arange(n0, n0 + 3)  # first 3 points of cluster 1 (x close to 0.5)

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

    ts = torch.arange(num_frames)
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(edges_payload, f.name)
        edges_path = f.name

        graph_bases = EdgeBoundaryGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
            baseline,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edges_path=edges_path,
            canonical_means=canonical_means,
            cluster_ids_all=cluster_ids_all,
            coefs_all=coefs,
            gnn_hidden_dim=16,
            gnn_num_layers=1,
            falloff_radius=0.3,
            gap_tolerance=0.05,
        )

    # --- 1. near-zero-init equivalence (sigmoid can't hit exactly 0, and the
    #        last layer's weight is now a small random init rather than exact
    #        zero, so this is "small", not bit-exact -- see
    #        EdgeBoundaryGNN._INIT_LOGIT_BIAS/_INIT_WEIGHT_STD) ---
    ref = baseline.compute_transforms(ts, coefs, cluster_ids_all)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    max_diff = (ref - out).abs().max().item()
    print(f"[near-zero-init] max |baseline - edge_boundary_corrected| = {max_diff:.3e} (expect small, not exactly 0)")
    assert max_diff < 5e-2

    magnitude0 = graph_bases.last_boundary_correction["magnitude"]
    print(f"[near-zero-init] |magnitude|_max = {magnitude0.abs().max().item():.3e} (expect small)")
    assert magnitude0.abs().max().item() < 5e-2
    alpha_init = torch.sigmoid(torch.tensor(EdgeBoundaryGNN._INIT_LOGIT_BIAS))
    print(f"[near-zero-init] alpha at init = {alpha_init.item():.3e} (expect small, ~= sigmoid(_INIT_LOGIT_BIAS) "
          "modulo the small-random-weight perturbation)")
    assert alpha_init.item() < 2e-2

    print(f"[max-displacement] max_displacement = {graph_bases.max_displacement.tolist()} (expect > 0)")
    assert bool((graph_bases.max_displacement > 0).all())

    # --- 3a. CRITICAL: a dummy render-style loss's gradient reaches the GNN's
    #        encoder even when the boundary is well within the OLD design's
    #        tolerance (edge 15(4-21) in production had gap_error == 0 for
    #        72% of frames -- reproduce that here by pushing gap_tolerance up
    #        past the actual falloff-weighted dist_before, exactly the "this
    #        edge has never been observed to open beyond normal" case the old
    #        design treated as nothing-to-fix). ---
    mean_a_pre, mean_b_pre, _ = graph_bases._falloff_side_means(ts, detach=True)
    dist_before_pre = (mean_b_pre - mean_a_pre).norm(dim=-1)
    huge_tolerance = float(dist_before_pre.max().item()) + 1.0
    allowed_pre = graph_bases.contact_reference_distance[:, None] + huge_tolerance
    gap_error_pre = F.relu(dist_before_pre - allowed_pre)
    print(f"[gradient-fix] gap_error under the OLD (removed) design = {gap_error_pre[0].tolist()} "
          "(expect all 0 -- old design would zero the gradient here)")
    assert bool((gap_error_pre[0] == 0).all())

    graph_bases.zero_grad()
    out_pre = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    loss_pre = out_pre.pow(2).mean()
    loss_pre.backward()
    encoder_grad = graph_bases.gnn.encoder[0].weight.grad
    print(f"[gradient-fix] gnn.encoder[0].weight.grad norm = "
          f"{0.0 if encoder_grad is None else encoder_grad.norm().item():.3e} "
          "(expect > 0 -- the old gap_error-gated design would give exactly 0 here)")
    assert encoder_grad is not None and encoder_grad.norm().item() > 0.0
    graph_bases.zero_grad()

    # --- 2. subset query without global_indices is rejected; WITH global_indices
    #        (mirrors flow3d/renderer.py's Renderer.__init__, which queries just
    #        10 arbitrary/unordered Gaussians via inds=torch.arange(10)) it must
    #        match the full-array computation sliced at those same indices.
    try:
        graph_bases.compute_transforms(ts, coefs[:5], cluster_ids_all[:5])
        raise AssertionError("expected ValueError for a subset query without global_indices")
    except ValueError as e:
        print(f"[subset-guard] raised as expected (no global_indices): {e}")

    full_out_for_subset_check = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    subset_idx = torch.tensor([5, 47, 0, 88, 39, 40, 41, 20, 60, 79])  # arbitrary order, spans all 3 clusters
    subset_out = graph_bases.compute_transforms(
        ts, coefs[subset_idx], cluster_ids_all[subset_idx], global_indices=subset_idx
    )
    subset_err = (subset_out - full_out_for_subset_check[subset_idx]).abs().max().item()
    print(f"[subset-query] max |subset(global_indices) - full[subset_idx]| = {subset_err:.3e} (expect ~0)")
    assert subset_err < 1e-5

    # global_indices whose length disagrees with the query is also rejected.
    try:
        graph_bases.compute_transforms(
            ts, coefs[subset_idx], cluster_ids_all[subset_idx], global_indices=subset_idx[:3]
        )
        raise AssertionError("expected ValueError for mismatched global_indices length")
    except ValueError as e:
        print(f"[subset-guard] raised as expected (length mismatch): {e}")

    # --- 3. force a real gap open, move the GNN head off its near-zero init,
    #        then check magnitude/direction/falloff behavior. (gap_error is
    #        computed here only for logging/comparison against the OLD
    #        design -- it plays no role in the new magnitude computation.) ---
    with torch.no_grad():
        graph_bases.params["transls"][1] += torch.tensor([0.5, 0.0, 0.0])  # separate cluster 1 from cluster 0

    mean_a0, mean_b0, has_both0 = graph_bases._falloff_side_means(ts, detach=True)
    gap_error0 = F.relu(
        (mean_b0 - mean_a0).norm(dim=-1) - graph_bases.contact_reference_distance[:, None] - graph_bases.gap_tolerance
    )
    print(f"[forced-open] gap_error under the OLD design (edge 0) = {gap_error0[0].tolist()} (expect > 0 everywhere)")
    assert bool((gap_error0[0] > 0).all())

    graph_bases.zero_grad()
    out2 = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    loss0 = out2.pow(2).mean()
    loss0.backward()
    with torch.no_grad():
        for p in graph_bases.gnn.edge_mlp[-1].parameters():
            if p.grad is not None:
                p += 3.0  # push the logit up (bias starts at _INIT_LOGIT_BIAS=-4.5 -> ~-1.5, alpha~0.18): meaningfully nonzero but still << 1

    magnitude1, direction1, gate1, alpha1 = graph_bases._edge_features_and_direction(ts, detach_base=False)
    print(f"[legacy-fallback] correction_gate all-ones: {bool((gate1 == 1.0).all())} (expect True)")
    assert bool((gate1 == 1.0).all())
    assert bool((graph_bases.connected_mask == True).all())  # noqa: E712 -- legacy payload has no per-frame fields
    print(f"[post-step] magnitude nonzero: {magnitude1.abs().max().item():.3e}")
    assert magnitude1.abs().max().item() > 0.0
    assert bool((alpha1 >= 0.0).all()) and bool((alpha1 <= 1.0).all())

    # magnitude must be in [0, gate * max_displacement]: never negative
    # (pull-only, no sign flip) and never larger than this edge's fixed
    # absolute cap. Unlike the removed gap_error-gated design, there is no
    # guarantee any more that magnitude <= the actual current gap
    # (gap_error0) -- max_displacement (a small, boundary-local-spacing-based
    # cap) is the safety valve instead, not the live gap size.
    cap1 = gate1 * graph_bases.max_displacement[:, None]
    print(
        f"[bounded] magnitude range: min={magnitude1.min().item():.3e} (expect >= 0), "
        f"max excess over gate*max_displacement={((magnitude1 - cap1).clamp_min(0)).max().item():.3e} (expect ~0)"
    )
    assert bool((magnitude1 >= -1e-6).all())
    assert bool((magnitude1 <= cap1 + 1e-5).all())

    out3 = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    means_out3 = torch.einsum(
        "pnij,pj->pni", out3, F.pad(canonical_means, (0, 1), value=1.0)
    )  # (G, B, 3)
    ref_forced = ScalableMotionBases.compute_transforms(graph_bases, ts, coefs, cluster_ids_all)
    means_ref = torch.einsum(
        "pnij,pj->pni", ref_forced, F.pad(canonical_means, (0, 1), value=1.0)
    )
    disp = (means_out3 - means_ref)[..., 0]  # x-component of displacement, (G, B)

    # points essentially AT the anchor (weight ~= 1) should move by close to +-0.5*m*dir_x
    dir_x = direction1[0, :, 0]  # (B,)
    m = magnitude1[0]  # (B,)
    expected_a = 0.5 * m * dir_x
    expected_b = -0.5 * m * dir_x
    err_a = (disp[boundary_a] - expected_a[None]).abs().max().item()
    err_b = (disp[boundary_b] - expected_b[None]).abs().max().item()
    print(f"[falloff] near-anchor displacement error: side_a={err_a:.3e} side_b={err_b:.3e} (expect small)")
    assert err_a < 5e-3 and err_b < 5e-3

    # a point in cluster 0 far from the boundary (x = -1) should move much less than the boundary itself.
    far_disp = disp[0].abs().max().item()
    near_disp = disp[boundary_a[-1]].abs().max().item()
    print(f"[falloff] |disp| far={far_disp:.3e} vs near-boundary={near_disp:.3e} (expect far << near)")
    assert far_disp < 0.3 * near_disp

    # cluster 2 (isolated, no incident edge) should be completely untouched.
    disp_c2 = (means_out3 - means_ref)[n0 + n1 :]
    print(f"[isolation] cluster-2 (no edge) max |disp| = {disp_c2.abs().max().item():.3e} (expect exactly 0)")
    assert disp_c2.abs().max().item() == 0.0

    # --- 3b. compute_boundary_gap_distances's dist_after must match the
    #         GROUND TRUTH: falloff-weighted mean of the Gaussians' ACTUAL
    #         rendered (post-correction) positions from compute_transforms --
    #         not the old "each side's mean moves by exactly 0.5*m*dir"
    #         shortcut, which is wrong whenever falloff_weight is non-uniform
    #         (true here: RBF decay, not every boundary_a/b point has
    #         weight==1) even with only one edge involved. ---
    result_gt_check = compute_boundary_gap_distances(graph_bases, ts)
    idx_a, idx_b = graph_bases.falloff_global_idx[graph_bases.falloff_sign > 0], graph_bases.falloff_global_idx[graph_bases.falloff_sign < 0]
    w_a, w_b = graph_bases.falloff_weight[graph_bases.falloff_sign > 0], graph_bases.falloff_weight[graph_bases.falloff_sign < 0]
    pos_after_all = torch.einsum("pnij,pj->pni", out3, F.pad(canonical_means, (0, 1), value=1.0))  # (G, B, 3), out3 = actual corrected render
    mean_a_gt = (pos_after_all[idx_a] * w_a[:, None, None]).sum(0) / w_a.sum()  # (B, 3)
    mean_b_gt = (pos_after_all[idx_b] * w_b[:, None, None]).sum(0) / w_b.sum()  # (B, 3)
    dist_after_gt = (mean_b_gt - mean_a_gt).norm(dim=-1)  # (B,)
    err_dist_after = (result_gt_check["dist_after"][0] - dist_after_gt).abs().max().item()
    print(
        f"[dist-after-gt] compute_boundary_gap_distances vs. ground-truth (from actual "
        f"compute_transforms render): max err={err_dist_after:.3e} (expect ~0)"
    )
    assert err_dist_after < 1e-4

    # alpha returned by compute_boundary_gap_distances must be in [0,1] and
    # consistent with magnitude = gate * max_displacement * alpha (where both
    # sides currently have falloff rows).
    alpha_gt = result_gt_check["alpha"]
    print(f"[alpha] range: min={alpha_gt.min().item():.3e}, max={alpha_gt.max().item():.3e} (expect within [0,1])")
    assert bool((alpha_gt >= 0.0).all()) and bool((alpha_gt <= 1.0).all())
    cap_gt = gate1 * graph_bases.max_displacement[:, None]
    expected_magnitude_from_alpha = cap_gt * alpha_gt
    err_alpha_consistency = (
        (result_gt_check["magnitude"] - expected_magnitude_from_alpha)[result_gt_check["has_both_sides"]]
        .abs()
        .max()
        .item()
    )
    print(f"[alpha] magnitude == gate * max_displacement * alpha: max err={err_alpha_consistency:.3e} (expect ~0)")
    assert err_alpha_consistency < 1e-5

    # --- 4. compute_boundary_gap_loss: 0 within tolerance, > 0 once forced open
    #        (still forced from step 3 above); grad isolation. tolerance now
    #        lives on motion_bases.gap_tolerance, not a per-call argument. ---
    graph_bases.zero_grad()
    graph_bases.gap_tolerance = torch.tensor(10.0)
    gap_loss_tight = compute_boundary_gap_loss(graph_bases, ts)
    print(f"[gap-loss] huge tolerance -> loss={gap_loss_tight.item():.3e} (expect 0)")
    assert gap_loss_tight.item() == 0.0

    graph_bases.gap_tolerance = torch.tensor(0.0)
    gap_loss_strict = compute_boundary_gap_loss(graph_bases, ts)
    print(f"[gap-loss] forced-open, zero tolerance -> loss={gap_loss_strict.item():.3e} (expect > 0)")
    assert gap_loss_strict.item() > 0.0
    gap_loss_strict.backward()
    head_grad = graph_bases.gnn.edge_mlp[-1].weight.grad
    print(f"[gap-loss] gnn.edge_mlp[-1].weight.grad norm = {head_grad.norm().item():.3e} (expect > 0)")
    assert head_grad.norm().item() > 0.0
    for name in ("rots", "transls", "centers", "fine_rots", "fine_transls"):
        grad = graph_bases.params[name].grad
        is_none_or_zero = grad is None or grad.abs().max().item() == 0.0
        print(
            f"[gap-loss] params[{name!r}].grad "
            f"{'is None' if grad is None else f'max={grad.abs().max().item():.3e}'} (expect None/0)"
        )
        assert is_none_or_zero
    graph_bases.gap_tolerance = torch.tensor(0.05)  # restore the constructor's value
    with torch.no_grad():
        graph_bases.params["transls"][1] -= torch.tensor([0.5, 0.0, 0.0])  # undo, for the tests below

    # --- 5. magnitude reg / smoothness losses ---
    reg0 = boundary_magnitude_reg_loss(torch.zeros(4, 3))
    assert reg0.item() == 0.0
    reg1 = boundary_magnitude_reg_loss(torch.ones(4, 3))
    assert reg1.item() > 0.0

    linear_triplet = torch.stack(
        [-torch.ones(4), torch.zeros(4), torch.ones(4)], dim=-1
    )  # linear in t -> 0 acceleration
    smooth_linear = boundary_magnitude_smoothness_loss(linear_triplet)
    jerky_triplet = linear_triplet.clone()
    jerky_triplet[:, 1] += 5.0
    smooth_jerky = boundary_magnitude_smoothness_loss(jerky_triplet)
    print(f"[smoothness] linear={smooth_linear.item():.3e} jerky={smooth_jerky.item():.3e}")
    assert smooth_linear.item() < 1e-5
    assert smooth_jerky.item() > smooth_linear.item()

    # --- 6. save -> init_from_state_dict round-trip ---
    full_state_dict = {f"motion_bases.{k}": v for k, v in graph_bases.state_dict().items()}
    assert EdgeBoundaryGraphCorrectedScalableMotionBases.has_gnn_state(full_state_dict, "motion_bases.")
    restored = EdgeBoundaryGraphCorrectedScalableMotionBases.init_from_state_dict(
        full_state_dict, prefix="motion_bases."
    )
    out_restored = restored.compute_transforms(ts, coefs, cluster_ids_all)
    out_original = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    max_diff_roundtrip = (out_restored - out_original).abs().max().item()
    print(f"[round-trip] max |original - restored| = {max_diff_roundtrip:.3e}")
    assert max_diff_roundtrip < 1e-6

    # --- 7. refresh_boundary_falloff after a simulated densify + cull ---
    # Simulate flow3d/params.py's GaussianParams.densify_params/cull_params: the
    # foreground array is rebuilt (split/dup + concat, then a cull mask), so both
    # the COUNT and the ORDER of Gaussians change -- old global indices are meaningless.
    torch.manual_seed(1)
    should_split = torch.zeros(num_fg, dtype=torch.bool)
    should_split[boundary_a[-1]] = True  # split the Gaussian right at the side-a anchor
    should_dup = torch.zeros(num_fg, dtype=torch.bool)
    new_far_idx = 0  # a far-from-any-boundary point in cluster 0
    should_dup[new_far_idx] = True

    def _densify(x):
        return torch.cat([x[~should_split], x[should_dup], x[should_split].repeat(2, *([1] * (x.dim() - 1)))], dim=0)

    means_densified = _densify(canonical_means)
    cluster_ids_densified = _densify(cluster_ids_all)
    coefs_densified = _densify(coefs)
    n_after_densify = means_densified.shape[0]
    assert n_after_densify == num_fg + 2  # +1 dup, +1 net from the 1->2 split

    graph_bases.refresh_boundary_falloff(means_densified, cluster_ids_densified, coefs_densified)
    print(
        f"[refresh] after densify: num_fg_gaussians={int(graph_bases.num_fg_gaussians)} "
        f"(expect {n_after_densify}), falloff rows={graph_bases.falloff_global_idx.numel()}"
    )
    assert int(graph_bases.num_fg_gaussians) == n_after_densify

    # compute_transforms must now accept exactly the new (post-densify) full array...
    out_densified = graph_bases.compute_transforms(ts, coefs_densified, cluster_ids_densified)
    # ...and reject the OLD (pre-densify) array size, since it no longer matches reality.
    try:
        graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
        raise AssertionError("expected ValueError: stale (pre-densify) array size")
    except ValueError as e:
        print(f"[refresh] stale full-array query correctly rejected: {e}")

    # The two new points from the split (indices n_after_densify-2, n_after_densify-1,
    # per the concat order in _densify above) sit essentially at the same canonical
    # position as the original split point (right at the side-a anchor) -- they should
    # pick up a nonzero falloff weight automatically, with no stale index anywhere.
    means_densified_out = torch.einsum(
        "pnij,pj->pni", out_densified, F.pad(means_densified, (0, 1), value=1.0)
    )
    means_densified_ref = torch.einsum(
        "pnij,pj->pni",
        ScalableMotionBases.compute_transforms(graph_bases, ts, coefs_densified, cluster_ids_densified),
        F.pad(means_densified, (0, 1), value=1.0),
    )
    disp_densified = (means_densified_out - means_densified_ref)[..., 0]
    split_child_disp = disp_densified[-2:].abs().max().item()
    print(f"[refresh] split children (at the old anchor point) |disp| = {split_child_disp:.3e} (expect > 0)")
    assert split_child_disp > 1e-4

    # The duplicated far-away point (index n0+n1+n2-ish, i.e. wherever should_dup
    # landed in the new array -- position n0 - 1 of the kept prefix, since it's
    # before should_split's removed index) should remain essentially undisplaced.
    kept_prefix_len = int((~should_split).sum())
    dup_new_idx = kept_prefix_len  # first of the appended dup block
    far_disp_after = disp_densified[dup_new_idx].abs().max().item()
    print(f"[refresh] duplicated far point |disp| = {far_disp_after:.3e} (expect ~0)")
    assert far_disp_after < 1e-6

    # Now cull down to a handful of points near the x=0.5 boundary band on BOTH
    # sides (cluster 0 and cluster 1) plus the untouched cluster 2 -- keeping at
    # least one point per side is required for the edge to produce any falloff
    # rows at all (see _compute_falloff_rows: a side with zero surviving
    # Gaussians can't define a boundary). num_fg_gaussians / falloff rows must
    # track this new (smaller, reordered) array.
    near_boundary = (means_densified[:, 0] - 0.5).abs() < 0.25
    keep_mask = near_boundary | (cluster_ids_densified == 2)
    assert bool((cluster_ids_densified[keep_mask] == 0).any())
    assert bool((cluster_ids_densified[keep_mask] == 1).any())
    means_culled = means_densified[keep_mask]
    cluster_ids_culled = cluster_ids_densified[keep_mask]
    coefs_culled = coefs_densified[keep_mask]

    graph_bases.refresh_boundary_falloff(means_culled, cluster_ids_culled, coefs_culled)
    print(
        f"[refresh] after cull: num_fg_gaussians={int(graph_bases.num_fg_gaussians)} "
        f"(expect {int(keep_mask.sum())}), falloff rows={graph_bases.falloff_global_idx.numel()}"
    )
    assert int(graph_bases.num_fg_gaussians) == int(keep_mask.sum())
    assert graph_bases.falloff_global_idx.numel() > 0
    assert int(graph_bases.falloff_global_idx.max()) < int(keep_mask.sum())

    _ = graph_bases.compute_transforms(ts, coefs_culled, cluster_ids_culled)  # must not raise
    print("[refresh] post-cull compute_transforms OK with the new (smaller) full array")

    # --- 8. new schema: contact core anchor, patch weight/local_scale snap,
    #        correction_gate interpolation, connected_mask-gated gap loss
    #        against contact_reference_distance (not canonical_distance),
    #        and the old-checkpoint KeyError guard. ---
    core_a = boundary_a[-1:]  # just the single point right at the boundary (side a's "core")
    core_b = boundary_b[:1]
    weight_a = torch.tensor([0.2, 0.6, 1.0])  # 1.0 exactly at the core point (last of boundary_a)
    weight_b = torch.tensor([1.0, 0.6, 0.2])  # 1.0 exactly at the core point (first of boundary_b)
    local_scale_a = torch.full((3,), 0.05)
    local_scale_b = torch.full((3,), 0.05)
    contact_ref_dist = 5.0  # deliberately far from canonical_distance (~0, boundary points touch)

    edges_payload_v2 = {
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edges_kept": [
            {
                "cluster_a": 0,
                "cluster_b": 1,
                "reason": "kept_persistence",
                "boundary_global_indices_a": boundary_a,
                "boundary_global_indices_b": boundary_b,
                "contact_core_global_indices_a": core_a,
                "contact_core_global_indices_b": core_b,
                "boundary_patch_global_indices_a": boundary_a,
                "boundary_patch_global_indices_b": boundary_b,
                "boundary_patch_weight_a": weight_a,
                "boundary_patch_weight_b": weight_b,
                "boundary_patch_local_scale_a": local_scale_a,
                "boundary_patch_local_scale_b": local_scale_b,
                "contact_reference_distance": contact_ref_dist,
                "connected_frame_indices": [0],
                "unknown_frame_indices": [1, 2, 3, 4],  # frame 5 is neither -> DISCONNECTED
            }
        ],
        "edges_cut": [],
        "cluster_ids": list(range(num_clusters)),
        "meta": {"num_frames": num_frames, "sampled_frame_indices": list(range(num_frames))},
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(edges_payload_v2, f.name)
        graph_bases2 = EdgeBoundaryGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
            baseline,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edges_path=f.name,
            canonical_means=canonical_means,
            cluster_ids_all=cluster_ids_all,
            coefs_all=coefs,
            gnn_hidden_dim=16,
            gnn_num_layers=1,
            falloff_radius=0.3,
        )

    # anchor uses the CONTACT CORE mean, not the wider boundary/patch mean.
    anchor_a_expected = canonical_means[core_a].mean(dim=0)
    anchor_a_from_patch_mean = canonical_means[boundary_a].mean(dim=0)
    print(
        f"[core-anchor] anchor_a matches contact-core mean: "
        f"{bool(torch.allclose(graph_bases2.anchor_a_canonical[0], anchor_a_expected))} (expect True)"
    )
    assert torch.allclose(graph_bases2.anchor_a_canonical[0], anchor_a_expected)
    assert not torch.allclose(graph_bases2.anchor_a_canonical[0], anchor_a_from_patch_mean)

    # correction_gate: 1.0 at the connected frame, 0.0 at the disconnected frame,
    # strictly interpolated (and monotonic) through the unknown stretch in between.
    gate_row = graph_bases2.correction_gate[0]
    print(f"[gate] frame0 (connected)={gate_row[0].item():.3f} (expect 1.0), "
          f"frame5 (disconnected)={gate_row[5].item():.3f} (expect 0.0)")
    assert gate_row[0].item() == 1.0
    assert gate_row[5].item() == 0.0
    for t in range(1, 5):
        assert 0.0 < gate_row[t].item() < 1.0, f"frame {t} gate should be strictly interpolated"
    assert bool((gate_row[1:5][:-1] > gate_row[1:5][1:]).all()), "gate should decrease monotonically 1..4"
    print(f"[gate] unknown-stretch values: {[round(v, 3) for v in gate_row[1:5].tolist()]} (expect strictly decreasing)")

    connected_mask_row = graph_bases2.connected_mask[0]
    assert connected_mask_row.tolist() == [True, False, False, False, False, False]

    # disconnected frame -> exactly zero correction displacement (gate=0 kills the magnitude).
    ts_disc = torch.tensor([5])
    _ = graph_bases2.compute_transforms(ts_disc, coefs, cluster_ids_all)
    mag_disc = graph_bases2.last_boundary_correction["magnitude"]
    print(f"[gate] disconnected-frame magnitude: {mag_disc.abs().max().item():.3e} (expect exactly 0)")
    assert mag_disc.abs().max().item() == 0.0

    # _build_edge_gate: UNKNOWN frames strictly BEFORE the first or AFTER the
    # last observed knot must be forced to 0, not held flat at the nearest
    # knot's value (the fix this session made -- previously np.interp's default
    # flat extrapolation could hold a gate open forever past anything ever
    # actually observed). connected=[3,4], unknown=[0,1,2,5,6,7] -> no
    # disconnected frames exist at all, so [3,4] are the ONLY knots and every
    # other frame is outside their range.
    gate_edge_case, connected_mask_edge_case = _build_edge_gate(
        connected_frames=[3, 4], unknown_frames=[0, 1, 2, 5, 6, 7], num_frames=8
    )
    print(
        f"[gate-extrapolation] gate = {[round(v, 3) for v in gate_edge_case.tolist()]} "
        "(expect frames 0-2 and 5-7 forced to 0.0, frames 3-4 at 1.0)"
    )
    assert gate_edge_case[3].item() == 1.0 and gate_edge_case[4].item() == 1.0
    assert bool((gate_edge_case[[0, 1, 2, 5, 6, 7]] == 0.0).all())
    assert connected_mask_edge_case.tolist() == [False, False, False, True, True, False, False, False]

    # boundary-gap loss uses contact_reference_distance (5.0), not canonical_distance (~0):
    # with a huge tolerance-agnostic target this far away, even the connected frame alone
    # should give exactly 0 loss (dist_after is nowhere near 5.0 in this toy scene).
    graph_bases2.gap_tolerance = torch.tensor(0.0)
    gap_loss_v2 = compute_boundary_gap_loss(graph_bases2, torch.tensor([0]))
    print(f"[gap-loss-v2] connected frame, contact_reference_distance target -> loss={gap_loss_v2.item():.3e} (expect 0)")
    assert gap_loss_v2.item() == 0.0

    # only the CONNECTED frame (0) contributes to the gap loss -- a batch of purely
    # unknown/disconnected frames (1..5) must give exactly 0 regardless of gap size.
    gap_loss_no_connected = compute_boundary_gap_loss(graph_bases2, torch.arange(1, 6))
    print(f"[gap-loss-v2] no connected frames in batch -> loss={gap_loss_no_connected.item():.3e} (expect 0)")
    assert gap_loss_no_connected.item() == 0.0

    # --- old-checkpoint (pre-schema) state_dict is rejected, not silently degraded ---
    full_state_dict_v2 = {f"motion_bases.{k}": v for k, v in graph_bases2.state_dict().items()}
    stale_state_dict = {
        k: v for k, v in full_state_dict_v2.items()
        if k not in ("motion_bases.patch_ref_points", "motion_bases.correction_gate", "motion_bases.connected_mask")
    }
    try:
        EdgeBoundaryGraphCorrectedScalableMotionBases.init_from_state_dict(stale_state_dict, prefix="motion_bases.")
        raise AssertionError("expected KeyError for a state_dict missing the new schema buffers")
    except KeyError as e:
        print(f"[old-checkpoint] correctly rejected: {e}")

    # --- checkpoint predating the falloff-weighted patch-centroid gap/direction
    #     schema (falloff_canonical_mean/falloff_coefs/gap_tolerance) is ALSO
    #     rejected, distinctly from the older patch-reference/frame-gating guard
    #     just above. ---
    stale_state_dict_v2 = {
        k: v for k, v in full_state_dict_v2.items()
        if k not in ("motion_bases.falloff_canonical_mean", "motion_bases.falloff_coefs", "motion_bases.gap_tolerance")
    }
    try:
        EdgeBoundaryGraphCorrectedScalableMotionBases.init_from_state_dict(stale_state_dict_v2, prefix="motion_bases.")
        raise AssertionError("expected KeyError for a state_dict missing the patch-centroid gap schema")
    except KeyError as e:
        print(f"[old-checkpoint] falloff-weighted patch-centroid schema correctly rejected: {e}")

    # --- checkpoint predating the gap_error-free correction schema
    #     (max_displacement) is ALSO rejected -- this is the schema change
    #     from THIS session (correction magnitude capped by max_displacement
    #     instead of gated by contact_reference_distance/gap_error). ---
    stale_state_dict_v3 = {
        k: v for k, v in full_state_dict_v2.items() if k != "motion_bases.max_displacement"
    }
    try:
        EdgeBoundaryGraphCorrectedScalableMotionBases.init_from_state_dict(stale_state_dict_v3, prefix="motion_bases.")
        raise AssertionError("expected KeyError for a state_dict missing max_displacement")
    except KeyError as e:
        print(f"[old-checkpoint] gap_error-free correction schema correctly rejected: {e}")

    # --- 9. refresh_falloff_state: cheap STATE-only refresh (no reassignment,
    #        no nearest-neighbor search) picks up a means/coefs change without
    #        touching falloff_global_idx/edge_id/sign/weight. Uses graph_bases2
    #        (untouched by section 7's densify/cull, so its falloff rows still
    #        index directly into the original canonical_means/boundary_a). ---
    old_assignment = graph_bases2.falloff_global_idx.clone()
    perturbed_means = canonical_means.clone()
    perturbed_means[boundary_a[-1]] += torch.tensor([0.01, 0.0, 0.0])  # simulate a gradient step on means
    graph_bases2.refresh_falloff_state(perturbed_means, coefs)
    same_assignment = bool((graph_bases2.falloff_global_idx == old_assignment).all())
    picked_up_new_mean = bool(
        torch.allclose(graph_bases2.falloff_canonical_mean[graph_bases2.falloff_global_idx == boundary_a[-1]][0], perturbed_means[boundary_a[-1]])
    )
    print(
        f"[refresh-state] assignment unchanged: {same_assignment} (expect True), "
        f"snapshot picked up the perturbed mean: {picked_up_new_mean} (expect True)"
    )
    assert same_assignment
    assert picked_up_new_mean
    graph_bases2.refresh_falloff_state(canonical_means, coefs)  # undo, for cleanliness

    # --- 10. _build_edge_gate: no CONNECTED/DISCONNECTED observation AT ALL
    #         (every sampled frame UNKNOWN) -> gate all 0, not all 1. ---
    gate_all_unknown, connected_mask_all_unknown = _build_edge_gate(
        connected_frames=[], unknown_frames=[0, 1, 2, 3], num_frames=4
    )
    print(f"[gate-all-unknown] gate = {gate_all_unknown.tolist()} (expect all 0.0, never observed)")
    assert bool((gate_all_unknown == 0.0).all())
    assert bool((connected_mask_all_unknown == False).all())  # noqa: E712

    # --- 11. a Gaussian shared by TWO edges (its cluster sits between two
    #         others) must receive BOTH edges' corrections summed -- and
    #         compute_boundary_gap_distances's dist_after for EACH of those
    #         edges must reflect that summed displacement, not just its own
    #         edge's contribution alone. Chain: cluster A -- Hub -- cluster C,
    #         Hub is narrow enough that every one of its points is within
    #         falloff range of BOTH neighbors. ---
    nA, nHub, nC = 30, 6, 30
    xA = torch.linspace(-1.0, -0.05, nA)
    xHub = torch.linspace(-0.05, 0.05, nHub)
    xC = torch.linspace(0.05, 1.0, nC)
    means_chain = torch.cat(
        [
            torch.stack([xA, torch.zeros(nA), torch.zeros(nA)], dim=-1),
            torch.stack([xHub, torch.zeros(nHub), torch.zeros(nHub)], dim=-1),
            torch.stack([xC, torch.zeros(nC), torch.zeros(nC)], dim=-1),
        ],
        dim=0,
    )
    cluster_ids_chain = torch.cat(
        [torch.zeros(nA, dtype=torch.long), torch.ones(nHub, dtype=torch.long), torch.full((nC,), 2, dtype=torch.long)]
    )
    num_fg_chain = means_chain.shape[0]
    hub_start = nA
    hub_idx = torch.arange(hub_start, hub_start + nHub)  # every Hub Gaussian, by construction near BOTH neighbors
    boundary_A = torch.arange(nA - 5, nA)
    boundary_C_local = torch.arange(nC - nC, nC - nC + 5)  # first 5 of cluster C
    boundary_C = hub_start + nHub + boundary_C_local

    edges_payload_chain = {
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edges_kept": [
            {
                "cluster_a": 0, "cluster_b": 1, "reason": "kept_all_frames_contact",
                "boundary_global_indices_a": boundary_A, "boundary_global_indices_b": hub_idx,
            },
            {
                "cluster_a": 1, "cluster_b": 2, "reason": "kept_all_frames_contact",
                "boundary_global_indices_a": hub_idx, "boundary_global_indices_b": boundary_C,
            },
        ],
        "edges_cut": [],
        "cluster_ids": list(range(num_clusters)),
        "meta": {},
    }
    coefs_chain = torch.softmax(torch.randn(num_fg_chain, num_fine), dim=-1)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(edges_payload_chain, f.name)
        chain_bases = EdgeBoundaryGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
            baseline,  # identity coarse transform regardless of centers -- fine to reuse
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edges_path=f.name,
            canonical_means=means_chain,
            cluster_ids_all=cluster_ids_chain,
            coefs_all=coefs_chain,
            gnn_hidden_dim=16,
            gnn_num_layers=1,
            falloff_radius=0.3,
            gap_tolerance=0.05,
        )
    assert chain_bases.num_edges == 2
    hub_falloff_edges = sorted(
        set(chain_bases.falloff_edge_id[chain_bases.falloff_global_idx == int(hub_idx[0])].tolist())
    )
    print(f"[multi-edge] Hub's first Gaussian carries falloff rows for edges: {hub_falloff_edges} (expect [0, 1])")
    assert hub_falloff_edges == [0, 1]

    # force BOTH gaps open and give both edges a nonzero (and DIFFERENT, so a
    # sum-vs-single-edge mixup would actually be caught) correction.
    with torch.no_grad():
        chain_bases.params["transls"][0] += torch.tensor([-0.4, 0.0, 0.0])  # A moves further from Hub -> opens edge 0
        chain_bases.params["transls"][2] += torch.tensor([0.4, 0.0, 0.0])  # C moves further from Hub -> opens edge 1
    chain_bases.zero_grad()
    chain_bases.compute_transforms(ts, coefs_chain, cluster_ids_chain).pow(2).mean().backward()
    with torch.no_grad():
        # edge_mlp's last Linear outputs a single scalar (shared across all
        # edges -- the "1" is the layer's output width, not a per-edge slot),
        # so bumping its bias raises alpha's baseline the same amount for
        # every edge; the two edges' alpha (and hence magnitude) still end up
        # different because the edge features feeding the Linear differ a lot
        # per edge (cluster A/Hub/C sit at very different `centers`, so
        # rel_center/rel_transl and the encoder's h_a/h_b differ), and
        # max_displacement happens to be equal for both edges here (both fall
        # back to falloff_radius * correction_max_disp_scale, since this toy
        # edges_kept payload has no boundary_patch_local_scale_a/b) so any
        # magnitude difference is attributable to alpha, not the cap.
        for p in chain_bases.gnn.edge_mlp[-1].parameters():
            if p.grad is not None:
                p += 3.0

    out_chain = chain_bases.compute_transforms(ts, coefs_chain, cluster_ids_chain)
    ref_chain = ScalableMotionBases.compute_transforms(chain_bases, ts, coefs_chain, cluster_ids_chain)
    actual_disp_chain = torch.einsum(
        "pnij,pj->pni", out_chain, F.pad(means_chain, (0, 1), value=1.0)
    ) - torch.einsum("pnij,pj->pni", ref_chain, F.pad(means_chain, (0, 1), value=1.0))  # (G, B, 3)

    magnitude_chain, direction_chain, _, _ = chain_bases._edge_features_and_direction(ts, detach_base=False)
    print(f"[multi-edge] magnitude edge0={magnitude_chain[0, 0].item():.4f} edge1={magnitude_chain[1, 0].item():.4f} (expect both > 0, different)")
    assert bool((magnitude_chain[:, 0] > 0).all())

    # A single Hub Gaussian's TRUE displacement must equal edge0's contribution
    # (as side b) PLUS edge1's contribution (as side a) -- not either alone.
    g = int(hub_idx[0])
    w_g = chain_bases.falloff_weight[chain_bases.falloff_global_idx == g]  # (2,) one per edge row
    edge_ids_g = chain_bases.falloff_edge_id[chain_bases.falloff_global_idx == g]
    sign_g = chain_bases.falloff_sign[chain_bases.falloff_global_idx == g]
    expected_disp_g = sum(
        0.5 * sign_g[i] * magnitude_chain[edge_ids_g[i], 0] * direction_chain[edge_ids_g[i], 0] * w_g[i]
        for i in range(edge_ids_g.shape[0])
    )
    err_shared = (actual_disp_chain[g, 0] - expected_disp_g).abs().max().item()
    print(f"[multi-edge] shared Hub Gaussian displacement error (both edges summed): {err_shared:.3e} (expect ~0)")
    assert err_shared < 5e-4

    # And compute_boundary_gap_distances's dist_after for BOTH edges must
    # match the ACTUAL rendered gap (ground truth via compute_transforms),
    # not just their own edge's contribution in isolation. Use the REAL
    # falloff assignment (chain_bases.falloff_*, which side_a/side_b actually
    # ended up including -- not our guessed boundary_A/hub_idx/boundary_C sets,
    # since falloff_radius=0.3 pulls in more of cluster A/C than just their
    # nominal "boundary" points) so this is a check of the AFTER-position
    # math specifically, not a restatement of the (separately tested)
    # assignment logic.
    result_chain = compute_boundary_gap_distances(chain_bases, ts)
    pos_after_chain = torch.einsum("pnij,pj->pni", out_chain, F.pad(means_chain, (0, 1), value=1.0))
    for e in range(2):
        row_mask_e = chain_bases.falloff_edge_id == e
        idx_a_e = chain_bases.falloff_global_idx[row_mask_e & (chain_bases.falloff_sign > 0)]
        idx_b_e = chain_bases.falloff_global_idx[row_mask_e & (chain_bases.falloff_sign < 0)]
        w_a_e = chain_bases.falloff_weight[row_mask_e & (chain_bases.falloff_sign > 0)]
        w_b_e = chain_bases.falloff_weight[row_mask_e & (chain_bases.falloff_sign < 0)]
        mean_a_e = (pos_after_chain[idx_a_e] * w_a_e[:, None, None]).sum(0) / w_a_e.sum()
        mean_b_e = (pos_after_chain[idx_b_e] * w_b_e[:, None, None]).sum(0) / w_b_e.sum()
        dist_after_gt_e = (mean_b_e - mean_a_e).norm(dim=-1)
        err_e = (result_chain["dist_after"][e] - dist_after_gt_e).abs().max().item()
        print(f"[multi-edge] edge {e} dist_after vs. ground truth: max err={err_e:.3e} (expect ~0)")
        assert err_e < 1e-4

    print("\nAll graph_relative_edge.py sanity checks passed.")

    print("OK")

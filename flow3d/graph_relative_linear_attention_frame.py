"""
flow3d/graph_relative_linear_attention_frame.py

`flow3d/graph_relative_linear_attention.py`
(RelativeVelLinearAttentionGraphCorrectedScalableMotionBases)의 드롭인
variant: cluster별 (omega, delta_t) correction의 node/edge feature, linear
velocity 계산, multi-head attention 구조, 6D head(0-init), omega/delta_t
합성 수식(exp map compose + translation add), coarse-to-fine 결합은 baseline과
**완전히 동일**하다. 유일한 차이는 attention과 최종 correction이 이제 그
edge/cluster가 "지금 이 프레임"에 실제로 관측된 연결 상태
(CONNECTED/DISCONNECTED/UNKNOWN)를 반영한다는 점이다.

frame gate
----------
`flow3d/graph_relative_edge.py`의 `load_edge_frame_gates`가 edges.pt의
`connected_frame_indices`/`unknown_frame_indices`를 읽어 만드는
per-edge/per-frame gate(CONNECTED=1, DISCONNECTED=0, UNKNOWN은 관측된 매듭
사이만 선형보간, 관측 범위 밖은 0 -- `_build_edge_gate` 참고)를 그대로
재사용한다. boundary index/falloff weight/patch centroid/alpha/magnitude/고정
pull 방향 등 `graph_relative_edge.py`의 나머지(edge별 국소 displacement
correction)는 이 파일과 무관하며 전혀 가져오지 않는다.

undirected edge (a, b)를 양방향 message edge (a->b, b->a)로 바꿀 때 같은 gate
행을 양쪽에 그대로 복제한다(`_build_directed_neighbor_edges_and_gate`) --
a->b와 b->a는 항상 같은 연결 상태를 쓴다.

attention: gate를 softmax 이전에 반영 (log-gate trick)
--------------------------------------------------------
`_compute_attention`은 baseline의 dense-masked softmax
((C, C, B, heads) 텐서를 -inf로 채우고 존재하는 edge 위치에만 score를 채운 뒤
neighbor 축에서 softmax)를 그대로 쓰되, dense score에 채우기 직전에 그 edge의
이번 프레임 gate의 로그를 raw score에 더한다:

    score'_ij = score_ij + log(gate_ij)     (gate_ij == 0이면 -inf)
    alpha_ij  = softmax_{j in N(i)}(score'_ij)
              = gate_ij * exp(score_ij) / sum_k gate_ik * exp(score_ik)

이게 "softmax를 먼저 계산한 뒤 gate를 곱하는" (틀린) 방식과 다른 이유는,
곱셈이 정규화 **이전**에 들어가 활성(gate>0) 이웃끼리 다시 정규화되기
때문이다(활성 이웃의 alpha 합은 항상 1). CONNECTED(gate=1)는 `log(1)=0`이라
score가 전혀 바뀌지 않아 baseline과 100% 동일하고, UNKNOWN(0<gate<1)은
`log(gate)<0`만큼 감점되어 비중이 줄어들되 완전히 배제되지는 않으며,
DISCONNECTED(gate=0)는 `log(0)=-inf`가 되어 존재하지 않는 edge와 정확히 같은
메커니즘으로 softmax 후보에서 완전히 제외된다. 한 cluster의 모든 in-neighbor가
이번 프레임에 비활성(DISCONNECTED거나 아예 이웃이 없음)이면 그 행이 전부
-inf가 된다. baseline은 이 경우 `torch.softmax` + `torch.nan_to_num(alpha,
nan=0.0)`로 처리하는데, 이 조합은 forward 값은 정확히 0으로 고쳐주지만
`torch.softmax`가 내부적으로 하는 `x - max(x)`가 `-inf - (-inf) = NaN`이
되고 softmax의 backward 공식이 그 NaN을 담은 raw output을 그대로 재사용하기
때문에 **gradient는 NaN으로 남는다** -- `nan_to_num`은 그 NaN을 forward에서만
가릴 뿐 backward 그래프에서 지우지 못한다. baseline은 이게 "edge가 전혀 없는
고립 cluster"에서만 일어나 거의 드러나지 않지만, 이 frame variant는 "edge는
있지만 이번 프레임에 전부 DISCONNECTED"인 경우가 실제 학습 데이터에서 흔해
(실측: 실제 edges.pt로 학습 시 특정 프레임 구간에서 여러 cluster가 동시에
이 상태가 되어 `loss.backward()`에서 곧바로 NaN이 발생했다) 이 버그가 훨씬
자주 발생한다. 그래서 `_compute_attention`은 `torch.softmax`/`nan_to_num`
대신 masked softmax를 **직접** 계산한다: 행의 max가 -inf(=행 전체가 -inf)이면
그 max를 유한한 placeholder(0)로 바꿔서 뺀다 -- `exp(-inf - 0) == 0`이 모든
원소에서 정확히 성립하므로 forward/backward 어디에도 NaN이 생기지 않고, 결과는
동일하게 정확히 0이다(`agg_i = 0`, `h_i' = h_i + act(0)`, residual만 유지 --
baseline의 "고립 노드" 처리 경로가 "이번 프레임에 활성 이웃이 없는 노드" 처리도
자연스럽게 겸한다는 의도는 그대로다). 적어도 하나의 유한한 entry가 있는 행은
`torch.softmax`와 비트 단위로 동일한 결과를 낸다.

cluster gate: incident edge gate의 최댓값
-------------------------------------------
attention에서 DISCONNECTED edge를 제외하는 것만으로는 correction이 완전히
0이 되지 않는다 -- 활성 이웃이 하나도 없어도 encoder와 residual hidden
feature를 거쳐 head가 omega/delta_t를 출력할 수 있기 때문이다. 그래서 GNN
head가 낸 raw omega/delta_t에 **cluster gate**(그 cluster에 연결된 모든
incident edge의 이번 프레임 gate 중 최댓값)를 곱한다. 그러면 하나라도
CONNECTED edge가 있으면 cluster gate가 1이 되어 그 cluster 전체의 correction이
켜지고(사용자가 확정한 이번 variant의 정의: "활성 edge가 하나라도 있는
cluster 전체를 보정한다" -- 특정 edge boundary만 움직이는 것은 이후 별도
variant의 몫), UNKNOWN edge만 있으면 그 보간값만큼만 켜지며, 모든 incident
edge가 DISCONNECTED면 정확히 0이 된다. incident edge가 하나도 없는(그래프에서
완전히 고립된) cluster는 관측된 연결이 없는 것과 동일하게 0으로 둔다.

head는 baseline과 완전히 동일하게 0-init이므로, gate/attention 구조와 무관하게
학습 시작 시점의 omega/delta_t는 정확히 0이다(`0 * cluster_gate == 0`은
cluster_gate 값과 무관하게 항상 성립) -- zero-init 등가성은 gate 도입과
무관하게 그대로 보존된다.

공통 기반
---------
- so3_exp_map / compose_rotation / cont_6d_to_rmat / rmat_to_cont_6d /
  CENTER_DIM / ROT_DIM / TRANSL_DIM / NODE_FEAT_DIM은
  flow3d/graph_coupling.py / flow3d/transforms.py에서 그대로 가져다 쓴다.
- load_edge_frame_gates는 flow3d/graph_relative_edge.py에서 가져다 쓴다(gate
  생성 로직만 재사용 -- 그 파일 자체는 수정하지 않는다).
- coarse correction 합성 수식, coarse-to-fine 결합 수식, checkpoint save/load
  포맷(directed edge_index_dir 저장 방식)은 baseline
  graph_relative_linear_attention.py와 동일한 패턴을 따른다.

이 파일에서 제공하는 것
------------------------
- RelativeVelLinearAttentionFrameClusterGraphGNN: baseline
  RelativeVelLinearAttentionClusterGraphGNN과 같은 API(node feature ->
  (omega, delta_t))를 갖지만, forward가 추가로 `edge_gate_dir (E_dir, B)`를
  받아 attention 마스킹 + cluster gate 곱셈에 쓴다.
- RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases: baseline
  RelativeVelLinearAttentionGraphCorrectedScalableMotionBases와 동일한 API
  (from_scalable_motion_bases, has_gnn_state, init_from_state_dict,
  compute_transforms_coarse, compute_transforms, last_correction)를 가진
  드롭인 대체. `from_scalable_motion_bases`는 baseline과 달리 `edges_path`도
  받아(gate를 읽기 위해) 내부에서 `load_edge_frame_gates`를 호출한다.
  `last_correction`에는 baseline의 `{"omega", "delta_t"}`에 더해
  `"edge_gate_dir"`(edge-consistency loss 가중치용)과 `"cluster_gate"`(디버깅용)
  가 추가된다 -- 한 번의 forward 결과를 항상 이 dict 하나로 원자적으로
  갱신하므로, 같은 loss 계산 안에서 `_corrected_coarse`가 여러 ts로 여러 번
  불려도(예: 현재 배치 ts 한 번, temporal smoothness triplet용 한 번)
  correction과 그 gate가 서로 다른 호출에서 섞일 수 없다.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.graph_coupling import (
    CENTER_DIM,
    NODE_FEAT_DIM,
    ROT_DIM,
    TRANSL_DIM,
    compose_rotation,
    so3_exp_map,
)
from flow3d.graph_relative_edge import load_edge_frame_gates
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "so3_exp_map",
    "compose_rotation",
    "load_edge_frame_gates",
    "RelativeVelLinearAttentionFrameClusterGraphGNN",
    "RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases",
]

_LEAKY_SLOPE = 0.2
VEL_DIM = 3
EDGE_FEAT_DIM = TRANSL_DIM + CENTER_DIM + ROT_DIM
NODE_FEAT_DIM_VEL = NODE_FEAT_DIM + VEL_DIM
EDGE_FEAT_DIM_VEL = EDGE_FEAT_DIM + VEL_DIM


def _compute_vel_scale(transls: torch.Tensor) -> float:
    """Return a stable scale for frame-to-frame coarse translation deltas.
    Identical to graph_relative_linear_attention.py's helper of the same
    name."""
    if transls.shape[1] < 2:
        return 1.0
    scale = (transls[:, 1:] - transls[:, :-1]).std().item()
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _undirected_pair_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """(2, E) -> order-independent set of canonicalized (min, max) cluster-id
    pairs. Used only to compare two edge_index tensors' TOPOLOGY without
    requiring identical row order -- build_edge_index_from_edges_pt and
    load_edge_frame_gates may enumerate the same edges.pt's edges_kept in
    different orders even though they describe the exact same graph."""
    if edge_index.numel() == 0:
        return set()
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    return set(zip(lo.tolist(), hi.tolist()))


def _build_directed_neighbor_edges_and_gate(
    edge_index: torch.Tensor,
    edge_gate: torch.Tensor,
    num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert undirected cluster pairs + their per-frame gate into
    deduplicated bidirectional directed edges, duplicating each pair's gate
    row to both directions (a->b and b->a always share the exact same gate --
    module docstring).

    Unlike graph_relative_linear_attention.py's _build_directed_neighbor_edges,
    this does NOT run the result through torch.unique -- reordering rows there
    would break the 1:1 row correspondence with edge_gate. Instead it assumes
    edge_index has no duplicate undirected pair (true for edges.pt-derived
    edge_index, e.g. load_edge_frame_gates's output) and raises if it finds
    one, rather than silently deduplicating.

    :param edge_index: (2, E) [cluster_a; cluster_b].
    :param edge_gate: (E, num_frames), same row order as edge_index's columns.
    :param num_clusters: kept for call-site symmetry with the baseline helper
        (unused -- cluster ids are validated by the caller).
    :return: (edge_index_dir (2, 2E'), edge_gate_dir_full (2E', num_frames)),
        E' = E minus any self-loops (cluster_a == cluster_b).
    """
    del num_clusters
    if edge_index.numel() == 0:
        num_frames = edge_gate.shape[1] if edge_gate.dim() == 2 else 0
        return (
            torch.empty(2, 0, dtype=torch.long, device=edge_index.device),
            edge_gate.new_zeros(0, num_frames),
        )

    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst, gate = src[keep], dst[keep], edge_gate[keep]

    pairs = _undirected_pair_set(torch.stack([src, dst], dim=0))
    if len(pairs) != int(src.shape[0]):
        raise ValueError(
            "edge_index contains a duplicate undirected cluster pair -- "
            "_build_directed_neighbor_edges_and_gate requires each pair to "
            "appear exactly once (as edges.pt's edges_kept does)."
        )

    edge_index_dir = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_gate_dir_full = torch.cat([gate, gate], dim=0)
    return edge_index_dir, edge_gate_dir_full


class _RelativeVelLinearAttentionFrameMessageLayer(nn.Module):
    """graph_relative_linear_attention.py's
    _RelativeVelLinearAttentionMessageLayer와 동일한 message MLP + multi-head
    GAT-style attention이지만, 이번 프레임의 per-directed-edge gate
    `edge_gate_dir (E, B)`를 raw attention score에 softmax **이전** 로그로
    더한다 (module docstring 참고) -- CONNECTED(gate=1)는 완전히 동일한 동작,
    UNKNOWN은 down-weight, DISCONNECTED(gate=0)는 -inf로 완전히 배제.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + EDGE_FEAT_DIM_VEL, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.ReLU()

        self.W_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_edge = nn.Linear(EDGE_FEAT_DIM_VEL, hidden_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)

    def _compute_attention(
        self,
        h: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_gate_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same dense-masked multi-head attention as the baseline, plus the
        log-gate term added to the raw score before the dense scatter+softmax.

        :param h: (C, B, H) node hidden features
        :param edge_index_dir: (2, E) directed [src(j), dst(i)] pairs, self-loop-free
        :param edge_feat: (E, B, EDGE_FEAT_DIM_VEL) relative geometry+velocity per edge
        :param edge_gate_dir: (E, B) this frame's per-directed-edge gate in [0, 1]
        :returns: alpha_dense (C, C, B, heads), mask (C, C) bool adjacency
                  (mask[i, j] == True iff edge j->i exists in the FIXED
                  topology -- independent of this frame's gate).
        """
        C, B, H = h.shape
        K, D = self.num_heads, self.head_dim
        device = h.device

        if edge_index_dir.numel() == 0:
            alpha_dense = h.new_zeros(C, C, B, K)
            mask = torch.zeros(C, C, dtype=torch.bool, device=device)
            return alpha_dense, mask

        src, dst = edge_index_dir[0], edge_index_dir[1]

        proj = self.W_score(h).view(C, B, K, D)  # (C, B, K, D)
        edge_proj = self.W_edge(edge_feat).view(-1, B, K, D)  # (E, B, K, D)

        proj_i = proj[dst]  # (E, B, K, D)
        proj_j = proj[src]  # (E, B, K, D)
        concat = torch.cat([proj_i, proj_j, edge_proj], dim=-1)  # (E, B, K, 3D)
        raw_score = (concat * self.attn).sum(-1)  # (E, B, K)
        score_e = F.leaky_relu(raw_score, negative_slope=_LEAKY_SLOPE)  # (E, B, K)

        # Gate applied BEFORE softmax (log-space), not after: alpha_ij ends up
        # exactly gate_ij * exp(score_ij) / sum_k gate_ik * exp(score_ik), so
        # active (gate > 0) neighbors' alpha still sums to 1 -- see module
        # docstring for why "softmax first, multiply by gate after" is wrong.
        log_gate = torch.where(
            edge_gate_dir > 0,
            torch.log(edge_gate_dir.clamp_min(1e-12)),
            torch.full_like(edge_gate_dir, float("-inf")),
        )  # (E, B)
        score_e = score_e + log_gate.unsqueeze(-1)  # broadcast over heads

        score_dense = h.new_full((C, C, B, K), float("-inf"))
        score_dense[dst, src] = score_e  # dense[i, j] = s'_{j->i}

        # Rows with no active (gate>0) in-neighbor -- either genuinely no edge,
        # or every incident edge DISCONNECTED this frame (routine with real
        # per-frame gates, unlike the baseline attention variant where this
        # only happens for a genuinely edge-less cluster) -- are entirely
        # -inf. torch.softmax(score_dense, dim=1) followed by
        # torch.nan_to_num(alpha, nan=0.0) gets the FORWARD value right
        # (exactly 0), but softmax's internal max-subtraction computes
        # `-inf - (-inf) = NaN` for such a row, and its backward formula
        # reuses that raw (NaN) output -- so nan_to_num does NOT stop NaN
        # gradients from reaching score_dense and poisoning the whole batch's
        # loss.backward() (confirmed against real training data: several
        # clusters have every incident edge DISCONNECTED within a single
        # frame window, crashing with "Loss is NaN" on the very first
        # backward call). Compute the masked softmax manually instead: for a
        # fully -inf row, replace its (also -inf) max with a finite
        # placeholder (0) before subtracting, so `exp(-inf - 0) == 0` exactly
        # for every entry -- never NaN, forward or backward -- giving the
        # same "exactly 0" result without ever touching nan_to_num. Rows with
        # at least one finite entry reduce to the exact same computation
        # torch.softmax performs (verified bit-for-bit).
        row_max = score_dense.detach().max(dim=1, keepdim=True).values  # (C, 1, B, K)
        row_max_safe = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        exp_score = torch.exp(score_dense - row_max_safe)
        denom = exp_score.sum(dim=1, keepdim=True).clamp_min(1e-30)
        alpha_dense = exp_score / denom  # normalize over j (neighbor) axis

        mask = torch.zeros(C, C, dtype=torch.bool, device=device)
        mask[dst, src] = True

        return alpha_dense, mask

    def forward(
        self,
        h: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_gate_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (C, B, H) node hidden features
        edge_index_dir: (2, E) directed [src(j), dst(i)] pairs, self-loop-free
        edge_feat: (E, B, EDGE_FEAT_DIM_VEL) relative geometry+velocity feature
                   per directed edge (shared across layers within one forward call)
        edge_gate_dir: (E, B) this frame's per-directed-edge gate in [0, 1]
                   (shared across layers, sliced once per forward call)
        """
        C, B, H = h.shape
        if edge_index_dir.numel() == 0:
            return h + self.act(h.new_zeros(C, B, H))

        src, dst = edge_index_dir[0], edge_index_dir[1]

        alpha_dense, _mask = self._compute_attention(h, edge_index_dir, edge_feat, edge_gate_dir)
        alpha_e = alpha_dense[dst, src]  # (E, B, K), gather back to edge order

        delta_h = h[src] - h[dst]  # (E, B, H), Δh_ij = h_j - h_i
        m = self.mlp(torch.cat([delta_h, edge_feat], dim=-1))  # (E, B, H)
        m_heads = m.view(-1, B, self.num_heads, self.head_dim)  # (E, B, K, D)
        weighted = alpha_e.unsqueeze(-1) * m_heads  # (E, B, K, D)
        weighted_flat = weighted.reshape(-1, B, H)  # concat heads back to H

        agg = h.new_zeros(C, B, H)
        agg.index_add_(0, dst, weighted_flat)

        return h + self.act(agg)


class RelativeVelLinearAttentionFrameClusterGraphGNN(nn.Module):
    """graph_relative_linear_attention.py's
    RelativeVelLinearAttentionClusterGraphGNN과 같은 API(node feature ->
    (omega, delta_t))를 갖지만, forward가 추가로 이번 프레임의
    per-directed-edge gate `edge_gate_dir (E, B)`를 받아 (1) attention 마스킹
    (module docstring), (2) 최종 raw omega/delta_t에 곱하는 cluster gate(그
    cluster에 연결된 모든 incident edge gate의 최댓값)에 쓴다. node/edge
    feature(linear velocity 포함)는 baseline과 완전히 동일하다.

    edge_index_dir/edge_gate_dir_full은 생성 시점에 고정돼 학습 중 바뀌지
    않는다(topology와 "그 edge가 무엇을 의미하는지"는 고정, 실제로 어느
    프레임에서 얼마나 열려 있는지는 gate로 매 forward마다 반영). head는
    0으로 초기화되므로 학습 시작 시점에는 correction이 정확히 0이다(baseline과
    동일한 zero-init 등가성).
    """

    def __init__(
        self,
        edge_index_dir: torch.Tensor,
        edge_gate_dir_full: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vel_scale: float = 1.0,
    ):
        """
        :param edge_index_dir: (2, E_dir) already-directed [src, dst] pairs,
            self-loop-free, bidirectional (both (a, b) and (b, a) present for
            every undirected pair) -- e.g. from
            _build_directed_neighbor_edges_and_gate. Not re-derived from an
            undirected edge_index here (that conversion happens once, in the
            caller -- see
            RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases).
        :param edge_gate_dir_full: (E_dir, num_frames), row-aligned with
            edge_index_dir's columns -- the full per-frame gate for every
            directed edge (both directions of a pair share the same row).
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.register_buffer("edge_index_dir", edge_index_dir)
        self.register_buffer("edge_gate_dir_full", edge_gate_dir_full)

        if not math.isfinite(vel_scale) or vel_scale <= 0.0:
            vel_scale = 1.0
        self.register_buffer("vel_scale", torch.tensor(float(vel_scale)))

        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM_VEL, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [
                _RelativeVelLinearAttentionFrameMessageLayer(hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, ROT_DIM // 2 + TRANSL_DIM)  # (omega(3), delta_t(3))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_edge_features(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        vel_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """Identical to graph_relative_linear_attention.py's
        RelativeVelLinearAttentionClusterGraphGNN._compute_edge_features --
        directed edge j->i마다
        e_ij = [t_j - t_i, c_j - c_i, rot6d(R_i^T @ R_j), v_j/vel_scale - v_i/vel_scale].
        """
        B = coarse_transl.shape[1]
        if self.edge_index_dir.numel() == 0:
            return coarse_rot_6d.new_zeros(0, B, EDGE_FEAT_DIM_VEL)

        src, dst = self.edge_index_dir[0], self.edge_index_dir[1]

        rel_transl = coarse_transl[src] - coarse_transl[dst]  # (E, B, 3)
        rel_center = centers[src] - centers[dst]  # (E, 3)
        rel_center = rel_center[:, None, :].expand(-1, B, -1)  # (E, B, 3)

        coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)  # (C, B, 3, 3)
        R_i = coarse_rotmats[dst]  # (E, B, 3, 3)
        R_j = coarse_rotmats[src]  # (E, B, 3, 3)
        R_rel = torch.matmul(R_i.transpose(-1, -2), R_j)
        rel_rot6d = rmat_to_cont_6d(R_rel)  # (E, B, 6)

        rel_vel = vel_scaled[src] - vel_scaled[dst]  # (E, B, 3)

        return torch.cat([rel_transl, rel_center, rel_rot6d, rel_vel], dim=-1)  # (E, B, 15)

    def _cluster_gate(self, edge_gate_dir: torch.Tensor, C: int, B: int) -> torch.Tensor:
        """(E_dir, B) directed gate -> (C, B): for each cluster, the max gate
        over all its incident edges (both directions of edge_index_dir are
        present, so scattering by `dst` alone already covers every incident
        edge of every cluster). A cluster with no incident edge at all
        (isolated in the graph) gets exactly 0 -- treated the same as a
        connection that was never observed."""
        cluster_gate = edge_gate_dir.new_zeros(C, B)
        if self.edge_index_dir.numel() == 0:
            return cluster_gate
        dst = self.edge_index_dir[1]
        index = dst.unsqueeze(-1).expand(-1, B)
        return cluster_gate.scatter_reduce(0, index, edge_gate_dir, reduce="amax", include_self=True)

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        coarse_vel: torch.Tensor,
        edge_gate_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        coarse_rot_6d: (C, B, 6)
        coarse_transl: (C, B, 3)
        centers: (C, 3) canonical cluster centers
        coarse_vel: (C, B, 3) raw (unnormalized) linear velocity
        edge_gate_dir: (E_dir, B) this batch's per-directed-edge gate
        returns: omega (C, B, 3), delta_t (C, B, 3), cluster_gate (C, B)
        """
        if coarse_rot_6d.shape[0] != self.num_clusters:
            raise ValueError(
                f"coarse_rot_6d has {coarse_rot_6d.shape[0]} clusters, "
                f"expected {self.num_clusters}"
            )

        C, B, _ = coarse_rot_6d.shape
        center_feat = centers[:, None, :].expand(C, B, CENTER_DIM)
        vel_scaled = coarse_vel / self.vel_scale
        node_feat = torch.cat(
            [coarse_rot_6d, coarse_transl, center_feat, vel_scaled], dim=-1
        )  # (C, B, 15)

        h = self.encoder(node_feat)
        edge_feat = self._compute_edge_features(coarse_rot_6d, coarse_transl, centers, vel_scaled)
        for layer in self.layers:
            h = layer(h, self.edge_index_dir, edge_feat, edge_gate_dir)

        raw = self.head(h)  # (C, B, 6)
        raw_omega, raw_delta_t = raw.split([3, 3], dim=-1)

        cluster_gate = self._cluster_gate(edge_gate_dir, C, B)  # (C, B)
        omega = raw_omega * cluster_gate.unsqueeze(-1)
        delta_t = raw_delta_t * cluster_gate.unsqueeze(-1)
        return omega, delta_t, cluster_gate


class RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + relative-message-with-velocity-attention graph-aware
    coarse transform correction, gated by per-frame edge connectivity
    (module docstring). API-compatible drop-in for baseline
    RelativeVelLinearAttentionGraphCorrectedScalableMotionBases, plus
    `from_scalable_motion_bases` additionally taking `edges_path` (to load the
    gate) and `last_correction` carrying two extra keys (`"edge_gate_dir"`,
    `"cluster_gate"`).
    """

    def __init__(
        self,
        centers: torch.Tensor,
        rots: torch.Tensor,
        transls: torch.Tensor,
        fine_rots: torch.Tensor,
        fine_transls: torch.Tensor,
        edge_index: torch.Tensor,
        edge_gate: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
    ):
        """
        :param edge_index: (2, E) [cluster_a; cluster_b], undirected, no
            duplicate pairs -- e.g. from load_edge_frame_gates.
        :param edge_gate: (E, num_frames), row-aligned with edge_index's
            columns -- e.g. from the SAME load_edge_frame_gates call (this is
            the only case this constructor is meant to be called with; the
            row alignment is not re-checked beyond the shape[0] match below).
        """
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        if edge_index.shape[1] != edge_gate.shape[0]:
            raise ValueError(
                f"edge_index has {edge_index.shape[1]} edges but edge_gate has "
                f"{edge_gate.shape[0]} rows -- both must come from the same "
                "load_edge_frame_gates(edges_path) call."
            )
        edge_index_dir, edge_gate_dir_full = _build_directed_neighbor_edges_and_gate(
            edge_index, edge_gate, self.num_clusters
        )
        vel_scale = _compute_vel_scale(transls)
        self.gnn = RelativeVelLinearAttentionFrameClusterGraphGNN(
            edge_index_dir=edge_index_dir,
            edge_gate_dir_full=edge_gate_dir_full,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            vel_scale=vel_scale,
        )
        # One _corrected_coarse() call's (omega, delta_t, edge_gate_dir,
        # cluster_gate) are always stored together as a single dict (module
        # docstring) -- never split across separate attributes, so a later
        # _corrected_coarse call (e.g. for the temporal-smoothness triplet)
        # can never leave this dict's pieces out of sync with each other.
        self._last_correction: dict[str, torch.Tensor] | None = None

    @classmethod
    def _new_from_directed(
        cls,
        base: ScalableMotionBases,
        edge_index_dir: torch.Tensor,
        edge_gate_dir_full: torch.Tensor,
        gnn_hidden_dim: int,
        gnn_num_layers: int,
        gnn_num_heads: int,
    ) -> "RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases":
        """Low-level constructor used only by init_from_state_dict: builds
        self.gnn directly from already-directed buffers (as saved in a
        checkpoint), bypassing __init__'s undirected edge_index/edge_gate ->
        directed conversion entirely (that conversion also rejects duplicate
        undirected pairs, which an already-bidirectional directed edge list
        would trigger if run through it again) -- and bypassing
        from_scalable_motion_bases's edges_path requirement, since a
        checkpoint already has the gate baked into edge_gate_dir_full and
        does not need to re-read edges.pt."""
        p = base.params
        self = cls.__new__(cls)
        ScalableMotionBases.__init__(
            self,
            p["centers"].detach().clone(),
            p["rots"].detach().clone(),
            p["transls"].detach().clone(),
            p["fine_rots"].detach().clone(),
            p["fine_transls"].detach().clone(),
        )
        vel_scale = _compute_vel_scale(p["transls"])
        self.gnn = RelativeVelLinearAttentionFrameClusterGraphGNN(
            edge_index_dir=edge_index_dir,
            edge_gate_dir_full=edge_gate_dir_full,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            vel_scale=vel_scale,
        )
        self._last_correction = None
        return self

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        edges_path: str | Path,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
    ) -> "RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 relative+velocity+attention+
        frame-gate graph-corrected 버전을 만든다. coarse/fine motion 파라미터
        값은 그대로 복사되고, GNN correction만 새로 추가된다(0으로 초기화됨).

        :param edge_index: 다른 *GraphCorrectedScalableMotionBases와
            call-site를 맞추기 위해 받지만, 실제 topology/gate는 edges_path에서
            load_edge_frame_gates가 만든 (edge_index, gate) 쌍을 단일 기준으로
            쓴다 -- build_edge_index_from_edges_pt가 만드는 edge_index는 이
            함수의 결과와 행 순서가 다를 수 있으므로(모듈 docstring, Part 1의
            load_edge_frame_gates 참고) 그 정합성에 의존하지 않는다. 여기서는
            두 edge_index가 같은 edge *집합*(순서 무관, {cluster_a, cluster_b}
            unordered pair 기준)을 가리키는지만 검증한다.
        :param edges_path: build_cluster_graph.py's edges.pt 경로.
        """
        p = bases.params
        num_clusters = p["centers"].shape[0]
        num_frames = p["transls"].shape[1]
        loaded_edge_index, edge_gate = load_edge_frame_gates(
            edges_path,
            num_clusters=num_clusters,
            num_frames=num_frames,
            device=p["centers"].device,
        )
        if _undirected_pair_set(edge_index) != _undirected_pair_set(loaded_edge_index):
            raise ValueError(
                "edge_index and load_edge_frame_gates(edges_path) disagree on "
                "which cluster pairs are edges -- both must be built from the "
                "same edges.pt."
            )
        return cls(
            centers=p["centers"].detach().clone(),
            rots=p["rots"].detach().clone(),
            transls=p["transls"].detach().clone(),
            fine_rots=p["fine_rots"].detach().clone(),
            fine_transls=p["fine_transls"].detach().clone(),
            edge_index=loaded_edge_index,
            edge_gate=edge_gate,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_num_heads=gnn_num_heads,
        )

    @staticmethod
    def has_gnn_state(state_dict: dict, prefix: str) -> bool:
        """state_dict의 motion_bases가
        RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases로
        저장된 것인지 확인한다. `edge_gate_dir_full` 버퍼는 baseline attention
        variant(vel_scale은 있지만 이 버퍼는 없음)에는 없으므로 이 마커로 두
        포맷을 구분할 수 있다. prefix는 "motion_bases." 처럼 끝에 점이 붙은
        형태여야 한다."""
        return f"{prefix}gnn.edge_gate_dir_full" in state_dict

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases":
        """체크포인트만으로
        RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases를
        통째로 복원한다. edge_index_dir/edge_gate_dir_full/hidden_dim/
        num_layers/num_heads는 저장된 텐서의 shape에서 읽어온다(baseline과
        동일한 패턴 -- num_heads는 저장된 attention 벡터 "layers.0.attn"의
        shape[0]). `_new_from_directed`로 이미 directed인 저장된 버퍼를 그대로
        생성자에 넘긴 뒤, 아래 load_state_dict가 strict하게 최종 값을
        덮어쓴다."""
        base = ScalableMotionBases.init_from_state_dict(
            state_dict, prefix=f"{prefix}params."
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
        num_heads = state_dict[f"{layers_prefix}0.attn"].shape[0]

        edge_index_dir = state_dict[f"{gnn_prefix}edge_index_dir"]
        edge_gate_dir_full = state_dict[f"{gnn_prefix}edge_gate_dir_full"]

        graph_bases = cls._new_from_directed(
            base,
            edge_index_dir=edge_index_dir,
            edge_gate_dir_full=edge_gate_dir_full,
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
            gnn_num_heads=num_heads,
        )

        gnn_state = {
            key[len(gnn_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(gnn_prefix)
        }
        graph_bases.gnn.load_state_dict(gnn_state, strict=True)

        return graph_bases

    def _corrected_coarse(
        self, ts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """coarse rotation/translation에 (gate 적용된) GNN correction을
        compose한다. 회전은 compose(exp(omega) @ R_coarse)이고, translation만
        덧셈이다 -- baseline과 완전히 동일한 compose 수식. velocity와 이번
        배치의 directed edge gate 슬라이스는 여기서만 계산된다: ts/self.params/
        self.gnn.edge_gate_dir_full에 동시에 접근할 수 있는 유일한 지점이기
        때문이다."""
        coarse_rot_6d = self.params["rots"][:, ts]  # (C, B, 6)
        coarse_transl = self.params["transls"][:, ts]  # (C, B, 3)
        centers = self.params["centers"]  # (C, 3)

        ts_prev = (ts - 1).clamp(min=0)
        coarse_transl_prev = self.params["transls"][:, ts_prev]  # (C, B, 3)
        coarse_vel = coarse_transl - coarse_transl_prev  # (C, B, 3)

        # This batch's ts slices the full (E_dir, num_frames) gate down to
        # (E_dir, B) -- the "dynamic edge gate" the module docstring refers to.
        edge_gate_dir = self.gnn.edge_gate_dir_full[:, ts]  # (E_dir, B)

        omega, delta_t, cluster_gate = self.gnn(
            coarse_rot_6d, coarse_transl, centers, coarse_vel, edge_gate_dir
        )
        self._last_correction = {
            "omega": omega,
            "delta_t": delta_t,
            "edge_gate_dir": edge_gate_dir,
            "cluster_gate": cluster_gate,
        }

        coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)  # (C, B, 3, 3)
        R_correction = so3_exp_map(omega)  # (C, B, 3, 3)
        corrected_rotmats = compose_rotation(R_correction, coarse_rotmats)  # exp(omega) @ R_coarse
        corrected_transl = coarse_transl + delta_t

        return corrected_rotmats, corrected_transl

    @property
    def last_correction(self) -> dict[str, torch.Tensor] | None:
        """가장 최근 forward에서 나온 (omega, delta_t, edge_gate_dir,
        cluster_gate) correction (디버깅/로깅 + edge-consistency loss
        가중치용)."""
        return self._last_correction

    def compute_transforms_coarse(
        self, ts: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        :param ts (B)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        coarse_rotmats, coarse_transls = self._corrected_coarse(ts)
        centers = self.params["centers"]  # (C, 3)

        transls_eff = (
            -torch.einsum("cbij,cj->cbi", coarse_rotmats, centers)
            + centers[:, None]
            + coarse_transls
        )  # (C, B, 3)

        return torch.cat(
            [coarse_rotmats[cluster_ids], transls_eff[cluster_ids].unsqueeze(-1)],
            dim=-1,
        )

    def compute_transforms(
        self, ts: torch.Tensor, coefs: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """coarse-to-fine 결합 수식은 ScalableMotionBases.compute_transforms /
        baseline RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.compute_transforms와
        완전히 동일하며, coarse rotation/translation만 gate-aware correction이
        compose된 값으로 바뀐다. fine motion 코드는 한 글자도 바뀌지 않는다.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        assert coefs.shape[0] == cluster_ids.shape[0]
        G, F_dim = coefs.shape
        C, _, B, _ = self.params["fine_transls"][:, :, ts].shape

        coarse_rotmats, coarse_transls = self._corrected_coarse(ts)
        centers = self.params["centers"]  # (C, 3)

        fine_transls = self.params["fine_transls"][:, :, ts]  # (C, F, B, 3)
        fine_rots = self.params["fine_rots"][:, :, ts]  # (C, F, B, 6)

        fine_rotmats = cont_6d_to_rmat(fine_rots)  # (C, F, B, 3, 3)
        total_rotmats = torch.einsum(
            "cbij,cfbjk->cfbik", coarse_rotmats, fine_rotmats
        )  # (C, F, B, 3, 3)
        total_6d = total_rotmats[..., :, :2].transpose(-1, -2).reshape(
            C, F_dim, B, 6
        )

        R_total_c = torch.einsum("cfbij,cj->cfbi", total_rotmats, centers)
        R_coarse_t_fine = torch.einsum(
            "cbij,cfbj->cfbi", coarse_rotmats, fine_transls
        )
        total_transls = (
            -R_total_c
            + R_coarse_t_fine
            + coarse_transls[:, None]
            + centers[:, None, None]
        )  # (C, F, B, 3)

        transls_flat = total_transls.contiguous().view(C * F_dim, -1)
        rots_flat = total_6d.contiguous().view(C * F_dim, -1)

        base_offsets = torch.arange(F_dim, device=cluster_ids.device)
        bag_indices = (cluster_ids.unsqueeze(1) * F_dim) + base_offsets  # (G, F)

        transls = torch.nn.functional.embedding_bag(
            weight=transls_flat,
            input=bag_indices,
            per_sample_weights=coefs,
            mode="sum",
        ).view(G, B, 3)

        rots_blended = torch.nn.functional.embedding_bag(
            weight=rots_flat,
            input=bag_indices,
            per_sample_weights=coefs,
            mode="sum",
        ).view(G, B, 6)
        rotmats = cont_6d_to_rmat(rots_blended)

        return torch.cat([rotmats, transls[..., None]], dim=-1)


if __name__ == "__main__":
    # Verification (per task spec):
    #   1. zero-init equivalence: step 0 == plain ScalableMotionBases exactly,
    #      |omega|_max == 0, |delta_t|_max == 0 -- holds regardless of gate values.
    #   2. grad flow: head.weight.grad norm > 0, and (after one optimizer step
    #      moves the head off zero) attention score params also get nonzero grad.
    #   3. gate == 1 everywhere reduces EXACTLY to the baseline attention variant
    #      (same score, same alpha, same cluster_gate == 1 -> no change in omega/delta_t).
    #   4. gate masking correctness: DISCONNECTED edges get exactly alpha == 0,
    #      an isolated-this-frame receiver's agg is exactly 0, both directions of
    #      an undirected pair always share the same gate.
    #   5. cluster gate correctness: "one CONNECTED incident edge turns the whole
    #      cluster's correction on" even if another incident edge is DISCONNECTED.
    #   6. num_heads=1 also runs.
    #   7. save -> init_from_state_dict round-trip reproduces the same output.
    #   8. end-to-end: load_edge_frame_gates + from_scalable_motion_bases from an
    #      actual (temp-file) edges.pt.
    import tempfile

    from flow3d.graph_relative_linear_attention import (
        RelativeVelLinearAttentionGraphCorrectedScalableMotionBases,
    )

    torch.manual_seed(0)
    num_clusters, num_frames, num_fine, num_fg = 6, 8, 3, 50

    centers = torch.randn(num_clusters, 3)
    rots = torch.randn(num_clusters, num_frames, 6)
    transls = torch.randn(num_clusters, num_frames, 3) * 0.1
    fine_rots = torch.randn(num_clusters, num_fine, num_frames, 6)
    fine_transls = torch.randn(num_clusters, num_fine, num_frames, 3) * 0.01

    baseline = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    # a simple chain graph: 0-1-2-3-4-5
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )
    edge_gate_ones = torch.ones(edge_index.shape[1], num_frames)
    gnn_hidden_dim, gnn_num_layers, gnn_num_heads = 32, 2, 4

    graph_bases = RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases(
        centers=centers.clone(), rots=rots.clone(), transls=transls.clone(),
        fine_rots=fine_rots.clone(), fine_transls=fine_transls.clone(),
        edge_index=edge_index, edge_gate=edge_gate_ones,
        gnn_hidden_dim=gnn_hidden_dim, gnn_num_layers=gnn_num_layers, gnn_num_heads=gnn_num_heads,
    )

    ts = torch.arange(num_frames)
    cluster_ids = torch.randint(0, num_clusters, (num_fg,))
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    # --- 1. zero-init equivalence ---
    ref_coarse = baseline.compute_transforms_coarse(ts, cluster_ids)
    out_coarse = graph_bases.compute_transforms_coarse(ts, cluster_ids)
    max_diff_coarse = (ref_coarse - out_coarse).abs().max().item()
    print(f"[coarse]  max |baseline - frame(zero-init)| = {max_diff_coarse:.3e}")
    assert max_diff_coarse < 1e-5

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"[full]    max |baseline - frame(zero-init)| = {max_diff:.3e}")
    assert max_diff < 1e-5

    omega = graph_bases.last_correction["omega"]
    delta_t = graph_bases.last_correction["delta_t"]
    print(f"[zero-init] |omega|_max={omega.abs().max().item():.3e}  |delta_t|_max={delta_t.abs().max().item():.3e}")
    assert omega.abs().max().item() == 0.0
    assert delta_t.abs().max().item() == 0.0

    # --- 2. grad flow ---
    graph_bases.zero_grad()
    out2 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss = out2.pow(2).mean()
    loss.backward()
    head_grad_norm = graph_bases.gnn.head.weight.grad.norm().item()
    attn_grad_norm_at_init = graph_bases.gnn.layers[0].attn.grad.norm().item()
    print(f"grad norm (zero-init): head.weight={head_grad_norm:.3e}  layers[0].attn={attn_grad_norm_at_init:.3e}")
    assert head_grad_norm > 0.0
    assert attn_grad_norm_at_init == 0.0

    with torch.no_grad():
        for p in graph_bases.gnn.head.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad

    graph_bases.zero_grad()
    out3 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss3 = out3.pow(2).mean()
    loss3.backward()
    attn_grad_norm = graph_bases.gnn.layers[0].attn.grad.norm().item()
    print(f"grad norm (post-step): layers[0].attn={attn_grad_norm:.3e}")
    assert attn_grad_norm > 0.0

    # --- 3. gate == 1 everywhere reduces exactly to the baseline attention variant ---
    torch.manual_seed(1)
    baseline_attn_bases = RelativeVelLinearAttentionGraphCorrectedScalableMotionBases(
        centers=centers.clone(), rots=rots.clone(), transls=transls.clone(),
        fine_rots=fine_rots.clone(), fine_transls=fine_transls.clone(),
        edge_index=edge_index, gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers, gnn_num_heads=gnn_num_heads,
    )
    torch.manual_seed(1)
    frame_bases_gate1 = RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases(
        centers=centers.clone(), rots=rots.clone(), transls=transls.clone(),
        fine_rots=fine_rots.clone(), fine_transls=fine_transls.clone(),
        edge_index=edge_index, edge_gate=edge_gate_ones,
        gnn_hidden_dim=gnn_hidden_dim, gnn_num_layers=gnn_num_layers, gnn_num_heads=gnn_num_heads,
    )
    # move both heads off zero identically so the comparison isn't trivially 0==0
    with torch.no_grad():
        for p_a, p_b in zip(baseline_attn_bases.gnn.head.parameters(), frame_bases_gate1.gnn.head.parameters()):
            p_a.add_(0.1 * torch.randn_like(p_a))
            p_b.copy_(p_a)
    out_baseline_attn = baseline_attn_bases.compute_transforms(ts, coefs, cluster_ids)
    out_frame_gate1 = frame_bases_gate1.compute_transforms(ts, coefs, cluster_ids)
    max_diff_gate1 = (out_baseline_attn - out_frame_gate1).abs().max().item()
    print(f"[gate==1 reduction] max |baseline_attention - frame(gate=1)| = {max_diff_gate1:.3e}")
    assert max_diff_gate1 < 1e-4, "gate==1 everywhere should reduce exactly to the baseline attention variant"

    # --- 5/6. feature dims + num_heads=1 smoke test ---
    assert graph_bases.gnn.encoder[0].in_features == NODE_FEAT_DIM_VEL == 15
    one_head_bases = RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases(
        centers=centers.clone(), rots=rots.clone(), transls=transls.clone(),
        fine_rots=fine_rots.clone(), fine_transls=fine_transls.clone(),
        edge_index=edge_index, edge_gate=edge_gate_ones,
        gnn_hidden_dim=gnn_hidden_dim, gnn_num_layers=gnn_num_layers, gnn_num_heads=1,
    )
    _ = one_head_bases.compute_transforms(ts, coefs, cluster_ids)
    print("[num_heads=1] forward pass OK")

    # --- 4. gate masking correctness on a toy graph: 0-1 CONNECTED@all frames,
    #     1-2 DISCONNECTED@all frames, cluster 3 fully isolated ---
    toy_num_clusters = 4
    toy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    toy_gate = torch.stack(
        [torch.ones(num_frames), torch.zeros(num_frames)], dim=0
    )  # edge (0,1) always CONNECTED, edge (1,2) always DISCONNECTED
    toy_centers = torch.randn(toy_num_clusters, 3)
    toy_rots = torch.randn(toy_num_clusters, num_frames, 6)
    toy_transls = torch.randn(toy_num_clusters, num_frames, 3) * 0.1
    toy_fine_rots = torch.randn(toy_num_clusters, num_fine, num_frames, 6)
    toy_fine_transls = torch.randn(toy_num_clusters, num_fine, num_frames, 3) * 0.01
    toy_bases = RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases(
        centers=toy_centers, rots=toy_rots, transls=toy_transls,
        fine_rots=toy_fine_rots, fine_transls=toy_fine_transls,
        edge_index=toy_edge_index, edge_gate=toy_gate,
        gnn_hidden_dim=16, gnn_num_layers=1, gnn_num_heads=2,
    )
    toy_ts = torch.arange(num_frames)
    toy_gnn = toy_bases.gnn
    toy_coarse_rot_6d = toy_bases.params["rots"][:, toy_ts]
    toy_coarse_transl = toy_bases.params["transls"][:, toy_ts]
    toy_ts_prev = (toy_ts - 1).clamp(min=0)
    toy_coarse_vel = toy_coarse_transl - toy_bases.params["transls"][:, toy_ts_prev]
    toy_edge_gate_dir = toy_gnn.edge_gate_dir_full[:, toy_ts]
    toy_vel_scaled = toy_coarse_vel / toy_gnn.vel_scale
    C_, B_ = toy_num_clusters, num_frames
    toy_center_feat = toy_centers[:, None, :].expand(C_, B_, CENTER_DIM)
    toy_node_feat = torch.cat(
        [toy_coarse_rot_6d, toy_coarse_transl, toy_center_feat, toy_vel_scaled], dim=-1
    )
    toy_h = toy_gnn.encoder(toy_node_feat)
    toy_edge_feat = toy_gnn._compute_edge_features(
        toy_coarse_rot_6d, toy_coarse_transl, toy_centers, toy_vel_scaled
    )
    toy_layer = toy_gnn.layers[0]
    alpha_dense, mask = toy_layer._compute_attention(
        toy_h, toy_gnn.edge_index_dir, toy_edge_feat, toy_edge_gate_dir
    )
    assert not torch.isnan(alpha_dense).any().item()

    # cluster 1's only active in-neighbor is 0 (edge 1-2 is DISCONNECTED) ->
    # its alpha row should sum to 1 with all weight on neighbor 0.
    row1_sum = alpha_dense[1].sum(dim=0)  # (B, K), sum over neighbor axis
    print(f"[gate masking] receiver 1 (only 0->1 active): max |row_sum - 1| = {(row1_sum - 1.0).abs().max().item():.3e}")
    assert (row1_sum - 1.0).abs().max().item() < 1e-5
    alpha_2_to_1 = alpha_dense[1, 2]
    print(f"[gate masking] alpha(2->1) (DISCONNECTED edge) max = {alpha_2_to_1.abs().max().item():.3e} (expect 0)")
    assert alpha_2_to_1.abs().max().item() == 0.0

    # cluster 2's only in-neighbor is 1, but edge 1-2 is DISCONNECTED ->
    # cluster 2 has zero active in-neighbors this "frame" -> agg exactly 0.
    row2_sum = alpha_dense[2].sum(dim=0)
    print(f"[gate masking] receiver 2 (no active in-neighbor): max |row_sum| = {row2_sum.abs().max().item():.3e} (expect 0)")
    assert row2_sum.abs().max().item() == 0.0
    toy_out = toy_layer(toy_h, toy_gnn.edge_index_dir, toy_edge_feat, toy_edge_gate_dir)
    isolated_contribution = (toy_out[2] - (toy_h[2] + toy_layer.act(torch.zeros_like(toy_h[2])))).abs().max().item()
    print(f"[gate masking] cluster 2 agg contribution = {isolated_contribution:.3e} (expect 0)")
    assert isolated_contribution < 1e-6

    # both directions of an undirected pair always share the same gate
    src, dst = toy_gnn.edge_index_dir[0], toy_gnn.edge_index_dir[1]
    for a, b in [(0, 1), (1, 2)]:
        fwd = toy_gnn.edge_gate_dir_full[(src == a) & (dst == b)]
        bwd = toy_gnn.edge_gate_dir_full[(src == b) & (dst == a)]
        assert torch.equal(fwd, bwd), f"gate({a}->{b}) must equal gate({b}->{a})"
    print("[gate masking] both directions of each undirected pair share the same gate: OK")

    # --- 5. cluster gate correctness: one CONNECTED incident edge is enough
    #     to turn the whole cluster's correction on, even with another
    #     DISCONNECTED incident edge (cluster 1: edges (0,1) CONNECTED, (1,2)
    #     DISCONNECTED -> cluster_gate[1] == 1; cluster 2: only incident edge
    #     is DISCONNECTED -> cluster_gate[2] == 0; cluster 3: no incident edge
    #     at all -> cluster_gate[3] == 0).
    _, _, toy_cluster_gate = toy_gnn(
        toy_coarse_rot_6d, toy_coarse_transl, toy_centers, toy_coarse_vel, toy_edge_gate_dir
    )
    print(f"[cluster gate] cluster_gate[1] (one CONNECTED incident edge) = {toy_cluster_gate[1].tolist()}")
    assert torch.allclose(toy_cluster_gate[1], torch.ones(num_frames))
    print(f"[cluster gate] cluster_gate[2] (only incident edge DISCONNECTED) = {toy_cluster_gate[2].tolist()}")
    assert torch.allclose(toy_cluster_gate[2], torch.zeros(num_frames))
    print(f"[cluster gate] cluster_gate[3] (no incident edge) = {toy_cluster_gate[3].tolist()}")
    assert torch.allclose(toy_cluster_gate[3], torch.zeros(num_frames))

    # --- regression: a cluster whose every incident edge is DISCONNECTED this
    #     frame (cluster 2 above) must produce a NaN-FREE backward pass, not
    #     just a clean forward value. torch.softmax(score_dense, dim=1) +
    #     torch.nan_to_num(alpha, nan=0.0) gets the forward value right but
    #     leaves NaN in softmax's own backward (repro: `-inf - (-inf) == NaN`
    #     inside softmax's internal max-subtraction, reused by its backward
    #     formula regardless of the upstream gradient) -- this crashed real
    #     training runs ("Loss is NaN" at the very first loss.backward() after
    #     resuming with --gnn-variant=relative_velocity_linear_attention_frame,
    #     since several clusters had every incident edge DISCONNECTED within a
    #     single frame window). _compute_attention's manual masked-softmax
    #     (module docstring) must keep every gradient finite here.
    toy_bases.zero_grad()
    toy_fg = 10
    toy_cluster_ids = torch.randint(0, toy_num_clusters, (toy_fg,))
    toy_coefs = torch.softmax(torch.randn(toy_fg, num_fine), dim=-1)
    toy_full_out = toy_bases.compute_transforms(toy_ts, toy_coefs, toy_cluster_ids)
    toy_full_out.pow(2).mean().backward()
    nan_grad_params = [
        name for name, p in toy_bases.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    print(f"[NaN-safe backward] params with NaN grad (all-DISCONNECTED-incident cluster in batch) = {nan_grad_params} (expect [])")
    assert not nan_grad_params, "manual masked softmax must keep every gradient finite"

    print(f"[vel_scale] {graph_bases.gnn.vel_scale.item():.3e}")

    # --- 7. save -> init_from_state_dict round-trip ---
    full_state_dict = {f"motion_bases.{k}": v for k, v in graph_bases.state_dict().items()}
    assert RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases.has_gnn_state(
        full_state_dict, prefix="motion_bases."
    )
    restored = RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases.init_from_state_dict(
        full_state_dict, prefix="motion_bases."
    )
    out_restored = restored.compute_transforms(ts, coefs, cluster_ids)
    out_original = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff_roundtrip = (out_restored - out_original).abs().max().item()
    num_heads_match = restored.gnn.num_heads == graph_bases.gnn.num_heads
    edge_gate_match = torch.equal(restored.gnn.edge_gate_dir_full, graph_bases.gnn.edge_gate_dir_full)
    print(
        f"[round-trip] max |original - restored| = {max_diff_roundtrip:.3e}  "
        f"num_heads match = {num_heads_match} ({restored.gnn.num_heads} vs {graph_bases.gnn.num_heads})  "
        f"edge_gate_dir_full match = {edge_gate_match}"
    )
    assert max_diff_roundtrip < 1e-6
    assert num_heads_match
    assert edge_gate_match

    # --- 8. end-to-end: load_edge_frame_gates + from_scalable_motion_bases from
    #     an actual edges.pt-shaped file (with connected/unknown frame indices).
    with tempfile.TemporaryDirectory() as tmp_dir:
        edges_path = f"{tmp_dir}/edges.pt"
        torch.save(
            {
                "edges_kept": [
                    {
                        "cluster_a": 0, "cluster_b": 1,
                        "connected_frame_indices": [0, 1, 2],
                        "unknown_frame_indices": [3, 4],
                    },
                    {
                        "cluster_a": 1, "cluster_b": 2,
                        "connected_frame_indices": [],
                        "unknown_frame_indices": [],
                    },
                    {
                        "cluster_a": 3, "cluster_b": 4,
                        "connected_frame_indices": [7],
                        "unknown_frame_indices": [],
                    },
                ],
            },
            edges_path,
        )
        e2e_edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)
        e2e_bases = RelativeVelLinearAttentionFrameGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
            baseline, edge_index=e2e_edge_index, edges_path=edges_path,
            gnn_hidden_dim=gnn_hidden_dim, gnn_num_layers=gnn_num_layers, gnn_num_heads=gnn_num_heads,
        )
        _ = e2e_bases.compute_transforms(ts, coefs, cluster_ids)
        print("[end-to-end] from_scalable_motion_bases(edges_path=...) OK")

    print("OK")

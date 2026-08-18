"""
flow3d/graph_relative_linear_attention_multihead.py

flow3d/graph_relative_linear_attention.py의 변형: GAT식 multi-head attention
score/softmax는 완전히 동일하게 유지한 채, message를 만드는 MLP만 진짜
multi-head 구조로 바꾼다.

message MLP: shared-then-reshape -> per-head 독립 MLP
-------------------------------------------------------
graph_relative_linear_attention.py는

    m_ij = MLP([h_j - h_i, e_ij])                # 출력 hidden_dim
    (m_ij를 K개 head로 reshape만 해서 나눠 씀 -- 모든 head가 파라미터를 공유)

였다. 이 파일은 message MLP 자체를 head마다 독립된 파라미터를 갖는 별도의
작은 MLP로 바꾼다:

    m_ij^k = MLP_k([h_j - h_i, e_ij])             k = 1..K, 각 head 출력 head_dim
    m_ij   = concat_k(m_ij^k)                     다시 hidden_dim으로 복원

    s_ij     = LeakyReLU( a^T [W h_i, W h_j, W_e e_ij] )   (attention score, per head -- 변경 없음)
    alpha_ij = softmax_{j in N(i)}(s_ij)
    agg_i    = sum_{j in N(i)} alpha_ij * m_ij   (head별로 alpha_ij^k * m_ij^k 후 concat)
    h_i'     = h_i + act(agg_i)

즉 head마다 "이웃을 얼마나 반영할지"(attention weight)뿐 아니라 "이웃 정보를
어떻게 변환할지"(message MLP)까지 서로 다른 파라미터를 갖게 된다. head 출력을
그대로 concat해서 hidden_dim을 복원하며(표준 GAT의 중간 layer 방식과 동일),
별도의 output projection은 두지 않는다 -- 요구사항이 "message mlp를
multi-head 구조로" 바꾸는 것이지 attention 결합 방식 자체를 바꾸는 것은
아니기 때문이다.

dense-masked softmax, zero-init 등가성, num_heads=1 지원, node/edge feature
차원(15/15) 등 나머지는 graph_relative_linear_attention.py와 완전히 동일하다
(head를 만드는 message MLP 내부 구조만 바뀌었을 뿐, attention score 계산/
softmax 정규화/최종 head Linear(0-init) 등은 그대로다).

이 파일에서 바뀌지 않는 것 (재구현하지 않고 import)
------------------------------------------------------
- so3_exp_map / compose_rotation / build_edge_index_from_edges_pt / cont_6d_to_rmat
  / rmat_to_cont_6d / CENTER_DIM / ROT_DIM / TRANSL_DIM / NODE_FEAT_DIM: 기존
  flow3d/graph_coupling.py 구현을 그대로 가져다 쓴다.
- _build_directed_neighbor_edges: flow3d/graph_coupling_relative.py 구현을
  그대로 가져다 쓴다 (topology 구성은 이 확장으로 바뀌지 않는다).
- NODE_FEAT_DIM_VEL / EDGE_FEAT_DIM_VEL / VEL_DIM / _compute_vel_scale: linear
  velocity 관련 상수/로직은 flow3d/graph_relative_velocity_linear.py의 구현을
  그대로 가져다 쓴다. node/edge feature 자체의 내용과 차원(15/15)은 이
  파일에서 전혀 바뀌지 않는다.
- attention score 계산(_compute_attention), dense-masked softmax, coarse
  correction 합성 수식, coarse-to-fine 결합 수식, checkpoint save/load
  포맷은 graph_relative_linear_attention.py와 동일하다.

이 파일에서 제공하는 것
------------------------
- RelativeVelLinearAttentionMultiHeadClusterGraphGNN: 위 두 클래스와 같은
  API(node feature -> (omega, delta_t))를 갖지만, message MLP가 head마다
  독립된 파라미터를 갖는 GNN.
- RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases:
  RelativeVelLinearAttentionGraphCorrectedScalableMotionBases와 동일한 API
  (from_scalable_motion_bases, has_gnn_state, init_from_state_dict,
  compute_transforms_coarse, compute_transforms, last_correction)를 가진
  드롭인 대체.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.graph_coupling import (
    CENTER_DIM,
    NODE_FEAT_DIM,
    ROT_DIM,
    TRANSL_DIM,
    build_edge_index_from_edges_pt,
    compose_rotation,
    so3_exp_map,
)
from flow3d.graph_coupling_relative import (
    EDGE_FEAT_DIM,
    _build_directed_neighbor_edges,
)
from flow3d.graph_relative_velocity_linear import (
    EDGE_FEAT_DIM_VEL,
    NODE_FEAT_DIM_VEL,
    VEL_DIM,
    _compute_vel_scale,
)
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "so3_exp_map",
    "compose_rotation",
    "build_edge_index_from_edges_pt",
    "RelativeVelLinearAttentionMultiHeadClusterGraphGNN",
    "RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases",
]

_LEAKY_SLOPE = 0.2


class _RelativeVelLinearAttentionMultiHeadMessageLayer(nn.Module):
    """Fixed-topology relative message passing, 1 layer, residual -- same
    attention score/softmax as
    graph_relative_linear_attention.py's _RelativeVelLinearAttentionMessageLayer,
    but the message MLP itself is now per-head independent instead of a
    single shared MLP whose output is merely reshaped into heads:

        m_ij^k   = MLP_k([h_j - h_i, e_ij])                    per head, output head_dim
        s_ij     = LeakyReLU(a^T [W h_i, W h_j, W_e e_ij])      (per head, unchanged)
        alpha_ij = softmax_{j in N(i)}(s_ij)
        agg_i    = sum_{j in N(i)} alpha_ij * m_ij              (per-head alpha * per-head message)
        h_i'     = h_i + act(agg_i)

    Self-loop을 쓰지 않는 이유는 baseline relative message layer와 동일하다
    (h_i - h_i == 0이라 self-message가 무의미한 상수 항이 되기 때문).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # message MLP: one independent 2-layer MLP per head (input ->
        # head_dim -> head_dim), instead of one shared hidden_dim MLP split
        # by reshape. Each head can learn a different neighbor-message
        # transform, not just a different aggregation weight.
        msg_input_dim = hidden_dim + EDGE_FEAT_DIM_VEL
        self.msg_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(msg_input_dim, self.head_dim),
                    nn.ReLU(),
                    nn.Linear(self.head_dim, self.head_dim),
                )
                for _ in range(num_heads)
            ]
        )
        self.act = nn.ReLU()

        # GAT-style attention score: W (shared for both i and j roles) + W_e
        # for the edge feature, then a per-head scoring vector `attn`.
        # Unchanged from graph_relative_linear_attention.py.
        self.W_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_edge = nn.Linear(EDGE_FEAT_DIM_VEL, hidden_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)

    def _compute_attention(
        self,
        h: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense-masked multi-head attention weights (identical to
        graph_relative_linear_attention.py -- only the message MLP that
        consumes alpha differs, not how alpha itself is computed).

        :param h: (C, B, H) node hidden features
        :param edge_index_dir: (2, E) directed [src(j), dst(i)] pairs, self-loop-free
        :param edge_feat: (E, B, EDGE_FEAT_DIM_VEL) relative geometry+velocity per edge
        :returns: alpha_dense (C, C, B, heads) with alpha_dense[i, j] = alpha_{j->i}
                  (0 where no edge j->i exists, or where receiver i has no
                  in-neighbors at all), and mask (C, C) bool adjacency
                  (mask[i, j] == True iff edge j->i exists).
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
        raw_score = (concat * self.attn).sum(-1)  # (E, B, K), a^T [Whi, Whj, We_eij]
        score_e = F.leaky_relu(raw_score, negative_slope=_LEAKY_SLOPE)  # (E, B, K)

        score_dense = h.new_full((C, C, B, K), float("-inf"))
        score_dense[dst, src] = score_e  # dense[i, j] = s_{j->i}

        alpha_dense = torch.softmax(score_dense, dim=1)  # normalize over j (neighbor) axis
        alpha_dense = torch.nan_to_num(alpha_dense, nan=0.0)  # in-degree-0 rows: all -inf -> NaN -> 0

        mask = torch.zeros(C, C, dtype=torch.bool, device=device)
        mask[dst, src] = True

        return alpha_dense, mask

    def forward(
        self,
        h: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (C, B, H) node hidden features
        edge_index_dir: (2, E) directed [src(j), dst(i)] pairs, self-loop-free
        edge_feat: (E, B, EDGE_FEAT_DIM_VEL) relative geometry+velocity feature
                   per directed edge (shared across layers within one forward call)
        """
        C, B, H = h.shape
        if edge_index_dir.numel() == 0:
            return h + self.act(h.new_zeros(C, B, H))

        src, dst = edge_index_dir[0], edge_index_dir[1]

        alpha_dense, _mask = self._compute_attention(h, edge_index_dir, edge_feat)
        alpha_e = alpha_dense[dst, src]  # (E, B, K), gather back to edge order

        delta_h = h[src] - h[dst]  # (E, B, H), Δh_ij = h_j - h_i
        msg_input = torch.cat([delta_h, edge_feat], dim=-1)  # (E, B, H + EDGE_FEAT_DIM_VEL)
        m_heads = torch.stack(
            [mlp(msg_input) for mlp in self.msg_mlps], dim=2
        )  # (E, B, K, D), each head from its own independent MLP
        weighted = alpha_e.unsqueeze(-1) * m_heads  # (E, B, K, D)
        weighted_flat = weighted.reshape(-1, B, H)  # concat heads back to H

        agg = h.new_zeros(C, B, H)
        agg.index_add_(0, dst, weighted_flat)

        return h + self.act(agg)


class RelativeVelLinearAttentionMultiHeadClusterGraphGNN(nn.Module):
    """RelativeVelLinearAttentionClusterGraphGNN과 같은 API(node feature ->
    (omega, delta_t))를 갖는 relative-message GNN이지만, message MLP가
    head마다 독립된 파라미터를 갖는다 (attention score/softmax 자체는 동일).
    node/edge feature(linear velocity 포함)는 baseline과 완전히 동일하다.

    Graph topology(directed neighbor edge, self-loop 제외)는 생성 시점에 고정돼
    학습 중 바뀌지 않는다. head(마지막 Linear)는 0으로 초기화되므로 학습 시작
    시점에는 correction이 정확히 0이다(baseline과 동일한 zero-init 등가성 --
    message MLP를 multi-head로 바꾼 것과 무관하게 유지된다).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vel_scale: float = 1.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        edge_index_dir = _build_directed_neighbor_edges(edge_index, num_clusters)
        self.register_buffer("edge_index_dir", edge_index_dir)

        if not math.isfinite(vel_scale) or vel_scale <= 0.0:
            vel_scale = 1.0
        self.register_buffer("vel_scale", torch.tensor(float(vel_scale)))

        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM_VEL, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [
                _RelativeVelLinearAttentionMultiHeadMessageLayer(hidden_dim, num_heads=num_heads)
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
        """directed edge j->i마다
        e_ij = [t_j - t_i, c_j - c_i, rot6d(R_i^T @ R_j), v_j/vel_scale - v_i/vel_scale]
        를 만든다 (flow3d/graph_relative_velocity_linear.py와 동일). 한 forward
        call 안에서 한 번만 계산되어 모든 layer에 재사용된다.

        :param vel_scaled: (C, B, 3) 이미 vel_scale로 나뉜 절대 속도.
        """
        B = coarse_transl.shape[1]
        if self.edge_index_dir.numel() == 0:
            return coarse_rot_6d.new_zeros(0, B, EDGE_FEAT_DIM_VEL)

        src, dst = self.edge_index_dir[0], self.edge_index_dir[1]

        rel_transl = coarse_transl[src] - coarse_transl[dst]  # (E, B, 3), t_j - t_i
        rel_center = centers[src] - centers[dst]  # (E, 3), c_j - c_i
        rel_center = rel_center[:, None, :].expand(-1, B, -1)  # (E, B, 3)

        coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)  # (C, B, 3, 3)
        R_i = coarse_rotmats[dst]  # (E, B, 3, 3)
        R_j = coarse_rotmats[src]  # (E, B, 3, 3)
        R_rel = torch.matmul(R_i.transpose(-1, -2), R_j)  # R_i^T @ R_j
        rel_rot6d = rmat_to_cont_6d(R_rel)  # (E, B, 6)

        rel_vel = vel_scaled[src] - vel_scaled[dst]  # (E, B, 3), already vel_scale-normalized

        return torch.cat([rel_transl, rel_center, rel_rot6d, rel_vel], dim=-1)  # (E, B, 15)

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        coarse_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        coarse_rot_6d: (C, B, 6)
        coarse_transl: (C, B, 3)
        centers: (C, 3) canonical cluster centers
        coarse_vel: (C, B, 3) raw (unnormalized) linear velocity, i.e.
            coarse_transl(t) - coarse_transl(t-1) (see module docstring)
        returns: omega (C, B, 3), delta_t (C, B, 3)
        """
        if coarse_rot_6d.shape[0] != self.num_clusters:
            raise ValueError(
                f"coarse_rot_6d has {coarse_rot_6d.shape[0]} clusters, "
                f"expected {self.num_clusters}"
            )

        C, B, _ = coarse_rot_6d.shape
        center_feat = centers[:, None, :].expand(C, B, CENTER_DIM)
        vel_scaled = coarse_vel / self.vel_scale  # (C, B, 3), O(1)-normalized
        # absolute node state + absolute velocity (baseline과 동일하게 relative로
        # 바꾸지 않는다 -- 이 확장은 message MLP 구조만 바꿀 뿐이다)
        node_feat = torch.cat(
            [coarse_rot_6d, coarse_transl, center_feat, vel_scaled], dim=-1
        )  # (C, B, 15)

        h = self.encoder(node_feat)
        edge_feat = self._compute_edge_features(coarse_rot_6d, coarse_transl, centers, vel_scaled)
        for layer in self.layers:
            h = layer(h, self.edge_index_dir, edge_feat)

        correction = self.head(h)  # (C, B, 6)
        omega, delta_t = correction.split([3, 3], dim=-1)
        return omega, delta_t


class RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + relative-message-with-velocity-and-multihead-attention
    graph-aware coarse transform correction.
    RelativeVelLinearAttentionGraphCorrectedScalableMotionBases와 API가 동일한
    드롭인 대체이며, 차이는 내부 GNN이
    RelativeVelLinearAttentionMultiHeadClusterGraphGNN(message MLP가 head마다
    독립적)이라는 점뿐이다.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        rots: torch.Tensor,
        transls: torch.Tensor,
        fine_rots: torch.Tensor,
        fine_transls: torch.Tensor,
        edge_index: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
    ):
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        vel_scale = _compute_vel_scale(transls)
        self.gnn = RelativeVelLinearAttentionMultiHeadClusterGraphGNN(
            edge_index=edge_index,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            vel_scale=vel_scale,
        )
        self._last_correction: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
    ) -> "RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 relative+velocity+multihead
        attention graph-corrected 버전을 만든다. coarse/fine motion 파라미터 값은
        그대로 복사되고, GNN correction만 새로 추가된다 (correction은 0으로
        초기화됨). vel_scale은 복사된 transls로부터 __init__ 안에서 다시
        계산된다."""
        p = bases.params
        return cls(
            centers=p["centers"].detach().clone(),
            rots=p["rots"].detach().clone(),
            transls=p["transls"].detach().clone(),
            fine_rots=p["fine_rots"].detach().clone(),
            fine_transls=p["fine_transls"].detach().clone(),
            edge_index=edge_index,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_num_heads=gnn_num_heads,
        )

    @staticmethod
    def has_gnn_state(state_dict: dict, prefix: str) -> bool:
        """state_dict의 motion_bases가
        RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases로
        저장된 것인지 확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은
        형태여야 한다 (params.가 아니라)."""
        gnn_prefix = f"{prefix}gnn."
        return any(key.startswith(gnn_prefix) for key in state_dict)

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases":
        """체크포인트만으로
        RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases를
        통째로 복원한다. edge_index/hidden_dim/num_layers는 저장된 텐서의
        shape에서 읽어온다. num_heads는 저장된 attention 벡터
        "layers.0.attn"의 shape[0]에서 그대로 읽는다 (attn은
        (num_heads, 3*head_dim) 모양으로 저장되므로 shape[0]가 곧
        num_heads이다). directed edge/vel_scale 버퍼는 __init__ 시점에
        placeholder로 재계산되고, 아래 load_state_dict가 저장된 실제 값으로
        strict하게 덮어쓴다 (baseline relative/velocity_linear와 동일한
        패턴)."""
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

        saved_edge_index_dir = state_dict[f"{gnn_prefix}edge_index_dir"]
        graph_bases = cls.from_scalable_motion_bases(
            base,
            edge_index=saved_edge_index_dir,
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
        """coarse rotation/translation에 GNN correction을 compose한다.

        회전은 compose(exp(omega) @ R_coarse)이고, translation만 덧셈이다
        (baseline과 완전히 동일한 compose 수식 -- 바뀐 건 omega/delta_t를 만드는
        message MLP가 head마다 독립적인 구조로 바뀐다는 점뿐이다).

        velocity는 여기서만 계산된다: ts/self.params에 동시에 접근할 수 있는
        유일한 지점이기 때문이다 (GNN.forward는 이미 gather된 텐서만 받는다).
        """
        coarse_rot_6d = self.params["rots"][:, ts]  # (C, B, 6)
        coarse_transl = self.params["transls"][:, ts]  # (C, B, 3)
        centers = self.params["centers"]  # (C, 3)

        ts_prev = (ts - 1).clamp(min=0)  # t=0 -> prev=0 -> v=0, no special-casing needed
        coarse_transl_prev = self.params["transls"][:, ts_prev]  # (C, B, 3)
        coarse_vel = coarse_transl - coarse_transl_prev  # (C, B, 3), raw frame-to-frame delta

        omega, delta_t = self.gnn(coarse_rot_6d, coarse_transl, centers, coarse_vel)  # (C, B, 3) x2
        self._last_correction = {"omega": omega, "delta_t": delta_t}

        coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)  # (C, B, 3, 3)
        R_correction = so3_exp_map(omega)  # (C, B, 3, 3)
        corrected_rotmats = compose_rotation(R_correction, coarse_rotmats)  # exp(omega) @ R_coarse
        corrected_transl = coarse_transl + delta_t  # translation only adds

        return corrected_rotmats, corrected_transl

    @property
    def last_correction(self) -> dict[str, torch.Tensor] | None:
        """가장 최근 forward에서 나온 (omega, delta_t) correction (디버깅/로깅용)."""
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
        완전히 동일하며, coarse rotation/translation만 multihead-message
        correction이 compose된 값으로 바뀐다. fine motion 코드는 한 글자도
        바뀌지 않는다.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        assert coefs.shape[0] == cluster_ids.shape[0]
        G, F_dim = coefs.shape
        C, _, B, _ = self.params["fine_transls"][:, :, ts].shape

        # --- graph-aware coarse transform (baseline과 다른 부분: multi-head message MLP) ---
        coarse_rotmats, coarse_transls = self._corrected_coarse(ts)
        centers = self.params["centers"]  # (C, 3)

        # --- 이하는 ScalableMotionBases.compute_transforms와 완전히 동일 ---
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
    # Verification (same shape as graph_relative_linear_attention.py's):
    #   1. zero-init equivalence: step 0 == plain ScalableMotionBases exactly,
    #      |omega|_max == 0, |delta_t|_max == 0.
    #   2. grad flow: head.weight.grad norm > 0, and (after one optimizer step
    #      moves the head off zero) per-head message MLP params also get
    #      nonzero grad.
    #   3. attention correctness: per-receiver neighbor alpha sums to 1,
    #      non-edge alpha == 0, isolated-node agg == 0, no NaN anywhere.
    #   4. attention is actually non-uniform for neighbors with different features.
    #   5. node/edge feature dims are 15/15 (unchanged), and message MLPs are
    #      truly per-head (independent parameters, input head_dim not shared hidden_dim).
    #   6. num_heads=1 also runs.
    #   7. save -> init_from_state_dict round-trip reproduces the same output.
    torch.manual_seed(0)
    num_clusters, num_frames, num_fine, num_fg = 6, 8, 3, 50

    centers = torch.randn(num_clusters, 3)
    rots = torch.randn(num_clusters, num_frames, 6)
    transls = torch.randn(num_clusters, num_frames, 3) * 0.1
    fine_rots = torch.randn(num_clusters, num_fine, num_frames, 6)
    fine_transls = torch.randn(num_clusters, num_fine, num_frames, 3) * 0.01

    baseline = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    # a simple chain graph: 0-1-2-3-4-5 (cluster 33/hand analog would connect to its forearm cluster)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )
    gnn_hidden_dim, gnn_num_layers, gnn_num_heads = 32, 2, 4
    graph_bases = RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline,
        edge_index=edge_index,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_num_heads=gnn_num_heads,
    )

    ts = torch.arange(num_frames)
    cluster_ids = torch.randint(0, num_clusters, (num_fg,))
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    # --- 1. zero-init equivalence ---
    ref_coarse = baseline.compute_transforms_coarse(ts, cluster_ids)
    out_coarse = graph_bases.compute_transforms_coarse(ts, cluster_ids)
    max_diff_coarse = (ref_coarse - out_coarse).abs().max().item()
    print(f"[coarse]  max |baseline - velocity_linear_attention_multihead_graph_corrected(zero-init)| = {max_diff_coarse:.3e}")
    assert max_diff_coarse < 1e-5, "zero-init correction should reproduce the baseline coarse transform exactly"

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"[full]    max |baseline - velocity_linear_attention_multihead_graph_corrected(zero-init)| = {max_diff:.3e}")
    assert max_diff < 1e-5, "zero-init correction should reproduce the baseline transform exactly"

    omega = graph_bases.last_correction["omega"]
    delta_t = graph_bases.last_correction["delta_t"]
    print(f"[zero-init] |omega|_max={omega.abs().max().item():.3e}  |delta_t|_max={delta_t.abs().max().item():.3e}")
    assert omega.abs().max().item() == 0.0
    assert delta_t.abs().max().item() == 0.0

    # --- 2. grad flow: head first, then per-head message MLP after head moves off zero ---
    graph_bases.zero_grad()
    out2 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss = out2.pow(2).mean()
    loss.backward()

    head_grad_norm = graph_bases.gnn.head.weight.grad.norm().item()
    encoder_grad_norm = graph_bases.gnn.encoder[0].weight.grad.norm().item()
    msg_grad_norm_at_init = graph_bases.gnn.layers[0].msg_mlps[0][0].weight.grad.norm().item()
    print(
        f"grad norm (zero-init): head.weight={head_grad_norm:.3e}  "
        f"encoder.weight={encoder_grad_norm:.3e}  layers[0].msg_mlps[0][0]={msg_grad_norm_at_init:.3e}"
    )
    assert head_grad_norm > 0.0, "GNN head should receive nonzero gradient -- it will actually train"
    # Encoder/message-passing/attention layers are (correctly) gradient-starved
    # at init: the zero-initialized head multiplies their contribution by zero
    # in dL/dW, same reasoning as graph_relative_linear_attention.py.
    assert msg_grad_norm_at_init == 0.0, "per-head message MLP params should also be zero-grad while head is exactly zero"

    with torch.no_grad():
        for p in graph_bases.gnn.head.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad  # a single manual SGD step moves head off zero

    graph_bases.zero_grad()
    out3 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss3 = out3.pow(2).mean()
    loss3.backward()
    attn_grad_norm = graph_bases.gnn.layers[0].attn.grad.norm().item()
    msg_grad_norm = graph_bases.gnn.layers[0].msg_mlps[0][0].weight.grad.norm().item()
    msg_grad_norm_other_head = graph_bases.gnn.layers[0].msg_mlps[1][0].weight.grad.norm().item()
    print(
        f"grad norm (post-step): layers[0].attn={attn_grad_norm:.3e}  "
        f"layers[0].msg_mlps[0][0].weight={msg_grad_norm:.3e}  "
        f"layers[0].msg_mlps[1][0].weight={msg_grad_norm_other_head:.3e}"
    )
    assert attn_grad_norm > 0.0, "attention score params should receive nonzero gradient once head is off zero"
    assert msg_grad_norm > 0.0, "head-0 message MLP should receive nonzero gradient once GNN head is off zero"
    assert msg_grad_norm_other_head > 0.0, "head-1 message MLP should receive nonzero gradient once GNN head is off zero"

    # --- 5/6. feature dims + per-head independence + num_heads=1 smoke test ---
    assert graph_bases.gnn.encoder[0].in_features == NODE_FEAT_DIM_VEL == 15, (
        "node encoder input dim should be unchanged from velocity_linear (12 + 3 velocity)"
    )
    head_dim = gnn_hidden_dim // gnn_num_heads
    first_layer = graph_bases.gnn.layers[0]
    assert len(first_layer.msg_mlps) == gnn_num_heads, "should have exactly num_heads independent message MLPs"
    assert first_layer.msg_mlps[0][0].in_features == gnn_hidden_dim + EDGE_FEAT_DIM_VEL, (
        "per-head message MLP input dim should be unchanged (hidden_dim + 12 + 3 velocity)"
    )
    assert first_layer.msg_mlps[0][0].out_features == head_dim, (
        "per-head message MLP should output head_dim, not hidden_dim"
    )
    assert not torch.equal(first_layer.msg_mlps[0][0].weight, first_layer.msg_mlps[1][0].weight), (
        "different heads' message MLPs should have independent (different) parameters"
    )
    print(f"[feature dims] node={NODE_FEAT_DIM_VEL}, edge={EDGE_FEAT_DIM_VEL} (unchanged); per-head msg MLP in={gnn_hidden_dim + EDGE_FEAT_DIM_VEL} out={head_dim}")

    one_head_bases = RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline, edge_index=edge_index, gnn_hidden_dim=gnn_hidden_dim, gnn_num_layers=gnn_num_layers, gnn_num_heads=1
    )
    _ = one_head_bases.compute_transforms(ts, coefs, cluster_ids)
    print("[num_heads=1] forward pass OK")

    # --- 3. attention correctness on a toy graph with an isolated node ---
    # toy graph: 4 clusters, edges 0-1 and 1-2 (cluster 3 is fully isolated).
    toy_num_clusters = 4
    toy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    toy_centers = torch.randn(toy_num_clusters, 3)
    toy_rots = torch.randn(toy_num_clusters, num_frames, 6)
    toy_transls = torch.randn(toy_num_clusters, num_frames, 3) * 0.1
    toy_fine_rots = torch.randn(toy_num_clusters, num_fine, num_frames, 6)
    toy_fine_transls = torch.randn(toy_num_clusters, num_fine, num_frames, 3) * 0.01
    toy_baseline = ScalableMotionBases(
        toy_centers, toy_rots, toy_transls, toy_fine_rots, toy_fine_transls
    )
    toy_bases = RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        toy_baseline, edge_index=toy_edge_index, gnn_hidden_dim=16, gnn_num_layers=1, gnn_num_heads=2
    )

    toy_ts = torch.arange(num_frames)
    toy_coarse_rot_6d = toy_bases.params["rots"][:, toy_ts]
    toy_coarse_transl = toy_bases.params["transls"][:, toy_ts]
    toy_ts_prev = (toy_ts - 1).clamp(min=0)
    toy_coarse_vel = toy_coarse_transl - toy_bases.params["transls"][:, toy_ts_prev]
    toy_gnn = toy_bases.gnn
    toy_vel_scaled = toy_coarse_vel / toy_gnn.vel_scale
    C_ = toy_num_clusters
    B_ = num_frames
    toy_center_feat = toy_centers[:, None, :].expand(C_, B_, CENTER_DIM)
    toy_node_feat = torch.cat(
        [toy_coarse_rot_6d, toy_coarse_transl, toy_center_feat, toy_vel_scaled], dim=-1
    )
    toy_h = toy_gnn.encoder(toy_node_feat)
    toy_edge_feat = toy_gnn._compute_edge_features(
        toy_coarse_rot_6d, toy_coarse_transl, toy_centers, toy_vel_scaled
    )
    toy_layer = toy_gnn.layers[0]
    alpha_dense, mask = toy_layer._compute_attention(toy_h, toy_gnn.edge_index_dir, toy_edge_feat)

    has_nan = torch.isnan(alpha_dense).any().item()
    print(f"[attention sanity] alpha_dense shape={tuple(alpha_dense.shape)}  has_nan={has_nan}")
    assert not has_nan, "attention weights must never be NaN"

    row_sums = alpha_dense.sum(dim=1)  # (C, B, K), sum over neighbor axis j
    has_neighbor = mask.any(dim=1)  # (C,) True if receiver i has >= 1 in-neighbor
    for i in range(toy_num_clusters):
        if has_neighbor[i]:
            max_err = (row_sums[i] - 1.0).abs().max().item()
            print(f"[attention sanity] receiver {i} (has neighbors): max |row_sum - 1| = {max_err:.3e}")
            assert max_err < 1e-5, f"receiver {i}'s attention weights should sum to 1 over its neighbors"
        else:
            max_val = row_sums[i].abs().max().item()
            print(f"[attention sanity] receiver {i} (isolated): max |row_sum| = {max_val:.3e} (expect 0)")
            assert max_val == 0.0, f"isolated receiver {i} should have all-zero attention weights"

    non_edge_max = alpha_dense[~mask].abs().max().item() if (~mask).any() else 0.0
    print(f"[attention sanity] max |alpha| at non-edge positions = {non_edge_max:.3e} (expect 0)")
    assert non_edge_max == 0.0, "attention weight must be exactly 0 at non-edge (i, j) positions"

    # isolated cluster 3 must contribute exactly zero aggregation
    toy_out = toy_layer(toy_h, toy_gnn.edge_index_dir, toy_edge_feat)
    isolated_contribution = (toy_out[3] - (toy_h[3] + toy_layer.act(torch.zeros_like(toy_h[3])))).abs().max().item()
    print(f"[attention sanity] isolated cluster agg contribution = {isolated_contribution:.3e} (expect 0)")
    assert isolated_contribution < 1e-6, "isolated node's message aggregation should be exactly 0"

    # --- 4. attention is actually non-uniform across differing neighbors ---
    # receiver 1 has two neighbors (0 and 2) with independently random features
    # -> generically their attention weights differ.
    alpha_n0 = alpha_dense[1, 0]  # (B, K)
    alpha_n2 = alpha_dense[1, 2]  # (B, K)
    non_uniform_gap = (alpha_n0 - alpha_n2).abs().max().item()
    print(f"[attention sanity] receiver 1: |alpha(0->1) - alpha(2->1)| max = {non_uniform_gap:.3e} (uniform would be 0)")
    assert non_uniform_gap > 1e-4, "attention over distinct neighbors should not be exactly uniform"

    print(f"[vel_scale] {graph_bases.gnn.vel_scale.item():.3e}")

    # --- 7. save -> init_from_state_dict round-trip ---
    full_state_dict = {f"motion_bases.{k}": v for k, v in graph_bases.state_dict().items()}
    restored = RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases.init_from_state_dict(
        full_state_dict, prefix="motion_bases."
    )
    out_restored = restored.compute_transforms(ts, coefs, cluster_ids)
    out_original = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff_roundtrip = (out_restored - out_original).abs().max().item()
    vel_scale_diff = (restored.gnn.vel_scale - graph_bases.gnn.vel_scale).abs().item()
    num_heads_match = restored.gnn.num_heads == graph_bases.gnn.num_heads
    print(
        f"[round-trip] max |original - restored| = {max_diff_roundtrip:.3e}  "
        f"vel_scale diff = {vel_scale_diff:.3e}  num_heads match = {num_heads_match} "
        f"({restored.gnn.num_heads} vs {graph_bases.gnn.num_heads})"
    )
    assert max_diff_roundtrip < 1e-6, "save -> init_from_state_dict round-trip should reproduce identical output"
    assert vel_scale_diff < 1e-6, "vel_scale buffer should round-trip exactly"
    assert num_heads_match, "num_heads should be recovered exactly from the saved attn buffer shape"

    print("OK")

"""
flow3d/graph_relative_linear_attention.py

Relative motion/geometry와 linear velocity를 node/edge feature(각 15차원)로
사용하고, 이웃 message를 학습된 multi-head attention(GAT류)으로 합치는 독립
구현이다.
Angular velocity와 temporal smoothness loss는 이 파일의 범위 밖이다.

mean aggregation -> attention aggregation
-------------------------------------------
기존 _RelativeVelocityLinearMessageLayer는

    m_ij  = MLP([h_j - h_i, e_ij])
    agg_i = mean_{j in N(i)} m_ij
    h_i'  = h_i + act(agg_i)

였다. 이 파일은 mean을 학습된 attention 가중치 alpha_ij로 대체한다:

    m_ij  = MLP([h_j - h_i, e_ij])                          # message, 기존과 동일
    s_ij  = LeakyReLU( a^T [W h_i, W h_j, W_e e_ij] )         # GAT식 attention score (per head)
    alpha_ij = softmax_{j in N(i)}(s_ij)                      # 같은 receiver i의 이웃들에 대해 정규화
    agg_i = sum_{j in N(i)} alpha_ij * m_ij
    h_i'  = h_i + act(agg_i)

score 입력에 e_ij(=상대기하 + 상대속도)가 반드시 포함되어, 예를 들어 빠르게
분리 중인 이웃을 프레임 단위로 down-weight할 수 있다.

multi-head
----------
num_heads(기본 4)로 hidden_dim을 균등하게 나눠(hidden_dim % num_heads == 0
필요) 각 head가 독립적인 raw score/softmax를 갖고, message도 head_dim 단위로
쪼개 각 head의 alpha로 가중합한 뒤 다시 hidden_dim으로 reshape해서 복원한다
(별도의 output projection 없이 reshape만으로 복원 -- head들이 이미 하나의
공유 message MLP의 출력을 나눠 쓰기 때문). num_heads=1로도 그대로 동작한다
(__main__ 참고).

dense-masked softmax (torch_scatter 미사용)
---------------------------------------------
cluster graph는 (수십 개 노드로) 작으므로 segment-softmax 라이브러리 없이
(C, C, B, heads) dense score 텐서를 만들어 softmax한다:

    1. sparse edge_index_dir(E개 directed edge)에서만 score s_ij를 계산한다
       (존재하지 않는 (i,j) 쌍에 대해서는 계산하지 않음 -- O(E)).
    2. (C, C, B, heads) 텐서를 -inf로 채우고, 실제 edge 위치 (dst=i, src=j)에만
       s_ij를 채워 넣는다 (dense scatter).
    3. src(=j, neighbor) 축(dim=1)에서 softmax한다. exp(-inf) == 0이므로 존재하지
       않는 이웃은 자동으로 정확히 0의 가중치를 받는다.
    4. 이웃이 하나도 없는 receiver i는 해당 row 전체가 -inf라서 softmax가
       0/0 = NaN이 된다 -- torch.nan_to_num(alpha, nan=0.0)으로 그 row를
       전부 0으로 강제한다 (agg_i도 자연히 0이 된다).

    이 dense (C, C) 텐서는 softmax 정규화에만 쓰이고, score/message 자체의
    계산은 sparse edge 목록(E개)에서만 이뤄지므로 O(C^2)로 느려지는 부분은
    (C, C, B, heads) 텐서 할당/softmax뿐이다 -- cluster 수가 작으므로 문제
    없다. torch_scatter 등 외부 의존성은 추가하지 않는다.

zero-init 등가성
-----------------
head(omega/delta_t를 내는 마지막 Linear)는 여전히 0으로 초기화된다. attention
내부 파라미터(W_score/W_edge/attn 벡터)는 임의로 초기화되지만, head가 0이라
GNN 전체 출력(omega, delta_t)은 attention 가중치와 무관하게 정확히 0이다 --
zero-init 등가성은 attention 도입과 무관하게 그대로 보존된다.

공통 기반
---------
- so3_exp_map / compose_rotation / build_edge_index_from_edges_pt / cont_6d_to_rmat
  / rmat_to_cont_6d / CENTER_DIM / ROT_DIM / TRANSL_DIM / NODE_FEAT_DIM은
  flow3d/graph_coupling.py에서 가져다 쓴다.
- directed neighbor topology 생성과 linear-velocity feature 차원/정규화는 이
  파일에 직접 구현하여 다른 GNN variant 모듈에 의존하지 않는다.
- coarse correction 합성 수식(R_new = exp(omega) @ R_coarse, t_new = t_coarse +
  delta_t), coarse-to-fine 결합 수식, checkpoint save/load 포맷은 baseline
  relative/velocity_linear와 동일하다.

이 파일에서 제공하는 것
------------------------
- RelativeVelLinearAttentionClusterGraphGNN: RelativeVelocityLinearClusterGraphGNN과
  같은 API(node feature -> (omega, delta_t))를 갖지만, message aggregation이
  mean이 아니라 multi-head attention인 GNN.
- RelativeVelLinearAttentionGraphCorrectedScalableMotionBases: baseline
  RelativeVelocityLinearGraphCorrectedScalableMotionBases와 동일한 API
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
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "so3_exp_map",
    "compose_rotation",
    "build_edge_index_from_edges_pt",
    "RelativeVelLinearAttentionClusterGraphGNN",
    "RelativeVelLinearAttentionGraphCorrectedScalableMotionBases",
]

_LEAKY_SLOPE = 0.2
VEL_DIM = 3
EDGE_FEAT_DIM = TRANSL_DIM + CENTER_DIM + ROT_DIM
NODE_FEAT_DIM_VEL = NODE_FEAT_DIM + VEL_DIM
EDGE_FEAT_DIM_VEL = EDGE_FEAT_DIM + VEL_DIM


def _build_directed_neighbor_edges(
    edge_index: torch.Tensor, num_clusters: int
) -> torch.Tensor:
    """Convert undirected cluster pairs to deduplicated directed edges."""
    del num_clusters  # Kept in the signature for construction-call compatibility.
    if edge_index.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=edge_index.device)

    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    directed = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    )
    return torch.unique(directed, dim=1)


def _compute_vel_scale(transls: torch.Tensor) -> float:
    """Return a stable scale for frame-to-frame coarse translation deltas."""
    if transls.shape[1] < 2:
        return 1.0
    scale = (transls[:, 1:] - transls[:, :-1]).std().item()
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


class _RelativeVelLinearAttentionMessageLayer(nn.Module):
    """Fixed-topology relative message passing, 1 layer, residual -- same
    message MLP aggregated with learned multi-head attention instead of a uniform
    (1/in_degree) mean:

        m_ij     = MLP([h_j - h_i, e_ij])
        s_ij     = LeakyReLU(a^T [W h_i, W h_j, W_e e_ij])   (per head)
        alpha_ij = softmax_{j in N(i)}(s_ij)
        agg_i    = sum_{j in N(i)} alpha_ij * m_ij
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

        # message MLP: identical shape/role to the mean-aggregation baseline.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + EDGE_FEAT_DIM_VEL, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.ReLU()

        # GAT-style attention score: W (shared for both i and j roles) + W_e
        # for the edge feature, then a per-head scoring vector `attn`.
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
        """Dense-masked multi-head attention weights.

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
        m = self.mlp(torch.cat([delta_h, edge_feat], dim=-1))  # (E, B, H)
        m_heads = m.view(-1, B, self.num_heads, self.head_dim)  # (E, B, K, D)
        weighted = alpha_e.unsqueeze(-1) * m_heads  # (E, B, K, D)
        weighted_flat = weighted.reshape(-1, B, H)  # concat heads back to H

        agg = h.new_zeros(C, B, H)
        agg.index_add_(0, dst, weighted_flat)

        return h + self.act(agg)


class RelativeVelLinearAttentionClusterGraphGNN(nn.Module):
    """RelativeVelocityLinearClusterGraphGNN과 같은 API(node feature ->
    (omega, delta_t))를 갖는 relative-message GNN이지만, 이웃 message
    aggregation이 mean이 아니라 multi-head attention이다. node/edge feature
    (linear velocity 포함)는 baseline과 완전히 동일하다.

    Graph topology(directed neighbor edge, self-loop 제외)는 생성 시점에 고정돼
    학습 중 바뀌지 않는다. head는 0으로 초기화되므로 학습 시작 시점에는 correction이
    정확히 0이다(baseline과 동일한 zero-init 등가성 -- attention 도입과 무관하게
    유지된다).
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
                _RelativeVelLinearAttentionMessageLayer(hidden_dim, num_heads=num_heads)
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
        를 만든다. 한 forward
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
        # 바꾸지 않는다 -- 이 확장은 aggregation 방식만 바꿀 뿐이다)
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


class RelativeVelLinearAttentionGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + relative-message-with-velocity-and-attention
    graph-aware coarse transform correction. baseline
    RelativeVelocityLinearGraphCorrectedScalableMotionBases와 API가 동일한
    드롭인 대체이며, 차이는 내부 GNN이 RelativeVelLinearAttentionClusterGraphGNN
    (attention aggregation)이라는 점뿐이다.
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
        self.gnn = RelativeVelLinearAttentionClusterGraphGNN(
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
    ) -> "RelativeVelLinearAttentionGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 relative+velocity+attention
        graph-corrected 버전을 만든다. coarse/fine motion 파라미터 값은 그대로
        복사되고, GNN correction만 새로 추가된다 (correction은 0으로 초기화됨).
        vel_scale은 복사된 transls로부터 __init__ 안에서 다시 계산된다."""
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
        RelativeVelLinearAttentionGraphCorrectedScalableMotionBases로 저장된
        것인지 확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은 형태여야
        한다 (params.가 아니라)."""
        gnn_prefix = f"{prefix}gnn."
        return any(key.startswith(gnn_prefix) for key in state_dict)

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "RelativeVelLinearAttentionGraphCorrectedScalableMotionBases":
        """체크포인트만으로
        RelativeVelLinearAttentionGraphCorrectedScalableMotionBases를 통째로
        복원한다. edge_index/hidden_dim/num_layers는 저장된 텐서의 shape에서
        읽어온다. num_heads는 저장된 attention 벡터 "layers.0.attn"의
        shape[0]에서 그대로 읽는다 (attn은 (num_heads, 3*head_dim) 모양으로
        저장되므로 shape[0]가 곧 num_heads이다). directed edge/vel_scale 버퍼는
        __init__ 시점에 placeholder로 재계산되고, 아래 load_state_dict가 저장된
        실제 값으로 strict하게 덮어쓴다 (baseline relative/velocity_linear와
        동일한 패턴)."""
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
        message passing의 aggregation이 mean에서 attention으로 바뀐다는 점뿐이다).

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
        baseline RelativeVelocityLinearGraphCorrectedScalableMotionBases.compute_transforms와
        완전히 동일하며, coarse rotation/translation만 attention-aware
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

        # --- graph-aware coarse transform (baseline과 다른 부분: attention aggregation) ---
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
    # Verification (per task spec):
    #   1. zero-init equivalence: step 0 == plain ScalableMotionBases exactly,
    #      |omega|_max == 0, |delta_t|_max == 0.
    #   2. grad flow: head.weight.grad norm > 0, and (after one optimizer step
    #      moves the head off zero) attention score params also get nonzero grad.
    #   3. attention correctness: per-receiver neighbor alpha sums to 1,
    #      non-edge alpha == 0, isolated-node agg == 0, no NaN anywhere.
    #   4. attention is actually non-uniform for neighbors with different features.
    #   5. node/edge feature dims are 15/15 (unchanged from velocity_linear).
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
    graph_bases = RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
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
    print(f"[coarse]  max |baseline - velocity_linear_attention_graph_corrected(zero-init)| = {max_diff_coarse:.3e}")
    assert max_diff_coarse < 1e-5, "zero-init correction should reproduce the baseline coarse transform exactly"

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"[full]    max |baseline - velocity_linear_attention_graph_corrected(zero-init)| = {max_diff:.3e}")
    assert max_diff < 1e-5, "zero-init correction should reproduce the baseline transform exactly"

    omega = graph_bases.last_correction["omega"]
    delta_t = graph_bases.last_correction["delta_t"]
    print(f"[zero-init] |omega|_max={omega.abs().max().item():.3e}  |delta_t|_max={delta_t.abs().max().item():.3e}")
    assert omega.abs().max().item() == 0.0
    assert delta_t.abs().max().item() == 0.0

    # --- 2. grad flow: head first, then attention params after head moves off zero ---
    graph_bases.zero_grad()
    out2 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss = out2.pow(2).mean()
    loss.backward()

    head_grad_norm = graph_bases.gnn.head.weight.grad.norm().item()
    encoder_grad_norm = graph_bases.gnn.encoder[0].weight.grad.norm().item()
    attn_grad_norm_at_init = graph_bases.gnn.layers[0].attn.grad.norm().item()
    print(
        f"grad norm (zero-init): head.weight={head_grad_norm:.3e}  "
        f"encoder.weight={encoder_grad_norm:.3e}  layers[0].attn={attn_grad_norm_at_init:.3e}"
    )
    assert head_grad_norm > 0.0, "GNN head should receive nonzero gradient -- it will actually train"
    # Encoder/message-passing/attention layers are (correctly) gradient-starved
    # at init: the zero-initialized head multiplies their contribution by zero
    # in dL/dW because the output head is exactly zero.
    assert attn_grad_norm_at_init == 0.0, "attention params should also be zero-grad while head is exactly zero"

    with torch.no_grad():
        for p in graph_bases.gnn.head.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad  # a single manual SGD step moves head off zero

    graph_bases.zero_grad()
    out3 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss3 = out3.pow(2).mean()
    loss3.backward()
    attn_grad_norm = graph_bases.gnn.layers[0].attn.grad.norm().item()
    w_score_grad_norm = graph_bases.gnn.layers[0].W_score.weight.grad.norm().item()
    print(f"grad norm (post-step): layers[0].attn={attn_grad_norm:.3e}  layers[0].W_score.weight={w_score_grad_norm:.3e}")
    assert attn_grad_norm > 0.0, "attention score params should receive nonzero gradient once head is off zero"
    assert w_score_grad_norm > 0.0, "attention W_score should receive nonzero gradient once head is off zero"

    # --- 5/6. feature dims + num_heads=1 smoke test ---
    assert graph_bases.gnn.encoder[0].in_features == NODE_FEAT_DIM_VEL == 15, (
        "node encoder input dim should be unchanged from velocity_linear (12 + 3 velocity)"
    )
    assert graph_bases.gnn.layers[0].mlp[0].in_features == gnn_hidden_dim + EDGE_FEAT_DIM_VEL, (
        "message MLP input dim should be unchanged from velocity_linear (hidden_dim + 12 + 3 velocity)"
    )
    print(f"[feature dims] node={NODE_FEAT_DIM_VEL}, edge={EDGE_FEAT_DIM_VEL} (both unchanged from velocity_linear)")

    one_head_bases = RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
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
    toy_bases = RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
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
    restored = RelativeVelLinearAttentionGraphCorrectedScalableMotionBases.init_from_state_dict(
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

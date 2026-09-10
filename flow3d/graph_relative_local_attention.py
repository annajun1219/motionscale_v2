"""
flow3d/graph_relative_local_attention.py

`flow3d/graph_local_target.py`가 만드는 local-control-node 진단 그래프
(`graph_relative_local.pt`: edge_index, target_mask, message_source_mask,
node_confidence, node_center, node_parent_cluster_id)를 학습 가능한 SE(3)
correction으로 이어주는 두 번째 단계. `flow3d/graph_coupling.py` /
`flow3d/graph_relative_edge.py` / `flow3d/graph_relative_linear_attention_frame.py`와
같은 "ScalableMotionBases 서브클래스 + nn.Module GNN submodule + zero-init
decoder" 패턴을 그대로 따르되, 기존 variant들과 근본적으로 다른 점이 하나
있다: 기존 variant는 correction이 **cluster 전체**에 균일하게 적용되는
per-cluster rigid transform이지만, 여기서는 correction이 **local node**
(하나의 cluster 안의 더 촘촘한 하위 영역) 단위로 예측된 뒤, 개별 live
Gaussian마다 그 Gaussian 주변 top-k local node의 RBF-weighted 혼합으로
적용된다 -- `EdgeBoundaryGraphCorrectedScalableMotionBases`(개별 Gaussian
identity에 의존)에 더 가까운 적용 지점(coarse+fine이 이미 blend된 뒤,
compute_transforms에서)을 쓰지만 그 확산 방식(top-k RBF)은 이 파일에서
새로 정의한다.

Node state: raw track이 아니라 MotionScale 적용 결과
------------------------------------------------------
매 프레임 각 local node의 위치/회전은 raw 2D-track이 아니라, 그 node의
hard-membership Gaussian들에 실제로 적용된 MotionScale pose
(coarse+fine blended `compute_transforms`의 결과를 canonical mean에
곱한 것, `R_g @ x0_g + t_g`)를 member별로 평균해서 얻는다 -- 단순히
transform의 translation 성분만 평균하는 게 아니라 실제 posed position을
평균한다. node rotation은 그 member들의 posed rotmat을 quaternion 평균한다.

Hard membership(어느 Gaussian이 어느 node에 속하는지, k=1 최근접
`node_center` 규칙 -- `assign_gaussians_to_local_nodes`와 동일한 규칙)과
top-k candidate set(어느 Gaussian이 어느 top-k node들의 RBF 혼합
대상인지)은 Gaussian 개수가 바뀔 때(densify/cull)뿐 아니라 학습 중 100
step마다도 `refresh_local_node_assignment`로 다시 계산된다 -- canonical
mean 자체가 학습되며 계속 움직이므로, 재계산 없이는 assignment가 서서히
낡는다. 이 재계산은 `rest_center_i`(그 node 현재 멤버들의 canonical 위치
평균 -- edge feature d_ij^0과 top-k RBF 거리 기준점으로 쓰이는, 저장된
`node_center` seed와는 다른 값)도 함께 갱신한다. `node_center` seed
자체는 이 reassignment의 anchor로만 쓰이고 절대 바뀌지 않는다. top-k RBF
거리/가중치는 매 forward마다 CURRENT canonical Gaussian 위치로 새로
계산된다(assignment 자체는 그대로 두고).

Message passing: 누가 source가 될 수 있는가
---------------------------------------------
`graph_local_target.py`가 이미 계산해 저장한 `message_source_mask[T,N]`
(초록 context node -- 항상 자기 confidence 기준, 그리고 초록 테두리 빨강
target -- red AND 자기 confidence도 높음, 주황 target은 항상 제외)를
그대로 gate로 쓴다. 이 파일은 message_source_mask를 다시 계산하지
않는다. edge j->i는 message_source_mask[t, j]가 True일 때만 메시지를
보낸다 -- `graph_relative_linear_attention_frame.py`의 "gate를 softmax
이전에 log-space로 더하는" NaN-안전 dense-masked-softmax를 그대로
재사용한다(모든 in-neighbor가 비활성인 행 -- 즉 source가 0개인 target --
이 forward/backward 어디서도 NaN을 내지 않고 정확히 0으로 떨어지는 것이
핵심이며, 실측으로 이미 검증된 메커니즘). target이 매 프레임 1개 이상의
유효 source를 가지면 정상적으로 attention을 계산하고, 0개면(temporal
fallback 없음 -- 첫 구현에서는 생략) decode 이후 명시적으로 correction을
0으로 마스킹한다. context node의 correction도 항상 0으로 마스킹된다
(target_mask로 게이트).

Edge feature (source j -> target i)
------------------------------------
- d_ij^0 = rest_center_i - rest_center_j (canonical, membership 갱신 시에만 바뀜)
- e_ij^t = p_j^t + R_j^t @ d_ij^0 - p_i^t (rigid-consistency 잔차: j와 강체로
  같이 움직였다면 i가 있어야 할 위치 - 실제 위치)
- 상대 velocity v_j^t - v_i^t, 상대 acceleration a_j^t - a_i^t (모두 backward
  difference, 경계 프레임은 clamp)
- source confidence: node_confidence[t, j]

Pivoted 적용과 anchor loss
----------------------------
correction은 Gaussian마다 top-k RBF로 혼합된 (omega, delta_t, pivot)을
"pivot 기준으로 회전 후 이동"으로 적용한다(원점 기준 아님 -- 원점 기준으로
composeatch하면 pivot에서 먼 Gaussian일수록 회전만으로도 부당하게 크게
움직인다). `flow3d/analysis/loss.py`의 `local_gnn_anchor_loss`가 쓰는
"실제 corrected 위치"도 이 pivoted per-Gaussian 공식을 각 member에 실제로
적용한 뒤 평균한 값이다 -- node 자신의 (p_i + delta_t_i)로 근사하지 않는다.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from flow3d.graph_coupling import CENTER_DIM, ROT_DIM, TRANSL_DIM, compose_rotation, so3_exp_map
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "LocalRelativeAttentionGNN",
    "LocalRelativeAttentionGraphCorrectedScalableMotionBases",
]

_LEAKY_SLOPE = 0.2
VEL_DIM = 3
ACCEL_DIM = 3
CONF_DIM = 1
# d_ij^0(3) + e_ij^t(3) + rel_vel(3) + rel_accel(3) + source_confidence(1)
EDGE_FEAT_DIM = CENTER_DIM + TRANSL_DIM + VEL_DIM + ACCEL_DIM + CONF_DIM
# rot6d(6) + p_i^t(3) + rest_center_i(3) + v_i(3) + a_i(3) + node_confidence(1)
NODE_FEAT_DIM = ROT_DIM + TRANSL_DIM + CENTER_DIM + VEL_DIM + ACCEL_DIM + CONF_DIM


def _quat_mean(rotmats: torch.Tensor, dim: int) -> torch.Tensor:
    """Sign-consistent quaternion mean + renormalize, reduced over `dim`.
    Adequate approximation of a full Karcher mean here since the inputs are
    always one local node's member Gaussians' rotations, which are spatially
    tight and hence already close to each other in rotation.

    :param rotmats: (..., M, ..., 3, 3), M at position `dim`.
    :return: (..., 3, 3), M reduced away.
    """
    quats = roma.rotmat_to_unitquat(rotmats)  # (..., M, ..., 4)
    ref = quats.select(dim, 0).unsqueeze(dim)
    sign = torch.where((quats * ref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    quats = quats * sign
    mean_quat = F.normalize(quats.mean(dim=dim), dim=-1, eps=1e-8)
    return roma.unitquat_to_rotmat(mean_quat)


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, num_out: int) -> torch.Tensor:
    """(G, ...) values grouped by (G,) index in [0, num_out) -> (num_out, ...)
    mean, 0 for an output row with no contributing input rows. index entries
    < 0 are dropped (unassigned)."""
    valid = index >= 0
    out_shape = (num_out,) + values.shape[1:]
    summed = values.new_zeros(out_shape)
    counts = values.new_zeros(num_out)
    if bool(valid.any()):
        idx = index[valid]
        summed.index_add_(0, idx, values[valid])
        counts.index_add_(0, idx, values.new_ones(idx.shape[0]))
    return summed / counts.clamp_min(1.0).view((num_out,) + (1,) * (values.dim() - 1))


class _LocalMessageLayer(nn.Module):
    """graph_relative_linear_attention_frame.py's
    _RelativeVelLinearAttentionFrameMessageLayer와 동일한 multi-head
    dense-masked-softmax attention + message MLP + residual 구조. gate는
    [0,1] 연속값이 아니라 message_source_mask 유래의 이진(0/1) 값이라는 점만
    다르다 -- 메커니즘(NaN-안전 masked softmax, log-gate를 score에 더하는
    방식)은 완전히 동일하게 재사용한다."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + EDGE_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.ReLU()

        self.W_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_edge = nn.Linear(EDGE_FEAT_DIM, hidden_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)

    def _compute_attention(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_source_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param h: (N, B, H)
        :param edge_index: (2, E) directed [src(j), dst(i)] pairs.
        :param edge_feat: (E, B, EDGE_FEAT_DIM)
        :param edge_source_gate: (E, B) in {0, 1} -- message_source_mask[ts, src].
        :return: alpha_dense (N, N, B, heads), alpha_dense[i, j] = weight of j->i.
        """
        N, B, H = h.shape
        K, D = self.num_heads, self.head_dim
        if edge_index.numel() == 0:
            return h.new_zeros(N, N, B, K)

        src, dst = edge_index[0], edge_index[1]
        proj = self.W_score(h).view(N, B, K, D)
        edge_proj = self.W_edge(edge_feat).view(-1, B, K, D)

        proj_i, proj_j = proj[dst], proj[src]
        concat = torch.cat([proj_i, proj_j, edge_proj], dim=-1)  # (E, B, K, 3D)
        raw_score = (concat * self.attn).sum(-1)  # (E, B, K)
        score_e = F.leaky_relu(raw_score, negative_slope=_LEAKY_SLOPE)

        # Binary gate in log-space, added to the raw score before the dense
        # scatter+softmax -- source (gate=1) contributes log(1)=0 (no change),
        # non-source (gate=0) becomes -inf (fully excluded), exactly the same
        # mechanism the frame-gated coarse variant uses for its continuous
        # [0,1] gate (see that file's module docstring for why this, not
        # "softmax first then multiply", is correct).
        log_gate = torch.where(
            edge_source_gate > 0,
            torch.zeros_like(edge_source_gate),
            torch.full_like(edge_source_gate, float("-inf")),
        )
        score_e = score_e + log_gate.unsqueeze(-1)

        score_dense = h.new_full((N, N, B, K), float("-inf"))
        score_dense[dst, src] = score_e

        # A target with 0 valid sources this frame -> its whole row is -inf.
        # Manual masked softmax (not torch.softmax + nan_to_num) so this is
        # exactly 0 in BOTH forward and backward, never NaN -- see
        # graph_relative_linear_attention_frame.py's module docstring for the
        # confirmed failure mode this avoids.
        row_max = score_dense.detach().max(dim=1, keepdim=True).values
        row_max_safe = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_score = torch.exp(score_dense - row_max_safe)
        denom = exp_score.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return exp_score / denom

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_source_gate: torch.Tensor,
    ) -> torch.Tensor:
        N, B, H = h.shape
        if edge_index.numel() == 0:
            return h + self.act(h.new_zeros(N, B, H))

        src, dst = edge_index[0], edge_index[1]
        alpha_dense = self._compute_attention(h, edge_index, edge_feat, edge_source_gate)
        alpha_e = alpha_dense[dst, src]  # (E, B, K)

        delta_h = h[src] - h[dst]
        m = self.mlp(torch.cat([delta_h, edge_feat], dim=-1))  # (E, B, H)
        m_heads = m.view(-1, B, self.num_heads, self.head_dim)
        weighted = alpha_e.unsqueeze(-1) * m_heads
        weighted_flat = weighted.reshape(-1, B, H)

        agg = h.new_zeros(N, B, H)
        agg.index_add_(0, dst, weighted_flat)
        return h + self.act(agg)


class LocalRelativeAttentionGNN(nn.Module):
    """Node feature -> (omega, delta_t) per local node, gated so only target
    nodes ever get a nonzero correction and only frames with >=1 valid
    spatial source do (module docstring). Holds the FROZEN local-node graph
    (loaded once from graph_relative_local.pt) as buffers -- topology and
    per-frame source/confidence data never change during training, only the
    live Gaussian<->node assignment (owned by the wrapper class, also stored
    here as buffers so a single `self.gnn.load_state_dict(...)` restores it
    on checkpoint load) does.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,  # (2, E) directed, already bidirectional
        target_mask: torch.Tensor,  # (N,) bool
        message_source_mask: torch.Tensor,  # (T, N) bool
        node_confidence: torch.Tensor,  # (T, N) float
        node_center_seed: torch.Tensor,  # (N, 3) float, frozen reassignment anchor
        node_parent_cluster_id: torch.Tensor,  # (N,) long
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        topk: int = 4,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        N = target_mask.shape[0]
        T = message_source_mask.shape[0]
        if message_source_mask.shape[1] != N or node_confidence.shape != (T, N):
            raise ValueError(
                f"Shape mismatch: target_mask {tuple(target_mask.shape)}, "
                f"message_source_mask {tuple(message_source_mask.shape)}, "
                f"node_confidence {tuple(node_confidence.shape)} must agree on N (and T)."
            )

        self.register_buffer("edge_index", edge_index.clone().long())
        self.register_buffer("target_mask", target_mask.clone().bool())
        self.register_buffer("message_source_mask", message_source_mask.clone().bool())
        self.register_buffer("node_confidence", node_confidence.clone().float())
        self.register_buffer("node_center_seed", node_center_seed.clone().float())
        self.register_buffer("node_parent_cluster_id", node_parent_cluster_id.clone().long())
        self.register_buffer("node_sigma", self._compute_node_sigma(node_center_seed, node_parent_cluster_id))

        self.num_nodes = N
        self.num_frames = T
        self.num_heads = num_heads
        self.topk = topk

        # Live Gaussian<->node assignment (see refresh_local_node_assignment on
        # the wrapper class): placeholders here, filled by the first refresh
        # call the wrapper's from_scalable_motion_bases always makes.
        self.register_buffer("hard_member_node_id", torch.zeros(0, dtype=torch.long))
        self.register_buffer("topk_node_id", torch.zeros(0, topk, dtype=torch.long))
        self.register_buffer("rest_center", node_center_seed.clone().float())
        # True iff this node currently has >=1 hard-membership Gaussian --
        # refresh_local_node_assignment recomputes this every refresh.
        # Defaults to all-True as a construction-time placeholder (before the
        # first real refresh_local_node_assignment call), matching how
        # rest_center defaults to node_center_seed above.
        self.register_buffer("active_node_mask", torch.ones(N, dtype=torch.bool))

        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [_LocalMessageLayer(hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, ROT_DIM // 2 + TRANSL_DIM)  # (omega(3), delta_t(3))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _compute_node_sigma(node_center_seed: torch.Tensor, node_parent_cluster_id: torch.Tensor) -> torch.Tensor:
        """Per-node RBF bandwidth: the median nearest-OTHER-node spacing
        within that node's own parent cluster (frozen seed positions, so this
        bandwidth doesn't flicker as rest_center drifts during training). A
        cluster with only 1 node gets a fallback of 1.0 (never divides by 0;
        with only one node the RBF weight is degenerately always 1 anyway)."""
        centers_np = node_center_seed.detach().cpu().double().numpy()
        cluster_ids_np = node_parent_cluster_id.detach().cpu().numpy()
        sigma = torch.ones(node_center_seed.shape[0], dtype=torch.float32)
        for cid in set(cluster_ids_np.tolist()):
            idx = (cluster_ids_np == cid).nonzero()[0]
            if idx.size < 2:
                continue
            pts = centers_np[idx]
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)  # rank 0 is self (dist 0)
            median_spacing = float(torch.tensor(dists[:, 1]).median())
            sigma[idx] = max(median_spacing, 1e-6)
        return sigma

    def forward(
        self,
        node_rot6d: torch.Tensor,  # (N, B, 6)
        node_pos: torch.Tensor,  # (N, B, 3)
        rest_center: torch.Tensor,  # (N, 3)
        node_vel: torch.Tensor,  # (N, B, 3)
        node_accel: torch.Tensor,  # (N, B, 3)
        ts: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: omega (N, B, 3), delta_t (N, B, 3) -- both exactly 0 for
            non-target nodes, for inactive nodes (0 current hard-membership
            Gaussians, active_node_mask False), and for (target, frame) pairs
            with 0 valid sources; predicted_by_source (N, B, 3) -- each
            node's attention-free rigid-consistency prediction averaged over
            its valid (active) sources (for the anchor loss, computed even
            for non-target rows -- cheap, and harmless since only target rows
            are used); has_source (N, B) bool, already False for inactive
            destinations too.
        """
        N, B = self.num_nodes, ts.shape[0]
        node_confidence_t = self.node_confidence[ts].transpose(0, 1)  # (N, B)

        node_feat = torch.cat(
            [
                node_rot6d,
                node_pos,
                rest_center[:, None, :].expand(N, B, CENTER_DIM),
                node_vel,
                node_accel,
                node_confidence_t.unsqueeze(-1),
            ],
            dim=-1,
        )  # (N, B, NODE_FEAT_DIM)
        h = self.encoder(node_feat)

        src, dst = (self.edge_index[0], self.edge_index[1]) if self.edge_index.numel() > 0 else (None, None)
        if src is not None:
            d0 = rest_center[dst] - rest_center[src]  # (E, 3): i's canonical offset from j
            d0 = d0[:, None, :].expand(-1, B, CENTER_DIM)
            R_j = cont_6d_to_rmat(node_rot6d[src])  # (E, B, 3, 3)
            predicted_i_by_j = node_pos[src] + torch.einsum("ebij,ebj->ebi", R_j, d0)  # (E, B, 3)
            e_ij = predicted_i_by_j - node_pos[dst]  # (E, B, 3)
            rel_vel = node_vel[src] - node_vel[dst]
            rel_accel = node_accel[src] - node_accel[dst]
            source_conf = node_confidence_t[src].unsqueeze(-1)  # (E, B, 1)
            edge_feat = torch.cat([d0, e_ij, rel_vel, rel_accel, source_conf], dim=-1)

            gate = self.message_source_mask[ts][:, src].float().transpose(0, 1)  # (E, B)
            # A node with 0 current hard-membership Gaussians has a
            # degenerate node_pos/node_rot6d (all-zero / identity, from
            # _scatter_mean's/_quat_mean's empty-group default in
            # _node_state) -- never a real message source, regardless of
            # message_source_mask (which is computed offline and can't know
            # about live membership dropping to 0 after densify/cull).
            gate = gate * self.active_node_mask[src].float()[:, None]
        else:
            edge_feat = h.new_zeros(0, B, EDGE_FEAT_DIM)
            gate = h.new_zeros(0, B)
            predicted_i_by_j = h.new_zeros(0, B, 3)

        for layer in self.layers:
            h = layer(h, self.edge_index, edge_feat, gate)

        raw = self.head(h)  # (N, B, 6)
        raw_omega, raw_delta_t = raw.split([3, 3], dim=-1)

        if src is not None:
            source_count = h.new_zeros(N, B)
            source_count.index_add_(0, dst, gate)
            has_source = source_count > 0  # (N, B)

            predicted_sum = h.new_zeros(N, B, 3)
            predicted_sum.index_add_(0, dst, gate.unsqueeze(-1) * predicted_i_by_j)
            predicted_by_source = predicted_sum / source_count.clamp_min(1.0).unsqueeze(-1)
        else:
            has_source = torch.zeros(N, B, dtype=torch.bool, device=h.device)
            predicted_by_source = h.new_zeros(N, B, 3)

        # An inactive destination (0 current members) has no real Gaussians
        # to correct or predict a position for -- excluded here so has_source
        # (and hence predicted_by_source) is already False/meaningless-but-
        # unused for it wherever a caller reads it (e.g. local_gnn_anchor_terms),
        # not just in the omega/delta_t masking below.
        has_source = has_source & self.active_node_mask[:, None]

        active = self.target_mask[:, None] & has_source  # (N, B)
        omega = raw_omega * active.unsqueeze(-1)
        delta_t = raw_delta_t * active.unsqueeze(-1)
        return omega, delta_t, predicted_by_source, has_source


def _load_local_graph(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"local graph not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("edge_index", "target_mask", "message_source_mask", "node_confidence", "node_center", "node_parent_cluster_id")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"{path} is missing required key(s) {missing} -- expected the output of "
            "flow3d/graph_local_target.py (which now saves message_source_mask; "
            "regenerate it if this file predates that field)."
        )
    return payload


class LocalRelativeAttentionGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + local-node attention GNN correction, applied
    per-Gaussian via a top-k RBF blend of nearby local nodes' (omega,
    delta_t), pivoted at the blended nodes' own live position -- see module
    docstring. compute_transforms_coarse is inherited unmodified (like
    EdgeBoundaryGraphCorrectedScalableMotionBases): this correction applies
    to the fully coarse+fine blended per-Gaussian result, not the cluster
    skeleton.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        rots: torch.Tensor,
        transls: torch.Tensor,
        fine_rots: torch.Tensor,
        fine_transls: torch.Tensor,
        edge_index: torch.Tensor,
        target_mask: torch.Tensor,
        message_source_mask: torch.Tensor,
        node_confidence: torch.Tensor,
        node_center_seed: torch.Tensor,
        node_parent_cluster_id: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        topk: int = 4,
    ):
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        self.gnn = LocalRelativeAttentionGNN(
            edge_index=edge_index,
            target_mask=target_mask,
            message_source_mask=message_source_mask,
            node_confidence=node_confidence,
            node_center_seed=node_center_seed,
            node_parent_cluster_id=node_parent_cluster_id,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            topk=topk,
        )
        # One compute_transforms() call's outputs, stored atomically -- see
        # last_local_correction. Live (non-buffer) references to the current
        # full foreground-Gaussian arrays, supplied by refresh_local_node_assignment
        # and re-read every forward (see module docstring: canonical means
        # are themselves trainable, so this must track the SAME tensor
        # object fg.params["means"] owns, not a stale detached copy). Held in
        # a plain dict, NOT bare attributes: canonical_means is an
        # nn.Parameter, and nn.Module.__setattr__ auto-registers any
        # nn.Parameter assigned to a bare attribute as one of THIS module's
        # own parameters (silently double-counting it in the optimizer, and
        # then refusing a later reassignment to a plain Tensor after
        # densify/cull creates a fresh non-Parameter means tensor) -- a dict
        # value is invisible to that auto-registration.
        self._last_local_correction: dict[str, torch.Tensor] | None = None
        self._live_refs: dict[str, torch.Tensor] = {}

    @property
    def _canonical_means_ref(self) -> torch.Tensor | None:
        return self._live_refs.get("canonical_means")

    @property
    def _cluster_ids_all_ref(self) -> torch.Tensor | None:
        return self._live_refs.get("cluster_ids_all")

    @property
    def _coefs_all_ref(self) -> torch.Tensor | None:
        return self._live_refs.get("coefs_all")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        local_graph_path: str | Path,
        canonical_means: torch.Tensor,
        cluster_ids_all: torch.Tensor,
        coefs_all: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        topk: int = 4,
    ) -> "LocalRelativeAttentionGraphCorrectedScalableMotionBases":
        """Wrap an existing (already-initialized) ScalableMotionBases with a
        fresh (zero-init) local-node attention GNN, loading the local-node
        graph from a flow3d/graph_local_target.py output.

        :param canonical_means: (G, 3) current canonical foreground means
            (e.g. model.fg.params["means"] -- a LIVE reference; do not detach,
            see refresh_local_node_assignment).
        :param cluster_ids_all: (G,) current foreground cluster ids, same order.
        :param coefs_all: (G, F) current foreground fine-basis blend weights,
            same order.
        """
        graph = _load_local_graph(local_graph_path)
        p = bases.params
        self = cls(
            centers=p["centers"].detach().clone(),
            rots=p["rots"].detach().clone(),
            transls=p["transls"].detach().clone(),
            fine_rots=p["fine_rots"].detach().clone(),
            fine_transls=p["fine_transls"].detach().clone(),
            edge_index=graph["edge_index"],
            target_mask=graph["target_mask"],
            message_source_mask=graph["message_source_mask"],
            node_confidence=graph["node_confidence"],
            node_center_seed=graph["node_center"],
            node_parent_cluster_id=graph["node_parent_cluster_id"],
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_num_heads=gnn_num_heads,
            topk=topk,
        )
        self.refresh_local_node_assignment(canonical_means, cluster_ids_all, coefs_all)
        return self

    @staticmethod
    def has_gnn_state(state_dict: dict, prefix: str) -> bool:
        """`node_center_seed` is a buffer unique to this variant among every
        *GraphCorrectedScalableMotionBases class -- see flow3d/scene_model.py's
        init_from_state_dict dispatch chain."""
        return f"{prefix}gnn.node_center_seed" in state_dict

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
        canonical_means: torch.Tensor | None = None,
        cluster_ids_all: torch.Tensor | None = None,
        coefs_all: torch.Tensor | None = None,
    ) -> "LocalRelativeAttentionGraphCorrectedScalableMotionBases":
        """
        :param canonical_means, cluster_ids_all, coefs_all: the FULL live
            foreground array this checkpoint's `fg` was just reconstructed
            into (e.g. flow3d/scene_model.py's SceneModel.init_from_state_dict
            already has `fg` in scope by the time it dispatches here -- see
            that method). REQUIRED, not optional in practice: without an
            immediate refresh_local_node_assignment call, self._live_refs
            stays empty (it's a plain dict, not a restorable buffer -- see
            __init__/refresh_local_node_assignment's docstring) until the
            trainer's own next periodic/densify-triggered refresh, which can
            be up to --optim.local-gnn-refresh-every steps away (or, on a
            resume whose checkpoint global_step isn't a clean multiple of it,
            arbitrarily delayed) -- so the very FIRST compute_transforms call
            after loading (the main render pass, not just the anchor loss)
            would raise. Kept optional in the signature only so a raw
            state-shape inspection (no live model) doesn't need them; passing
            None here reproduces that crash on first use, by design (no
            guessing a fallback).
        """
        base = ScalableMotionBases.init_from_state_dict(state_dict, prefix=f"{prefix}params.")

        gnn_prefix = f"{prefix}gnn."
        gnn_keys = [k for k in state_dict if k.startswith(gnn_prefix)]
        if not gnn_keys:
            raise KeyError(f"No '{gnn_prefix}*' keys found in state_dict.")

        hidden_dim = state_dict[f"{gnn_prefix}encoder.0.weight"].shape[0]
        layers_prefix = f"{gnn_prefix}layers."
        layer_indices = {
            int(k[len(layers_prefix):].split(".", 1)[0]) for k in gnn_keys if k.startswith(layers_prefix)
        }
        num_layers = max(layer_indices) + 1
        num_heads = state_dict[f"{layers_prefix}0.attn"].shape[0]
        topk = state_dict[f"{gnn_prefix}topk_node_id"].shape[1]

        p = base.params
        self = cls(
            centers=p["centers"].detach().clone(),
            rots=p["rots"].detach().clone(),
            transls=p["transls"].detach().clone(),
            fine_rots=p["fine_rots"].detach().clone(),
            fine_transls=p["fine_transls"].detach().clone(),
            edge_index=state_dict[f"{gnn_prefix}edge_index"],
            target_mask=state_dict[f"{gnn_prefix}target_mask"],
            message_source_mask=state_dict[f"{gnn_prefix}message_source_mask"],
            node_confidence=state_dict[f"{gnn_prefix}node_confidence"],
            node_center_seed=state_dict[f"{gnn_prefix}node_center_seed"],
            node_parent_cluster_id=state_dict[f"{gnn_prefix}node_parent_cluster_id"],
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
            gnn_num_heads=num_heads,
            topk=topk,
        )
        # Placeholder buffers for the LIVE (non-frozen) assignment state, sized
        # to match the checkpoint, so the strict load below can fill them in --
        # refresh_local_node_assignment will recompute them again from live
        # data at the next scheduled refresh regardless.
        num_fg_at_save = state_dict[f"{gnn_prefix}hard_member_node_id"].shape[0]
        self.gnn.hard_member_node_id = torch.zeros(num_fg_at_save, dtype=torch.long)
        self.gnn.topk_node_id = torch.zeros(num_fg_at_save, topk, dtype=torch.long)

        gnn_state = {k[len(gnn_prefix):]: v for k, v in state_dict.items() if k.startswith(gnn_prefix)}
        self.gnn.load_state_dict(gnn_state, strict=True)

        if canonical_means is None or cluster_ids_all is None or coefs_all is None:
            raise ValueError(
                "LocalRelativeAttentionGraphCorrectedScalableMotionBases.init_from_state_dict requires "
                "canonical_means/cluster_ids_all/coefs_all (the just-reconstructed fg's live foreground "
                "array) to populate self._live_refs and re-derive the hard-membership/top-k assignment -- "
                "without this, the first compute_transforms call would raise "
                "'refresh_local_node_assignment must be called at least once'."
            )
        # Re-derives hard_member_node_id/topk_node_id/rest_center/active_node_mask
        # from the CURRENT live canonical means (the same array this checkpoint's
        # fg was just built from, so this reproduces exactly what was saved) and,
        # critically, populates self._live_refs -- load_state_dict above restores
        # every BUFFER correctly, but _live_refs is a plain dict (see __init__),
        # invisible to state_dict, and must be set immediately here rather than
        # left to the trainer's next periodic/densify-triggered refresh.
        self.refresh_local_node_assignment(canonical_means, cluster_ids_all, coefs_all)
        return self

    # ------------------------------------------------------------------
    # Live Gaussian<->node assignment refresh
    # ------------------------------------------------------------------

    @torch.no_grad()
    def refresh_local_node_assignment(
        self, canonical_means: torch.Tensor, cluster_ids_all: torch.Tensor, coefs_all: torch.Tensor
    ) -> None:
        """Recomputes, from CURRENT canonical positions:
          - hard_member_node_id (G,): each live fg Gaussian's k=1 nearest node
            (nearest-`node_center_seed`, same rule assign_gaussians_to_local_nodes
            uses), restricted to nodes sharing that Gaussian's parent cluster;
            -1 for a Gaussian whose cluster has no nodes in this graph at all.
          - topk_node_id (G, k): the k nearest such nodes (candidates for the
            RBF blend), -1-padded if the cluster has fewer than k nodes.
          - rest_center (N, 3): mean canonical position of each node's CURRENT
            hard-membership Gaussians (0 for a node with 0 members this refresh
            -- see active_node_mask below).
          - active_node_mask (N,): True iff the node has >=1 hard-membership
            Gaussian this refresh. A node CAN legitimately drop to 0 members
            (e.g. every Gaussian nearest it got culled) even though
            local_nodes.pt's own candidacy criteria required a minimum member
            count AT BUILD TIME -- that guarantee doesn't survive densify/cull
            or reassignment against live positions. Such a node is excluded
            everywhere downstream (message source, target correction,
            predicted-anchor aggregation, anchor loss -- see
            LocalRelativeAttentionGNN.forward and local_gnn_anchor_terms) since
            its p_i^t/R_i^t would otherwise be a degenerate all-zero/identity
            value with no real Gaussian behind it.
        Also re-anchors the live (non-buffer) full-array references this
        class's compute_transforms needs every forward (canonical means are
        themselves trainable and keep moving between refreshes -- holding the
        SAME tensor object, not a detached snapshot, keeps every forward
        pass using genuinely current values without needing a refresh call
        of its own).

        Call on densify/cull (num_fg_gaussians changed) AND periodically
        (e.g. every 100 training steps) even without a count change -- see
        module docstring.
        """
        self._live_refs["canonical_means"] = canonical_means
        self._live_refs["cluster_ids_all"] = cluster_ids_all
        self._live_refs["coefs_all"] = coefs_all

        device = canonical_means.device
        G = canonical_means.shape[0]
        N = self.gnn.num_nodes
        k = self.gnn.topk

        canonical_np = canonical_means.detach().cpu().double().numpy()
        cluster_ids_np = cluster_ids_all.detach().cpu().numpy()
        node_cluster_np = self.gnn.node_parent_cluster_id.detach().cpu().numpy()
        node_center_np = self.gnn.node_center_seed.detach().cpu().double().numpy()

        hard_member = torch.full((G,), -1, dtype=torch.long)
        topk_ids = torch.full((G, k), -1, dtype=torch.long)

        for cid in set(node_cluster_np.tolist()):
            node_idx = (node_cluster_np == cid).nonzero()[0]
            if node_idx.size == 0:
                continue
            gaussian_idx = (cluster_ids_np == cid).nonzero()[0]
            if gaussian_idx.size == 0:
                continue
            tree = cKDTree(node_center_np[node_idx])
            k_eff = min(k, node_idx.size)
            dists, nearest = tree.query(canonical_np[gaussian_idx], k=k_eff)
            nearest = nearest.reshape(gaussian_idx.size, k_eff)
            global_node_ids = node_idx[nearest]  # (num_gaussians_in_cluster, k_eff)
            hard_member[gaussian_idx] = torch.from_numpy(global_node_ids[:, 0]).long()
            topk_ids[gaussian_idx, :k_eff] = torch.from_numpy(global_node_ids).long()

        hard_member = hard_member.to(device)
        topk_ids = topk_ids.to(device)
        rest_center = _scatter_mean(canonical_means.detach().float(), hard_member, N)
        member_count = torch.bincount(hard_member[hard_member >= 0], minlength=N)
        active_node_mask = member_count > 0

        self.gnn.hard_member_node_id = hard_member
        self.gnn.topk_node_id = topk_ids
        self.gnn.rest_center = rest_center
        self.gnn.active_node_mask = active_node_mask

    # ------------------------------------------------------------------
    # Node state (per frame) from the FULL live foreground array
    # ------------------------------------------------------------------

    def _node_state(
        self, ts: torch.Tensor, full_coefs: torch.Tensor, full_cluster_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(rot6d, pos) per node at ts, plus the FULL per-Gaussian base
        (coarse+fine, pre-correction) transform this call needed anyway --
        the caller reuses it to avoid recomputing compute_transforms twice
        for the same ts.

        :param full_coefs, full_cluster_ids: the FULL foreground array's
            fine-basis coefs / cluster ids for THIS call -- must be freshly
            fetched (e.g. model.fg.get_coefs()) by the caller, not cached
            across calls: get_coefs() recomputes F.softmax(motion_coefs)
            every call, so it's a fresh graph node each time, unlike
            _canonical_means_ref (a stable nn.Parameter object whose .data is
            updated in-place, so caching THAT reference is safe/correct)."""
        if self._canonical_means_ref is None:
            raise RuntimeError(
                "refresh_local_node_assignment must be called at least once (from_scalable_motion_bases "
                "already does this) before compute_transforms."
            )
        full_transforms = ScalableMotionBases.compute_transforms(
            self, ts, full_coefs, full_cluster_ids
        )  # (G, B, 3, 4)
        rotmat, transl = full_transforms[..., :3], full_transforms[..., 3]
        posed = torch.einsum("gbij,gj->gbi", rotmat, self._canonical_means_ref) + transl  # (G, B, 3)

        N = self.gnn.num_nodes
        member = self.gnn.hard_member_node_id
        node_pos = _scatter_mean(posed, member, N)  # (N, B, 3)

        # Node rotation = quaternion mean of member rotmats, grouped per node
        # (N is small -- a handful to a few dozen local nodes per checkpoint
        # -- so a python loop over N, not G, is cheap).
        B = rotmat.shape[1]
        node_rotmat = torch.eye(3, device=rotmat.device, dtype=rotmat.dtype)[None, None].expand(N, B, 3, 3).clone()
        for n in range(N):
            sel = member == n
            if bool(sel.any()):
                node_rotmat[n] = _quat_mean(rotmat[sel], dim=0)  # (B, 3, 3)

        return node_rotmat, node_pos, full_transforms

    # ------------------------------------------------------------------
    # correction + application
    # ------------------------------------------------------------------

    def _corrected_full(
        self, ts: torch.Tensor, full_coefs: torch.Tensor, full_cluster_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the GNN for this batch of frames and returns:
        omega_g, delta_t_g, pivot_g (G_full, B, 3) -- the top-k RBF-blended,
        per-Gaussian correction for EVERY live foreground Gaussian -- and
        full_transforms (G_full, B, 3, 4), the pre-correction base transform
        (so the caller doesn't need to recompute it)."""
        node_rotmat, node_pos, full_transforms = self._node_state(ts, full_coefs, full_cluster_ids)
        node_rot6d = rmat_to_cont_6d(node_rotmat)  # (N, B, 6)

        B = ts.shape[0]
        ts_prev = (ts - 1).clamp(min=0)
        ts_prev2 = (ts - 2).clamp(min=0)
        _, node_pos_prev, _ = self._node_state(ts_prev, full_coefs, full_cluster_ids)
        _, node_pos_prev2, _ = self._node_state(ts_prev2, full_coefs, full_cluster_ids)
        node_vel = node_pos - node_pos_prev
        node_vel_prev = node_pos_prev - node_pos_prev2
        node_accel = node_vel - node_vel_prev

        rest_center = self.gnn.rest_center
        omega_n, delta_t_n, predicted_by_source, has_source = self.gnn(
            node_rot6d, node_pos, rest_center, node_vel, node_accel, ts
        )
        self._last_local_correction = {
            "omega": omega_n,
            "delta_t": delta_t_n,
            "node_pos": node_pos,
            "predicted_by_source": predicted_by_source,
            "has_source": has_source,
        }

        # Top-k RBF blend onto every live Gaussian.
        topk_ids = self.gnn.topk_node_id  # (G, k)
        canonical = self._canonical_means_ref  # (G, 3)
        valid_k = topk_ids >= 0  # (G, k)
        safe_ids = topk_ids.clamp_min(0)
        node_center_for_dist = self.gnn.rest_center[safe_ids]  # (G, k, 3)
        dist2 = (canonical[:, None, :] - node_center_for_dist).pow(2).sum(-1)  # (G, k)
        sigma = self.gnn.node_sigma[safe_ids]  # (G, k)
        logits = torch.where(valid_k, -dist2 / (2.0 * sigma.pow(2).clamp_min(1e-12)), torch.full_like(dist2, float("-inf")))
        # Manual NaN-safe masked softmax (NOT torch.softmax + nan_to_num) --
        # same reasoning as _LocalMessageLayer._compute_attention: a Gaussian
        # with 0 valid top-k nodes (e.g. its cluster has no nodes in this
        # graph at all) has an all -inf row, and torch.softmax's own
        # max-subtraction computes -inf - (-inf) = NaN there, which its
        # backward formula reuses -- nan_to_num only hides that in the
        # forward value, not the gradient (see
        # graph_relative_linear_attention_frame.py's module docstring for the
        # confirmed failure mode; reproduced here too before this fix).
        row_max = logits.detach().max(dim=-1, keepdim=True).values
        row_max_safe = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_logits = torch.exp(logits - row_max_safe)
        denom = exp_logits.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        weight = exp_logits / denom

        omega_g = torch.einsum("gk,gkbc->gbc", weight, omega_n[safe_ids])
        delta_t_g = torch.einsum("gk,gkbc->gbc", weight, delta_t_n[safe_ids])
        pivot_g = torch.einsum("gk,gkbc->gbc", weight, node_pos[safe_ids])
        return omega_g, delta_t_g, pivot_g, full_transforms

    def _apply_pivoted_correction(
        self,
        base_transforms: torch.Tensor,  # (G, B, 3, 4)
        omega_g: torch.Tensor,
        delta_t_g: torch.Tensor,
        pivot_g: torch.Tensor,
    ) -> torch.Tensor:
        """R_new = R_correction @ R_base; t_new = R_correction @ (t_base - pivot)
        + pivot + delta_t -- rotate around `pivot` (in world space), then
        translate, NOT the origin-pivoted `t_base + delta_t` shortcut."""
        base_rotmat, base_transl = base_transforms[..., :3], base_transforms[..., 3]
        R_correction = so3_exp_map(omega_g)
        corrected_rotmat = compose_rotation(R_correction, base_rotmat)
        corrected_transl = (
            torch.einsum("gbij,gbj->gbi", R_correction, base_transl - pivot_g) + pivot_g + delta_t_g
        )
        return torch.cat([corrected_rotmat, corrected_transl[..., None]], dim=-1)

    def compute_transforms(
        self,
        ts: torch.Tensor,
        coefs: torch.Tensor,
        cluster_ids: torch.Tensor,
        global_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the correction for the FULL live foreground array
        internally (node aggregation inherently needs every member, not just
        the queried subset) and gathers the queried rows out at the end --
        simpler than a scatter-into-subset since the full computation is
        unavoidable anyway.

        Whether `coefs`/`cluster_ids` themselves can be used directly (fresh,
        differentiable) or a cached full-array snapshot must be substituted is
        decided by comparing the QUERY size against the current live
        foreground count -- NOT by `global_indices is None`.
        flow3d/scene_model.py's SceneModel.compute_transforms(ts, inds) ALWAYS
        passes a non-None global_indices for this class (`inds` itself, or
        `torch.arange(G)` when inds is None) -- so a `global_indices is None`
        check is dead code on the real training path and was silently forcing
        EVERY forward (the main render call included, not just the anchor
        loss) onto the stale cached-snapshot branch: `coefs`/`cluster_ids` are
        detached wherever refresh_local_node_assignment is invoked (see
        run_training.py / Trainer.compute_losses), so this both broke
        gradient flow from the render loss into fg.motion_coefs and rendered
        every Gaussian from up-to-`--optim.local-gnn-refresh-every`-steps-old
        fine-basis blend weights.

        :param global_indices: (G,) long, optional -- which rows of the full
            canonical foreground array `coefs`/`cluster_ids` are, in order.
            Only consulted for a genuine SUBSET query (see below); a full-array
            query doesn't need it (the result is already in canonical order).
        returns transforms (G, B, 3, 4), G = coefs.shape[0] (the QUERY size).
        """
        if self._canonical_means_ref is None:
            raise RuntimeError(
                "refresh_local_node_assignment must be called at least once (from_scalable_motion_bases "
                "already does this) before compute_transforms."
            )
        full_G = self._canonical_means_ref.shape[0]
        is_full_query = coefs.shape[0] == full_G

        if is_full_query:
            # The standard training call: coefs/cluster_ids ARE the fresh,
            # differentiable full foreground array in canonical order (same
            # order as _canonical_means_ref / hard_member_node_id) -- get_coefs()
            # recomputes a softmax every call, so it must never be cached
            # across forward passes the way _canonical_means_ref safely is
            # (see _node_state).
            full_coefs, full_cluster_ids = coefs, cluster_ids
        else:
            full_coefs, full_cluster_ids = self._coefs_all_ref, self._cluster_ids_all_ref
            if full_coefs is None:
                raise RuntimeError(
                    "compute_transforms was called with a partial query (fewer rows than the live "
                    "foreground count) before refresh_local_node_assignment ever ran -- no full-array "
                    "coefs/cluster_ids snapshot available to fall back to."
                )
            if global_indices is None:
                raise ValueError(
                    "LocalRelativeAttentionGraphCorrectedScalableMotionBases.compute_transforms was "
                    f"called with a partial query ({coefs.shape[0]} rows, full array has {full_G}) but "
                    "no global_indices to know which rows -- cannot gather the result."
                )

        omega_g, delta_t_g, pivot_g, full_transforms = self._corrected_full(ts, full_coefs, full_cluster_ids)
        full_corrected = self._apply_pivoted_correction(full_transforms, omega_g, delta_t_g, pivot_g)

        if is_full_query:
            return full_corrected
        return full_corrected[global_indices]

    # ------------------------------------------------------------------
    # Anchor-loss inputs
    # ------------------------------------------------------------------

    def local_gnn_anchor_terms(
        self, ts: torch.Tensor, coefs_all: torch.Tensor, cluster_ids_all: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(corrected, predicted), each (M, 3), M = number of (target, frame)
        pairs in this batch with >=1 valid source -- see
        flow3d/analysis/loss.py's local_gnn_anchor_loss. `corrected` is the
        mean, over each such target's CURRENT hard-membership Gaussians, of
        that member's REAL final corrected position (the same top-k
        RBF-blended, pivoted formula compute_transforms applies, evaluated
        for real, not the node-only p_i + delta_t_i shortcut).

        :param coefs_all, cluster_ids_all: this step's FRESH full foreground
            array (e.g. model.fg.get_coefs()/model.fg.get_cluster_ids()) --
            same freshness requirement as compute_transforms's global_indices=None
            path (see _node_state); typically the exact same tensors the
            caller already fetched for this step's main compute_transforms call.
        """
        omega_g, delta_t_g, pivot_g, full_transforms = self._corrected_full(ts, coefs_all, cluster_ids_all)
        full_corrected = self._apply_pivoted_correction(full_transforms, omega_g, delta_t_g, pivot_g)
        # full_corrected[..., 3] is only the transform's TRANSLATION column,
        # not a Gaussian's actual position -- that requires applying the full
        # (rotmat, transl) transform to the CURRENT canonical mean, exactly
        # like every other posed-position computation in this file (e.g.
        # _node_state's `posed`). Using the translation column alone silently
        # drops the rotation's effect on off-pivot members entirely.
        corrected_rotmat, corrected_transl = full_corrected[..., :3], full_corrected[..., 3]
        canonical = self._canonical_means_ref  # (G, 3) -- same reference _corrected_full used internally
        corrected_pos = (
            torch.einsum("gbij,gj->gbi", corrected_rotmat, canonical) + corrected_transl
        )  # (G, B, 3)

        N, B = self.gnn.num_nodes, ts.shape[0]
        member = self.gnn.hard_member_node_id  # (G,)
        corrected_flat = corrected_pos.reshape(-1, 3)
        member_expanded = member[:, None].expand(-1, B).reshape(-1)
        batch_offset = torch.arange(B, device=member.device).repeat(member.shape[0])
        combined_index = member_expanded * B + batch_offset
        valid = member_expanded >= 0
        node_corrected = _scatter_mean(
            corrected_flat[valid], combined_index[valid], N * B
        ).view(N, B, 3)

        info = self._last_local_correction
        assert info is not None
        predicted = info["predicted_by_source"]  # (N, B, 3)
        # has_source is already False for inactive (0-member) nodes -- see
        # LocalRelativeAttentionGNN.forward -- repeated here via
        # active_node_mask explicitly so the exclusion is visible at this
        # call site too, not just relied on transitively.
        has_source = (
            info["has_source"] & self.gnn.target_mask[:, None] & self.gnn.active_node_mask[:, None]
        )  # (N, B)

        return node_corrected[has_source], predicted[has_source]

    @property
    def last_local_correction(self) -> dict[str, torch.Tensor] | None:
        """Most recent forward's node-level (omega, delta_t, node_pos,
        predicted_by_source, has_source) -- for the magnitude/smoothness
        regularizers (flow3d/analysis/loss.py) and diagnostics."""
        return self._last_local_correction

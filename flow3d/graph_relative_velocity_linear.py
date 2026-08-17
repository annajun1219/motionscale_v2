"""
flow3d/graph_relative_velocity_linear.py

flow3d/graph_coupling_relative.py(relative baseline)의 ablation 확장: node/edge
feature에 cluster의 linear velocity를 추가한다. Angular velocity와 temporal
smoothness loss는 이 파일의 범위 밖이다.

velocity 정의
--------------
프레임 t의 cluster linear velocity는 coarse translation의 프레임 간 차분이다:

    v(t) = coarse_transl[:, t] - coarse_transl[:, (t-1).clamp(min=0)]

t=0이면 prev도 frame 0이 되어 v=0이 자동으로 나온다(특수 처리 불필요). ts가
batch of frame indices일 때도 동일하게 gather로 계산된다.

어디에 추가되는가
-------------------
1. node feature: 절대 속도 v_i를 그대로 붙인다.
       node_feat = [coarse_rot_6d(6), coarse_transl(3), center(3), v_i/vel_scale(3)]
       NODE_FEAT_DIM_VEL = NODE_FEAT_DIM + 3 (12 -> 15)
2. edge feature (directed edge j->i): 상대 속도 v_j - v_i를 붙인다.
       edge_feat = [t_j-t_i(3), c_j-c_i(3), rot6d(R_i^T@R_j)(6), (v_j-v_i)/vel_scale(3)]
       EDGE_FEAT_DIM_VEL = EDGE_FEAT_DIM + 3 (12 -> 15)

velocity는 raw translation 차분(~0.05 크기)이라 rot6d(~1 크기)에 묻히므로,
생성 시점에 전체 transls 시퀀스의 frame-to-frame delta 표준편차를 vel_scale로
계산해 GNN에 buffer로 저장하고, velocity feature를 이 값으로 나눠 O(1) 크기로
맞춘다 (degenerate하면 1.0으로 fallback). vel_scale은 register_buffer라서
체크포인트 저장/복원 시 edge_index_dir과 동일하게 그대로 따라간다.

이 파일에서 바뀌지 않는 것 (재구현하지 않고 import)
------------------------------------------------------
- so3_exp_map / compose_rotation / build_edge_index_from_edges_pt / cont_6d_to_rmat
  / rmat_to_cont_6d / CENTER_DIM / ROT_DIM / TRANSL_DIM / NODE_FEAT_DIM: 기존
  flow3d/graph_coupling.py 구현을 그대로 가져다 쓴다.
- EDGE_FEAT_DIM / _build_directed_neighbor_edges: flow3d/graph_coupling_relative.py
  구현을 그대로 가져다 쓴다 (topology 구성/relative geometry dim은 이 확장으로
  바뀌지 않는다).
- graph topology(fixed, self-loop 제외 directed edge + in_degree 평균),
  residual(h_i + act(agg)) message passing 구조, coarse correction 합성 수식
  (R_new = exp(omega) @ R_coarse, t_new = t_coarse + delta_t), coarse-to-fine
  결합 수식, checkpoint save/load 포맷은 baseline relative와 동일하다.

이 파일에서 제공하는 것
------------------------
- RelativeVelocityLinearClusterGraphGNN: RelativeClusterGraphGNN과 같은 역할이지만
  node/edge feature에 linear velocity가 추가된 GNN.
- RelativeVelocityLinearGraphCorrectedScalableMotionBases: baseline
  RelativeGraphCorrectedScalableMotionBases와 동일한 API(from_scalable_motion_bases,
  has_gnn_state, init_from_state_dict, compute_transforms_coarse,
  compute_transforms, last_correction)를 가진 드롭인 대체.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

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
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "so3_exp_map",
    "compose_rotation",
    "build_edge_index_from_edges_pt",
    "RelativeVelocityLinearClusterGraphGNN",
    "RelativeVelocityLinearGraphCorrectedScalableMotionBases",
]

VEL_DIM = 3
# node_feat = [coarse_rot_6d(6), coarse_transl(3), center(3), v_i(3)]
NODE_FEAT_DIM_VEL = NODE_FEAT_DIM + VEL_DIM
# edge_feat = [t_j-t_i(3), c_j-c_i(3), rot6d(R_i^T@R_j)(6), (v_j-v_i)(3)]
EDGE_FEAT_DIM_VEL = EDGE_FEAT_DIM + VEL_DIM


def _compute_vel_scale(transls: torch.Tensor) -> float:
    """(C, T, 3) coarse translation 시퀀스 전체에서 frame-to-frame delta의
    표준편차를 계산한다. velocity feature를 O(1) 크기로 정규화하기 위한 상수.
    T < 2(delta를 낼 프레임이 없음)이거나 표준편차가 0/NaN/Inf인 degenerate
    케이스는 1.0으로 fallback한다."""
    if transls.shape[1] < 2:
        return 1.0
    deltas = transls[:, 1:] - transls[:, :-1]  # (C, T-1, 3)
    scale = deltas.std().item()
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


class _RelativeVelocityLinearMessageLayer(nn.Module):
    """flow3d/graph_coupling_relative.py의 _RelativeMessageLayer와 구조가 완전히
    동일한 fixed-topology relative message passing (residual, self-loop 제외,
    mean over non-self neighbors) -- 유일한 차이는 edge_feat 차원이
    EDGE_FEAT_DIM_VEL(velocity 포함)이라는 점뿐이다."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + EDGE_FEAT_DIM_VEL, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.ReLU()

    def forward(
        self,
        h: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat: torch.Tensor,
        in_degree_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (C, B, H) node hidden features
        edge_index_dir: (2, E) directed [src(j), dst(i)] pairs, self-loop-free
        edge_feat: (E, B, EDGE_FEAT_DIM_VEL) relative geometry+velocity feature
                   per directed edge (shared across layers within one forward call)
        in_degree_inv: (C,) 1 / (non-self in-degree), clamped to >= 1
        """
        C, B, H = h.shape
        if edge_index_dir.numel() == 0:
            return h + self.act(h.new_zeros(C, B, H))

        src, dst = edge_index_dir[0], edge_index_dir[1]
        delta_h = h[src] - h[dst]  # (E, B, H), Δh_ij = h_j - h_i
        m = self.mlp(torch.cat([delta_h, edge_feat], dim=-1))  # (E, B, H)

        agg = h.new_zeros(C, B, H)
        agg.index_add_(0, dst, m)
        agg = agg * in_degree_inv[:, None, None]  # mean over non-self neighbors

        return h + self.act(agg)


class RelativeVelocityLinearClusterGraphGNN(nn.Module):
    """RelativeClusterGraphGNN과 같은 API(node feature -> (omega, delta_t))를
    갖는 relative-message GNN에, cluster의 linear velocity를 node/edge feature로
    추가한 변형.

    Graph topology(directed neighbor edge, self-loop 제외)는 생성 시점에 고정돼
    학습 중 바뀌지 않는다. head는 0으로 초기화되므로 학습 시작 시점에는 correction이
    정확히 0이다(baseline과 동일한 zero-init 등가성 -- 입력 feature 차원이
    늘어도 이 성질은 그대로 유지된다).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        vel_scale: float = 1.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        edge_index_dir = _build_directed_neighbor_edges(edge_index, num_clusters)
        self.register_buffer("edge_index_dir", edge_index_dir)

        in_degree = torch.zeros(num_clusters, device=edge_index_dir.device)
        if edge_index_dir.numel() > 0:
            dst = edge_index_dir[1]
            in_degree.index_add_(0, dst, in_degree.new_ones(dst.shape[0]))
        self.register_buffer("in_degree_inv", 1.0 / in_degree.clamp_min(1.0))

        if not math.isfinite(vel_scale) or vel_scale <= 0.0:
            vel_scale = 1.0
        self.register_buffer("vel_scale", torch.tensor(float(vel_scale)))

        self.num_clusters = num_clusters
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM_VEL, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [_RelativeVelocityLinearMessageLayer(hidden_dim) for _ in range(num_layers)]
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
        를 만든다. 한 forward call 안에서 한 번만 계산되어 모든 layer에 재사용된다.

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
        # 바꾸지 않는다 -- 이 확장은 velocity feature를 "추가"할 뿐이다)
        node_feat = torch.cat(
            [coarse_rot_6d, coarse_transl, center_feat, vel_scaled], dim=-1
        )  # (C, B, 15)

        h = self.encoder(node_feat)
        edge_feat = self._compute_edge_features(coarse_rot_6d, coarse_transl, centers, vel_scaled)
        for layer in self.layers:
            h = layer(h, self.edge_index_dir, edge_feat, self.in_degree_inv)

        correction = self.head(h)  # (C, B, 6)
        omega, delta_t = correction.split([3, 3], dim=-1)
        return omega, delta_t


class RelativeVelocityLinearGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + relative-message-with-velocity graph-aware coarse
    transform correction. baseline RelativeGraphCorrectedScalableMotionBases와
    API가 동일한 드롭인 대체이며, 차이는 내부 GNN이
    RelativeVelocityLinearClusterGraphGNN이라는 점과, cluster linear velocity를
    계산해 그 GNN에 넘긴다는 점뿐이다.
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
    ):
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        vel_scale = _compute_vel_scale(transls)
        self.gnn = RelativeVelocityLinearClusterGraphGNN(
            edge_index=edge_index,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
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
    ) -> "RelativeVelocityLinearGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 relative+velocity
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
        )

    @staticmethod
    def has_gnn_state(state_dict: dict, prefix: str) -> bool:
        """state_dict의 motion_bases가
        RelativeVelocityLinearGraphCorrectedScalableMotionBases로 저장된 것인지
        확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은 형태여야 한다
        (params.가 아니라)."""
        gnn_prefix = f"{prefix}gnn."
        return any(key.startswith(gnn_prefix) for key in state_dict)

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "RelativeVelocityLinearGraphCorrectedScalableMotionBases":
        """체크포인트만으로
        RelativeVelocityLinearGraphCorrectedScalableMotionBases를 통째로
        복원한다. edge_index/hidden_dim/num_layers는 저장된 텐서의 shape에서
        읽어온다. directed edge/in-degree/vel_scale 버퍼는 __init__ 시점에
        placeholder로 재계산되고, 아래 load_state_dict가 저장된 실제 값으로
        strict하게 덮어쓴다 (baseline relative와 동일한 패턴)."""
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

        saved_edge_index_dir = state_dict[f"{gnn_prefix}edge_index_dir"]
        graph_bases = cls.from_scalable_motion_bases(
            base,
            edge_index=saved_edge_index_dir,
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
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
        message passing에 linear velocity feature가 추가로 들어간다는 점뿐이다).

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
        baseline RelativeGraphCorrectedScalableMotionBases.compute_transforms와
        완전히 동일하며, coarse rotation/translation만 velocity-aware
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

        # --- graph-aware coarse transform (baseline과 다른 부분: velocity feature) ---
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
    #   1. zero-init equivalence: step 0 == plain ScalableMotionBases exactly.
    #   2. zero-init: |omega|_max == 0, |delta_t|_max == 0.
    #   3. grad flow: head receives nonzero gradient from a downstream loss.
    #   4. velocity feature is actually wired in: node/edge feature dims, and
    #      nonzero velocity for a moving (per-frame-random) sequence.
    #   5. save -> init_from_state_dict round-trip reproduces the same output.
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
    gnn_hidden_dim, gnn_num_layers = 32, 2
    graph_bases = RelativeVelocityLinearGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline, edge_index=edge_index, gnn_hidden_dim=gnn_hidden_dim, gnn_num_layers=gnn_num_layers
    )

    ts = torch.arange(num_frames)
    cluster_ids = torch.randint(0, num_clusters, (num_fg,))
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    # --- 1. zero-init equivalence ---
    ref_coarse = baseline.compute_transforms_coarse(ts, cluster_ids)
    out_coarse = graph_bases.compute_transforms_coarse(ts, cluster_ids)
    max_diff_coarse = (ref_coarse - out_coarse).abs().max().item()
    print(f"[coarse]  max |baseline - velocity_linear_graph_corrected(zero-init)| = {max_diff_coarse:.3e}")
    assert max_diff_coarse < 1e-5, "zero-init correction should reproduce the baseline coarse transform exactly"

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"[full]    max |baseline - velocity_linear_graph_corrected(zero-init)| = {max_diff:.3e}")
    assert max_diff < 1e-5, "zero-init correction should reproduce the baseline transform exactly"

    # --- 2. zero-init correction is exactly zero ---
    omega = graph_bases.last_correction["omega"]
    delta_t = graph_bases.last_correction["delta_t"]
    print(f"[zero-init] |omega|_max={omega.abs().max().item():.3e}  |delta_t|_max={delta_t.abs().max().item():.3e}")
    assert omega.abs().max().item() == 0.0
    assert delta_t.abs().max().item() == 0.0

    # --- 3. grad flow check: GNN head must receive a nonzero gradient from a downstream loss ---
    graph_bases.zero_grad()
    out2 = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    loss = out2.pow(2).mean()
    loss.backward()

    head_grad_norm = graph_bases.gnn.head.weight.grad.norm().item()
    encoder_grad_norm = graph_bases.gnn.encoder[0].weight.grad.norm().item()
    print(f"grad norm: head.weight={head_grad_norm:.3e}  encoder.weight={encoder_grad_norm:.3e}")
    assert head_grad_norm > 0.0, "GNN head should receive nonzero gradient -- it will actually train"
    # Encoder/message-passing layers are (correctly) gradient-starved at init: the
    # zero-initialized head multiplies their contribution by zero in dL/dW. This
    # is expected -- they start learning once the head moves off zero.

    # --- 4. velocity feature is actually wired in ---
    assert graph_bases.gnn.encoder[0].in_features == NODE_FEAT_DIM_VEL == 15, (
        "node encoder input dim should be NODE_FEAT_DIM_VEL (12 + 3 velocity)"
    )
    assert graph_bases.gnn.layers[0].mlp[0].in_features == gnn_hidden_dim + EDGE_FEAT_DIM_VEL, (
        "message MLP input dim should be hidden_dim + EDGE_FEAT_DIM_VEL (12 + 3 velocity)"
    )
    print(
        f"[feature dims] node={NODE_FEAT_DIM_VEL} (base {NODE_FEAT_DIM} + {VEL_DIM}), "
        f"edge={EDGE_FEAT_DIM_VEL} (base {EDGE_FEAT_DIM} + {VEL_DIM})"
    )

    captured: dict[str, torch.Tensor] = {}

    def _capture_coarse_vel(module, args):
        captured["coarse_vel"] = args[3]

    handle = graph_bases.gnn.register_forward_pre_hook(_capture_coarse_vel)
    _ = graph_bases.compute_transforms_coarse(ts, cluster_ids)
    handle.remove()

    coarse_vel = captured["coarse_vel"]
    frame0_max = coarse_vel[:, 0].abs().max().item()
    other_frames_max = coarse_vel[:, 1:].abs().max().item()
    print(f"[velocity] shape={tuple(coarse_vel.shape)}  frame0 max|v|={frame0_max:.3e}  frames>=1 max|v|={other_frames_max:.3e}")
    assert coarse_vel.shape == (num_clusters, num_frames, VEL_DIM)
    assert frame0_max == 0.0, "t=0 velocity must be exactly zero (prev clamps to frame 0)"
    assert other_frames_max > 0.0, "velocity should be nonzero for a moving (per-frame-random) sequence"

    print(f"[vel_scale] {graph_bases.gnn.vel_scale.item():.3e}")

    # --- 5. save -> init_from_state_dict round-trip ---
    full_state_dict = {f"motion_bases.{k}": v for k, v in graph_bases.state_dict().items()}
    restored = RelativeVelocityLinearGraphCorrectedScalableMotionBases.init_from_state_dict(
        full_state_dict, prefix="motion_bases."
    )
    out_restored = restored.compute_transforms(ts, coefs, cluster_ids)
    out_original = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff_roundtrip = (out_restored - out_original).abs().max().item()
    vel_scale_diff = (restored.gnn.vel_scale - graph_bases.gnn.vel_scale).abs().item()
    print(f"[round-trip] max |original - restored| = {max_diff_roundtrip:.3e}  vel_scale diff = {vel_scale_diff:.3e}")
    assert max_diff_roundtrip < 1e-6, "save -> init_from_state_dict round-trip should reproduce identical output"
    assert vel_scale_diff < 1e-6, "vel_scale buffer should round-trip exactly"

    print("OK")

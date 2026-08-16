"""
flow3d/graph_coupling_relative.py

flow3d/graph_coupling.py(baseline)의 ablation 변형: message passing에서 이웃
cluster와의 relative motion/geometry feature를 명시적으로 쓴다.

baseline과의 차이는 오직 "GNN이 fixed graph의 이웃 정보를 어떻게 message로
만드는가" 뿐이다. baseline은 이웃의 (변환되지 않은 절대) hidden feature를 그대로
평균 냈지만, 이 변형은 directed edge j->i마다

    - relative translation: t_j - t_i        (3)
    - relative center:      c_j - c_i        (3)
    - relative rotation:    R_i^T @ R_j (6D) (6)
    - hidden-state 차이:     h_j - h_i        (hidden_dim)

를 만들어 MLP에 통과시킨 뒤, node i로 들어오는 (self 제외) message만 mean
aggregate해서 residual로 더한다. node feature, coarse correction 방식
(R_new = exp(omega) @ R_coarse, t_new = t_coarse + delta_t), graph topology,
edges.pt 파싱, fine motion 결합 수식, checkpoint save/load 포맷은 baseline과
동일하다 -- graph_coupling.py는 이 파일에서 한 줄도 바뀌지 않는다.

이 파일에서 제공하는 것
------------------------
- RelativeClusterGraphGNN: baseline ClusterGraphGNN과 같은 시그니처의
  node feature -> (omega, delta_t) GNN. message passing만 relative.
- RelativeGraphCorrectedScalableMotionBases: baseline
  GraphCorrectedScalableMotionBases와 동일한 API(from_scalable_motion_bases,
  has_gnn_state, init_from_state_dict, compute_transforms_coarse,
  compute_transforms, last_correction)를 가진 드롭인 대체.
- so3_exp_map / compose_rotation / build_edge_index_from_edges_pt: baseline
  구현을 그대로 재사용(re-export)한다 -- 이 파일에서 재구현하지 않는다.
"""

from __future__ import annotations

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
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat, rmat_to_cont_6d

__all__ = [
    "so3_exp_map",
    "compose_rotation",
    "build_edge_index_from_edges_pt",
    "RelativeClusterGraphGNN",
    "RelativeGraphCorrectedScalableMotionBases",
]

# edge feature e_ij = [t_j - t_i (3), c_j - c_i (3), rot6d(R_i^T @ R_j) (6)]
EDGE_FEAT_DIM = TRANSL_DIM + CENTER_DIM + ROT_DIM


def _build_directed_neighbor_edges(
    edge_index: torch.Tensor, num_clusters: int
) -> torch.Tensor:
    """(2, E) undirected cluster-id pairs -> (2, E_dir) directed edges (both
    directions, self-loops excluded, deduplicated).

    row0 = src (j, sender), row1 = dst (i, receiver) so that column e
    represents the message direction j -> i.
    """
    if edge_index.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)

    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]

    all_src = torch.cat([src, dst])
    all_dst = torch.cat([dst, src])
    directed = torch.stack([all_src, all_dst], dim=0)
    directed = torch.unique(directed, dim=1)
    return directed


class _RelativeMessageLayer(nn.Module):
    """Fixed-topology relative message passing, 1 layer, residual.

    baseline(_MeanAggLayer)과 달리 이웃의 절대 hidden feature를 그대로 쓰지
    않고, sender-receiver 쌍마다 (hidden 차이, relative geometry)로 만든
    message를 mean aggregate한다:

        m_{j->i} = MLP([h_j - h_i, e_ij]),   e_ij = relative geometry(j, i)
        agg_i    = mean_{j in N(i)} m_{j->i}
        h_i'     = h_i + act(agg_i)

    Self-loop을 쓰지 않는 이유: h_i - h_i = 0이라 self-message는 e_ii(자기 자신과의
    relative geometry, 즉 0벡터)만 인코딩하는 무의미한 상수 항이 되기 때문이다.
    baseline과 달리 이 레이어의 이웃 집합(N(i))에는 self가 포함되지 않고, 자기
    정보는 residual path(h_i)만으로 보존한다.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + EDGE_FEAT_DIM, hidden_dim),
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
        edge_feat: (E, B, 12) relative geometry feature per directed edge
                   (shared across layers within one forward call)
        in_degree_inv: (C,) 1 / (non-self in-degree), clamped to >= 1
        """
        C, B, H = h.shape
        if edge_index_dir.numel() == 0:
            # No edges at all (isolated graph): nothing to aggregate, pure residual.
            return h + self.act(h.new_zeros(C, B, H))

        src, dst = edge_index_dir[0], edge_index_dir[1]
        delta_h = h[src] - h[dst]  # (E, B, H), Δh_ij = h_j - h_i
        m = self.mlp(torch.cat([delta_h, edge_feat], dim=-1))  # (E, B, H)

        agg = h.new_zeros(C, B, H)
        agg.index_add_(0, dst, m)
        agg = agg * in_degree_inv[:, None, None]  # mean over non-self neighbors

        return h + self.act(agg)


class RelativeClusterGraphGNN(nn.Module):
    """baseline ClusterGraphGNN과 같은 API(node feature -> (omega, delta_t))를
    갖는, relative-message 기반 GNN.

    Graph topology(directed neighbor edge, self-loop 제외)는 생성 시점에 고정돼
    학습 중 바뀌지 않는다. head는 0으로 초기화되므로 학습 시작 시점에는 correction이
    정확히 0이다(baseline과 동일한 zero-init 등가성).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        edge_index_dir = _build_directed_neighbor_edges(edge_index, num_clusters)
        self.register_buffer("edge_index_dir", edge_index_dir)

        in_degree = torch.zeros(num_clusters)
        if edge_index_dir.numel() > 0:
            dst = edge_index_dir[1]
            in_degree.index_add_(0, dst, torch.ones(dst.shape[0]))
        self.register_buffer("in_degree_inv", 1.0 / in_degree.clamp_min(1.0))

        self.num_clusters = num_clusters
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [_RelativeMessageLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, ROT_DIM // 2 + TRANSL_DIM)  # (omega(3), delta_t(3))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_edge_features(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """directed edge j->i마다 e_ij = [t_j - t_i, c_j - c_i, rot6d(R_i^T @ R_j)]
        를 만든다. 한 forward call 안에서 한 번만 계산되어 모든 layer에 재사용된다
        (geometric feature는 frame의 coarse transform/center에서만 나오고 layer의
        hidden state와 무관하기 때문).
        """
        B = coarse_transl.shape[1]
        if self.edge_index_dir.numel() == 0:
            return coarse_rot_6d.new_zeros(0, B, EDGE_FEAT_DIM)

        src, dst = self.edge_index_dir[0], self.edge_index_dir[1]

        rel_transl = coarse_transl[src] - coarse_transl[dst]  # (E, B, 3), t_j - t_i
        rel_center = centers[src] - centers[dst]  # (E, 3), c_j - c_i
        rel_center = rel_center[:, None, :].expand(-1, B, -1)  # (E, B, 3)

        coarse_rotmats = cont_6d_to_rmat(coarse_rot_6d)  # (C, B, 3, 3)
        R_i = coarse_rotmats[dst]  # (E, B, 3, 3)
        R_j = coarse_rotmats[src]  # (E, B, 3, 3)
        R_rel = torch.matmul(R_i.transpose(-1, -2), R_j)  # R_i^T @ R_j
        rel_rot6d = rmat_to_cont_6d(R_rel)  # (E, B, 6)

        return torch.cat([rel_transl, rel_center, rel_rot6d], dim=-1)  # (E, B, 12)

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        coarse_rot_6d: (C, B, 6)
        coarse_transl: (C, B, 3)
        centers: (C, 3) canonical cluster centers
        returns: omega (C, B, 3), delta_t (C, B, 3)
        """
        if coarse_rot_6d.shape[0] != self.num_clusters:
            raise ValueError(
                f"coarse_rot_6d has {coarse_rot_6d.shape[0]} clusters, "
                f"expected {self.num_clusters}"
            )

        C, B, _ = coarse_rot_6d.shape
        center_feat = centers[:, None, :].expand(C, B, CENTER_DIM)
        # absolute node state (baseline과 동일 -- relative로 바꾸지 않는다)
        node_feat = torch.cat([coarse_rot_6d, coarse_transl, center_feat], dim=-1)  # (C, B, 12)

        h = self.encoder(node_feat)
        edge_feat = self._compute_edge_features(coarse_rot_6d, coarse_transl, centers)
        for layer in self.layers:
            h = layer(h, self.edge_index_dir, edge_feat, self.in_degree_inv)

        correction = self.head(h)  # (C, B, 6)
        omega, delta_t = correction.split([3, 3], dim=-1)
        return omega, delta_t


class RelativeGraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + relative-message graph-aware coarse transform
    correction. baseline GraphCorrectedScalableMotionBases와 API가 동일한
    드롭인 대체이며, 유일한 차이는 내부 GNN이 RelativeClusterGraphGNN이라는 점.
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
        self.gnn = RelativeClusterGraphGNN(
            edge_index=edge_index,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
        )
        self._last_correction: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 2,
    ) -> "RelativeGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 relative graph-corrected
        버전을 만든다. coarse/fine motion 파라미터 값은 그대로 복사되고, GNN
        correction만 새로 추가된다 (correction은 0으로 초기화됨)."""
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
        """state_dict의 motion_bases가 RelativeGraphCorrectedScalableMotionBases로
        저장된 것인지 확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은
        형태여야 한다 (params.가 아니라)."""
        gnn_prefix = f"{prefix}gnn."
        return any(key.startswith(gnn_prefix) for key in state_dict)

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "RelativeGraphCorrectedScalableMotionBases":
        """체크포인트만으로 RelativeGraphCorrectedScalableMotionBases를 통째로
        복원한다. edge_index/hidden_dim/num_layers는 저장된 텐서의 shape에서
        읽어온다. directed edge/in-degree 버퍼는 num_clusters만으로 placeholder를
        만들 수 있고, 아래 load_state_dict가 실제 값으로 덮어쓴다."""
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

        회전은 compose(exp(omega) @ R_coarse)이고, translation만 덧셈이다.
        (baseline과 완전히 동일한 compose 수식 -- 바뀐 건 omega/delta_t를
        만드는 message passing 뿐이다.)
        """
        coarse_rot_6d = self.params["rots"][:, ts]  # (C, B, 6)
        coarse_transl = self.params["transls"][:, ts]  # (C, B, 3)
        centers = self.params["centers"]  # (C, 3)

        omega, delta_t = self.gnn(coarse_rot_6d, coarse_transl, centers)  # (C, B, 3) x2
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
        baseline GraphCorrectedScalableMotionBases.compute_transforms와 완전히
        동일하며, coarse rotation/translation만 relative-message correction이
        compose된 값으로 바뀐다. fine motion 코드는 한 글자도 바뀌지 않는다.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        assert coefs.shape[0] == cluster_ids.shape[0]
        G, F_dim = coefs.shape
        C, _, B, _ = self.params["fine_transls"][:, :, ts].shape

        # --- graph-aware coarse transform (baseline과 다른 부분: message가 relative) ---
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
    #   1. zero-init check: at step 0, RelativeGraphCorrected == plain ScalableMotionBases exactly.
    #   2. GNN head params actually receive nonzero gradient (they will train).
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
    graph_bases = RelativeGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline, edge_index=edge_index, gnn_hidden_dim=32, gnn_num_layers=2
    )

    ts = torch.arange(num_frames)
    cluster_ids = torch.randint(0, num_clusters, (num_fg,))
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    # --- 1. zero-init equivalence ---
    ref_coarse = baseline.compute_transforms_coarse(ts, cluster_ids)
    out_coarse = graph_bases.compute_transforms_coarse(ts, cluster_ids)
    max_diff_coarse = (ref_coarse - out_coarse).abs().max().item()
    print(f"[coarse]  max |baseline - relative_graph_corrected(zero-init)| = {max_diff_coarse:.3e}")
    assert max_diff_coarse < 1e-5, "zero-init correction should reproduce the baseline coarse transform exactly"

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"[full]    max |baseline - relative_graph_corrected(zero-init)| = {max_diff:.3e}")
    assert max_diff < 1e-5, "zero-init correction should reproduce the baseline transform exactly"

    omega = graph_bases.last_correction["omega"]
    delta_t = graph_bases.last_correction["delta_t"]
    print(f"[zero-init] |omega|_max={omega.abs().max().item():.3e}  |delta_t|_max={delta_t.abs().max().item():.3e}")
    assert omega.abs().max().item() == 0.0
    assert delta_t.abs().max().item() == 0.0

    # --- 2. grad flow check: GNN head must receive a nonzero gradient from a downstream loss ---
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

    print("OK")

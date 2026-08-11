"""
flow3d/graph_coupling.py

Per-cluster coarse transform을 그래프 이웃과 결합해 residual 보정하는 GNN.

목적: 손 cluster가 팔 등 이웃 cluster와 조율돼 움직이도록 해서, coarse motion이
다른 신체 부위(identity)로 잘못 넘어가는 것(예: 오른손으로의 identity swap)을
억제한다. fine(local) motion과 ARAP rigidity는 이 모듈과 무관하게 그대로 둔다.

파이프라인 (프레임 t마다)
--------------------------
1. cluster 상태를 feature로 인코딩: [rot6d(coarse), transl(coarse), canonical center]
2. 고정된 cluster adjacency graph 위에서 message passing(mean aggregation,
   residual, 1~2 layer): 각 cluster가 이웃 feature를 모아 자기 feature에 섞는다.
3. head가 보정량 (omega, delta_t)을 출력한다. head는 0으로 초기화되므로 학습
   시작 시점에는 보정이 정확히 0이다.
4. compose: R_new = exp(omega) @ R_coarse (회전은 합성), t_new = t_coarse + delta_t
   (이동만 덧셈).
5. 보정된 coarse rotation/translation을 기존 자리에 넣고, 이후 fine motion과의
   결합은 ScalableMotionBases.compute_transforms와 완전히 동일한 수식을 쓴다.

이 파일에서 제공하는 것
------------------------
- ClusterGraphGNN: node feature(coarse rot6d + transl + canonical center) ->
  (omega, delta_t) correction을 만드는 작은 mean-aggregation message-passing GNN.
  Graph topology는 생성 시점에 고정된 (C, C) adjacency로 저장된다.
- so3_exp_map / compose_rotation: so(3) exponential map과 회전 합성 유틸.
- GraphCorrectedScalableMotionBases: ScalableMotionBases를 상속해서
  compute_transforms_coarse / compute_transforms 두 곳에서만 correction을
  끼워 넣는다. fine motion 결합 수식은 부모 구현과 동일하다.
- build_edge_index_from_edges_pt: flow3d/analysis/build_cluster_graph.py가 만든
  edges.pt (Trainer._load_cluster_graph_file이 읽는 것과 동일한 포맷)로부터
  GNN의 edge_index를 만드는 헬퍼.
"""

from __future__ import annotations

from pathlib import Path

import roma
import torch
import torch.nn as nn

from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat

ROT_DIM = 6
TRANSL_DIM = 3
CENTER_DIM = 3
NODE_FEAT_DIM = ROT_DIM + TRANSL_DIM + CENTER_DIM


def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    """so(3) exponential map: axis-angle (..., 3) -> rotation matrix (..., 3, 3)."""
    return roma.rotvec_to_rotmat(omega)


def compose_rotation(R_correction: torch.Tensor, R_coarse: torch.Tensor) -> torch.Tensor:
    """R_new = R_correction @ R_coarse (rotation is composed, never added)."""
    return torch.einsum("...ij,...jk->...ik", R_correction, R_coarse)


def build_edge_index_from_edges_pt(path: str | Path, num_clusters: int) -> torch.Tensor:
    """flow3d/analysis/build_cluster_graph.py가 저장한 edges.pt를 읽어 (2, E)
    edge_index를 만든다 (Trainer._load_cluster_graph_file과 동일한 포맷 가정).

    edges.pt는 {"edges_kept": [{"cluster_a": a, "cluster_b": b, ...}, ...],
    "cluster_ids": [...]} 형태이며, a, b는 model.fg.get_cluster_ids() 기준
    cluster index(0..C-1)다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"edges.pt not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    edges_kept = payload["edges_kept"]
    if not edges_kept:
        raise ValueError(f"{path} has no edges_kept -- cannot build a cluster graph.")

    src = torch.tensor([int(e["cluster_a"]) for e in edges_kept], dtype=torch.long)
    dst = torch.tensor([int(e["cluster_b"]) for e in edges_kept], dtype=torch.long)

    if src.numel() and (int(src.max()) >= num_clusters or int(dst.max()) >= num_clusters):
        raise ValueError(
            f"{path} references cluster ids up to "
            f"{max(int(src.max()), int(dst.max()))}, but num_clusters={num_clusters}. "
            "The cluster set must be unchanged since edges.pt was built "
            "(pass --optim.no-enable-bases-control)."
        )

    return torch.stack([src, dst], dim=0)


def _build_adjacency_with_self_loops(edge_index: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """(2, E) cluster-id pairs -> symmetric (C, C) bool adjacency with self-loops."""
    adj = torch.zeros(num_clusters, num_clusters, dtype=torch.bool)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = True
        adj[dst, src] = True
    adj.fill_diagonal_(True)
    return adj


class _MeanAggLayer(nn.Module):
    """Fixed-topology mean-aggregation message passing, 1 layer, residual.

    각 cluster는 (self-loop을 포함한) 고정 이웃 집합의 hidden feature 평균을
    자기 feature에 residual로 더한다. 이웃이 누구인지(topology)는 학습 중
    절대 바뀌지 않는다.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        h: (C, B, H) node hidden features
        adj_norm: (C, C) row-normalized adjacency (mean aggregation weights)
        """
        msg = self.lin(h)  # (C, B, H)
        agg = torch.einsum("ij,jbh->ibh", adj_norm, msg)  # mean over neighbors(+self)
        return h + self.act(agg)


class ClusterGraphGNN(nn.Module):
    """cluster adjacency graph 위에서 coarse rotation/translation에 더할
    (omega, delta_t) correction만 출력하는 GNN.

    Graph topology는 생성 시점에 고정된 (C, C) adjacency(self-loop 포함)로
    저장되고 학습 중 바뀌지 않는다. head(decoder)를 0으로 초기화하므로 학습
    시작 시점에는 correction이 정확히 0이 되어, graph-uninformed 기존 방식과
    동일하게 동작한다.
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

        adj = _build_adjacency_with_self_loops(edge_index, num_clusters)  # (C, C) bool
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1).float()
        adj_norm = adj.float() / degree
        self.register_buffer("adj_norm", adj_norm)

        self.num_clusters = num_clusters
        self.encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [_MeanAggLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, ROT_DIM // 2 + TRANSL_DIM)  # (omega(3), delta_t(3))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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
        node_feat = torch.cat([coarse_rot_6d, coarse_transl, center_feat], dim=-1)  # (C, B, 12)

        h = self.encoder(node_feat)
        for layer in self.layers:
            h = layer(h, self.adj_norm)

        correction = self.head(h)  # (C, B, 6)
        omega, delta_t = correction.split([3, 3], dim=-1)
        return omega, delta_t


class GraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + graph-aware coarse transform correction.

    기존 coarse rotation/translation(self.params["rots"]/["transls"])은 그대로
    두고, ClusterGraphGNN이 예측한 (omega, delta_t) correction을 compose한
    coarse transform으로 이후 계산을 진행한다. fine(local) motion 결합은 부모
    클래스(ScalableMotionBases.compute_transforms)와 완전히 동일한 수식을 쓴다.
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
        self.gnn = ClusterGraphGNN(
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
    ) -> "GraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 graph-corrected 버전을 만든다.
        coarse/fine motion 파라미터 값은 그대로 복사되고, GNN correction만 새로
        추가된다 (correction은 0으로 초기화됨 -- 기존 동작과 동일하게 시작)."""
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
        """state_dict의 motion_bases가 GraphCorrectedScalableMotionBases로 저장된
        것인지 확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은 형태여야
        한다 (params.가 아니라)."""
        gnn_prefix = f"{prefix}gnn."
        return any(key.startswith(gnn_prefix) for key in state_dict)

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "GraphCorrectedScalableMotionBases":
        """체크포인트만으로 GraphCorrectedScalableMotionBases를 통째로 복원한다.
        edge_index/hidden_dim/num_layers 인자를 따로 넘길 필요가 없다 -- 전부
        저장된 텐서의 shape에서 읽어온다. adjacency(adj_norm)는 (C, C)라서
        edge 개수와 무관하게 num_clusters만으로 placeholder를 만들 수 있고,
        아래 load_state_dict가 실제 값으로 덮어쓴다."""
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

        # adj_norm depends only on num_clusters, not edge count, so an empty
        # placeholder edge_index is safe -- load_state_dict below overwrites
        # its contents with the real (saved) adjacency.
        placeholder_edge_index = torch.empty((2, 0), dtype=torch.long)
        graph_bases = cls.from_scalable_motion_bases(
            base,
            edge_index=placeholder_edge_index,
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
        """coarse-to-fine 결합 수식은 ScalableMotionBases.compute_transforms와
        동일하며, coarse rotation/translation만 graph correction이 compose된
        값으로 바뀐다. fine motion 코드는 한 글자도 바뀌지 않는다.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        assert coefs.shape[0] == cluster_ids.shape[0]
        G, F_dim = coefs.shape
        C, _, B, _ = self.params["fine_transls"][:, :, ts].shape

        # --- graph-aware coarse transform (기존 코드와 다른 부분) ---
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
    #   1. zero-init check: at step 0, GraphCorrected == plain ScalableMotionBases exactly.
    #   2. GNN params actually receive nonzero gradient (they will train).
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
    graph_bases = GraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline, edge_index=edge_index, gnn_hidden_dim=32, gnn_num_layers=2
    )

    ts = torch.arange(num_frames)
    cluster_ids = torch.randint(0, num_clusters, (num_fg,))
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    # --- 1. zero-init equivalence ---
    ref_coarse = baseline.compute_transforms_coarse(ts, cluster_ids)
    out_coarse = graph_bases.compute_transforms_coarse(ts, cluster_ids)
    max_diff_coarse = (ref_coarse - out_coarse).abs().max().item()
    print(f"[coarse]  max |baseline - graph_corrected(zero-init)| = {max_diff_coarse:.3e}")
    assert max_diff_coarse < 1e-5, "zero-init correction should reproduce the baseline coarse transform exactly"

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"[full]    max |baseline - graph_corrected(zero-init)| = {max_diff:.3e}")
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

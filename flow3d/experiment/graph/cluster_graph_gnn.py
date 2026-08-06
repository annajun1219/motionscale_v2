"""
flow3d/experiment/graph/cluster_graph_gnn.py

Graph-aware cluster global transform 실험용 모듈.

기존 파이프라인 (ScalableMotionBases, flow3d/params.py):
    cluster별 global(coarse) transform + 기존 local(fine) motion

이 실험에서 발전시키는 부분:
    graph 관계를 반영한 cluster별 global transform + 기존 local motion

    **graph-aware global transform + 기존 local transform = Gaussian 최종 transform**

파이프라인
----------
1. Cluster의 기존 coarse rotation(6D)/translation(3D) 값을 그대로 유지하고,
   둘을 이어 붙여 하나의 node feature로 만든다. (ClusterGraphGNN 입력)
2. cluster adjacency graph(canonical space에서 인접한 cluster 쌍, 예:
   flow3d/analysis/cluster_pairs.py 가 만드는 fixed_boundary_indices.pt)의
   topology는 생성 시점에 한 번 고정된다. 이 topology 자체는 학습 중 바뀌지
   않고, 대신 각 edge를 얼마나 신뢰할지(attention weight)를 프레임마다 그
   시점의 cluster motion feature로부터 다시 예측한다 (GAT 스타일 message
   passing) -- "누가 이웃인가"는 고정, "지금 그 이웃을 얼마나 참고할까"만 가변.
3. GNN은 최종 transform을 직접 출력하지 않고, 기존 coarse rotation/translation에
   더할 "보정량(correction)"만 출력한다. 마지막 layer를 0으로 초기화해서 학습
   초기에는 correction이 0이 되도록 하고(=기존 방식과 동일하게 시작),
   correction_regularization_loss()로 correction 크기에 loss를 걸어
   GNN이 완전히 새로운 transform을 만들지 않고 "보정"만 하도록 유도한다.
   같은 이유로 attention_entropy_regularization_loss()는 attention이
   (고정된) 이웃 중 한둘로 완전히 쏠리지 않도록 정규화한다.
4. 이 보정량을 기존 coarse rotation/translation에 더해 graph-aware coarse
   transform을 만든다.
5. 이후 fine(local) motion을 얹는 계산은 ScalableMotionBases.compute_transforms
   와 완전히 동일한 수식을 사용한다 (coarse-to-fine 결합 로직은 건드리지 않음).

이 파일에서 제공하는 것
------------------------
- ClusterGraphGNN: node feature(coarse rot+transl) -> correction 을 만드는
  작은 graph attention 네트워크 (canonical topology 고정, edge attention은
  시간-가변; 외부 그래프 라이브러리 의존 없이 dense masked attention으로 구현).
- GraphCorrectedScalableMotionBases: ScalableMotionBases를 상속해서
  compute_transforms_coarse / compute_transforms 두 곳에서만 correction을
  끼워 넣은 버전. 나머지(fine motion 결합 등)는 부모 구현과 동일한 수식.
- build_edge_index_from_pairs / build_edge_index_from_boundary_file:
  cluster pair 목록 (또는 cluster_pairs.py의 fixed_boundary_indices.pt) 로부터
  GNN의 edge_index를 만드는 헬퍼.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat

ROT_DIM = 6
TRANSL_DIM = 3
NODE_FEAT_DIM = ROT_DIM + TRANSL_DIM


def _make_bidirectional(edge_index: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """(2, E) -> 양방향 + self-loop 포함, 중복 제거된 (2, E') edge_index."""
    src = torch.cat([edge_index[0], edge_index[1], torch.arange(num_clusters)])
    dst = torch.cat([edge_index[1], edge_index[0], torch.arange(num_clusters)])
    pairs = torch.stack([src, dst], dim=0)
    unique_pairs = torch.unique(pairs, dim=1)
    return unique_pairs


def build_edge_index_from_pairs(
    pairs: Sequence[tuple[int, int]],
    num_clusters: int,
) -> torch.Tensor:
    """cluster id pair 목록 [(a, b), ...] -> (2, E) edge_index.

    cluster id는 ScalableMotionBases의 cluster 차원 index(0..C-1)와
    동일한 값이어야 한다 (model.fg.get_cluster_ids()가 이미 이 규약을 따른다).
    """
    if not pairs:
        raise ValueError("pairs must contain at least one cluster pair.")

    src = torch.tensor([int(a) for a, _ in pairs], dtype=torch.long)
    dst = torch.tensor([int(b) for _, b in pairs], dtype=torch.long)

    if src.numel() and (src.max() >= num_clusters or dst.max() >= num_clusters):
        raise ValueError(
            f"Pair index exceeds num_clusters={num_clusters}: "
            f"max(src)={int(src.max())}, max(dst)={int(dst.max())}"
        )

    edge_index = torch.stack([src, dst], dim=0)
    return _make_bidirectional(edge_index, num_clusters)


def _normalize_pair_entries(raw: Any) -> list[Mapping[str, Any]]:
    if isinstance(raw, Mapping) and "pairs" in raw and "cluster_a" not in raw:
        raw = raw["pairs"]

    if isinstance(raw, Mapping):
        entries = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = list(raw)
    else:
        raise TypeError(
            f"Unsupported boundary-relation file content: {type(raw).__name__}"
        )

    for idx, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise TypeError(f"Pair entry {idx} must be a mapping, got {type(item).__name__}.")
    return entries


def build_edge_index_from_boundary_file(
    path: str | Path,
    num_clusters: int,
) -> torch.Tensor:
    """flow3d/analysis/cluster_pairs.py가 저장한 fixed_boundary_indices.pt를 읽어
    cluster adjacency graph의 edge_index를 만든다.

    해당 파일은 {"{a}_{b}": {"cluster_a": a, "cluster_b": b, ...}, ...} 형태이며,
    a, b는 model.fg.get_cluster_ids() 기준 cluster index다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary-relation file not found: {path}")

    raw = torch.load(path, map_location="cpu", weights_only=False)
    entries = _normalize_pair_entries(raw)

    pairs = [(int(item["cluster_a"]), int(item["cluster_b"])) for item in entries]
    return build_edge_index_from_pairs(pairs, num_clusters)


class _GraphAttentionBlock(nn.Module):
    """Canonical(고정) topology 위에서 edge attention weight만 프레임마다 새로
    계산하는 multi-head graph attention (self-loop 포함 GAT 스타일) 1-layer.

    어떤 cluster 쌍이 연결되는가(topology, adj_mask)는 절대 바뀌지 않고,
    "그 edge를 이번 프레임에 얼마나 신뢰할지"만 현재 프레임의 node hidden
    feature로부터 매번 다시 예측된다.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, negative_slope: float = 0.2):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim({hidden_dim}) must be divisible by num_heads({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.attn_src)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(
        self, h: torch.Tensor, adj_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (C, B, H) node hidden features
        adj_mask: (C, C) bool, 고정된 canonical topology (self-loop 포함).
            adj_mask[i, j] = True 면 j -> i 로 message passing.
        returns: 갱신된 h (C, B, H), attention weight (C_dst, C_src, B, num_heads)
        """
        C, B, _ = h.shape
        Wh = self.proj(h).view(C, B, self.num_heads, self.head_dim)  # (C, B, heads, d)

        e_dst = torch.einsum("ibhd,hd->ibh", Wh, self.attn_dst)  # (C, B, heads)
        e_src = torch.einsum("jbhd,hd->jbh", Wh, self.attn_src)  # (C, B, heads)
        logits = self.leaky_relu(e_dst[:, None] + e_src[None, :])  # (C_dst, C_src, B, heads)

        mask = adj_mask[:, :, None, None]
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)  # 각 dst node의 (고정된) neighbor 집합에 대해서만 정규화

        aggregated = torch.einsum("ijbh,jbhd->ibhd", attn, Wh).reshape(C, B, -1)
        out = self.out_proj(aggregated)
        return h + self.act(out), attn


class ClusterGraphGNN(nn.Module):
    """cluster adjacency graph를 반영해 coarse rotation/translation에 더할
    correction만 출력하는 GNN.

    Graph topology(canonical space에서 인접한 cluster 쌍)는 생성 시점에 고정되고
    학습 중 절대 바뀌지 않는다. 대신 각 edge의 영향력(attention weight)은
    frame마다 그 시점 cluster motion feature로부터 다시 예측된다 -- 즉
    "누가 이웃인가"는 고정, "이번 프레임에 그 이웃을 얼마나 참고할까"만 가변.

    최종 layer(decoder)를 0으로 초기화하므로, 학습 시작 시점에는
    correction이 정확히 0이 되어 graph-uninformed 기존 방식과 동일하게 동작한다.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_clusters: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        edge_index = _make_bidirectional(edge_index, num_clusters)
        adj_mask = torch.zeros(num_clusters, num_clusters, dtype=torch.bool)
        adj_mask[edge_index[0], edge_index[1]] = True
        self.register_buffer("adj_mask", adj_mask)
        # log(degree) per dst node, used to normalize attention entropy to [0, 1].
        # Degree-1 nodes (self-loop only, no real neighbor) have exactly one valid
        # choice, so their attention is trivially peaked and excluded below.
        degree = adj_mask.sum(dim=1).float()
        self.register_buffer("_entropy_valid", degree > 1)
        self.register_buffer("_log_degree", torch.log(degree.clamp_min(1.0)))

        self.num_clusters = num_clusters
        self.encoder = nn.Linear(NODE_FEAT_DIM, hidden_dim)
        self.layers = nn.ModuleList(
            [
                _GraphAttentionBlock(hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Linear(hidden_dim, NODE_FEAT_DIM)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        self._last_attn_weights: list[torch.Tensor] | None = None

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        coarse_rot_6d: (C, B, 6)
        coarse_transl: (C, B, 3)
        returns: delta_rot_6d (C, B, 6), delta_transl (C, B, 3)
        """
        if coarse_rot_6d.shape[0] != self.num_clusters:
            raise ValueError(
                f"coarse_rot_6d has {coarse_rot_6d.shape[0]} clusters, "
                f"expected {self.num_clusters}"
            )

        node_feat = torch.cat([coarse_rot_6d, coarse_transl], dim=-1)  # (C, B, 9)
        h = self.encoder(node_feat)
        attn_weights = []
        for layer in self.layers:
            h, attn = layer(h, self.adj_mask)
            attn_weights.append(attn)
        self._last_attn_weights = attn_weights

        delta = self.decoder(h)  # (C, B, 9)
        delta_rot, delta_transl = delta.split([ROT_DIM, TRANSL_DIM], dim=-1)
        return delta_rot, delta_transl

    def get_last_attention_weights(self) -> list[torch.Tensor] | None:
        """가장 최근 forward에서 나온 layer별 attention weight
        (C_dst, C_src, B, num_heads) 목록. topology(adj_mask)는 고정이고 이
        값만 프레임(B)마다 달라진다 -- 어떤 edge가 언제 강하게 쓰이는지 확인할 때 사용."""
        return self._last_attn_weights

    def attention_entropy_regularization_loss(self, eps: float = 1e-8) -> torch.Tensor:
        """attention이 (고정된) neighbor 집합 중 한두 개로 너무 쏠리는 것을
        막는 정규화 loss.

        각 dst node/frame/head별로 attention 분포의 entropy를
        log(degree)로 정규화해 [0, 1] 범위로 만들고 (0 = 한 neighbor에 완전히
        쏠림, 1 = 이웃들에 균등), (1 - normalized_entropy)의 평균을 반환한다.
        즉 최소화하면 attention이 collapse하지 않고 이웃들에 고르게 퍼지도록
        유도된다. self-loop만 있고 실제 이웃이 없는 cluster(degree == 1)는
        선택의 여지가 없어 계산에서 제외한다.

        get_last_attention_weights()와 마찬가지로 forward가 먼저 호출되어
        있어야 한다.
        """
        if self._last_attn_weights is None:
            raise RuntimeError(
                "forward() must be called before "
                "attention_entropy_regularization_loss()."
            )
        if not bool(self._entropy_valid.any()):
            # Every cluster is isolated (self-loop only) -- no attention to regularize.
            return self._last_attn_weights[0].new_zeros(())

        losses = []
        for attn in self._last_attn_weights:  # (C_dst, C_src, B, heads)
            entropy = -(attn * torch.log(attn.clamp_min(eps))).sum(dim=1)  # (C_dst, B, heads)
            normalized_entropy = entropy / self._log_degree.clamp_min(eps)[:, None, None]
            losses.append((1.0 - normalized_entropy)[self._entropy_valid].mean())

        return torch.stack(losses).mean()


class GraphCorrectedScalableMotionBases(ScalableMotionBases):
    """ScalableMotionBases + graph-aware coarse transform correction.

    기존 coarse rotation/translation(self.params["rots"]/["transls"])은 그대로
    두고, ClusterGraphGNN이 예측한 correction을 더한 값으로 coarse transform을
    계산한 뒤, fine(local) motion 결합은 부모 클래스와 완전히 동일한 수식을 쓴다.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        rots: torch.Tensor,
        transls: torch.Tensor,
        fine_rots: torch.Tensor,
        fine_transls: torch.Tensor,
        edge_index: torch.Tensor,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
    ):
        super().__init__(centers, rots, transls, fine_rots, fine_transls)
        self.graph_gnn = ClusterGraphGNN(
            edge_index=edge_index,
            num_clusters=self.num_clusters,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
        )
        self._last_correction: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
    ) -> "GraphCorrectedScalableMotionBases":
        """기존에 학습된(혹은 초기화된) ScalableMotionBases로부터 graph-corrected
        버전을 만든다. coarse/fine motion 파라미터 값은 그대로 복사되고,
        GNN correction만 새로 추가된다 (correction은 0으로 초기화됨)."""
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
    def has_graph_gnn_state(state_dict: dict, prefix: str) -> bool:
        """state_dict의 motion_bases가 GraphCorrectedScalableMotionBases로 저장된
        것인지 확인한다. prefix는 "motion_bases." 처럼 끝에 점이 붙은 형태여야
        한다 (params.가 아니라)."""
        graph_gnn_prefix = f"{prefix}graph_gnn."
        return any(key.startswith(graph_gnn_prefix) for key in state_dict)

    @classmethod
    def init_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "motion_bases.",
    ) -> "GraphCorrectedScalableMotionBases":
        """체크포인트만으로 GraphCorrectedScalableMotionBases를 통째로 복원한다.

        edge_index/hidden_dim/num_layers/num_heads 같은 architecture 인자를
        따로 넘길 필요가 없다 -- 전부 저장된 텐서의 shape에서 그대로 읽어온다.
        canonical adjacency topology(adj_mask)도 학습 때 그대로 저장돼 있으므로
        cluster_pairs.py의 fixed_boundary_indices.pt를 다시 읽을 필요도 없다.
        (이 값들은 어차피 아래에서 load_state_dict로 다시 덮어써지므로, 생성자에
        넘기는 placeholder edge_index 자체는 shape만 맞으면 무엇이든 상관없다.)
        """
        base = ScalableMotionBases.init_from_state_dict(
            state_dict, prefix=f"{prefix}params."
        )

        graph_gnn_prefix = f"{prefix}graph_gnn."
        gnn_keys = [
            key for key in state_dict if key.startswith(graph_gnn_prefix)
        ]
        if not gnn_keys:
            raise KeyError(f"No '{graph_gnn_prefix}*' keys found in state_dict.")

        num_clusters = state_dict[f"{graph_gnn_prefix}adj_mask"].shape[0]
        hidden_dim = state_dict[f"{graph_gnn_prefix}encoder.weight"].shape[0]
        num_heads = state_dict[f"{graph_gnn_prefix}layers.0.attn_dst"].shape[0]

        layers_prefix = f"{graph_gnn_prefix}layers."
        layer_indices = {
            int(key[len(layers_prefix):].split(".", 1)[0])
            for key in gnn_keys
            if key.startswith(layers_prefix)
        }
        num_layers = max(layer_indices) + 1

        placeholder_edge_index = torch.empty((2, 0), dtype=torch.long)
        graph_bases = cls.from_scalable_motion_bases(
            base,
            edge_index=placeholder_edge_index,
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
            gnn_num_heads=num_heads,
        )

        gnn_state = {
            key[len(graph_gnn_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(graph_gnn_prefix)
        }
        graph_bases.graph_gnn.load_state_dict(gnn_state, strict=True)

        return graph_bases

    def _corrected_coarse(
        self, ts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_rot_6d = self.params["rots"][:, ts]  # (C, B, 6)
        coarse_transl = self.params["transls"][:, ts]  # (C, B, 3)
        delta_rot, delta_transl = self.graph_gnn(coarse_rot_6d, coarse_transl)

        self._last_correction = {
            "delta_rot": delta_rot,
            "delta_transl": delta_transl,
        }

        corrected_rot_6d = coarse_rot_6d + delta_rot
        corrected_transl = coarse_transl + delta_transl
        return corrected_rot_6d, corrected_transl

    def correction_regularization_loss(self) -> torch.Tensor:
        """가장 최근 forward에서 나온 correction 크기에 대한 L2 penalty.

        GNN이 "최종 transform"이 아니라 "보정량"만 출력하도록 유도하는 loss.
        compute_transforms 혹은 compute_transforms_coarse를 먼저 호출해야 한다.
        """
        if self._last_correction is None:
            raise RuntimeError(
                "compute_transforms(_coarse) must be called before "
                "correction_regularization_loss()."
            )
        delta_rot = self._last_correction["delta_rot"]
        delta_transl = self._last_correction["delta_transl"]
        return delta_rot.pow(2).mean() + delta_transl.pow(2).mean()

    def attention_entropy_regularization_loss(self) -> torch.Tensor:
        """graph_gnn.attention_entropy_regularization_loss()로 위임.

        attention이 (고정된) neighbor 중 한둘에 쏠리지 않도록 막는 정규화 loss.
        compute_transforms(_coarse)를 먼저 호출해야 한다.
        """
        return self.graph_gnn.attention_entropy_regularization_loss()

    def compute_transforms_coarse(
        self, ts: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        :param ts (B)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        corrected_rot_6d, corrected_transl = self._corrected_coarse(ts)
        coarse_rotmats = cont_6d_to_rmat(corrected_rot_6d)  # (C, B, 3, 3)
        centers = self.params["centers"]  # (C, 3)

        transls_eff = (
            -torch.einsum("cbij,cj->cbi", coarse_rotmats, centers)
            + centers[:, None]
            + corrected_transl
        )  # (C, B, 3)

        return torch.cat(
            [coarse_rotmats[cluster_ids], transls_eff[cluster_ids].unsqueeze(-1)],
            dim=-1,
        )

    def compute_transforms(
        self, ts: torch.Tensor, coefs: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """coarse-to-fine 결합 수식은 ScalableMotionBases.compute_transforms와
        동일하며, coarse rotation/translation만 graph correction이 적용된
        값으로 바뀐다.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        assert coefs.shape[0] == cluster_ids.shape[0]
        G, F_dim = coefs.shape
        C, _, B, _ = self.params["fine_transls"][:, :, ts].shape

        # --- graph-aware coarse transform (기존 코드와 다른 부분) ---
        corrected_rot_6d, corrected_transl = self._corrected_coarse(ts)
        coarse_rotmats = cont_6d_to_rmat(corrected_rot_6d)  # (C, B, 3, 3)
        centers = self.params["centers"]  # (C, 3)

        # --- 이하는 ScalableMotionBases.compute_transforms와 동일 ---
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
            + corrected_transl[:, None]
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
    # Smoke test: 0-init GNN이면 GraphCorrectedScalableMotionBases는
    # baseline ScalableMotionBases와 동일한 transform을 내야 한다.
    torch.manual_seed(0)
    num_clusters, num_frames, num_fine, num_fg = 5, 8, 3, 40

    centers = torch.randn(num_clusters, 3)
    rots = torch.randn(num_clusters, num_frames, 6)
    transls = torch.randn(num_clusters, num_frames, 3) * 0.1
    fine_rots = torch.randn(num_clusters, num_fine, num_frames, 6)
    fine_transls = torch.randn(num_clusters, num_fine, num_frames, 3) * 0.01

    baseline = ScalableMotionBases(centers, rots, transls, fine_rots, fine_transls)

    edge_index = build_edge_index_from_pairs(
        [(0, 1), (1, 2), (2, 3), (3, 4)], num_clusters=num_clusters
    )
    graph_bases = GraphCorrectedScalableMotionBases.from_scalable_motion_bases(
        baseline, edge_index=edge_index
    )

    ts = torch.arange(num_frames)
    cluster_ids = torch.randint(0, num_clusters, (num_fg,))
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    ref = baseline.compute_transforms(ts, coefs, cluster_ids)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids)
    max_diff = (ref - out).abs().max().item()
    print(f"max |baseline - graph_corrected(zero-init)| = {max_diff:.3e}")
    assert max_diff < 1e-5, "zero-init correction should reproduce the baseline exactly"

    reg_loss = graph_bases.correction_regularization_loss()
    print(f"correction_regularization_loss (should be 0 at init) = {reg_loss.item():.3e}")

    # attention itself is not zero-initialized (only the decoder is), so its
    # entropy loss is generally nonzero even at init -- just sanity check range.
    attn_reg_loss = graph_bases.attention_entropy_regularization_loss()
    print(f"attention_entropy_regularization_loss (in [0, 1]) = {attn_reg_loss.item():.3e}")
    assert 0.0 <= attn_reg_loss.item() <= 1.0 + 1e-5

    # a maximally peaked attention (one-hot) should give a loss near 1.
    fake_attn = torch.zeros(num_clusters, num_clusters, num_frames, 4)
    fake_attn[torch.arange(num_clusters), torch.arange(num_clusters)] = 1.0  # self-loop only
    fake_attn[0, 1] = 1.0
    fake_attn[0, 0] = 0.0  # cluster 0 puts all weight on neighbor 1
    graph_bases.graph_gnn._last_attn_weights = [fake_attn]
    peaked_loss = graph_bases.attention_entropy_regularization_loss()
    print(f"attention_entropy_regularization_loss (near-peaked case) = {peaked_loss.item():.3e}")

    print("OK")

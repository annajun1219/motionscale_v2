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
1. GNN은 cluster별이 아니라 cluster graph의 각 (무방향) edge (a, b)마다 스칼라
   correction 크기 m_{ab}(t)만 예측한다 (회전 없음).
2. 방향은 학습하지 않는다: edge (a, b)의 "두 경계를 잇는 방향"
   dir_{ab}(t) = normalize(anchor_b_coarse(t) - anchor_a_coarse(t))를 coarse
   (per-cluster rigid) transform만으로 매 프레임 기하학적으로 계산하고, 이
   값은 항상 detach한다 -- 오직 스칼라 m_{ab}만 gradient를 받는다.
3. correction은 world-space additive translation으로, 최종 (coarse+fine
   blended) 위치의 translation 성분에만 더해진다 (회전은 절대 건드리지 않는다):
     Gaussian i (side a, edge (a,b)) -> position_i += +0.5 * m_{ab} * dir_{ab} * w_i
     Gaussian i (side b, edge (a,b)) -> position_i += -0.5 * m_{ab} * dir_{ab} * w_i
   양쪽을 반씩 움직여 이음매를 좁힌다.
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
5. m_{ab}는 head를 0으로 초기화하므로 학습 시작 시점에는 correction이 정확히
   0이다 (zero-init 등가성, 기존 *GraphCorrectedScalableMotionBases와 동일한
   관례).

Loss는 이 파일이 아니라 flow3d/trainer.py가 부른다 (다른 GNN correction
variant와 동일한 분리: 이 파일은 correction의 정의/적용, loss는 이 파일이
제공하는 compute_boundary_gap_loss/boundary_magnitude_* 함수를 trainer가
호출). compute_boundary_gap_loss는:
  - "현재 boundary에 속한 Gaussian"(falloff row, 위와 동일하게 densify/cull에
    따라 갱신됨)의 실제(coarse+fine blended) 위치를 falloff weight로 가중
    평균해서 각 side의 현재 위치로 삼되, base coarse/fine motion 입력은
    detach하고 correction의 스칼라 m_{ab}에만 gradient가 흐르도록 격리한다
    (flow3d/analysis/loss_joint_gnn_only.py와 동일한 철학).
  - 그 거리가 (canonical_distance + tolerance)보다 벌어질 때만 (hinge) loss를
    준다 -- 허용 범위 안에서는 정확히 0.
boundary_magnitude_reg_loss/boundary_magnitude_smoothness_loss는 correction
크기 자체와 그 시간 변화(가속도)를 제한한다.

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

flow3d/trainer.py는 매 control_step()(densify/cull/bases-control이 실제로
foreground Gaussian 개수를 바꿀 수 있는 지점) 끝에서
motion_bases.num_fg_gaussians를 model.fg.num_gaussians와 비교해 다르면
refresh_boundary_falloff를 호출한다.

이 파일에서 제공하는 것
------------------------
- EdgeBoundarySets / load_edge_boundary_sets: build_cluster_graph.py의
  edges.pt를 읽어 edge별 (cluster_a, cluster_b, canonical anchor)를 만든다
  (anchor는 그 파일이 저장한 boundary Gaussian 집합의 canonical 평균 위치일
  뿐이므로, 만든 뒤에는 index를 더 이상 갖고 있지 않는다).
- EdgeBoundaryGNN: cluster graph 위 mean-aggregation node encoding +
  edge-level MLP head로 edge별 스칼라 correction 크기를 예측하는 작은 GNN.
- EdgeBoundaryGraphCorrectedScalableMotionBases: ScalableMotionBases를
  상속해 compute_transforms에서만 (falloff-weighted, translation-only) 보정을
  끼워 넣는 드롭인 대체. compute_transforms_coarse는 건드리지 않는다(상속
  그대로). refresh_boundary_falloff(canonical_means, cluster_ids_all)로
  densify/cull 이후 falloff row를 다시 계산한다.
- compute_boundary_gap_loss / boundary_magnitude_reg_loss /
  boundary_magnitude_smoothness_loss: flow3d/trainer.py가 호출하는 loss 함수.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.graph_coupling import CENTER_DIM, NODE_FEAT_DIM, ROT_DIM, TRANSL_DIM
from flow3d.params import ScalableMotionBases
from flow3d.transforms import cont_6d_to_rmat

__all__ = [
    "EdgeBoundarySets",
    "load_edge_boundary_sets",
    "EdgeBoundaryGNN",
    "EdgeBoundaryGraphCorrectedScalableMotionBases",
    "compute_boundary_gap_loss",
    "boundary_magnitude_reg_loss",
    "boundary_magnitude_smoothness_loss",
]

EDGE_MLP_EXTRA_DIM = TRANSL_DIM + CENTER_DIM + 2  # rel_transl(3) + rel_center(3) + gap_dist(1) + canonical_distance(1)
DEFAULT_FALLOFF_MIN_WEIGHT = 1e-3


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_long_indices(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


@dataclass
class EdgeBoundarySets:
    """edges.pt(build_cluster_graph.py)에서 읽은, edge별 "boundary의 의미"
    (고정): topology와 각 side의 canonical anchor 위치. 만든 시점의 boundary
    Gaussian index는 anchor를 계산하는 데만 잠깐 쓰이고 저장되지 않는다 --
    학습 중 densify/cull로 그 index들이 stale해져도 이 구조체 자체는 영향받지
    않는다 (모듈 docstring 참고).

    :param cluster_a / cluster_b: (E,) long, 무방향 edge의 두 cluster id.
    :param anchor_a_canonical / anchor_b_canonical: (E, 3), 각 side
        boundary Gaussian들의 canonical 평균 위치 (고정된 3D 점).
    """

    cluster_a: torch.Tensor
    cluster_b: torch.Tensor
    anchor_a_canonical: torch.Tensor
    anchor_b_canonical: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.cluster_a.shape[0])

    @property
    def canonical_distance(self) -> torch.Tensor:
        """(E,) ||anchor_a - anchor_b|| in canonical space -- 허용 거리(tolerance)의 기준값."""
        return (self.anchor_a_canonical - self.anchor_b_canonical).norm(dim=-1)


def load_edge_boundary_sets(
    edges_path: str | Path,
    canonical_means: torch.Tensor,
    device: torch.device | None = None,
) -> EdgeBoundarySets:
    """
    flow3d/analysis/build_cluster_graph.py의 edges.pt를 읽어 EdgeBoundarySets
    (고정된 edge topology + anchor 위치)를 만든다.

    :param edges_path: edges.pt 경로. "edges_kept" 리스트를 가진 dict여야 하며,
        각 원소는 최소 cluster_a, cluster_b, boundary_global_indices_a,
        boundary_global_indices_b를 가져야 한다.
    :param canonical_means: (G, 3) 저장된 index들이 가리키는, 이 함수를 호출하는
        시점의 canonical foreground Gaussian means (예:
        model.fg.params["means"].detach()) -- anchor 위치를 한 번 계산하는
        데만 쓰이고, 이후로는 참조하지 않는다.
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

        cluster_a_rows.append(int(entry["cluster_a"]))
        cluster_b_rows.append(int(entry["cluster_b"]))
        anchor_a_rows.append(means_cpu[idx_a].mean(dim=0))
        anchor_b_rows.append(means_cpu[idx_b].mean(dim=0))

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
    )


def _compute_falloff_rows(
    canonical_means: torch.Tensor,
    cluster_ids_all: torch.Tensor,
    edge_cluster_a: torch.Tensor,
    edge_cluster_b: torch.Tensor,
    falloff_radius: float,
    min_weight: float = DEFAULT_FALLOFF_MIN_WEIGHT,
) -> dict[str, torch.Tensor]:
    """
    edge/side마다, "그 side의 cluster에 속한 (지금 존재하는) Gaussian이 반대편
    cluster까지 얼마나 가까운가"로 falloff weight를 계산하고, min_weight보다
    큰 것만 sparse row로 남긴다.

    거리 기준은 flow3d/analysis/cluster_graph.py가 애초에 boundary_global_
    indices_a/b를 고를 때 쓰는 것과 정확히 같다 -- directed_nn/
    select_boundary_local_indices도 "반대편 cluster까지의 최근접-이웃 거리"로
    boundary 후보를 뽑는다 (hard top-k 컷). 여기서는 그 동일한 신호를 hard
    cutoff 대신 부드러운 RBF weight로 재사용한다. 저장된 index가 아니라 두
    cluster의 CURRENT canonical 위치만 있으면 계산되므로, densify/cull로
    canonical_means/cluster_ids_all이 통째로 바뀐 뒤에도 옛 상태 없이 처음부터
    다시 계산할 수 있다 (refresh_boundary_falloff가 매번 이 함수를 다시
    부른다). anchor_a/b_canonical(고정된 "boundary의 의미")은 여기서 전혀
    쓰이지 않는다 -- direction/canonical_distance 계산에만 쓰인다.

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
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue

        # (na, nb) cross-cluster distance -- same nearest-neighbor-to-the-
        # other-cluster signal directed_nn uses, just kept dense here since
        # per-edge cluster sizes are small enough for a plain cdist.
        cross_dist = torch.cdist(canonical_means[idx_a], canonical_means[idx_b])
        dist_a_to_b = cross_dist.min(dim=1).values  # (na,) nearest point in B, for each point in A
        dist_b_to_a = cross_dist.min(dim=0).values  # (nb,) nearest point in A, for each point in B

        for idx_side, dist_side, sign in ((idx_a, dist_a_to_b, 1.0), (idx_b, dist_b_to_a, -1.0)):
            weight = torch.exp(-(dist_side / falloff_radius) ** 2)
            keep = weight > min_weight
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


def _build_adjacency_with_self_loops(edge_index: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """(2, E) cluster-id pairs -> symmetric (C, C) bool adjacency with self-loops."""
    adj = torch.zeros(num_clusters, num_clusters, dtype=torch.bool, device=edge_index.device)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = True
        adj[dst, src] = True
    adj.fill_diagonal_(True)
    return adj


class _MeanAggLayer(nn.Module):
    """flow3d/graph_coupling.py's _MeanAggLayer와 동일한 fixed-topology
    mean-aggregation message passing (1 layer, residual). cluster node hidden
    state를 만드는 데만 쓰이고, correction 자체는 이 레이어가 아니라
    EdgeBoundaryGNN의 edge_mlp가 출력한다.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        msg = self.lin(h)  # (C, B, H)
        agg = torch.einsum("ij,jbh->ibh", adj_norm, msg)
        return h + self.act(agg)


def _coarse_rigid_transform(
    rots_6d: torch.Tensor, transls: torch.Tensor, centers: torch.Tensor, cluster_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ScalableMotionBases.compute_transforms_coarse와 동일한 수식(rotation은
    합성 없이 그대로, translation은 R_eff = -R@c + c + t)을, 주어진(임의로
    detach될 수 있는) rots_6d/transls/centers에 대해 임의의 cluster_idx 질의로
    계산하는 헬퍼. cluster_idx가 (K,)이면 K개 cluster 각각의 rigid transform을
    반환한다 (self-composition/recursion 없이 순수 함수로 재사용하기 위함).

    :param rots_6d: (C, B, 6)
    :param transls: (C, B, 3)
    :param centers: (C, 3)
    :param cluster_idx: (K,) long
    :return: R (K, B, 3, 3), t_eff (K, B, 3)
    """
    R = cont_6d_to_rmat(rots_6d[cluster_idx])  # (K, B, 3, 3)
    c = centers[cluster_idx]  # (K, 3)
    t_eff = (
        -torch.einsum("kbij,kj->kbi", R, c) + c[:, None, :] + transls[cluster_idx]
    )  # (K, B, 3)
    return R, t_eff


class EdgeBoundaryGNN(nn.Module):
    """cluster graph 위에서 mean-aggregation으로 node feature를 encode한 뒤,
    각 (무방향) edge마다 스칼라 correction magnitude m_{ab}(t)만 출력하는 작은
    GNN. 방향은 여기서 예측하지 않는다 (호출하는 쪽에서 기하학적으로 계산).

    edge_head(edge_mlp)의 마지막 Linear는 0으로 초기화되므로, 학습 시작
    시점에는 magnitude가 정확히 0이다 (zero-init 등가성).
    """

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

        edge_index = torch.stack([edge_cluster_a, edge_cluster_b], dim=0)
        adj = _build_adjacency_with_self_loops(edge_index, num_clusters)
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1).float()
        self.register_buffer("adj_norm", adj.float() / degree)
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
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(
        self,
        coarse_rot_6d: torch.Tensor,
        coarse_transl: torch.Tensor,
        centers: torch.Tensor,
        gap_dist: torch.Tensor,
        canonical_distance: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param coarse_rot_6d: (C, B, 6)
        :param coarse_transl: (C, B, 3)
        :param centers: (C, 3)
        :param gap_dist: (E, B) current coarse anchor-to-anchor distance (detached by caller).
        :param canonical_distance: (E,) fixed canonical anchor-to-anchor distance.
        :return: magnitude (E, B)
        """
        if coarse_rot_6d.shape[0] != self.num_clusters:
            raise ValueError(
                f"coarse_rot_6d has {coarse_rot_6d.shape[0]} clusters, expected {self.num_clusters}"
            )

        C, B, _ = coarse_rot_6d.shape
        center_feat = centers[:, None, :].expand(C, B, CENTER_DIM)
        node_feat = torch.cat([coarse_rot_6d, coarse_transl, center_feat], dim=-1)  # (C, B, 12)

        h = self.encoder(node_feat)
        for layer in self.layers:
            h = layer(h, self.adj_norm)

        a, b = self.edge_cluster_a, self.edge_cluster_b
        h_a, h_b = h[a], h[b]  # (E, B, H) each
        rel_transl = coarse_transl[b] - coarse_transl[a]  # (E, B, 3)
        rel_center = (centers[b] - centers[a])[:, None, :].expand(-1, B, -1)  # (E, B, 3)
        gap_feat = gap_dist[..., None]  # (E, B, 1)
        canon_feat = canonical_distance[:, None, None].expand(-1, B, 1)  # (E, B, 1)

        edge_feat = torch.cat([h_a, h_b, rel_transl, rel_center, gap_feat, canon_feat], dim=-1)
        magnitude = self.edge_mlp(edge_feat).squeeze(-1)  # (E, B)
        return magnitude


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
    refresh_boundary_falloff(canonical_means, cluster_ids_all)를 호출해야
    한다 -- edge topology/anchor("boundary의 의미")는 그대로 두고, falloff row
    ("지금 boundary에 속한 Gaussian이 누구인지")만 다시 계산한다.
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
        falloff_global_idx: torch.Tensor,
        falloff_edge_id: torch.Tensor,
        falloff_sign: torch.Tensor,
        falloff_weight: torch.Tensor,
        num_fg_gaussians: int,
        falloff_radius: float,
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
        self.register_buffer("falloff_radius", torch.tensor(float(falloff_radius)))
        self.register_buffer("falloff_min_weight", torch.tensor(float(falloff_min_weight)))

        # Live ("지금 boundary에 속한 Gaussian"): replaced wholesale by
        # refresh_boundary_falloff whenever foreground Gaussian count/order changes.
        self.register_buffer("falloff_global_idx", falloff_global_idx.clone().long())
        self.register_buffer("falloff_edge_id", falloff_edge_id.clone().long())
        self.register_buffer("falloff_sign", falloff_sign.clone().float())
        self.register_buffer("falloff_weight", falloff_weight.clone().float())
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
        """가장 최근 compute_transforms 호출에서 나온 {"magnitude": (E, B)} (디버깅/로깅/loss용)."""
        return self._last_boundary_correction

    @torch.no_grad()
    def refresh_boundary_falloff(
        self, canonical_means: torch.Tensor, cluster_ids_all: torch.Tensor
    ) -> None:
        """densify/cull(flow3d/trainer.py's _densify_control_step/
        _cull_control_step/_bases_control_step) 이후, foreground Gaussian
        배열이 바뀐 CURRENT canonical_means/cluster_ids_all로 falloff row와
        num_fg_gaussians를 다시 계산해 덮어쓴다. edge_cluster_a/b와
        anchor_a/b_canonical("boundary의 의미")는 건드리지 않는다 -- 옛 index를
        전혀 참조하지 않으므로 densify/cull이 몇 번 일어났든 항상 처음부터
        다시 계산 가능하다.

        :param canonical_means: (G, 3) 현재 canonical foreground Gaussian means
            (예: model.fg.params["means"].detach()).
        :param cluster_ids_all: (G,) 현재 foreground Gaussian cluster id, 같은
            순서 (예: model.fg.get_cluster_ids().reshape(-1).long()).
        """
        device = self.falloff_global_idx.device
        falloff = _compute_falloff_rows(
            canonical_means,
            cluster_ids_all,
            self.edge_cluster_a,
            self.edge_cluster_b,
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

    @classmethod
    def from_scalable_motion_bases(
        cls,
        bases: ScalableMotionBases,
        edge_index: torch.Tensor,
        edges_path: str | Path,
        canonical_means: torch.Tensor,
        cluster_ids_all: torch.Tensor,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 1,
        falloff_radius: float = 0.05,
        falloff_min_weight: float = DEFAULT_FALLOFF_MIN_WEIGHT,
    ) -> "EdgeBoundaryGraphCorrectedScalableMotionBases":
        """기존 (초기화된) ScalableMotionBases로부터 edge-boundary-corrected
        버전을 만든다. coarse/fine motion 파라미터 값은 그대로 복사되고,
        correction만 새로 추가된다 (0으로 초기화됨).

        :param edge_index: 다른 *GraphCorrectedScalableMotionBases와
            call-site를 맞추기 위해 받지만 사용하지 않는다 -- edge topology는
            edges_path의 cluster_a/cluster_b에서 그대로 다시 읽는다 (같은
            파일이므로 edge_index와 항상 일치한다).
        :param edges_path: build_cluster_graph.py's edges.pt 경로.
        :param canonical_means: (G, 3) canonical foreground Gaussian means
            (예: fg_params.params["means"].detach()).
        :param cluster_ids_all: (G,) canonical foreground Gaussian cluster id
            (예: fg_params.get_cluster_ids().reshape(-1).long()).
        :param falloff_radius: falloff weight의 RBF 반경(scene 단위).
        """
        del edge_index
        boundary_sets = load_edge_boundary_sets(edges_path, canonical_means)
        falloff = _compute_falloff_rows(
            canonical_means,
            cluster_ids_all,
            boundary_sets.cluster_a,
            boundary_sets.cluster_b,
            falloff_radius,
            falloff_min_weight,
        )
        if falloff["global_idx"].numel() == 0:
            raise RuntimeError(
                f"No boundary falloff rows were produced from {edges_path} -- "
                f"falloff_radius ({falloff_radius}) may be too small relative to the scene scale."
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
            falloff_global_idx=falloff["global_idx"],
            falloff_edge_id=falloff["edge_id"],
            falloff_sign=falloff["sign"],
            falloff_weight=falloff["weight"],
            num_fg_gaussians=canonical_means.shape[0],
            falloff_radius=falloff_radius,
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
        값으로 덮어쓴다."""
        base = ScalableMotionBases.init_from_state_dict(state_dict, prefix=f"{prefix}params.")

        if f"{prefix}edge_cluster_a" not in state_dict:
            raise KeyError(f"No '{prefix}edge_cluster_a' buffer found in state_dict.")

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
        falloff_global_idx = state_dict[f"{prefix}falloff_global_idx"]
        falloff_edge_id = state_dict[f"{prefix}falloff_edge_id"]
        falloff_sign = state_dict[f"{prefix}falloff_sign"]
        falloff_weight = state_dict[f"{prefix}falloff_weight"]
        num_fg_gaussians = int(state_dict[f"{prefix}num_fg_gaussians"].item())
        falloff_radius = float(state_dict[f"{prefix}falloff_radius"].item())
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
            falloff_global_idx=falloff_global_idx,
            falloff_edge_id=falloff_edge_id,
            falloff_sign=falloff_sign,
            falloff_weight=falloff_weight,
            num_fg_gaussians=num_fg_gaussians,
            falloff_radius=falloff_radius,
            falloff_min_weight=falloff_min_weight,
            gnn_hidden_dim=hidden_dim,
            gnn_num_layers=num_layers,
        )

        own_state = {
            key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        graph_bases.load_state_dict(own_state, strict=True)
        return graph_bases

    def _edge_features_and_direction(
        self, ts: torch.Tensor, detach_base: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """edge별 correction magnitude(gradient 보유)와 방향(항상 detach)을 계산한다.

        :param detach_base: True면 GNN에 들어가는 coarse rot/transl/centers
            입력 자체를 detach한다 (isolated loss 계산용 -- gradient가 오직
            self.gnn 자신의 파라미터로만 흐르게 함,
            flow3d/analysis/loss_joint_gnn_only.py와 동일한 격리 방식).
            False면(기본 forward 경로) 평소처럼 gradient가 base coarse motion
            에도 정상적으로 흐른다 (렌더링/track 등 다른 loss가 이미 그렇게
            쓰고 있으므로).
        :return: magnitude (E, B) [gradient는 self.gnn 파라미터(및
            detach_base=False일 때 base motion)로 흐름], direction (E, B, 3)
            [항상 detached].
        """
        rots = self.params["rots"][:, ts]  # (C, B, 6)
        transls = self.params["transls"][:, ts]  # (C, B, 3)
        centers = self.params["centers"]  # (C, 3)
        if detach_base:
            rots = rots.detach()
            transls = transls.detach()
            centers = centers.detach()

        with torch.no_grad():
            R_a, t_a = _coarse_rigid_transform(rots, transls, centers, self.edge_cluster_a)
            R_b, t_b = _coarse_rigid_transform(rots, transls, centers, self.edge_cluster_b)
            anchor_a_t = torch.einsum("ebij,ej->ebi", R_a, self.anchor_a_canonical) + t_a
            anchor_b_t = torch.einsum("ebij,ej->ebi", R_b, self.anchor_b_canonical) + t_b
            gap_vec = anchor_b_t - anchor_a_t  # (E, B, 3)
            gap_dist = gap_vec.norm(dim=-1).clamp_min(1e-8)  # (E, B)
            direction = gap_vec / gap_dist[..., None]  # (E, B, 3)

        magnitude = self.gnn(rots, transls, centers, gap_dist, self.canonical_distance)  # (E, B)
        return magnitude, direction

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

        magnitude, direction = self._edge_features_and_direction(ts, detach_base=False)  # (E,B), (E,B,3)
        self._last_boundary_correction = {"magnitude": magnitude}

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


def compute_boundary_gap_loss(
    motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases,
    ts: torch.Tensor,
    canonical_means: torch.Tensor,
    coefs_all: torch.Tensor,
    cluster_ids_all: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """
    실제(coarse+fine blended, base 입력은 detach) 경계 거리가
    (canonical_distance + tolerance)보다 벌어질 때만 loss를 준다 (hinge,
    벌어지지 않으면 정확히 0). Gradient는 오직 motion_bases.gnn 자신의
    파라미터로만 흐른다 (base coarse/fine motion, cluster centers는 모두 이
    함수 안에서 detach된 입력) -- flow3d/analysis/loss_joint_gnn_only.py와
    동일한 격리 철학.

    "현재 side의 위치"는 motion_bases의 (라이브, densify/cull에 따라 갱신되는)
    falloff row를 그대로 재사용해서 falloff weight로 가중 평균한다 -- 별도로
    edges.pt를 다시 읽거나 boundary Gaussian index를 저장해 둘 필요가 없다
    (그 index는 densify/cull 이후 stale해지지만, 이 함수는 애초에 index를
    쓰지 않는다).

    :param motion_bases: EdgeBoundaryGraphCorrectedScalableMotionBases.
    :param ts: (B,) frame indices.
    :param canonical_means: (G_fg, 3) canonical foreground Gaussian means (detached),
        motion_bases.falloff_global_idx가 가리키는 것과 같은 순서.
    :param coefs_all: (G_fg, F) per-Gaussian fine-basis blend weights (detached).
    :param cluster_ids_all: (G_fg,) per-Gaussian raw cluster id.
    :param tolerance: 허용 거리(canonical_distance 위에 추가로 더 벌어져도 되는 여유량, scene 단위).
    :return: scalar; falloff row가 하나도 없거나 벌어진 pair가 없으면 0.0.
    """
    mb = motion_bases
    if not hasattr(mb, "gnn") or not hasattr(mb, "edge_cluster_a"):
        raise TypeError(
            "motion_bases must be an EdgeBoundaryGraphCorrectedScalableMotionBases."
        )

    idx = mb.falloff_global_idx  # (M,)
    if idx.numel() == 0:
        return canonical_means.new_zeros(())

    base_transforms = ScalableMotionBases.compute_transforms(
        mb, ts, coefs_all[idx], cluster_ids_all[idx]
    ).detach()  # (M, B, 3, 4) -- base coarse+fine motion detached, isolation point
    positions = _apply_transform(base_transforms, canonical_means[idx])  # (M, B, 3)

    weight = mb.falloff_weight  # (M,)
    side = (mb.falloff_sign < 0).long()  # 0 = side a, 1 = side b
    group_ids = mb.falloff_edge_id * 2 + side  # (M,)

    E = mb.num_edges
    B = positions.shape[1]
    weight_sums = positions.new_zeros(E * 2)
    pos_sums = positions.new_zeros(E * 2, B, 3)
    weight_sums.index_add_(0, group_ids, weight)
    pos_sums.index_add_(0, group_ids, positions * weight[:, None, None])
    means = pos_sums / weight_sums[:, None, None].clamp_min(1e-8)
    mean_a = means[0::2]  # (E, B, 3), falloff-weighted mean position, side a
    mean_b = means[1::2]  # (E, B, 3), side b

    magnitude, direction = mb._edge_features_and_direction(ts, detach_base=True)  # (E,B), (E,B,3)

    # side a moves +0.5*m*dir, side b moves -0.5*m*dir -> gap = (b - a) - m*dir
    gap_vec = (mean_b - mean_a) - direction * magnitude[..., None]  # (E, B, 3)
    dist = gap_vec.norm(dim=-1)  # (E, B)

    allowed = mb.canonical_distance[:, None] + tolerance  # (E, 1)
    hinge = F.relu(dist - allowed)
    # Edges with zero weight on one side (e.g. every Gaussian near that side was
    # just culled, before the next refresh_boundary_falloff runs) have an
    # undefined mean position (0/0 clamped to 0) -- exclude them rather than
    # spuriously penalizing/rewarding a meaningless gap.
    has_both_sides = (weight_sums[0::2] > 0) & (weight_sums[1::2] > 0)
    if not bool(has_both_sides.any()):
        return canonical_means.new_zeros(())
    return hinge[has_both_sides].pow(2).mean()


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
    # Sanity checks:
    #   1. zero-init: EdgeBoundary-corrected == plain ScalableMotionBases exactly.
    #   2. compute_transforms rejects a subset query (only full array supported).
    #   3. after the head moves off zero: correction is nonzero, boundary-anchor-adjacent
    #      points move by close to +-0.5*m*dir, and displacement decays smoothly away
    #      from the boundary (falloff).
    #   4. compute_boundary_gap_loss: 0 within tolerance, > 0 once forced open;
    #      gradient reaches only motion_bases.gnn's own parameters.
    #   5. boundary_magnitude_reg_loss / boundary_magnitude_smoothness_loss basic behavior.
    #   6. save -> init_from_state_dict round-trip reproduces identical output.
    #   7. refresh_boundary_falloff after a simulated densify/cull (Gaussian count AND
    #      order both change) keeps the correction correctly localized, with no stale
    #      indices anywhere -- and compute_transforms's full-array check tracks the new count.
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
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(edges_payload, f.name)
        edges_path = f.name

        graph_bases = EdgeBoundaryGraphCorrectedScalableMotionBases.from_scalable_motion_bases(
            baseline,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edges_path=edges_path,
            canonical_means=canonical_means,
            cluster_ids_all=cluster_ids_all,
            gnn_hidden_dim=16,
            gnn_num_layers=1,
            falloff_radius=0.3,
        )

    ts = torch.arange(num_frames)
    coefs = torch.softmax(torch.randn(num_fg, num_fine), dim=-1)

    # --- 1. zero-init equivalence ---
    ref = baseline.compute_transforms(ts, coefs, cluster_ids_all)
    out = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    max_diff = (ref - out).abs().max().item()
    print(f"[zero-init] max |baseline - edge_boundary_corrected| = {max_diff:.3e}")
    assert max_diff < 1e-5

    magnitude0 = graph_bases.last_boundary_correction["magnitude"]
    print(f"[zero-init] |magnitude|_max = {magnitude0.abs().max().item():.3e}")
    assert magnitude0.abs().max().item() == 0.0

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

    # --- 3. move the GNN's edge head off zero, then check magnitude/falloff behavior ---
    graph_bases.zero_grad()
    out2 = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    loss0 = out2.pow(2).mean()
    loss0.backward()
    with torch.no_grad():
        for p in graph_bases.gnn.edge_mlp[-1].parameters():
            if p.grad is not None:
                p += 0.05  # push magnitude off zero (positive: pull the two sides together)

    magnitude1, direction1 = graph_bases._edge_features_and_direction(ts, detach_base=False)
    print(f"[post-step] magnitude nonzero: {magnitude1.abs().max().item():.3e}")
    assert magnitude1.abs().max().item() > 0.0

    out3 = graph_bases.compute_transforms(ts, coefs, cluster_ids_all)
    means_out3 = torch.einsum(
        "pnij,pj->pni", out3, F.pad(canonical_means, (0, 1), value=1.0)
    )  # (G, B, 3)
    means_ref = torch.einsum(
        "pnij,pj->pni", ref, F.pad(canonical_means, (0, 1), value=1.0)
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

    # --- 4. compute_boundary_gap_loss: 0 within tolerance, > 0 once forced open; grad isolation ---
    graph_bases.zero_grad()
    gap_loss_tight = compute_boundary_gap_loss(
        graph_bases, ts, canonical_means, coefs, cluster_ids_all, tolerance=10.0
    )
    print(f"[gap-loss] huge tolerance -> loss={gap_loss_tight.item():.3e} (expect 0)")
    assert gap_loss_tight.item() == 0.0

    # Force cluster 1 to actually separate from cluster 0 (well beyond what the
    # small correction learned above can close) so the hinge has something to penalize.
    with torch.no_grad():
        graph_bases.params["transls"][1] += torch.tensor([0.5, 0.0, 0.0])

    gap_loss_strict = compute_boundary_gap_loss(
        graph_bases, ts, canonical_means, coefs, cluster_ids_all, tolerance=0.0
    )
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

    graph_bases.refresh_boundary_falloff(means_densified, cluster_ids_densified)
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

    graph_bases.refresh_boundary_falloff(means_culled, cluster_ids_culled)
    print(
        f"[refresh] after cull: num_fg_gaussians={int(graph_bases.num_fg_gaussians)} "
        f"(expect {int(keep_mask.sum())}), falloff rows={graph_bases.falloff_global_idx.numel()}"
    )
    assert int(graph_bases.num_fg_gaussians) == int(keep_mask.sum())
    assert graph_bases.falloff_global_idx.numel() > 0
    assert int(graph_bases.falloff_global_idx.max()) < int(keep_mask.sum())

    _ = graph_bases.compute_transforms(ts, coefs_culled, cluster_ids_culled)  # must not raise
    print("[refresh] post-cull compute_transforms OK with the new (smaller) full array")

    print("OK")

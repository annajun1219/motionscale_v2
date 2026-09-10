#!/usr/bin/env python3
"""
flow3d/graph_local_target.py

이 파일에서 생성하거나 --local-nodes-path로 읽은 local_nodes.pt에
있는 local control node들을 target/context로 분류하고, 같은 parent cluster
내부의 local kNN 그래프를 진단하는 스크립트.

local_nodes.pt는 기하학적 정보(각 노드의 parent_cluster_id, canonical
center, 그리고 체크포인트 시점의 member_global_indices snapshot)만 담고
있을 뿐, confidence/visibility는 전혀 포함하지 않는다 (module-level
docstring 및 meta["member_indices_caveat"] 참고). 따라서 confidence는
이 스크립트가 별도로 계산해야 하는데, 자체적으로 새 신호를 만들어내지 않고
flow3d/analysis/build_cluster_graph.py가 이미 사용 중인 raw 2D-track
confidence 파이프라인(load_model_and_clusters + load_raw_track_positions)을
그대로 재사용한다:

  - CasualDataset이 캐시해 둔 raw 2D track(.npy, cotracker3/tapir)을 여러
    query frame에서 풀링하고, get_tracks_3d_for_query_frame으로 3D로
    lift한 뒤, 각 포인트의 confidence(0~1, data/utils.py의
    parse_cotracker3_track_info/parse_tapir_track_info 참고: 확실히
    visible도 invisible도 아니면 0으로 clamp됨 -- 이 0이 곧 "unknown"
    프레임의 신호)를 함께 가져온다.
  - 이 raw track pool을 체크포인트의 canonical Gaussian 위치에 그리디
    최근접 매칭(_greedy_unique_match)시켜, "이 Gaussian은 대략 이 raw
    track과 같은 궤적을 갖는다"는 근사로 Gaussian별 confidence를 얻는다.

이 스크립트는 그 매칭된 raw-track 포인트들을 (node_centers에 대한 nearest-
center 규칙으로) 다시 local node에 배정한다. 매칭된 포인트의 canonical
위치(positions_all_frames[:, 0, :])는 애초에 실제 canonical Gaussian
위치에 최근접 매칭된 것이므로, node 배정 기준으로 그대로 써도
assign_gaussians_to_local_nodes를 원본 Gaussian에 직접 돌린 것과 사실상
같은 결과를 준다 -- 원본 fg 배열 global index까지 되짚어 갈 필요가 없다.
raw track은 오직 이 confidence 집계에만 쓰인다 -- 노드의 위치(아래
node_position)에는 관여하지 않는다.

node_position[T, N, 3]은 raw track이 아니라 MotionScale 자신의 학습된
모션으로 계산한다: 체크포인트의 canonical Gaussian(model.fg.params
["means"])을 같은 parent cluster 안에서 가장 가까운 local node에
assign_gaussians_to_local_nodes로 배정하고(로컬 노드가 원래 정의되는
바로 그 recomputable 규칙), 그 멤버 Gaussian들을 model.compute_poses_fg
로 매 프레임 posing한 뒤 노드별 평균을 취한다. target/context 구분 없이
모든 노드에 동일하게 적용된다. 시각화에서 찍는 점과 kNN edge의 끝점도 항상
이 node_position을 사용한다.

local_nodes.pt가 만들어질 때와 다른 --min-cluster-size, 다른 체크포인트,
혹은 cfg.yaml/track 캐시가 없는 work_dir로 이 스크립트를 돌리면 confidence
계산에 필요한 데이터가 애초에 존재하지 않는 것이므로, 절대 0이나 1로
추정하지 않고 어떤 파일/필드가 없는지 명시한 에러로 즉시 중단한다. 마찬가지로
어떤 노드에 배정되는 canonical Gaussian이 0개면(체크포인트가 densify/cull로
local_nodes.pt 생성 시점과 달라진 경우) node_position도 추정하지 않고 에러로
중단한다.

Target / context 분류 -- 더 이상 배타적이지 않다
-------------------------------------------------
node 단위로 두 스칼라를 raw-track confidence로부터 집계한다:
  - mean_confidence_known[n]: confidence > --unknown-confidence-floor인
    (member, frame) 관측치들만의 평균 confidence ("추적이 됐을 때 얼마나
    확신도가 높았나").
  - unknown_ratio[n]: confidence <= floor인 (member, frame) 관측치의 비율
    ("애초에 추적 자체가 안 된 비율").

target_mask[N] (정적, 시퀀스 전체 기준)은 "이 노드는 (전체적으로 신뢰도가
낮아) correction이 필요한 대상이다"라는 고정된 라벨이다:
    target_mask[n] := mean_confidence_known[n] < --target-confidence-threshold
                       OR unknown_ratio[n] > --target-unknown-ratio-threshold

이건 "이 노드가 다른 노드에게 context 역할을 할 수 있는가"와는 별개 질문이다
-- target으로 분류된 노드도 특정 프레임에는 (일시적으로 안 가려져서) 아주
confidence가 높을 수 있고, 그런 프레임에는 다른 target에게 context
message를 줄 수 있어야 한다. 그래서 context 자격은 target_mask와 무관하게
매 프레임 독립적으로 계산한다:
    active_context_mask[T, N] := node_confidence[T, N] >= --frame-context-confidence-threshold
target 여부와 상관없이 모든 노드에 대해 계산되므로, target_mask와
active_context_mask는 (같은 노드, 같은 프레임에 대해서도) 동시에 참일 수
있다 -- 배타적 분류가 아니다.

같은 parent cluster 안에서 node center 간 kNN 그래프(--knn-k)를 기본
edge_index로 두되, target 노드마다는 그 위에 별도로 "source 후보 목록"을
시퀀스 전체 기준으로 한 번만(= 모든 프레임 공통, 프레임마다 달라지지 않음)
고정해 둔다 (select_target_source_candidates). 이 후보 목록은 3단계로,
앞 단계에서 --target-context-goal개의 활성 source(target/non-target 무관,
아래 message_source_mask 참고)를 이미 최대한 많은 프레임에서 확보했다면
뒤 단계는 열지 않는다:
  1. 같은 parent cluster 내 이 target의 기존 kNN 이웃(--knn-k)부터, 가까운
     순으로 채운다.
  2. 그래도 부족하면 같은 parent cluster의 다음으로 가까운 node를 순서대로
     추가한다.
  3. 그래도 부족하면, 기존 mesh cluster graph(edges.pt의 kept edge_index --
     --cluster-edges-path, 기본값은 local_nodes.pt와 같은 디렉터리)에서 이
     target의 parent cluster와 실제로 연결된 인접 parent cluster에 한해,
     그 안의 node 중 이 target과 canonical 위치가 가까운 순으로 추가한다.
     mesh cluster graph에 연결되지 않은 cluster의 node는 후보에서 아예
     제외한다.
target별 source 후보는 총 --max-source-candidates개(기본 6)를 넘지 않는다.
이렇게 고른 후보 집합이 최종 edge_index에 반영되므로 edge_index 자체는
시퀀스 내내 고정이고, 프레임마다 달라지는 것은 이 고정된 edge 위에 적용하는
message_source_mask뿐이다.

source 자격(message_source_mask)은 red/orange(target_ok_mask/fallback_mask)
상태와 완전히 분리되어, target 여부와도 무관하게 오직 confidence로만 먼저
확정된다:
    message_source_mask[T, N] := (node_confidence[T, N] >= --frame-context-confidence-threshold)
                                  AND NOT missing_confidence_mask[N]
missing_confidence_mask[n]은 이 노드에 매칭된 raw track이 0개였던 경우다
(compute_node_confidence 참고) -- 그런 노드는 confidence 자체가 없으므로
confidence가 아무리 높게 보여도(0으로 채워져 있을 뿐이다) 영원히 source에서
제외된다. active_context_mask도 정확히 같은 식으로 계산되므로 실질적으로
message_source_mask == active_context_mask다.

target_context_count[T, N] := 매 프레임 이 target의 고정 source 후보
(target_source_candidates, 곧 최종 edge_index로 이 target에 연결된 모든
node) 중 message_source_mask인 것의 개수 -- target인 이웃, non-target인
이웃 가리지 않고 전부 센다(target이 아닌 노드 자신에 대해서는 계산하지
않는다 -- 어차피 쓰이지 않는다). target인 노드가 어떤 프레임에
target_context_count가 --min-context-neighbors(기본 2) 미만이면 그 프레임은
fallback_mask[T, N](주황)에 표시되고(correction 시점에 이웃 context가
부족해 fallback 처리가 필요하다는 뜻), 그 이상이면 target_ok_mask[T, N]
(빨강)이다. fallback_mask/target_ok_mask는 target이 아닌 노드에 대해서는
항상 False다. 어떤 (node, frame)이 fallback됐는지 진단으로 출력한다.

message_source_mask가 confidence만으로 먼저 완전히 확정되고 target_ok_mask/
fallback_mask는 그 결과(target_context_count)로부터 "나중에" 계산되므로,
둘 사이에 순환 의존성이 없다 -- 빨강이든 주황이든 target의 message-source
자격은 오직 자신의 confidence에만 좌우되고, 자신이 빨강인지 주황인지와는
무관하다(그래서 주황 target도 confidence만 높으면 다른 target에게 context
message를 전달할 수 있다).

이 스크립트는 local node 탐색, target/context 분류와 그래프 진단을 한다 -- GNN
correction이나 학습 코드는 포함하지 않는다.

시각화
------
분류 결과를 눈으로 검증할 수 있도록, 실제 렌더된 RGB 위에 각 노드를 위의
MotionScale 기반 node_position[frame_index]에 투영해 점으로 찍는다
(이 파일의 render_local_nodes_overlay_frame과
동일한 model.render + _project_world_points 방식 재사용).
  - 빨강: target, 이 프레임에 min-context-neighbors 이상 확보(정상,
    target_ok_mask)
  - 주황(테두리 포함, 크게): target, 이 프레임에 fallback_mask True(문제
    프레임)
  - 초록: non-target, active_context_mask True (이 상태일 때만 그려짐)
  - (회색은 그리지 않는다: non-target이면서 active_context_mask False인
    노드는 데이터엔 남지만 시각화에서는 완전히 숨긴다 -- 닿는 엣지도 함께)
빨강이든 주황이든, target이 이 프레임에 message_source_mask True면(즉 자신의
confidence가 --frame-context-confidence-threshold(기본 0.5) 이상이고
missing_confidence_mask가 아니면) 다른 target에게 context message를 전달할
수 있다는 뜻으로 초록색 테두리를 추가로 그린다 -- red/orange 여부와 전혀
무관하다(위 순환-의존성 설명 참고). 같은 parent cluster 내부
kNN 엣지도 얇은 선으로 함께 그린다. 문제가 보고된 프레임 위주로 PNG 스냅샷을
자동 선택해 저장하고, 원하면 전체 시퀀스 mp4도 만든다 (기본 동작;
--no-visualization으로 끌 수 있다).

Local node 생성
--------------
    python -m flow3d.graph_local_target --work-dir outputs/davis/spaceout/<run>

가시성 기반 candidate cluster 선택 → 위치/상대 속도 K-means → nearest-center
membership → local_nodes.pt/cluster_visibility.csv 저장 후 target/source 분류로
이어진다. --nodes-only는 node 생성까지만 실행한다. 기존 local_nodes.pt로
분류만 다시 실행하려면 아래 --local-nodes-path 경로를 사용한다.

Example
-------
    python -m flow3d.graph_local_target \\
        --local-nodes-path outputs/davis/spaceout/.../analysis/cluster_graph_mesh/local_nodes.pt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow3d.analysis.build_cluster_graph import load_model_and_clusters, load_raw_track_positions
from flow3d.analysis.build_cluster_graph_mesh import (
    FrameRenderData,
    _id_to_pos_lookup,
    build_cluster_colormap,
    load_frame_snapshot,
    render_frame_mesh,
)
from flow3d.analysis.cluster_pairs import (
    _get_camera_w2cs, _get_dynamic_fg_means, _load_overlay_font, _project_world_points, write_csv,
)

# Local-node discovery and its checkpoint-time preview outputs.
_LOCAL_NODE_SEED_COLOR = (1.0, 0.0, 0.0)
_LOCAL_NODE_DIM_FACTOR = 0.35


@dataclass
class LocalNodeVisibilityConfig:
    tau_o: float
    mask_alpha_threshold: float
    depth_jump_ratio: float
    visibility_depth_tol: float


@dataclass
class LocalNodeConfig:
    local_in_frame_ratio_threshold: float  # in_bounds_count(t)/p90_in_bounds >= this -> "in-frame" frame
    local_occlusion_ratio_threshold: float  # occluded_count(t)/in_bounds_count(t) >= this -> "occluded" frame
    local_occlusion_rate_min: float  # cluster-level occlusion_rate band for candidacy
    local_occlusion_rate_max: float
    local_min_known_frames: int  # min num_in_frame_frames required
    local_min_cluster_gaussians: int
    num_local_nodes: int
    max_local_candidates: int | None
    kmeans_seed: int


@dataclass
class ClusterVisibilityStats:
    cluster_id: int
    num_active_gaussians: int
    num_sampled_frames: int
    p90_in_bounds_count: float
    mean_in_bounds_count: float
    num_in_frame_frames: int
    num_out_of_frame_frames: int
    out_of_frame_rate: float
    mean_occlusion_ratio: float  # averaged over in-frame frames only
    num_occluded_frames: int  # counted over in-frame frames only
    occlusion_rate: float  # num_occluded_frames / num_in_frame_frames
    is_candidate: bool
    num_local_nodes_used: int
    reason: str


@dataclass
class LocalNode:
    parent_cluster_id: int
    node_index: int
    center: np.ndarray  # (3,) canonical, from seed Gaussian
    seed_global_index: int  # checkpoint-only preview field -- see local_nodes.pt meta caveat
    member_global_indices: np.ndarray  # nearest-center reassignment result, checkpoint-only snapshot


def _active_gaussians_by_cluster(
    model: Any, valid_ids: list[int], cfg: LocalNodeVisibilityConfig
) -> dict[int, np.ndarray]:
    """Per-cluster active-Gaussian global indices (opacity > cfg.tau_o), grouped
    by cluster id. Opacity is a static per-Gaussian param in this codebase (not
    time-varying), so this set is identical every frame -- computed once via a
    single frame_index=0 call to load_frame_snapshot rather than per-frame."""
    _, cluster_ids, global_indices = load_frame_snapshot(model, valid_ids, 0, cfg.tau_o)
    result: dict[int, np.ndarray] = {cid: np.zeros(0, dtype=np.int64) for cid in valid_ids}
    for cid in valid_ids:
        result[cid] = global_indices[cluster_ids == cid]
    return result


def _all_active_gaussians_visibility_masks(
    frame_data: FrameRenderData,
    cfg: LocalNodeVisibilityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized sibling of _core_side_visible_fraction: instead of one fixed
    side_cluster_id and a single contact core, tests EVERY active Gaussian in
    frame_data (frame_data.gaussian_positions / gaussian_cluster_ids, already
    opacity+valid-cluster filtered) against its OWN cluster id in one pass --
    used for the local-control-node candidate discovery below, not the
    mesh/contact-core pipeline above (which is untouched).

    :return: (in_bounds, visible), both (G,) bool aligned with
        frame_data.gaussian_global_indices / gaussian_cluster_ids.
        in_bounds is the raw _project_world_points validity mask alone (in
        front of camera, within pixel bounds) -- "on screen" independent of
        occlusion. visible additionally requires alpha/depth/label agreement
        (the same 3 extra conditions as _core_side_visible_fraction). Caller
        derives occluded = in_bounds & ~visible.
    """
    height, width = frame_data.alpha_map.shape
    if frame_data.gaussian_global_indices.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty

    device = frame_data.w2c.device
    points = torch.from_numpy(frame_data.gaussian_positions).to(device=device, dtype=torch.float64)
    w2c64 = frame_data.w2c.to(dtype=torch.float64)
    intrinsic64 = frame_data.intrinsic.to(dtype=torch.float64)
    pixels, in_bounds = _project_world_points(points, w2c64, intrinsic64, width, height)

    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    camera_points = (w2c64 @ homogeneous.T).T[:, :3]
    z_g = camera_points[:, 2].detach().cpu().numpy()

    px = np.clip(np.floor(pixels[:, 0]).astype(np.int64), 0, width - 1)
    py = np.clip(np.floor(pixels[:, 1]).astype(np.int64), 0, height - 1)

    rendered_depth = frame_data.depth_map[py, px]
    rendered_alpha = frame_data.alpha_map[py, px]
    rendered_label = frame_data.vertex_cluster_ids_grid[py, px]

    visible = (
        in_bounds
        & (rendered_alpha > cfg.mask_alpha_threshold)
        & (np.abs(z_g - rendered_depth) <= cfg.visibility_depth_tol * np.maximum(rendered_depth, 1e-8))
        & (rendered_label == frame_data.gaussian_cluster_ids)
    )
    return in_bounds, visible


def compute_cluster_visibility_stats(
    active_by_cluster: dict[int, np.ndarray],
    in_bounds_count_by_cluster: dict[int, list[int]],
    visible_count_by_cluster: dict[int, list[int]],
    sampled_frames: list[int],
    node_cfg: LocalNodeConfig,
) -> dict[int, ClusterVisibilityStats]:
    """Per valid cluster: separate "off-screen" (in_bounds low relative to
    this cluster's own p90 -- an in-bounds-presence problem) from
    "occluded/mismatched while on-screen" (occluded = in_bounds & ~visible,
    restricted to in-frame frames -- the actual articulation signal). Two
    clusters that both have a high raw not-fully-visible rate can tell very
    different stories here: one barely on camera (high out_of_frame_rate,
    occlusion_rate not meaningfully defined) vs. one on camera often but
    self-occluding (low out_of_frame_rate, high occlusion_rate) -- only the
    latter is a good local-node candidate (see select_candidate_clusters).
    """
    num_sampled_frames = len(sampled_frames)
    stats: dict[int, ClusterVisibilityStats] = {}
    for cid, active_indices in active_by_cluster.items():
        num_active = int(active_indices.shape[0])
        in_bounds_counts = np.asarray(in_bounds_count_by_cluster.get(cid, []), dtype=np.float64)
        visible_counts = np.asarray(visible_count_by_cluster.get(cid, []), dtype=np.float64)
        occluded_counts = in_bounds_counts - visible_counts

        p90_in_bounds = float(np.percentile(in_bounds_counts, 90)) if in_bounds_counts.size else 0.0
        mean_in_bounds = float(in_bounds_counts.mean()) if in_bounds_counts.size else 0.0

        if p90_in_bounds < 1e-8:
            stats[cid] = ClusterVisibilityStats(
                cluster_id=cid, num_active_gaussians=num_active,
                num_sampled_frames=num_sampled_frames,
                p90_in_bounds_count=p90_in_bounds, mean_in_bounds_count=mean_in_bounds,
                num_in_frame_frames=0, num_out_of_frame_frames=num_sampled_frames,
                out_of_frame_rate=1.0, mean_occlusion_ratio=0.0,
                num_occluded_frames=0, occlusion_rate=0.0,
                is_candidate=False, num_local_nodes_used=0, reason="never_in_frame",
            )
            continue

        in_frame_ratio = in_bounds_counts / p90_in_bounds
        is_in_frame = in_frame_ratio >= node_cfg.local_in_frame_ratio_threshold
        num_in_frame = int(is_in_frame.sum())
        num_out_of_frame = num_sampled_frames - num_in_frame
        out_of_frame_rate = num_out_of_frame / max(num_sampled_frames, 1)

        if num_in_frame > 0:
            occlusion_ratio = occluded_counts[is_in_frame] / np.maximum(in_bounds_counts[is_in_frame], 1.0)
            is_occluded_frame = occlusion_ratio >= node_cfg.local_occlusion_ratio_threshold
            mean_occlusion_ratio = float(occlusion_ratio.mean())
            num_occluded_frames = int(is_occluded_frame.sum())
            occlusion_rate = num_occluded_frames / num_in_frame
        else:
            mean_occlusion_ratio = 0.0
            num_occluded_frames = 0
            occlusion_rate = 0.0

        if num_active < node_cfg.local_min_cluster_gaussians:
            is_candidate, reason = False, "below_min_gaussians"
        elif num_in_frame < node_cfg.local_min_known_frames:
            is_candidate, reason = False, "insufficient_in_frame_frames"
        elif not (node_cfg.local_occlusion_rate_min <= occlusion_rate <= node_cfg.local_occlusion_rate_max):
            is_candidate, reason = False, "occlusion_rate_out_of_range"
        else:
            is_candidate, reason = True, "candidate"

        stats[cid] = ClusterVisibilityStats(
            cluster_id=cid, num_active_gaussians=num_active,
            num_sampled_frames=num_sampled_frames,
            p90_in_bounds_count=p90_in_bounds, mean_in_bounds_count=mean_in_bounds,
            num_in_frame_frames=num_in_frame, num_out_of_frame_frames=num_out_of_frame,
            out_of_frame_rate=out_of_frame_rate, mean_occlusion_ratio=mean_occlusion_ratio,
            num_occluded_frames=num_occluded_frames, occlusion_rate=occlusion_rate,
            is_candidate=is_candidate, num_local_nodes_used=0, reason=reason,
        )
    return stats


def select_candidate_clusters(
    stats: dict[int, ClusterVisibilityStats], node_cfg: LocalNodeConfig
) -> list[int]:
    """Ranks is_candidate clusters by (-occlusion_rate, -num_in_frame_frames,
    -num_active_gaussians, cluster_id) -- "appears enough on screen AND is
    frequently occluded/mismatched while there" ranked first, cluster_id only
    as a deterministic final tie-break -- and applies --max-local-candidates.
    Clusters cut by the cap are demoted IN PLACE (is_candidate=False,
    reason="capped_by_max_candidates") so cluster_visibility.csv reflects the
    final decision, not just the raw gate.
    """
    candidates = [s for s in stats.values() if s.is_candidate]
    candidates.sort(
        key=lambda s: (-s.occlusion_rate, -s.num_in_frame_frames, -s.num_active_gaussians, s.cluster_id)
    )
    if node_cfg.max_local_candidates is not None and len(candidates) > node_cfg.max_local_candidates:
        kept = candidates[: node_cfg.max_local_candidates]
        cut = candidates[node_cfg.max_local_candidates :]
        for s in cut:
            stats[s.cluster_id] = replace(s, is_candidate=False, reason="capped_by_max_candidates")
        candidates = kept
    return [s.cluster_id for s in candidates]


def build_cluster_visibility_csv_rows(stats: dict[int, ClusterVisibilityStats]) -> list[dict[str, Any]]:
    """Flattens per-cluster stats to cluster_visibility.csv's schema, sorted
    by cluster id for a stable, readable file."""
    rows = []
    for cid in sorted(stats.keys()):
        s = stats[cid]
        rows.append(
            {
                "cluster_id": s.cluster_id,
                "num_active_gaussians": s.num_active_gaussians,
                "num_sampled_frames": s.num_sampled_frames,
                "p90_in_bounds_count": s.p90_in_bounds_count,
                "mean_in_bounds_count": s.mean_in_bounds_count,
                "num_in_frame_frames": s.num_in_frame_frames,
                "num_out_of_frame_frames": s.num_out_of_frame_frames,
                "out_of_frame_rate": s.out_of_frame_rate,
                "mean_occlusion_ratio": s.mean_occlusion_ratio,
                "num_occluded_frames": s.num_occluded_frames,
                "occlusion_rate": s.occlusion_rate,
                "is_candidate": s.is_candidate,
                "num_local_nodes_used": s.num_local_nodes_used,
                "reason": s.reason,
            }
        )
    return rows


def collect_candidate_trajectories(
    model: Any,
    active_by_cluster: dict[int, np.ndarray],
    candidate_ids: list[int],
    sampled_frames: list[int],
) -> dict[int, np.ndarray]:
    """Pose-only pass restricted to candidate clusters' active Gaussians,
    after visibility statistics determine candidacy. Accumulating trajectories
    during the visibility pass would materialize every valid cluster's active
    Gaussians before knowing which clusters need local nodes. model.compute_poses_fg
    computes every fg Gaussian's pose per call regardless of subset, so this
    pass saves nothing on the pose compute itself -- the saving is skipping
    render() entirely and only materializing the (small) candidate subset's
    positions per frame.

    :return: cid -> (T, N_cid, 3) posed positions across sampled_frames.
    """
    if not candidate_ids:
        return {}
    device = model.fg.params["means"].device
    concat_indices = np.concatenate([active_by_cluster[cid] for cid in candidate_ids])
    splits = np.cumsum([active_by_cluster[cid].shape[0] for cid in candidate_ids])[:-1]

    per_frame_rows: list[np.ndarray] = []
    for frame_index in sampled_frames:
        with torch.no_grad():
            means, _ = model.compute_poses_fg(torch.tensor([frame_index], device=device))
        means = means[:, 0, :].detach().double().cpu().numpy()
        per_frame_rows.append(means[concat_indices])

    stacked = np.stack(per_frame_rows, axis=0)  # (T, N_total, 3)
    per_cluster_chunks = np.split(stacked, splits, axis=1)
    return {cid: chunk for cid, chunk in zip(candidate_ids, per_cluster_chunks)}


def assign_gaussians_to_local_nodes(
    canonical_positions: np.ndarray, node_centers: np.ndarray
) -> np.ndarray:
    """Plain nearest-node-center assignment in raw 3D canonical space -- no
    velocity feature, no standardization. This is the RECOMPUTABLE rule: a
    future Local-GNN trainer re-runs this against its own live (post-
    densify/cull) Gaussian population, after filtering to one
    parent_cluster_id's Gaussians and passing that cluster's node_centers --
    unlike seed_global_index/member_global_indices in local_nodes.pt, which
    are only a frozen checkpoint-time snapshot of this same rule (see
    build_local_nodes_payload's meta caveat).
    """
    if node_centers.shape[0] == 0:
        return np.full(canonical_positions.shape[0], -1, dtype=np.int64)
    tree = cKDTree(node_centers)
    _, labels = tree.query(canonical_positions, k=1)
    return np.atleast_1d(labels).astype(np.int64)


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per column, guarded against a genuinely
    zero-variance column (e.g. a near-rigid cluster's velocity column)."""
    if x.size == 0:
        return x
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def build_local_nodes_for_cluster(
    cluster_id: int,
    canonical_positions: np.ndarray,  # (N, 3)
    posed_trajectory: np.ndarray,  # (T, N, 3)
    active_global_indices: np.ndarray,  # (N,)
    node_cfg: LocalNodeConfig,
) -> list[LocalNode]:
    """K-means feature pipeline: per-frame global-motion-removed velocity
    (subtract each frame's cluster-mean velocity, isolating each Gaussian's
    motion RELATIVE to its cluster's own rigid/bulk motion) -> per-block
    (position, velocity) column-wise standardize -> per-block 1/sqrt(dim)
    rebalance (column standardization alone doesn't equalize each block's
    total contribution to Euclidean distance when block widths differ this
    much) -> concat -> kmeans2 -> drop empty clusters + remap node ids ->
    per-node medoid seed (search restricted to that node's OWN members in
    feature space, guaranteeing the seed is a member of its own node) ->
    center = seed's canonical position -> FINAL membership via
    assign_gaussians_to_local_nodes (raw 3D nearest-center reassignment over
    every active Gaussian, NOT the raw kmeans2 labels -- see module
    docstring for why this is the recomputable, densify/cull-safe rule).
    """
    num_active = canonical_positions.shape[0]
    num_frames = posed_trajectory.shape[0]

    if num_frames > 1:
        velocity = np.diff(posed_trajectory, axis=0)  # (T-1, N, 3)
        mean_velocity_per_frame = velocity.mean(axis=1, keepdims=True)  # (T-1, 1, 3)
        velocity_centered = velocity - mean_velocity_per_frame
        velocity_flat = velocity_centered.transpose(1, 0, 2).reshape(num_active, -1)  # (N, (T-1)*3)
    else:
        velocity_flat = np.zeros((num_active, 0), dtype=np.float64)

    pos_std = _standardize_columns(canonical_positions)
    pos_block = pos_std / np.sqrt(pos_std.shape[1]) if pos_std.shape[1] > 0 else pos_std

    if velocity_flat.shape[1] > 0:
        vel_std = _standardize_columns(velocity_flat)
        vel_block = vel_std / np.sqrt(vel_std.shape[1])
    else:
        vel_block = velocity_flat

    features = np.concatenate([pos_block, vel_block], axis=1)

    k = max(1, min(node_cfg.num_local_nodes, num_active))
    centroids, labels = kmeans2(features, k, minit="++", rng=node_cfg.kmeans_seed)

    present_labels = sorted(set(int(l) for l in labels.tolist()))
    label_remap = {old: new for new, old in enumerate(present_labels)}
    labels = np.asarray([label_remap[int(l)] for l in labels], dtype=np.int64)
    k_final = len(present_labels)

    seed_global_by_node: list[int] = []
    center_by_node: list[np.ndarray] = []
    for node_index in range(k_final):
        member_mask = labels == node_index
        member_features = features[member_mask]
        member_global = active_global_indices[member_mask]
        member_canonical = canonical_positions[member_mask]
        tree = cKDTree(member_features)
        _, nearest_local = tree.query(centroids[present_labels[node_index]], k=1)
        seed_global_by_node.append(int(member_global[nearest_local]))
        center_by_node.append(member_canonical[nearest_local])

    node_centers = np.stack(center_by_node, axis=0)  # (k_final, 3)
    final_labels = assign_gaussians_to_local_nodes(canonical_positions, node_centers)

    nodes: list[LocalNode] = []
    for node_index in range(k_final):
        nodes.append(
            LocalNode(
                parent_cluster_id=cluster_id,
                node_index=node_index,
                center=node_centers[node_index],
                seed_global_index=seed_global_by_node[node_index],
                member_global_indices=active_global_indices[final_labels == node_index],
            )
        )
    return nodes


def save_local_nodes_ply(
    active_by_cluster: dict[int, np.ndarray],
    canonical_means: np.ndarray,  # (N_fg, 3)
    color_by_id: dict[int, tuple[float, float, float, float]],
    nodes_by_cluster: dict[int, list[LocalNode]],
    candidate_ids: set[int],
    output_path: Path,
) -> None:
    """Colored canonical point cloud (o3d.geometry.PointCloud): non-candidate
    clusters dimmed to their normal cluster color, candidate clusters colored
    per assigned local node using each node's FINAL, nearest-center-
    reassigned membership (see build_local_nodes_for_cluster) -- not the raw
    kmeans2 labels, so what's drawn matches what's stored in local_nodes.pt.
    Seed Gaussians are recolored to a fixed highlight color last so they
    stand out (a plain point cloud can't vary point size per-point)."""
    node_cmap = plt.get_cmap("tab10")

    all_indices: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for cid, active_indices in active_by_cluster.items():
        if active_indices.size == 0:
            continue
        if cid in candidate_ids:
            colors = np.tile(np.asarray(color_by_id[cid][:3], dtype=np.float64), (active_indices.shape[0], 1))
            index_to_row = {int(g): row for row, g in enumerate(active_indices.tolist())}
            for node in nodes_by_cluster.get(cid, []):
                node_color = np.asarray(node_cmap(node.node_index % 10)[:3])
                rows = [index_to_row[int(g)] for g in node.member_global_indices.tolist()]
                if rows:
                    colors[rows] = node_color
            seed_rows = [
                index_to_row[node.seed_global_index]
                for node in nodes_by_cluster.get(cid, [])
                if node.seed_global_index in index_to_row
            ]
            if seed_rows:
                colors[seed_rows] = np.asarray(_LOCAL_NODE_SEED_COLOR)
        else:
            colors = np.tile(
                np.asarray(color_by_id[cid][:3], dtype=np.float64) * _LOCAL_NODE_DIM_FACTOR,
                (active_indices.shape[0], 1),
            )
        all_indices.append(active_indices)
        all_colors.append(colors)

    if not all_indices:
        return
    indices = np.concatenate(all_indices)
    colors = np.concatenate(all_colors, axis=0)
    points = canonical_means[indices]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), cloud)


def _representative_frame_for_cluster(
    cluster_id: int,
    in_bounds_count_by_cluster: dict[int, list[int]],
    sampled_frames: list[int],
    default_frame_index: int,
) -> int:
    """Sampled frame where this cluster is best represented on screen (max
    in-bounds count) -- same fallback shape as _representative_frame_for_pair."""
    counts = in_bounds_count_by_cluster.get(cluster_id, [])
    if counts:
        best = int(np.argmax(np.asarray(counts)))
        return sampled_frames[best]
    return default_frame_index if default_frame_index in sampled_frames else sampled_frames[0]


def render_local_nodes_overlay_frame(
    model: Any,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    frame_index: int,
    nodes_by_cluster: dict[int, list[LocalNode]],
) -> np.ndarray:
    """One rendered RGB frame + every candidate cluster's Gaussians (via each
    node's FINAL member_global_indices, see build_local_nodes_for_cluster)
    projected and colored by node, members small and seeds larger/
    highlighted -- shared by the PNG and video wrappers below, mirroring
    _render_graph_overlay_frame_mesh."""
    width, height = image_size
    with torch.no_grad():
        render_output = model.render(
            frame_index, w2c[None], intrinsic[None], image_size,
            use_learned_poses=False,
        )
    rgb = render_output["img"][0].detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8).convert("RGBA")
    draw = ImageDraw.Draw(image)

    dynamic_means = _get_dynamic_fg_means(model, frame_index)
    node_cmap = plt.get_cmap("tab10")

    def _draw_points(global_indices: np.ndarray, color: tuple[int, int, int, int], radius: int) -> None:
        if global_indices.size == 0:
            return
        points = dynamic_means[global_indices]
        pixels, valid = _project_world_points(points, w2c, intrinsic, width, height)
        for x, y in pixels[valid]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    for nodes in nodes_by_cluster.values():
        for node in nodes:
            color = tuple(int(round(c * 255)) for c in node_cmap(node.node_index % 10)[:3]) + (200,)
            _draw_points(node.member_global_indices, color, 2)
    for nodes in nodes_by_cluster.values():
        for node in nodes:
            _draw_points(np.asarray([node.seed_global_index], dtype=np.int64), (255, 0, 0, 230), 5)

    return np.asarray(image.convert("RGB"))


def render_local_nodes_png(
    model: Any,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    frame_index: int,
    nodes_by_cluster: dict[int, list[LocalNode]],
    output_path: Path,
) -> Path:
    frame = render_local_nodes_overlay_frame(model, w2c, intrinsic, image_size, frame_index, nodes_by_cluster)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)
    return output_path


def render_local_nodes_video(
    model: Any,
    w2cs: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    sampled_frames: list[int],
    nodes_by_cluster: dict[int, list[LocalNode]],
    output_path: Path,
    fps: int = 10,
    frame_stride: int = 1,
) -> Path:
    """Same overlay as render_local_nodes_png, across sampled_frames (strided
    by frame_stride) instead of a single frame -- mirrors render_2d_overlay_video."""
    frame_indices = sampled_frames[:: max(frame_stride, 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), imageio.get_writer(output_path, fps=fps) as writer:
        for output_index, frame_index in enumerate(frame_indices, start=1):
            frame = render_local_nodes_overlay_frame(
                model, w2cs[frame_index], intrinsics[frame_index], image_size, frame_index, nodes_by_cluster
            )
            writer.append_data(frame)
            print(f"[local-nodes video {output_index:03d}/{len(frame_indices):03d}] frame={frame_index:04d}")
    return output_path


def build_local_nodes_payload(
    nodes_by_cluster: dict[int, list[LocalNode]],
    stats: dict[int, ClusterVisibilityStats],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Assembles local_nodes.pt: a flat node list + per-cluster stats + meta,
    preserving the original dict-of-dicts + meta schema.
    meta carries "member_indices_caveat" -- see its
    text below for what a downstream consumer needs to know."""
    nodes = []
    for cid in sorted(nodes_by_cluster.keys()):
        for node in nodes_by_cluster[cid]:
            nodes.append(
                {
                    "parent_cluster_id": int(node.parent_cluster_id),
                    "node_index": int(node.node_index),
                    "center": torch.from_numpy(np.asarray(node.center, dtype=np.float64)).float(),
                    "seed_global_index": int(node.seed_global_index),
                    "member_global_indices": torch.from_numpy(node.member_global_indices).long(),
                }
            )
    cluster_stats = [
        {
            "cluster_id": s.cluster_id,
            "num_active_gaussians": s.num_active_gaussians,
            "num_sampled_frames": s.num_sampled_frames,
            "p90_in_bounds_count": s.p90_in_bounds_count,
            "mean_in_bounds_count": s.mean_in_bounds_count,
            "num_in_frame_frames": s.num_in_frame_frames,
            "num_out_of_frame_frames": s.num_out_of_frame_frames,
            "out_of_frame_rate": s.out_of_frame_rate,
            "mean_occlusion_ratio": s.mean_occlusion_ratio,
            "num_occluded_frames": s.num_occluded_frames,
            "occlusion_rate": s.occlusion_rate,
            "is_candidate": s.is_candidate,
            "num_local_nodes_used": s.num_local_nodes_used,
            "reason": s.reason,
        }
        for s in (stats[cid] for cid in sorted(stats.keys()))
    ]
    full_meta = dict(meta)
    full_meta["member_indices_caveat"] = (
        "seed_global_index/member_global_indices are a checkpoint-time snapshot for "
        "visualization only -- they index into THIS checkpoint's foreground Gaussian array "
        "and do not survive densification/culling. A training-time consumer must recompute "
        "node membership fresh against its own live Gaussian population by filtering to a "
        "node's parent_cluster_id and calling assign_gaussians_to_local_nodes(live_positions, "
        "node_centers) -- node 'center' (and parent_cluster_id) is the durable, recomputable "
        "reference; the stored indices are not."
    )
    return {"nodes": nodes, "cluster_stats": cluster_stats, "meta": full_meta}


REQUIRED_META_KEYS = (
    "work_dir",
    "checkpoint",
    "num_frames",
    "candidate_cluster_ids",
    "local_min_cluster_gaussians",
)
REQUIRED_NODE_KEYS = ("parent_cluster_id", "node_index", "center")


@dataclass
class LocalNodeRecord:
    global_id: int  # index into the flattened nodes list -- this script's own node id
    parent_cluster_id: int
    node_index: int  # local index within parent_cluster_id, as stored in local_nodes.pt
    center: np.ndarray  # (3,) canonical position, the durable/recomputable reference
    snapshot_member_count: int  # stale checkpoint-time member_global_indices count, diagnostic only


def load_local_nodes(path: Path) -> tuple[list[LocalNodeRecord], dict[str, Any]]:
    """Loads and validates local_nodes.pt's structure. Raises ValueError naming
    exactly which required field is missing rather than filling in a guess."""
    if not path.is_file():
        raise FileNotFoundError(f"local_nodes.pt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "nodes" not in payload or "meta" not in payload:
        raise ValueError(
            f"{path} does not look like a local_nodes.pt payload: expected a dict with "
            f"'nodes' and 'meta' keys, got type={type(payload)}"
            + (f" keys={sorted(payload.keys())}" if isinstance(payload, dict) else "")
        )
    meta = payload["meta"]
    missing_meta = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing_meta:
        raise ValueError(
            f"{path}'s meta is missing required key(s) {missing_meta} -- cannot locate the "
            "checkpoint/dataset or know which clusters were candidates without them. "
            f"Available meta keys: {sorted(meta.keys())}"
        )
    nodes_raw = payload["nodes"]
    if not nodes_raw:
        raise ValueError(f"{path} contains zero nodes; nothing to classify.")

    records: list[LocalNodeRecord] = []
    for i, n in enumerate(nodes_raw):
        missing = [k for k in REQUIRED_NODE_KEYS if k not in n]
        if missing:
            raise ValueError(f"node #{i} in {path} is missing required key(s) {missing}")
        center = np.asarray(n["center"], dtype=np.float64).reshape(3)
        snapshot_count = int(len(n["member_global_indices"])) if "member_global_indices" in n else -1
        records.append(
            LocalNodeRecord(
                global_id=i,
                parent_cluster_id=int(n["parent_cluster_id"]),
                node_index=int(n["node_index"]),
                center=center,
                snapshot_member_count=snapshot_count,
            )
        )
    return records, meta


def compute_node_confidence(
    node_records: list[LocalNodeRecord],
    positions_all_frames: np.ndarray,  # (N_matched, T, 3)
    confidences_all_frames: np.ndarray,  # (N_matched, T)
    global_indices_by_cluster: dict[int, np.ndarray],
    unknown_confidence_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregates raw-track confidence onto each local node. Position is
    NOT computed here -- see compute_node_position, which uses the model's
    own dynamic Gaussian poses instead of raw tracks.

    :return: (node_confidence [T, N] float64, mean_confidence_known [N]
        float64, unknown_ratio [N] float64, matched_track_count [N] int64).
    :raises RuntimeError: for any node with zero matched raw-track points --
        there is genuinely no confidence data to aggregate for it, so this is
        surfaced as an error instead of silently defaulting to 0 or 1.
    """
    T = confidences_all_frames.shape[1]
    num_nodes = len(node_records)
    node_confidence = np.zeros((T, num_nodes), dtype=np.float64)
    mean_confidence_known = np.zeros(num_nodes, dtype=np.float64)
    unknown_ratio = np.zeros(num_nodes, dtype=np.float64)
    matched_track_count = np.zeros(num_nodes, dtype=np.int64)
    missing_confidence_mask = np.zeros(num_nodes, dtype=bool)

    nodes_by_cluster: dict[int, list[LocalNodeRecord]] = {}
    for r in node_records:
        nodes_by_cluster.setdefault(r.parent_cluster_id, []).append(r)

    for cluster_id, recs in nodes_by_cluster.items():
        recs_sorted = sorted(recs, key=lambda r: r.node_index)
        node_centers = np.stack([r.center for r in recs_sorted], axis=0)  # (k, 3)
        matched_rows = global_indices_by_cluster[cluster_id]
        # positions_all_frames[:, 0, :] is the raw-track pool's own frame-0 lift,
        # already nearest-matched to real canonical Gaussian positions by
        # load_raw_track_positions -- reusing it here as the node-assignment
        # position (for CONFIDENCE aggregation only) avoids needing to recover
        # the original fg global index.
        matched_canonical = positions_all_frames[matched_rows, 0, :]
        local_labels = assign_gaussians_to_local_nodes(matched_canonical, node_centers)

        for local_idx, r in enumerate(recs_sorted):
            member_rows = matched_rows[local_labels == local_idx]
            gid = r.global_id
            matched_track_count[gid] = int(member_rows.size)
            if member_rows.size == 0:
                matched_track_count[gid] = 0
                node_confidence[:, gid] = 0.0
                mean_confidence_known[gid] = 0.0
                unknown_ratio[gid] = 1.0
                missing_confidence_mask[gid] = True

                print(
                    f"[graph_relative_local] WARNING: node {gid} "
                    f"(parent_cluster_id={cluster_id}, node_index={r.node_index}) "
                    "has 0 matched raw tracks; treating it as an unknown target "
                    "and excluding it from message sources."
                )
                continue
            member_conf = confidences_all_frames[member_rows]  # (m, T)
            node_confidence[:, gid] = member_conf.mean(axis=0)
            known_mask = member_conf > unknown_confidence_floor
            unknown_ratio[gid] = 1.0 - float(known_mask.mean())
            if known_mask.any():
                mean_confidence_known[gid] = float(member_conf[known_mask].mean())
            else:
                mean_confidence_known[gid] = 0.0

    return node_confidence, mean_confidence_known, unknown_ratio, matched_track_count, missing_confidence_mask


def compute_node_position(
    model: Any,
    cluster_by_id: dict[int, Any],  # cluster_id -> ClusterInfo
    node_records: list[LocalNodeRecord],
    num_frames: int,
) -> np.ndarray:
    """Live node_position[T, N, 3] from MotionScale's own learned motion --
    NOT raw 2D-track lifts (those are used only for confidence, see
    compute_node_confidence). Node membership is recomputed fresh from each
    node's parent cluster's CURRENT canonical Gaussians via
    assign_gaussians_to_local_nodes (the same durable/recomputable
    nearest-center rule local_nodes.pt's own caveat documents), then each
    member's dynamic per-frame position comes from model.compute_poses_fg.
    Applies uniformly to every node, target and context alike.

    :raises RuntimeError: for any node with zero canonical Gaussians
        currently assigned to it -- this checkpoint's Gaussian population no
        longer matches what local_nodes.pt was built from, and there is no
        way to derive a position for that node without guessing.
    """
    device = model.fg.params["means"].device
    num_nodes = len(node_records)

    nodes_by_cluster: dict[int, list[LocalNodeRecord]] = {}
    for r in node_records:
        nodes_by_cluster.setdefault(r.parent_cluster_id, []).append(r)

    member_indices_by_node: dict[int, torch.Tensor] = {}
    for cluster_id, recs in nodes_by_cluster.items():
        cluster = cluster_by_id[cluster_id]
        recs_sorted = sorted(recs, key=lambda r: r.node_index)
        node_centers = np.stack([r.center for r in recs_sorted], axis=0)  # (k, 3)
        canonical_positions_np = cluster.canonical_points.detach().double().cpu().numpy()
        local_labels = assign_gaussians_to_local_nodes(canonical_positions_np, node_centers)

        for local_idx, r in enumerate(recs_sorted):
            member_mask = local_labels == local_idx
            gid = r.global_id
            if not member_mask.any():
                raise RuntimeError(
                    f"node {gid} (parent_cluster_id={cluster_id}, node_index={r.node_index}): "
                    "0 canonical Gaussians in the current checkpoint were assigned to this "
                    "node's region, so no MotionScale-based node_position can be computed for "
                    "it -- refusing to guess. This checkpoint's Gaussian population for this "
                    "cluster likely no longer matches what local_nodes.pt was built from "
                    "(e.g. densify/cull since then)."
                )
            member_local_mask = torch.from_numpy(member_mask).to(cluster.global_indices.device)
            member_indices_by_node[gid] = cluster.global_indices[member_local_mask]

    all_indices = torch.cat([member_indices_by_node[gid] for gid in range(num_nodes)], dim=0)
    member_counts = [int(member_indices_by_node[gid].numel()) for gid in range(num_nodes)]

    ts = torch.arange(num_frames, device=device)
    with torch.no_grad():
        means, _ = model.compute_poses_fg(ts, inds=all_indices)  # (sum_members, T, 3)

    node_position = np.zeros((num_frames, num_nodes, 3), dtype=np.float64)
    offset = 0
    for gid in range(num_nodes):
        n = member_counts[gid]
        node_position[:, gid, :] = means[offset : offset + n].mean(dim=0).detach().double().cpu().numpy()
        offset += n

    return node_position


def build_local_knn_edges(
    node_records: list[LocalNodeRecord], knn_k: int
) -> np.ndarray:
    """Node-level kNN graph, edges restricted to node pairs sharing the same
    parent_cluster_id ("같은 parent cluster 내부에서 local kNN graph"). Returns
    a symmetric (both directions present) (2, E) int64 edge_index."""
    nodes_by_cluster: dict[int, list[LocalNodeRecord]] = {}
    for r in node_records:
        nodes_by_cluster.setdefault(r.parent_cluster_id, []).append(r)

    edge_pairs: set[tuple[int, int]] = set()
    for recs in nodes_by_cluster.values():
        recs_sorted = sorted(recs, key=lambda r: r.node_index)
        ids = [r.global_id for r in recs_sorted]
        n = len(ids)
        if n < 2:
            continue
        centers = np.stack([r.center for r in recs_sorted], axis=0)
        k_eff = min(knn_k, n - 1)
        tree = cKDTree(centers)
        _, idxs = tree.query(centers, k=k_eff + 1)  # rank 0 is always self
        idxs = np.atleast_2d(idxs)
        for i in range(n):
            for rank in range(1, k_eff + 1):
                j = int(idxs[i, rank])
                edge_pairs.add((ids[i], ids[j]))
                edge_pairs.add((ids[j], ids[i]))

    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(edge_pairs), dtype=np.int64).T


def load_cluster_adjacency(path: Path) -> dict[int, set[int]]:
    """Cluster-level adjacency from an existing mesh cluster graph's edges.pt
    (flow3d/analysis/build_cluster_graph_mesh.py's payload -- "edge_index" is
    already just the KEPT edges, symmetric). Used only to let
    select_target_source_candidates cross into an ADJACENT parent cluster
    when a target's own cluster runs out of source candidates -- a cluster
    with no kept edge to a target's cluster is never used, regardless of how
    close its nodes are in space."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "edge_index" not in payload:
        raise ValueError(
            f"{path} does not look like a mesh cluster graph edges.pt payload: expected a "
            f"dict with an 'edge_index' key, got type={type(payload)}"
            + (f" keys={sorted(payload.keys())}" if isinstance(payload, dict) else "")
        )
    edge_index = np.asarray(payload["edge_index"])
    adjacency: dict[int, set[int]] = {}
    for a, b in edge_index.T.tolist():
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    return adjacency


def select_target_source_candidates(
    node_records: list[LocalNodeRecord],
    node_center: np.ndarray,  # (N, 3)
    base_neighbors_by_node: dict[int, list[int]],  # from build_local_knn_edges
    cluster_adjacency: dict[int, set[int]] | None,
    message_source_mask: np.ndarray,  # (T, N) bool, confidence-only -- see main()
    target_mask: np.ndarray,  # (N,)
    goal: int,
    max_candidates: int,
) -> dict[int, list[int]]:
    """Builds each TARGET node's fixed, whole-sequence source-candidate list
    (non-target nodes are not given one -- target_context_count is never
    computed for them). Candidates are ANY other node -- target or
    non-target alike, since message_source_mask (source eligibility) is
    confidence-only and does not depend on target_mask/red/orange status
    (see main()). Three tiers, opened in order and only as far as needed,
    capped at max_candidates total:
      1. this target's existing same-parent-cluster kNN neighbors
         (base_neighbors_by_node, see build_local_knn_edges), closest first;
      2. further same-parent-cluster nodes, next-closest first;
      3. nodes in OTHER parent clusters with a kept edge (cluster_adjacency,
         see load_cluster_adjacency) to this target's own parent cluster,
         closest by canonical position first -- clusters with no such kept
         edge are never used, even if geometrically close.
    A later tier is only opened once the previous tier is exhausted (or the
    cap is hit) AND at least one frame still has fewer than `goal` active
    sources (message_source_mask, computed once over every frame here)
    among the candidates picked so far. Because the whole decision is made
    from message_source_mask across ALL frames at once, the resulting
    candidate list -- and therefore edge_index -- is identical for every
    frame; only message_source_mask itself (unchanged, frame-by-frame)
    decides which of these fixed candidates are actually active in a given
    frame. This is a best-effort fill, not a guarantee: a target may still
    fall short of `goal` (or even of --min-context-neighbors) in some frames
    once max_candidates is exhausted.
    """
    T = message_source_mask.shape[0]
    nodes_by_cluster: dict[int, list[LocalNodeRecord]] = {}
    for r in node_records:
        nodes_by_cluster.setdefault(r.parent_cluster_id, []).append(r)

    ever_active_source = message_source_mask.any(axis=0)
    candidates_by_target: dict[int, list[int]] = {}
    for r in node_records:
        gid = r.global_id
        if not target_mask[gid]:
            continue

        def _dist(oid: int, _center: np.ndarray = r.center) -> float:
            return float(np.linalg.norm(node_center[oid] - _center))


        def _is_valid_source(oid: int) -> bool:
            # Only spend a candidate slot on a node that can ever actually
            # count toward target_context_count -- target or non-target,
            # any node with at least one active (message_source_mask) frame.
            return oid != gid and bool(ever_active_source[oid])


        # 1단계: 기존 same-cluster kNN 중 실제 source가 될 수 있는 node
        tier1_ranked = sorted(
            {
                oid
                for oid in base_neighbors_by_node.get(gid, [])
                if _is_valid_source(oid)
            },
            key=_dist,
        )

        # 2단계: 같은 cluster의 나머지 source
        same_cluster_ranked = sorted(
            (
                o.global_id
                for o in nodes_by_cluster[r.parent_cluster_id]
                if _is_valid_source(o.global_id)
            ),
            key=_dist,
        )

        # 3단계: mesh edge로 연결된 cluster의 source
        adjacent_clusters = (cluster_adjacency or {}).get(
            r.parent_cluster_id, set()
        )

        cross_cluster_ranked = sorted(
            (
                o.global_id
                for cid in adjacent_clusters
                for o in nodes_by_cluster.get(cid, [])
                if _is_valid_source(o.global_id)
            ),
            key=_dist,
        )

        selected: list[int] = []
        seen: set[int] = {gid}

        def _satisfied() -> bool:
            if not selected:
                return T == 0
            counts = message_source_mask[:, selected].sum(axis=1)
            return bool(np.all(counts >= goal))

        for tier in (tier1_ranked, same_cluster_ranked, cross_cluster_ranked):
            if len(selected) >= max_candidates or _satisfied():
                break
            for oid in tier:
                if len(selected) >= max_candidates or _satisfied():
                    break
                if oid in seen:
                    continue
                seen.add(oid)
                selected.append(oid)

        candidates_by_target[gid] = selected

    return candidates_by_target


def merge_target_candidate_edges(
    base_edge_index: np.ndarray,  # (2, E) symmetric, see build_local_knn_edges
    candidates_by_target: dict[int, list[int]],
) -> np.ndarray:
    """Final, whole-sequence-fixed edge_index: base_edge_index (plain
    same-cluster kNN, unchanged, covers every node) unioned with every
    target's selected source-candidate edges (both directions, so the
    within-cluster and cross-cluster "rescue" edges added by
    select_target_source_candidates are visible in the graph/visualization
    the same way the base kNN edges are)."""
    edge_pairs: set[tuple[int, int]] = {(int(a), int(b)) for a, b in base_edge_index.T.tolist()}
    for gid, candidates in candidates_by_target.items():
        for oid in candidates:
            edge_pairs.add((gid, oid))
            edge_pairs.add((oid, gid))
    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(edge_pairs), dtype=np.int64).T


# ---------------------------------------------------------------------------
# Visualization: target/context nodes drawn over the actual rendered RGB
# ---------------------------------------------------------------------------
#
# target_mask and active_context_mask are NOT exclusive (see module
# docstring): a node's dot color is driven purely by target_mask (its fixed
# correction-target role) and fallback_mask (red vs orange, itself derived
# from target_context_count -- see main()). The green ring is NOT tied to
# red/orange at all: it is drawn on ANY target (red or orange) whose own
# message_source_mask is True this frame -- message_source_mask is decided
# purely from that node's own confidence (see main()), before red/orange
# even exists, so a normally-"problem" orange target can still ring and
# forward a context message just like a red one. A non-target node is only
# ever drawn when active, so its green fill alone already says "source" --
# gray/inactive non-target nodes are kept in every saved array but hidden
# here entirely (dot AND any edge touching it).
#
# Colors (RGBA):
_COLOR_TARGET_OK = (235, 50, 50, 255)  # target, NOT in fallback_mask this frame
_COLOR_TARGET_FALLBACK = (255, 140, 0, 255)  # target, fallback_mask True this frame (short of --min-context-neighbors)
_COLOR_CONTEXT_ACTIVE = (60, 200, 90, 255)  # non-target node, active_context_mask True this frame (only state ever drawn for non-targets)
_COLOR_MESSAGE_SOURCE_RING = (60, 200, 90, 255)  # ring around ANY target (red or orange) with message_source_mask True this frame
_COLOR_EDGE = (255, 255, 255, 90)


def render_target_context_overlay_frame(
    model: Any,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    frame_index: int,
    node_records: list[LocalNodeRecord],
    node_position: np.ndarray,  # (T, N, 3)
    target_mask: np.ndarray,  # (N,)
    active_context_mask: np.ndarray,  # (T, N)
    fallback_mask: np.ndarray,  # (T, N)
    message_source_mask: np.ndarray,  # (T, N)
    edge_index: np.ndarray,  # (2, E)
    label_node_ids: bool = False,
) -> np.ndarray:
    """One rendered RGB frame with every node drawn at its MotionScale-posed
    position this frame, colored by its (static) target/context role and,
    for targets, whether fallback_mask flags this frame -- mirrors
    render_local_nodes_overlay_frame
    (same model.render + _project_world_points recipe), but colored by this
    script's classification instead of by node id. Non-target nodes with
    active_context_mask False this frame (gray/inactive) are hidden, along
    with any edge touching a hidden node."""
    width, height = image_size
    with torch.no_grad():
        render_output = model.render(frame_index, w2c[None], intrinsic[None], image_size, use_learned_poses=False)
    rgb = render_output["img"][0].detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    device = w2c.device
    points = torch.from_numpy(node_position[frame_index]).to(device=device, dtype=w2c.dtype)  # (N, 3)
    pixels, valid = _project_world_points(points, w2c, intrinsic, width, height)

    num_nodes = len(node_records)
    # Targets are always drawable; non-targets only when active_context_mask
    # this frame (a gray/inactive dot is hidden, per spec).
    drawable = np.zeros(num_nodes, dtype=bool)
    for r in node_records:
        gid = r.global_id
        drawable[gid] = bool(target_mask[gid]) or bool(active_context_mask[frame_index, gid])

    # kNN edges first, so node dots draw on top. Hidden endpoints hide the edge too.
    for src, dst in edge_index.T:
        if src >= dst:
            continue  # edge_index is symmetric; draw each undirected pair once
        if not (valid[src] and valid[dst] and drawable[src] and drawable[dst]):
            continue
        draw.line(
            [tuple(pixels[src]), tuple(pixels[dst])],
            fill=_COLOR_EDGE, width=1,
        )

    font = _load_overlay_font(11) if label_node_ids else None
    for r in node_records:
        gid = r.global_id
        if not valid[gid] or not drawable[gid]:
            continue
        x, y = pixels[gid]
        is_target = bool(target_mask[gid])
        if is_target:
            in_fallback = bool(fallback_mask[frame_index, gid])
            color = _COLOR_TARGET_FALLBACK if in_fallback else _COLOR_TARGET_OK
            radius = 6 if in_fallback else 4
            if in_fallback:
                draw.ellipse((x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1), outline=(0, 0, 0, 255), width=1)
        else:
            color = _COLOR_CONTEXT_ACTIVE
            radius = 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if is_target and bool(message_source_mask[frame_index, gid]):
            # message_source_mask is confidence-only (see main()) and does not depend on
            # red/orange -- an orange (fallback) target gets this ring exactly like a red one
            # whenever its own confidence clears --frame-context-confidence-threshold.
            ring_r = radius + 4
            draw.ellipse((x - ring_r, y - ring_r, x + ring_r, y + ring_r), outline=_COLOR_MESSAGE_SOURCE_RING, width=2)
        if font is not None:
            draw.text((x + radius + 1, y - radius - 1), str(gid), fill=(255, 255, 255, 230), font=font)

    _draw_legend(draw, _load_overlay_font(13))
    return np.asarray(image.convert("RGB"))


def _draw_legend(draw: ImageDraw.ImageDraw, font: Any) -> None:
    pad, row_h, swatch_r = 6, 16, 5
    entries: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None, str]] = [
        (_COLOR_TARGET_OK, None, "target (ok)"),
        (_COLOR_TARGET_FALLBACK, None, "target (fallback: low context)"),
        (_COLOR_CONTEXT_ACTIVE, None, "active context"),
        (_COLOR_TARGET_OK, _COLOR_MESSAGE_SOURCE_RING, "target: message source (red or orange)"),
    ]
    box_w, box_h = 220, pad * 2 + row_h * len(entries)
    draw.rectangle((4, 4, 4 + box_w, 4 + box_h), fill=(0, 0, 0, 140))
    for i, (fill_color, ring_color, label) in enumerate(entries):
        cy = 4 + pad + row_h * i + row_h // 2
        cx = 4 + pad + swatch_r + (2 if ring_color is not None else 0)
        draw.ellipse((cx - swatch_r, cy - swatch_r, cx + swatch_r, cy + swatch_r), fill=fill_color)
        if ring_color is not None:
            ring_r = swatch_r + 3
            draw.ellipse((cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r), outline=ring_color, width=2)
        draw.text((cx + swatch_r + 9, cy - 7), label, fill=(255, 255, 255, 255), font=font)


def render_target_context_png(
    model: Any,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple[int, int],
    frame_index: int,
    node_records: list[LocalNodeRecord],
    node_position: np.ndarray,
    target_mask: np.ndarray,
    active_context_mask: np.ndarray,
    fallback_mask: np.ndarray,
    message_source_mask: np.ndarray,
    edge_index: np.ndarray,
    output_path: Path,
    label_node_ids: bool = False,
) -> Path:
    frame = render_target_context_overlay_frame(
        model, w2c, intrinsic, image_size, frame_index, node_records, node_position,
        target_mask, active_context_mask, fallback_mask, message_source_mask, edge_index,
        label_node_ids=label_node_ids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)
    return output_path


def render_target_context_video(
    model: Any,
    w2cs: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    frame_indices: list[int],
    node_records: list[LocalNodeRecord],
    node_position: np.ndarray,
    target_mask: np.ndarray,
    active_context_mask: np.ndarray,
    fallback_mask: np.ndarray,
    message_source_mask: np.ndarray,
    edge_index: np.ndarray,
    output_path: Path,
    fps: int = 10,
    label_node_ids: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), imageio.get_writer(output_path, fps=fps) as writer:
        for output_index, frame_index in enumerate(frame_indices, start=1):
            frame = render_target_context_overlay_frame(
                model, w2cs[frame_index], intrinsics[frame_index], image_size, frame_index,
                node_records, node_position, target_mask, active_context_mask, fallback_mask,
                message_source_mask, edge_index, label_node_ids=label_node_ids,
            )
            writer.append_data(frame)
            print(f"[graph_relative_local video {output_index:04d}/{len(frame_indices):04d}] frame={frame_index:04d}")
    return output_path


def select_diagnostic_frames(
    fallback_records: list[dict[str, Any]], num_frames: int, max_frames: int
) -> list[int]:
    """Picks which frames get a PNG snapshot: prioritize frames that actually
    exhibit the reported target-fallback problem (spread across different
    target nodes, not all from the same node), then pad with evenly-spaced
    frames if there's room left or nothing was flagged at all."""
    problem_frames: list[int] = []
    seen: set[int] = set()
    if fallback_records:
        max_per_node = max(1, max_frames // len(fallback_records))
        for rec in fallback_records:
            for f in rec["bad_frames"][:max_per_node]:
                if f not in seen:
                    seen.add(f)
                    problem_frames.append(f)
    problem_frames = sorted(problem_frames)[:max_frames]

    if len(problem_frames) < max_frames:
        num_extra = max_frames - len(problem_frames)
        # Oversample evenly-spaced candidates since some will collide with
        # already-picked problem frames (e.g. many share frame 0 as their
        # first bad frame) -- keep drawing until num_extra new ones are found
        # or the candidate pool (every frame) is exhausted.
        candidates = np.linspace(0, num_frames - 1, num=min(num_frames, max_frames * 8), dtype=int).tolist()
        added = 0
        for f in candidates:
            if f in seen:
                continue
            seen.add(f)
            problem_frames.append(f)
            added += 1
            if added >= num_extra:
                break
    return sorted(problem_frames)


def build_local_nodes(
    args: argparse.Namespace,
    model: Any,
    valid_ids: list[int],
    work_dir: Path,
    checkpoint: Path,
    output_dir: Path,
) -> Path:
    """Discover local nodes from checkpoint visibility and relative motion.

    Reuses the mesh builder's per-frame renderer, but does not build contact
    pairs, boundary patches, or cluster edges. The selection/K-means and
    local_nodes.pt schema are the same as the former mesh-local entry point.
    """
    local_nodes_pt = output_dir / "local_nodes.pt"
    if local_nodes_pt.exists() and not args.overwrite:
        raise FileExistsError(f"{local_nodes_pt} already exists. Pass --overwrite to replace it.")
    if not valid_ids:
        raise ValueError("No valid clusters remain for local-node discovery.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LocalNodeVisibilityConfig(
        tau_o=args.tau_o,
        mask_alpha_threshold=args.mask_alpha_threshold,
        depth_jump_ratio=args.depth_jump_ratio,
        visibility_depth_tol=args.visibility_depth_tol,
    )
    node_cfg = LocalNodeConfig(
        local_in_frame_ratio_threshold=args.local_in_frame_ratio_threshold,
        local_occlusion_ratio_threshold=args.local_occlusion_ratio_threshold,
        local_occlusion_rate_min=args.local_occlusion_rate_min,
        local_occlusion_rate_max=args.local_occlusion_rate_max,
        local_min_known_frames=args.local_min_known_frames,
        local_min_cluster_gaussians=args.local_min_cluster_gaussians,
        num_local_nodes=args.num_local_nodes,
        max_local_candidates=args.max_local_candidates,
        kmeans_seed=args.kmeans_seed,
    )
    device = model.fg.params["means"].device
    w2cs = _get_camera_w2cs(model).to(device)
    intrinsics = model.Ks.to(device)
    principal_x = float(intrinsics[0, 0, 2].item())
    principal_y = float(intrinsics[0, 1, 2].item())
    image_size = (max(int(round(principal_x * 2.0)), 2), max(int(round(principal_y * 2.0)), 2))
    num_frames = model.num_frames
    sampled_frames = list(range(0, num_frames, args.frame_interval))
    color_by_id = build_cluster_colormap(valid_ids)
    active_by_cluster = _active_gaussians_by_cluster(model, valid_ids, cfg)
    in_bounds_count_by_cluster: dict[int, list[int]] = {cid: [] for cid in valid_ids}
    visible_count_by_cluster: dict[int, list[int]] = {cid: [] for cid in valid_ids}
    id_to_pos_arr = _id_to_pos_lookup(valid_ids)
    for i, frame_index in enumerate(sampled_frames, start=1):
        frame_data = render_frame_mesh(
            model, valid_ids, frame_index, w2cs[frame_index], intrinsics[frame_index], image_size, cfg
        )
        in_bounds_mask, visible_mask = _all_active_gaussians_visibility_masks(frame_data, cfg)
        positions_in_valid_order = id_to_pos_arr[frame_data.gaussian_cluster_ids]
        in_bounds_counts_this_frame = np.bincount(
            positions_in_valid_order[in_bounds_mask], minlength=len(valid_ids)
        )
        visible_counts_this_frame = np.bincount(
            positions_in_valid_order[visible_mask], minlength=len(valid_ids)
        )
        for pos, cid in enumerate(sorted(valid_ids)):
            in_bounds_count_by_cluster[cid].append(int(in_bounds_counts_this_frame[pos]))
            visible_count_by_cluster[cid].append(int(visible_counts_this_frame[pos]))
        print(f"[local-nodes visibility {i:03d}/{len(sampled_frames):03d}] frame={frame_index:04d}")

    stats = compute_cluster_visibility_stats(
        active_by_cluster, in_bounds_count_by_cluster, visible_count_by_cluster,
        sampled_frames, node_cfg,
    )
    candidate_ids = select_candidate_clusters(stats, node_cfg)
    nodes_by_cluster: dict[int, list[LocalNode]] = {}
    canonical_means_np = model.fg.params["means"].detach().double().cpu().numpy()

    if candidate_ids:
        trajectories = collect_candidate_trajectories(
            model, active_by_cluster, candidate_ids, sampled_frames
        )
        for cid in candidate_ids:
            active_indices = active_by_cluster[cid]
            canonical_positions = canonical_means_np[active_indices]
            nodes = build_local_nodes_for_cluster(
                cid, canonical_positions, trajectories[cid], active_indices, node_cfg
            )
            nodes_by_cluster[cid] = nodes
            stats[cid] = replace(stats[cid], num_local_nodes_used=len(nodes))
        print(f"[local-nodes] {len(candidate_ids)} candidate cluster(s): {candidate_ids}")
    else:
        print("[local-nodes] no candidate clusters found (see cluster_visibility.csv for reasons)")

    cluster_visibility_csv = output_dir / "cluster_visibility.csv"
    write_csv(cluster_visibility_csv, build_cluster_visibility_csv_rows(stats))
    print(f"[local-nodes] wrote {cluster_visibility_csv}")

    local_nodes_meta = {
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint),
        "num_frames": int(num_frames),
        "min_cluster_size": args.min_cluster_size,
        "tau_o": args.tau_o,
        "mask_alpha_threshold": args.mask_alpha_threshold,
        "visibility_depth_tol": args.visibility_depth_tol,
        "depth_jump_ratio": args.depth_jump_ratio,
        "sampled_frame_indices": sampled_frames,
        "local_in_frame_ratio_threshold": node_cfg.local_in_frame_ratio_threshold,
        "local_occlusion_ratio_threshold": node_cfg.local_occlusion_ratio_threshold,
        "local_occlusion_rate_min": node_cfg.local_occlusion_rate_min,
        "local_occlusion_rate_max": node_cfg.local_occlusion_rate_max,
        "local_min_known_frames": node_cfg.local_min_known_frames,
        "local_min_cluster_gaussians": node_cfg.local_min_cluster_gaussians,
        "num_local_nodes": node_cfg.num_local_nodes,
        "max_local_candidates": node_cfg.max_local_candidates,
        "kmeans_seed": node_cfg.kmeans_seed,
        "candidate_cluster_ids": candidate_ids,
    }
    local_nodes_pt = output_dir / "local_nodes.pt"
    torch.save(build_local_nodes_payload(nodes_by_cluster, stats, local_nodes_meta), local_nodes_pt)
    print(f"[local-nodes] wrote {local_nodes_pt}")

    if candidate_ids and not args.no_visualization and not args.no_local_visualization:
        local_nodes_dir = output_dir / "local_nodes"
        local_nodes_dir.mkdir(parents=True, exist_ok=True)
        save_local_nodes_ply(
            active_by_cluster, canonical_means_np, color_by_id, nodes_by_cluster,
            set(candidate_ids), local_nodes_dir / "canonical_nodes.ply",
        )
        representative_cluster = candidate_ids[0]  # ranked by select_candidate_clusters
        representative_frame = _representative_frame_for_cluster(
            representative_cluster, in_bounds_count_by_cluster, sampled_frames, args.frame_index_2d
        )
        render_local_nodes_png(
            model, w2cs[representative_frame], intrinsics[representative_frame], image_size,
            representative_frame, nodes_by_cluster, local_nodes_dir / "overlay.png",
        )
        print(f"[local-nodes] saved canonical PLY + overlay PNG under {local_nodes_dir}")
        if not args.no_local_video:
            local_video_path = render_local_nodes_video(
                model, w2cs, intrinsics, image_size, sampled_frames, nodes_by_cluster,
                local_nodes_dir / "overlays.mp4", fps=args.video_fps, frame_stride=args.video_frame_stride,
            )
            print(f"[local-nodes] saved overlay video: {local_video_path}")
    return local_nodes_pt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover local control nodes from a checkpoint (or load local_nodes.pt), "
            "then classify nodes into target/context "
            "using aggregated raw-track confidence, and diagnose per-target context "
            "coverage over the local kNN graph."
        ),
        allow_abbrev=False,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--work-dir", "--work_dir", type=Path,
                        help="Discover local nodes from this training run before building the source graph.")
    inputs.add_argument("--local-nodes-path", "--local_nodes_path", type=Path,
                        help="Reuse existing local_nodes.pt and only classify/build the source graph.")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="With --work-dir: default <work-dir>/checkpoints/last.ckpt.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="With --work-dir: default <work-dir>/analysis/cluster_graph_mesh.")
    parser.add_argument("--nodes-only", action="store_true",
                        help="With --work-dir: save local nodes and skip target/source graph construction.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacement of local_nodes.pt when discovering nodes.")
    parser.add_argument("--tau-o", type=float, default=0.1)
    parser.add_argument("--mask-alpha-threshold", type=float, default=0.5)
    parser.add_argument("--visibility-depth-tol", type=float, default=0.05)
    parser.add_argument("--depth-jump-ratio", type=float, default=0.05)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--frame-index-2d", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Default: <local_nodes_path's dir>/graph_relative_local.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="Discovery default: 20. Existing local_nodes.pt default: its "
        "meta.local_min_cluster_gaussians, preserving the existing classification path.",
    )
    parser.add_argument("--raw-track-num-query-frames", type=int, default=20)
    parser.add_argument("--raw-track-samples-per-query", type=int, default=20000)
    parser.add_argument("--raw-track-max-match-distance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--unknown-confidence-floor", type=float, default=1e-6,
        help="A (Gaussian, frame) observation counts as 'unknown' iff its raw-track "
        "confidence is <= this. data/utils.py already zeroes confidence whenever a point is "
        "neither confidently visible nor confidently invisible, so a floor near 0 (not 0.5) "
        "picks out exactly those zeroed/unknown observations.",
    )
    parser.add_argument(
        "--target-confidence-threshold", type=float, default=0.5,
        help="Node-level, static: a node is a target (fixed correction candidate) if its mean "
        "confidence over known (non-unknown) observations is < this, OR its unknown_ratio "
        "exceeds --target-unknown-ratio-threshold. target_mask is NOT the complement of any "
        "context notion -- see --frame-context-confidence-threshold below, which is computed "
        "independently and applies to every node including targets.",
    )
    parser.add_argument(
        "--target-unknown-ratio-threshold", type=float, default=0.3,
        help="Node-level, static: a node is a target if its fraction of unknown observations "
        "over the whole sequence is > this, OR its mean_confidence_known is below "
        "--target-confidence-threshold.",
    )
    parser.add_argument(
        "--frame-context-confidence-threshold", type=float, default=0.5,
        help="The single threshold for ALL confidence-based per-frame source eligibility: "
        "active_context_mask[t, n] := node_confidence[t, n] >= this, computed for EVERY node "
        "regardless of target_mask. message_source_mask uses this exact same threshold (plus "
        "excluding any node with missing_confidence_mask True, i.e. 0 matched raw tracks) -- "
        "source eligibility is decided purely from this, BEFORE red/orange exists, so a target's "
        "own red/orange status never affects whether it can source a message (see main()). "
        "target_context_count -- and therefore red-vs-orange (--min-context-neighbors) -- counts "
        "message_source_mask over EVERY node reachable via the final edge_index, target or not.",
    )
    parser.add_argument("--knn-k", type=int, default=6, help="k for the within-parent-cluster local kNN graph, "
        "and the size of tier 1 of each target's fixed source-candidate list (see "
        "--max-source-candidates / --target-context-goal).")
    parser.add_argument(
        "--min-context-neighbors", type=int, default=2,
        help="Each target node must have at least this many ACTIVE sources (message_source_mask "
        "True, any node -- target or non-target -- reachable via the final edge_index) among "
        "its fixed source-candidate list (see --max-source-candidates) in a given frame to "
        "count as target_ok_mask (red) there, else fallback_mask (orange) is set there and "
        "printed as a diagnostic.",
    )
    parser.add_argument(
        "--target-context-goal", type=int, default=3,
        help="Per-target source-candidate selection heuristic: keep opening tiers (see "
        "select_target_source_candidates / --max-source-candidates) while at least one frame "
        "has fewer than this many active sources (message_source_mask True, target or "
        "non-target) among the candidates picked so far. Best-effort -- selection stops at "
        "--max-source-candidates regardless of whether every frame reached this goal. Distinct "
        "from --min-context-neighbors, which is the (lower) threshold actually used to decide "
        "target_ok_mask vs fallback_mask.",
    )
    parser.add_argument(
        "--max-source-candidates", type=int, default=6,
        help="Hard cap on the total size of each target's fixed source-candidate list, across "
        "all three tiers (own kNN neighbors, further same-cluster nodes, then nodes in "
        "mesh-adjacent parent clusters).",
    )
    parser.add_argument(
        "--cluster-edges-path", type=Path, default=None,
        help="edges.pt from the existing mesh cluster graph (flow3d/analysis/"
        "build_cluster_graph_mesh.py), used for tier 3 of each target's source-candidate list "
        "(nodes in a parent cluster connected to the target's own cluster by a KEPT edge). "
        "Default: <local_nodes_path's dir>/edges.pt if it exists; if not found there, tier 3 "
        "is simply skipped (only same-cluster candidates are used). If this flag is passed "
        "explicitly, the path must exist.",
    )

    parser.add_argument(
        "--no-visualization", action="store_true",
        help="Skip all local-node and target/context visualizations.",
    )
    parser.add_argument(
        "--render-output-dir", type=Path, default=None,
        help="Default: <local_nodes_path's dir>/graph_relative_local_vis",
    )
    parser.add_argument(
        "--render-png-frames", type=int, nargs="*", default=None,
        help="Explicit frame indices to save as overlay PNGs. Default: auto-pick, prioritizing "
        "frames flagged in fallback_mask (a target short of active sources), padded with "
        "evenly-spaced frames up to --render-max-png-frames.",
    )
    parser.add_argument("--render-max-png-frames", type=int, default=6)
    parser.add_argument(
        "--no-video", action="store_true", help="Skip the full-sequence overlay mp4 (PNGs are still saved)."
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-frame-stride", type=int, default=1,
        help="Render every Nth frame for the overlay video (>1 speeds up rendering for long sequences).",
    )
    parser.add_argument(
        "--label-node-ids", action="store_true", help="Draw each node's global id next to its dot.",
    )
    # Local-node discovery settings (used with --work-dir).
    parser.add_argument(
        "--local-in-frame-ratio-threshold", type=float, default=0.3,
        help="A sampled frame counts as 'in-frame' for a cluster if its in-bounds active-"
        "Gaussian count is at least this fraction of that cluster's own p90 in-bounds count "
        "over the sampled sequence -- purely a screen-presence test, independent of occlusion.",
    )
    parser.add_argument(
        "--local-occlusion-ratio-threshold", type=float, default=0.3,
        help="Within an in-frame frame, that frame counts as 'occluded' for a cluster if at "
        "least this fraction of its in-bounds active Gaussians fail the alpha/depth/rendered-"
        "cluster-label visibility test (same 3 conditions as the existing contact-core "
        "visibility check).",
    )
    parser.add_argument(
        "--local-occlusion-rate-min", type=float, default=0.3,
        help="Local-node candidacy band (with --local-occlusion-rate-max): a cluster's "
        "occlusion_rate (fraction of in-frame frames that are 'occluded') must be >= this.",
    )
    parser.add_argument(
        "--local-occlusion-rate-max", type=float, default=0.8,
        help="Local-node candidacy band (with --local-occlusion-rate-min): a cluster's "
        "occlusion_rate must be <= this -- excludes clusters that are essentially always "
        "occluded whenever on-screen (little clean signal) as well as ones that are barely "
        "ever occluded (no self-articulation signal to subdivide on).",
    )
    parser.add_argument(
        "--local-min-known-frames", type=int, default=5,
        help="A cluster needs at least this many IN-FRAME sampled frames to be eligible as a "
        "local-node candidate -- rejects a cluster that's essentially never substantially "
        "on-screen, regardless of its occlusion_rate.",
    )
    parser.add_argument(
        "--local-min-cluster-gaussians", type=int, default=50,
        help="A cluster needs at least this many active (opacity > --tau-o) Gaussians to be "
        "eligible as a local-node candidate -- too few points to meaningfully subdivide.",
    )
    parser.add_argument(
        "--num-local-nodes", type=int, default=8,
        help="Target number of local control nodes to build per candidate cluster via K-means "
        "(clamped down to the cluster's own active-Gaussian count if smaller, and further "
        "reduced if K-means produces an empty cluster -- see --local-min-cluster-gaussians).",
    )
    parser.add_argument(
        "--max-local-candidates", type=int, default=None,
        help="Cap on the number of local-node candidate clusters kept, ranked by "
        "(occlusion_rate desc, num_in_frame_frames desc, num_active_gaussians desc, "
        "cluster_id asc). Default: no cap -- keep every cluster that clears the gates.",
    )
    parser.add_argument(
        "--kmeans-seed", type=int, default=0,
        help="Fixed rng seed for scipy.cluster.vq.kmeans2, so reruns of this script produce "
        "identical local-node membership/seeds/centers.",
    )
    parser.add_argument(
        "--no-local-visualization", action="store_true",
        help="Skip local-node visualization (canonical_nodes.ply, overlay.png/.mp4) even if "
        "--no-visualization is not set.",
    )
    parser.add_argument(
        "--no-local-video", action="store_true",
        help="Skip the local-node overlays.mp4 (the representative-frame overlay.png is still "
        "saved). Independent of --no-video for the target/source overlay.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.local_nodes_path is not None and (
        args.ckpt is not None or args.output_dir is not None or args.nodes_only
    ):
        raise ValueError("--ckpt, --output-dir and --nodes-only require --work-dir.")
    if args.min_cluster_size is not None and args.min_cluster_size < 1:
        raise ValueError("--min-cluster-size must be >= 1")
    if not 0.0 <= args.tau_o < 1.0:
        raise ValueError("--tau-o must be in [0, 1)")
    if not 0.0 < args.mask_alpha_threshold < 1.0:
        raise ValueError("--mask-alpha-threshold must be in (0, 1)")
    if args.visibility_depth_tol <= 0 or args.depth_jump_ratio <= 0:
        raise ValueError("--visibility-depth-tol and --depth-jump-ratio must be > 0")
    if args.frame_interval < 1 or args.video_frame_stride < 1 or args.video_fps < 1:
        raise ValueError("--frame-interval, --video-frame-stride and --video-fps must be >= 1")
    if not 0.0 < args.local_in_frame_ratio_threshold <= 1.0:
        raise ValueError("--local-in-frame-ratio-threshold must be in (0, 1]")
    if not 0.0 < args.local_occlusion_ratio_threshold <= 1.0:
        raise ValueError("--local-occlusion-ratio-threshold must be in (0, 1]")
    if not 0.0 <= args.local_occlusion_rate_min < args.local_occlusion_rate_max <= 1.0:
        raise ValueError(
            "--local-occlusion-rate-min/--local-occlusion-rate-max must satisfy "
            "0 <= min < max <= 1"
        )
    if args.local_min_known_frames < 1:
        raise ValueError("--local-min-known-frames must be >= 1")
    if args.local_min_cluster_gaussians < 1:
        raise ValueError("--local-min-cluster-gaussians must be >= 1")
    if args.num_local_nodes < 1:
        raise ValueError("--num-local-nodes must be >= 1")
    if args.max_local_candidates is not None and args.max_local_candidates < 1:
        raise ValueError("--max-local-candidates must be >= 1 if set")
    if args.knn_k < 1:
        raise ValueError("--knn-k must be >= 1")
    if args.min_context_neighbors < 1:
        raise ValueError("--min-context-neighbors must be >= 1")
    if args.target_context_goal < args.min_context_neighbors:
        raise ValueError(
            "--target-context-goal must be >= --min-context-neighbors (the selection goal "
            "cannot be looser than the target_ok_mask/fallback_mask threshold it feeds)"
        )
    if args.max_source_candidates < args.min_context_neighbors:
        raise ValueError(
            "--max-source-candidates must be >= --min-context-neighbors, otherwise a target "
            "could never reach target_ok_mask even with every candidate active"
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    np.random.seed(args.seed)

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
        checkpoint = (
            args.ckpt.expanduser().resolve() if args.ckpt is not None
            else work_dir / "checkpoints" / "last.ckpt"
        )
        output_dir = (
            args.output_dir.expanduser().resolve() if args.output_dir is not None
            else work_dir / "analysis" / "cluster_graph_mesh"
        )
        args.min_cluster_size = 20 if args.min_cluster_size is None else args.min_cluster_size
        model, clusters, filtered_ids = load_model_and_clusters(
            work_dir, checkpoint, args.device, args.min_cluster_size
        )
        local_nodes_path = build_local_nodes(
            args, model, sorted(c.cluster_id for c in clusters), work_dir, checkpoint, output_dir
        )
        if args.nodes_only:
            return
        node_records, meta = load_local_nodes(local_nodes_path)
        # Reuse the already-loaded model and cluster set.
        min_cluster_size = args.min_cluster_size
    else:
        local_nodes_path = args.local_nodes_path.expanduser().resolve()
        node_records, meta = load_local_nodes(local_nodes_path)
        work_dir = Path(meta["work_dir"]).expanduser().resolve()
        checkpoint = Path(meta["checkpoint"]).expanduser().resolve()
        min_cluster_size = (
            args.min_cluster_size if args.min_cluster_size is not None
            else int(meta["local_min_cluster_gaussians"])
        )
        model, clusters, filtered_ids = load_model_and_clusters(
            work_dir, checkpoint, args.device, min_cluster_size
        )

    num_nodes = len(node_records)
    meta_num_frames = int(meta["num_frames"])
    if num_nodes == 0:
        raise ValueError(
            f"{local_nodes_path} contains no local nodes; adjust discovery thresholds before building the source graph."
        )
    # Discovery/preview should not change the raw-track sampling RNG stream.
    np.random.seed(args.seed)
    needed_cluster_ids = sorted({r.parent_cluster_id for r in node_records})
    print(f"[graph_relative_local] {num_nodes} nodes across parent clusters {needed_cluster_ids}")

    cluster_by_id = {c.cluster_id: c for c in clusters}
    missing_clusters = [cid for cid in needed_cluster_ids if cid not in cluster_by_id]
    if missing_clusters:
        raise RuntimeError(
            f"local_nodes.pt references parent_cluster_id(s) {missing_clusters} that are not "
            f"present as valid (>= --min-cluster-size={min_cluster_size}) clusters in checkpoint "
            f"{checkpoint} (filtered out as too small: {filtered_ids}). The checkpoint may not "
            "match local_nodes.pt's meta.checkpoint, or --min-cluster-size may be stricter than "
            "the local_min_cluster_gaussians used when local_nodes.pt was built."
        )
    needed_clusters = [cluster_by_id[cid] for cid in needed_cluster_ids]

    try:
        positions_all_frames, confidences_all_frames, global_indices_by_cluster, dataset_num_frames = (
            load_raw_track_positions(
                work_dir,
                needed_clusters,
                num_query_frames=args.raw_track_num_query_frames,
                num_samples_per_query=args.raw_track_samples_per_query,
                max_match_distance=args.raw_track_max_match_distance,
            )
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Could not load the raw 2D-track confidence data this script relies on for "
            f"node_confidence -- missing file: {e}. This requires {work_dir}/cfg.yaml plus the "
            "dataset's cached target-track .npy files (see CasualDataset.load_target_tracks); "
            "it will not fabricate confidence values when this data is absent."
        ) from e

    if dataset_num_frames != meta_num_frames:
        raise RuntimeError(
            f"local_nodes.pt's meta.num_frames={meta_num_frames} does not match the dataset's "
            f"actual num_frames={dataset_num_frames} read from {work_dir}/cfg.yaml -- "
            "local_nodes.pt likely no longer corresponds to this work_dir's dataset config."
        )
    T = dataset_num_frames

    (
        node_confidence,
        mean_confidence_known,
        unknown_ratio,
        matched_track_count,
        missing_confidence_mask,
    ) = compute_node_confidence(
        node_records,
        positions_all_frames,
        confidences_all_frames,
        global_indices_by_cluster,
        args.unknown_confidence_floor,
    )
    node_position = compute_node_position(model, cluster_by_id, node_records, T)

    # target_mask: fixed, whole-sequence correction-target label. NOT the
    # complement of context capability -- a target can still be an active
    # context source in frames where it's confident (see active_context_mask
    # below, which every node gets regardless of target_mask).
    target_mask = (
        (mean_confidence_known < args.target_confidence_threshold)
        | (unknown_ratio > args.target_unknown_ratio_threshold)
        | missing_confidence_mask
    )

    active_context_mask = (
        node_confidence >= args.frame_context_confidence_threshold
    ) & ~missing_confidence_mask[None, :]

    print(
        f"[graph_relative_local] {int(target_mask.sum())}/{num_nodes} node(s) are targets "
        f"(mean_confidence_known<{args.target_confidence_threshold} OR "
        f"unknown_ratio>{args.target_unknown_ratio_threshold}); active_context_mask is "
        "computed independently per-frame for every node, targets included."
    )

    base_edge_index = build_local_knn_edges(node_records, args.knn_k)
    base_neighbors_by_node: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    for src, dst in base_edge_index.T:
        base_neighbors_by_node[int(src)].append(int(dst))

    if args.cluster_edges_path is not None:
        cluster_edges_path = args.cluster_edges_path.expanduser().resolve()
        if not cluster_edges_path.is_file():
            raise FileNotFoundError(f"--cluster-edges-path {cluster_edges_path} not found.")
        cluster_adjacency = load_cluster_adjacency(cluster_edges_path)
    else:
        cluster_edges_path = local_nodes_path.parent / "edges.pt"
        if cluster_edges_path.is_file():
            cluster_adjacency = load_cluster_adjacency(cluster_edges_path)
        else:
            cluster_adjacency = None
            print(
                f"[graph_relative_local] no mesh cluster graph found at {cluster_edges_path} -- "
                "tier 3 (cross-parent-cluster source candidates) is unavailable; targets will "
                "only draw from same-parent-cluster nodes."
            )

    # Source eligibility is decided FIRST, from confidence alone, completely independent of
    # target_mask/red/orange -- this is what breaks the circularity: red/orange (below) is
    # computed FROM message_source_mask, so message_source_mask can never be computed from
    # red/orange in turn. Identical to active_context_mask (same threshold, same
    # missing_confidence_mask exclusion for nodes with 0 matched raw tracks); kept as its own
    # name because it now IS the thing target_context_count counts, target or non-target alike.
    message_source_mask = active_context_mask
    node_center = np.stack([r.center for r in node_records], axis=0)
    target_source_candidates = select_target_source_candidates(
        node_records, node_center, base_neighbors_by_node, cluster_adjacency,
        message_source_mask, target_mask, args.target_context_goal, args.max_source_candidates,
    )
    edge_index = merge_target_candidate_edges(base_edge_index, target_source_candidates)

    num_cross_cluster_targets = sum(
        1
        for gid, candidates in target_source_candidates.items()
        if any(
            node_records[oid].parent_cluster_id != node_records[gid].parent_cluster_id
            for oid in candidates
        )
    )
    print(
        f"[graph_relative_local] built fixed source-candidate lists for "
        f"{len(target_source_candidates)} target node(s) (goal={args.target_context_goal} active "
        f"sources/frame, cap={args.max_source_candidates}/target); "
        f"{num_cross_cluster_targets} target(s) needed a tier-3 cross-cluster candidate."
    )

    # target_context_count counts EVERY node reachable via the final edge_index with
    # message_source_mask True this frame -- target or non-target, since source eligibility no
    # longer excludes targets (see message_source_mask above). >= --min-context-neighbors is
    # red (target_ok_mask), below is orange (fallback_mask); this is the ONLY place red/orange
    # is decided, and it is decided strictly after message_source_mask, never the other way.
    target_context_count = np.zeros((T, num_nodes), dtype=np.int64)
    for gid, candidates in target_source_candidates.items():
        if candidates:
            target_context_count[:, gid] = message_source_mask[:, candidates].sum(axis=1)

    fallback_mask = target_mask[None, :] & (target_context_count < args.min_context_neighbors)
    target_ok_mask = target_mask[None, :] & (target_context_count >= args.min_context_neighbors)

    fallback_records: list[dict[str, Any]] = []
    for r in node_records:
        gid = r.global_id
        if not target_mask[gid]:
            continue
        neigh = target_source_candidates.get(gid, [])
        bad_frames = np.nonzero(fallback_mask[:, gid])[0]
        if bad_frames.size > 0:
            fallback_records.append(
                {
                    "node_id": gid,
                    "parent_cluster_id": r.parent_cluster_id,
                    "node_index": r.node_index,
                    "num_available_neighbors": len(neigh),
                    "num_bad_frames": int(bad_frames.size),
                    "bad_frames": bad_frames.tolist(),
                }
            )

    if fallback_records:
        print(
            f"[graph_relative_local] {len(fallback_records)}/{int(target_mask.sum())} target "
            f"node(s) fall back to <--min-context-neighbors={args.min_context_neighbors} "
            "active sources in at least one frame:"
        )
        for rec in fallback_records:
            frames = rec["bad_frames"]
            shown = frames if len(frames) <= 20 else frames[:20] + ["..."]
            structural = rec["num_available_neighbors"] < args.min_context_neighbors
            note = (
                " (STRUCTURAL: fewer source candidates were found across all 3 tiers -- own "
                "kNN neighbors, further same-cluster nodes, mesh-adjacent-cluster nodes -- than "
                "--min-context-neighbors requires at all)"
                if structural
                else ""
            )
            print(
                f"  node {rec['node_id']} (cluster {rec['parent_cluster_id']}, "
                f"local idx {rec['node_index']}, {rec['num_available_neighbors']} source "
                f"candidate(s)){note}: {rec['num_bad_frames']}/{T} frame(s) short -> {shown}"
            )
    else:
        print("[graph_relative_local] every target node has enough active sources in every frame.")

    output_path = args.output.expanduser().resolve() if args.output is not None else local_nodes_path.parent / "graph_relative_local.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_mask": torch.from_numpy(target_mask),
            "active_context_mask": torch.from_numpy(active_context_mask),
            "message_source_mask": torch.from_numpy(message_source_mask),
            "fallback_mask": torch.from_numpy(fallback_mask),
            "node_confidence": torch.from_numpy(node_confidence).float(),
            "edge_index": torch.from_numpy(edge_index),
            "target_context_count": torch.from_numpy(target_context_count),
            "node_parent_cluster_id": torch.tensor([r.parent_cluster_id for r in node_records], dtype=torch.int64),
            "node_index_in_cluster": torch.tensor([r.node_index for r in node_records], dtype=torch.int64),
            "node_center": torch.from_numpy(np.stack([r.center for r in node_records])).float(),
            "mean_confidence_known": torch.from_numpy(mean_confidence_known).float(),
            "unknown_ratio": torch.from_numpy(unknown_ratio).float(),
            "matched_track_count": torch.from_numpy(matched_track_count),
            "missing_confidence_mask": torch.from_numpy(missing_confidence_mask),
            "fallback_records": fallback_records,
            "target_source_candidates": [
                {"node_id": gid, "candidate_ids": candidates}
                for gid, candidates in sorted(target_source_candidates.items())
            ],
            "meta": {
                "local_nodes_path": str(local_nodes_path),
                "work_dir": str(work_dir),
                "checkpoint": str(checkpoint),
                "num_frames": T,
                "min_cluster_size": min_cluster_size,
                "unknown_confidence_floor": args.unknown_confidence_floor,
                "target_confidence_threshold": args.target_confidence_threshold,
                "target_unknown_ratio_threshold": args.target_unknown_ratio_threshold,
                "frame_context_confidence_threshold": args.frame_context_confidence_threshold,
                "knn_k": args.knn_k,
                "min_context_neighbors": args.min_context_neighbors,
                "target_context_goal": args.target_context_goal,
                "max_source_candidates": args.max_source_candidates,
                "cluster_edges_path": str(cluster_edges_path) if cluster_adjacency is not None else None,
                "raw_track_num_query_frames": args.raw_track_num_query_frames,
                "raw_track_samples_per_query": args.raw_track_samples_per_query,
                "raw_track_max_match_distance": args.raw_track_max_match_distance,
            },
        },
        output_path,
    )
    print(f"[graph_relative_local] wrote {output_path}")

    if not args.no_visualization:
        render_output_dir = (
            args.render_output_dir.expanduser().resolve()
            if args.render_output_dir is not None
            else local_nodes_path.parent / "graph_relative_local_vis"
        )
        device = model.fg.params["means"].device
        w2cs = _get_camera_w2cs(model).to(device)
        intrinsics = model.Ks.to(device)
        if w2cs.shape[0] != T or intrinsics.shape[0] != T:
            raise RuntimeError(
                f"model camera count (w2cs={w2cs.shape[0]}, Ks={intrinsics.shape[0]}) does not "
                f"match num_frames={T} used for node_confidence -- the checkpoint and dataset "
                "are out of sync, refusing to guess which frames line up for rendering."
            )
        principal_x = float(intrinsics[0, 0, 2].item())
        principal_y = float(intrinsics[0, 1, 2].item())
        image_size = (max(int(round(principal_x * 2.0)), 2), max(int(round(principal_y * 2.0)), 2))

        if args.render_png_frames is not None:
            bad = [f for f in args.render_png_frames if not (0 <= f < T)]
            if bad:
                raise ValueError(
                    f"--render-png-frames contains out-of-range frame index/indices {bad} "
                    f"(valid range: 0..{T - 1})"
                )
            png_frames = sorted(set(args.render_png_frames))
        else:
            png_frames = select_diagnostic_frames(fallback_records, T, args.render_max_png_frames)

        render_output_dir.mkdir(parents=True, exist_ok=True)
        for f in png_frames:
            out = render_target_context_png(
                model, w2cs[f], intrinsics[f], image_size, f, node_records, node_position,
                target_mask, active_context_mask, fallback_mask, message_source_mask,
                edge_index, render_output_dir / f"overlay_frame{f:04d}.png",
                label_node_ids=args.label_node_ids,
            )
            print(f"[graph_relative_local] saved overlay PNG: {out}")

        if not args.no_video:
            frame_indices = list(range(0, T, max(args.video_frame_stride, 1)))
            video_path = render_target_context_video(
                model, w2cs, intrinsics, image_size, frame_indices, node_records, node_position,
                target_mask, active_context_mask, fallback_mask, message_source_mask,
                edge_index, render_output_dir / "overlay.mp4", fps=args.video_fps,
                label_node_ids=args.label_node_ids,
            )
            print(f"[graph_relative_local] saved overlay video: {video_path}")


if __name__ == "__main__":
    main()

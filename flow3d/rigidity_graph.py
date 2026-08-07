"""
Body-connectivity neighbor graphs for ARAP rigidity.

The default ARAP neighbor graph (see Trainer.update_rigidity_weights) connects
each motion-basis cluster to its k Euclidean-nearest cluster centers. That
graph is wrong whenever two unrelated body parts are spatially close but not
attached (e.g. clasped hands, a hand resting on a knee): the Euclidean
neighbor set for a hand cluster can end up dominated by the *other* hand,
crowding out the actual anchor (that hand's own forearm/wrist), so nothing in
the neighbor set can constrain the hand's identity.

`build_body_connectivity_graph` builds an alternative, fixed-degree neighbor
graph from two independent signals instead:
  1. spatial candidacy -- a Gaussian-level spatial kNN graph on canonical
     positions; two clusters are adjacency *candidates* only if enough
     Gaussian-kNN edges connect their members.
  2. motion consistency -- among spatial candidates, keep only the pairs whose
     center-to-center distance stays stable over time (low coefficient of
     variation), i.e. the two clusters move together rigidly.

A hand and its own forearm satisfy both (adjacent AND rigidly attached). Two
clasped hands satisfy (1) only, since they can move independently -- the CV
gate drops that edge.
"""
from __future__ import annotations

import numpy as np
import torch
from loguru import logger as guru
from scipy.spatial import cKDTree


def build_cluster_spatial_adjacency(
    means_cano: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_clusters: int,
    spatial_k: int = 12,
    min_shared_edges: int = 2,
) -> torch.Tensor:
    """
    Count Gaussian-level spatial-kNN edges crossing each pair of clusters.

    :param means_cano: (N, 3) canonical Gaussian positions.
    :param cluster_ids: (N,) int64 cluster id per Gaussian, aligned to means_cano.
    :param num_clusters: C, total number of clusters.
    :param spatial_k: number of spatial nearest neighbors per Gaussian.
    :param min_shared_edges: pairs with fewer crossing edges than this are
        zeroed out (treated as not adjacent).
    :return: (C, C) int64 symmetric edge-count matrix with the diagonal zero.
    """
    device = means_cano.device
    N = means_cano.shape[0]
    edge_counts = torch.zeros((num_clusters, num_clusters), dtype=torch.int64)
    if N <= 1:
        return edge_counts.to(device)

    k_eff = max(1, min(spatial_k, N - 1))
    means_np = means_cano.detach().cpu().numpy()
    tree = cKDTree(means_np)
    _, knn_idx = tree.query(means_np, k=k_eff + 1)  # (N, k_eff+1), col 0 is self
    knn_idx = np.atleast_2d(knn_idx)

    src = np.repeat(np.arange(N), k_eff)
    dst = knn_idx[:, 1:].reshape(-1)

    cluster_ids_np = cluster_ids.detach().cpu().numpy()
    ci = cluster_ids_np[src]
    cj = cluster_ids_np[dst]
    cross = ci != cj
    ci, cj = ci[cross], cj[cross]
    if ci.size == 0:
        return edge_counts.to(device)

    lo = np.minimum(ci, cj).astype(np.int64)
    hi = np.maximum(ci, cj).astype(np.int64)
    pair_id = lo * num_clusters + hi
    unique_pairs, counts = np.unique(pair_id, return_counts=True)

    a = torch.from_numpy(unique_pairs // num_clusters)
    b = torch.from_numpy(unique_pairs % num_clusters)
    cnt = torch.from_numpy(counts)
    edge_counts[a, b] = cnt
    edge_counts[b, a] = cnt

    edge_counts[edge_counts < min_shared_edges] = 0
    return edge_counts.to(device)


def build_body_connectivity_graph(
    means_cano: torch.Tensor,
    cluster_ids: torch.Tensor,
    centers_ts: torch.Tensor,
    num_clusters: int,
    spatial_k: int = 12,
    min_shared_edges: int = 2,
    cv_threshold: float = 0.05,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a fixed-degree (padded), body-connectivity neighbor graph for ARAP
    rigidity, as an alternative to Euclidean-nearest cluster centers.

    :param means_cano: (N, 3) canonical Gaussian positions.
    :param cluster_ids: (N,) int64 cluster id per Gaussian, aligned to means_cano.
    :param centers_ts: (C, T, 3) cluster center trajectories across all frames.
    :param num_clusters: C.
    :param spatial_k: Gaussian-level spatial kNN size used for adjacency candidacy.
    :param min_shared_edges: minimum shared spatial-kNN edges for two clusters
        to be considered an adjacency candidate ("sufficiently shared").
    :param cv_threshold: a candidate pair is kept as a real neighbor only if
        the coefficient of variation (std/mean) of its center-to-center
        distance over time is below this value (i.e. it moves rigidly).
    :param eps: numerical floor for the CV denominator.
    :return: (knn_idx, valid_mask), both (C, max_degree) on the same device as
        centers_ts. Padding slots use the cluster's own index in knn_idx and
        False in valid_mask, so they carry zero weight downstream.
    """
    device = centers_ts.device
    C = num_clusters

    edge_counts = build_cluster_spatial_adjacency(
        means_cano, cluster_ids, num_clusters,
        spatial_k=spatial_k, min_shared_edges=min_shared_edges,
    ).to(device)
    cand_i, cand_j = torch.nonzero(torch.triu(edge_counts, diagonal=1), as_tuple=True)

    neighbors: list[list[int]] = [[] for _ in range(C)]
    if cand_i.numel() > 0:
        dist_t = torch.norm(centers_ts[cand_i] - centers_ts[cand_j], dim=-1)  # (M, T)
        mean_d = dist_t.mean(dim=-1)
        std_d = dist_t.std(dim=-1, unbiased=False)
        cv = std_d / mean_d.clamp_min(eps)

        keep = cv < cv_threshold
        for i, j in zip(cand_i[keep].tolist(), cand_j[keep].tolist()):
            neighbors[i].append(j)
            neighbors[j].append(i)

    max_degree = max(1, max((len(n) for n in neighbors), default=1))
    knn_idx = torch.arange(C, device=device).unsqueeze(1).repeat(1, max_degree)
    valid_mask = torch.zeros((C, max_degree), dtype=torch.bool, device=device)
    for c, ns in enumerate(neighbors):
        if not ns:
            continue
        knn_idx[c, : len(ns)] = torch.tensor(ns, device=device, dtype=torch.long)
        valid_mask[c, : len(ns)] = True

    return knn_idx, valid_mask


def log_connectivity_graph(
    knn_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    highlight_cluster_ids: list[int] | None = None,
) -> None:
    """
    Log degree distribution and (optionally) specific clusters' neighbor
    lists, for verifying the connectivity graph at init time.
    """
    degrees = valid_mask.sum(dim=1)
    isolated = int((degrees == 0).sum().item())
    guru.info(
        f"[rigidity] connectivity graph: C={knn_idx.shape[0]} max_degree={knn_idx.shape[1]} "
        f"degree(min/median/max)=({degrees.min().item()}/{degrees.float().median().item():.1f}/"
        f"{degrees.max().item()}) isolated_clusters={isolated}"
    )
    for c in (highlight_cluster_ids or range(knn_idx.shape[0])):
        ns = knn_idx[c][valid_mask[c]].tolist()
        guru.info(f"[rigidity] cluster {c} neighbors ({len(ns)}): {ns}")

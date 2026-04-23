import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from flow3d.transforms import cont_6d_to_rmat


def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )

        if not quantile_mask.any():
            return torch.tensor(0.0, device=sum_loss.device)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_cos_loss(pred, gt, mask=None, normalize_vec=True, quantile: float = 1.0):
    """
    Apply cosine similarity loss. Use mask and quantile (quantile is applied after mask) to filter some pixels.
    The implementation is a bit different from other masked losses.
    """
    if normalize_vec:
        pred = F.normalize(pred, dim=-1)
        gt = F.normalize(gt, dim=-1)

    sum_loss = 1 - (pred * gt).sum(dim=-1)
    if mask is None:
        mask = torch.ones_like(sum_loss, dtype=torch.bool)
    else:
        mask = mask.bool()
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        assert mask.shape == sum_loss.shape

    if quantile < 1.0:
        valid_losses = sum_loss[mask]
        n = valid_losses.numel()
        if n > 0:
            if valid_losses.numel() < 16_000_000:
                threshold = torch.quantile(valid_losses, quantile)
            else:
                flat, _ = torch.sort(valid_losses.reshape(-1))
                rank = quantile * (n - 1)
                lo = int(rank)
                hi = min(lo + 1, n - 1)
                w = rank - lo
                threshold = (1 - w) * flat[lo] + w * flat[hi]
            # apply quantile_mask
            mask = mask & (sum_loss <= threshold)

    valid_count = mask.sum()
    if valid_count.item() > 0:
        loss = (sum_loss * mask).sum() / (valid_count + 1e-8)
    else:
        loss = torch.tensor(0.0, device=pred.device)

    return loss


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        # sum_loss.shape
        # block     [39048, 447, 1]     17,454,456
        # apple     [36673, 475, 1]     17,419,675
        # creeper   [37587, 360, 1]     13,531,320
        # backpack  [37828, 180, 1]     6,809,040
        # quantile_mask = (
        #     (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
        #     if quantile < 1
        #     else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        # )
        # use torch.sort instead of torch.quantile when input too large
        if quantile < 1:
            num = sum_loss.numel()
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else:
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        if not quantile_mask.any():
            return torch.tensor(0.0, device=sum_loss.device)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    if not (loss < loss_at_quantile).any():
        return torch.tensor(0.0, device=loss.device)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    if not (loss < loss_at_quantile).any():
        return torch.tensor(0.0, device=loss.device)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def knn_query(x: torch.Tensor, k: int, query=None) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k, algorithm="auto", metric="euclidean"
    ).fit(x)
    query = x if query is None else query.cpu().numpy()
    distances, indices = knn_model.kneighbors(query)
    return distances.astype(np.float32), indices


def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x)
    distances, indices = knn_model.kneighbors(x)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_weights_for_procrustes(clusters, visibilities=None):
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (*, T, 3)
    :param motion_rots (*, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[..., 1:-1, :] - transls[..., :-2, :] - transls[..., 2:, :]
    loss = accel.norm(dim=-1).mean()
    return loss


def compute_se3_reg_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    :param transls (*, 3)
    :param rots (*, 6)
    """
    rotmats = cont_6d_to_rmat(rots)

    I = torch.eye(3, dtype=rots.dtype, device=rots.device)
    rots_loss = ((rotmats - I) ** 2).sum(dim=(-2, -1)).mean()
    transls_loss = (transls ** 2).sum(dim=-1).mean()

    return rots_loss * weight_rot + transls_loss * weight_transl


def cluster_mean(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_clusters: int,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
):
    """
    Compute per-cluster mean of x.
    Args:
        x: (G, ..., D)
        cluster_ids: (G,) int64
        weights: (G, ..., 1) or (G, ...) or None
    Returns:
        mean:   (C, ..., D)
        counts: (C, ..., 1)
    """
    assert cluster_ids.dtype == torch.long
    assert cluster_ids.shape[0] == x.shape[0]
    C = num_clusters
    G, D = x.shape[0], x.shape[-1]
    device, dtype = x.device, x.dtype

    sums = torch.zeros((C,) + x.shape[1:], device=device, dtype=dtype)  # (C, ..., D)
    counts = torch.zeros((C,) + x.shape[1:-1] + (1,), device=device, dtype=dtype)  # (C, ..., 1)

    # Ensure weights shape matches x except last dim D
    if weights is None:
        sums.index_add_(0, cluster_ids, x)
        ones = torch.ones(x.shape[:-1] + (1,), device=device, dtype=dtype)
        counts.index_add_(0, cluster_ids, ones)
    else:
        if weights.dim() == x.dim() - 1:
            weights = weights.unsqueeze(-1)  # (G, ..., 1)
        assert weights.shape[:-1] == x.shape[:-1]
        w = weights.to(dtype)
        sums.index_add_(0, cluster_ids, x * w)
        counts.index_add_(0, cluster_ids, w)

    mean = sums / counts.clamp_min(eps)
    return mean, counts


def center_to_cluster_mean_loss(
    points: torch.Tensor,
    centers: torch.Tensor,
    cluster_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
):
    """
    Generic center-to-cluster-mean loss.

    points, (G, ..., 3): Gaussian means
    centers, (C, ..., 3): cluster centers
    cluster_ids, (G,)
    weights, (G, ...) : optional
    """
    C = centers.shape[0]

    # mean position of each cluster
    mean_pts, counts = cluster_mean(points, cluster_ids, C, weights=weights)  # (C, ..., 3)

    per_center = F.mse_loss(centers, mean_pts, reduction="none").mean(dim=-1)  # (C, ...)
    valid = (counts.squeeze(-1) > 0).float()
    loss = (per_center * valid).sum() / valid.sum().clamp_min(1.0)

    return loss


def compute_arap_distance_loss(centers, ref_t=0, batch_idxs=None, knn_idx=None, k=10, weights=None, detach_ref=True):
    """
    Penalizes changes in pairwise distances between KNN cluster centers.
    Allows articulation (bending at joints) while preventing detachment/stretching.

    Args:
        centers: (C, T, 3) cluster centers for all frames.
        ref_t: (int) Index of the reference frame (default 0).
        batch_idxs: (Tensor, optional) Time indices to compute the loss on. If None, uses all frames besides ref_t.
        knn_idx: (C, k) precomputed KNN indices. Built from ref frame if None.
        k: (int) Number of neighbors (used when knn_idx is None).
        weights: (C, k) per-edge weights.
        detach_ref: (bool) Whether to detach reference distances.

    Returns:
        loss: (Scalar)
    """
    device = centers.device
    C, T, _ = centers.shape

    ref_points = centers[:, ref_t]  # (C, 3)

    if knn_idx is None:
        with torch.no_grad():
            dists = torch.cdist(ref_points, ref_points)
            _, knn_idx = dists.topk(k + 1, largest=False)
            knn_idx = knn_idx[:, 1:]  # remove self (C, k)

    assert knn_idx.ndim == 2
    assert knn_idx.shape[0] == C
    k = knn_idx.shape[1]

    if batch_idxs is None:
        batch_idxs = torch.arange(T, device=device)
        batch_idxs = batch_idxs[batch_idxs != ref_t]

    curr_points = centers[:, batch_idxs].transpose(0, 1)  # (B, C, 3)
    B = curr_points.shape[0]

    if weights is None:
        weights = torch.ones((C, k), dtype=torch.float32, device=device)
    assert weights.shape == knn_idx.shape
    weights = weights.unsqueeze(0).expand(B, -1, -1)  # (B, C, k)

    w_sum = weights.sum(dim=2)  # (B, C)
    valid = w_sum > 1e-6

    # Reference distances
    dist_ref = torch.norm(ref_points.unsqueeze(1) - ref_points[knn_idx], dim=-1)  # (C, k)
    if detach_ref:
        dist_ref = dist_ref.detach()
    dist_ref = dist_ref.unsqueeze(0).expand(B, -1, -1)  # (B, C, k)

    # Current distances
    idx_expanded = knn_idx.unsqueeze(0).expand(B, -1, -1)
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, C, k)
    curr_neighbors = curr_points[batch_indices, idx_expanded]  # (B, C, k, 3)
    dist_curr = torch.norm(curr_points.unsqueeze(2) - curr_neighbors, dim=-1)  # (B, C, k)

    # Difference to the knn distances in reference frame
    diff_sq = (dist_curr - dist_ref) ** 2

    # Weighted mean
    weights = weights * valid.unsqueeze(-1)
    loss = (diff_sq * weights).sum() / (weights.sum() + 1e-6)

    return loss
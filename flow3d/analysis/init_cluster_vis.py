"""
Diagnostic visualization for cluster initialization.

Unlike cluster_pairs.py, none of this needs a trained checkpoint/model: the static
multi-view PNG is built directly from canonical positions and cluster ids, and the
overlay video is built directly from the raw tracked 3D points (tracks_3d.xyz) and the
already-known camera parameters -- no learned motion bases required. Both can run
right after cluster initialization (e.g. cluster_by_motion_affinity in init_utils.py)
to visually check that clusters were separated as expected.
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from matplotlib.pyplot import get_cmap

from flow3d.analysis.cluster_pairs import (
    ClusterInfo,
    _prepare_cluster_visual_data,
    _save_multiview_cluster_png,
)
from flow3d.vis.utils import draw_keypoints_video, project_2d_tracks


def save_cluster_init_png(
    means_cano: torch.Tensor,
    cluster_ids: torch.Tensor,
    output_dir: str,
    filename: str = "clusters_3d.png",
    max_points_per_cluster: int = 2000,
) -> str:
    """
    :param means_cano: (N, 3) canonical positions
    :param cluster_ids: (N,) cluster id per point
    :param output_dir: directory to save the PNG into (created if missing)
    :return: path to the saved PNG
    """
    means_cano = means_cano.detach().cpu()
    cluster_ids = cluster_ids.detach().cpu().long()

    clusters = []
    for cid in cluster_ids.unique().tolist():
        idx = torch.where(cluster_ids == cid)[0]
        points = means_cano[idx]
        clusters.append(
            ClusterInfo(
                cluster_id=int(cid),
                global_indices=idx,
                canonical_points=points,
                canonical_center=points.mean(dim=0),
            )
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    png_path = output_path / filename

    sampled_by_id, center_by_id, color_by_id, all_points = _prepare_cluster_visual_data(
        clusters=clusters,
        max_points_per_cluster=max_points_per_cluster,
    )
    _save_multiview_cluster_png(
        output_path=png_path,
        clusters=clusters,
        sampled_by_id=sampled_by_id,
        center_by_id=center_by_id,
        color_by_id=color_by_id,
        all_points=all_points,
    )
    return str(png_path)


def save_cluster_overlay_video(
    tracks_3d,
    cluster_ids: torch.Tensor,
    imgs: torch.Tensor | np.ndarray,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    output_dir: str,
    filename: str = "clusters_2d_overlay.mp4",
    fps: int = 5,
    radius: int = 3,
) -> str:
    """
    Overlay cluster-colored track points onto the real RGB frames across time, so
    cluster separation can be checked by watching whether same-colored points move
    together or drift apart (e.g. a "hand" cluster sliding independently over a
    "thigh" cluster). Uses the raw tracked 3D points -- no trained model needed, so
    this can run before training even starts.

    :param tracks_3d: TrackObservations with xyz (N, T, 3), visibles (N, T)
    :param cluster_ids: (N,) cluster id per point
    :param imgs: (T, H, W, 3), float [0, 1] or uint8 [0, 255]
    :param Ks: (T, 3, 3) camera intrinsics for the same T frames as tracks_3d/imgs
    :param w2cs: (T, 4, 4) world-to-camera extrinsics for the same T frames
    :return: path to the saved mp4
    """
    device = tracks_3d.xyz.device
    Ks = Ks.to(device)
    w2cs = w2cs.to(device)

    # (T, N, 2) -> (N, T, 2), matching draw_keypoints_video's expected kps layout
    tracks_2d = (
        project_2d_tracks(tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs)
        .swapaxes(0, 1)
        .detach()
        .cpu()
        .numpy()
    )
    occs = (~tracks_3d.visibles.bool()).cpu().numpy()  # (N, T); occluded points drawn hollow

    cluster_ids_np = cluster_ids.detach().cpu().long().numpy()
    cmap = get_cmap("tab20")
    colors = np.stack([np.asarray(cmap(int(c) % cmap.N)[:3]) for c in cluster_ids_np])  # (N, 3)

    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    if imgs.dtype != np.uint8:
        imgs = np.clip(imgs * 255.0, 0, 255).astype(np.uint8)

    frames = draw_keypoints_video(imgs, tracks_2d, colors, occs, radius=radius)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_path = output_path / filename
    iio.imwrite(video_path, np.asarray(frames), fps=fps)
    return str(video_path)

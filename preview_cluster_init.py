#!/usr/bin/env python3
"""
Run only cluster initialization -- no procrustes solve, no optimization, no training,
no checkpoint -- and save a clusters_3d.png plus a clusters_2d_overlay.mp4 so cluster
separation can be checked visually before committing to a full training run. The video
overlays cluster-colored points onto the real RGB frames across time using the raw
tracked 3D points -- no trained model needed, so same-colored points sliding apart
(e.g. a hand over a thigh) can be seen directly.

Uses the exact same TrainConfig CLI/YAML as run_training.py, so all the usual flags
work unchanged, including the motion-affinity ones added alongside "means"/"velocities":
    --cluster-init-type {means,velocities,motion_affinity}
    --affinity-method {agglomerative,components}  (default: agglomerative)
    --affinity-k, --affinity-min-cluster-size
    --affinity-n-clusters          (agglomerative only)
    --affinity-cut-percentile      (components only)

Example:
    python preview_cluster_init.py --config configs/davis/default.yaml \\
        --seq_name bike-packing --cluster-init-type motion_affinity \\
        --affinity-method agglomerative --affinity-k 12 --affinity-n-clusters 40
"""
import os

import torch
from loguru import logger as guru

from flow3d.analysis.init_cluster_vis import save_cluster_init_png, save_cluster_overlay_video
from flow3d.configs import TrainConfig
from flow3d.data import get_train_val_datasets
from flow3d.init_utils import (
    cluster_by_motion_affinity,
    sample_bases_centers_by_means,
    sample_initial_bases_centers,
)
from flow3d.tensor_dataclass import TrackObservations


def main():
    cfg = TrainConfig.build_from_cli()

    train_dataset, _, _, _ = get_train_val_datasets(cfg.data, load_val=False)
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    cano_t = 0
    tracks_3d = TrackObservations(
        *train_dataset.get_tracks_3d(cfg.num_init_samples, start=0, end=cfg.num_init_frames)
    )
    guru.info(f"loaded tracks_3d.xyz.shape={tracks_3d.xyz.shape}")

    # same cano-frame visibility filter used in init_motion_params_with_procrustes
    valid_mask = tracks_3d.visibles[:, cano_t]
    tracks_3d = tracks_3d.filter_valid(valid_mask)
    means_cano = tracks_3d.xyz[:, cano_t].clone()
    guru.info(f"{means_cano.shape[0]} valid tracks on canonical frame {cano_t}")

    if cfg.cluster_init_type == "means":
        _, num_bases, labels = sample_bases_centers_by_means(
            "kmeans", means_cano, cfg.num_motion_bases
        )
    elif cfg.cluster_init_type == "velocities":
        _, num_bases, labels = sample_initial_bases_centers(
            "kmeans", cano_t, tracks_3d, cfg.num_motion_bases
        )
    elif cfg.cluster_init_type == "motion_affinity":
        labels = cluster_by_motion_affinity(
            means_cano, tracks_3d,
            k=cfg.affinity_k,
            cut_percentile=cfg.affinity_cut_percentile,
            min_cluster_size=cfg.affinity_min_cluster_size,
            method=cfg.affinity_method,
            n_clusters=cfg.affinity_n_clusters,
        )
        num_bases = int(labels.max().item()) + 1
    else:
        raise ValueError(f"Invalid cluster_init_type: {cfg.cluster_init_type}")

    guru.info(f"cluster_init_type={cfg.cluster_init_type!r}: {num_bases} clusters")

    output_dir = os.path.join(cfg.work_dir, "analysis", f"{cfg.cluster_init_type}_init_preview")
    png_path = save_cluster_init_png(means_cano, labels, output_dir)
    guru.info(f"Saved cluster preview PNG to {png_path}")

    imgs = torch.stack([train_dataset.get_image(t) for t in range(cfg.num_init_frames)])
    Ks = train_dataset.get_Ks()[: cfg.num_init_frames]
    w2cs = train_dataset.get_w2cs()[: cfg.num_init_frames]
    video_path = save_cluster_overlay_video(tracks_3d, labels, imgs, Ks, w2cs, output_dir)
    guru.info(f"Saved cluster preview video to {video_path}")


if __name__ == "__main__":
    main()

import os
import os.path as osp
import time
from dataclasses import asdict
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from flow3d.configs import TrainConfig
from flow3d.data import (
    BaseDataset,
    get_train_val_datasets,
)
from flow3d.data.base_dataset import CustomBatchSampler, CustomSequentialSampler
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_new_bases,
    optim_new_bases_by_velocity,
    init_bg,
    init_shad_motion,
    init_shad_params,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    run_motion_optim,
    run_bg_optim,
    vis_init_params,
    init_trainable_poses,
    init_new_camera_poses,
)
from flow3d.loss_utils import knn_query
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def _graph_corrected_bases_cls(gnn_variant: str):
    """Resolve which *GraphCorrectedScalableMotionBases class implements
    cfg.gnn_variant ("absolute" -> flow3d/graph_coupling.py's
    GraphCorrectedScalableMotionBases, "relative" -> flow3d/graph_coupling_relative.py's
    RelativeGraphCorrectedScalableMotionBases, "relative_velocity_linear" ->
    flow3d/graph_relative_velocity_linear.py's
    RelativeVelocityLinearGraphCorrectedScalableMotionBases,
    "relative_velocity_angular" -> flow3d/graph_relative_velocity_angular.py's
    RelativeVelocityAngularGraphCorrectedScalableMotionBases,
    "relative_velocity_linear_attention" ->
    flow3d/graph_relative_linear_attention.py's
    RelativeVelLinearAttentionGraphCorrectedScalableMotionBases,
    "relative_velocity_linear_attention_multihead" ->
    flow3d/graph_relative_linear_attention_multihead.py's
    RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases, same
    as the attention variant but with a per-head independent message MLP
    instead of a shared one). All six share the exact same
    constructor/from_scalable_motion_bases signature (modulo the two
    attention variants' extra gnn_num_heads kwarg, see
    _graph_bases_extra_kwargs), so callers can swap the class without
    touching the rest of the wrapping code.
    """
    if gnn_variant == "relative":
        from flow3d.graph_coupling_relative import (
            RelativeGraphCorrectedScalableMotionBases,
        )

        return RelativeGraphCorrectedScalableMotionBases
    elif gnn_variant == "relative_attention":
        from flow3d.graph_coupling_relative_attention import (
            RelativeAttentionGraphCorrectedScalableMotionBases,
        )

        return RelativeAttentionGraphCorrectedScalableMotionBases
    elif gnn_variant == "relative_velocity_linear":
        from flow3d.graph_relative_velocity_linear import (
            RelativeVelocityLinearGraphCorrectedScalableMotionBases,
        )

        return RelativeVelocityLinearGraphCorrectedScalableMotionBases
    elif gnn_variant == "relative_velocity_angular":
        from flow3d.graph_relative_velocity_angular import (
            RelativeVelocityAngularGraphCorrectedScalableMotionBases,
        )

        return RelativeVelocityAngularGraphCorrectedScalableMotionBases
    elif gnn_variant == "relative_velocity_linear_attention":
        from flow3d.graph_relative_linear_attention import (
            RelativeVelLinearAttentionGraphCorrectedScalableMotionBases,
        )

        return RelativeVelLinearAttentionGraphCorrectedScalableMotionBases
    elif gnn_variant == "relative_velocity_linear_attention_multihead":
        from flow3d.graph_relative_linear_attention_multihead import (
            RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases,
        )

        return RelativeVelLinearAttentionMultiHeadGraphCorrectedScalableMotionBases
    elif gnn_variant == "absolute":
        from flow3d.graph_coupling import GraphCorrectedScalableMotionBases

        return GraphCorrectedScalableMotionBases
    else:
        raise ValueError(f"Unknown gnn_variant: {gnn_variant!r}")


def _graph_bases_extra_kwargs(cfg: TrainConfig) -> dict:
    """Extra from_scalable_motion_bases kwargs specific to some gnn_variant
    values (currently just gnn_num_heads for the attention variants -- every
    other variant's from_scalable_motion_bases takes no extra args beyond
    edge_index/gnn_hidden_dim/gnn_num_layers)."""
    if cfg.gnn_variant in {
        "relative_attention",
        "relative_velocity_linear_attention",
        "relative_velocity_linear_attention_multihead",
    }:
        return {"gnn_num_heads": cfg.gnn_heads}
    return {}


def get_git_info() -> str:
    """
    Get the current Git commit hash of the codebase.
    """
    # git hash
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    # git commit message
    commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()

    # check if clean
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]).decode("utf-8").strip()
    is_clean = len(status) == 0

    return commit_hash, commit_msg, is_clean


# Example: python run_training.py --config configs/davis/default.yaml --seq_name bike-packing --exp_name new_run
def main():
    ## Configuration
    # Build train config
    cfg = TrainConfig.build_from_cli()

    # Create output directory with timestamp
    timestr = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    cfg.exp_name = timestr if cfg.exp_name is None else f"{timestr}__{cfg.exp_name}"
    commit_hash, commit_msg, is_clean = get_git_info()
    if cfg.hash:
        # Record commit hash
        cfg.exp_name += f"_{commit_hash}_SHA"

    cfg.work_dir = os.path.join(cfg.work_dir, cfg.exp_name)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Save config
    with open(os.path.join(cfg.work_dir, "cfg.yaml"), "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    ## Data Preparation
    # Load datasets
    train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Model Preparation
    ckpt_path = os.path.join(cfg.work_dir, "checkpoints/last.ckpt") if cfg.ckpt_path is None else cfg.ckpt_path
    load_ckpt_path = initialize_and_checkpoint_model(
        cfg,
        train_dataset,
        device,
        ckpt_path,
        use_2dgs=cfg.use_2dgs,
        vis=cfg.vis_debug,
        port=cfg.port,
    )

    trainer, start_epoch = Trainer.init_from_checkpoint(
        load_ckpt_path,
        device,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
        checkpoint_every=cfg.optim.checkpoint_every_steps,
    )

    custom_batch_sampler = CustomBatchSampler(
        ranges=[(0, cfg.num_init_frames, 0, cfg.num_init_frames)],
        batch_size=cfg.batch_size,
        num_batches=int(np.ceil(cfg.num_init_frames / cfg.batch_size)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=custom_batch_sampler,
        num_workers=cfg.num_dl_workers,
        collate_fn=BaseDataset.train_collate_fn,
    )

    validator = None
    if (
        train_video_view is not None
        or val_img_dataset is not None
        or val_kpt_dataset is not None
    ):
        custom_seq_sampler = CustomSequentialSampler(train_video_view, max_frames=cfg.num_init_frames)
        validator = Validator(
            model=trainer.model,
            device=device,
            train_loader=(
                DataLoader(train_video_view, batch_size=1, sampler=custom_seq_sampler) if train_video_view else None
            ),
            val_img_loader=(
                DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None
            ),
            val_kpt_loader=(
                DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None
            ),
            save_dir=cfg.work_dir,
            save_as_imgs=cfg.save_video_frames,
        )

    ## Training start
    guru.info(f"Training start.")
    num_frames = train_dataset.num_frames
    prop_interval = cfg.track.prop_interval
    prop_epochs = cfg.track.prop_epochs
    warmup_epochs = cfg.track.warmup_epochs

    # Build propagation schedule: frame ranges and their corresponding epoch windows.
    prop_start_frames = [0] + list(range(cfg.num_init_frames, num_frames, prop_interval))
    prop_end_frames = prop_start_frames[1:] + [num_frames]
    prop_start_epochs = [0] + [warmup_epochs + i * prop_epochs for i in range(len(prop_start_frames) - 1)]
    prop_end_epochs = prop_start_epochs[1:] + [prop_start_epochs[-1] + prop_epochs]
    guru.info(f"{prop_start_frames=}")

    # Resume: find which propagation stage start_epoch falls into.
    assert start_epoch >= 0
    start_idx = next((i for i, (a, b) in enumerate(zip(prop_start_epochs, prop_end_epochs)) if a <= start_epoch < b), -1)
    start, end = prop_start_frames[start_idx], prop_end_frames[start_idx]

    # Apply schedule to samplers.
    train_loader.batch_sampler.set_ranges([(0, end, 0, end)])
    train_loader.batch_sampler.set_num_batches(int(np.ceil(end / cfg.batch_size)))
    if validator is not None:
        validator.train_loader.sampler.set_max_frames(end)

    # Apply schedule to trainer.
    num_epochs = prop_end_epochs[-1] + cfg.num_glob_epochs
    trainer.reset_opacity_every = prop_end_epochs[-1] + cfg.reset_opacity_epochs
    trainer.optim_cfg.stop_control_steps = prop_end_epochs[-1] + cfg.stop_control_epochs

    glob_start_epoch = prop_end_epochs[-1]
    trainer.pose_optimize_intervals = sorted(
        [(warmup_epochs // 2, warmup_epochs)]
        + [(s, s + cfg.pose_optim_window) for s in prop_start_epochs[1:]]
        + [(glob_start_epoch, glob_start_epoch + cfg.reset_opacity_epochs)]
    )

    # Training steps
    for epoch in (pbar := tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs)):
        trainer.set_epoch(epoch)
        trainer.update_pose_grad()

        # train step
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = trainer.train_step(batch)
            pbar.set_description(f"Training [{start}, {end}): loss: {loss:.6f}")

        # save checkpoints
        if (epoch + 1) % cfg.optim.checkpoint_every_steps == 0:
            trainer.save_checkpoint(f"{cfg.work_dir}/checkpoints/last.ckpt")
        if cfg.save_more_ckpts and ((epoch + 1) % cfg.save_videos_every == 0):
            trainer.save_checkpoint(f"{cfg.work_dir}/checkpoints/epoch_{epoch:04d}.ckpt")

        if validator is not None:
            if (epoch + 1) % cfg.save_videos_every == 0:
                validator.save_train_videos(epoch)
            if (epoch + 1) % cfg.eval_every == 0 and (epoch + 1) > num_epochs - cfg.eval_last_n_epochs:
                eval_logs = validator.eval_metrics()
                trainer.log_dict({f"eval/{k}": v for k, v in eval_logs.items()})

        # adaptive control
        trainer.control_step(epoch)

        # update between steps
        if (epoch + 1) >= warmup_epochs and (epoch + 1) in prop_start_epochs:
            # update rigidity graph
            trainer.update_rigidity_weights()

            # initialize new background
            prop_end = min(num_frames, end + prop_interval)
            update_bg_model(
                train_dataset, trainer, validator,
                start_frame=end, new_frames=prop_end - end, num_samples=cfg.num_bg_samples,
            )

            # initialize motion on new frames
            for end_frame in range(end + 1, prop_end + 1):
                update_model(
                    train_dataset,
                    trainer,
                    old_frame_end=end,
                    new_frames=1,
                    win_size=15,
                    fg_only=cfg.prop_fg_only,
                    vis=cfg.vis_debug,
                )

                if validator is not None:
                    validator.train_loader.sampler.set_max_frames(end_frame)

            # next step start end
            start = prop_end - prop_interval
            end = prop_end
            train_loader.batch_sampler.set_ranges([(0, end, 0, end)])
            train_loader.batch_sampler.set_num_batches(int(np.ceil(end / cfg.batch_size)))

    ## Finish Training
    # Log final results
    hparam_dict = {
        "Run_Name": cfg.exp_name,
        "Git_Hash": commit_hash,
        "Commit_Message": commit_msg,
        "Is_Clean": is_clean,
    }
    metrics_dict = {f"eval/{k}": v for k, v in eval_logs.items() if "psnr" in k}
    final_metrics = validator.get_metrics()
    metrics_dict.update({f"summary/{k}":v for k, v in final_metrics.items() if "psnr" in k})
    trainer.log_hparams(hparam_dict, metrics_dict)


def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
    use_2dgs: bool,
    vis: bool = False,
    port: int | None = None,
) -> str:
    """
    :return: path main() should actually load via Trainer.init_from_checkpoint.
        Normally just ckpt_path unchanged. The one exception is resuming with
        --enable_graph_coupling from a checkpoint that doesn't have the GNN
        yet (see _wrap_checkpoint_with_graph_coupling below) -- that writes a
        wrapped copy under cfg.work_dir's own checkpoints/ instead of back to
        ckpt_path, so an externally-supplied --ckpt_path source is never
        overwritten, and returns that copy's path.
    """
    if os.path.exists(ckpt_path):
        if cfg.enable_graph_coupling:
            wrapped_path = _wrap_checkpoint_with_graph_coupling(cfg, ckpt_path)
            if wrapped_path is not None:
                return wrapped_path
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return ckpt_path

    fg_params, motion_bases, bg_params, tracks_3d, shad_params, shad_bases = init_model_from_tracks(
        train_dataset,
        cfg.num_fg,
        cfg.num_bg,
        cfg.sample_bg_stride,
        num_samples=cfg.num_init_samples,
        num_init_frames=cfg.num_init_frames,
        bases_type=cfg.bases_type,
        num_motion_bases=cfg.num_motion_bases,
        num_fine_bases=cfg.num_fine_bases,
        cluster_init_type=cfg.cluster_init_type,
        affinity_k=cfg.affinity_k,
        affinity_cut_percentile=cfg.affinity_cut_percentile,
        affinity_min_cluster_size=cfg.affinity_min_cluster_size,
        affinity_method=cfg.affinity_method,
        affinity_n_clusters=cfg.affinity_n_clusters,
        work_dir=cfg.work_dir,
        coefs_type=cfg.optim.coefs_type,
        coefs_sigma=cfg.optim.coefs_sigma,
        vis=vis,
        port=port,
    )

    # run initial optimization
    init_t = cfg.num_init_frames
    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)
    Ks_init, w2cs_init = Ks[:init_t], w2cs[:init_t]
    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks_init, w2cs_init)

    if vis and cfg.port is not None:
        server = get_server(port=cfg.port)
        vis_init_params(server, fg_params, motion_bases)

    if cfg.enable_graph_coupling:
        assert not cfg.optim.enable_bases_control, (
            "enable_graph_coupling requires --optim.no-enable-bases-control: "
            "bases split/cull would remap cluster ids that the graph topology "
            "doesn't know about."
        )
        assert cfg.graph_coupling_path, (
            "enable_graph_coupling=True requires --graph_coupling_path to point "
            "to an edges.pt built by flow3d/analysis/build_cluster_graph.py."
        )
        from flow3d.graph_coupling import build_edge_index_from_edges_pt

        GraphBasesCls = _graph_corrected_bases_cls(cfg.gnn_variant)
        edge_index = build_edge_index_from_edges_pt(
            cfg.graph_coupling_path, num_clusters=motion_bases.num_clusters
        )
        motion_bases = GraphBasesCls.from_scalable_motion_bases(
            motion_bases,
            edge_index=edge_index,
            gnn_hidden_dim=cfg.gnn_hidden,
            gnn_num_layers=cfg.gnn_layers,
            **_graph_bases_extra_kwargs(cfg),
        ).to(device)
        guru.info(
            f"Graph coupling enabled: wrapped motion_bases with "
            f"{GraphBasesCls.__name__} (variant={cfg.gnn_variant}, "
            f"hidden={cfg.gnn_hidden}, layers={cfg.gnn_layers}, "
            f"edges={edge_index.shape[1]}, from {cfg.graph_coupling_path})"
        )

    # Initialize scene model — camera poses only for init frames, new frames added during propagation
    camera_poses = init_trainable_poses(w2cs_init)
    model = SceneModel(Ks, w2cs, fg_params, motion_bases, bg_params, shad_params, shad_bases, camera_poses=camera_poses, use_2dgs=use_2dgs)

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)
    return ckpt_path


def _wrap_checkpoint_with_graph_coupling(cfg: TrainConfig, source_ckpt_path: str) -> str | None:
    """
    Resume support for --enable_graph_coupling starting from a checkpoint
    trained WITHOUT it (plain ScalableMotionBases motion_bases -- e.g. a
    warmup run's last.ckpt): wraps motion_bases in the
    *GraphCorrectedScalableMotionBases class selected by cfg.gnn_variant
    (see _graph_corrected_bases_cls) in place and writes the result to
    cfg.work_dir's own checkpoints/last.ckpt, leaving source_ckpt_path itself
    untouched (it may be an external run's checkpoint that other work still
    depends on).

    The wrap is exactly GraphBasesCls.from_scalable_motion_bases: coarse/fine
    motion params are copied as-is and only the GNN correction is added,
    zero-initialized (both ClusterGraphGNN and RelativeClusterGraphGNN zero
    the head layer in __init__), so it's a no-op the instant training resumes
    -- identical outputs to the source checkpoint until gradients move the
    GNN off zero.

    Optimizer/scheduler state from source_ckpt_path is intentionally dropped:
    it has no entries for the new motion_bases.gnn.* params, and
    Trainer.load_checkpoint_optimizers indexes every one of the resumed
    model's param groups into that dict, so keeping it would KeyError on
    those params. Dropping it means ALL params (not just the GNN's) get a
    fresh Adam optimizer on resume -- epoch/global_step are preserved from
    source_ckpt_path so the epoch-based propagation schedule and checkpoint
    cadence still continue from where it left off.

    :return: path to the wrapped checkpoint, or None if source_ckpt_path is
        already graph-coupled (has "motion_bases.gnn." keys) -- nothing to do,
        SceneModel.init_from_state_dict already restores it correctly as-is.
    """
    assert cfg.graph_coupling_path, (
        "enable_graph_coupling=True requires --graph_coupling_path to point "
        "to an edges.pt built by flow3d/analysis/build_cluster_graph.py."
    )
    assert not cfg.optim.enable_bases_control, (
        "enable_graph_coupling requires --optim.no-enable-bases-control: "
        "bases split/cull would remap cluster ids that the graph topology "
        "doesn't know about."
    )

    ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    if any("motion_bases.gnn." in k for k in state_dict):
        return None

    from flow3d.graph_coupling import build_edge_index_from_edges_pt
    from flow3d.params import ScalableMotionBases

    GraphBasesCls = _graph_corrected_bases_cls(cfg.gnn_variant)
    plain_motion_bases = ScalableMotionBases.init_from_state_dict(
        state_dict, prefix="motion_bases.params."
    )
    edge_index = build_edge_index_from_edges_pt(
        cfg.graph_coupling_path, num_clusters=plain_motion_bases.num_clusters
    )
    model = SceneModel.init_from_state_dict(state_dict)
    model.motion_bases = GraphBasesCls.from_scalable_motion_bases(
        plain_motion_bases,
        edge_index=edge_index,
        gnn_hidden_dim=cfg.gnn_hidden,
        gnn_num_layers=cfg.gnn_layers,
        **_graph_bases_extra_kwargs(cfg),
    )

    target_ckpt_path = os.path.join(cfg.work_dir, "checkpoints/last.ckpt")
    guru.info(
        f"Resume: {source_ckpt_path} has plain motion_bases -- wrapping with "
        f"{GraphBasesCls.__name__} (variant={cfg.gnn_variant}, "
        f"hidden={cfg.gnn_hidden}, layers={cfg.gnn_layers}, "
        f"edges={edge_index.shape[1]}, from "
        f"{cfg.graph_coupling_path}; correction zero-initialized) and saving "
        f"to {target_ckpt_path} (source left untouched, epoch="
        f"{ckpt.get('epoch', 0)}, global_step={ckpt.get('global_step', 0)} "
        f"preserved, optimizer/scheduler state dropped -- fresh optimizer "
        f"for all params)."
    )
    os.makedirs(os.path.dirname(target_ckpt_path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": ckpt.get("epoch", 0),
            "global_step": ckpt.get("global_step", 0),
        },
        target_ckpt_path,
    )
    return target_ckpt_path


def init_model_from_tracks(
    train_dataset,
    num_fg: int,
    num_bg: int,
    sample_stride: int,
    num_samples: int,
    num_init_frames: int,
    bases_type: str,
    num_motion_bases: int,
    num_fine_bases: int,
    cluster_init_type: str,
    coefs_type: str,
    coefs_sigma: float,
    affinity_k: int = 12,
    affinity_cut_percentile: float = 85.0,
    affinity_min_cluster_size: int = 20,
    affinity_method: str = "agglomerative",
    affinity_n_clusters: int = 40,
    work_dir: str | None = None,
    vis: bool = False,
    port: int | None = None,
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_samples, start=0, end=num_init_frames))
    num_fg = tracks_3d.xyz.shape[0]
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    rot_type = "6d"
    cano_t = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d, cluster_ids = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t,
        cluster_init_type=cluster_init_type, bases_type=bases_type, num_fine_bases=num_fine_bases,
        coefs_type=coefs_type, coefs_sigma=coefs_sigma, vis=vis, port=port,
        affinity_k=affinity_k, affinity_cut_percentile=affinity_cut_percentile,
        affinity_min_cluster_size=affinity_min_cluster_size, affinity_method=affinity_method,
        affinity_n_clusters=affinity_n_clusters, work_dir=work_dir,
        train_dataset=train_dataset,
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs, cluster_ids)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.sample_bkgd_points(start=0, end=num_init_frames, down_rate=4))
        assert bg_points.check_sizes()
        num_bg = len(bg_points.xyz)
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")

    # initialize shadows
    if train_dataset.shadows is not None:
        shad_masks = torch.stack([train_dataset.get_shadow(i) for i in range(num_init_frames)], dim=0)

        shad_points = StaticObservations(*train_dataset.get_bkgd_points(
            num_samples=None, start=0, end=num_init_frames, sample_masks=shad_masks,
            use_kf_tstamps=False, stride=1, down_rate=4, return_lists=True,
        ))

        shad_bases, shad_coefs, shad_cluster_ids = init_shad_motion(
            shad_points, 5, rot_type, cano_t,
            cluster_init_type=cluster_init_type, bases_type=bases_type, num_fine_bases=num_fine_bases,
        )
        shad_params = init_shad_params(cano_t, shad_points, shad_coefs, shad_cluster_ids)
        shad_params = shad_params.to(device)
        shad_bases = shad_bases.to(device)

        guru.info(f"{shad_params.num_gaussians=}, {shad_bases.num_clusters=}")
    else:
        shad_params, shad_bases = None, None

    # save pcd as ply file
    if num_bg > 0:
        fg_xyz = tracks_3d.xyz[:, cano_t].numpy()
        fg_colors = tracks_3d.colors.numpy()
        bg_xyz = bg_points.xyz.numpy()
        bg_colors = bg_points.colors.numpy()
        write_ply(osp.join(train_dataset.cache_dir, 'fg_pcd.ply'), fg_xyz, fg_colors)
        write_ply(osp.join(train_dataset.cache_dir, 'bg_pcd.ply'), bg_xyz, bg_colors)
    else:
        save_dir = osp.join(train_dataset.cache_dir, 'init_pcd.ply')
        pcd_points = tracks_3d.xyz[:, cano_t].numpy()
        pcd_colors = tracks_3d.colors.numpy()
        write_ply(save_dir, pcd_points, pcd_colors)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d, shad_params, shad_bases


def update_bg_model(
    train_dataset,
    trainer,
    validator,
    start_frame: int,
    new_frames: int,
    num_samples: int,
):
    # sample new gaussians and update the model
    update_model_by_sampling(
        train_dataset, trainer,
        num_samples=num_samples, start_frame=start_frame, end_frame=start_frame + new_frames, vis=False,
    )

    # new frame shadow bases
    if trainer.model.has_shad:
        new_bases = init_new_bases(trainer.model.shad_bases, new_frames)
        trainer._add_shad_bases(new_bases)

    # create new frame camera poses (before run_bg_optim, merged after)
    new_poses = None
    if trainer.model.camera_poses is not None:
        new_poses = init_new_camera_poses(trainer, train_dataset, new_frames)

    # run a basic optimization on background (new poses injected temporarily if optimize_poses)
    run_bg_optim(trainer, train_dataset, new_frames=new_frames, optimize_poses=True, new_poses=new_poses)

    # merge new camera poses after optimization
    if new_poses is not None:
        trainer._add_camera_poses(new_poses)


def update_model_by_sampling(
    train_dataset,
    trainer,
    num_samples: int,
    start_frame: int = 0,
    end_frame: int = -1,
    radius: int = 5,
    vis: bool = False,
):
    if end_frame == -1:
        end_frame = train_dataset.num_frames

    # sample new background points
    bg_points = trainer.model.bg.params["means"].detach().cpu()
    new_points = StaticObservations(*train_dataset.sample_bkgd_points(
        start=start_frame, end=end_frame, bg_input=bg_points, down_rate=4, erode_radius=radius,
    ))
    if new_points.xyz.shape[0] == 0:
        return

    if vis:
        save_dir = osp.join(trainer.work_dir, "pcds")
        os.makedirs(save_dir, exist_ok=True)
        write_ply(os.path.join(save_dir, f"{start_frame:04d}_sampled.ply"), new_points.xyz.numpy(), new_points.colors.numpy())

        bg_means = trainer.model.bg.params["means"].detach().cpu().numpy()
        bg_colors = trainer.model.bg.get_colors().detach().cpu().numpy()
        write_ply(os.path.join(save_dir, f"{start_frame:04d}_bg.ply"), bg_means, bg_colors)

    # initialize background gaussians from sampled points
    bg_scales = trainer.model.bg.params["scales"].detach().cpu()
    _, indices = knn_query(bg_points, k=3, query=new_points.xyz)
    new_scales = bg_scales[indices].mean(dim=1)
    new_params = init_bg(new_points, init_scales=new_scales)

    # update gs params
    trainer.add_control_step(fg_params=None, bg_params=new_params)


def update_model(
    train_dataset,
    trainer,
    old_frame_end: int,
    new_frames: int,
    win_size: int,
    fg_only: bool = False,
    vis: bool = False,
):
    # initialize new frame bases
    new_bases = init_new_bases(trainer.model.motion_bases, new_frames)
    optim_new_bases_by_velocity(new_bases, trainer.model)

    # update motion bases
    trainer._add_motion_bases(new_bases)

    # run initial optim
    run_motion_optim(trainer, train_dataset, old_frame_end, new_frames, win_size, fg_only=fg_only)


def write_ply(filename, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)


if __name__ == "__main__":
    main()
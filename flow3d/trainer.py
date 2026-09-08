import functools
import random
import time
from dataclasses import asdict
from typing import cast
import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from pytorch_msssim import SSIM
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fnc
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from flow3d.configs import LossesConfig, OptimConfig, SceneLRConfig
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
    masked_cos_loss,
    compute_se3_reg_loss,
    center_to_cluster_mean_loss,
    compute_arap_distance_loss,
)
from flow3d.analysis.loss import (
    gnn_correction_magnitude_loss,
    gnn_correction_smoothness_loss,
    gnn_correction_edge_consistency_loss,
)
from flow3d.analysis.loss_joint import joint_anchor_loss
from flow3d.analysis.loss_joint_gnn_only import (
    JointAnchorBoundarySets,
    build_joint_anchor_boundary_sets,
    transform_joint_anchor_boundary_sets_gnn_only,
)
from flow3d.graph_relative_edge import (
    EdgeBoundaryGraphCorrectedScalableMotionBases,
    boundary_magnitude_reg_loss,
    boundary_magnitude_smoothness_loss,
    compute_boundary_gap_loss,
)
from flow3d.graph_relative_linear_attention_boundary import (
    RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases,
    compute_edge_patch_gap_loss,
)
from flow3d.data.utils import compute_depth_normal_mask, to_device
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import get_server
from flow3d.vis.viewer import DynamicViewer
from flow3d.init_utils import cluster_by_velocities, init_motion_params_for_split
from flow3d.rigidity_graph import build_body_connectivity_graph, log_connectivity_graph
from sklearn.cluster import AgglomerativeClustering


def capture_rng_state() -> dict:
    """
    Snapshot every RNG stream that training-loop randomness (e.g. the
    per-batch frame sampling in CustomBatchSampler.__iter__, which draws from
    np.random) can come from, so an exact resume replays the same sequence of
    random draws as an uninterrupted run.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Inverse of capture_rng_state(). No-op if state is None/empty (e.g. a
    checkpoint saved before this field existed, or a fresh from-scratch init)."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        log_every: int = 10,
        checkpoint_every: int = 200,
        validate_every: int = 500,
        validate_video_every: int = 1000,
        validate_viewer_assets_every: int = 100,
    ):
        self.device = device
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.scheduler = self.configure_optimizers()

        # running stats for adaptive density control
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(self.model.num_gaussians, device=device),
            "vis_count": torch.zeros(
                self.model.num_gaussians, device=device, dtype=torch.int64
            ),
            "max_radii": torch.zeros(self.model.num_gaussians, device=device),
        }
        self.knn_idx, self.rigid_weights = None, None
        # cached body-connectivity graph (rigidity_graph_type="connectivity"):
        # built once from the fixed cluster assignment, never rebuilt during training.
        self.connectivity_knn_idx, self.connectivity_valid_mask = None, None
        # cached offline cluster graph (rigidity_graph_type="cluster_graph_file"):
        # loaded once from optim_cfg.rigidity_graph_path, never rebuilt during training.
        self.cluster_graph_knn_idx, self.cluster_graph_valid_mask = None, None
        # cached joint anchor boundary sets (see
        # flow3d/analysis/loss_joint_gnn_only.py): loaded once from
        # optim_cfg.joint_anchor_path, never rebuilt during training.
        self.joint_anchor_boundary_sets: JointAnchorBoundarySets | None = None

        self.work_dir = work_dir
        self.writer = SummaryWriter(log_dir=work_dir)
        self.global_step = 0
        self.epoch = 0
        self.pose_optimize_intervals: list[tuple[int, int]] = []  # [(start, end), ...], empty = never optimize
        # RNG state loaded from a checkpoint (see init_from_checkpoint), held
        # here until restore_pending_rng_state() is called. Deliberately not
        # applied immediately: model/trainer/DataLoader construction itself
        # consumes RNG draws (e.g. weight init for any freshly-added params),
        # so applying it too early would perturb those instead of just the
        # training loop's own draws.
        self.pending_rng_state: dict | None = None

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="training"
            )

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()
        self.bg_psnr_metric = mPSNR()
        self.fg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.fg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS()
        self.fg_lpips_metric = mLPIPS()

    @property
    def num_frames(self) -> int:
        return self.model.num_frames

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_pose_grad(self, optimize: bool):
        if self.model.camera_poses is None:
            return
        for p in self.model.camera_poses.parameters():
            p.requires_grad = optimize

    def update_pose_grad(self):
        if self.model.camera_poses is None or not self.pose_optimize_intervals:
            return
        optimize = any(start <= self.epoch < end for start, end in self.pose_optimize_intervals)
        self.set_pose_grad(optimize)

    def save_checkpoint(self, path: str, resume_epoch: int | None = None):
        """
        :param resume_epoch: the epoch a resumed run should start at (i.e.
            the *next* epoch to train, not the one just finished). Callers
            that save mid-epoch (e.g. a legacy/manual snapshot) can omit this
            to fall back to self.epoch; run_training.py's main loop always
            passes epoch+1, since it only calls this after that epoch's
            training, control_step, and propagation update have all
            completed -- so a resumed run starts clean at the next epoch
            instead of redoing (part of) this one.
        """
        epoch_to_save = self.epoch if resume_epoch is None else resume_epoch
        model_dict = self.model.state_dict()
        optimizer_dict = {k: v.state_dict() for k, v in self.optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in self.scheduler.items()}
        ckpt = {
            "model": model_dict,
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": self.global_step,
            "epoch": epoch_to_save,
            "control_state": self._get_control_state(),
            "rng_state": capture_rng_state(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)
        guru.info(f"Saved checkpoint at {self.global_step=} epoch={epoch_to_save} to {path}")

    def _get_control_state(self) -> dict:
        """
        Everything adaptive density control (densify/cull) and the rigidity
        cache accumulate between checkpoints that isn't captured by the model
        state_dict itself -- without this, a resumed run would restart
        density control with empty gradient/visibility stats (an
        under-informed, noisy first control decision) and rebuild the
        rigidity graph from scratch instead of reusing the cached one.
        """
        def _cpu(t):
            return t.detach().cpu() if t is not None else None

        return {
            "running_stats": {k: _cpu(v) for k, v in self.running_stats.items()},
            "knn_idx": _cpu(self.knn_idx),
            "rigid_weights": _cpu(self.rigid_weights),
            "connectivity_knn_idx": _cpu(self.connectivity_knn_idx),
            "connectivity_valid_mask": _cpu(self.connectivity_valid_mask),
            "cluster_graph_knn_idx": _cpu(self.cluster_graph_knn_idx),
            "cluster_graph_valid_mask": _cpu(self.cluster_graph_valid_mask),
        }

    def _load_control_state(self, state: dict | None) -> None:
        if not state:
            return

        def _dev(t):
            return t.to(self.device) if t is not None else None

        running_stats = state.get("running_stats")
        if running_stats:
            for k, v in running_stats.items():
                self.running_stats[k] = _dev(v)
        self.knn_idx = _dev(state.get("knn_idx"))
        self.rigid_weights = _dev(state.get("rigid_weights"))
        self.connectivity_knn_idx = _dev(state.get("connectivity_knn_idx"))
        self.connectivity_valid_mask = _dev(state.get("connectivity_valid_mask"))
        self.cluster_graph_knn_idx = _dev(state.get("cluster_graph_knn_idx"))
        self.cluster_graph_valid_mask = _dev(state.get("cluster_graph_valid_mask"))

    def restore_pending_rng_state(self) -> None:
        """
        Apply the RNG state loaded by init_from_checkpoint (if any). Call
        this once, right before the training loop starts -- after model,
        trainer, and DataLoader construction are all done, since those
        themselves draw from the RNG and would be perturbed by restoring
        earlier.
        """
        if self.pending_rng_state is not None:
            restore_rng_state(self.pending_rng_state)
            guru.info("Restored RNG state (python/numpy/torch CPU+CUDA) from checkpoint.")
            self.pending_rng_state = None

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, weights_only=False)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        trainer = Trainer(model, device, *args, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer._load_control_state(ckpt.get("control_state"))
        trainer.pending_rng_state = ckpt.get("rng_state")
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def load_checkpoint_optimizers(self, opt_ckpt):
        missing = [k for k in self.optimizers if k not in opt_ckpt]
        for k, v in self.optimizers.items():
            if k in opt_ckpt:
                v.load_state_dict(opt_ckpt[k])
        if missing:
            guru.info(
                f"[resume] No optimizer state in checkpoint for {len(missing)} param(s) "
                f"(new params added since the checkpoint was saved, e.g. GNN wrapping): "
                f"{missing} -- initialized fresh."
            )

    def load_checkpoint_schedulers(self, sched_ckpt):
        missing = [k for k in self.scheduler if k not in sched_ckpt]
        for k, v in self.scheduler.items():
            if k in sched_ckpt:
                v.load_state_dict(sched_ckpt[k])
        if missing:
            guru.info(
                f"[resume] No scheduler state in checkpoint for {len(missing)} param(s): "
                f"{missing} -- initialized fresh."
            )

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = 0
        if self.viewer is not None:
            t = (
                int(self.viewer._playback_guis[0].value)
                if not self.viewer._canonical_checkbox.value
                else None
            )
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh, use_learned_poses=False)["img"][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss, stats, num_rays_per_step, num_rays_per_sec = self.compute_losses(batch)

        extra_loss, extra_stats = self.compute_extra_view_losses(batch["ts"])
        loss = loss + extra_loss
        stats.update(extra_stats)

        if loss.isnan():
            guru.info(f"Loss is NaN at step {self.global_step}!!")
            import ipdb

            ipdb.set_trace()

        loss.backward()

        for opt_name, opt in self.optimizers.items():
            opt.step()
            opt.zero_grad(set_to_none=True)

        for sched_name, sched in self.scheduler.items():
            sched.step()

        self.log_dict(stats)
        self.global_step += 1
        self._prepare_control_step()

        if self.viewer is not None:
            self.viewer.lock.release()
            self.viewer.state.num_train_rays_per_sec = num_rays_per_sec
            if self.viewer.mode == "training":
                self.viewer.update(self.global_step, num_rays_per_step)

        return loss.item()

    def compute_extra_view_losses(self, ts: torch.Tensor):
        """
        Photometric-only (RGB + mask + depth) supervision from extra camera
        views (e.g. other cameras in a multi-view rig), in addition to the
        primary view's full compute_losses(). No 2D-track/normal losses here
        -- extra views don't have (or need) CoTracker-based motion init, and
        gaussian poses at a given timestep don't depend on which camera
        renders them, so we just re-render the same means/quats used for the
        primary view from each extra view's camera.

        A no-op (returns (0.0, {})) unless self.extra_view_datasets has been
        set externally (see run_training.py) -- keeps single-view training
        completely unaffected when multi-view isn't requested.
        """
        extra_view_datasets = getattr(self, "extra_view_datasets", None)
        if not extra_view_datasets:
            return 0.0, {}

        self.model.training = True
        device = ts.device
        B = ts.shape[0]

        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)

        loss = 0.0
        rgb_loss_sum, mask_loss_sum, depth_loss_sum = 0.0, 0.0, 0.0
        n_terms = 0
        for view_name, ds in extra_view_datasets.items():
            for i in range(B):
                idx = ts[i].item()
                sample = to_device(
                    {
                        "img": ds.get_image(idx),
                        "fg_mask": ds.get_fg_mask(idx).float(),
                        "depth": ds.get_depth(idx),
                        "depth_mask": ds.get_depth_mask(idx),
                        "w2c": ds.w2cs[idx],
                        "K": ds.Ks[idx],
                    },
                    device,
                )
                H, W = sample["img"].shape[:2]
                bg_color = torch.ones(1, 3, device=device)
                rendered = self.model.render(
                    idx,
                    sample["w2c"][None],
                    sample["K"][None],
                    (W, H),
                    means=means[i],
                    quats=quats[i],
                    bg_color=bg_color,
                    return_depth=True,
                )

                rgb_loss = 0.8 * F.l1_loss(rendered["img"], sample["img"][None]) + 0.2 * (
                    1 - self.ssim(
                        rendered["img"].permute(0, 3, 1, 2),
                        sample["img"][None].permute(0, 3, 1, 2),
                    )
                )
                mask_loss = F.mse_loss(rendered["acc"], sample["fg_mask"][None, ..., None])

                pred_disp = 1.0 / (cast(torch.Tensor, rendered["depth"]) + 1e-5)
                tgt_disp = 1.0 / (sample["depth"][None, ..., None] + 1e-5)
                depth_loss = masked_l1_loss(
                    pred_disp, tgt_disp,
                    mask=sample["depth_mask"][None, ..., None],
                    quantile=0.98,
                )

                view_loss = (
                    rgb_loss * self.losses_cfg.w_rgb
                    + mask_loss * self.losses_cfg.w_mask
                    + depth_loss * self.losses_cfg.w_depth_reg
                )
                loss = loss + view_loss
                rgb_loss_sum += rgb_loss.item()
                mask_loss_sum += mask_loss.item()
                depth_loss_sum += depth_loss.item()
                n_terms += 1

        loss = loss / n_terms * self.losses_cfg.w_multiview
        stats = {
            "train/multiview_rgb_loss": rgb_loss_sum / n_terms,
            "train/multiview_mask_loss": mask_loss_sum / n_terms,
            "train/multiview_depth_loss": depth_loss_sum / n_terms,
        }
        return loss, stats

    def compute_bg_losses(self, batch):
        """
        Compute losses on background only.
        """
        self.model.training = True
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W)
        depth_masks = batch.get("depth_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]
        # [(N, P), ...].
        target_track_masks = batch["target_track_masks"]
        _tic = time.time()
        # device = means.device
        device = w2cs.device
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                bg_color=bg_color,
                return_depth=True,
                bg_only=True,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        # Compute losses.
        # mask images
        bg_masks = valid_masks * (1.0 - masks)
        imgs = (
            imgs * bg_masks[..., None]
            + (1.0 - bg_masks[..., None]) * bg_colors[:, None, None]
        )

        ## RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        rendered_imgs = (
            rendered_imgs * bg_masks[..., None]
            + (1.0 - bg_masks[..., None]) * bg_colors[:, None, None]
        )

        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb

        ## Acc loss
        rendered_acc = cast(torch.Tensor, rendered_all["acc"])
        rendered_acc = rendered_acc * bg_masks[..., None]
        acc_loss = F.mse_loss(rendered_acc, bg_masks[..., None])
        loss += acc_loss * self.losses_cfg.w_mask

        ## Depth loss on the current frame
        depth_masks = depth_masks * valid_masks if self.model.has_bg else depth_masks * masks
        depth_masks = depth_masks[..., None]
        depth_masks *= bg_masks[..., None]

        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
        loss += depth_loss * self.losses_cfg.w_depth_reg

        ## Difference in depth between adjacent pixels <==> that of gt
        #  depth_gradient_loss = 0.0
        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )
        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/acc_loss": acc_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
        }

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec

    def compute_motion_losses(self, batch, fg_only=False):
        self.model.training = True
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]
        # [(N, P), ...].
        target_track_masks = batch["target_track_masks"]
        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts) if not fg_only else self.model.compute_poses_fg(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        # [(N, G, 3), ...].
        target_ts_vec = torch.cat(target_ts)
        # (B * N, G, 3).
        target_means, _ = self.model.compute_poses_all(target_ts_vec) if not fg_only else self.model.compute_poses_fg(target_ts_vec)
        target_means = target_means.transpose(0, 1)
        target_mean_list = target_means.split(N)  # (N, G, 3) x B
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            # bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_color=False,
                fg_only=fg_only,
            )
            rendered_all.append(rendered)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }

        # Compute losses.
        # (B * N).
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0)

        ## 2D track loss for targets
        # (B * N, H * W, 3).
        pred_tracks_3d = (
            rendered_all["tracks_3d"].permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3)  # type: ignore
        )
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
        )  # w2c is applied before render
        # (B * N, H * W, 1).
        mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

        # (B * N).
        w_interval = torch.exp(-2 * frame_intervals / num_frames)
        # w_track_loss = min(1, (self.max_steps - self.global_step) / 6000)

        # (B, H, W).
        masks_flatten = torch.zeros_like(masks)

        # get masks and weights for the 2d track points
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0

        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )

        # Add weights for the loss on tracks: confidence; penalty on distant frames
        track_weights = []
        track_depth_weights = []
        for i in range(B):
            weight = target_confidences[i] * w_interval[i: i + N].unsqueeze(-1)
            track_weights.append(weight.reshape(-1))
            track_depth_weights.append(target_track_masks[i].reshape(-1))
        track_weights = torch.cat(track_weights, dim=0).unsqueeze(-1)
        track_depth_weights = torch.cat(track_depth_weights, dim=0).unsqueeze(-1)

        track_2d_loss = masked_l1_loss(
            pred_tracks_2d[masks_flatten][visibles],
            tracks_2d[visibles],
            mask=track_weights[visibles],
            quantile=0.98,
        ) / max(H, W)
        loss += track_2d_loss * self.losses_cfg.w_track

        ## Depth loss for track targets
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        mapped_depth_loss = masked_l1_loss(
            1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
            1 / (mapped_depth_gt[visibles, None] + 1e-5),
            mask=track_weights[visibles] * track_depth_weights[visibles],
            quantile=0.98,
        )

        loss += mapped_depth_loss * self.losses_cfg.w_depth_const

        ## Align coarse transformed gaussian means
        loss_coarse_align = torch.tensor(0.0)
        num_fg = self.model.num_fg_gaussians
        positions = target_means.transpose(0, 1)[:num_fg, 0:1].detach()  # (G, 1, 3)
        transfms_coarse = self.model.compute_transforms_coarse(target_ts[0])
        pos_coarse = torch.einsum(
            "pnij,pj->pni",
            transfms_coarse,
            F.pad(self.model.fg.params["means"].detach(), (0, 1), value=1.0),
        )  # (G, T, 3)
        loss_coarse_align = masked_l1_loss(pos_coarse, positions)
        loss += loss_coarse_align * self.losses_cfg.w_coarse_align

        ## fine bases should be close to identity
        bases_reg_loss = torch.tensor(0.0)
        if "fine_rots" in self.model.motion_bases.params:
            bases_reg_loss = compute_se3_reg_loss(
                self.model.motion_bases.params["fine_rots"], self.model.motion_bases.params["fine_transls"]
            )
            loss += bases_reg_loss * 0.5

        ## Rigidity loss
        rigid_body_loss = torch.tensor(0.0, device=device)
        if self.knn_idx is not None and self.rigid_weights is not None:
            ref_t = 0
            target_t = target_ts[0].item()
            frame_ids = torch.tensor([ref_t, target_t], device=device)
            centers_ts = self.model.motion_bases.get_centers(frame_ids, freeze_centers=True)  # (C, T, 3)
            rigid_body_loss = compute_arap_distance_loss(
                centers_ts, ref_t=0, knn_idx=self.knn_idx, weights=self.rigid_weights, detach_ref=True,
            )
            loss += rigid_body_loss * self.losses_cfg.w_rigidity

        # Prepare stats for logging.
        stats = {
            "loss": loss.item(),
            "mapped_depth_loss": mapped_depth_loss.item(),
            "track_2d_loss": track_2d_loss.item(),
            "loss_coarse_align": loss_coarse_align.item(),
            "rigid_body_loss": rigid_body_loss.item(),
            "bases_reg_loss": bases_reg_loss.item(),
        }

        return loss, stats, num_rays_per_step, num_rays_per_sec


    def compute_losses(self, batch):
        self.model.training = True

        # Edge-boundary correction (flow3d/graph_relative_edge.py): cheap,
        # every-step refresh of the falloff-row STATE snapshot (current
        # means/motion_coefs for the Gaussians already assigned to each
        # edge's boundary) -- unlike refresh_boundary_falloff (called only
        # from _control_step, on densify/cull) this does no nearest-neighbor
        # search, just re-gathers by the existing falloff_global_idx, so it's
        # safe to call every step. Without it, means/motion_coefs keep moving
        # via their own gradient between control_steps and the GNN's
        # gap_error/direction (and compute_boundary_gap_loss's dist_before)
        # would silently drift away from the position actually being
        # rendered. No-op unless motion_bases is
        # EdgeBoundaryGraphCorrectedScalableMotionBases.
        if isinstance(self.model.motion_bases, EdgeBoundaryGraphCorrectedScalableMotionBases):
            self.model.motion_bases.refresh_falloff_state(
                self.model.fg.params["means"].detach(), self.model.fg.get_coefs().detach()
            )

        # flow3d/graph_relative_linear_attention_boundary.py: same rationale
        # -- cheap, every-step refresh of the per-membership-row STATE
        # snapshot (current means/motion_coefs for Gaussians already
        # assigned to an edge/side patch), no nearest-neighbor search. Keeps
        # the patch pivot (centroid) the GNN sees from drifting away from
        # the position actually being rendered between control_steps.
        if isinstance(
            self.model.motion_bases, RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases
        ):
            self.model.motion_bases.refresh_edge_state(
                self.model.fg.params["means"].detach(), self.model.fg.get_coefs().detach()
            )

        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W)
        depth_masks = batch.get("depth_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W)
        fg_masks = batch["fg_masks"]
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # (B, H, W, 3)
        normals = batch.get("normals", None)
        # (B, H, W, 1)
        normal_masks = batch.get("normal_masks", None)
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]
        # [(N, P), ...].
        target_track_masks = batch["target_track_masks"]
        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        # [(N, G, 3), ...].
        target_ts_vec = torch.cat(target_ts)
        # (B * N, G, 3).
        target_means, _ = self.model.compute_poses_all(target_ts_vec)
        target_means = target_means.transpose(0, 1)
        target_mean_list = target_means.split(N)  # (N, G, 3) x B
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_depth=True,
                return_mask=self.model.has_bg,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)
            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        # Compute losses.
        # (B * N).
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0)

        ## RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb

        ## Mask loss.
        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore
        else:
            mask_loss = F.mse_loss(
                rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
            ) + masked_l1_loss(
                rendered_all["mask"],
                masks[..., None],
                quantile=0.98,  # type: ignore
            )
        loss += mask_loss * self.losses_cfg.w_mask

        ## 2D track loss for targets
        # (B * N, H * W, 3).
        pred_tracks_3d = (
            rendered_all["tracks_3d"].permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3)  # type: ignore
        )
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
        )  # w2c is applied before render
        # (B * N, H * W, 1).
        mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

        # (B * N).
        w_interval = torch.exp(-2 * frame_intervals / num_frames)

        # (B, H, W).
        masks_flatten = torch.zeros_like(masks)

        # get masks and weights for the 2d track points
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0

        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )

        # Add weights for the loss on tracks: confidence; penalty on distant frames
        track_weights = []
        track_depth_weights = []
        for i in range(B):
            weight = target_confidences[i] * w_interval[i: i + N].unsqueeze(-1)
            track_weights.append(weight.reshape(-1))
            track_depth_weights.append(target_track_masks[i].reshape(-1))
        track_weights = torch.cat(track_weights, dim=0).unsqueeze(-1)
        track_depth_weights = torch.cat(track_depth_weights, dim=0).unsqueeze(-1)

        track_2d_loss = masked_l1_loss(
            pred_tracks_2d[masks_flatten][visibles],
            tracks_2d[visibles],
            mask=track_weights[visibles],
            quantile=0.98,
        ) / max(H, W)
        loss += track_2d_loss * self.losses_cfg.w_track

        ## Depth loss on the current frame
        depth_masks = depth_masks * valid_masks if self.model.has_bg else depth_masks * masks
        depth_masks = depth_masks[..., None]

        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
        loss += depth_loss * self.losses_cfg.w_depth_reg

        ## Depth loss for track targets
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        mapped_depth_loss = masked_l1_loss(
            1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
            1 / (mapped_depth_gt[visibles, None] + 1e-5),
            mask=track_weights[visibles] * track_depth_weights[visibles],
            quantile=0.98,
        )

        loss += mapped_depth_loss * self.losses_cfg.w_depth_const

        ## Difference in depth between adjacent pixels <==> that of gt
        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )
        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

        ## Normal loss
        if rendered_all["rend_normal"] is not None and rendered_all["surf_normal"] is not None and normals is not None:
            # 2DGS normal consistency
            rendered_normals = F.normalize(rendered_all["rend_normal"], dim=-1)
            surf_normals = F.normalize(rendered_all["surf_normal"], dim=-1)
            surf_normals = surf_normals.reshape(rendered_normals.shape)
            normal_depth_masks = ~compute_depth_normal_mask(pred_depth.detach(), thresh_rel=0.03)
            # plt.imshow(masked_normals[0].detach().cpu()*0.5+0.5); plt.show()

            normal_geometry_loss = masked_cos_loss(
                rendered_normals, normals, mask=normal_masks, normalize_vec=False, quantile=0.95,
            )
            normal_consist_loss = masked_cos_loss(
                surf_normals, normals, mask=(normal_masks & normal_depth_masks), normalize_vec=False, quantile=0.95,
            )

            loss += self.losses_cfg.w_normal * normal_geometry_loss + self.losses_cfg.w_normal * normal_consist_loss
        else:
            normal_geometry_loss, normal_consist_loss = torch.tensor(0.0), torch.tensor(0.0)

        ## bases should be smooth.
        small_accel_loss = compute_se3_smoothness_loss(
            self.model.motion_bases.params["rots"],
            self.model.motion_bases.params["transls"],
        )
        if "fine_rots" in self.model.motion_bases.params:
            small_accel_loss += compute_se3_smoothness_loss(
                self.model.motion_bases.params["fine_rots"],
                self.model.motion_bases.params["fine_transls"],
            )
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases

        if self.model.has_shad:
            shad_accel_loss = compute_se3_smoothness_loss(
                self.model.shad_bases.params["rots"],
                self.model.shad_bases.params["transls"],
            )
            if "fine_rots" in self.model.shad_bases.params:
                shad_accel_loss += compute_se3_smoothness_loss(
                    self.model.shad_bases.params["fine_rots"],
                    self.model.shad_bases.params["fine_transls"],
                )
            loss += shad_accel_loss * self.losses_cfg.w_smooth_bases
        else:
            shad_accel_loss = torch.tensor(0.0)

        ## Tracks (means) in adjacent frames should be close
        ts_clamp = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts_clamp - 1, ts_clamp, ts_clamp + 1))
        transfms_nbs = self.model.compute_transforms(ts_neighbors)  # (G, 3n, 3, 4)

        # GNN correction (t-1, t, t+1) triplet, captured as a side effect of the
        # compute_transforms(ts_neighbors) call above (None for non-graph-coupled
        # motion_bases, e.g. plain ScalableMotionBases). See flow3d/analysis/loss.py.
        gnn_correction_nbs = getattr(self.model.motion_bases, "last_correction", None)
        if gnn_correction_nbs is not None:
            n = ts_clamp.shape[0]
            C_gnn = gnn_correction_nbs["omega"].shape[0]
            # (C, 3n, 3) -> (C, n, 3, 3): axis=-2 becomes the (t-1, t, t+1) triplet.
            omega_triplet = gnn_correction_nbs["omega"].reshape(C_gnn, 3, n, 3).permute(0, 2, 1, 3)
            delta_t_triplet = gnn_correction_nbs["delta_t"].reshape(C_gnn, 3, n, 3).permute(0, 2, 1, 3)

        # Edge-boundary correction (t-1, t, t+1) triplet, same side-effect
        # capture as gnn_correction_nbs above. See flow3d/graph_relative_edge.py.
        # None for non-graph-coupled motion_bases or the per-cluster gnn_variant.
        boundary_correction_nbs = getattr(self.model.motion_bases, "last_boundary_correction", None)
        if boundary_correction_nbs is not None:
            n = ts_clamp.shape[0]
            E_edge = boundary_correction_nbs["magnitude"].shape[0]
            # (E, 3n) -> (E, n, 3): axis=-1 becomes the (t-1, t, t+1) triplet.
            boundary_magnitude_triplet = boundary_correction_nbs["magnitude"].reshape(E_edge, 3, n).permute(0, 2, 1)

        # Edge-PATCH correction (t-1, t, t+1) triplet (flow3d/
        # graph_relative_linear_attention_boundary.py) -- same side-effect
        # capture pattern, but shaped (E, B, 2, 3) instead of (C, B, 3): the
        # "2" is side (a/b), not a cluster axis. None for every other variant
        # (this class defines last_edge_correction, not last_correction, so
        # gnn_correction_nbs above is always None for it and vice versa).
        edge_correction_nbs = getattr(self.model.motion_bases, "last_edge_correction", None)
        if edge_correction_nbs is not None:
            n = ts_clamp.shape[0]
            E_patch = edge_correction_nbs["omega"].shape[0]
            # (E, 3n, 2, 3) -> (E, n, 2, 3, 3): axis=-2 becomes the (t-1, t, t+1) triplet.
            edge_omega_triplet = (
                edge_correction_nbs["omega"].reshape(E_patch, 3, n, 2, 3).permute(0, 2, 3, 1, 4)
            )
            edge_delta_t_triplet = (
                edge_correction_nbs["delta_t"].reshape(E_patch, 3, n, 2, 3).permute(0, 2, 3, 1, 4)
            )

        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(self.model.fg.params["means"], (0, 1), value=1.0),
        )
        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        if self.losses_cfg.w_smooth_tracks > 0:
            small_accel_loss_tracks = 0.5 * (
                (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks

        if self.model.has_shad:
            shad_transfms_nbs = self.model.compute_shad_transforms(ts_neighbors)  # (G, 3n, 3, 4)
            means_shad_nbs = torch.einsum(
                "pnij,pj->pni",
                shad_transfms_nbs,
                F.pad(self.model.shad.params["means"], (0, 1), value=1.0),
            )
            means_shad_nbs = means_shad_nbs.reshape(
                means_shad_nbs.shape[0], 3, -1, 3
            )  # [G, 3, n, 3]
            if self.losses_cfg.w_smooth_tracks > 0:
                small_accel_loss_shad = 0.5 * (
                    (2 * means_shad_nbs[:, 1:-1] - means_shad_nbs[:, :-2] - means_shad_nbs[:, 2:])
                    .norm(dim=-1)
                    .mean()
                )
                loss += small_accel_loss_shad * self.losses_cfg.w_smooth_tracks
        else:
            small_accel_loss_shad = torch.tensor(0.0)

        ## Constrain the std of scales.
        if self.losses_cfg.use_log_scale_var:
            func_scale_var = lambda x: torch.var(x, dim=-1).mean()
        else:
            func_scale_var = lambda x: torch.var(torch.exp(x), dim=-1).mean()

        scale_var_loss = func_scale_var(self.model.fg.params["scales"])
        if self.model.bg is not None:
            scale_var_loss += func_scale_var(self.model.bg.params["scales"])
        loss += self.losses_cfg.w_scale_var * scale_var_loss

        ## Distance along ray direction (depth?) between adjacent frames should be close
        # Acceleration along ray direction should be small.
        z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)
        loss += self.losses_cfg.w_z_accel * z_accel_loss

        ## Regularize centers
        loss_center_cano = torch.tensor(0.0)
        if "centers" in self.model.motion_bases.params:
            # align the centers in canonical frame
            cluster_ids = self.model.fg.get_cluster_ids()
            loss_center_cano = center_to_cluster_mean_loss(
                self.model.fg.params["means"].detach(), self.model.motion_bases.params["centers"], cluster_ids,
            )
            loss += loss_center_cano * self.losses_cfg.w_center_cano

        ## Align coarse transformed gaussian means
        num_fg = self.model.num_fg_gaussians
        positions = means.transpose(0, 1)[:num_fg].detach()
        transfms_coarse = self.model.compute_transforms_coarse(ts)
        pos_coarse = torch.einsum(
            "pnij,pj->pni",
            transfms_coarse,
            F.pad(self.model.fg.params["means"].detach(), (0, 1), value=1.0),
        )  # (G, T, 3)
        loss_coarse_align = masked_l1_loss(pos_coarse, positions)
        loss += loss_coarse_align * self.losses_cfg.w_coarse_align

        # GNN correction at the current batch's ts, captured as a side effect
        # of the compute_transforms_coarse(ts) call just above (overwrites
        # gnn_correction_nbs, which is why the smoothness loss below reads
        # from the triplet captured earlier instead of from here).
        gnn_correction_cur = getattr(self.model.motion_bases, "last_correction", None)
        # Same side effect, edge-patch shape (flow3d/graph_relative_linear_
        # attention_boundary.py). Always None together with gnn_correction_cur
        # (mutually exclusive: no motion_bases variant defines both).
        edge_correction_cur = getattr(self.model.motion_bases, "last_edge_correction", None)

        ## GNN correction (omega, delta_t) regularizers -- see
        ## flow3d/analysis/loss.py. All no-op (0.0, not computed) for
        ## non-graph-coupled motion_bases (gnn_correction_cur is None then).
        gnn_correction_reg_loss = torch.tensor(0.0, device=device)
        gnn_correction_smooth_loss = torch.tensor(0.0, device=device)
        gnn_correction_edge_loss = torch.tensor(0.0, device=device)
        if gnn_correction_cur is not None:
            omega_cur, delta_t_cur = gnn_correction_cur["omega"], gnn_correction_cur["delta_t"]

            gnn_correction_reg_loss = gnn_correction_magnitude_loss(omega_cur, delta_t_cur)
            loss += gnn_correction_reg_loss * self.losses_cfg.w_gnn_correction_reg

            if gnn_correction_nbs is not None:
                gnn_correction_smooth_loss = gnn_correction_smoothness_loss(omega_triplet, delta_t_triplet)
                loss += gnn_correction_smooth_loss * self.losses_cfg.w_gnn_correction_smooth

            edge_index_dir = getattr(self.model.motion_bases.gnn, "edge_index_dir", None)
            if edge_index_dir is not None:
                # (flow3d/graph_relative_linear_attention_frame.py only)
                # gnn_correction_cur was captured right after the
                # compute_transforms_coarse(ts) call above, from the SAME
                # forward pass's last_correction dict -- so its
                # "edge_gate_dir" (if present) is guaranteed to be this exact
                # frame's gate, never a stale value from a different
                # _corrected_coarse call. Every other variant's
                # last_correction dict has no such key, so .get(...) is None
                # and the call below is byte-identical to before.
                edge_weight = gnn_correction_cur.get("edge_gate_dir")
                gnn_correction_edge_loss = gnn_correction_edge_consistency_loss(
                    omega_cur,
                    delta_t_cur,
                    edge_index_dir,
                    cos_margin=self.losses_cfg.gnn_correction_edge_consistency_cos_margin,
                    edge_weight=edge_weight,
                )
                loss += gnn_correction_edge_loss * self.losses_cfg.w_gnn_correction_edge_consistency

        ## Edge-PATCH correction (omega + delta_t per edge-SIDE) regularizers
        ## -- see flow3d/graph_relative_linear_attention_boundary.py. No-op
        ## (0.0, not computed) unless motion_bases is
        ## RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases
        ## (--gnn_variant=relative_velocity_linear_attention_boundary).
        ## gnn_correction_magnitude_loss/gnn_correction_smoothness_loss are
        ## shape-agnostic (operate on the last dim=3 / last two dims=(3,3)),
        ## so the SAME functions used for the per-cluster (C,B,3) shape above
        ## apply unmodified to this (E,B,2,3) shape -- only the reshape/
        ## permute producing the triplet differs (captured above as
        ## edge_omega_triplet/edge_delta_t_triplet). Regularizes the RAW
        ## (patch-weight-*before*-application) edge decoder output, matching
        ## how the per-cluster variant regularizes its own raw last_correction
        ## -- not the per-Gaussian combined correction, so a raw prediction
        ## that's currently small only because patch weight is small doesn't
        ## silently escape regularization. No edge-consistency analog is
        ## computed here: "neighboring CLUSTERS' single correction vector
        ## shouldn't oppose" doesn't translate to per-edge-side corrections;
        ## edge_boundary_gap_loss below is this variant's geometric-
        ## consistency mechanism instead.
        edge_correction_reg_loss = torch.tensor(0.0, device=device)
        edge_correction_smooth_loss = torch.tensor(0.0, device=device)
        if edge_correction_cur is not None:
            edge_correction_reg_loss = gnn_correction_magnitude_loss(
                edge_correction_cur["omega"], edge_correction_cur["delta_t"]
            )
            loss += edge_correction_reg_loss * self.losses_cfg.w_edge_correction_reg

            if edge_correction_nbs is not None:
                edge_correction_smooth_loss = gnn_correction_smoothness_loss(
                    edge_omega_triplet, edge_delta_t_triplet
                )
                loss += edge_correction_smooth_loss * self.losses_cfg.w_edge_correction_smooth

        ## Edge-patch boundary gap loss: OFF by default
        ## (self.losses_cfg.w_edge_boundary_gap == 0.0). Only applied to
        ## edges the class itself already judged reliable_edge_mask=True
        ## (persistence/num_known_frames gated at construction time, see
        ## flow3d/graph_relative_linear_attention_boundary.py's
        ## _load_edge_topology_from_edges_pt) AND observed CONNECTED this
        ## frame -- not every CONNECTED edge. Gradient reaches only
        ## motion_bases.gnn's own parameters (detach_base=True internally);
        ## does not disturb last_edge_correction (see that function's
        ## docstring).
        edge_boundary_gap_loss_value = torch.tensor(0.0, device=device)
        if self.losses_cfg.w_edge_boundary_gap > 0 and isinstance(
            self.model.motion_bases, RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases
        ):
            edge_boundary_gap_loss_value = compute_edge_patch_gap_loss(
                self.model.motion_bases, ts, tolerance=self.losses_cfg.edge_boundary_gap_tolerance
            )
            loss += edge_boundary_gap_loss_value * self.losses_cfg.w_edge_boundary_gap

        ## Edge-boundary correction (omega-free, per-edge translation
        ## magnitude) regularizers -- see flow3d/graph_relative_edge.py. Both
        ## no-op (0.0, not computed) unless motion_bases is
        ## EdgeBoundaryGraphCorrectedScalableMotionBases
        ## (--gnn_variant=relative_edge_boundary). Read from the (t-1, t, t+1)
        ## triplet captured above; the middle frame (index 1) is ts_clamp,
        ## which equals ts except at the very first/last frame of the sequence.
        boundary_correction_reg_loss = torch.tensor(0.0, device=device)
        boundary_correction_smooth_loss = torch.tensor(0.0, device=device)
        if boundary_correction_nbs is not None:
            magnitude_cur = boundary_magnitude_triplet[:, :, 1]  # (E, n)
            boundary_correction_reg_loss = boundary_magnitude_reg_loss(magnitude_cur)
            loss += boundary_correction_reg_loss * self.losses_cfg.w_boundary_correction_reg

            boundary_correction_smooth_loss = boundary_magnitude_smoothness_loss(boundary_magnitude_triplet)
            loss += boundary_correction_smooth_loss * self.losses_cfg.w_boundary_correction_smooth

        ## Boundary gap loss: OFF by default (self.losses_cfg.w_boundary_gap
        ## defaults to 0.0) -- correction magnitude is now learned directly
        ## from the render loss (RGB/depth/mask/track, via compute_transforms's
        ## detach_base=False path into EdgeBoundaryGraphCorrectedScalableMotion
        ## Bases._edge_features_and_direction), not gated by this hinge any
        ## more (see flow3d/graph_relative_edge.py's EdgeBoundaryGNN docstring
        ## for why: contact_reference_distance is a median over this SAME
        ## reconstruction's own "connected" frames, so a persistently-open
        ## edge has it absorbed as "normal" and the old gap_error-gated
        ## correction could never learn to close it). Kept as an opt-in extra
        ## loss (set --loss.w-boundary-gap > 0 to enable) and for
        ## compute_boundary_gap_distances-based diagnostics
        ## (flow3d/analysis/gnn_check_edge.py) -- guarded so the (otherwise
        ## redundant) extra forward pass through _edge_features_and_direction
        ## is skipped while the weight is 0.
        boundary_gap_loss_value = torch.tensor(0.0, device=device)
        if self.losses_cfg.w_boundary_gap > 0 and isinstance(
            self.model.motion_bases, EdgeBoundaryGraphCorrectedScalableMotionBases
        ):
            boundary_gap_loss_value = compute_boundary_gap_loss(self.model.motion_bases, ts)
            loss += boundary_gap_loss_value * self.losses_cfg.w_boundary_gap

        ## Joint anchor loss (GNN-only, true-transform target): keeps each
        ## cluster pair connected by an edge in the cluster graph
        ## (optim_cfg.joint_anchor_path -- the SAME build_cluster_graph.py
        ## edges.pt used for --graph-coupling-path) from opening in the
        ## ACTUAL rendered (coarse+fine-blended) geometry, while still
        ## allowing ordinary joint rotation -- see
        ## flow3d/analysis/loss_joint_gnn_only.py. Measured directly: the
        ## coarse-only version of this loss (flow3d/analysis/loss_joint.py)
        ## can converge its own (coarse-only) distance to ~1x canonical
        ## while the true rendered boundary gap barely moves, since fine
        ## motion bases are per-cluster and free to diverge at a shared
        ## boundary regardless of coarse alignment. This version measures
        ## the true gap and restricts gradient to ONLY motion_bases.gnn's
        ## own parameters (base coarse motion basis, centers, and fine
        ## motion are all frozen inputs) -- a deliberate isolation so the
        ## GNN's own contribution is unambiguous, not because it's expected
        ## to fully close the gap by itself (a rigid per-cluster correction
        ## can only close the *coherent* component of a boundary gap; the
        ## non-rigid remainder needs a separate fine-level treatment, not
        ## implemented here). No-op (0.0, not even computed) unless
        ## optim_cfg.joint_anchor_path is set.
        joint_anchor_loss_value = torch.tensor(0.0, device=device)
        if self.optim_cfg.joint_anchor_path:
            if isinstance(self.model.motion_bases, EdgeBoundaryGraphCorrectedScalableMotionBases):
                raise ValueError(
                    "optim_cfg.joint_anchor_path targets the per-cluster (omega, delta_t) "
                    "GNN correction and is not supported with "
                    "gnn_variant='relative_edge_boundary' -- use "
                    "optim_cfg.boundary_gap_path instead (see flow3d/graph_relative_edge.py)."
                )
            if isinstance(
                self.model.motion_bases, RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases
            ):
                raise ValueError(
                    "optim_cfg.joint_anchor_path targets the per-cluster (omega, delta_t) "
                    "GNN correction (flow3d/analysis/loss_joint_gnn_only.py's "
                    "_gnn_only_compute_transforms calls motion_bases.gnn with the per-cluster "
                    "4-argument signature) and is not supported with "
                    "gnn_variant='relative_velocity_linear_attention_boundary' (its GNN takes an "
                    "extra edge-patch-feature argument and returns a per-edge-side shape) -- use "
                    "--loss.w_edge_boundary_gap instead (see "
                    "flow3d/graph_relative_linear_attention_boundary.py's compute_edge_patch_gap_loss)."
                )
            if self.joint_anchor_boundary_sets is None:
                self.joint_anchor_boundary_sets = build_joint_anchor_boundary_sets(
                    self.optim_cfg.joint_anchor_path,
                    self.model.fg.params["means"].detach(),
                    device=device,
                )
                num_clusters = self.model.motion_bases.num_clusters
                max_cluster_id = int(self.joint_anchor_boundary_sets.cluster_ids.max())
                if max_cluster_id >= num_clusters:
                    raise ValueError(
                        f"{self.optim_cfg.joint_anchor_path} references cluster ids up to "
                        f"{max_cluster_id}, but the model currently has {num_clusters} "
                        "clusters (0.."
                        f"{num_clusters - 1}) -- joint_anchor_path requires the cluster set to "
                        "stay exactly as it was when the file was built. Pass "
                        "--optim.no-enable-bases-control."
                    )
                guru.info(
                    f"[joint-anchor] loaded {self.joint_anchor_boundary_sets.num_pairs} joint "
                    f"anchor pair(s) from edges.pt at {self.optim_cfg.joint_anchor_path}"
                )

            transformed_anchors = transform_joint_anchor_boundary_sets_gnn_only(
                self.model.motion_bases,
                ts,
                self.joint_anchor_boundary_sets,
                self.model.fg.params["means"].detach(),
                self.model.fg.get_coefs().detach(),
                self.model.fg.get_cluster_ids().reshape(-1).long(),
            )
            joint_anchor_loss_value = joint_anchor_loss(
                transformed_anchors,
                self.joint_anchor_boundary_sets.canonical_distance,
                huber_delta=self.losses_cfg.joint_anchor_huber_delta,
            )
            loss += joint_anchor_loss_value * self.losses_cfg.w_joint_anchor

        ## Rigidity loss (ARAP). update_rigidity_weights() is normally only
        ## triggered as a side effect of _bases_control_step, so this only
        ## ever fires once basis control has been explicitly disabled and
        ## control_step() has populated knn_idx/rigid_weights itself -- this
        ## keeps default (bases-control-on) runs byte-for-byte unaffected.
        rigid_body_loss = torch.tensor(0.0, device=device)
        if (
            not self.optim_cfg.enable_bases_control
            and self.knn_idx is not None
            and self.rigid_weights is not None
        ):
            ref_t = 0
            target_t = target_ts[0][0].item()  # one target frame is enough for the ARAP graph
            frame_ids = torch.tensor([ref_t, target_t], device=device)
            centers_ts = self.model.motion_bases.get_centers(frame_ids, freeze_centers=True)  # (C, T, 3)
            rigid_body_loss = compute_arap_distance_loss(
                centers_ts, ref_t=0, knn_idx=self.knn_idx, weights=self.rigid_weights, detach_ref=True,
            )
            loss += rigid_body_loss * self.losses_cfg.w_rigidity

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/mapped_depth_loss": mapped_depth_loss.item(),
            "train/track_2d_loss": track_2d_loss.item(),
            "train/normal_geometry_loss": normal_geometry_loss.item(),
            "train/normal_consist_loss": normal_consist_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/shad_accel_loss": shad_accel_loss.item(),
            "train/small_accel_loss_tracks": small_accel_loss_tracks.item(),
            "train/small_accel_loss_shad": small_accel_loss_shad.item(),
            "train/scale_var_loss": scale_var_loss.item(),
            "train/z_accel_loss": z_accel_loss.item(),
            "train/loss_center_cano": loss_center_cano.item(),
            "train/loss_coarse_align": loss_coarse_align.item(),
            "train/rigid_body_loss": rigid_body_loss.item(),
            "train/gnn_correction_reg_loss": gnn_correction_reg_loss.item(),
            "train/gnn_correction_smooth_loss": gnn_correction_smooth_loss.item(),
            "train/gnn_correction_edge_loss": gnn_correction_edge_loss.item(),
            "train/boundary_correction_reg_loss": boundary_correction_reg_loss.item(),
            "train/boundary_correction_smooth_loss": boundary_correction_smooth_loss.item(),
            "train/boundary_gap_loss": boundary_gap_loss_value.item(),
            "train/edge_correction_reg_loss": edge_correction_reg_loss.item(),
            "train/edge_correction_smooth_loss": edge_correction_smooth_loss.item(),
            "train/edge_boundary_gap_loss": edge_boundary_gap_loss_value.item(),
            "train/joint_anchor_loss": joint_anchor_loss_value.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
        }

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, fg_masks if not self.model.has_bg else torch.ones_like(valid_masks)
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            if self.model.has_bg:
                bg_psnr = self.bg_psnr_metric(rendered_imgs, imgs, 1.0 - fg_masks)
                fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, fg_masks)
                self.bg_psnr_metric.reset()
                self.fg_psnr_metric.reset()
                stats["train/bg_psnr"] = bg_psnr
                stats["train/fg_psnr"] = fg_psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec

    def log_dict(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.global_step)

    def log_hparams(self, hparams_dict, metric_dict):
        self.writer.add_hparams(
            hparam_dict=hparams_dict, metric_dict=metric_dict, run_name=".", global_step=self.global_step,
        )

    @torch.no_grad()
    def add_control_step(self, fg_params, bg_params):
        """
        Add a set of new gaussians into the existing model.
        """
        num_fg = self.model.num_fg_gaussians
        num_new_fg = fg_params.num_gaussians if fg_params is not None else 0
        num_new_bg = bg_params.num_gaussians if bg_params is not None else 0

        # add new params to fg model
        if fg_params is not None:
            fg_param_map = self.model.fg.add_params(fg_params)

            # fix the optimizers
            for param_name, new_params in fg_param_map.items():
                full_param_name = f"fg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                add_in_optim(optimizer, [new_params], dim=0)

        # add bg model
        if bg_params is not None:
            bg_param_map = self.model.bg.add_params(bg_params)

            # fix the optimizers
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                add_in_optim(optimizer, [new_params], dim=0)

        # update running stats
        for k, v in self.running_stats.items():
            v_fg, v_bg = v[:num_fg], v[num_fg:]
            new_v = torch.cat(
                [
                    v_fg,
                    v.new_zeros(num_new_fg),
                    v_bg,
                    v.new_zeros(num_new_bg),
                ],
                dim=0,
            )
            self.running_stats[k] = new_v

        guru.info(f"Added {num_new_fg + num_new_bg} gaussians.")

    @torch.no_grad()
    def add_motion_bases(self, should_dup):
        """
        Duplicate bases.
        """
        bases_param_map = self.model.motion_bases.dup_bases(should_dup)

        # update optimizer
        num_bases_dup = int(should_dup.sum().item())
        for param_name, new_params in bases_param_map.items():
            full_param_name = f"motion_bases.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            # all bases are kept and add more to the end
            dup_in_optim(
                optimizer,
                [new_params],
                torch.zeros_like(should_dup),
                num_bases_dup,
            )

        guru.info(f"Duplicate {num_bases_dup} bases.")

    def control_step(self, step):
        """
        Apply control at valid steps
        """
        cfg = self.optim_cfg

        is_bases_control_interval = (
            step >= cfg.start_control_steps
            and step < cfg.stop_control_steps
            and (step + 1) % cfg.control_bases_every == cfg.control_bases_offset
        )

        # control bases
        if cfg.enable_bases_control and is_bases_control_interval:
            self._bases_control_step(step)

        if is_bases_control_interval:
            guru.info(
                f"[control] {step=} num_clusters={self.model.motion_bases.num_clusters} "
                f"(bases_control={'ON' if cfg.enable_bases_control else 'OFF'})"
            )

        # rigidity refresh: normally a side effect of _bases_control_step, so
        # when basis control is disabled it must be driven explicitly instead.
        if not cfg.enable_bases_control:
            if self.knn_idx is None:
                self.update_rigidity_weights()
                guru.info(f"[rigidity] Initial rigidity weights computed at {step=}.")
            elif (
                cfg.rigidity_refresh_every > 0
                and (step + 1) % cfg.rigidity_refresh_every == 0
            ):
                self.update_rigidity_weights()
                guru.info(f"[rigidity] Refreshed rigidity weights at {step=}.")

        # densify and cull
        if (
            step >= cfg.warmup_steps
            and step < cfg.stop_control_steps
            and (step + 1) % cfg.control_every == 0
        ):
            if (
                step < cfg.stop_densify_steps
                and (step + 1) % self.reset_opacity_every > cfg.control_every
            ):
                self._densify_control_step(step)
            if (step + 1) % self.reset_opacity_every > 3 * cfg.control_every:
                self._cull_control_step(step)

            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

            guru.info(f"Adaptive control at {self.global_step=} {self.epoch=}")

        # reset opacity
        if (
            step >= cfg.warmup_steps
            and step < cfg.stop_control_steps
            and (step + 1) % self.reset_opacity_every == 0
        ):
            self._reset_opacity_control_step()

        # Edge-boundary correction (flow3d/graph_relative_edge.py): densify/cull
        # above (_densify_control_step/_cull_control_step) run regardless of
        # cfg.enable_bases_control -- only cluster/motion-basis *count* is
        # frozen by --optim.no-enable-bases-control, not per-Gaussian adaptive
        # density control. So the foreground Gaussian array can still be
        # split/dup'd/culled (reordered and resized) on every control_step,
        # and motion_bases's falloff rows (which Gaussian belongs to which
        # edge's boundary, and how much) must be recomputed against the new
        # array -- refresh_boundary_falloff does this from live positions only
        # (no stale index dependency), leaving edge topology/anchors untouched.
        if isinstance(self.model.motion_bases, EdgeBoundaryGraphCorrectedScalableMotionBases):
            mb = self.model.motion_bases
            if int(mb.num_fg_gaussians) != self.model.fg.num_gaussians:
                mb.refresh_boundary_falloff(
                    self.model.fg.params["means"].detach(),
                    self.model.fg.get_cluster_ids().reshape(-1).long(),
                    self.model.fg.get_coefs().detach(),
                )
                guru.info(
                    f"[boundary-falloff] refreshed at {step=}: "
                    f"num_fg_gaussians={mb.num_fg_gaussians.item()} "
                    f"falloff_rows={mb.falloff_global_idx.numel()}"
                )

        # flow3d/graph_relative_linear_attention_boundary.py: same rationale
        # as the edge-boundary refresh above -- the per-Gaussian edge/side
        # membership ASSIGNMENT (which current Gaussian belongs to which
        # edge/side patch, and how much) is keyed by canonical identity, so
        # it must be recomputed whenever the foreground Gaussian array is
        # resized/reordered by densify/cull. The canonical boundary
        # reference (patch_ref_*) and edge topology (edge_cluster_a/b) are
        # untouched. Expensive (nearest-neighbor search) -- only run on an
        # actual count change, same as the edge-boundary variant above.
        if isinstance(
            self.model.motion_bases, RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases
        ):
            mb = self.model.motion_bases
            if int(mb.num_fg_gaussians) != self.model.fg.num_gaussians:
                mb.refresh_edge_membership(
                    self.model.fg.params["means"].detach(),
                    self.model.fg.get_cluster_ids().reshape(-1).long(),
                    self.model.fg.get_coefs().detach(),
                )
                guru.info(
                    f"[edge-patch-membership] refreshed at {step=}: "
                    f"num_fg_gaussians={mb.num_fg_gaussians.item()} "
                    f"membership_rows={mb.row_gaussian_idx.numel()}"
                )

    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        # Prepare for adaptive gaussian control based on the current stats.
        if not (
            self.model._current_radii is not None
            and self.model._current_xys is not None
        ):
            raise ValueError("Model not training, skipping control step preparation")
            return False

        batch_size = len(self._batched_xys)
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views
        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):
            _current_radii = _current_radii.max(dim=-1)[0]
            sel = _current_radii > 0
            gidcs = torch.where(sel)[1]
            # normalize grads to [-1, 1] screen space
            xys_grad = _current_xys.grad.clone()
            xys_grad[..., 0] *= _current_img_wh[0] / 2.0 * batch_size
            xys_grad[..., 1] *= _current_img_wh[1] / 2.0 * batch_size
            self.running_stats["xys_grad_norm_acc"].index_add_(
                0, gidcs, xys_grad[sel].norm(dim=-1)
            )
            self.running_stats["vis_count"].index_add_(
                0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
            )
            max_radii = torch.maximum(
                self.running_stats["max_radii"].index_select(0, gidcs),
                _current_radii[sel] / max(_current_img_wh),
            )
            self.running_stats["max_radii"].index_put_((gidcs,), max_radii)
        return True

    def _load_cluster_graph_file(
        self, path: str, num_clusters: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load flow3d/analysis/build_cluster_graph.py's edges.pt and pad it to
        the same (C, max_degree) knn_idx/valid_mask convention
        build_body_connectivity_graph returns, so the rest of
        update_rigidity_weights doesn't care which source built the graph.

        edges.pt's cluster ids are 0-indexed against the checkpoint it was
        built from (model.fg.get_cluster_ids()), the same indexing
        motion_bases uses -- but only valid as long as the cluster set hasn't
        changed since (bases split/cull remaps cluster ids), hence the
        num_clusters check below.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        graph_cluster_ids = payload["cluster_ids"]
        if graph_cluster_ids and max(graph_cluster_ids) >= num_clusters:
            raise ValueError(
                f"{path} was built for cluster ids up to {max(graph_cluster_ids)}, but the "
                f"model currently has {num_clusters} clusters (0..{num_clusters - 1}) -- "
                "cluster_graph_file requires the cluster set to stay exactly as it was when "
                "the graph was built. Pass --optim.no-enable-bases-control."
            )

        neighbors: list[list[int]] = [[] for _ in range(num_clusters)]
        for e in payload["edges_kept"]:
            a, b = int(e["cluster_a"]), int(e["cluster_b"])
            neighbors[a].append(b)
            neighbors[b].append(a)

        max_degree = max(1, max((len(n) for n in neighbors), default=1))
        knn_idx = torch.arange(num_clusters, device=self.device).unsqueeze(1).repeat(1, max_degree)
        valid_mask = torch.zeros((num_clusters, max_degree), dtype=torch.bool, device=self.device)
        for c, ns in enumerate(neighbors):
            if not ns:
                continue
            knn_idx[c, : len(ns)] = torch.tensor(ns, device=self.device, dtype=torch.long)
            valid_mask[c, : len(ns)] = True

        guru.info(
            f"[rigidity] loaded cluster graph file: {path} "
            f"(kept_edges={len(payload['edges_kept'])}, cut_edges={len(payload.get('edges_cut', []))})"
        )
        return knn_idx, valid_mask

    @torch.no_grad()
    def update_rigidity_weights(self, k=5, sigma_rel=0.05, eps=1e-6):
        # get all tracks
        num_frames = self.model.num_frames
        all_tracks = self.model.motion_bases.get_centers(
            torch.arange(num_frames, device=self.device), freeze_centers=True,
        )  # (C, T, 3)

        graph_type = self.optim_cfg.rigidity_graph_type
        if graph_type == "connectivity":
            # graph structure is fixed for the whole run: build it once from the
            # (fixed) cluster assignment, then only refresh the edge weights.
            if self.connectivity_knn_idx is None:
                cluster_ids = self.model.fg.get_cluster_ids().reshape(-1).long()
                means_cano = self.model.fg.params["means"]
                self.connectivity_knn_idx, self.connectivity_valid_mask = build_body_connectivity_graph(
                    means_cano=means_cano,
                    cluster_ids=cluster_ids,
                    centers_ts=all_tracks,
                    num_clusters=self.model.motion_bases.num_clusters,
                    spatial_k=self.optim_cfg.connectivity_spatial_k,
                    min_shared_edges=self.optim_cfg.connectivity_min_shared_edges,
                    cv_threshold=self.optim_cfg.connectivity_cv_threshold,
                )
                log_connectivity_graph(self.connectivity_knn_idx, self.connectivity_valid_mask)

            knn_idx = self.connectivity_knn_idx
            valid_mask = self.connectivity_valid_mask
        elif graph_type == "cluster_graph_file":
            # graph structure comes from an offline edges.pt (see
            # flow3d/analysis/build_cluster_graph.py, typically run with
            # --position-source raw_tracks so it isn't corrupted by the same
            # learned-motion swaps rigidity is meant to prevent): load and
            # pad it once, then only refresh the edge weights.
            if self.cluster_graph_knn_idx is None:
                if not self.optim_cfg.rigidity_graph_path:
                    raise ValueError(
                        "rigidity_graph_type='cluster_graph_file' requires optim_cfg.rigidity_graph_path."
                    )
                self.cluster_graph_knn_idx, self.cluster_graph_valid_mask = self._load_cluster_graph_file(
                    self.optim_cfg.rigidity_graph_path,
                    num_clusters=self.model.motion_bases.num_clusters,
                )
                log_connectivity_graph(self.cluster_graph_knn_idx, self.cluster_graph_valid_mask)

            knn_idx = self.cluster_graph_knn_idx
            valid_mask = self.cluster_graph_valid_mask
        elif graph_type == "euclidean":
            # compute knn graph
            points_ref = all_tracks[:, 0]  # (C, 3)
            dists = torch.cdist(points_ref, points_ref)
            _, knn_idx = dists.topk(k + 1, largest=False)
            knn_idx = knn_idx[:, 1:]  # (C, k)
            valid_mask = None
        else:
            raise ValueError(f"Unknown rigidity_graph_type: {graph_type}")

        self.knn_idx = knn_idx

        # lengths inside local neighbors
        neighbors_all = all_tracks[self.knn_idx]  # (C, k, T, 3)
        lengths_all = torch.norm(neighbors_all - all_tracks.unsqueeze(1), dim=-1)  # (C, k, T)
        lengths_mean = lengths_all.mean(dim=-1)  # (C, k)

        # std of the lengths over time
        lengths_std = lengths_all.std(dim=-1, unbiased=False)  # (C, k)
        rel = lengths_std / (lengths_mean.clamp_min(eps))  # (C, k)
        rigid_weights = torch.exp(-(rel ** 2) / (2 * sigma_rel ** 2))  # (C, k)
        if valid_mask is not None:
            # padded (self-index) slots carry zero weight -> zero contribution
            # to compute_arap_distance_loss, matching a cluster with no valid
            # connectivity neighbor.
            rigid_weights = rigid_weights * valid_mask.float()
        self.rigid_weights = rigid_weights


    @torch.no_grad()
    def _add_motion_bases(self, bases):
        """
        Add new time dimensions to the existing motion bases.
        """
        # add new motion bases
        bases_map = self.model.motion_bases.add_new_frames(bases)

        # fix the optimizers
        for param_name, new_params in bases_map.items():
            full_param_name = f"motion_bases.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            add_in_optim(optimizer, [new_params], dim=-2)

    @torch.no_grad()
    def _add_shad_bases(self, bases):
        """
        Add new time dimensions to the existing motion bases.
        """
        # add new motion bases
        bases_map = self.model.shad_bases.add_new_frames(bases)

        # fix the optimizers
        for param_name, new_params in bases_map.items():
            full_param_name = f"shad_bases.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            add_in_optim(optimizer, [new_params], dim=-2)

    @torch.no_grad()
    def _add_camera_poses(self, new_poses):
        """
        Add camera poses for new frames.
        """
        poses_map = self.model.camera_poses.add_new_frames(new_poses)

        # fix the optimizers
        for param_name, new_params in poses_map.items():
            full_param_name = f"camera_poses.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            add_in_optim(optimizer, [new_params], dim=0)

    @torch.no_grad()
    def _bases_control_step(self, global_step):
        """
        Split and cull motion bases.
        """
        num_frames = self.model.num_frames
        num_clus = self.model.motion_bases.num_clusters
        num_fg = self.model.num_fg_gaussians
        cfg = self.optim_cfg

        ## Split bases
        opacities = self.model.get_opacities_all()
        should_cull = torch.zeros_like(opacities, dtype=torch.bool)
        num_bases_split = 0
        if num_clus < cfg.max_num_bases:
            cluster_ids = self.model.fg.get_cluster_ids()
            clus_ids, clus_counts = cluster_ids.unique(return_counts=True)

            # check each cluster
            should_clus_split = cluster_ids.new_zeros(num_clus).bool()
            num_clus_new = num_clus
            means, _ = self.model.compute_poses_fg(
                torch.arange(max(num_frames - cfg.clustering_window_size, 0), num_frames)
            )
            split_clus_info = {}
            for clus_id in clus_ids:
                mask_in_cluster = cluster_ids == clus_id
                ids_in_cluster = torch.where(mask_in_cluster)[0]

                # use velocity for clustering
                means_clus = means[mask_in_cluster]
                num_pts_in_clus = means_clus.shape[0]
                if num_pts_in_clus < 50:
                    continue
                velocities = means_clus[:, 1:] - means_clus[:, :-1]
                velocities = velocities.reshape((num_pts_in_clus, -1))
                labels, _, cluster_persistence = cluster_by_velocities(velocities)
                labels = labels.long()
                cluster_persistence = cluster_persistence.squeeze()

                # check valid clusters
                ids, counts = labels.unique(return_counts=True)
                valid_ids = ids[ids >= 0]
                if -1 in ids:
                    # check the ratio of noise, skip when the clustering is unstable
                    ratio = counts[0] / labels.shape[0]
                    if ratio < cfg.max_noise_ratio:
                        # cull noise points
                        if cfg.cull_clustering_noise:
                            mask = ids_in_cluster[labels == -1]
                            should_cull[mask] = True
                    else:
                        continue
                if valid_ids.shape[0] <= 1:
                    continue

                # distance between clusters
                centroids = torch.stack([velocities[labels == id].mean(dim=0) for id in valid_ids])
                dist_matrix = torch.cdist(centroids, centroids, p=2)
                max_dist = dist_matrix.max()

                # meta-clustering on centers
                if max_dist > cfg.cluster_dist_threshold:
                    # merge small clusters
                    agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
                    clus_labels = agg.fit_predict(centroids.detach().cpu().numpy())
                    clus_labels = torch.from_numpy(clus_labels)

                    # overwrite point labels
                    point_labels = labels.clone()
                    for cid, m in zip(valid_ids.tolist(), clus_labels.tolist()):
                        point_labels[labels == cid] = m

                    # deal with noise, add to the nearest cluster
                    noise_mask = labels == -1
                    if noise_mask.any() and not cfg.cull_clustering_noise:
                        # compute macro centers
                        macro_centers = []
                        for m in [0, 1]:
                            mask = point_labels == m
                            mc = velocities[mask].mean(dim=0)
                            macro_centers.append(mc)
                        macro_centers = torch.stack(macro_centers, dim=0)

                        # assign each noise point to the nearest macro center
                        d = torch.cdist(velocities[noise_mask], macro_centers, p=2)
                        nearest = d.argmin(dim=1)
                        point_labels[noise_mask] = nearest

                    # do not split when new cluster is too small
                    new_ids, new_counts = point_labels.unique(return_counts=True)
                    if (new_counts[new_ids >= 0] > cfg.split_bases_threshold * num_fg).all():
                        # update cluster_ids, new clusters are at the end of current bases
                        # noise is still in the cluster, will be deleted later
                        for id in new_ids:
                            if id >= 0:
                                mask = ids_in_cluster[point_labels == id]
                                cluster_ids[mask] = num_clus_new
                                num_clus_new += 1

                        # should_clus_split for later update
                        split_clus_info[clus_id] = {
                            "old_ids": ids.detach().cpu(),
                            "old_counts": counts.detach().cpu(),
                            "new_ids": new_ids.detach().cpu(),
                            "new_counts": new_counts.detach().cpu(),
                        }
                        should_clus_split[clus_id] = True

            # Update model bases
            num_new_bases = num_clus_new - num_clus
            if num_new_bases > 0:
                # initialize new cluster parameters
                new_bases, new_coefs = init_motion_params_for_split(
                    self.model,
                    torch.arange(num_clus, num_clus_new),
                    cluster_ids,
                    # add cluster_method
                    coefs_type=self.optim_cfg.coefs_type,
                    coefs_sigma=self.optim_cfg.coefs_sigma,
                )

                # update fg coefs
                self.model.fg.params["motion_coefs"].copy_(new_coefs)

                # update motion_bases
                bases_param_map = self.model.motion_bases.add_bases(new_bases)
                self.model.fg.set_cluster_ids(cluster_ids)

                # update optimizer
                for param_name, new_params in bases_param_map.items():
                    full_param_name = f"motion_bases.params.{param_name}"
                    optimizer = self.optimizers[full_param_name]
                    # all bases are kept and add more to the end
                    dup_in_optim(
                        optimizer,
                        [new_params],
                        torch.zeros_like(should_clus_split),
                        num_new_bases,
                    )

            num_bases_split = int(should_clus_split.sum().item())
            guru.info(f"Split {num_bases_split} bases.")

        ## Cull bases
        cluster_ids = self.model.fg.get_cluster_ids()
        ids, counts = cluster_ids.unique(return_counts=True)

        # cull bases that are too small, cull all gaussians inside
        should_clus_cull = counts < cfg.cull_bases_threshold * num_fg
        cull_clus_ids = torch.where(should_clus_cull)[0]
        should_cull_from_clus = torch.isin(cluster_ids, cull_clus_ids)

        ## Cull gaussians
        should_cull[:num_fg] |= should_cull_from_clus
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        ## Remap cluster ids
        # update motion bases, clean empty bases
        cluster_ids = self.model.fg.get_cluster_ids()  # ids has been updated
        ids, inv_ids, counts = cluster_ids.unique(return_inverse=True, return_counts=True)
        should_clus_remove = torch.ones(self.model.motion_bases.num_clusters, dtype=torch.bool, device=ids.device)
        should_clus_remove[ids] = False

        # remove empty bases
        bases_param_map = self.model.motion_bases.cull_bases(should_clus_remove)
        for param_name, new_params in bases_param_map.items():
            full_param_name = f"motion_bases.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_clus_remove)

        # remap cluster_ids
        self.model.fg.set_cluster_ids(inv_ids)

        num_bases_culled = should_clus_remove.sum().item() - num_bases_split
        guru.info(
            f"Culled {num_bases_culled} bases, "
            f"Culled {should_cull.sum().item()} gaussians"
        )

        ## Update rigidity map if bases have been changed
        if (
            (self.knn_idx is not None and self.rigid_weights is not None)
            and (num_bases_split > 0 or num_bases_culled > 0)
        ):
            # cluster ids/count just changed (split/cull/remap) -- a cached
            # connectivity or cluster-graph-file graph is now stale (wrong or
            # out-of-range cluster indices), so force a rebuild. Euclidean
            # mode already rebuilds knn_idx from scratch every call and needs
            # no such reset. cluster_graph_file's rebuild will raise if the
            # new cluster count no longer matches the offline graph -- that
            # combination (bases control + a fixed offline graph) isn't
            # supported, use --optim.no-enable-bases-control instead.
            self.connectivity_knn_idx, self.connectivity_valid_mask = None, None
            self.cluster_graph_knn_idx, self.cluster_graph_valid_mask = None, None
            self.update_rigidity_weights()

    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.optim_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold

        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        # Optional hard caps (env vars) to prevent gaussian count from exploding
        # over long schedules: MAX_NUM_GAUSSIANS bounds the total count, and
        # MAX_DENSIFY_PER_STEP bounds how many new gaussians a single control
        # step may add. Splits/dups are net +1 gaussian each, so when the
        # candidate count exceeds the cap we keep only the highest-gradient
        # candidates (the ones the threshold logic is most confident about).
        max_num_gaussians = int(os.environ["MAX_NUM_GAUSSIANS"]) if "MAX_NUM_GAUSSIANS" in os.environ else None
        max_densify_per_step = int(os.environ["MAX_DENSIFY_PER_STEP"]) if "MAX_DENSIFY_PER_STEP" in os.environ else None
        if max_num_gaussians is not None or max_densify_per_step is not None:
            cap = max_densify_per_step
            if max_num_gaussians is not None:
                room = max(0, max_num_gaussians - self.model.num_gaussians)
                cap = room if cap is None else min(cap, room)
            candidate_mask = should_split | should_dup
            num_candidates = int(candidate_mask.sum().item())
            if cap is not None and num_candidates > cap:
                candidate_idx = torch.where(candidate_mask)[0]
                priority = xys_grad_avg[candidate_idx]
                keep_idx = (
                    candidate_idx[torch.topk(priority, cap).indices]
                    if cap > 0
                    else candidate_idx[:0]
                )
                keep_mask = torch.zeros_like(candidate_mask)
                keep_mask[keep_idx] = True
                should_split = should_split & keep_mask
                should_dup = should_dup & keep_mask
                guru.info(
                    f"Densify cap active: {num_candidates} candidates -> kept {cap} "
                    f"(num_gaussians={self.model.num_gaussians}, MAX_NUM_GAUSSIANS={max_num_gaussians}, "
                    f"MAX_DENSIFY_PER_STEP={max_densify_per_step})"
                )

        num_fg = self.model.num_fg_gaussians
        num_bg = self.model.num_bg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())

        should_bg_split = should_split[num_fg:num_fg + num_bg]
        num_bg_splits = int(should_bg_split.sum().item())
        should_bg_dup = should_dup[num_fg:num_fg + num_bg]
        num_bg_dups = int(should_bg_dup.sum().item())

        should_shad_split = should_split[num_fg + num_bg:]
        num_shad_splits = int(should_shad_split.sum().item())
        should_shad_dup = should_dup[num_fg + num_bg:]
        num_shad_dups = int(should_shad_dup.sum().item())

        fg_param_map = self.model.fg.densify_params(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_fg_split,
                num_fg_splits * 2 + num_fg_dups,
            )

        if self.model.bg is not None:
            bg_param_map = self.model.bg.densify_params(should_bg_split, should_bg_dup)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                dup_in_optim(
                    optimizer,
                    [new_params],
                    should_bg_split,
                    num_bg_splits * 2 + num_bg_dups,
                )

        if self.model.shad is not None:
            shad_param_map = self.model.shad.densify_params(should_shad_split, should_shad_dup)
            for param_name, new_params in shad_param_map.items():
                full_param_name = f"shad.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                dup_in_optim(
                    optimizer,
                    [new_params],
                    should_shad_split,
                    num_shad_splits * 2 + num_shad_dups,
                )

        # update running stats
        for k, v in self.running_stats.items():
            v_fg, v_bg, v_shad = v[:num_fg], v[num_fg:num_fg + num_bg], v[num_fg + num_bg:]
            new_v = torch.cat(
                [
                    v_fg[~should_fg_split],
                    v_fg[should_fg_dup],
                    v_fg[should_fg_split].repeat(2),
                    v_bg[~should_bg_split],
                    v_bg[should_bg_dup],
                    v_bg[should_bg_split].repeat(2),
                    v_shad[~should_shad_split],
                    v_shad[should_shad_dup],
                    v_shad[should_shad_split].repeat(2),
                ],
                dim=0,
            )
            self.running_stats[k] = new_v

        split_radius_too_big = is_grad_too_high & is_radius_too_big
        guru.info(
            f"Split {should_split.sum().item()} gaussians ({split_radius_too_big.sum().item()} by radius), "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"Split {should_shad_split.sum().item()} gaussians (shadows), "
            f"Duplicated {should_shad_dup.sum().item()} gaussians (shadows), "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        num_fg = self.model.num_fg_gaussians
        num_bg = self.model.num_bg_gaussians
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )

        # fg bg mask
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:num_fg + num_bg]
        should_shad_cull = should_cull[num_fg + num_bg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        if self.model.bg is not None:
            bg_param_map = self.model.bg.cull_params(should_bg_cull)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                remove_from_optim(optimizer, [new_params], should_bg_cull)

        if self.model.shad is not None:
            shad_param_map = self.model.shad.cull_params(should_shad_cull)
            for param_name, new_params in shad_param_map.items():
                full_param_name = f"shad.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                remove_from_optim(optimizer, [new_params], should_shad_cull)

        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        guru.info(
            f"Culled {should_cull.sum().item()} gaussians ({is_radius_too_big.sum().item()} by radius), "
            f"Culled {should_shad_cull.sum().item()} gaussians (shadows), "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        for part in ["fg", "bg"]:
            part_params = getattr(self.model, part).reset_opacities(new_val)
            # Modify optimizer states by new assignment.
            for param_name, new_params in part_params.items():
                full_param_name = f"{part}.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                reset_in_optim(optimizer, [new_params])
        guru.info("Reset opacities")

    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.model.named_parameters():
            name_fields = name.split(".")
            part, field = name_fields[0], name_fields[-1]
            if len(name_fields) > 1 and name_fields[1] == "gnn":
                # flow3d/graph_coupling.py: GraphCorrectedScalableMotionBases's
                # GNN params (e.g. "motion_bases.gnn.encoder.0.weight") aren't a
                # "part.params.field" leaf, so they're not in lr_dict -- give
                # them their own fixed-lr param group instead.
                lr = self.optim_cfg.gnn_lr
            else:
                lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers


def add_in_optim(optimizer, new_params: list, dim=0):
    """
    Add more elements in the optimizer along a given dimension
    """
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return

        # check new dimension
        new_dim = p_new.shape[dim] - old_params.shape[dim]
        assert new_dim > 0
        new_shape = p_new.shape[:dim] + (new_dim,) + p_new.shape[dim + 1:]  # new_shape: [..., new_dim, ...]

        # modify the states
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat([p, p.new_zeros(new_shape)], dim=dim)

        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()

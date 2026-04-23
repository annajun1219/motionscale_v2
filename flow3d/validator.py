import functools
import os
import os.path as osp
import time
from dataclasses import asdict
from typing import cast
import json

import imageio as iio
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fnc
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flow3d.data.utils import normalize_coords, to_device, to_serializable
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.loss_utils import knn_query
from flow3d.vis.utils import (
    apply_depth_colormap,
    make_video_divisble,
    plot_correspondences,
)


class Validator:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        train_loader: DataLoader | None,
        val_img_loader: DataLoader | None,
        val_kpt_loader: DataLoader | None,
        save_dir: str,
        save_as_imgs: bool = False,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_img_loader = val_img_loader
        self.val_kpt_loader = val_kpt_loader
        self.save_dir = save_dir
        self.has_bg = self.model.has_bg if self.model is not None else False
        self.save_as_imgs = save_as_imgs

        # metrics
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS().to(device)
        self.fg_psnr_metric = mPSNR()
        self.fg_ssim_metric = mSSIM()
        self.fg_lpips_metric = mLPIPS().to(device)
        self.bg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS().to(device)
        self.pck_metric = PCK()
        self.metrics = {}

    def reset_metrics(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()
        self.fg_psnr_metric.reset()
        self.fg_ssim_metric.reset()
        self.fg_lpips_metric.reset()
        self.bg_psnr_metric.reset()
        self.bg_ssim_metric.reset()
        self.bg_lpips_metric.reset()
        self.pck_metric.reset()

    def save_metrics(self, metrics):
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(metric_value)

    def get_metrics(self):
        # compute average, max, min of each metric
        summary = {}
        for metric_name, values_list in self.metrics.items():
            if len(values_list) > 0:
                summary[f"{metric_name}_max"] = float(np.max(values_list))
                # summary[f"{metric_name}_min"] = float(np.min(values_list))
                summary[f"{metric_name}_mean"] = float(np.mean(values_list))

        return summary

    @torch.no_grad()
    def eval_metrics(self):
        self.reset_metrics()

        if self.train_loader is None:
            return

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Evaluating train images", leave=False)):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            # ().
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = torch.ones_like(batch["imgs"][..., 0])  # don't mask edges
            # (1, H, W).
            fg_mask = batch["fg_masks"]

            img_wh = img[0].shape[-2::-1]
            rendered = self.model.render(t, w2c, K, img_wh)

            # Compute metrics.
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)

                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

        metrics = {
            "psnr": self.psnr_metric.compute().item(),
            "ssim": self.ssim_metric.compute().item(),
            "lpips": self.lpips_metric.compute().item(),
            "fg_psnr": self.fg_psnr_metric.compute().item(),
            "fg_ssim": self.fg_ssim_metric.compute().item(),
            "fg_lpips": self.fg_lpips_metric.compute().item(),
            "bg_psnr": self.bg_psnr_metric.compute().item(),
            "bg_ssim": self.bg_ssim_metric.compute().item(),
            "bg_lpips": self.bg_lpips_metric.compute().item(),
        }
        self.save_metrics(metrics)
        self.reset_metrics()
        return metrics

    @torch.no_grad()
    def validate(self):
        self.reset_metrics()
        metric_imgs = self.validate_imgs() or {}
        metric_kpts = self.validate_keypoints() or {}
        return {**metric_imgs, **metric_kpts}

    @torch.no_grad()
    def validate_imgs(self):
        guru.info("rendering validation images...")
        if self.val_img_loader is None:
            return

        for batch in tqdm(self.val_img_loader, desc="render val images"):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch["masks"]

            # (H, W).
            covisible_mask = batch.get(
                "covisible_masks",
                torch.ones_like(fg_mask)[None],
            )
            W, H = img_wh = img[0].shape[-2::-1]
            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True)

            # Compute metrics.
            valid_mask *= covisible_mask
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)

                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(
                osp.join(results_dir, f"{frame_name}.jpg"),
                (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
            )

            # co-visibility mask
            mask_color = torch.tensor([0.0, 1.0, 0.0]).float()
            blend_img = rendered["img"][0].cpu().clone()
            invalid_mask = ~(valid_mask.squeeze(0) > 0).cpu()
            blend_img[invalid_mask] = 0.5 * blend_img[invalid_mask] + 0.5 * mask_color
            results_dir = osp.join(self.save_dir, "results", "masked_rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(osp.join(results_dir, f"{frame_name}.jpg"), (blend_img.numpy() * 255).astype(np.uint8))

        return {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }

    @torch.no_grad()
    def validate_keypoints(self):
        if self.val_kpt_loader is None:
            return
        pred_keypoints_3d_all = []
        time_ids = self.val_kpt_loader.dataset.time_ids.tolist()
        h, w = self.val_kpt_loader.dataset.dataset.imgs.shape[1:3]
        pred_train_depths = np.zeros((len(time_ids), h, w))

        for batch in tqdm(self.val_kpt_loader, desc="render val keypoints"):
            batch = to_device(batch, self.device)
            # (2,).
            ts = batch["ts"][0]
            # (2, 4, 4).
            w2cs = batch["w2cs"][0]
            # (2, 3, 3).
            Ks = batch["Ks"][0]
            # (2, H, W, 3).
            imgs = batch["imgs"][0]
            # (2, P, 3).
            keypoints = batch["keypoints"][0]
            # (P,)
            keypoint_masks = (keypoints[..., -1] > 0.5).all(dim=0)
            src_keypoints, target_keypoints = keypoints[:, keypoint_masks, :2]
            W, H = img_wh = imgs.shape[-2:0:-1]
            rendered = self.model.render(
                ts[0].item(),
                w2cs[:1],
                Ks[:1],
                img_wh,
                target_ts=ts[1:],
                target_w2cs=w2cs[1:],
                return_depth=True,
            )
            pred_tracks_3d = rendered["tracks_3d"][0, ..., 0, :]
            pred_tracks_2d = torch.einsum("ij,hwj->hwi", Ks[1], pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., -1:], min=1e-6
            )
            pred_keypoints = F.grid_sample(
                pred_tracks_2d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]

            # Compute metrics.
            self.pck_metric.update(pred_keypoints, target_keypoints, max(img_wh) * 0.05)

            padded_keypoints_3d = torch.zeros_like(keypoints[0])
            pred_keypoints_3d = F.grid_sample(
                pred_tracks_3d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]
            # Transform 3D keypoints back to world space.
            pred_keypoints_3d = torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(w2cs[1])[:3],
                F.pad(pred_keypoints_3d, (0, 1), value=1.0),
            )
            padded_keypoints_3d[keypoint_masks] = pred_keypoints_3d
            # Cache predicted keypoints.
            pred_keypoints_3d_all.append(padded_keypoints_3d.cpu().numpy())
            pred_train_depths[time_ids.index(ts[0].item())] = (
                rendered["depth"][0, ..., 0].cpu().numpy()
            )

        # Dump unified results.
        all_Ks = self.val_kpt_loader.dataset.dataset.Ks
        all_w2cs = self.val_kpt_loader.dataset.dataset.w2cs

        keypoint_result_dict = {
            "Ks": all_Ks[time_ids].cpu().numpy(),
            "w2cs": all_w2cs[time_ids].cpu().numpy(),
            "pred_keypoints_3d": np.stack(pred_keypoints_3d_all, 0),
            "pred_train_depths": pred_train_depths,
        }

        results_dir = osp.join(self.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            osp.join(results_dir, "keypoints.npz"),
            **keypoint_result_dict,
        )
        guru.info(
            f"Dumped keypoint results to {results_dir=} {keypoint_result_dict['pred_keypoints_3d'].shape=}"
        )

        return {"val/pck": self.pck_metric.compute()}


    @torch.no_grad()
    def save_train_videos(self, epoch=None, out_name=None):
        if self.train_loader is None:
            return
        assert (epoch is not None) or (out_name is not None)
        out_name = f"epoch_{epoch:04d}" if out_name is None else out_name
        video_dir = osp.join(self.save_dir, "videos", out_name)
        os.makedirs(video_dir, exist_ok=True)
        fps = getattr(self.train_loader.dataset.dataset, "fps", 15.0)
        input_palette = self.train_loader.dataset.dataset.segment_palette
        palette_torch = torch.tensor(input_palette, dtype=torch.uint8).reshape(-1, 3)

        os.makedirs(osp.join(video_dir, "rgbs"), exist_ok=True)
        os.makedirs(osp.join(video_dir, "masks"), exist_ok=True)

        ## Render video.
        video = []
        ref_pred_depths = []
        video_masks = []
        frame_names = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            #
            frame_name = batch["frame_names"][0]
            frame_names.append(frame_name)
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]
            # (H, W).
            depth = batch["depths"][0]
            # (H, W)
            masks = batch["masks"][0]

            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True,
            )

            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            # rgb
            # video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
            rendered_img = (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8)
            iio.imwrite(osp.join(video_dir, "rgbs", f"{frame_name}.jpg"), rendered_img)
            # depth
            ref_pred_depth = torch.cat((depth.unsqueeze(-1), rendered["depth"][0]), dim=1).cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())

            # mask
            if rendered["mask"] is not None:
                # video_masks.append(torch.cat([masks, rendered["mask"][0].squeeze(-1)], dim=1).cpu())
                rendered_mask = (rendered["mask"][0].squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
                iio.imwrite(osp.join(video_dir, "masks", f"{frame_name}.jpg"), rendered_mask)

        ## Save results
        # depth video
        def process_depth(x, near, far):
            x = apply_depth_colormap(x, near_plane=near, far_plane=far)
            return (x.numpy() * 255).astype(np.uint8)

        self.save_frames(
            output_dir=video_dir,
            data_type="depths",
            data_list=ref_pred_depths,
            apply_func=lambda x: process_depth(x, depth_min, depth_max),
            save_as_imgs=self.save_as_imgs,
        )

        ## Render 2D track video.
        tracks_2d, target_imgs = [], []
        sample_interval = 8
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        # ().
        t = batch0["ts"]
        # (4, 4).
        w2c = batch0["w2cs"]
        # (3, 3).
        K = batch0["Ks"]
        # (H, W, 3).
        img = batch0["imgs"]
        # (H, W).
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        for batch in tqdm(
            self.train_loader, desc="Rendering 2D track video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            # (1, H, W, 3).
            target_imgs.append(batch["imgs"].cpu())
            # (1,).
            target_ts = batch["ts"]
            # (1, 4, 4).
            target_w2cs = batch["w2cs"]
            # (1, 3, 3).
            target_Ks = batch["Ks"]
            rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                target_ts=target_ts,
                target_w2cs=target_w2cs,
                fg_only=True,
            )
            pred_tracks_3d = rendered["tracks_3d"][0][
                ::sample_interval, ::sample_interval
            ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., 2:], min=1e-6
            )
            tracks_2d.append(pred_tracks_2d.cpu())
        tracks_2d = torch.cat(tracks_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        track_2d_video = plot_correspondences(
            target_imgs.numpy(),
            tracks_2d.numpy(),
            query_id=cast(int, t),
        )

        # save track video
        self.save_frames(
            output_dir=video_dir,
            data_type="tracks_2d",
            data_list=track_2d_video,
            apply_func=None,
            save_as_imgs=self.save_as_imgs,
        )

        ## Render motion coefficient video.
        # assign colors according to motion bases
        palette = palette_torch.to(self.device)
        if hasattr(self.model.fg, "cluster_ids"):
            cluster_ids = self.model.fg.get_cluster_ids()
            motion_coef_colors = (palette[cluster_ids, :] / 255.0).float()
        else:
            coefs = self.model.fg.get_coefs()
            coefs_max_idx = torch.argmax(coefs, dim=1)
            motion_coef_colors = (palette[coefs_max_idx, :] / 255.0).float()

        if self.model.has_bg:
            motion_coef_colors = F.pad(
                motion_coef_colors, (0, 0, 0, self.model.num_bg_gaussians + self.model.num_shad_gaussians), value=1.0
            )
        video = []
        for batch in tqdm(
            self.train_loader, desc="Rendering motion coefficient video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (3, 3).
            img = batch["imgs"][0]
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, colors_override=motion_coef_colors
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())

        # motion coefficients video
        self.save_frames(
            output_dir=video_dir,
            data_type="motion_coefs",
            data_list=video,
            apply_func=lambda x: (x.numpy() * 255).astype(np.uint8),
            save_as_imgs=self.save_as_imgs,
        )


    def save_frames(self, output_dir, data_type, data_list, apply_func=None, fps=15, save_as_imgs=False):
        """
        Stack a list of frame data and save them as a video or images.
        """
        if save_as_imgs:
            output_dir = osp.join(output_dir, data_type)
            os.makedirs(output_dir, exist_ok=True)

        # save frames as images
        video = []
        for frame_id, frame_data in enumerate(tqdm(data_list, desc=f"Saving {data_type}", leave=False)):
            img = apply_func(frame_data) if apply_func is not None else frame_data
            video.append(img)
            if save_as_imgs:
                iio.imwrite(osp.join(output_dir, f"{frame_id:04d}.jpg"), img)

        # save as one video
        video = np.stack(video, axis=0)
        if not save_as_imgs:
            output_path = osp.join(output_dir, f"{data_type}.mp4")
            iio.mimwrite(output_path, make_video_divisble(video), fps=fps)

        return video

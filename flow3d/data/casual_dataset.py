import os
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import cv2
import imageio
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tyro
from loguru import logger as guru
from roma import roma
from tqdm import tqdm
import matplotlib.pyplot as plt

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    UINT16_MAX,
    SceneNormDict,
    get_tracks_3d_for_query_frame,
    median_filter_2d,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
    parse_cotracker3_track_info,
    depth_map_edge,
    depth_map_aliasing,
)
from flow3d.transforms import rt_to_mat4


@dataclass
class DavisDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = "480p"
    image_type: str = "JPEGImages"
    mask_type: str = "Segmentation"
    depth_type: str = "aligned_depth_anything"
    camera_type: Literal["droid_recon", "megasam"] = "megasam"
    slam_type: str = "megasam/outputs_cvd"
    track_2d_type: str = "cotracker3"
    normals_type: str | None = None
    mask_erosion_radius: int = 3
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    shadow_type: str | None = None


@dataclass
class CustomDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = ""
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: str = "aligned_depth_anything"
    camera_type: Literal["droid_recon", "megasam"] = "megasam"
    slam_type: str = "megasam/outputs_cvd"
    track_2d_type: str = "cotracker3"
    normals_type: str | None = None
    mask_erosion_radius: int = 7
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    shadow_type: str | None = None


class CasualDataset(BaseDataset):
    def __init__(
        self,
        seq_name: str,
        root_dir: str,
        start: int = 0,
        end: int = -1,
        res: str = "480p",
        image_type: str = "JPEGImages",
        mask_type: str = "Segmentation",
        depth_type: str = "aligned_depth_anything",
        camera_type: Literal["droid_recon", "megasam"] = "megasam",
        slam_type: str = "megasam/outputs_cvd",
        track_2d_type: str = "cotracker3",
        normals_type: str | None = None,
        mask_erosion_radius: int = 3,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        shadow_type: str | None = None,
        **_,
    ):
        super().__init__()

        self.seq_name = seq_name
        self.root_dir = root_dir
        self.res = res
        self.depth_type = depth_type
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius
        self.track_2d_type = track_2d_type

        self.img_dir = os.path.join(root_dir, image_type, res, seq_name)
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        preproc_dir = os.path.join(root_dir, "flow3d_preprocessed", res, seq_name)
        self.depth_dir = os.path.join(preproc_dir, depth_type)
        self.mask_dir = os.path.join(preproc_dir, mask_type)
        self.tracks_dir = os.path.join(preproc_dir, track_2d_type)
        self.cache_dir = os.path.join(preproc_dir, "cache")
        self.slam_dir = os.path.join(preproc_dir, slam_type)
        self.normals_dir = os.path.join(preproc_dir, normals_type) if normals_type is not None else None
        self.shadow_dir = os.path.join(preproc_dir, shadow_type) if shadow_type is not None else None

        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        self.frame_names = frame_names[start:end]

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.fg_masks: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depth_masks: list[torch.Tensor | None] = [None for _ in self.frame_names]

        # load cameras
        img = self.get_image(0)
        H, W = img.shape[:2]
        if camera_type == "megasam":
            w2cs, Ks, tstamps = load_cameras_megasam(os.path.join(self.slam_dir, f"{seq_name}.npz"), H, W)
        elif camera_type == "droid_recon":
            w2cs, Ks, tstamps = load_cameras(os.path.join(preproc_dir, f"{camera_type}.npy"), H, W)
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        # load depths
        if depth_type == "megasam":
            self.depths = load_depths_megasam(os.path.join(self.slam_dir, f"{seq_name}.npz"), H, W)
        else:
            # use default depths under self.depth_dir
            pass

        # load normals
        if self.normals_dir is not None:
            # (N, H, W, 3)
            self.normals = torch.stack(
                [
                    torch.from_numpy(np.load(os.path.join(self.normals_dir, f"{frame_name}.npy")))
                    for frame_name in self.frame_names
                ]
            )
            # normal mask
            self.normal_masks = self.normals.norm(dim=-1, keepdim=True) > 0.1
            self.normals = F.normalize(self.normals, dim=-1)
        else:
            self.normals = None
            self.normal_masks = None

        # load shadow masks
        if self.shadow_dir is not None and os.path.exists(self.shadow_dir):
            # (N, H, W)
            self.shadows = torch.stack(
                [
                    torch.from_numpy(np.array(
                        Image.open(os.path.join(self.shadow_dir, f"{frame_name}.png"))
                    ))
                    for frame_name in self.frame_names
                ]
            )
            self.shadows = self.shadows > 0
        else:
            self.shadows = None

        assert (
            len(frame_names) == len(w2cs) == len(Ks)
        ), f"{len(frame_names)}, {len(w2cs)}, {len(Ks)}"
        self.w2cs = w2cs[start:end]
        self.Ks = Ks[start:end]
        tstamps = torch.from_numpy(np.arange(0, end))
        tmask = (tstamps >= start) & (tstamps < end)
        self._keyframe_idcs = tstamps[tmask] - start
        self.scale = 1

        if scene_norm_dict is None:
            cached_scene_norm_dict_path = os.path.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if os.path.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                guru.info("loading cached scene norm dict...")
                scene_norm_dict = torch.load(
                    os.path.join(self.cache_dir, "scene_norm_dict.pth")
                )
            else:
                tracks_3d = self.get_tracks_3d(5000, step=self.num_frames // 10)[0]
                scale, transfm = compute_scene_norm(tracks_3d, self.w2cs)
                scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(scene_norm_dict, cached_scene_norm_dict_path)

        # transform cameras
        self.scene_norm_dict = cast(SceneNormDict, scene_norm_dict)
        self.scale = self.scene_norm_dict["scale"]
        transform = self.scene_norm_dict["transfm"]
        guru.info(f"scene norm {self.scale=}, {transform=}")
        self.w2cs = torch.einsum("nij,jk->nik", self.w2cs, torch.linalg.inv(transform))
        self.w2cs[:, :3, 3] /= self.scale

        # Apply transform to normals
        if self.normals is not None:
            self.normals = torch.einsum("nij,nhwj->nhwi", torch.linalg.inv(self.w2cs)[:, :3, :3], self.normals)
            self.normals = F.normalize(self.normals, dim=-1)


    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return self._keyframe_idcs

    def __len__(self):
        return len(self.frame_names)

    def parse_track_info(self, visibility, confidence):
        if self.track_2d_type in ["bootstapir", "tapir"]:
            return parse_tapir_track_info(visibility, confidence)
        elif self.track_2d_type == "cotracker3":
            return parse_cotracker3_track_info(visibility, confidence)
        else:
            raise ValueError(f"Unknown track type: {self.track_2d_type}")

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img

    def get_mask(self, index) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        mask = cast(torch.Tensor, self.masks[index])
        return mask

    def get_shadow(self, index) -> torch.Tensor:
        return self.shadows[index] if self.shadows is not None else None

    def get_normal(self, index) -> torch.Tensor:
        return self.normals[index] if self.normals is not None else None

    def get_fg_mask(self, index):
        if self.fg_masks[index] is None:
            self.fg_masks[index] = self.load_mask(index, use_erosion=False)
        return self.fg_masks[index]

    def get_depth(self, index) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index] / self.scale

    def get_depth_mask(self, index) -> torch.Tensor:
        if self.depth_masks[index] is None:
            self.depth_masks[index] = self.compute_depth_mask(index)
        depth_mask = cast(torch.Tensor, self.depth_masks[index])
        return depth_mask

    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0

    def load_mask(self, index, use_erosion=True) -> torch.Tensor:
        path = f"{self.mask_dir}/{self.frame_names[index]}.png"
        r = self.mask_erosion_radius
        mask = imageio.imread(path)
        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        bg_mask = ~fg_mask

        # Return a simple boolean mask without erode
        if not use_erosion:
            return torch.from_numpy(fg_mask).bool()

        # shrink the mask inward a bit by cv2.erode
        fg_mask_erode = cv2.erode(
            fg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        bg_mask_erode = cv2.erode(
            bg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        # the out mask has a zeros region covering small edges of both fg and bg
        out_mask = np.zeros_like(fg_mask, dtype=np.float32)
        out_mask[bg_mask_erode > 0] = -1
        out_mask[fg_mask_erode > 0] = 1
        return torch.from_numpy(out_mask).float()

    def load_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        disp = np.load(path)
        depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
        depth = torch.from_numpy(depth).float()
        depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
        return depth

    def compute_depth_mask(self, index) -> torch.Tensor:
        depth = self.get_depth(index)
        bool_mask = (depth > 0).to(torch.bool)
        non_edge = ~depth_map_aliasing(depth, rtol=0.02, kernel_size=5)
        bool_mask = bool_mask & non_edge
        return bool_mask

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1, return_vis: bool = False
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """
        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)

        # visible, confidence
        if return_vis:
            all_tracks = torch.from_numpy(np.stack(all_tracks, axis=dim))
            valid_visible, valid_invisible, confidence = self.parse_track_info(all_tracks[..., 2], all_tracks[..., 3])
            return all_tracks, valid_visible, valid_invisible, confidence

        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))
        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        depth_masks = torch.stack([self.get_depth_mask(i) for i in target_idcs], dim=0)
        fg_masks = (masks == 1).float() * (depth_masks == 1).float()
        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
        c2ws = torch.linalg.inv(self.w2cs[target_idcs])

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0
        tracks_all_queries = []
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )
            if num_sel < len(tracks_2d):
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]
            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            tidx = target_idcs.index(q_idx)
            tracks_tuple = get_tracks_3d_for_query_frame(
                tidx, img, tracks_2d, depths, fg_masks, inv_Ks, c2ws, track_type=self.track_2d_type,
            )
            tracks_all_queries.append(tracks_tuple)
        tracks_3d, colors, visibles, invisibles, confidences, depths = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )
        return tracks_3d, visibles, invisibles, confidences, colors, depths


    def sample_bkgd_points(
        self,
        num_samples: int | None = None,
        start: int = 0,
        end: int = -1,
        sample_stride: int = 1,
        bg_input = None,
        use_kf_tstamps: bool = False,
        stride: int = 1,
        down_rate: int | None = None,
        min_per_frame: int = 64,
        erode_radius = 3,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert erode_radius % 2 == 1
        if end == -1:
            end = self.num_frames

        # sample pixels in consistent grids
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )
        grid_mask = torch.zeros((H, W), dtype=torch.bool)
        grid_mask[::sample_stride, ::sample_stride] = True

        if use_kf_tstamps:
            query_idcs = self.keyframe_idcs.tolist()
        else:
            query_idcs = list(range(start, end, stride))

        # existing background points
        if bg_input is not None:
            N = bg_input.shape[0]
            bg_points = bg_input.clone().float()
            bg_normals = torch.zeros((N, 3), dtype=torch.float32)
            bg_colors = torch.zeros((N, 3), dtype=torch.float32)
        else:
            bg_points = torch.empty((0, 3), dtype=torch.float32)
            bg_normals = torch.empty((0, 3), dtype=torch.float32)
            bg_colors = torch.empty((0, 3), dtype=torch.float32)

        # sample points in empty regions
        for i, query_idx in enumerate(tqdm(query_idcs, desc="Sampling bkgd points")):
            img = self.get_image(query_idx)
            depth = self.get_depth(query_idx)
            normal = self.get_normal(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            depth_mask = self.get_depth_mask(query_idx)
            bool_mask = (bg_mask * depth_mask * (depth > 0)).to(torch.bool)
            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]

            # mask of empty regions
            if len(bg_points) > 0:
                bg_points_3d = torch.einsum(
                    "ij,pj->pi",
                    w2c[:3, :],
                    F.pad(bg_points, (0, 1), value=1.0),
                )
                bg_points_2d = torch.einsum("ij,pj->pi", K, bg_points_3d)
                bg_points_2d = bg_points_2d[..., :2] / torch.clamp(bg_points_2d[..., 2:], min=1e-6)

                # project onto image grids
                u, v = bg_points_2d[:, 0], bg_points_2d[:, 1]
                visible = (bg_points_3d[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                u, v = u[visible], v[visible]

                # occlusion mask
                empty_mask = torch.ones((H, W), dtype=torch.bool)
                empty_mask[v.long(), u.long()] = False

                # erode the mask a bit to remove noise
                bool_mask = bool_mask & empty_mask
                bool_mask = -F.max_pool2d(
                    -bool_mask[None, None].float(), kernel_size=erode_radius, stride=1, padding=erode_radius // 2
                )
                bool_mask = bool_mask[0, 0].bool()

            # apply grid mask
            bool_mask = bool_mask & grid_mask

            # get sampled 3D points via projection
            points = torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(K),
                F.pad(grid[bool_mask], (0, 1), value=1.0),
            ) * depth[bool_mask][:, None]
            points = torch.einsum("ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0))
            point_normals = normal[bool_mask] if normal is not None else normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]

            # add to existing bg points
            bg_points = torch.cat([bg_points, points], dim=0)
            bg_normals = torch.cat([bg_normals, point_normals], dim=0)
            bg_colors = torch.cat([bg_colors, point_colors], dim=0)

            guru.debug(f"{query_idx=} {points.shape=}")

        if bg_input is not None:
            bg_points, bg_normals, bg_colors = bg_points[N:], bg_normals[N:], bg_colors[N:]

        if num_samples is None and down_rate is not None:
            num_samples = int(len(bg_points) / down_rate)

        if num_samples is not None and len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]

        return bg_points, bg_normals, bg_colors


    def get_bkgd_points(
        self,
        num_samples: int,
        start: int = 0,
        end: int = -1,
        sample_masks=None,
        use_kf_tstamps: bool = True,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        return_lists: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if end == -1:
            end = self.num_frames
        num_sample_frames = end - start
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )

        if use_kf_tstamps:
            query_idcs = self.keyframe_idcs.tolist()
        else:
            query_idcs = list(range(start, end, stride))
            # num_query_frames = num_sample_frames // stride
            # query_endpts = torch.linspace(start, end, num_query_frames + 1)
            # query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        for i, query_idx in enumerate(tqdm(query_idcs, desc="Loading bkgd points", leave=False)):
            img = self.get_image(query_idx)
            depth = self.get_depth(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            depth_mask = self.get_depth_mask(query_idx)
            bool_mask = (bg_mask * depth_mask * (depth > 0)).to(torch.bool)
            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]
            sample_mask = sample_masks[i] if sample_masks is not None else None

            # get the mask of area to sample on this frame
            if sample_mask is not None:
                overlap_mask = sample_mask
            else:
                # get the bounding box of previous points that reproject into frame
                # inefficient but works for now
                bmax_x, bmax_y, bmin_x, bmin_y = 0, 0, W, H
                for p3d, _, _ in bg_geometry:
                    if len(p3d) < 1:
                        continue
                    # reproject into current frame
                    p2d = torch.einsum(
                        "ij,jk,pk->pi", K, w2c[:3], F.pad(p3d, (0, 1), value=1.0)
                    )
                    p2d = p2d[:, :2] / p2d[:, 2:].clamp(min=1e-6)
                    xmin, xmax = p2d[:, 0].min().item(), p2d[:, 0].max().item()
                    ymin, ymax = p2d[:, 1].min().item(), p2d[:, 1].max().item()

                    bmin_x = min(bmin_x, int(xmin))
                    bmin_y = min(bmin_y, int(ymin))
                    bmax_x = max(bmax_x, int(xmax))
                    bmax_y = max(bmax_y, int(ymax))

                # don't include points that are covered by previous points
                bmin_x = max(0, bmin_x)
                bmin_y = max(0, bmin_y)
                bmax_x = min(W, bmax_x)
                bmax_y = min(H, bmax_y)
                overlap_mask = torch.ones_like(bool_mask)
                overlap_mask[bmin_y:bmax_y, bmin_x:bmax_x] = 0

            bool_mask &= overlap_mask
            if bool_mask.sum() < min_per_frame:
                guru.debug(f"skipping {query_idx=}")
                continue

            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]

            num_sel = max(len(points) // down_rate, min_per_frame)
            sel_idcs = np.random.choice(len(points), num_sel, replace=False)
            points = points[sel_idcs]
            point_normals = point_normals[sel_idcs]
            point_colors = point_colors[sel_idcs]
            guru.debug(f"{query_idx=} {points.shape=}")
            bg_geometry.append((points, point_normals, point_colors))

        if return_lists:
            bg_points, bg_normals, bg_colors = map(list, zip(*bg_geometry))
            return bg_points, bg_normals, bg_colors

        bg_points, bg_normals, bg_colors = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]

        return bg_points, bg_normals, bg_colors

    def __getitem__(self, index):
        target_inds = None
        if isinstance(index, int):
            index, target_start, target_end = index, 0, self.num_frames
        elif isinstance(index, tuple):
            if isinstance(index[1], (list, np.ndarray)):
                index, target_inds = index[0], torch.as_tensor(index[1])
            else:
                index, target_start, target_end = index
        else:
            raise ValueError(f"index {index} is not supported.")

        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": torch.tensor(index),
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.get_image(index),
            "depths": self.get_depth(index),
        }
        # eroded fg masks
        tri_mask = self.get_mask(index)
        valid_mask = tri_mask != 0  # not fg or bg
        mask = tri_mask == 1  # fg mask
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()
        data["depth_masks"] = self.get_depth_mask(index).float()

        # (H, W, 3)
        if self.normals is not None:
            data["normals"] = self.get_normal(index)
            data["normal_masks"] = self.normal_masks[index]

        # (H, W)
        if self.shadows is not None:
            data["shadows"] = self.get_shadow(index)

        # raw fg masks
        data["fg_masks"] = self.get_fg_mask(index).float()

        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        if target_inds is None:
            target_inds = torch.from_numpy(
                np.random.choice(
                    np.arange(target_start, target_end),
                    (self.num_targets_per_frame,),
                    replace=self.num_targets_per_frame >= (target_end - target_start),
                )
            )
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)
        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]
        # (N, P).
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = self.parse_track_info(target_tracks[..., 2], target_tracks[..., 3])

        # target track depths
        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        # (N, P)
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]

        # target depth masks
        target_masks = torch.stack([self.get_depth_mask(i) for i in target_inds], dim=0).float()
        # (N, P)
        data["target_track_masks"] = F.grid_sample(
            target_masks[:, None],  # (N, 1, H, W)
            normalize_coords(target_tracks[..., None, :2], H, W),  # (N, P, 1, 2)
            align_corners=True,
        )[:, 0, :, 0] == 1  # (N, P)

        return data


class CasualDatasetVideoView(Dataset):
    """Return a dataset view of the video trajectory."""

    def __init__(self, dataset: CasualDataset):
        super().__init__()
        self.dataset = dataset
        self.fps = self.dataset.fps if hasattr(self.dataset, "fps") else 15

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, index):
        data = {
            "frame_names": self.dataset.frame_names[index],
            "ts": torch.tensor(index),
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.dataset.get_image(index),
            "depths": self.dataset.get_depth(index),
        }

        tri_mask = self.dataset.get_mask(index)
        valid_mask = tri_mask != 0  # not fg or bg
        mask = tri_mask == 1  # fg mask
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()

        # raw fg masks
        data["fg_masks"] = self.dataset.get_fg_mask(index).float()

        return data


def load_cameras_megasam(
    path: str, H: int, W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert os.path.exists(path), f"Camera file {path} does not exist."
    recon = np.load(path)
    traj_c2w = recon["cam_c2w"]
    h, w = recon["images"].shape[1:3]
    K_ = recon["intrinsic"]
    fx, fy, cx, cy = K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2]
    kf_tstamps = np.arange(recon["images"].shape[0])

    sy, sx = H / h, W / w
    traj_w2c = np.linalg.inv(traj_c2w)
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
    Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)

    return (
        torch.from_numpy(traj_w2c).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(kf_tstamps),
    )


def load_depths_megasam(
    path: str, H: int, W: int, depth_min: float = 1e-3, depth_max: float = 100.0
) -> torch.Tensor:
    data = np.load(path)
    depths = data["depths"].copy()
    scale = float(data["normalize_scale"]) if "normalize_scale" in data else 1.0
    assert scale > 0, f"normalize_scale must be positive, got {scale}"
    # set invalid pixels (e.g. sky) to depth_max expressed in DROID scale
    if "valid_masks" in data:
        depths[~data["valid_masks"]] = depth_max * scale
    depths = np.clip(depths, depth_min * scale, depth_max * scale)
    depths = F.interpolate(
        torch.from_numpy(depths).float().unsqueeze(1),
        size=(H, W), mode="nearest-exact",
    )
    return depths.squeeze()


def load_cameras(
    path: str, H: int, W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert os.path.exists(path), f"Camera file {path} does not exist."
    recon = np.load(path, allow_pickle=True).item()
    guru.debug(f"{recon.keys()=}")
    traj_c2w = recon["traj_c2w"]  # (N, 4, 4)
    h, w = recon["img_shape"]
    fx, fy, cx, cy = recon["intrinsics"]  # (4,)
    kf_tstamps = recon["tstamps"].astype("int")

    sy, sx = H / h, W / w
    traj_w2c = np.linalg.inv(traj_c2w)
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
    Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)

    return (
        torch.from_numpy(traj_w2c).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(kf_tstamps),
    )


def compute_scene_norm(
    X: torch.Tensor, w2cs: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """
    :param X: [N*T, 3]
    :param w2cs: [N, 4, 4]
    """
    X = X.reshape(-1, 3)
    scene_center = X.mean(dim=0)
    X = X - scene_center[None]
    min_scale = X.quantile(0.05, dim=0)
    max_scale = X.quantile(0.95, dim=0)
    scale = (max_scale - min_scale).max().item() / 2.0
    original_up = -F.normalize(w2cs[:, 1, :3].mean(0), dim=-1)
    target_up = original_up.new_tensor([0.0, 0.0, 1.0])
    R = roma.rotvec_to_rotmat(
        F.normalize(original_up.cross(target_up), dim=-1)
        * original_up.dot(target_up).acos_()
    )
    transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
    return scale, transfm


if __name__ == "__main__":
    d = CasualDataset("bear", "/shared/vye/datasets/DAVIS", camera_type="droid_recon")

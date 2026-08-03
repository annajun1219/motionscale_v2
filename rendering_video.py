import os
import os.path as osp
import re
from dataclasses import dataclass

import imageio.v3 as iio
import numpy as np
import tyro
from loguru import logger as guru

from flow3d.vis.utils import make_video_divisble


@dataclass
class VideoConfig:
    # Run directory, e.g.
    # outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3
    work_dir: str
    # Epoch subfolder under {work_dir}/videos to render. If None, uses the
    # latest available epoch.
    epoch: str | None = None
    # Which rendered output to turn into a video: rgbs, depths, masks,
    # tracks_2d, or motion_coefs.
    data_type: str = "rgbs"
    fps: int = 15
    # Output mp4 path. If None, saved next to the source frames.
    out_path: str | None = None


def get_latest_epoch_dir(videos_dir: str) -> str:
    epoch_dirs = [d for d in os.listdir(videos_dir) if d.startswith("epoch_")]
    assert len(epoch_dirs) > 0, f"No epoch_* dirs found in {videos_dir}"
    epoch_dirs.sort(key=lambda d: int(re.search(r"\d+", d).group()))
    return epoch_dirs[-1]


def main(cfg: VideoConfig):
    videos_dir = osp.join(cfg.work_dir, "videos")
    epoch_dir_name = cfg.epoch if cfg.epoch is not None else get_latest_epoch_dir(videos_dir)
    frames_dir = osp.join(videos_dir, epoch_dir_name, cfg.data_type)
    assert osp.isdir(frames_dir), f"{frames_dir} does not exist"

    frame_names = sorted(os.listdir(frames_dir))
    assert len(frame_names) > 0, f"No frames found in {frames_dir}"

    guru.info(f"Reading {len(frame_names)} frames from {frames_dir}")
    frames = np.stack([iio.imread(osp.join(frames_dir, name)) for name in frame_names], axis=0)
    frames = make_video_divisble(frames)

    out_path = cfg.out_path
    if out_path is None:
        out_path = osp.join(videos_dir, epoch_dir_name, f"{cfg.data_type}.mp4")
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    iio.imwrite(out_path, frames, fps=cfg.fps)
    guru.info(f"Saved video to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))

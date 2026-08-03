"""
Composes several per-camera CasualDataset instances (sharing one scene
scale/transform) into a single multi-view dataset, so a future training loop
can supervise from more than one camera per frame index.

Does NOT subclass BaseDataset: BaseDataset's abstract methods (get_w2cs,
get_Ks, get_image, ...) assume exactly one camera per sample, which
multi-view breaks by construction. Use .get_view(name) to get a plain
CasualDataset for any code that still wants that single-camera contract
(e.g. run_training.py's init_model_from_tracks, which only ever needs the
primary view).
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import torch
from torch.utils.data import Dataset

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig


@dataclass
class MultiViewDavisDataConfig:
    # {view_name: seq_name}, e.g. {"cam00": "dog_cam00_multiview",
    # "cam08": "dog_cam08", "cam17": "dog_cam17", "cam26": "dog_cam26"}
    seq_names: dict[str, str] = field(default_factory=dict)
    root_dir: str = ""
    primary_view: str = "cam00"
    start: int = 0
    end: int = -1
    res: str = "480p"
    image_type: str = "JPEGImages"
    mask_type: str = "sam2_masks"
    depth_type: str = "moge_calib"
    depth_is_metric: bool = True
    camera_type: Literal["droid_recon", "megasam", "static_rig"] = "static_rig"
    track_2d_type: str = "cotracker3"
    normals_type: str | None = None
    mask_erosion_radius: int = 3
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    shadow_type: str | None = None


class MultiViewCasualDataset(Dataset):
    """Indexing this dataset returns, for a given frame index, a dict mapping
    each view name to that view's CasualDataset sample dict (same schema
    CasualDataset.__getitem__ already produces -- see that class for the
    exact keys)."""

    def __init__(self, cfg: MultiViewDavisDataConfig):
        assert cfg.primary_view in cfg.seq_names, (
            f"primary_view={cfg.primary_view!r} not in seq_names={list(cfg.seq_names)}"
        )
        self.cfg = cfg
        self.view_names = list(cfg.seq_names)
        self.primary_view = cfg.primary_view

        common_kwargs = dict(
            root_dir=cfg.root_dir, start=cfg.start, end=cfg.end, res=cfg.res,
            image_type=cfg.image_type, mask_type=cfg.mask_type,
            depth_type=cfg.depth_type, depth_is_metric=cfg.depth_is_metric,
            camera_type=cfg.camera_type, track_2d_type=cfg.track_2d_type,
            normals_type=cfg.normals_type, mask_erosion_radius=cfg.mask_erosion_radius,
            num_targets_per_frame=cfg.num_targets_per_frame,
            load_from_cache=cfg.load_from_cache, shadow_type=cfg.shadow_type,
        )

        # Build the primary view first so every other view can share its
        # scene_norm_dict (scale + transform) instead of each independently
        # re-estimating one from its own tracks -- otherwise the views would
        # end up in mutually inconsistent coordinate frames/scales.
        primary_ds = CasualDataset(seq_name=cfg.seq_names[self.primary_view], **common_kwargs)
        self.datasets: dict[str, CasualDataset] = {self.primary_view: primary_ds}

        for view, seq_name in cfg.seq_names.items():
            if view == self.primary_view:
                continue
            self.datasets[view] = CasualDataset(
                seq_name=seq_name, scene_norm_dict=primary_ds.scene_norm_dict, **common_kwargs
            )

        n = primary_ds.num_frames
        for view, ds in self.datasets.items():
            assert ds.num_frames == n, (
                f"view {view!r} ({cfg.seq_names[view]}) has {ds.num_frames} frames, "
                f"expected {n} (from primary view {self.primary_view!r})"
            )
            assert ds.frame_names == primary_ds.frame_names, (
                f"view {view!r} frame names don't match the primary view -- "
                f"per-index alignment across views would be broken"
            )

    @property
    def num_frames(self) -> int:
        return self.datasets[self.primary_view].num_frames

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index) -> dict[str, dict]:
        return {view: ds[index] for view, ds in self.datasets.items()}

    def get_view(self, view_name: str) -> CasualDataset:
        return self.datasets[view_name]

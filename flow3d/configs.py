import os
from dataclasses import dataclass
from typing import Any
import tyro
import yaml

from flow3d.data import DATASET_REGISTRY


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class ShadowConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4

    centers: float = 1.6e-4
    coarse_rots: float = 1.6e-4
    coarse_transls: float = 1.6e-4
    fine_rots: float = 1.6e-4
    fine_transls: float = 1.6e-4


@dataclass
class ShadowMotionConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4

    centers: float = 1.6e-4
    coarse_rots: float = 1.6e-4
    coarse_transls: float = 1.6e-4
    fine_rots: float = 1.6e-4
    fine_transls: float = 1.6e-4


@dataclass
class CameraPoseLRConfig:
    Rs: float = 1e-5
    ts: float = 1e-5


@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig
    shad: ShadowConfig
    shad_bases: ShadowMotionConfig
    camera_poses: CameraPoseLRConfig


@dataclass
class LossesConfig:
    w_rgb: float = 1.0
    w_normal: float = 0.05
    w_depth_reg: float = 10
    w_depth_const: float = 1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0
    w_center_cano: float = 1.0
    w_center_coarse: float = 1.0
    w_coarse_align: float = 0.1
    w_rigidity: float = 0.5
    use_log_scale_var: bool = True


@dataclass
class OptimConfig:
    """Hyperparameters for Gaussian densification, culling, and motion bases control."""
    max_steps: int = 5000
    checkpoint_every_steps: int = 10
    ## Adaptive gaussian control
    warmup_steps: int = 20
    control_every: int = 10
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 400
    stop_control_steps: int = 400
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 1500
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15
    ## Motion bases control
    enable_bases_control: bool = True
    start_control_steps: int = 150
    control_bases_every: int = 50
    control_bases_offset: int = 30
    rigidity_refresh_every: int = 50
    max_num_bases: int = 100
    split_bases_threshold: float = 0.005
    cull_bases_threshold: float = 0.003
    clustering_window_size: int = 20
    max_noise_ratio: float = 0.25
    cull_clustering_noise: bool = False
    cluster_dist_threshold: float = 0.05
    ### Coefs Initialization
    coefs_type: str = "linear"
    coefs_sigma: float = 0.45


@dataclass
class TrackConfig:
    prop_interval: int = 5
    prop_epochs: int = 50
    warmup_epochs: int = 200


@dataclass
class TrainConfig:
    # Config classes
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimConfig
    track: TrackConfig
    config: str

    data: Any | None = None
    work_dir: str | None = None
    ckpt_path: str | None = None
    dataset: str | None = None
    data_dir: str | None = None
    seq_name: str | None = None
    exp_name: str | None = None
    # Gaussian
    use_2dgs: bool = False
    num_fg: int = 40_000
    num_bg: int = 100_000
    sample_bg_stride: int = 2
    num_bg_samples: int = 1000
    num_init_frames: int = 10
    num_init_samples: int = 40_000
    bases_type: str = "scalable"
    num_motion_bases: int = 40
    num_fine_bases: int = 30
    cluster_init_type: str = "means"
    # only used when cluster_init_type == "motion_affinity"
    affinity_k: int = 12
    affinity_cut_percentile: float = 85.0  # only used when affinity_method == "components"
    affinity_min_cluster_size: int = 20
    affinity_method: str = "agglomerative"
    affinity_n_clusters: int = 40  # only used when affinity_method == "agglomerative"

    # Training
    num_glob_epochs: int = 400
    stop_control_epochs: int = 200
    reset_opacity_epochs: int = 100
    pose_optim_window: int = 25
    train_features_every: int = 10
    train_features_epochs: int = 5
    stop_features_epoch: int = 400
    port: int = 8890
    vis_debug: bool = False
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50
    hash: bool = False
    save_video_frames: bool = True
    update_data: bool = False
    update_every: int = 50
    save_more_ckpts: bool = False
    eval_every: int = 10
    eval_last_n_epochs: int = 100
    prop_fg_only: bool = False

    @classmethod
    def build_from_cli(cls) -> "TrainConfig":
        """
        Build config from CLI arguments, load data config accordingly, and overwrite fields used in the scene config.
        """
        # Initialize config from CLI input
        cfg = tyro.cli(cls)

        # Load scene specific config from YAML
        with open(cfg.config, "r") as f:
            scene_config = yaml.safe_load(f)

        # Resolve seq_name: CLI takes priority, YAML is the fallback.
        if cfg.seq_name is not None:
            scene_config["seq_name"] = cfg.seq_name
        assert scene_config.get("seq_name") is not None, "seq_name must be provided via --seq_name or set in the config file."

        # Append seq_name to work_dir when it comes from the CLI.
        if cfg.seq_name:
            scene_config["work_dir"] = os.path.join(scene_config["work_dir"], cfg.seq_name)

        # Validate required fields and instantiate the dataset config.
        for field in ["dataset", "data_dir", "work_dir"]:
            assert scene_config.get(field) is not None, f"{field} is not specified in the configuration."

        # Load dataset config
        dataset_type = scene_config["dataset"]
        if dataset_type not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{dataset_type}' not recognized in registry.")

        data_class = DATASET_REGISTRY[dataset_type]
        cfg.data = data_class(
            root_dir=scene_config["data_dir"],
            seq_name=scene_config["seq_name"],
        )

        # Merge remaining YAML fields into cfg.
        merge_configs(cfg, scene_config)

        return cfg


def merge_configs(target: Any, source: dict) -> None:
    """
    Recursively merges a source configuration dictionary into a target object.
    """
    for key, value in source.items():
        if not hasattr(target, key):
            raise AttributeError(f"The target configuration does not have an attribute: '{key}'")

        if isinstance(value, dict):
            # Recursively merge dictionaries
            target_attr = getattr(target, key)
            merge_configs(target_attr, value)
        else:
            # Set the value directly for non-dictionary attributes
            setattr(target, key, value)
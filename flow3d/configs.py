import os
from dataclasses import dataclass
from typing import Any, Literal
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

    ### GNN correction (omega, delta_t) regularizers -- see flow3d/analysis/loss.py.
    # Only active when motion_bases is one of the *GraphCorrectedScalableMotionBases
    # classes (--enable_graph_coupling), no-op (0.0 loss, not even computed)
    # otherwise. All three are 0.0 while the GNN head is still zero-initialized,
    # so they don't disturb the zero-init-equivalence baseline.
    # L2 penalty on the correction magnitude, so it stays a small nudge on the
    # coarse transform rather than a free-floating residual.
    w_gnn_correction_reg: float = 0.01
    # second-order (central-difference) temporal smoothness on the correction
    # across (t-1, t, t+1), independent of the fixed-graph rigidity loss above.
    w_gnn_correction_smooth: float = 0.01
    # weak: only penalizes graph-connected clusters whose corrections point in
    # genuinely opposite directions (see cos_margin below); doesn't restrict
    # ordinary joint articulation.
    w_gnn_correction_edge_consistency: float = 0.01
    gnn_correction_edge_consistency_cos_margin: float = 0.5

    ### Edge-boundary correction (omega-free, per-edge translation magnitude)
    # regularizers -- see flow3d/graph_relative_edge.py. Only active when
    # motion_bases is EdgeBoundaryGraphCorrectedScalableMotionBases
    # (--enable_graph_coupling --gnn_variant=relative_edge_boundary), no-op
    # (0.0, not even computed) otherwise. Both are 0.0 while the edge head is
    # still zero-initialized.
    w_boundary_correction_reg: float = 0.01
    w_boundary_correction_smooth: float = 0.01

    ### Boundary gap loss -- see flow3d/graph_relative_edge.py's
    # compute_boundary_gap_loss. Only active (nonzero) when motion_bases is
    # EdgeBoundaryGraphCorrectedScalableMotionBases
    # (--gnn_variant=relative_edge_boundary); reads that class's own live,
    # densify/cull-refreshed boundary state directly, so no separate file path
    # is needed here. Penalizes the TRUE (coarse+fine-blended) boundary-Gaussian
    # gap only once
    # it exceeds (canonical_distance + boundary_gap_tolerance) -- inside the
    # tolerance band the loss is exactly 0, so ordinary articulation and small
    # canonical gaps are never fought. Gradient reaches only the edge GNN's
    # own parameters (base coarse/fine motion is a frozen input), so this loss
    # can only close the gap via the small localized correction, not by moving
    # (and thereby distorting) the rest of either cluster.
    #
    # Default 0.0: OFF. The correction magnitude itself is no longer gated by
    # this hinge (see graph_relative_edge.py's EdgeBoundaryGNN docstring) --
    # contact_reference_distance is a median over this SAME reconstruction's
    # own "connected" frames, so an edge with a persistent (roughly constant)
    # misalignment has contact_reference_distance already absorb that gap as
    # "normal", making the old gap_error-gated correction permanently 0 there
    # even though the underlying gap is real. Set > 0 explicitly to opt back
    # into this hinge as an ADDITIONAL loss on top of the render-loss-driven
    # correction (kept for backward CLI/config compatibility and as a
    # diagnostic/opt-in knob, not removed).
    w_boundary_gap: float = 0.0
    boundary_gap_tolerance: float = 0.01

    ### Edge-PATCH correction (omega + delta_t per edge-SIDE) regularizers --
    # see flow3d/graph_relative_linear_attention_boundary.py. Only active
    # when motion_bases is
    # RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases
    # (--gnn_variant=relative_velocity_linear_attention_boundary), no-op
    # (0.0, not even computed) otherwise. Applied to the RAW (patch-weight-
    # before-application) edge decoder output, same target as
    # w_gnn_correction_reg/w_gnn_correction_smooth for the per-cluster
    # variant. Both 0.0 while the edge decoder head is still zero-initialized.
    w_edge_correction_reg: float = 0.01
    w_edge_correction_smooth: float = 0.01

    ### Edge-patch boundary gap loss -- see flow3d/graph_relative_linear_
    # attention_boundary.py's compute_edge_patch_gap_loss. Only active
    # (nonzero) when motion_bases is
    # RelativeVelLinearAttentionBoundaryGraphCorrectedScalableMotionBases.
    # Only applied to edges that class already judged reliable
    # (persistence/num_known_frames gated at construction, not every
    # CONNECTED edge) and observed CONNECTED this frame. Gradient reaches
    # only the edge decoder GNN's own parameters (base motion is a frozen,
    # detached input), so this loss can only close the gap via the small
    # localized correction. Default 0.0: OFF, same rationale as
    # w_boundary_gap above (opt-in additional pressure on top of the
    # render-loss-driven correction).
    w_edge_boundary_gap: float = 0.0
    edge_boundary_gap_tolerance: float = 0.01

    ### Joint anchor loss -- see flow3d/analysis/loss_joint.py. Only active when
    # optim_cfg.joint_anchor_path is set (an edges.pt built by
    # flow3d/analysis/build_cluster_graph.py -- the SAME file passed to
    # --graph-coupling-path). Keeps each cluster pair connected by a kept
    # edge in that graph from opening under the (possibly GNN-corrected)
    # coarse transform -- e.g. flow3d/graph_relative_linear_attention.py's
    # RelativeVelLinearAttentionGraphCorrectedScalableMotionBases -- while
    # still allowing ordinary joint rotation (see loss_joint.py's module
    # docstring for why rotation about the anchor is unaffected, and for why
    # edges.pt is used instead of cluster_pairs.py's fixed_boundary_indices.pt).
    w_joint_anchor: float = 0.1
    # Huber transition point, in canonical/world scene units: below this the
    # penalty on (transformed distance - canonical distance) is quadratic,
    # above it linear, so a few badly-open frames don't dominate the gradient.
    joint_anchor_huber_delta: float = 0.01


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
    ### ARAP rigidity neighbor graph
    # "euclidean" (default): k Euclidean-nearest cluster centers, exactly the
    # historical behavior. "connectivity": body-connectivity graph (spatial
    # adjacency + motion consistency), see flow3d/rigidity_graph.py -- avoids
    # anchoring a cluster to a spatially-close-but-unrelated one (e.g. the
    # other hand during a clasped-hands pose). "cluster_graph_file": load a
    # fixed graph from an offline edges.pt built by
    # flow3d/analysis/build_cluster_graph.py (run against the init checkpoint
    # with --position-source raw_tracks, before this training run, so the
    # graph isn't built from this run's own learned motion) -- see
    # rigidity_graph_path. Requires --optim.no-enable-bases-control, since
    # bases split/cull remaps cluster ids the offline graph doesn't know
    # about.
    rigidity_graph_type: str = "euclidean"
    rigidity_graph_path: str | None = None
    connectivity_spatial_k: int = 12
    connectivity_min_shared_edges: int = 2
    connectivity_cv_threshold: float = 0.05
    ### Graph-coupled cluster GNN (see flow3d/graph_coupling.py)
    # learning rate for the GNN's own parameters (encoder/message-passing/head);
    # these don't appear in SceneLRConfig since they're not a "fg"/"bg"/"motion_bases"
    # leaf param, so Trainer.configure_optimizers gives them their own param group.
    gnn_lr: float = 1e-3
    ### Joint anchor loss (see flow3d/analysis/loss_joint.py). Path to an
    # edges.pt built by flow3d/analysis/build_cluster_graph.py -- normally
    # the exact same path passed to --graph-coupling-path, so the loss only
    # ever anchors cluster pairs the GNN's message passing already treats as
    # connected. Loaded and cached once by Trainer, the same lazy-load-once
    # pattern as rigidity_graph_path above; None disables the loss entirely
    # (0.0, not even computed). Like rigidity_graph_path's
    # "cluster_graph_file" mode, the saved cluster ids only stay valid while
    # the cluster set doesn't change, so this is meant to be combined with
    # --optim.no-enable-bases-control (the same requirement --enable_graph_coupling
    # already has).
    joint_anchor_path: str | None = None


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

    ### Graph-coupled cluster GNN
    # Four implementations are supported (--gnn_variant):
    # "relative_velocity_linear_attention" (default): a single rigid
    #   (omega, delta_t) correction per CLUSTER (flow3d/graph_relative_linear_attention.py).
    # "relative_velocity_linear_attention_frame": same per-CLUSTER (omega,
    #   delta_t) correction and multi-head attention as the variant above,
    #   but attention (and the final correction) is gated by each edge's
    #   per-frame CONNECTED/DISCONNECTED/UNKNOWN state, read from
    #   graph_coupling_path's connected_frame_indices/unknown_frame_indices
    #   (see flow3d/graph_relative_linear_attention_frame.py). A cluster's
    #   correction is scaled by the max gate over its incident edges, so one
    #   CONNECTED incident edge is enough to turn the whole cluster's
    #   correction on even if another incident edge is DISCONNECTED (this is
    #   the defined behavior of this variant, not a bug -- see that module's
    #   docstring). Uses gnn_hidden/gnn_layers/gnn_heads the same way as
    #   "relative_velocity_linear_attention".
    # "relative_edge_boundary": a small translation-only correction per EDGE
    #   (cluster pair), applied only to boundary Gaussians and smoothly
    #   falling off with distance from the boundary, so it can close a
    #   seam without rotating/translating the rest of either cluster (see
    #   flow3d/graph_relative_edge.py). Uses gnn_hidden/gnn_layers the same
    #   way (gnn_heads is unused -- no attention heads); its falloff radius is
    #   boundary_falloff_radius below.
    # "relative_velocity_linear_attention_boundary": the SAME node encoder/
    #   relative edge feature/multi-head attention message passing as
    #   "relative_velocity_linear_attention" (cluster embeddings are still
    #   contextual over the whole graph), but the readout is a per-EDGE,
    #   per-SIDE decoder instead of a per-cluster 6D head: each kept edge
    #   (a, b) gets its OWN (omega, delta_t) for side a and side b, applied
    #   only to that side's boundary-patch Gaussians (read from
    #   boundary_patch_path, a boundary_patch.pt built by flow3d/analysis/
    #   build_cluster_graph_mesh.py) and pivoted at that patch's own current
    #   centroid -- NOT the cluster center (see
    #   flow3d/graph_relative_linear_attention_boundary.py). A cluster
    #   touching several edges (e.g. 19-12, 19-8, 19-24) gets an
    #   independent correction per edge; a Gaussian in more than one edge's
    #   patch gets a weighted combination (never a naive sum). Interior
    #   (non-boundary) Gaussians are therefore byte-identical to the
    #   uncorrected MotionScale baseline no matter what the GNN predicts.
    #   Uses gnn_hidden/gnn_layers/gnn_heads the same way as
    #   "relative_velocity_linear_attention". edge topology and per-edge
    #   gate/reference-distance/persistence are read directly from
    #   graph_coupling_path's edges.pt (not from a separately-built
    #   edge_index) -- boundary_patch_path only supplies patch Gaussians/
    #   weights, matched to edges.pt's kept edges by cluster pair.
    # All four require a fixed graph and disabled bases control so cluster
    # ids remain stable during training.
    enable_graph_coupling: bool = False
    graph_coupling_path: str | None = None
    gnn_hidden: int = 128
    gnn_layers: int = 2
    # hidden_dim must be divisible by the number of attention heads.
    gnn_heads: int = 4
    gnn_variant: Literal[
        "relative_velocity_linear_attention",
        "relative_velocity_linear_attention_frame",
        "relative_edge_boundary",
        "relative_velocity_linear_attention_boundary",
    ] = "relative_velocity_linear_attention"
    # "relative_edge_boundary" only: RBF radius (scene units) of the
    # per-Gaussian falloff weight around each edge's boundary Gaussian set --
    # exactly at the boundary set weight == 1, decaying smoothly to ~0 by a
    # few multiples of this radius.
    #
    # "relative_velocity_linear_attention_boundary" reuses this same field as
    # its Gaussian<->canonical-boundary-reference snap radius fallback
    # (assign_edge_memberships's snap_radius, only used for a reference point
    # whose own local_scale is <= 0 -- normally that per-point local_scale is
    # used instead, density-adaptive).
    boundary_falloff_radius: float = 0.05
    # "relative_edge_boundary" only: multiplier on each edge's boundary patch's
    # own local Gaussian spacing (patch_ref_local_scale) to get that edge's
    # max_displacement -- the absolute cap on |correction| per step (see
    # flow3d/graph_relative_edge.py's _compute_edge_max_displacement). Larger
    # allows bigger single-step pulls; since correction is learned directly
    # from the render loss (no gap_error cap any more), this is the main
    # safety valve against an overly large single-step correction.
    #
    # "relative_velocity_linear_attention_boundary" reuses this same field as
    # its per-edge translation clamp scale (same _compute_edge_max_displacement
    # design, see flow3d/graph_relative_linear_attention_boundary.py).
    correction_max_disp_scale: float = 2.0
    # "relative_velocity_linear_attention_boundary" only: absolute cap
    # (radians) on the edge decoder's rotation correction (see
    # flow3d/graph_relative_linear_attention_boundary.py's
    # _clamp_vector_magnitude) -- unlike the translation cap, this isn't
    # derived from patch local_scale (rotation has no natural length scale),
    # so it's its own fixed hyperparameter.
    edge_correction_max_omega: float = 0.2
    # "relative_velocity_linear_attention_boundary" only: an edge must have at
    # least this much persistence AND this many known frames (both read from
    # graph_coupling_path's edges.pt) to be included in the edge-patch
    # boundary gap loss (w_edge_boundary_gap) -- a sparsely- or unreliably-
    # observed edge's distance measurements shouldn't be trusted as a gap-loss
    # target even if it happens to be CONNECTED this frame.
    edge_gap_loss_min_persistence: float = 0.5
    edge_gap_loss_min_known_frames: int = 1
    # "relative_velocity_linear_attention_boundary" only: path to the
    # boundary_patch.pt built by flow3d/analysis/build_cluster_graph_mesh.py
    # (typically <work_dir>/analysis/cluster_graph_mesh/boundary_patch.pt).
    # Required when gnn_variant is this value; unused otherwise.
    boundary_patch_path: str | None = None

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
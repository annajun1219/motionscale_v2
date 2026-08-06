#!/usr/bin/env python3
"""
Continue training from an existing checkpoint with a *fresh* optimizer for
exactly N more epochs, swapping the model's cluster motion bases
(ScalableMotionBases) for the graph-aware version in cluster_graph_gnn.py --
to test whether reflecting cluster-adjacency graph structure in the coarse
(global) transform helps, on top of the exact same fine (local) motion.

This mirrors flow3d/experiment/contactloss/run_contact_experiment.py's
"fresh optimizer, N more epochs" isolation strategy so any gap between runs
is attributable to the graph-aware correction, not to additional training
time.

Baseline vs treatment
----------------------
    # A) baseline: fresh optimizer, N more epochs, motion bases untouched
    python flow3d/experiment/graph/run_graph_experiment.py \\
        --work-dir outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3 \\
        --exp-name graph_baseline --disable-gnn

    # B) treatment: identical, but coarse transform is graph-corrected
    python flow3d/experiment/graph/run_graph_experiment.py \\
        --work-dir outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3 \\
        --exp-name graph_treated --reg-weight 0.01

Both share: the exact source model weights, a freshly initialized optimizer,
the same --num-epochs additional epochs, and every other training
hyperparameter from the original run (re-read from <work-dir>/cfg.yaml).

Prerequisite
------------
--boundary-file must point at the adjacency file produced by
flow3d/analysis/cluster_pairs.py (default:
<work-dir>/analysis/all_cluster_boundary_relations/fixed_boundary_indices.pt).
Run that script against the source checkpoint first if it doesn't exist yet.
--disable-gnn (the baseline run) does not need this file.

Implementation note
--------------------
Like run_contact_experiment.py, this does not modify run_training.py or
trainer.py. It monkey-patches three things and then calls run_training.main()
in-process:

  1. SceneModel.init_from_state_dict -- after building the model, replaces
     model.motion_bases with GraphCorrectedScalableMotionBases wrapping it.
  2. Trainer.configure_optimizers -- otherwise it looks up each parameter's
     learning rate as lr_cfg[part][field] where field is the parameter's own
     name (e.g. "rots", "means"); the new graph_gnn.* submodule parameters
     ("weight"/"bias" of nn.Linear layers) don't exist in that lr config,
     so they're special-cased to use --gnn-lr instead.
  3. Trainer.compute_losses -- adds two optional terms:
       - motion_bases.correction_regularization_loss() (weighted by --reg-weight)
         so the GNN is pushed to output a small correction rather than a
         free-standing transform, per the "loss를 통해 correction만 보정" step.
       - motion_bases.attention_entropy_regularization_loss() (weighted by
         --attn-reg-weight) so the per-frame edge attention doesn't collapse
         onto a single neighbor in the (fixed) canonical adjacency graph.

Known limitation
----------------
If adaptive density control splits/culls clusters during these extra epochs,
ScalableMotionBases.dup_bases/add_bases/cull_bases only resize the coarse/fine
motion parameters -- the GNN's cluster-adjacency graph (built once, from the
source checkpoint's cluster count) would then no longer match. This is fine
for a short post-hoc continuation (the common case here) but would need
explicit handling for a from-scratch run with density control enabled.
"""

from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from flow3d.experiment.graph.cluster_graph_gnn import (
    GraphCorrectedScalableMotionBases,
    build_edge_index_from_boundary_file,
)
from flow3d.params import ScalableMotionBases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True,
        help="Source run dir, containing checkpoints/last.ckpt and cfg.yaml.",
    )
    parser.add_argument("--src-ckpt", type=Path, default=None, help="Default: <work-dir>/checkpoints/last.ckpt")
    parser.add_argument(
        "--boundary-file", type=Path, default=None,
        help="Default: <work-dir>/analysis/all_cluster_boundary_relations/fixed_boundary_indices.pt "
             "(output of flow3d/analysis/cluster_pairs.py). Not needed with --disable-gnn.",
    )
    parser.add_argument(
        "--disable-gnn", action="store_true",
        help="Baseline run: skip the graph-motion-bases swap entirely (plain fresh-optimizer continuation).",
    )
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument(
        "--gnn-num-heads", type=int, default=4,
        help="Attention heads per graph-attention layer. Must evenly divide --gnn-hidden-dim.",
    )
    parser.add_argument("--gnn-lr", type=float, default=1.6e-4, help="LR for the new graph_gnn.* parameters.")
    parser.add_argument(
        "--reg-weight", type=float, default=0.0,
        help="Weight for motion_bases.correction_regularization_loss(). 0 = no regularization.",
    )
    parser.add_argument(
        "--attn-reg-weight", type=float, default=0.0,
        help="Weight for motion_bases.attention_entropy_regularization_loss(), which "
             "discourages attention from collapsing onto one neighbor. 0 = disabled.",
    )
    parser.add_argument("--num-epochs", type=int, default=50, help="Additional epochs to train beyond the checkpoint.")
    parser.add_argument(
        "--exp-name", type=str, required=True,
        help="e.g. 'graph_baseline' or 'graph_w1' -- must differ between the two runs.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def make_fresh_optimizer_checkpoint(src_ckpt: Path, dst_ckpt: Path) -> int:
    """Copy a checkpoint, dropping optimizer/scheduler state. Returns its saved epoch."""
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    ckpt.pop("optimizers", None)
    ckpt.pop("schedulers", None)
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst_ckpt)
    return int(ckpt.get("epoch", 0))


def compute_prop_end_final(
    num_frames: int, num_init_frames: int, prop_interval: int, warmup_epochs: int, prop_epochs: int,
) -> int:
    """Mirrors run_training.py's propagation-schedule math (kept in sync manually)."""
    prop_start_frames = [0] + list(range(num_init_frames, num_frames, prop_interval))
    prop_start_epochs = [0] + [warmup_epochs + i * prop_epochs for i in range(len(prop_start_frames) - 1)]
    prop_end_epochs = prop_start_epochs[1:] + [prop_start_epochs[-1] + prop_epochs]
    return prop_end_epochs[-1]


def install_graph_bases_patch(
    edge_index: torch.Tensor, gnn_hidden_dim: int, gnn_num_layers: int, gnn_num_heads: int
) -> None:
    """Make SceneModel.init_from_state_dict return a graph-corrected motion_bases."""
    import flow3d.scene_model as scene_model_mod

    original_init_from_state_dict = scene_model_mod.SceneModel.init_from_state_dict

    def patched_init_from_state_dict(state_dict, prefix=""):
        model = original_init_from_state_dict(state_dict, prefix=prefix)
        if not isinstance(model.motion_bases, ScalableMotionBases):
            raise TypeError(
                "Graph-aware correction requires a ScalableMotionBases motion_bases "
                f"(coarse+fine cluster motion); got {type(model.motion_bases).__name__}."
            )
        model.motion_bases = GraphCorrectedScalableMotionBases.from_scalable_motion_bases(
            model.motion_bases,
            edge_index=edge_index,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_num_heads=gnn_num_heads,
        )
        return model

    scene_model_mod.SceneModel.init_from_state_dict = staticmethod(patched_init_from_state_dict)


def install_optimizer_patch(graph_gnn_lr: float) -> None:
    """Route motion_bases.graph_gnn.* params to --gnn-lr instead of the (nonexistent)
    MotionLRConfig entry that Trainer.configure_optimizers would otherwise look up."""
    import flow3d.trainer as trainer_mod

    def patched_configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        for name, params in self.model.named_parameters():
            name_fields = name.split(".")
            part, field = name_fields[0], name_fields[-1]
            lr = graph_gnn_lr if "graph_gnn" in name_fields else lr_dict[part][field]
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

    trainer_mod.Trainer.configure_optimizers = patched_configure_optimizers


def install_regularization_loss_patch(reg_weight: float, attn_reg_weight: float) -> None:
    """Add motion_bases.correction_regularization_loss() and
    motion_bases.attention_entropy_regularization_loss(), weighted, to every training step."""
    import flow3d.trainer as trainer_mod

    original_compute_losses = trainer_mod.Trainer.compute_losses

    def patched_compute_losses(self, batch):
        loss, stats, num_rays_per_step, num_rays_per_sec = original_compute_losses(self, batch)
        if reg_weight > 0:
            reg_value = self.model.motion_bases.correction_regularization_loss()
            loss = loss + reg_weight * reg_value
            stats["train/graph_correction_reg_loss"] = reg_value.detach()
        if attn_reg_weight > 0:
            attn_reg_value = self.model.motion_bases.attention_entropy_regularization_loss()
            loss = loss + attn_reg_weight * attn_reg_value
            stats["train/graph_attention_entropy_reg_loss"] = attn_reg_value.detach()
        return loss, stats, num_rays_per_step, num_rays_per_sec

    trainer_mod.Trainer.compute_losses = patched_compute_losses


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    src_ckpt = (
        args.src_ckpt.expanduser().resolve() if args.src_ckpt is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )
    boundary_file = (
        args.boundary_file.expanduser().resolve() if args.boundary_file is not None
        else work_dir / "analysis" / "all_cluster_boundary_relations" / "fixed_boundary_indices.pt"
    )
    src_cfg_path = work_dir / "cfg.yaml"

    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt}")
    if not src_cfg_path.is_file():
        raise FileNotFoundError(f"Source cfg.yaml not found: {src_cfg_path}")
    if not args.disable_gnn and not boundary_file.is_file():
        raise FileNotFoundError(
            f"Boundary-relation file not found: {boundary_file}. Run "
            "flow3d/analysis/cluster_pairs.py first, pass --boundary-file, or use --disable-gnn."
        )

    src_cfg = yaml.safe_load(src_cfg_path.read_text())

    fresh_ckpt = work_dir / "checkpoints" / f"last_fresh_optim_for_{args.exp_name}.ckpt"
    start_epoch = make_fresh_optimizer_checkpoint(src_ckpt, fresh_ckpt)
    print(f"Source checkpoint: {src_ckpt} (epoch={start_epoch})")
    print(f"Fresh-optimizer copy (model weights only): {fresh_ckpt}")

    if args.disable_gnn:
        print("Graph-aware motion bases DISABLED (baseline run).")
    else:
        state_dict = torch.load(fresh_ckpt, map_location="cpu", weights_only=False)["model"]
        num_clusters = state_dict["motion_bases.params.rots"].shape[0]
        edge_index = build_edge_index_from_boundary_file(boundary_file, num_clusters=num_clusters)
        print(
            f"Graph-aware motion bases ENABLED: num_clusters={num_clusters} "
            f"edges={edge_index.shape[1]} hidden_dim={args.gnn_hidden_dim} "
            f"num_layers={args.gnn_num_layers} num_heads={args.gnn_num_heads} "
            f"gnn_lr={args.gnn_lr} reg_weight={args.reg_weight} attn_reg_weight={args.attn_reg_weight}"
        )
        install_graph_bases_patch(edge_index, args.gnn_hidden_dim, args.gnn_num_layers, args.gnn_num_heads)
        install_optimizer_patch(args.gnn_lr)
        install_regularization_loss_patch(args.reg_weight, args.attn_reg_weight)

    img_dir = (
        Path(src_cfg["data"]["root_dir"])
        / src_cfg["data"]["image_type"]
        / src_cfg["data"]["res"]
        / src_cfg["data"]["seq_name"]
    )
    num_frames = len(list(img_dir.iterdir()))

    prop_end_final = compute_prop_end_final(
        num_frames=num_frames,
        num_init_frames=int(src_cfg["num_init_frames"]),
        prop_interval=int(src_cfg["track"]["prop_interval"]),
        warmup_epochs=int(src_cfg["track"]["warmup_epochs"]),
        prop_epochs=int(src_cfg["track"]["prop_epochs"]),
    )
    num_glob_epochs_new = (start_epoch + args.num_epochs) - prop_end_final
    if num_glob_epochs_new <= 0:
        raise ValueError(
            f"start_epoch({start_epoch}) + num_epochs({args.num_epochs}) does not exceed "
            f"the source run's propagation-end epoch ({prop_end_final}); "
            f"got num_glob_epochs={num_glob_epochs_new} <= 0. This script only supports "
            f"continuing a checkpoint that is already past propagation."
        )
    target_total_epochs = prop_end_final + num_glob_epochs_new
    print(
        f"num_frames={num_frames}, propagation-end epoch={prop_end_final}, "
        f"start_epoch={start_epoch} -> target_total_epochs={target_total_epochs} "
        f"(+{args.num_epochs} epochs), --num-glob-epochs override={num_glob_epochs_new}"
    )

    sys.argv = [
        "run_training.py",
        "--config", src_cfg["config"],
        "--seq-name", src_cfg["seq_name"],
        "--exp-name", args.exp_name,
        "--ckpt-path", str(fresh_ckpt),
        "--num-glob-epochs", str(num_glob_epochs_new),
        "--track.warmup-epochs", str(src_cfg["track"]["warmup_epochs"]),
        "--track.prop-epochs", str(src_cfg["track"]["prop_epochs"]),
        "--track.prop-interval", str(src_cfg["track"]["prop_interval"]),
    ]
    print(f"run_training.py argv: {sys.argv[1:]}")

    import run_training

    run_training.main()


if __name__ == "__main__":
    main()

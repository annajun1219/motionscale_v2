#!/usr/bin/env python3
"""
Continue training from an existing checkpoint with a *fresh* optimizer for
exactly N more epochs, combining two experiments that were each tested in
isolation first:

  1. flow3d/experiment/graph/run_graph_experiment.py -- graph-aware coarse
     (global) cluster transform correction (canonical adjacency topology
     fixed, per-frame edge attention).
  2. flow3d/experiment/contactloss/frame_contact_loss.py -- an explicit
     contact loss that only fires on (pair, frame) combinations that
     flow3d/analysis/contactpatch_distance.py already flagged as separating.

Why combine them
-----------------
The graph-correction-only experiment gave a real but small joint-separation
improvement (median max_ratio_to_reference down ~3-5%) -- the structural
mechanism (neighbor clusters share information) works, but nothing in the
training loss actually penalizes joint separation directly, so there was no
strong signal pushing the GNN to use that mechanism for that purpose. The
frame-contact-loss-only experiment gave that direct signal (raw contact loss
dropped ~36x over 200 epochs, with no measurable rgb/psnr/ssim cost) but only
acts on the existing (non-graph) coarse+fine motion.

Combining them lets the explicit contact-loss gradient flow through the
graph-corrected positions: SceneModel.compute_poses_fg (which the contact
loss calls to get means_fg) goes through motion_bases.compute_transforms,
which for GraphCorrectedScalableMotionBases already applies the GNN
correction -- so gradients from the contact loss reach the GNN's attention
and correction layers directly, giving the graph structure the target signal
it was missing on its own.

Usage
-----
    python flow3d/experiment/graph/run_graph_contact_experiment.py \\
        --work-dir outputs/davis/spaceout/2026_08_04_11_15_39__spaceout_run1 \\
        --exp-name graph_contact_combined --num-epochs 200

Defaults for --reg-weight/--attn-reg-weight/--margin/--contact-weight are the
values already settled on in the two standalone experiments for this
sequence (reg=0.01, attn=0.005, margin=0.03, contact-weight=0.1) -- override
any of them explicitly if re-tuning.

Prerequisites (same as the two standalone scripts)
----------------------------------------------------
- <work-dir>/analysis/all_cluster_boundary_relations/fixed_boundary_indices.pt
  (flow3d/analysis/cluster_pairs.py)
- <work-dir>/analysis/contact_patches/contact_patches.pt
  (flow3d/analysis/contactpatch.py)
- <work-dir>/analysis/contact_patch_distances/contact_patch_pair_summary.csv
  (flow3d/analysis/contactpatch_distance.py)

Implementation note
--------------------
Does not modify run_training.py or trainer.py. Installs, in order, all of:
  1. SceneModel.init_from_state_dict patch (motion_bases -> GraphCorrectedScalableMotionBases)
  2. Trainer.configure_optimizers patch (graph_gnn.* params -> --gnn-lr)
  3. Trainer.compute_losses patch #1: + graph correction/attention regularization
  4. Trainer.compute_losses patch #2: + frame-gated contact loss
then calls run_training.main() in-process. Patches #3 and #4 both capture
"whatever Trainer.compute_losses currently is" at install time and wrap it,
so installing them in sequence chains them (both terms get added) rather
than one overwriting the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from flow3d.experiment.contactloss.frame_contact_loss import FrameContactLoss
from flow3d.experiment.contactloss.run_contact_experiment import (
    install_contact_loss_patch,
)
from flow3d.experiment.graph.cluster_graph_gnn import build_edge_index_from_boundary_file
from flow3d.experiment.graph.run_graph_experiment import (
    compute_prop_end_final,
    install_graph_bases_patch,
    install_optimizer_patch,
    install_regularization_loss_patch,
    make_fresh_optimizer_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True,
        help="Source run dir, containing checkpoints/last.ckpt and cfg.yaml.",
    )
    parser.add_argument("--src-ckpt", type=Path, default=None, help="Default: <work-dir>/checkpoints/last.ckpt")

    # --- graph correction args (see run_graph_experiment.py) ---
    parser.add_argument(
        "--boundary-file", type=Path, default=None,
        help="Default: <work-dir>/analysis/all_cluster_boundary_relations/fixed_boundary_indices.pt",
    )
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-num-heads", type=int, default=4)
    parser.add_argument("--gnn-lr", type=float, default=1.6e-4, help="LR for the new graph_gnn.* parameters.")
    parser.add_argument(
        "--reg-weight", type=float, default=0.01,
        help="Weight for motion_bases.correction_regularization_loss(). 0 = disabled.",
    )
    parser.add_argument(
        "--attn-reg-weight", type=float, default=0.005,
        help="Weight for motion_bases.attention_entropy_regularization_loss(). 0 = disabled.",
    )

    # --- frame contact loss args (see run_contact_experiment.py) ---
    parser.add_argument(
        "--patch-file", type=Path, default=None,
        help="Default: <work-dir>/analysis/contact_patches/contact_patches.pt",
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="Default: <work-dir>/analysis/contact_patch_distances/contact_patch_pair_summary.csv",
    )
    parser.add_argument(
        "--pairs", type=str, default=None,
        help='Comma list of cluster pairs to penalize, e.g. "29-33,7-25". '
             "Default: every pair passing --min-flagged-frames.",
    )
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--contact-beta", type=float, default=0.002, help="Smooth-L1 beta for the contact loss.")
    parser.add_argument("--contact-weight", type=float, default=0.1)
    parser.add_argument("--min-flagged-frames", type=int, default=5)

    parser.add_argument("--num-epochs", type=int, default=200, help="Additional epochs to train beyond the checkpoint.")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def parse_pairs_arg(pairs_arg: str) -> set[tuple[int, int]]:
    requested: set[tuple[int, int]] = set()
    for token in pairs_arg.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.replace(":", "-").split("-")
        requested.add((min(int(a), int(b)), max(int(a), int(b))))
    if not requested:
        raise ValueError("--pairs must list at least one cluster pair.")
    return requested


def build_frame_contact_loss(
    patch_file: Path, summary_csv: Path, pairs_arg: str | None,
    margin: float, beta: float, min_flagged_frames: int, device: str,
) -> FrameContactLoss:
    from flow3d.experiment.contactloss.frame_contact_loss import load_frame_gated_pairs

    gated_pairs = load_frame_gated_pairs(patch_file, summary_csv, min_flagged_frames)

    if pairs_arg is not None:
        requested = parse_pairs_arg(pairs_arg)
        canon = lambda p: (min(p.cluster_a, p.cluster_b), max(p.cluster_a, p.cluster_b))
        gated_pairs = [p for p in gated_pairs if canon(p) in requested]
        missing = requested - {canon(p) for p in gated_pairs}
        if missing:
            raise ValueError(
                f"Requested pair(s) not found (or didn't clear --min-flagged-frames="
                f"{min_flagged_frames}) in {summary_csv}: {sorted(missing)}"
            )

    return FrameContactLoss(gated_pairs=gated_pairs, margin=margin, beta=beta).to(torch.device(device))


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
    patch_file = (
        args.patch_file.expanduser().resolve() if args.patch_file is not None
        else work_dir / "analysis" / "contact_patches" / "contact_patches.pt"
    )
    summary_csv = (
        args.summary_csv.expanduser().resolve() if args.summary_csv is not None
        else work_dir / "analysis" / "contact_patch_distances" / "contact_patch_pair_summary.csv"
    )
    src_cfg_path = work_dir / "cfg.yaml"

    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt}")
    if not src_cfg_path.is_file():
        raise FileNotFoundError(f"Source cfg.yaml not found: {src_cfg_path}")
    if not boundary_file.is_file():
        raise FileNotFoundError(
            f"Boundary-relation file not found: {boundary_file}. "
            "Run flow3d/analysis/cluster_pairs.py first, or pass --boundary-file."
        )
    if not patch_file.is_file():
        raise FileNotFoundError(
            f"Contact patch file not found: {patch_file}. "
            "Run flow3d/analysis/contactpatch.py first, or pass --patch-file."
        )
    if not summary_csv.is_file():
        raise FileNotFoundError(
            f"Pair summary CSV not found: {summary_csv}. "
            "Run flow3d/analysis/contactpatch_distance.py first, or pass --summary-csv."
        )

    src_cfg = yaml.safe_load(src_cfg_path.read_text())

    fresh_ckpt = work_dir / "checkpoints" / f"last_fresh_optim_for_{args.exp_name}.ckpt"
    start_epoch = make_fresh_optimizer_checkpoint(src_ckpt, fresh_ckpt)
    print(f"Source checkpoint: {src_ckpt} (epoch={start_epoch})")
    print(f"Fresh-optimizer copy (model weights only): {fresh_ckpt}")

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

    contact_loss = build_frame_contact_loss(
        patch_file, summary_csv, args.pairs, args.margin, args.contact_beta,
        args.min_flagged_frames, args.device,
    )
    print(
        f"Frame contact loss ENABLED: patch_file={patch_file} summary_csv={summary_csv} "
        f"pairs={args.pairs or 'all passing --min-flagged-frames'} "
        f"min_flagged_frames={args.min_flagged_frames} margin={args.margin} "
        f"beta={args.contact_beta} weight={args.contact_weight}"
    )
    install_contact_loss_patch(contact_loss, args.contact_weight)

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

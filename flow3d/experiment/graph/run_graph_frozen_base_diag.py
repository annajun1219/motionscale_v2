#!/usr/bin/env python3
"""
flow3d/experiment/graph/run_graph_frozen_base_diag.py

Diagnostic (b) for "why doesn't graph help contact stability":

Starts from a checkpoint that frame_contact_loss has ALREADY converged on
(e.g. outputs/davis/spaceout/2026_08_05_15_05_43__frame_contact_w0.1_200ep),
freezes every existing coarse/fine motion parameter (lr=0, so Adam applies an
exactly-zero update regardless of gradient), attaches the graph GNN fresh
(zero-initialized decoder, so it starts as a no-op), and continues training
with the SAME frame-gated contact loss active -- but now the ONLY parameters
that can move are motion_bases.graph_gnn.*.

This isolates one question from run_graph_contact_experiment.py's combined
run: given a base motion that's already resolved most flagged pairs, and an
explicit contact-loss gradient with nothing else competing for it (no
photometric fine-tuning happening at the same time), can the graph's
rigid-per-cluster correction mechanism reduce the *remaining* flagged pairs
any further? If not, the bottleneck is the correction's granularity (whole
cluster rigid transform), not gradient interference from simultaneous
photometric fine-tuning.

reg-weight/attn-reg-weight default to 0 here (unlike
run_graph_contact_experiment.py's 0.01/0.005) since the point is to give the
correction maximum freedom to prove it can help at all before reintroducing
any regularization.

Usage
-----
    python flow3d/experiment/graph/run_graph_frozen_base_diag.py \\
        --work-dir outputs/davis/spaceout/2026_08_05_15_05_43__frame_contact_w0.1_200ep \\
        --exp-name diagB_graph_frozen_base_200ep --num-epochs 200
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from flow3d.experiment.contactloss.run_contact_experiment import install_contact_loss_patch
from flow3d.experiment.graph.cluster_graph_gnn import build_edge_index_from_boundary_file
from flow3d.experiment.graph.run_graph_contact_experiment import build_frame_contact_loss
from flow3d.experiment.graph.run_graph_experiment import (
    compute_prop_end_final,
    install_graph_bases_patch,
    install_regularization_loss_patch,
    make_fresh_optimizer_checkpoint,
)


def install_frozen_base_optimizer_patch(graph_gnn_lr: float) -> None:
    """Same shape as run_graph_experiment.install_optimizer_patch, except every
    parameter outside motion_bases.graph_gnn.* gets lr=0 (Adam still creates an
    optimizer entry for it -- required for checkpoint save/load and any
    density-control lookups by full_param_name -- but its update is exactly
    zero every step, regardless of gradient)."""
    import flow3d.trainer as trainer_mod

    def patched_configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = None  # unused for frozen params, kept for parity/debug
        optimizers = {}
        schedulers = {}
        for name, params in self.model.named_parameters():
            name_fields = name.split(".")
            is_graph_gnn = "graph_gnn" in name_fields
            lr = graph_gnn_lr if is_graph_gnn else 0.0
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if is_graph_gnn:
                fnc = lambda _, **__: 1.0
            else:
                fnc = lambda _, **__: 1.0  # lr already 0; decay is a no-op either way

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=(lr if lr > 0 else 1.0))
            )
        n_frozen = sum(1 for n, _ in self.model.named_parameters() if "graph_gnn" not in n.split("."))
        n_trainable = sum(1 for n, _ in self.model.named_parameters() if "graph_gnn" in n.split("."))
        print(f"[frozen-base diag] {n_frozen} param groups frozen (lr=0), {n_trainable} graph_gnn param groups trainable (lr={graph_gnn_lr})")
        return optimizers, schedulers

    trainer_mod.Trainer.configure_optimizers = patched_configure_optimizers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True,
        help="Run dir to continue from (its own checkpoints/last.ckpt + analysis/* are used).",
    )
    parser.add_argument("--src-ckpt", type=Path, default=None, help="Default: <work-dir>/checkpoints/last.ckpt")
    parser.add_argument("--boundary-file", type=Path, default=None)
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-num-heads", type=int, default=4)
    parser.add_argument("--gnn-lr", type=float, default=1.6e-4)
    parser.add_argument("--reg-weight", type=float, default=0.0)
    parser.add_argument("--attn-reg-weight", type=float, default=0.0)
    parser.add_argument("--patch-file", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--contact-beta", type=float, default=0.002)
    parser.add_argument("--contact-weight", type=float, default=0.1)
    parser.add_argument("--min-flagged-frames", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


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

    for p, desc in [
        (src_ckpt, "Source checkpoint"), (src_cfg_path, "Source cfg.yaml"),
        (boundary_file, "Boundary-relation file"), (patch_file, "Contact patch file"),
        (summary_csv, "Pair summary CSV"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"{desc} not found: {p}")

    src_cfg = yaml.safe_load(src_cfg_path.read_text())

    fresh_ckpt = work_dir / "checkpoints" / f"last_fresh_optim_for_{args.exp_name}.ckpt"
    start_epoch = make_fresh_optimizer_checkpoint(src_ckpt, fresh_ckpt)
    print(f"Source checkpoint: {src_ckpt} (epoch={start_epoch})")
    print(f"Fresh-optimizer copy (model weights only): {fresh_ckpt}")

    state_dict = torch.load(fresh_ckpt, map_location="cpu", weights_only=False)["model"]
    num_clusters = state_dict["motion_bases.params.rots"].shape[0]
    edge_index = build_edge_index_from_boundary_file(boundary_file, num_clusters=num_clusters)
    print(
        f"Graph-aware motion bases ENABLED (base FROZEN): num_clusters={num_clusters} "
        f"edges={edge_index.shape[1]} hidden_dim={args.gnn_hidden_dim} "
        f"num_layers={args.gnn_num_layers} num_heads={args.gnn_num_heads} "
        f"gnn_lr={args.gnn_lr} reg_weight={args.reg_weight} attn_reg_weight={args.attn_reg_weight}"
    )
    install_graph_bases_patch(edge_index, args.gnn_hidden_dim, args.gnn_num_layers, args.gnn_num_heads)
    install_frozen_base_optimizer_patch(args.gnn_lr)
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
            f"the source run's propagation-end epoch ({prop_end_final})."
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

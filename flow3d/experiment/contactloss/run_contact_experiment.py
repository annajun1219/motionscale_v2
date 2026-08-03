#!/usr/bin/env python3
"""
Continue training from an existing checkpoint with a *fresh* optimizer for
exactly N more epochs, optionally adding a contact loss on one or more saved
contact-patch cluster pairs -- to test whether the loss (not just more
training) reduces contact-patch separation.

Two loss implementations are available via --loss-type:

  simple (flow3d/experiment/contactloss/simple_contact_loss.py):
      Always-on for the pairs listed in --pairs, every frame, every step.

  frame (flow3d/experiment/contactloss/frame_contact_loss.py):
      Only penalizes (pair, frame) combinations that
      flow3d/analysis/contactpatch_distance.py already flagged as actually
      separating, read from --summary-csv (its contact_patch_pair_summary.csv
      output). Pairs that rarely separate are not registered at all
      (--min-flagged-frames); frames where a registered pair is currently
      fine contribute exactly zero. --pairs is optional here -- omit it to
      use every pair that clears --min-flagged-frames.

Why a fresh optimizer
----------------------
Just resuming normally (`python run_training.py --ckpt-path ...`) would also
restore Adam's momentum/variance state from the original run. Any change
seen afterwards could then be "the tail end of the original optimization
trajectory" rather than something attributable to the contact loss. This
script instead loads ONLY the model weights from the checkpoint and lets
run_training.py build a brand-new optimizer/LR-scheduler for it (Adam's
state is zero-initialized either way -- no seed is needed to make the two
runs' starting optimizer state identical).

Isolating the contact loss from "just more training"
------------------------------------------------------
Run this script twice from the SAME source checkpoint:

    # A) baseline: fresh optimizer, N more epochs, no contact loss
    python flow3d/experiment/contactloss/run_contact_experiment.py \\
        --work-dir outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3 \\
        --exp-name contact_baseline

    # B) treatment: identical, but with the contact loss enabled
    python flow3d/experiment/contactloss/run_contact_experiment.py \\
        --work-dir outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3 \\
        --exp-name contact_treated --contact-weight 1.0

Both (A) and (B) share: the exact source model weights, a freshly
initialized optimizer, the same --num-epochs additional epochs, and every
other training hyperparameter from the original run (re-read from
<work-dir>/cfg.yaml and replayed as run_training.py CLI flags). The ONLY
difference between the two invocations is --contact-weight, so any gap
between them (e.g. rerun flow3d/analysis/contactpatch_distance.py against
each new checkpoint and compare contact_patch_pair_summary.csv) is
attributable to the loss itself, not to additional training time.

Implementation note
--------------------
This does not reimplement the training loop: it monkey-patches
Trainer.compute_losses to add the (optional) contact loss term, builds an
equivalent CLI argv for run_training.py from <work-dir>/cfg.yaml, and calls
run_training.main() in-process. Aside from --ckpt-path (pointed at an
optimizer/scheduler-stripped copy of the source checkpoint) and
--num-glob-epochs (bumped so exactly --num-epochs more epochs run), this is
byte-for-byte the same training code path as the original run.

Default pair/margin
--------------------
Picked from flow3d/analysis/contactpatch_distance.py's own output for run3:
cluster pair 29-33 separates ~513x by frame 99 (the worst offender across
all retained pairs there), and 0.016 is that same script's "flagged"
threshold (1.5x the frame-0 contact distance of 0.0106 for that pair).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from flow3d.experiment.contactloss.frame_contact_loss import (
    FrameContactLoss,
    load_frame_gated_pairs,
)
from flow3d.experiment.contactloss.simple_contact_loss import (
    SimpleContactLoss,
    load_contact_patch_pairs,
)

ContactLoss = SimpleContactLoss | FrameContactLoss


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
        "--patch-file", type=Path, default=None,
        help="Default: <work-dir>/analysis/contact_patches/contact_patches.pt",
    )
    parser.add_argument(
        "--loss-type", type=str, default="simple", choices=["simple", "frame"],
        help="'simple' = simple_contact_loss.py (always-on for --pairs). "
             "'frame' = frame_contact_loss.py (only flagged pair/frame combos).",
    )
    parser.add_argument(
        "--pairs", type=str, default=None,
        help='Comma list of cluster pairs to penalize, e.g. "29-33" or "29-33,7-25". '
             "Required for --loss-type simple. Optional for --loss-type frame "
             "(omit to use every pair passing --min-flagged-frames).",
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="Only used with --loss-type frame. Default: "
             "<work-dir>/analysis/contact_patch_distances/contact_patch_pair_summary.csv",
    )
    parser.add_argument(
        "--min-flagged-frames", type=int, default=5,
        help="Only used with --loss-type frame. A pair is only registered in the "
             "loss if contactpatch_distance.py flagged at least this many frames for it.",
    )
    parser.add_argument("--margin", type=float, default=0.016)
    parser.add_argument("--contact-beta", type=float, default=0.002, help="Smooth-L1 beta for the contact loss.")
    parser.add_argument(
        "--contact-weight", type=float, default=0.0,
        help="Contact loss weight. 0 (default) = baseline / contact loss disabled.",
    )
    parser.add_argument("--num-epochs", type=int, default=50, help="Additional epochs to train beyond the checkpoint.")
    parser.add_argument(
        "--exp-name", type=str, required=True,
        help="e.g. 'contact_baseline' or 'contact_w1' -- must differ between the two runs.",
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


def build_simple_loss(
    patch_file: Path, pairs_arg: str | None, margin: float, beta: float, device: str,
) -> SimpleContactLoss:
    if pairs_arg is None:
        raise ValueError("--pairs is required for --loss-type simple.")
    requested = parse_pairs_arg(pairs_arg)

    all_pairs = load_contact_patch_pairs(patch_file)
    canon = lambda p: (min(p.cluster_a, p.cluster_b), max(p.cluster_a, p.cluster_b))
    filtered = [p for p in all_pairs if canon(p) in requested]
    missing = requested - {canon(p) for p in filtered}
    if missing:
        raise ValueError(f"Requested pair(s) not found in {patch_file}: {sorted(missing)}")

    return SimpleContactLoss(patch_pairs=filtered, margin=margin, beta=beta).to(torch.device(device))


def build_frame_loss(
    patch_file: Path, summary_csv: Path, pairs_arg: str | None,
    margin: float, beta: float, min_flagged_frames: int, device: str,
) -> FrameContactLoss:
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


def install_contact_loss_patch(contact_loss: ContactLoss | None, contact_weight: float) -> None:
    """Add an optional contact loss term to every training step, additively.

    Installed unconditionally (even for the baseline run, with contact_loss=None)
    so both the baseline and treatment runs execute the identical code path --
    only the injected loss term differs.
    """
    import flow3d.trainer as trainer_mod

    original_compute_losses = trainer_mod.Trainer.compute_losses
    is_frame_gated = isinstance(contact_loss, FrameContactLoss)

    def patched_compute_losses(self, batch):
        loss, stats, num_rays_per_step, num_rays_per_sec = original_compute_losses(self, batch)
        if contact_loss is not None and contact_weight > 0:
            ts = batch["ts"]
            means_fg, _ = self.model.compute_poses_fg(ts)  # (G, B, 3)
            if is_frame_gated:
                per_frame_losses = [
                    contact_loss(means_fg[:, i], frame_idx=int(ts[i].item()))
                    for i in range(means_fg.shape[1])
                ]
            else:
                per_frame_losses = [contact_loss(means_fg[:, i]) for i in range(means_fg.shape[1])]
            contact_value = torch.stack(per_frame_losses).mean()
            loss = loss + contact_weight * contact_value
            stats["train/contact_loss"] = contact_value.detach()
        return loss, stats, num_rays_per_step, num_rays_per_sec

    trainer_mod.Trainer.compute_losses = patched_compute_losses


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    src_ckpt = (
        args.src_ckpt.expanduser().resolve() if args.src_ckpt is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )
    patch_file = (
        args.patch_file.expanduser().resolve() if args.patch_file is not None
        else work_dir / "analysis" / "contact_patches" / "contact_patches.pt"
    )
    src_cfg_path = work_dir / "cfg.yaml"

    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt}")
    if not src_cfg_path.is_file():
        raise FileNotFoundError(f"Source cfg.yaml not found: {src_cfg_path}")
    if args.contact_weight > 0 and not patch_file.is_file():
        raise FileNotFoundError(f"Contact patch file not found: {patch_file}")

    src_cfg = yaml.safe_load(src_cfg_path.read_text())

    fresh_ckpt = work_dir / "checkpoints" / f"last_fresh_optim_for_{args.exp_name}.ckpt"
    start_epoch = make_fresh_optimizer_checkpoint(src_ckpt, fresh_ckpt)
    print(f"Source checkpoint: {src_ckpt} (epoch={start_epoch})")
    print(f"Fresh-optimizer copy (model weights only): {fresh_ckpt}")

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

    contact_loss = None
    if args.contact_weight > 0:
        if args.loss_type == "simple":
            contact_loss = build_simple_loss(
                patch_file, args.pairs, args.margin, args.contact_beta, args.device
            )
            print(
                f"Contact loss ENABLED (simple): pairs={args.pairs} margin={args.margin} "
                f"beta={args.contact_beta} weight={args.contact_weight}"
            )
        else:
            summary_csv = (
                args.summary_csv.expanduser().resolve() if args.summary_csv is not None
                else work_dir / "analysis" / "contact_patch_distances" / "contact_patch_pair_summary.csv"
            )
            if not summary_csv.is_file():
                raise FileNotFoundError(
                    f"Pair summary CSV not found: {summary_csv}. Run "
                    "flow3d/analysis/contactpatch_distance.py first, or pass --summary-csv."
                )
            contact_loss = build_frame_loss(
                patch_file, summary_csv, args.pairs, args.margin, args.contact_beta,
                args.min_flagged_frames, args.device,
            )
            print(
                f"Contact loss ENABLED (frame-gated): summary_csv={summary_csv} "
                f"pairs={args.pairs or 'all passing --min-flagged-frames'} "
                f"min_flagged_frames={args.min_flagged_frames} margin={args.margin} "
                f"beta={args.contact_beta} weight={args.contact_weight}"
            )
    else:
        print("Contact loss DISABLED (baseline run; pass --contact-weight > 0 to enable).")

    install_contact_loss_patch(contact_loss, args.contact_weight)

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

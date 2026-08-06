#!/usr/bin/env python3
"""
Run the full MotionScale contact-analysis pipeline for one trained checkpoint:

    cluster_pairs.py -> contactpatch.py -> contactpatch_distance.py
        -> contactpatch_render.py -> render_output_novelview.py -> gt_analyze.py

Each stage writes into <work-dir>/analysis/... (or, for render_output_novelview.py,
<work-dir>/novel_views/...) using its own defaults, and the next stage reads
from those same defaults (--ckpt/--pair-file/--contact-patch-file all default
to the previous stage's output path), so this script just runs them in order
with --work-dir (and a few shared overrides) forwarded to each.

Dependency note
----------------
gt_analyze.py requires flow3d/analysis/2drender/render_output_novelview.py to
have already been run for this work-dir (it reads
<work-dir>/novel_views/selection_summary.json). This script now runs that
stage automatically, right before gt_analyze.py.

Example
-------
    python flow3d/analysis/run_analysis.py \\
        --work-dir outputs/davis/spin/2026_08_03_08_41_11__spin_run1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    "cluster_pairs",
    "contactpatch",
    "contactpatch_distance",
    "contactpatch_render",
    "render_novelview",
    "gt_analyze",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument(
        "--dycheck-dir", type=Path, default=Path("data/DyCheck/spin"),
        help="Forwarded to contactpatch_render.py and gt_analyze.py.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/davis/spin.yaml"),
        help="Forwarded to gt_analyze.py.",
    )
    parser.add_argument("--seq-name", type=str, default="spin", help="Forwarded to gt_analyze.py.")
    parser.add_argument(
        "--contactpatch-render-frames", type=str, default="0,20,40,60,80,100,120,140",
        help="Forwarded to contactpatch_render.py as --frames. Its own default "
             "(0,20,40,60,80,99) was tuned for 100-frame scenes; spin has 142.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip", type=str, default="",
        help=f"Comma list of stages to skip, from: {','.join(STAGES)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    invalid = skip - set(STAGES)
    if invalid:
        raise ValueError(f"Unknown stage(s) in --skip: {invalid}; choose from {STAGES}")

    work_dir = args.work_dir
    ckpt = work_dir / "checkpoints" / "last.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    commands = {
        "cluster_pairs": [
            sys.executable, "flow3d/analysis/cluster_pairs.py",
            "--work-dir", str(work_dir), "--device", args.device,
        ],
        "contactpatch": [
            sys.executable, "flow3d/analysis/contactpatch.py",
            "--work-dir", str(work_dir), "--device", args.device,
        ],
        "contactpatch_distance": [
            sys.executable, "flow3d/analysis/contactpatch_distance.py",
            "--work-dir", str(work_dir), "--device", args.device,
        ],
        "contactpatch_render": [
            sys.executable, "flow3d/analysis/2drender/contactpatch_render.py",
            "--work-dir", str(work_dir),
            "--dycheck-dir", str(args.dycheck_dir),
            "--frames", args.contactpatch_render_frames,
            "--device", args.device,
        ],
        "render_novelview": [
            sys.executable, "flow3d/analysis/2drender/render_output_novelview.py",
            "--ckpt", str(ckpt),
            "--config", str(args.config),
            "--seq_name", args.seq_name,
            "--dycheck_dir", str(args.dycheck_dir),
            "--device", args.device,
        ],
        "gt_analyze": [
            sys.executable, "flow3d/analysis/2drender/gt_analyze.py",
            "--work-dir", str(work_dir),
            "--dycheck-dir", str(args.dycheck_dir),
            "--config", str(args.config),
            "--seq-name", args.seq_name,
            "--device", args.device,
        ],
    }

    for stage in STAGES:
        if stage in skip:
            print(f"[skip] {stage}")
            continue
        cmd = commands[stage]
        print(f"\n{'=' * 72}\n[run] {stage}\n{'=' * 72}")
        print(" ".join(str(c) for c in cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)

    if args.dry_run:
        print("\n[dry-run] no commands were executed.")
    else:
        print(f"\nAll stages complete. See {work_dir}/analysis/ for outputs.")


if __name__ == "__main__":
    main()

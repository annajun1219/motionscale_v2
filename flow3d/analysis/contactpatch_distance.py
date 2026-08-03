#!/usr/bin/env python3
"""
Measure temporal separation between fixed canonical contact patches.

Input
-----
<work-dir>/analysis/contact_patches/contact_patches.pt

The patch Gaussian identities are fixed in canonical space, but Gaussian-to-
Gaussian correspondences are NOT fixed. For every frame and every cluster pair:

1. Transform the saved contact-patch Gaussians to their current 3D positions.
2. Compute A -> B nearest-neighbour distances.
3. Compute B -> A nearest-neighbour distances.
4. Concatenate both directed distance sets.
5. Report the median as the main set-to-set contact-patch distance.

This measures whether the entire saved junction region opens over time while
allowing the nearest neighbour of each Gaussian to change at every frame.

Outputs
-------
<output-dir>/
    contact_patch_distances_all.csv
    contact_patch_pair_summary.csv
    analysis_report.json
    plots/
        pair_<A>_<B>_distance.png
        all_pairs_median_distance.png
        all_pairs_relative_distance.png

Example
-------
cd /workspace/motionscale

PYTHONPATH=/workspace/motionscale \
/workspace/miniconda3/envs/motionscale/bin/python \
flow3d/analysis/contactpatch_distance.py \
  --work-dir /workspace/motionscale/outputs/davis/camel/2026_07_12_15_35_50__camel_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from flow3d.renderer import Renderer


EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute bidirectional patch-to-patch nearest-neighbour "
            "distances at every frame."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--contact-patch-file",
        type=Path,
        default=None,
        help=(
            "Default: <work-dir>/analysis/contact_patches/contact_patches.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <work-dir>/analysis/contact_patch_distances"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="all",
        help='Use "all" or a subset such as "0-16,1-28".',
    )
    parser.add_argument(
        "--exclude-pairs",
        type=str,
        default=None,
        help='Pairs to exclude, such as "41-45,36-39".',
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="all",
        help='Use "all", a comma-separated list, or ranges such as "0-20,40,50-60".',
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=None,
        help=(
            "Frame used for relative distance and increase calculations. "
            "Default: first selected frame."
        ),
    )
    parser.add_argument(
        "--increase-threshold",
        type=float,
        default=1.5,
        help=(
            "A frame is flagged when median distance / reference median "
            "is at least this value."
        ),
    )
    parser.add_argument(
        "--absolute-increase-threshold",
        type=float,
        default=0.0,
        help=(
            "Additional absolute increase required for flagging. "
            "Use 0 to disable the absolute requirement."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Save CSV/JSON only.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.increase_threshold <= 0:
        raise ValueError("--increase-threshold must be > 0")
    if args.absolute_increase_threshold < 0:
        raise ValueError("--absolute-increase-threshold must be >= 0")


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_pair_subset(text: str) -> set[tuple[int, int]] | None:
    if text.strip().lower() == "all":
        return None

    pairs: set[tuple[int, int]] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.replace(":", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair token: {token}")
        a, b = int(parts[0]), int(parts[1])
        if a == b:
            raise ValueError(f"Self-pair is invalid: {token}")
        pairs.add((min(a, b), max(a, b)))

    if not pairs:
        raise ValueError("No valid pair was supplied.")
    return pairs


def parse_frames(text: str, total_frames: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total_frames))

    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid frame range: {token}")
            start, end = int(parts[0]), int(parts[1])
            if end < start:
                raise ValueError(f"Descending frame range is invalid: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    frames = sorted(selected)
    if not frames:
        raise ValueError("No frame was selected.")

    invalid = [frame for frame in frames if frame < 0 or frame >= total_frames]
    if invalid:
        raise ValueError(
            f"Invalid frames {invalid}; checkpoint contains {total_frames} frames."
        )
    return frames


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(
        *points.shape[:-1],
        1,
        device=points.device,
        dtype=points.dtype,
    )
    return torch.cat([points, ones], dim=-1)


def normalize_transform_tensor(
    value: Any,
    num_gaussians: int,
    num_frames: int,
) -> torch.Tensor:
    if isinstance(value, dict):
        for key in (
            "transforms",
            "full_transforms",
            "gaussian_transforms",
            "means_transforms",
        ):
            if key in value:
                value = value[key]
                break

    if isinstance(value, (tuple, list)):
        candidates = [
            item
            for item in value
            if isinstance(item, torch.Tensor)
            and item.ndim == 4
            and item.shape[-2:] == (3, 4)
        ]
        if candidates:
            value = candidates[0]

    if not isinstance(value, torch.Tensor):
        raise TypeError("Could not extract a Gaussian transform tensor.")

    if value.ndim != 4 or value.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected [G,T,3,4] or [T,G,3,4], got {tuple(value.shape)}."
        )

    if value.shape[:2] == (num_gaussians, num_frames):
        return value
    if value.shape[:2] == (num_frames, num_gaussians):
        return value.permute(1, 0, 2, 3)

    raise ValueError(
        f"Unexpected transform shape {tuple(value.shape)} for "
        f"G={num_gaussians}, T={num_frames}."
    )


def compute_positions(
    model: Any,
    frame_ids: torch.Tensor,
    canonical_means: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    errors: list[str] = []

    for method_name in ("compute_transforms", "compute_full_transforms"):
        method = getattr(model, method_name, None)
        if method is None:
            continue

        try:
            raw = method(frame_ids)
            transforms = normalize_transform_tensor(
                raw,
                num_gaussians=int(canonical_means.shape[0]),
                num_frames=int(frame_ids.numel()),
            )
            positions = torch.einsum(
                "gtij,gj->gti",
                transforms,
                to_homogeneous(canonical_means),
            )
            return positions, f"model.{method_name}(frame_ids)"
        except Exception as exc:
            errors.append(f"{method_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not compute frame-specific Gaussian positions:\n"
        + "\n".join(errors)
    )


def as_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)


def bidirectional_nn_metrics(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
) -> dict[str, float]:
    array_a = points_a.detach().float().cpu().numpy()
    array_b = points_b.detach().float().cpu().numpy()

    if len(array_a) == 0 or len(array_b) == 0:
        raise ValueError("Both contact patches must contain at least one Gaussian.")

    tree_a = cKDTree(array_a)
    tree_b = cKDTree(array_b)

    distance_ab, _ = tree_b.query(array_a, k=1)
    distance_ba, _ = tree_a.query(array_b, k=1)

    distance_ab = np.asarray(distance_ab, dtype=np.float64)
    distance_ba = np.asarray(distance_ba, dtype=np.float64)
    balanced = np.concatenate([distance_ab, distance_ba])

    return {
        "median": float(np.median(balanced)),
        "mean": float(np.mean(balanced)),
        "p90": float(np.quantile(balanced, 0.90)),
        "p95": float(np.quantile(balanced, 0.95)),
        "max": float(np.max(balanced)),
        "a_to_b_median": float(np.median(distance_ab)),
        "b_to_a_median": float(np.median(distance_ba)),
        "a_to_b_p90": float(np.quantile(distance_ab, 0.90)),
        "b_to_a_p90": float(np.quantile(distance_ba, 0.90)),
    }


def safe_ratio(value: float, reference: float) -> float:
    if abs(reference) <= EPS:
        return 1.0 if abs(value) <= EPS else float("inf")
    return value / reference


def save_pair_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    cluster_a: int,
    cluster_b: int,
    reference_frame: int,
    threshold: float,
) -> None:
    frames = np.asarray([row["frame"] for row in rows], dtype=np.int64)
    medians = np.asarray(
        [row["bidirectional_median_distance"] for row in rows],
        dtype=np.float64,
    )
    p90 = np.asarray(
        [row["bidirectional_p90_distance"] for row in rows],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [row["median_distance_ratio_to_reference"] for row in rows],
        dtype=np.float64,
    )

    fig, left_axis = plt.subplots(figsize=(10, 5))
    left_axis.plot(frames, medians, marker="o", markersize=3, label="median")
    left_axis.plot(frames, p90, linewidth=1.5, label="p90")
    left_axis.set_xlabel("Frame")
    left_axis.set_ylabel("3D patch-to-patch NN distance")
    left_axis.grid(alpha=0.3)
    left_axis.axvline(
        reference_frame,
        linestyle="--",
        linewidth=1,
        label=f"reference frame {reference_frame}",
    )

    right_axis = left_axis.twinx()
    right_axis.plot(
        frames,
        ratios,
        linestyle=":",
        linewidth=1.5,
        label="median / reference",
    )
    right_axis.axhline(
        threshold,
        linestyle="--",
        linewidth=1,
        label=f"ratio threshold {threshold:g}",
    )
    right_axis.set_ylabel("Median distance ratio")

    left_lines, left_labels = left_axis.get_legend_handles_labels()
    right_lines, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(
        left_lines + right_lines,
        left_labels + right_labels,
        loc="upper left",
    )

    plt.title(f"Contact-patch distance: cluster {cluster_a}-{cluster_b}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_all_pairs_heatmap(
    rows: list[dict[str, Any]],
    output_path: Path,
    value_key: str,
    title: str,
    colorbar_label: str,
) -> None:
    if not rows:
        return

    pairs = sorted(
        {
            (int(row["cluster_a"]), int(row["cluster_b"]))
            for row in rows
        }
    )
    frames = sorted({int(row["frame"]) for row in rows})

    pair_to_index = {pair: i for i, pair in enumerate(pairs)}
    frame_to_index = {frame: i for i, frame in enumerate(frames)}

    matrix = np.full((len(pairs), len(frames)), np.nan, dtype=np.float64)
    for row in rows:
        pair = (int(row["cluster_a"]), int(row["cluster_b"]))
        matrix[pair_to_index[pair], frame_to_index[int(row["frame"])]] = float(
            row[value_key]
        )

    figure_width = max(10, len(frames) * 0.08)
    figure_height = max(5, len(pairs) * 0.22)

    plt.figure(figsize=(figure_width, figure_height))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=colorbar_label)

    frame_tick_step = max(1, len(frames) // 20)
    frame_tick_indices = list(range(0, len(frames), frame_tick_step))
    plt.xticks(
        frame_tick_indices,
        [str(frames[index]) for index in frame_tick_indices],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(len(pairs)),
        [f"{a}-{b}" for a, b in pairs],
    )
    plt.xlabel("Frame")
    plt.ylabel("Cluster pair")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    work_dir = args.work_dir.expanduser().resolve()
    checkpoint = (
        args.ckpt.expanduser().resolve()
        if args.ckpt is not None
        else work_dir / "checkpoints" / "last.ckpt"
    )
    contact_patch_file = (
        args.contact_patch_file.expanduser().resolve()
        if args.contact_patch_file is not None
        else work_dir / "analysis" / "contact_patches" / "contact_patches.pt"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "analysis" / "contact_patch_distances"
    )
    plot_dir = output_dir / "plots"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not contact_patch_file.is_file():
        raise FileNotFoundError(
            f"Contact-patch file not found: {contact_patch_file}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    payload = torch_load_cpu(contact_patch_file)
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise TypeError(
            "contact_patches.pt must contain a dictionary with a 'pairs' field."
        )

    requested_pairs = parse_pair_subset(args.pairs)
    excluded_pairs = (
        parse_pair_subset(args.exclude_pairs)
        if args.exclude_pairs is not None
        else set()
    )

    with torch.inference_mode():
        renderer = Renderer.init_from_checkpoint(
            str(checkpoint),
            device,
            work_dir=str(work_dir),
            port=None,
        )
        model = renderer.model
        model.eval()

        canonical_means = model.fg.params["means"].detach()
        total_frames = int(model.num_frames)
        frames = parse_frames(args.frames, total_frames)

        reference_frame = (
            frames[0]
            if args.reference_frame is None
            else int(args.reference_frame)
        )
        if reference_frame not in frames:
            raise ValueError(
                f"Reference frame {reference_frame} is not among selected frames."
            )

        frame_ids = torch.tensor(
            frames,
            dtype=torch.long,
            device=canonical_means.device,
        )
        positions, transform_source = compute_positions(
            model=model,
            frame_ids=frame_ids,
            canonical_means=canonical_means,
        )

        selected_entries: list[tuple[str, dict[str, Any], int, int]] = []
        available_pairs: set[tuple[int, int]] = set()

        for pair_key, entry in payload["pairs"].items():
            if not isinstance(entry, dict):
                raise TypeError(f"Invalid pair entry: {pair_key!r}")

            cluster_a = int(entry["cluster_a"])
            cluster_b = int(entry["cluster_b"])
            canonical_pair = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
            available_pairs.add(canonical_pair)

            if requested_pairs is not None and canonical_pair not in requested_pairs:
                continue
            if canonical_pair in excluded_pairs:
                continue

            selected_entries.append(
                (str(pair_key), entry, cluster_a, cluster_b)
            )

        if requested_pairs is not None:
            missing = sorted(requested_pairs - available_pairs)
            if missing:
                raise ValueError(
                    "Requested pairs are absent from contact_patches.pt: "
                    + ", ".join(f"{a}-{b}" for a, b in missing)
                )

        missing_excluded = sorted(excluded_pairs - available_pairs)
        if missing_excluded:
            print(
                "[Warning] Excluded pairs absent from contact_patches.pt: "
                + ", ".join(f"{a}-{b}" for a, b in missing_excluded)
            )

        if not selected_entries:
            raise RuntimeError("No contact-patch pair was selected.")

        all_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        local_frame_of_reference = frames.index(reference_frame)

        for pair_index, (pair_key, entry, cluster_a, cluster_b) in enumerate(
            selected_entries,
            start=1,
        ):
            patch_a = as_long_tensor(
                entry["contact_patch_global_indices_a"],
                device=canonical_means.device,
            )
            patch_b = as_long_tensor(
                entry["contact_patch_global_indices_b"],
                device=canonical_means.device,
            )

            if patch_a.numel() == 0 or patch_b.numel() == 0:
                raise RuntimeError(
                    f"Pair {cluster_a}-{cluster_b} has an empty contact patch."
                )

            pair_metrics: list[dict[str, float]] = []
            for local_t, frame in enumerate(frames):
                metrics = bidirectional_nn_metrics(
                    positions[patch_a, local_t],
                    positions[patch_b, local_t],
                )
                pair_metrics.append(metrics)

            reference_metrics = pair_metrics[local_frame_of_reference]
            reference_median = reference_metrics["median"]

            pair_rows: list[dict[str, Any]] = []
            for frame, metrics in zip(frames, pair_metrics):
                median_increase = metrics["median"] - reference_median
                median_ratio = safe_ratio(metrics["median"], reference_median)

                flagged = (
                    median_ratio >= args.increase_threshold
                    and median_increase >= args.absolute_increase_threshold
                )

                row = {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "pair_key": pair_key,
                    "frame": frame,
                    "reference_frame": reference_frame,
                    "patch_count_a": int(patch_a.numel()),
                    "patch_count_b": int(patch_b.numel()),
                    "bidirectional_median_distance": metrics["median"],
                    "bidirectional_mean_distance": metrics["mean"],
                    "bidirectional_p90_distance": metrics["p90"],
                    "bidirectional_p95_distance": metrics["p95"],
                    "bidirectional_max_distance": metrics["max"],
                    "a_to_b_median_distance": metrics["a_to_b_median"],
                    "b_to_a_median_distance": metrics["b_to_a_median"],
                    "a_to_b_p90_distance": metrics["a_to_b_p90"],
                    "b_to_a_p90_distance": metrics["b_to_a_p90"],
                    "reference_median_distance": reference_median,
                    "median_distance_absolute_increase": median_increase,
                    "median_distance_ratio_to_reference": median_ratio,
                    "flagged_increase": bool(flagged),
                }
                pair_rows.append(row)
                all_rows.append(row)

            median_values = np.asarray(
                [row["bidirectional_median_distance"] for row in pair_rows],
                dtype=np.float64,
            )
            ratio_values = np.asarray(
                [row["median_distance_ratio_to_reference"] for row in pair_rows],
                dtype=np.float64,
            )
            p90_values = np.asarray(
                [row["bidirectional_p90_distance"] for row in pair_rows],
                dtype=np.float64,
            )
            flagged_frames = [
                int(row["frame"])
                for row in pair_rows
                if bool(row["flagged_increase"])
            ]

            max_median_index = int(np.argmax(median_values))
            max_ratio_index = int(np.argmax(ratio_values))

            summary = {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "pair_key": pair_key,
                "patch_count_a": int(patch_a.numel()),
                "patch_count_b": int(patch_b.numel()),
                "reference_frame": reference_frame,
                "reference_median_distance": reference_median,
                "minimum_median_distance": float(np.min(median_values)),
                "median_of_frame_medians": float(np.median(median_values)),
                "maximum_median_distance": float(np.max(median_values)),
                "maximum_median_distance_frame": int(frames[max_median_index]),
                "maximum_p90_distance": float(np.max(p90_values)),
                "maximum_ratio_to_reference": float(np.max(ratio_values)),
                "maximum_ratio_frame": int(frames[max_ratio_index]),
                "flagged_frame_count": len(flagged_frames),
                "first_flagged_frame": (
                    flagged_frames[0] if flagged_frames else ""
                ),
                "flagged_frames": ",".join(map(str, flagged_frames)),
            }
            summary_rows.append(summary)

            if not args.no_plots:
                save_pair_plot(
                    rows=pair_rows,
                    output_path=plot_dir / f"pair_{cluster_a}_{cluster_b}_distance.png",
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    reference_frame=reference_frame,
                    threshold=args.increase_threshold,
                )

            print(
                f"[{pair_index:03d}/{len(selected_entries):03d}] "
                f"{cluster_a}-{cluster_b}: "
                f"reference={reference_median:.6f}, "
                f"max={summary['maximum_median_distance']:.6f} "
                f"(frame {summary['maximum_median_distance_frame']}), "
                f"max_ratio={summary['maximum_ratio_to_reference']:.3f}, "
                f"flagged={summary['flagged_frame_count']}"
            )

    write_csv(
        output_dir / "contact_patch_distances_all.csv",
        all_rows,
    )
    write_csv(
        output_dir / "contact_patch_pair_summary.csv",
        summary_rows,
    )

    if not args.no_plots:
        save_all_pairs_heatmap(
            rows=all_rows,
            output_path=plot_dir / "all_pairs_median_distance.png",
            value_key="bidirectional_median_distance",
            title="Bidirectional contact-patch median distance",
            colorbar_label="3D median NN distance",
        )
        save_all_pairs_heatmap(
            rows=all_rows,
            output_path=plot_dir / "all_pairs_relative_distance.png",
            value_key="median_distance_ratio_to_reference",
            title="Contact-patch median distance relative to reference",
            colorbar_label="Median / reference median",
        )

    report = {
        "work_dir": str(work_dir),
        "checkpoint": str(checkpoint),
        "contact_patch_file": str(contact_patch_file),
        "output_dir": str(output_dir),
        "pair_count": len(summary_rows),
        "excluded_pairs": [list(pair) for pair in sorted(excluded_pairs)],
        "frame_count": len(frames),
        "frames": frames,
        "reference_frame": reference_frame,
        "transform_source": transform_source,
        "distance_definition": {
            "patch_identity": (
                "Canonical contact-patch Gaussian indices are fixed for all frames."
            ),
            "correspondence": (
                "No fixed A-B Gaussian pairs. A-to-B and B-to-A nearest "
                "neighbours are recomputed independently at every frame."
            ),
            "main_metric": (
                "Median of the concatenated A-to-B and B-to-A nearest-neighbour "
                "3D distances."
            ),
        },
        "flagging": {
            "ratio_threshold": args.increase_threshold,
            "absolute_increase_threshold": args.absolute_increase_threshold,
            "condition": (
                "median/reference >= ratio threshold AND "
                "median-reference >= absolute threshold"
            ),
        },
        "outputs": {
            "per_frame_csv": str(
                output_dir / "contact_patch_distances_all.csv"
            ),
            "pair_summary_csv": str(
                output_dir / "contact_patch_pair_summary.csv"
            ),
            "plot_dir": "" if args.no_plots else str(plot_dir),
        },
        "pair_summary": summary_rows,
    }

    with (output_dir / "analysis_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 78)
    print("Contact-patch temporal distance analysis complete")
    print(f"Pairs       : {len(summary_rows)}")
    print(f"Frames      : {len(frames)}")
    print(f"Output      : {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
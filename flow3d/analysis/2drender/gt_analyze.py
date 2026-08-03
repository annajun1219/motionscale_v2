#!/usr/bin/env python3
"""
Compare novel-view renders against real DyCheck photographs.

Input
-----
The outputs of flow3d/analysis/2drender/render_output_novelview.py:
    <novel-views-dir>/selection_summary.json
    <novel-views-dir>/<view>_frame<NNNN>.png (+ camera0_frame<NNNN>_ref.png)

Output
------
<output-dir>/
    gt_comparison_all.csv          (one row per render/GT pair)
    gt_comparison_cam_summary.csv  (metrics averaged per view)
    analysis_report.json
    comparisons/<stem>_compare.png (render | GT | abs-diff side by side)
    plots/psnr_by_cam.png, plots/ssim_by_cam.png

IMPORTANT CAVEAT -- read before trusting the camera1/camera2 numbers
----------------------------------------------------------------------
For the reference camera (camera0), the render uses the exact pose the model
was trained and evaluated with, so its metrics measure genuine reconstruction
fidelity (in-distribution).

camera1 and camera2 are real, physically-captured DyCheck cameras that were
never used for training, so they have real ground-truth photos -- but
render_output_novelview.py can only *approximate* their pose: it transplants
DyCheck's real relative rotation (camera0 -> camera1/camera2) onto the
trained scene, keeping camera0's own intrinsics and an approximate orbit
radius, because the trained (mega-sam) coordinate frame has no known metric
scale relating it to DyCheck's real camera positions. So a large pixel error
against camera1/camera2's real photo can mean either (a) genuinely poor
novel-view reconstruction, or (b) a real but imperfect viewpoint/FOV match --
this script cannot tell the two apart. Treat the camera0 row as the
trustworthy quality signal, and camera1/camera2 as a rough,
pessimistically-biased indicator.

novel1/novel2/novel3 are purely synthetic viewpoints with no corresponding
physical camera, so they have no ground truth and are always skipped here.

Example
-------
    python flow3d/analysis/2drender/gt_analyze.py \\
        --work-dir outputs/davis/spin/2026_08_03_08_41_11__spin_run1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dataclasses import asdict
from PIL import Image

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.metrics import mLPIPS, mPSNR, mSSIM, compute_psnr

# Real, physically-captured DyCheck cameras that have ground truth but were
# never used for training. novel1/novel2/novel3 are synthetic and have none.
REAL_DYCHECK_VIEWS = ["camera1", "camera2"]

# data/DAVIS/JPEGImages/480p/<seq>/ keeps every 3rd camera0 frame from
# data/DyCheck/<seq>/rgb/1x/ -- must match render_output_novelview.py.
DYCHECK_SUBSAMPLE_INTERVAL = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument(
        "--novel-views-dir", type=Path, default=None,
        help="Default: <work-dir>/novel_views",
    )
    parser.add_argument(
        "--dycheck-dir", type=Path, default=Path("data/DyCheck/spin"),
        help="DyCheck sequence dir. GT for camera1/camera2 is read from "
             "<dycheck-dir>/rgb/1x/<cam_id>_<NNNNN>.png.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/spin.yaml"))
    parser.add_argument("--seq-name", type=str, default="spin")
    parser.add_argument("--ref-view", type=str, default="camera0")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <work-dir>/analysis/gt_comparison",
    )
    parser.add_argument(
        "--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
    )
    parser.add_argument("--metrics-only", action="store_true", help="Skip saving comparison images.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def load_dataset(config_path: Path, seq_name: str) -> CasualDataset:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(
        root_dir=scene_cfg["data_dir"], seq_name=seq_name, **scene_cfg.get("data", {}),
    )
    return CasualDataset(**asdict(data_cfg))


def load_render_entries(novel_views_dir: Path) -> list[dict[str, Any]]:
    """Read render_output_novelview.py's selection_summary.json (view/frame/file per render)."""
    summary_path = novel_views_dir / "selection_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{summary_path} not found. Run render_output_novelview.py first "
            "(this script reads its selection_summary.json, not filenames directly)."
        )
    summary = json.loads(summary_path.read_text())
    return summary["views"]


def find_gt_path(
    view: str, frame_idx: int, ref_view: str, ref_img_dir: Path, dycheck_dir: Path
) -> Path | None:
    if view == ref_view:
        # Handled by the caller via dataset.frame_names; kept here for symmetry.
        return None
    if view not in REAL_DYCHECK_VIEWS:
        # novel1/novel2/novel3: synthetic viewpoints, no physical camera exists there.
        return None
    cam_id = view.replace("camera", "")
    orig_idx = frame_idx * DYCHECK_SUBSAMPLE_INTERVAL
    cam_dir = dycheck_dir / "rgb" / "1x"
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = cam_dir / f"{cam_id}_{orig_idx:05d}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_image_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_comparison_image(path: Path, render: np.ndarray, gt: np.ndarray, title: str) -> None:
    diff = np.abs(render - gt).mean(axis=-1)
    diff_vis = plt.get_cmap("inferno")(np.clip(diff * 3.0, 0.0, 1.0))[..., :3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(render)
    axes[0].set_title("Render")
    axes[1].imshow(gt)
    axes[1].set_title("GT")
    axes[2].imshow(diff_vis)
    axes[2].set_title(f"abs diff (mean={diff.mean():.4f}, x3 gain)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(rows: list[dict[str, Any]], value_key: str, ylabel: str, output_path: Path) -> None:
    if not rows:
        return
    views = sorted({row["view"] for row in rows}, key=lambda c: (c != rows[0]["ref_view"], c))
    means = []
    is_ref_by_view = {}
    for view in views:
        values = [row[value_key] for row in rows if row["view"] == view and np.isfinite(row[value_key])]
        means.append(float(np.mean(values)) if values else float("nan"))
        is_ref_by_view[view] = any(row["is_ref"] for row in rows if row["view"] == view)

    colors = ["tab:orange" if is_ref_by_view[view] else "tab:blue" for view in views]
    plt.figure(figsize=(max(6, len(views) * 1.2), 4.5))
    plt.bar(views, means, color=colors)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by view (orange = reference / in-distribution)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    novel_views_dir = (
        args.novel_views_dir.expanduser().resolve()
        if args.novel_views_dir is not None
        else work_dir / "novel_views"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "analysis" / "gt_comparison"
    )
    comparisons_dir = output_dir / "comparisons"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        comparisons_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not novel_views_dir.is_dir():
        raise FileNotFoundError(f"Novel-views dir not found: {novel_views_dir}")
    if not args.dycheck_dir.is_dir():
        raise FileNotFoundError(f"DyCheck sequence dir not found: {args.dycheck_dir}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = load_dataset(args.config, args.seq_name)
    ref_img_dir = Path(dataset.img_dir)

    entries = load_render_entries(novel_views_dir)
    if not entries:
        raise RuntimeError(f"No renders listed in {novel_views_dir}/selection_summary.json.")

    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS(net_type=args.lpips_net).to(device)

    rows: list[dict[str, Any]] = []
    missing_gt: list[str] = []
    skipped_synthetic = 0

    for entry in entries:
        view = entry["view"]
        frame_idx = entry["frame_idx"]
        is_ref = entry.get("is_ref", view == args.ref_view)
        render_path = novel_views_dir / entry["file"]
        if not render_path.is_file():
            print(f"[Warning] {render_path} listed in summary but missing on disk, skipping.")
            continue
        if not (0 <= frame_idx < len(dataset.frame_names)):
            print(f"[Warning] {render_path.name}: frame_idx {frame_idx} out of dataset range, skipping.")
            continue
        frame_name = dataset.frame_names[frame_idx]

        if is_ref:
            gt_path = next(
                (ref_img_dir / f"{frame_name}{ext}" for ext in (".jpg", ".jpeg", ".png")
                 if (ref_img_dir / f"{frame_name}{ext}").is_file()),
                None,
            )
        elif view in REAL_DYCHECK_VIEWS:
            gt_path = find_gt_path(view, frame_idx, args.ref_view, ref_img_dir, args.dycheck_dir)
        else:
            skipped_synthetic += 1
            continue

        if gt_path is None:
            missing_gt.append(f"{view}/{frame_name}")
            continue

        render = load_image_float(render_path)
        gt = load_image_float(gt_path)
        if render.shape != gt.shape:
            gt_img = Image.fromarray((gt * 255).astype(np.uint8)).resize(
                (render.shape[1], render.shape[0]), Image.BILINEAR
            )
            gt = np.asarray(gt_img, dtype=np.float32) / 255.0

        render_t = torch.from_numpy(render).to(device)[None]
        gt_t = torch.from_numpy(gt).to(device)[None]

        mae = float(np.abs(render - gt).mean())
        mse = float(((render - gt) ** 2).mean())
        psnr = compute_psnr(render_t, gt_t)

        ssim_metric.reset()
        ssim_metric.update(render_t, gt_t)
        ssim = float(ssim_metric.compute().item())

        lpips_metric.reset()
        lpips_metric.update(render_t, gt_t)
        lpips = float(lpips_metric.compute().item())

        row = {
            "view": view,
            "ref_view": args.ref_view,
            "is_ref": is_ref,
            "azimuth_deg": entry.get("azimuth_deg", float("nan")),
            "elevation_deg": entry.get("elevation_deg", float("nan")),
            "frame_idx": frame_idx,
            "frame_name": frame_name,
            "mae": mae,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "render_path": str(render_path),
            "gt_path": str(gt_path),
            "comparison_path": "",
        }

        if not args.metrics_only:
            compare_path = comparisons_dir / f"{render_path.stem}_compare.png"
            kind = "reference (in-distribution)" if is_ref else f"real GT (az={row['azimuth_deg']:.1f} deg)"
            save_comparison_image(
                compare_path, render, gt,
                title=f"{view} frame {frame_name} [{kind}]  "
                      f"PSNR={psnr:.2f} SSIM={ssim:.3f} LPIPS={lpips:.3f}",
            )
            row["comparison_path"] = str(compare_path)

        rows.append(row)
        print(
            f"{view} frame={frame_name} {'REF' if is_ref else 'real-GT'} "
            f"psnr={psnr:.2f} ssim={ssim:.3f} lpips={lpips:.3f} mae={mae:.4f}"
        )

    if skipped_synthetic:
        print(f"[Info] Skipped {skipped_synthetic} synthetic-view render(s) (novel1/2/3 have no GT).")
    if missing_gt:
        print(f"[Warning] No GT image found for {len(missing_gt)} render(s): {missing_gt[:10]}"
              + (" ..." if len(missing_gt) > 10 else ""))
    if not rows:
        raise RuntimeError("No render/GT pairs could be compared (see warnings above).")

    with (output_dir / "gt_comparison_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    views = sorted({row["view"] for row in rows}, key=lambda c: (c != args.ref_view, c))
    view_summary = []
    for view in views:
        view_rows = [row for row in rows if row["view"] == view]
        view_summary.append({
            "view": view,
            "is_ref": view_rows[0]["is_ref"],
            "azimuth_deg": view_rows[0]["azimuth_deg"],
            "elevation_deg": view_rows[0]["elevation_deg"],
            "num_frames": len(view_rows),
            "mean_mae": float(np.mean([r["mae"] for r in view_rows])),
            "mean_mse": float(np.mean([r["mse"] for r in view_rows])),
            "mean_psnr": float(np.mean([r["psnr"] for r in view_rows])),
            "mean_ssim": float(np.mean([r["ssim"] for r in view_rows])),
            "mean_lpips": float(np.mean([r["lpips"] for r in view_rows])),
        })
    with (output_dir / "gt_comparison_cam_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(view_summary[0].keys()))
        writer.writeheader()
        writer.writerows(view_summary)

    save_bar_plot(rows, "psnr", "PSNR (dB, higher is better)", plots_dir / "psnr_by_cam.png")
    save_bar_plot(rows, "ssim", "SSIM (higher is better)", plots_dir / "ssim_by_cam.png")
    save_bar_plot(rows, "lpips", "LPIPS (lower is better)", plots_dir / "lpips_by_cam.png")

    ref_summary = next((s for s in view_summary if s["is_ref"]), None)
    real_novel_summary = [s for s in view_summary if not s["is_ref"]]

    report = {
        "work_dir": str(work_dir),
        "novel_views_dir": str(novel_views_dir),
        "dycheck_dir": str(args.dycheck_dir),
        "ref_view": args.ref_view,
        "lpips_net": args.lpips_net,
        "num_pairs_compared": len(rows),
        "num_pairs_missing_gt": len(missing_gt),
        "num_synthetic_skipped": skipped_synthetic,
        "missing_gt": missing_gt,
        "reference_view_summary": ref_summary,
        "real_camera_summary": real_novel_summary,
        "per_view_summary": view_summary,
        "caveat": (
            "camera0 metrics use the exact trained camera pose and measure "
            "genuine reconstruction fidelity. camera1/camera2 metrics use an "
            "approximate pose (transplanted relative rotation, reference "
            "intrinsics, approximate radius) and conflate reconstruction "
            "error with viewpoint/FOV mismatch -- see the module docstring. "
            "novel1/novel2/novel3 have no ground truth and are never scored."
        ),
    }
    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("GT comparison complete")
    print(f"Pairs compared      : {len(rows)}")
    print(f"Missing GT          : {len(missing_gt)}")
    print(f"Synthetic (no GT)   : {skipped_synthetic}")
    if ref_summary is not None:
        print(
            f"Reference ({args.ref_view}, in-distribution): "
            f"PSNR={ref_summary['mean_psnr']:.2f} SSIM={ref_summary['mean_ssim']:.3f} "
            f"LPIPS={ref_summary['mean_lpips']:.3f}"
        )
    for summary in real_novel_summary:
        print(
            f"{summary['view']} (az={summary['azimuth_deg']:.1f} deg, real GT, approximate pose): "
            f"PSNR={summary['mean_psnr']:.2f} SSIM={summary['mean_ssim']:.3f} "
            f"LPIPS={summary['mean_lpips']:.3f}"
        )
    print(f"Output               : {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()

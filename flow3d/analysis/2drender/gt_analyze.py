#!/usr/bin/env python3
"""
Compare novel-view renders against real DiVa-360 photographs.

Input
-----
The PNGs written by flow3d/analysis/2drender/render_output_novelview.py:
    <novel-views-dir>/<ref_cam>_frame<NNNN>_ref.png
    <novel-views-dir>/<cam>_angle<AA.A>_frame<NNNN>.png

Output
------
<output-dir>/
    gt_comparison_all.csv          (one row per render/GT pair)
    gt_comparison_cam_summary.csv  (metrics averaged per camera)
    analysis_report.json
    comparisons/<stem>_compare.png (render | GT | abs-diff side by side)
    plots/psnr_by_cam.png, plots/ssim_by_cam.png

IMPORTANT CAVEAT -- read before trusting the novel-camera numbers
------------------------------------------------------------------
For the reference camera (--ref-cam, e.g. cam00), the render uses the exact
pose the model was trained and evaluated with, so its metrics measure
genuine reconstruction fidelity (in-distribution).

For every OTHER camera, render_output_novelview.py can only *approximate*
the real camera: it transplants DiVa-360's real relative rotation onto the
trained scene, but keeps the reference camera's own intrinsics and an
approximate orbit radius, because the trained (mega-sam) coordinate frame
has no known metric scale relating it to DiVa-360's real camera positions.
So a large pixel error against a novel camera's real photo can mean either
(a) genuinely poor novel-view reconstruction, or (b) a real but imperfect
viewpoint/FOV match -- this script cannot tell the two apart. Treat the
reference-camera row as the trustworthy quality signal, and the novel-camera
rows as a rough, pessimistically-biased indicator (misalignment inflates
the error on top of whatever the reconstruction itself gets wrong).

Example
-------
    python flow3d/analysis/2drender/gt_analyze.py \\
        --work-dir outputs/davis/dog_cam00/2026_07_26_03_03_54__dog_cam00_run3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

REF_PATTERN = re.compile(r"^(?P<cam>[a-zA-Z0-9]+)_frame(?P<frame>\d+)_ref\.png$")
NOVEL_PATTERN = re.compile(r"^(?P<cam>[a-zA-Z0-9]+)_angle(?P<angle>[-+]?\d+\.\d+)_frame(?P<frame>\d+)\.png$")


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
        "--diva-dir", type=Path, default=Path("data/DiVa360/processed_data/dog"),
        help="DiVa-360 sequence dir. GT for non-reference cameras is read from "
             "<diva-dir>/image/<cam>/<frame_name>.jpg.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--seq-name", type=str, default="dog_cam00")
    parser.add_argument("--ref-cam", type=str, default="cam00")
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


def parse_render_filename(path: Path, ref_cam: str) -> dict[str, Any] | None:
    match = REF_PATTERN.match(path.name)
    if match is not None:
        return {
            "cam": match.group("cam"),
            "frame_idx": int(match.group("frame")),
            "angle_deg": 0.0,
            "is_ref": match.group("cam") == ref_cam,
        }
    match = NOVEL_PATTERN.match(path.name)
    if match is not None:
        return {
            "cam": match.group("cam"),
            "frame_idx": int(match.group("frame")),
            "angle_deg": float(match.group("angle")),
            "is_ref": False,
        }
    return None


def find_gt_path(cam: str, frame_name: str, ref_cam: str, ref_img_dir: Path, diva_dir: Path) -> Path | None:
    if cam == ref_cam:
        candidates = [ref_img_dir / f"{frame_name}{ext}" for ext in (".jpg", ".jpeg", ".png")]
    else:
        cam_dir = diva_dir / "image" / cam
        candidates = [cam_dir / f"{frame_name}{ext}" for ext in (".jpg", ".jpeg", ".png")]
    for candidate in candidates:
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
    cams = sorted({row["cam"] for row in rows}, key=lambda c: (c != rows[0]["ref_cam"], c))
    means = []
    is_ref_by_cam = {}
    for cam in cams:
        values = [row[value_key] for row in rows if row["cam"] == cam and np.isfinite(row[value_key])]
        means.append(float(np.mean(values)) if values else float("nan"))
        is_ref_by_cam[cam] = any(row["is_ref"] for row in rows if row["cam"] == cam)

    colors = ["tab:orange" if is_ref_by_cam[cam] else "tab:blue" for cam in cams]
    plt.figure(figsize=(max(6, len(cams) * 1.2), 4.5))
    plt.bar(cams, means, color=colors)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by camera (orange = reference / in-distribution)")
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
    if not args.diva_dir.is_dir():
        raise FileNotFoundError(f"DiVa-360 sequence dir not found: {args.diva_dir}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = load_dataset(args.config, args.seq_name)
    ref_img_dir = Path(dataset.img_dir)

    render_paths = sorted(novel_views_dir.glob("*.png"))
    parsed = [(path, parse_render_filename(path, args.ref_cam)) for path in render_paths]
    unparsed = [path for path, info in parsed if info is None]
    if unparsed:
        print(f"[Warning] Skipping {len(unparsed)} file(s) with unrecognized name pattern.")
    entries = [(path, info) for path, info in parsed if info is not None]
    if not entries:
        raise RuntimeError(f"No render_output_novelview.py outputs found in {novel_views_dir}.")

    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS(net_type=args.lpips_net).to(device)

    rows: list[dict[str, Any]] = []
    missing_gt: list[str] = []

    for path, info in entries:
        cam, frame_idx, angle_deg, is_ref = (
            info["cam"], info["frame_idx"], info["angle_deg"], info["is_ref"]
        )
        if not (0 <= frame_idx < len(dataset.frame_names)):
            print(f"[Warning] {path.name}: frame_idx {frame_idx} out of dataset range, skipping.")
            continue
        frame_name = dataset.frame_names[frame_idx]

        gt_path = find_gt_path(cam, frame_name, args.ref_cam, ref_img_dir, args.diva_dir)
        if gt_path is None:
            missing_gt.append(f"{cam}/{frame_name}")
            continue

        render = load_image_float(path)
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
            "cam": cam,
            "ref_cam": args.ref_cam,
            "is_ref": is_ref,
            "angle_deg": angle_deg,
            "frame_idx": frame_idx,
            "frame_name": frame_name,
            "mae": mae,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "render_path": str(path),
            "gt_path": str(gt_path),
            "comparison_path": "",
        }

        if not args.metrics_only:
            compare_path = comparisons_dir / f"{path.stem}_compare.png"
            kind = "reference (in-distribution)" if is_ref else f"novel (~{angle_deg:.1f} deg)"
            save_comparison_image(
                compare_path, render, gt,
                title=f"{cam} frame {frame_name} [{kind}]  "
                      f"PSNR={psnr:.2f} SSIM={ssim:.3f} LPIPS={lpips:.3f}",
            )
            row["comparison_path"] = str(compare_path)

        rows.append(row)
        print(
            f"{cam} frame={frame_name} {'REF' if is_ref else f'novel({angle_deg:+.1f}deg)'} "
            f"psnr={psnr:.2f} ssim={ssim:.3f} lpips={lpips:.3f} mae={mae:.4f}"
        )

    if missing_gt:
        print(f"[Warning] No GT image found for {len(missing_gt)} render(s): {missing_gt[:10]}"
              + (" ..." if len(missing_gt) > 10 else ""))
    if not rows:
        raise RuntimeError("No render/GT pairs could be compared (see warnings above).")

    with (output_dir / "gt_comparison_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cams = sorted({row["cam"] for row in rows}, key=lambda c: (c != args.ref_cam, c))
    cam_summary = []
    for cam in cams:
        cam_rows = [row for row in rows if row["cam"] == cam]
        cam_summary.append({
            "cam": cam,
            "is_ref": cam_rows[0]["is_ref"],
            "angle_deg": cam_rows[0]["angle_deg"],
            "num_frames": len(cam_rows),
            "mean_mae": float(np.mean([r["mae"] for r in cam_rows])),
            "mean_mse": float(np.mean([r["mse"] for r in cam_rows])),
            "mean_psnr": float(np.mean([r["psnr"] for r in cam_rows])),
            "mean_ssim": float(np.mean([r["ssim"] for r in cam_rows])),
            "mean_lpips": float(np.mean([r["lpips"] for r in cam_rows])),
        })
    with (output_dir / "gt_comparison_cam_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cam_summary[0].keys()))
        writer.writeheader()
        writer.writerows(cam_summary)

    save_bar_plot(rows, "psnr", "PSNR (dB, higher is better)", plots_dir / "psnr_by_cam.png")
    save_bar_plot(rows, "ssim", "SSIM (higher is better)", plots_dir / "ssim_by_cam.png")
    save_bar_plot(rows, "lpips", "LPIPS (lower is better)", plots_dir / "lpips_by_cam.png")

    ref_summary = next((s for s in cam_summary if s["is_ref"]), None)
    novel_summary = [s for s in cam_summary if not s["is_ref"]]

    report = {
        "work_dir": str(work_dir),
        "novel_views_dir": str(novel_views_dir),
        "diva_dir": str(args.diva_dir),
        "ref_cam": args.ref_cam,
        "lpips_net": args.lpips_net,
        "num_pairs_compared": len(rows),
        "num_pairs_missing_gt": len(missing_gt),
        "missing_gt": missing_gt,
        "reference_camera_summary": ref_summary,
        "novel_camera_summary": novel_summary,
        "per_camera_summary": cam_summary,
        "caveat": (
            "Reference-camera metrics use the exact trained camera pose and "
            "measure genuine reconstruction fidelity. Novel-camera metrics use "
            "an approximate pose (transplanted rig rotation, reference "
            "intrinsics, approximate radius) and conflate reconstruction "
            "error with viewpoint/FOV mismatch -- see the module docstring."
        ),
    }
    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("GT comparison complete")
    print(f"Pairs compared : {len(rows)}")
    print(f"Missing GT     : {len(missing_gt)}")
    if ref_summary is not None:
        print(
            f"Reference ({args.ref_cam}, in-distribution): "
            f"PSNR={ref_summary['mean_psnr']:.2f} SSIM={ref_summary['mean_ssim']:.3f} "
            f"LPIPS={ref_summary['mean_lpips']:.3f}"
        )
    for summary in novel_summary:
        print(
            f"Novel {summary['cam']} (~{summary['angle_deg']:.1f} deg, approximate pose): "
            f"PSNR={summary['mean_psnr']:.2f} SSIM={summary['mean_ssim']:.3f} "
            f"LPIPS={summary['mean_lpips']:.3f}"
        )
    print(f"Output         : {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hand-crop PSNR/SSIM/LPIPS comparison between two (or more) checkpoints.

Motivation
----------
Whole-frame metrics on camera1/camera2 are dominated by background pixels and
by the pose-approximation error described in render_output_novelview.py's
docstring, so a genuine improvement confined to a small hand blob barely
moves the global number. This script crops every evaluated frame down to the
region that the hand Gaussian clusters actually paint, using the *union* of
each checkpoint's hand coverage so that a model which smears the hand outside
the box is not let off the hook, and scores PSNR/SSIM/LPIPS only inside that
box.

Everything here is assembled from existing building blocks -- no pose
approximation, rendering, dataset loading, or metric math is reimplemented:
  - camera0/1/2 pose handling, GT loading, dataset loading: imported from
    render_output_novelview.py, gt_analyze.py, and contactpatch_render.py
    (which already factors the camera1/2 pose-transplant math into
    `transplanted_novel_camera`).
  - PSNR/SSIM/LPIPS: flow3d/metrics.py, same classes gt_analyze.py uses.
  - hand-only rendering: uses SceneModel.render's existing `filter_mask`
    argument (it already accepts a per-Gaussian keep-mask) rather than
    mutating a copy of the model's opacities -- this reads model state only
    and can never corrupt it, no try/finally restore needed.

IMPORTANT CAVEAT (inherited from render_output_novelview.py / gt_analyze.py)
------------------------------------------------------------------------
camera1/camera2 use an *approximate* transplanted pose (see those modules'
docstrings). Cropping to the hand region removes background dominance but
does not remove this pose-approximation error. camera0 is the only view
rendered at the exact trained pose, so treat its numbers as the trustworthy
signal and camera1/camera2 as a rough, pessimistically-biased indicator.

Example
-------
    python flow3d/analysis/2drender/gt_analyze_handcrop.py \\
        --work-dir outputs/davis/spin/2026_08_03_08_41_11__spin_run1 \\
        --ckpts relative:outputs/.../relative/checkpoints/last.ckpt \\
                linattn:outputs/.../linear_attention/checkpoints/last.ckpt \\
        --frames 90,130,140 --views camera0,camera1,camera2
    # -> prints candidate hand-cluster sizes and exits (no --hand-clusters yet)

    python flow3d/analysis/2drender/gt_analyze_handcrop.py \\
        --work-dir outputs/davis/spin/2026_08_03_08_41_11__spin_run1 \\
        --ckpts relative:...last.ckpt linattn:...last.ckpt \\
        --hand-clusters 5,12 --frames 90,130,140
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from flow3d.metrics import compute_psnr, mLPIPS, mSSIM
from flow3d.renderer import Renderer

# GT loading / dataset loading -- reused verbatim, not reimplemented.
from gt_analyze import (
    DYCHECK_SUBSAMPLE_INTERVAL,
    REAL_DYCHECK_VIEWS,
    find_gt_path,
    load_dataset,
    load_image_float,
)

# camera1/camera2 pose-approximation geometry -- reused verbatim.
from render_output_novelview import (
    build_candidate_views,
    load_dycheck_ref_cam,
    parse_frames,
)

# The already-factored version of render_output_novelview.py's per-frame
# novel-camera math (see its module docstring), plus a render-dict key
# sniffer that already falls back across alpha/acc/accumulation/... names.
from contactpatch_render import extract_rgb_alpha_depth, transplanted_novel_camera

ALL_SUPPORTED_VIEWS = ["camera0", "camera1", "camera2"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", type=Path, required=True)
    parser.add_argument(
        "--ckpts", type=str, nargs="+", required=True,
        help='Two or more "label:path" pairs, e.g. relative:.../last.ckpt linattn:.../last.ckpt',
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/spin.yaml"))
    parser.add_argument("--seq-name", "--seq_name", dest="seq_name", type=str, default="spin")
    parser.add_argument(
        "--dycheck-dir", "--dycheck_dir", dest="dycheck_dir", type=Path,
        default=Path("data/DyCheck/spin"),
    )
    parser.add_argument("--ref-view", "--ref_view", dest="ref_view", type=str, default="camera0")
    parser.add_argument(
        "--hand-clusters", "--hand_clusters", dest="hand_clusters", type=str, default=None,
        help="Comma-separated cluster ids to treat as the hand. Omit to print candidate "
             "cluster sizes per checkpoint and exit.",
    )
    parser.add_argument("--frames", type=str, default="90,130,140")
    parser.add_argument("--views", type=str, default="camera0,camera1,camera2")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path, default=None)
    parser.add_argument("--bbox-pad", "--bbox_pad", dest="bbox_pad", type=int, default=12)
    parser.add_argument("--min-crop", "--min_crop", dest="min_crop", type=int, default=48)
    parser.add_argument("--lpips-resize", "--lpips_resize", dest="lpips_resize", type=int, default=128)
    parser.add_argument(
        "--coverage-threshold", "--coverage_threshold", dest="coverage_threshold",
        type=float, default=0.1,
    )
    parser.add_argument("--lpips-net", "--lpips_net", dest="lpips_net", type=str, default="alex",
                         choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def parse_ckpts(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f'--ckpts entry {spec!r} must be "label:path"')
        label, path_str = spec.split(":", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not label:
            raise ValueError(f'--ckpts entry {spec!r} has an empty label')
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for label {label!r}: {path}")
        out.append((label, path))
    if len(out) < 2:
        raise ValueError("--ckpts needs at least 2 entries to compare.")
    labels = [label for label, _ in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate --ckpts labels: {labels}")
    return out


def parse_views_arg(text: str, ref_view: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    valid = {ref_view, *REAL_DYCHECK_VIEWS}
    invalid = [n for n in names if n not in valid]
    if invalid:
        raise ValueError(f"Unknown view(s) {invalid}; this script only supports {sorted(valid)}")
    if not names:
        raise ValueError("No view was selected.")
    return names


def parse_cluster_ids(text: str) -> list[int]:
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ids:
        raise ValueError("--hand-clusters was given but parsed to an empty list.")
    return sorted(set(ids))


def load_model(ckpt_path: Path, device: torch.device, work_dir: Path):
    renderer = Renderer.init_from_checkpoint(str(ckpt_path), device, work_dir=str(work_dir), port=None)
    model = renderer.model
    model.eval()
    return model


def print_cluster_size_table(label: str, model) -> None:
    cluster_ids = model.fg.get_cluster_ids()
    if cluster_ids is None:
        print(f"[{label}] model.fg has no cluster_ids buffer -- cannot list hand-cluster candidates.")
        return
    unique_ids, counts = torch.unique(cluster_ids, return_counts=True)
    order = torch.argsort(counts, descending=True)
    print(f"\nCandidate hand clusters for ckpt {label!r} ({int(cluster_ids.shape[0])} fg Gaussians total):")
    print(f"{'cluster_id':>10}  {'num_points':>10}")
    for idx in order.tolist():
        print(f"{int(unique_ids[idx]):>10}  {int(counts[idx]):>10}")


def build_hand_filter_mask(model, hand_cluster_ids: list[int], device: torch.device) -> torch.Tensor:
    cluster_ids = model.fg.get_cluster_ids()
    if cluster_ids is None:
        raise RuntimeError("model.fg has no cluster_ids buffer; cannot select hand clusters.")
    num_fg = model.num_fg_gaussians
    assert cluster_ids.shape[0] == num_fg
    fg_keep = torch.zeros(num_fg, dtype=torch.bool, device=cluster_ids.device)
    for cid in hand_cluster_ids:
        fg_keep |= (cluster_ids == cid)
    if not bool(fg_keep.any()):
        raise RuntimeError(f"--hand-clusters {hand_cluster_ids} matched 0 Gaussians.")
    # SceneModel.render(fg_only=False) rasterizes [foreground, background,
    # shadow] in that order (see scene_model.py's get_*_all()/compute_poses_all),
    # so a full-length filter_mask must place fg_keep first and leave every
    # background/shadow Gaussian False (excluded from the hand-only render).
    mask = torch.zeros(model.num_gaussians, dtype=torch.bool, device=cluster_ids.device)
    mask[:num_fg] = fg_keep
    return mask.to(device)


def get_view_pose(
    view: str,
    ref_view: str,
    frame_idx: int,
    dataset,
    model,
    device: torch.device,
    relative_rotations: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """(w2c, K, img_wh) for one view/frame, using this model's own foreground
    pose at this frame (needed for the camera1/2 orbit pivot)."""
    img = dataset.get_image(frame_idx)
    H, W = img.shape[:2]
    img_wh = (W, H)
    K = dataset.Ks[frame_idx].to(device)

    if view == ref_view:
        w2c = dataset.w2cs[frame_idx].to(device)
        return w2c, K, img_wh

    assert view in REAL_DYCHECK_VIEWS
    reference_w2c = dataset.w2cs[frame_idx].to(device)
    fg_means = model.compute_poses_fg(torch.tensor([frame_idx], device=device))[0]
    pivot = fg_means[:, 0, :].mean(dim=0)
    w2c = transplanted_novel_camera(
        reference_w2c=reference_w2c,
        target=pivot,
        r_rel=relative_rotations[view],
        radius_scale=1.0,
    )
    return w2c, K, img_wh


def render_rgb_and_hand_coverage(
    model, frame_idx: int, w2c: torch.Tensor, K: torch.Tensor, img_wh: tuple[int, int],
    hand_filter_mask: torch.Tensor, log_keys: bool,
) -> tuple[np.ndarray, np.ndarray]:
    full_out = model.render(frame_idx, w2c[None], K[None], img_wh, use_learned_poses=False)
    if log_keys:
        print(f"[render keys] full render dict keys: {sorted(full_out.keys())}")
    rgb, _, _ = extract_rgb_alpha_depth(full_out)

    hand_out = model.render(
        frame_idx, w2c[None], K[None], img_wh,
        filter_mask=hand_filter_mask, use_learned_poses=False,
    )
    if log_keys:
        print(f"[render keys] hand-only render dict keys: {sorted(hand_out.keys())}")
    _, alpha, _ = extract_rgb_alpha_depth(hand_out)
    if alpha is None:
        raise RuntimeError(
            "Could not find an alpha/accumulation channel in the hand-only render "
            f"(keys={sorted(hand_out.keys())}); extract_rgb_alpha_depth's fallback list "
            "was exhausted. SceneModel.render always returns 'acc', so this should not happen."
        )
    if log_keys:
        print("[coverage mask] hand coverage taken from extract_rgb_alpha_depth's alpha/acc channel")
    return rgb, alpha


def coverage_union_bbox(
    masks: list[np.ndarray], threshold: float, pad: int, min_crop: int,
) -> tuple[int, int, int, int] | None:
    H, W = masks[0].shape
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        union |= (m > threshold)
    ys, xs = np.nonzero(union)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    x0, x1 = x0 - pad, x1 + pad
    y0, y1 = y0 - pad, y1 + pad

    def expand(lo: int, hi: int, min_size: int) -> tuple[int, int]:
        if hi - lo < min_size:
            center = (lo + hi) / 2.0
            lo = int(round(center - min_size / 2.0))
            hi = lo + min_size
        return lo, hi

    x0, x1 = expand(x0, x1, min_crop)
    y0, y1 = expand(y0, y1, min_crop)

    def clip(lo: int, hi: int, size: int) -> tuple[int, int]:
        if lo < 0:
            hi -= lo
            lo = 0
        if hi > size:
            lo -= hi - size
            hi = size
        lo = max(lo, 0)
        hi = min(hi, size)
        return lo, hi

    x0, x1 = clip(x0, x1, W)
    y0, y1 = clip(y0, y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def load_gt_for_view(
    view: str, ref_view: str, frame_idx: int, dataset, dycheck_dir: Path, render_shape: tuple[int, int, int],
) -> np.ndarray | None:
    frame_name = dataset.frame_names[frame_idx]
    ref_img_dir = Path(dataset.img_dir)
    if view == ref_view:
        gt_path = next(
            (ref_img_dir / f"{frame_name}{ext}" for ext in (".jpg", ".jpeg", ".png")
             if (ref_img_dir / f"{frame_name}{ext}").is_file()),
            None,
        )
    else:
        gt_path = find_gt_path(view, frame_idx, ref_view, ref_img_dir, dycheck_dir)
    if gt_path is None:
        return None
    gt = load_image_float(gt_path)
    if gt.shape != render_shape:
        gt_img = Image.fromarray((gt * 255).astype(np.uint8)).resize(
            (render_shape[1], render_shape[0]), Image.BILINEAR
        )
        gt = np.asarray(gt_img, dtype=np.float32) / 255.0
    return gt


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def save_handcrop_comparison(
    path: Path, view: str, frame_name: str, rows: list[dict[str, Any]],
) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4.2 * n), squeeze=False)
    for i, row in enumerate(rows):
        render, gt = row["crop_render"], row["crop_gt"]
        diff = np.abs(render - gt).mean(axis=-1)
        diff_vis = plt.get_cmap("inferno")(np.clip(diff * 3.0, 0.0, 1.0))[..., :3]
        axes[i][0].imshow(render)
        axes[i][0].set_title(f"[{row['label']}] render")
        axes[i][1].imshow(gt)
        axes[i][1].set_title("GT")
        axes[i][2].imshow(diff_vis)
        axes[i][2].set_title(
            f"abs diff (x3)  PSNR={row['psnr']:.2f} SSIM={row['ssim']:.3f} LPIPS={row['lpips']:.3f}"
        )
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(f"{view} frame {frame_name} -- hand crop, same box across all checkpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else work_dir / "analysis" / "handcrop"
    comparisons_dir = out_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    if not args.dycheck_dir.is_dir():
        raise FileNotFoundError(f"DyCheck sequence dir not found: {args.dycheck_dir}")

    ckpt_specs = parse_ckpts(args.ckpts)
    views = parse_views_arg(args.views, args.ref_view)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = load_dataset(args.config, args.seq_name)
    frames = parse_frames(args.frames, dataset.num_frames)

    with torch.no_grad():
        models = {label: load_model(path, device, out_dir) for label, path in ckpt_specs}

        if args.hand_clusters is None:
            for label, model in models.items():
                print_cluster_size_table(label, model)
            print(
                "\nNo --hand-clusters given -- pick ids from the table(s) above and rerun with "
                "--hand-clusters id1,id2,..."
            )
            return

        hand_cluster_ids = parse_cluster_ids(args.hand_clusters)
        print(f"Using hand clusters: {hand_cluster_ids}")
        for label, model in models.items():
            print_cluster_size_table(label, model)
            available = set(torch.unique(model.fg.get_cluster_ids()).tolist())
            missing = [c for c in hand_cluster_ids if c not in available]
            if missing:
                print(f"[Warning] ckpt {label!r} has no Gaussians for cluster id(s) {missing}.")

        hand_filter_masks = {
            label: build_hand_filter_mask(model, hand_cluster_ids, device)
            for label, model in models.items()
        }

        real_views_needed = [v for v in views if v in REAL_DYCHECK_VIEWS]
        candidate_views = build_candidate_views(args.dycheck_dir, real_views_needed) if real_views_needed else {}
        relative_rotations = {}
        if real_views_needed:
            _, r_ref_dycheck = load_dycheck_ref_cam(args.dycheck_dir, 0)
            relative_rotations = {
                name: r_ref_dycheck.T @ v["r_c2w"] for name, v in candidate_views.items()
            }

        ssim_metric = mSSIM().to(device)
        lpips_metric = mLPIPS(net_type=args.lpips_net).to(device)

        labels = [label for label, _ in ckpt_specs]
        baseline_label = labels[0]

        metric_rows: list[dict[str, Any]] = []
        logged_keys_once = False

        for frame_idx in frames:
            frame_name = dataset.frame_names[frame_idx]
            for view in views:
                per_ckpt_render: dict[str, np.ndarray] = {}
                per_ckpt_coverage: dict[str, np.ndarray] = {}
                per_ckpt_pose: dict[str, tuple[torch.Tensor, torch.Tensor, tuple[int, int]]] = {}

                for label, model in models.items():
                    w2c, K, img_wh = get_view_pose(
                        view, args.ref_view, frame_idx, dataset, model, device, relative_rotations
                    )
                    per_ckpt_pose[label] = (w2c, K, img_wh)
                    rgb, coverage = render_rgb_and_hand_coverage(
                        model, frame_idx, w2c, K, img_wh,
                        hand_filter_masks[label], log_keys=not logged_keys_once,
                    )
                    logged_keys_once = True
                    per_ckpt_render[label] = rgb
                    per_ckpt_coverage[label] = coverage

                bbox = coverage_union_bbox(
                    list(per_ckpt_coverage.values()), args.coverage_threshold, args.bbox_pad, args.min_crop
                )
                if bbox is None:
                    print(f"[Skip] {view} frame={frame_name}: no hand coverage above threshold for any ckpt.")
                    continue
                x0, y0, x1, y1 = bbox
                print(
                    f"[bbox] {view} frame={frame_name}: box=(x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}) "
                    f"-- identical box used for all {len(labels)} checkpoint(s)"
                )

                render_shape = per_ckpt_render[baseline_label].shape
                gt_full = load_gt_for_view(view, args.ref_view, frame_idx, dataset, args.dycheck_dir, render_shape)
                if gt_full is None:
                    print(f"[Skip] {view} frame={frame_name}: no GT image found.")
                    continue

                comparison_rows = []
                for label in labels:
                    render_full = per_ckpt_render[label]
                    crop_render = render_full[y0:y1, x0:x1]
                    crop_gt = gt_full[y0:y1, x0:x1]

                    render_t = torch.from_numpy(crop_render).to(device)[None]
                    gt_t = torch.from_numpy(crop_gt).to(device)[None]

                    psnr = compute_psnr(render_t, gt_t)
                    ssim_metric.reset()
                    ssim_metric.update(render_t, gt_t)
                    ssim = float(ssim_metric.compute().item())

                    render_lp = resize_square(crop_render, args.lpips_resize)
                    gt_lp = resize_square(crop_gt, args.lpips_resize)
                    render_lp_t = torch.from_numpy(render_lp).to(device)[None]
                    gt_lp_t = torch.from_numpy(gt_lp).to(device)[None]
                    lpips_metric.reset()
                    lpips_metric.update(render_lp_t, gt_lp_t)
                    lpips = float(lpips_metric.compute().item())

                    row = {
                        "frame": frame_idx,
                        "frame_name": frame_name,
                        "view": view,
                        "ckpt_label": label,
                        "bbox_x": x0, "bbox_y": y0, "bbox_w": x1 - x0, "bbox_h": y1 - y0,
                        "crop_px": f"{x1 - x0}x{y1 - y0}",
                        "psnr": psnr, "ssim": ssim, "lpips": lpips,
                    }
                    metric_rows.append(row)
                    comparison_rows.append({
                        "label": label, "crop_render": crop_render, "crop_gt": crop_gt,
                        "psnr": psnr, "ssim": ssim, "lpips": lpips,
                    })
                    print(
                        f"  {view} frame={frame_name} ckpt={label} crop={row['crop_px']} "
                        f"psnr={psnr:.2f} ssim={ssim:.3f} lpips={lpips:.3f}"
                    )

                labels_joined = "_vs_".join(labels)
                compare_path = comparisons_dir / f"{view}_frame{frame_idx:04d}_{labels_joined}_crop.png"
                save_handcrop_comparison(compare_path, view, frame_name, comparison_rows)

    if not metric_rows:
        raise RuntimeError("No (frame, view) pair produced a scored crop -- see [Skip] warnings above.")

    with (out_dir / "handcrop_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    by_key: dict[tuple[int, str, str], dict[str, Any]] = {
        (r["frame"], r["view"], r["ckpt_label"]): r for r in metric_rows
    }
    pair_labels = labels[1:]
    delta_rows: list[dict[str, Any]] = []
    for other_label in pair_labels:
        pair_deltas = []
        for frame_idx in frames:
            for view in views:
                base_row = by_key.get((frame_idx, view, baseline_label))
                other_row = by_key.get((frame_idx, view, other_label))
                if base_row is None or other_row is None:
                    continue
                delta = {
                    "frame": frame_idx,
                    "view": view,
                    "baseline": baseline_label,
                    "compared": other_label,
                    "delta_psnr": other_row["psnr"] - base_row["psnr"],
                    "delta_ssim": other_row["ssim"] - base_row["ssim"],
                    "delta_lpips": other_row["lpips"] - base_row["lpips"],
                }
                delta_rows.append(delta)
                pair_deltas.append(delta)
        if pair_deltas:
            delta_rows.append({
                "frame": "ALL", "view": "ALL",
                "baseline": baseline_label, "compared": other_label,
                "delta_psnr": float(np.mean([d["delta_psnr"] for d in pair_deltas])),
                "delta_ssim": float(np.mean([d["delta_ssim"] for d in pair_deltas])),
                "delta_lpips": float(np.mean([d["delta_lpips"] for d in pair_deltas])),
            })

    with (out_dir / "handcrop_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_rows)

    print("\n" + "=" * 78)
    print("Hand-crop delta summary (compared - baseline; PSNR/SSIM higher=better, LPIPS lower=better)")
    print(f"Baseline: {baseline_label!r}")
    print("=" * 78)
    for other_label in pair_labels:
        summary = next(
            (d for d in delta_rows if d["frame"] == "ALL" and d["compared"] == other_label), None
        )
        if summary is None:
            continue
        print(
            f"{other_label!r} vs {baseline_label!r} (mean over {len(views)} view(s) x {len(frames)} frame(s)): "
            f"dPSNR={summary['delta_psnr']:+.3f}dB  dSSIM={summary['delta_ssim']:+.4f}  "
            f"dLPIPS={summary['delta_lpips']:+.4f}"
        )
        cam0_summary = [
            d for d in delta_rows if d["frame"] != "ALL" and d["compared"] == other_label and d["view"] == args.ref_view
        ]
        if cam0_summary:
            print(
                f"  -> {args.ref_view} only (exact trained pose, most trustworthy): "
                f"dPSNR={np.mean([d['delta_psnr'] for d in cam0_summary]):+.3f}dB  "
                f"dSSIM={np.mean([d['delta_ssim'] for d in cam0_summary]):+.4f}  "
                f"dLPIPS={np.mean([d['delta_lpips'] for d in cam0_summary]):+.4f}"
            )
    print(f"\nOutput: {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()

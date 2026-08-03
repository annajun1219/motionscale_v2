#!/usr/bin/env python3
"""
Occlusion-aware novel-view contact-patch distance and frame classification.

For each selected cluster pair, frame, and novel camera:
1. Transform canonical contact-patch Gaussian identities to their current 3D positions.
2. Build a novel camera by transplanting a nearby DiVa-360 rig camera's
   orientation (relative to the training camera) onto the trained scene,
   orbiting around the target at the reference camera's own distance. DiVa-360
   poses live in a different coordinate frame/scale/axis-convention than the
   trained (mega-sam) scene, so only the *relative* rotation to the reference
   camera is transplanted -- see flow3d/analysis/2drender/render_output_novelview.py
   for the same technique and a longer explanation.
3. Render the full scene, then re-render it with patch A removed and with
   patch B removed.
4. Use the full-scene image difference to estimate which projected patch
   Gaussians actually contribute to the final rendered image. This respects
   occlusion by the opposite patch, other clusters, and the background.
5. Mark a view valid only when both patches have sufficient visible support.
6. Recompute bidirectional 2D nearest-neighbour distances only in valid views.
7. Aggregate valid views per frame into connected, separated, or
   unknown_due_to_occlusion.

Default requested views follow the user's spec:
- frames: 0,20,40,60,80,99
- novel cameras (DiVa-360 rig, closest to cam00): cam28,cam39,cam06,cam07,cam43,cam32
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
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from flow3d.renderer import Renderer

# Fixed local-axis flip between the OpenGL/NeRF convention (DiVa-360
# transforms.json: camera looks down -Z, Y-up) and the OpenCV convention
# (mega-sam / the trained scene: camera looks down +Z, Y-down).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def load_diva_cam_poses(diva_dir: Path) -> dict[str, np.ndarray]:
    """Load DiVa-360 camera-to-world matrices (OpenGL/NeRF axis convention)."""
    cams: dict[str, np.ndarray] = {}
    for name in ("transforms_train.json", "transforms_test.json", "transforms_val.json"):
        path = diva_dir / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for frame in data.get("frames", []):
            parts = frame["file_path"].split("/")
            if len(parts) < 2:
                continue
            cam = parts[1]
            cams.setdefault(cam, np.array(frame["transform_matrix"], dtype=np.float64))
    return cams


def diva_cam_angle_deg(cams: dict[str, np.ndarray], ref_cam: str, cam_name: str) -> float:
    """Angular distance of cam_name from ref_cam, about the DiVa-360 rig center."""
    names = list(cams.keys())
    positions = np.stack([cams[n][:3, 3] for n in names])
    center = positions.mean(axis=0)
    dirs = positions - center
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    ref_dir = dirs[names.index(ref_cam)]
    cam_dir = dirs[names.index(cam_name)]
    return float(np.degrees(np.arccos(np.clip(np.dot(ref_dir, cam_dir), -1.0, 1.0))))


def relative_rotation_cv(r_ref_gl: np.ndarray, r_cam_gl: np.ndarray) -> np.ndarray:
    """Relative rotation ref->cam (DiVa-360, OpenGL axes) re-expressed in OpenCV axes."""
    rel_gl = r_ref_gl.T @ r_cam_gl
    return _GL_TO_CV @ rel_gl @ _GL_TO_CV


EPS = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Novel-view rendering and 2D visible-patch analysis for contact patches."
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
        help="Default: <work-dir>/analysis/contact_patches/contact_patches.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <work-dir>/analysis/2drender/contactpatch_render",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="all",
        help='Use "all" or a subset such as "41-42,14-22".',
    )
    parser.add_argument(
        "--exclude-pairs",
        type=str,
        default="41-45,36-39",
        help=(
            'Pairs excluded from analysis. Default: "41-45,36-39". '
            'Pass an empty string only after changing the code or use --pairs '
            'to select an explicit subset.'
        ),
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0,20,40,60,80,99",
        help='Comma list or "all". Default matches the requested frames.',
    )
    parser.add_argument(
        "--diva-dir",
        type=Path,
        default=Path("data/DiVa360/processed_data/dog"),
        help="DiVa-360 sequence dir containing transforms_{train,test,val}.json.",
    )
    parser.add_argument(
        "--ref-cam",
        type=str,
        default="cam00",
        help="DiVa-360 camera used for training (the reference orientation).",
    )
    parser.add_argument(
        "--novel-cams",
        type=str,
        default="cam28,cam39,cam06,cam07,cam43,cam32",
        help=(
            "Comma list of DiVa-360 camera names to render novel views from. "
            "Each camera's orientation relative to --ref-cam is transplanted "
            "onto the trained scene; see the module docstring."
        ),
    )
    parser.add_argument(
        "--reference-camera",
        type=str,
        default="per-frame",
        choices=["per-frame", "first"],
        help="Use each frame camera or always the first frame camera.",
    )
    parser.add_argument(
        "--orbit-center",
        type=str,
        default="patch-center",
        choices=["patch-center", "scene-center"],
        help="Center the orbit around the current patch center or the scene center.",
    )
    parser.add_argument(
        "--orbit-radius-scale",
        type=float,
        default=1.0,
        help="Multiply reference camera radius relative to target center.",
    )
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument(
        "--contribution-threshold",
        type=float,
        default=0.01,
        help=(
            "A projected patch Gaussian is visible when removing its entire patch "
            "changes the final full-scene render by at least this amount at the "
            "Gaussian center. The contribution map is max(RGB-L1, positive alpha difference)."
        ),
    )
    parser.add_argument(
        "--min-visible-gaussians",
        type=int,
        default=8,
        help="Minimum visible projected Gaussian count required for each patch.",
    )
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.02,
        help=(
            "Minimum visible fraction of in-frame projected Gaussians required "
            "for each patch."
        ),
    )
    parser.add_argument(
        "--contact-threshold-px",
        type=float,
        default=4.0,
        help=(
            "A visible projected point counts as touching when its nearest "
            "opposite-patch point is within this many pixels."
        ),
    )
    parser.add_argument(
        "--separation-gap-threshold-px",
        type=float,
        default=4.0,
        help="A valid view is a separation candidate when its median 2D gap exceeds this.",
    )
    parser.add_argument(
        "--separation-contact-ratio-threshold",
        type=float,
        default=0.30,
        help=(
            "A valid view is separated only when its joint contact ratio is below "
            "this value as well as exceeding the gap threshold."
        ),
    )
    parser.add_argument(
        "--connected-gap-threshold-px",
        type=float,
        default=4.0,
        help=(
            "A valid view is connected only when its median 2D gap is at or below "
            "this threshold."
        ),
    )
    parser.add_argument(
        "--connected-contact-ratio-threshold",
        type=float,
        default=0.50,
        help=(
            "A valid view is connected only when its joint contact ratio is at or "
            "above this threshold."
        ),
    )
    parser.add_argument(
        "--pair-separation-iqr-multiplier",
        type=float,
        default=1.5,
        help=(
            "A cluster pair is treated as globally separated when its pair-level "
            "median valid-view gap is above Q3 + multiplier * IQR across pairs."
        ),
    )
    parser.add_argument(
        "--pair-separation-min-gap-px",
        type=float,
        default=8.0,
        help=(
            "Minimum pair-level median valid-view gap required before a pair can "
            "be classified as globally separated."
        ),
    )
    parser.add_argument(
        "--temporal-reference-frame-count",
        type=int,
        default=2,
        help=(
            "Number of earliest valid frame-level gap values used to compute "
            "the temporal reference gap."
        ),
    )
    parser.add_argument(
        "--temporal-gap-ratio-threshold",
        type=float,
        default=1.8,
        help=(
            "A frame is temporally separated when its median valid-view gap is "
            "greater than this multiple of the temporal reference gap."
        ),
    )
    parser.add_argument(
        "--temporal-persistent-min-frames",
        type=int,
        default=3,
        help=(
            "Minimum number of consecutive temporally separated valid frames "
            "required for persistent separation."
        ),
    )
    parser.add_argument(
        "--temporal-jump-threshold-px",
        type=float,
        default=4.0,
        help=(
            "Minimum absolute change between consecutive valid frame-level gaps "
            "counted as a temporal jump."
        ),
    )
    parser.add_argument(
        "--temporal-unstable-min-jumps",
        type=int,
        default=2,
        help=(
            "Minimum number of large consecutive-frame gap jumps required to "
            "classify a pair as temporally unstable."
        ),
    )
    parser.add_argument(
        "--min-valid-views",
        type=int,
        default=3,
        help=(
            "A frame is unknown_due_to_occlusion when fewer than this many views "
            "show both patches sufficiently."
        ),
    )
    parser.add_argument(
        "--separation-view-fraction",
        type=float,
        default=0.50,
        help=(
            "Among valid views, at least this fraction must be separated for the "
            "frame to be classified as separated."
        ),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Do not save overlay/alpha images.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.orbit_radius_scale <= 0:
        raise ValueError("--orbit-radius-scale must be > 0")
    if args.contribution_threshold < 0:
        raise ValueError("--contribution-threshold must be >= 0")
    if args.min_visible_gaussians < 1:
        raise ValueError("--min-visible-gaussians must be >= 1")
    if not (0.0 <= args.min_visible_fraction <= 1.0):
        raise ValueError("--min-visible-fraction must be in [0, 1]")
    if args.contact_threshold_px <= 0:
        raise ValueError("--contact-threshold-px must be > 0")
    if args.separation_gap_threshold_px <= 0:
        raise ValueError("--separation-gap-threshold-px must be > 0")
    if not (0.0 <= args.separation_contact_ratio_threshold <= 1.0):
        raise ValueError("--separation-contact-ratio-threshold must be in [0, 1]")
    if args.connected_gap_threshold_px <= 0:
        raise ValueError("--connected-gap-threshold-px must be > 0")
    if not (0.0 <= args.connected_contact_ratio_threshold <= 1.0):
        raise ValueError("--connected-contact-ratio-threshold must be in [0, 1]")
    if args.pair_separation_iqr_multiplier < 0:
        raise ValueError("--pair-separation-iqr-multiplier must be >= 0")
    if args.pair_separation_min_gap_px <= 0:
        raise ValueError("--pair-separation-min-gap-px must be > 0")
    if args.temporal_reference_frame_count < 1:
        raise ValueError("--temporal-reference-frame-count must be >= 1")
    if args.temporal_gap_ratio_threshold <= 1.0:
        raise ValueError("--temporal-gap-ratio-threshold must be > 1")
    if args.temporal_persistent_min_frames < 1:
        raise ValueError("--temporal-persistent-min-frames must be >= 1")
    if args.temporal_jump_threshold_px <= 0:
        raise ValueError("--temporal-jump-threshold-px must be > 0")
    if args.temporal_unstable_min_jumps < 1:
        raise ValueError("--temporal-unstable-min-jumps must be >= 1")
    if args.min_valid_views < 1:
        raise ValueError("--min-valid-views must be >= 1")
    if not (0.0 <= args.separation_view_fraction <= 1.0):
        raise ValueError("--separation-view-fraction must be in [0, 1]")


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_pair_subset(text: str | None) -> set[tuple[int, int]] | None:
    if text is None:
        return set()
    if text.strip().lower() == "all":
        return None
    result: set[tuple[int, int]] = set()
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
        result.add((min(a, b), max(a, b)))
    if not result:
        raise ValueError("No valid pair was supplied.")
    return result


def parse_frames(text: str, total_frames: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total_frames))
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("No frame was selected.")
    invalid = [x for x in values if x < 0 or x >= total_frames]
    if invalid:
        raise ValueError(f"Invalid frames {invalid}; total frame count is {total_frames}.")
    return values


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


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
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
            item for item in value
            if isinstance(item, torch.Tensor)
            and item.ndim == 4
            and item.shape[-2:] == (3, 4)
        ]
        if candidates:
            value = candidates[0]

    if not isinstance(value, torch.Tensor):
        raise TypeError("Could not extract a Gaussian transform tensor.")
    if value.ndim != 4 or value.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected transform shape: {tuple(value.shape)}")

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
                value=raw,
                num_gaussians=int(canonical_means.shape[0]),
                num_frames=int(frame_ids.numel()),
            )
        except TypeError:
            try:
                transforms = normalize_transform_tensor(
                    value=raw,
                    num_gaussians=int(canonical_means.shape[0]),
                    num_frames=int(frame_ids.numel()),
                )
            except Exception as exc:
                errors.append(f"{method_name}: {type(exc).__name__}: {exc}")
                continue
        except Exception as exc:
            errors.append(f"{method_name}: {type(exc).__name__}: {exc}")
            continue

        positions = torch.einsum(
            "gtij,gj->gti",
            transforms,
            to_homogeneous(canonical_means),
        )
        return positions, f"model.{method_name}(frame_ids)"

    raise RuntimeError(
        "Could not compute frame-specific Gaussian positions:\n" + "\n".join(errors)
    )


def get_reference_w2cs(model: Any) -> tuple[torch.Tensor, str]:
    camera_poses = getattr(model, "camera_poses", None)
    get_camera_matrix = (
        getattr(camera_poses, "get_camera_matrix", None)
        if camera_poses is not None
        else None
    )
    if callable(get_camera_matrix):
        w2cs = get_camera_matrix()
        if not isinstance(w2cs, torch.Tensor):
            raise TypeError("camera_poses.get_camera_matrix() did not return a tensor.")
        return w2cs, "model.camera_poses.get_camera_matrix()"

    w2cs = getattr(model, "w2cs", None)
    if not isinstance(w2cs, torch.Tensor):
        raise TypeError("Could not obtain camera matrices from model.")
    return w2cs, "model.w2cs"


def infer_image_size(model: Any) -> tuple[int, int]:
    for name in ("img_wh", "image_wh", "resolution"):
        value = getattr(model, name, None)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return int(value[0]), int(value[1])
    width = getattr(model, "image_width", None)
    height = getattr(model, "image_height", None)
    if width is not None and height is not None:
        return int(width), int(height)
    return 854, 480


def scale_intrinsics(
    K: torch.Tensor,
    base_width: int,
    base_height: int,
    width: int,
    height: int,
) -> torch.Tensor:
    K_view = K.clone()
    K_view[0, :] *= float(width) / float(base_width)
    K_view[1, :] *= float(height) / float(base_height)
    return K_view


def camera_center_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return -(rotation.transpose(0, 1) @ translation)


def diva_novel_camera(
    reference_w2c: torch.Tensor,
    target: torch.Tensor,
    r_ref_diva: np.ndarray,
    r_cam_diva: np.ndarray,
    radius_scale: float,
) -> torch.Tensor:
    """Build a novel-view w2c by transplanting a DiVa-360 camera's orientation
    relative to the reference camera onto the trained scene, orbiting around
    `target` at the reference camera's own distance (DiVa-360's absolute
    translation scale is not usable here -- see the module docstring).
    """
    center = camera_center_from_w2c(reference_w2c)
    radius = (center - target).norm().clamp_min(EPS) * radius_scale

    r_ref_train = reference_w2c[:3, :3].transpose(0, 1)  # c2w rotation
    r_rel_cv = torch.from_numpy(relative_rotation_cv(r_ref_diva, r_cam_diva)).to(
        device=r_ref_train.device, dtype=r_ref_train.dtype
    )
    r_novel = r_ref_train @ r_rel_cv  # c2w rotation, OpenCV local axes
    forward = r_novel[:, 2]
    forward = forward / forward.norm().clamp_min(EPS)
    new_center = target - radius * forward

    w2c = torch.eye(4, device=target.device, dtype=target.dtype)
    w2c[:3, :3] = r_novel.transpose(0, 1)
    w2c[:3, 3] = -(r_novel.transpose(0, 1) @ new_center)
    return w2c


def project_points(
    points_world: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_h = to_homogeneous(points_world)

    if w2c.shape[-2:] == (4, 4):
        camera = (w2c @ points_h.T).T[:, :3]
    else:
        camera = (w2c @ points_h.T).T

    depth = camera[:, 2]
    projected = (K @ camera.T).T
    uv = projected[:, :2] / projected[:, 2:3].clamp_min(EPS)

    valid = (
        (depth > EPS)
        & torch.isfinite(uv).all(dim=-1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return (
        uv[valid].detach().cpu().numpy(),
        depth[valid].detach().cpu().numpy(),
        torch.where(valid)[0].detach().cpu().numpy(),
    )


def extract_rgb_alpha_depth(
    render_result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if "img" not in render_result:
        raise KeyError(f"Render result has no 'img'. Keys: {list(render_result.keys())}")

    rgb = render_result["img"][0].detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    alpha = None
    for key in ("alpha", "acc", "accumulation", "opacity", "render_alpha", "mask"):
        value = render_result.get(key)
        if isinstance(value, torch.Tensor):
            alpha = value[0].detach().float().cpu().numpy()
            break
    if alpha is not None:
        alpha = np.squeeze(alpha)
        if alpha.ndim == 3:
            if alpha.shape[0] == 1:
                alpha = alpha[0]
            elif alpha.shape[-1] == 1:
                alpha = alpha[..., 0]
            else:
                alpha = alpha.mean(axis=-1)
        alpha = np.clip(alpha, 0, 1).astype(np.float32)

    depth = None
    for key in ("depth", "render_depth", "disp_depth", "expected_depth", "z", "depth_map"):
        value = render_result.get(key)
        if isinstance(value, torch.Tensor):
            depth = value[0].detach().float().cpu().numpy()
            break
    if depth is not None:
        depth = np.squeeze(depth)
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                depth = depth.mean(axis=-1)
        depth = depth.astype(np.float32)

    return rgb, alpha, depth


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = xs - x0
    wy = ys - y0

    return (
        image[y0, x0] * (1 - wx) * (1 - wy)
        + image[y0, x1] * wx * (1 - wy)
        + image[y1, x0] * (1 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def filter_visible_points(
    uv: np.ndarray,
    contribution_map: np.ndarray,
    contribution_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep projected Gaussian centers whose patch contributes to the final
    full-scene image at that pixel.

    contribution_map is computed by comparing:
      full scene
      vs. full scene with the entire patch removed.

    Therefore a patch hidden by the opposite patch, another cluster, or the
    background should have near-zero contribution at the covered pixels.
    """
    if len(uv) == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)

    samples = bilinear_sample(contribution_map, uv[:, 0], uv[:, 1])
    visible = np.isfinite(samples) & (samples >= contribution_threshold)
    return visible, samples.astype(np.float32)


def bidirectional_projected_gap(uv_a: np.ndarray, uv_b: np.ndarray) -> dict[str, float]:
    if len(uv_a) == 0 or len(uv_b) == 0:
        return {
            "median_px": float("nan"),
            "p90_px": float("nan"),
            "mean_px": float("nan"),
            "a_to_b_median_px": float("nan"),
            "b_to_a_median_px": float("nan"),
            "contact_ratio_a": float("nan"),
            "contact_ratio_b": float("nan"),
            "contact_ratio_joint": float("nan"),
        }

    distances = np.linalg.norm(uv_a[:, None, :] - uv_b[None, :, :], axis=-1)
    a_to_b = distances.min(axis=1)
    b_to_a = distances.min(axis=0)
    both = np.concatenate([a_to_b, b_to_a])

    return {
        "median_px": float(np.median(both)),
        "p90_px": float(np.quantile(both, 0.9)),
        "mean_px": float(np.mean(both)),
        "a_to_b_median_px": float(np.median(a_to_b)),
        "b_to_a_median_px": float(np.median(b_to_a)),
    }


def contact_ratios(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    threshold_px: float,
) -> dict[str, float]:
    if len(uv_a) == 0 or len(uv_b) == 0:
        return {
            "contact_ratio_a": float("nan"),
            "contact_ratio_b": float("nan"),
            "contact_ratio_joint": float("nan"),
        }
    distances = np.linalg.norm(uv_a[:, None, :] - uv_b[None, :, :], axis=-1)
    a_to_b = distances.min(axis=1)
    b_to_a = distances.min(axis=0)
    ratio_a = float(np.mean(a_to_b <= threshold_px))
    ratio_b = float(np.mean(b_to_a <= threshold_px))
    ratio_joint = float((ratio_a + ratio_b) * 0.5)
    return {
        "contact_ratio_a": ratio_a,
        "contact_ratio_b": ratio_b,
        "contact_ratio_joint": ratio_joint,
    }


def pair_filter_from_indices(
    total_gaussians: int,
    patch_indices: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros(total_gaussians, dtype=torch.bool, device=patch_indices.device)
    mask[patch_indices] = True
    return mask


def render_views(
    model: Any,
    frame: int,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    patch_filter_a: torch.Tensor,
    patch_filter_b: torch.Tensor,
) -> dict[str, Any]:
    """
    Estimate full-scene-visible patch contribution without requiring a depth map.

    We render:
      1. the complete scene;
      2. the complete scene with patch A removed;
      3. the complete scene with patch B removed.

    If removing a patch changes the final image at a pixel, that patch was
    contributing there after all rasterization and occlusion effects.
    """
    full_result = model.render(
        frame,
        w2c[None],
        K[None],
        (width, height),
        use_learned_poses=False,
    )
    full_rgb, full_acc, _ = extract_rgb_alpha_depth(full_result)

    without_a_result = model.render(
        frame,
        w2c[None],
        K[None],
        (width, height),
        filter_mask=~patch_filter_a,
        use_learned_poses=False,
    )
    without_a_rgb, without_a_acc, _ = extract_rgb_alpha_depth(without_a_result)

    without_b_result = model.render(
        frame,
        w2c[None],
        K[None],
        (width, height),
        filter_mask=~patch_filter_b,
        use_learned_poses=False,
    )
    without_b_rgb, without_b_acc, _ = extract_rgb_alpha_depth(without_b_result)

    rgb_contrib_a = np.mean(np.abs(full_rgb - without_a_rgb), axis=-1)
    rgb_contrib_b = np.mean(np.abs(full_rgb - without_b_rgb), axis=-1)

    if full_acc is not None and without_a_acc is not None:
        alpha_contrib_a = np.clip(full_acc - without_a_acc, 0.0, None)
        contribution_a = np.maximum(rgb_contrib_a, alpha_contrib_a)
    else:
        contribution_a = rgb_contrib_a

    if full_acc is not None and without_b_acc is not None:
        alpha_contrib_b = np.clip(full_acc - without_b_acc, 0.0, None)
        contribution_b = np.maximum(rgb_contrib_b, alpha_contrib_b)
    else:
        contribution_b = rgb_contrib_b

    return {
        "rgb": full_rgb,
        "contribution_a": contribution_a.astype(np.float32),
        "contribution_b": contribution_b.astype(np.float32),
        "full_keys": list(full_result.keys()),
        "without_a_keys": list(without_a_result.keys()),
        "without_b_keys": list(without_b_result.keys()),
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_alpha(path: Path, alpha: np.ndarray | None, title: str) -> None:
    if alpha is None:
        return
    plt.figure(figsize=(6, 4))
    plt.imshow(alpha, vmin=0, vmax=1)
    plt.colorbar(label="alpha")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def save_overlay(
    path: Path,
    rgb: np.ndarray,
    all_uv_a: np.ndarray,
    all_uv_b: np.ndarray,
    vis_uv_a: np.ndarray,
    vis_uv_b: np.ndarray,
    title: str,
) -> None:
    image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)

    # faint all projected points
    for x, y in all_uv_a:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(80, 180, 255), width=1)
    for x, y in all_uv_b:
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline=(255, 120, 120), width=1)

    # visible points highlighted
    draw_points(draw, vis_uv_a, (0, 255, 255), 4)
    draw_points(draw, vis_uv_b, (255, 0, 255), 4)

    banner_width = min(image.width - 1, 1200)
    draw.rectangle((0, 0, banner_width, 34), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    image.save(path)


def save_heatmap(
    rows: list[dict[str, Any]],
    output_path: Path,
    value_key: str,
    title: str,
    colorbar_label: str,
) -> None:
    if not rows:
        return
    pair_labels = sorted({f"{int(r['cluster_a'])}-{int(r['cluster_b'])}" for r in rows})
    view_labels = sorted({
        f"f{int(r['frame'])}_{r['novel_cam']}"
        for r in rows
    })
    pair_to_index = {label: i for i, label in enumerate(pair_labels)}
    view_to_index = {label: i for i, label in enumerate(view_labels)}

    matrix = np.full((len(pair_labels), len(view_labels)), np.nan, dtype=np.float64)
    for row in rows:
        pair = f"{int(row['cluster_a'])}-{int(row['cluster_b'])}"
        view = f"f{int(row['frame'])}_{row['novel_cam']}"
        matrix[pair_to_index[pair], view_to_index[view]] = float(row[value_key])

    plt.figure(figsize=(max(14, len(view_labels) * 0.12), max(5, len(pair_labels) * 0.22)))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=colorbar_label)
    tick_step = max(1, len(view_labels) // 20)
    tick_indices = list(range(0, len(view_labels), tick_step))
    plt.xticks(tick_indices, [view_labels[i] for i in tick_indices], rotation=45, ha="right")
    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel("View (frame / novel camera)")
    plt.ylabel("Cluster pair")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def max_consecutive_true(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def temporal_pair_metrics(
    pair_frame_rows: list[dict[str, Any]],
    absolute_gap_threshold_px: float,
    reference_frame_count: int,
    gap_ratio_threshold: float,
    persistent_min_frames: int,
    jump_threshold_px: float,
    unstable_min_jumps: int,
) -> dict[str, Any]:
    valid_series = [
        (
            int(row["frame"]),
            float(row["median_valid_view_gap_px"]),
        )
        for row in sorted(pair_frame_rows, key=lambda item: int(item["frame"]))
        if np.isfinite(float(row["median_valid_view_gap_px"]))
    ]

    if not valid_series:
        return {
            "temporal_valid_frame_count": 0,
            "temporal_reference_gap_px": float("nan"),
            "temporal_separation_threshold_px": float("nan"),
            "temporal_separated_frame_count": 0,
            "temporal_separated_frames": "",
            "temporal_max_consecutive_separated_frames": 0,
            "persistent_separation": False,
            "temporal_jump_count": 0,
            "temporal_jump_frames": "",
            "temporally_unstable": False,
            "temporal_gap_range_px": float("nan"),
        }

    reference_values = [
        gap for _, gap in valid_series[:reference_frame_count]
    ]
    reference_gap = float(np.median(reference_values))
    temporal_separation_threshold = max(
        absolute_gap_threshold_px,
        gap_ratio_threshold * reference_gap,
    )

    separated_flags = [
        gap > temporal_separation_threshold for _, gap in valid_series
    ]
    separated_frames = [
        frame
        for (frame, _), separated in zip(valid_series, separated_flags)
        if separated
    ]
    max_consecutive = max_consecutive_true(separated_flags)
    persistent_separation = max_consecutive >= persistent_min_frames

    jump_frames: list[str] = []
    jump_count = 0
    for (prev_frame, prev_gap), (frame, gap) in zip(
        valid_series[:-1], valid_series[1:]
    ):
        if abs(gap - prev_gap) > jump_threshold_px:
            jump_count += 1
            jump_frames.append(f"{prev_frame}->{frame}")

    temporally_unstable = jump_count >= unstable_min_jumps
    gaps = [gap for _, gap in valid_series]

    return {
        "temporal_valid_frame_count": len(valid_series),
        "temporal_reference_gap_px": reference_gap,
        "temporal_separation_threshold_px": temporal_separation_threshold,
        "temporal_separated_frame_count": len(separated_frames),
        "temporal_separated_frames": ",".join(map(str, separated_frames)),
        "temporal_max_consecutive_separated_frames": max_consecutive,
        "persistent_separation": bool(persistent_separation),
        "temporal_jump_count": jump_count,
        "temporal_jump_frames": ",".join(jump_frames),
        "temporally_unstable": bool(temporally_unstable),
        "temporal_gap_range_px": float(np.max(gaps) - np.min(gaps)),
    }


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
        else work_dir / "analysis" / "2drender" / "contactpatch_render"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not contact_patch_file.is_file():
        raise FileNotFoundError(f"Contact-patch file not found: {contact_patch_file}")

    overlay_dir = output_dir / "overlays"
    alpha_dir = output_dir / "contribution_maps"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

    requested_pairs = parse_pair_subset(args.pairs)
    excluded_pairs = parse_pair_subset(args.exclude_pairs) if args.exclude_pairs is not None else set()

    if not args.diva_dir.is_dir():
        raise FileNotFoundError(f"DiVa-360 sequence dir not found: {args.diva_dir}")
    diva_cams = load_diva_cam_poses(args.diva_dir)
    novel_cam_names = [name.strip() for name in args.novel_cams.split(",") if name.strip()]
    if not novel_cam_names:
        raise ValueError("--novel-cams must list at least one DiVa-360 camera.")
    missing_cams = [name for name in [args.ref_cam, *novel_cam_names] if name not in diva_cams]
    if missing_cams:
        raise ValueError(
            f"Camera(s) not found in DiVa-360 calibration ({args.diva_dir}): {missing_cams}"
        )
    r_ref_diva = diva_cams[args.ref_cam][:3, :3]
    novel_cam_angle_deg = {
        name: diva_cam_angle_deg(diva_cams, args.ref_cam, name) for name in novel_cam_names
    }
    print(f"Novel views from {len(novel_cam_names)} DiVa-360 camera(s) relative to {args.ref_cam}:")
    for name in novel_cam_names:
        print(f"  {name}: {novel_cam_angle_deg[name]:.1f} deg")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    payload = torch_load_cpu(contact_patch_file)
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise TypeError("contact_patches.pt must contain a top-level dictionary with a 'pairs' field.")

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(
            str(checkpoint),
            device,
            work_dir=str(work_dir),
            port=None,
        )
        model = renderer.model
        model.eval()

        canonical_means = model.fg.params["means"].detach()

        # Contact-patch indices refer only to foreground Gaussians. However,
        # SceneModel.render(..., fg_only=False) renders Gaussians in this order:
        #   foreground -> background -> shadow
        # and therefore requires filter_mask.shape == (model.num_gaussians,).
        #
        # Keep the foreground count separate for trajectory computation, while
        # building render masks with the full rendered Gaussian count.
        num_fg_gaussians = int(model.num_fg_gaussians)
        num_bg_gaussians = int(model.num_bg_gaussians)
        num_shad_gaussians = int(model.num_shad_gaussians)
        num_render_gaussians = int(model.num_gaussians)

        if int(canonical_means.shape[0]) != num_fg_gaussians:
            raise RuntimeError(
                "Foreground Gaussian count mismatch: "
                f"fg means={int(canonical_means.shape[0])}, "
                f"model.num_fg_gaussians={num_fg_gaussians}"
            )
        if (
            num_render_gaussians
            != num_fg_gaussians + num_bg_gaussians + num_shad_gaussians
        ):
            raise RuntimeError(
                "SceneModel Gaussian count mismatch: "
                f"total={num_render_gaussians}, "
                f"fg={num_fg_gaussians}, "
                f"bg={num_bg_gaussians}, "
                f"shad={num_shad_gaussians}"
            )

        print(
            "[Gaussian count] "
            f"foreground={num_fg_gaussians}, "
            f"background={num_bg_gaussians}, "
            f"shadow={num_shad_gaussians}, "
            f"render_total={num_render_gaussians}"
        )

        scene_center = canonical_means.mean(dim=0)
        total_frames = int(model.num_frames)
        frames = parse_frames(args.frames, total_frames)
        frame_ids = torch.tensor(frames, dtype=torch.long, device=canonical_means.device)

        positions, transform_source = compute_positions(model, frame_ids, canonical_means)
        reference_w2cs, reference_camera_source = get_reference_w2cs(model)
        reference_w2cs = reference_w2cs.to(canonical_means.device)

        base_width, base_height = infer_image_size(model)
        width = args.image_width or base_width
        height = args.image_height or base_height

        selected_entries: list[tuple[str, dict[str, Any], int, int]] = []
        available_pairs: set[tuple[int, int]] = set()
        for pair_key, entry in payload["pairs"].items():
            if not isinstance(entry, dict):
                continue
            cluster_a = int(entry["cluster_a"])
            cluster_b = int(entry["cluster_b"])
            pair = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
            available_pairs.add(pair)
            if requested_pairs is not None and pair not in requested_pairs:
                continue
            if pair in excluded_pairs:
                continue
            selected_entries.append((str(pair_key), entry, cluster_a, cluster_b))

        if requested_pairs is not None:
            missing = sorted(requested_pairs - available_pairs)
            if missing:
                raise ValueError(
                    "Requested pairs absent from contact_patches.pt: "
                    + ", ".join(f"{a}-{b}" for a, b in missing)
                )
        if not selected_entries:
            raise RuntimeError("No contact-patch pair was selected.")

        all_rows: list[dict[str, Any]] = []
        frame_summary_rows: list[dict[str, Any]] = []
        pair_summary_rows: list[dict[str, Any]] = []
        render_keys: dict[str, list[str]] | None = None

        for pair_index, (pair_key, entry, cluster_a, cluster_b) in enumerate(selected_entries, start=1):
            patch_a = torch.as_tensor(
                entry["contact_patch_global_indices_a"], dtype=torch.long, device=canonical_means.device
            ).reshape(-1)
            patch_b = torch.as_tensor(
                entry["contact_patch_global_indices_b"], dtype=torch.long, device=canonical_means.device
            ).reshape(-1)

            if patch_a.numel() == 0 or patch_b.numel() == 0:
                raise RuntimeError(f"Pair {cluster_a}-{cluster_b} has an empty contact patch.")

            # patch_a/patch_b are foreground-global indices. SceneModel places
            # foreground Gaussians first, so these indices remain valid in the
            # full [foreground, background, shadow] render mask. Background and
            # shadow entries remain False here and become True after inversion,
            # meaning only the selected foreground patch is removed.
            if int(patch_a.max().item()) >= num_fg_gaussians:
                raise IndexError(
                    f"Pair {cluster_a}-{cluster_b}: patch A contains index "
                    f"{int(patch_a.max().item())}, but foreground count is "
                    f"{num_fg_gaussians}."
                )
            if int(patch_b.max().item()) >= num_fg_gaussians:
                raise IndexError(
                    f"Pair {cluster_a}-{cluster_b}: patch B contains index "
                    f"{int(patch_b.max().item())}, but foreground count is "
                    f"{num_fg_gaussians}."
                )

            patch_filter_a = pair_filter_from_indices(
                num_render_gaussians,
                patch_a,
            )
            patch_filter_b = pair_filter_from_indices(
                num_render_gaussians,
                patch_b,
            )

            pair_rows: list[dict[str, Any]] = []
            pair_frame_rows: list[dict[str, Any]] = []

            for local_t, frame in enumerate(frames):
                current_pos_a = positions[patch_a, local_t]
                current_pos_b = positions[patch_b, local_t]

                target = (
                    torch.cat([current_pos_a, current_pos_b], dim=0).mean(dim=0)
                    if args.orbit_center == "patch-center"
                    else scene_center
                )
                reference_frame = frame if args.reference_camera == "per-frame" else 0
                reference_w2c = reference_w2cs[reference_frame]
                K = scale_intrinsics(
                    model.Ks[reference_frame].to(canonical_means.device),
                    base_width=base_width,
                    base_height=base_height,
                    width=width,
                    height=height,
                )

                current_frame_rows: list[dict[str, Any]] = []

                for cam_name in novel_cam_names:
                    novel_w2c = diva_novel_camera(
                        reference_w2c=reference_w2c,
                        target=target,
                        r_ref_diva=r_ref_diva,
                        r_cam_diva=diva_cams[cam_name][:3, :3],
                        radius_scale=args.orbit_radius_scale,
                    )

                    render_data = render_views(
                        model=model,
                        frame=frame,
                        w2c=novel_w2c,
                        K=K,
                        width=width,
                        height=height,
                        patch_filter_a=patch_filter_a,
                        patch_filter_b=patch_filter_b,
                    )
                    render_keys = {
                        "full": render_data["full_keys"],
                        "without_a": render_data["without_a_keys"],
                        "without_b": render_data["without_b_keys"],
                    }

                    uv_a, _, _ = project_points(
                        current_pos_a, novel_w2c, K, width, height
                    )
                    uv_b, _, _ = project_points(
                        current_pos_b, novel_w2c, K, width, height
                    )

                    visible_mask_a, contribution_samples_a = filter_visible_points(
                        uv=uv_a,
                        contribution_map=render_data["contribution_a"],
                        contribution_threshold=args.contribution_threshold,
                    )
                    visible_mask_b, contribution_samples_b = filter_visible_points(
                        uv=uv_b,
                        contribution_map=render_data["contribution_b"],
                        contribution_threshold=args.contribution_threshold,
                    )

                    visible_uv_a = uv_a[visible_mask_a]
                    visible_uv_b = uv_b[visible_mask_b]

                    visible_fraction_a = float(
                        len(visible_uv_a) / max(len(uv_a), 1)
                    )
                    visible_fraction_b = float(
                        len(visible_uv_b) / max(len(uv_b), 1)
                    )

                    view_valid = (
                        len(visible_uv_a) >= args.min_visible_gaussians
                        and len(visible_uv_b) >= args.min_visible_gaussians
                        and visible_fraction_a >= args.min_visible_fraction
                        and visible_fraction_b >= args.min_visible_fraction
                    )

                    if view_valid:
                        gap = bidirectional_projected_gap(
                            visible_uv_a, visible_uv_b
                        )
                        ratios = contact_ratios(
                            visible_uv_a,
                            visible_uv_b,
                            args.contact_threshold_px,
                        )
                        connected_view = bool(
                            gap["median_px"]
                            <= args.connected_gap_threshold_px
                            and ratios["contact_ratio_joint"]
                            >= args.connected_contact_ratio_threshold
                        )

                        # Per-view classification intentionally uses only:
                        # connected / ambiguous / unknown_due_to_occlusion.
                        # Even a very large visible gap is ambiguous here;
                        # globally separated pairs are detected later using
                        # pair-level robust outlier statistics.
                        separated_view = False
                        view_status = (
                            "connected" if connected_view else "ambiguous"
                        )
                    else:
                        gap = bidirectional_projected_gap(
                            np.empty((0, 2), dtype=np.float32),
                            np.empty((0, 2), dtype=np.float32),
                        )
                        ratios = contact_ratios(
                            np.empty((0, 2), dtype=np.float32),
                            np.empty((0, 2), dtype=np.float32),
                            args.contact_threshold_px,
                        )
                        separated_view = False
                        view_status = "unknown_due_to_occlusion"

                    row = {
                        "cluster_a": cluster_a,
                        "cluster_b": cluster_b,
                        "pair_key": pair_key,
                        "frame": frame,
                        "reference_frame": reference_frame,
                        "novel_cam": cam_name,
                        "novel_cam_angle_deg": novel_cam_angle_deg[cam_name],
                        "patch_count_a": int(patch_a.numel()),
                        "patch_count_b": int(patch_b.numel()),
                        "projected_in_frame_a": int(len(uv_a)),
                        "projected_in_frame_b": int(len(uv_b)),
                        "visible_projected_count_a": int(len(visible_uv_a)),
                        "visible_projected_count_b": int(len(visible_uv_b)),
                        "visible_fraction_a": visible_fraction_a,
                        "visible_fraction_b": visible_fraction_b,
                        "mean_contribution_sample_a": (
                            float(np.mean(contribution_samples_a))
                            if len(contribution_samples_a) else float("nan")
                        ),
                        "mean_contribution_sample_b": (
                            float(np.mean(contribution_samples_b))
                            if len(contribution_samples_b) else float("nan")
                        ),
                        "view_valid": bool(view_valid),
                        "view_status": view_status,
                        "projected_gap_median_px": gap["median_px"],
                        "projected_gap_mean_px": gap["mean_px"],
                        "projected_gap_p90_px": gap["p90_px"],
                        "projected_gap_a_to_b_median_px": gap[
                            "a_to_b_median_px"
                        ],
                        "projected_gap_b_to_a_median_px": gap[
                            "b_to_a_median_px"
                        ],
                        "contact_ratio_a": ratios["contact_ratio_a"],
                        "contact_ratio_b": ratios["contact_ratio_b"],
                        "contact_ratio_joint": ratios[
                            "contact_ratio_joint"
                        ],
                        "separated_view": bool(separated_view),
                        "contribution_threshold": args.contribution_threshold,
                        "contact_threshold_px": args.contact_threshold_px,
                        "separation_gap_threshold_px": (
                            args.separation_gap_threshold_px
                        ),
                        "separation_contact_ratio_threshold": (
                            args.separation_contact_ratio_threshold
                        ),
                        "connected_gap_threshold_px": (
                            args.connected_gap_threshold_px
                        ),
                        "connected_contact_ratio_threshold": (
                            args.connected_contact_ratio_threshold
                        ),
                        "overlay_path": "",
                        "contribution_a_path": "",
                        "contribution_b_path": "",
                    }

                    stem = f"pair_{cluster_a}_{cluster_b}_frame_{frame:04d}_cam_{cam_name}"

                    if not args.metrics_only:
                        overlay_path = overlay_dir / f"{stem}.png"
                        gap_text = (
                            f"{gap['median_px']:.2f}px"
                            if np.isfinite(gap["median_px"])
                            else "NA"
                        )
                        contact_text = (
                            f"{ratios['contact_ratio_joint']:.2f}"
                            if np.isfinite(ratios["contact_ratio_joint"])
                            else "NA"
                        )
                        save_overlay(
                            path=overlay_path,
                            rgb=render_data["rgb"],
                            all_uv_a=uv_a,
                            all_uv_b=uv_b,
                            vis_uv_a=visible_uv_a,
                            vis_uv_b=visible_uv_b,
                            title=(
                                f"{cluster_a}-{cluster_b} f={frame} "
                                f"cam={cam_name} ({novel_cam_angle_deg[cam_name]:.0f} deg) "
                                f"{view_status} gap={gap_text} "
                                f"contact={contact_text}"
                            ),
                        )
                        row["overlay_path"] = str(overlay_path)

                        contribution_a_path = alpha_dir / f"{stem}_A.png"
                        contribution_b_path = alpha_dir / f"{stem}_B.png"
                        save_alpha(
                            contribution_a_path,
                            render_data["contribution_a"],
                            title=(
                                f"{cluster_a}-{cluster_b} full-scene "
                                "contribution of patch A"
                            ),
                        )
                        save_alpha(
                            contribution_b_path,
                            render_data["contribution_b"],
                            title=(
                                f"{cluster_a}-{cluster_b} full-scene "
                                "contribution of patch B"
                            ),
                        )
                        row["contribution_a_path"] = str(
                            contribution_a_path
                        )
                        row["contribution_b_path"] = str(
                            contribution_b_path
                        )

                    current_frame_rows.append(row)
                    pair_rows.append(row)
                    all_rows.append(row)

                    print(
                        f"[{pair_index:03d}/{len(selected_entries):03d}] "
                        f"{cluster_a}-{cluster_b} frame={frame:03d} "
                        f"cam={cam_name} ({novel_cam_angle_deg[cam_name]:+.1f} deg) "
                        f"vis=({len(visible_uv_a)},{len(visible_uv_b)}) "
                        f"status={view_status}"
                    )

                valid_rows = [
                    row for row in current_frame_rows
                    if bool(row["view_valid"])
                ]
                connected_rows = [
                    row for row in valid_rows
                    if row["view_status"] == "connected"
                ]
                valid_view_count = len(valid_rows)
                separated_view_count = 0
                connected_view_count = len(connected_rows)
                separated_fraction = 0.0 if valid_view_count > 0 else float("nan")
                connected_fraction = (
                    connected_view_count / valid_view_count
                    if valid_view_count > 0
                    else float("nan")
                )

                if valid_view_count < args.min_valid_views:
                    frame_status = "unknown_due_to_occlusion"
                elif connected_fraction >= 0.50:
                    frame_status = "connected"
                else:
                    frame_status = "ambiguous"

                valid_gaps = [
                    float(row["projected_gap_median_px"])
                    for row in valid_rows
                    if np.isfinite(float(row["projected_gap_median_px"]))
                ]
                valid_contacts = [
                    float(row["contact_ratio_joint"])
                    for row in valid_rows
                    if np.isfinite(float(row["contact_ratio_joint"]))
                ]

                frame_row = {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "pair_key": pair_key,
                    "frame": frame,
                    "total_view_count": len(current_frame_rows),
                    "valid_view_count": valid_view_count,
                    "occluded_or_invalid_view_count": (
                        len(current_frame_rows) - valid_view_count
                    ),
                    "separated_view_count": separated_view_count,
                    "separated_view_fraction": separated_fraction,
                    "connected_view_count": connected_view_count,
                    "connected_view_fraction": connected_fraction,
                    "frame_status": frame_status,
                    "median_valid_view_gap_px": (
                        float(np.median(valid_gaps))
                        if valid_gaps else float("nan")
                    ),
                    "maximum_valid_view_gap_px": (
                        float(np.max(valid_gaps))
                        if valid_gaps else float("nan")
                    ),
                    "median_valid_view_contact_ratio": (
                        float(np.median(valid_contacts))
                        if valid_contacts else float("nan")
                    ),
                    "minimum_valid_view_contact_ratio": (
                        float(np.min(valid_contacts))
                        if valid_contacts else float("nan")
                    ),
                    "min_valid_views_required": args.min_valid_views,
                    "separation_view_fraction_threshold": (
                        args.separation_view_fraction
                    ),
                }
                pair_frame_rows.append(frame_row)
                frame_summary_rows.append(frame_row)

                print(
                    f"    frame summary: valid={valid_view_count}/"
                    f"{len(current_frame_rows)}, "
                    f"separated={separated_view_count}, "
                    f"status={frame_status}"
                )

            status_counts = {
                status: sum(
                    1 for row in pair_frame_rows
                    if row["frame_status"] == status
                )
                for status in (
                    "connected",
                    "ambiguous",
                    "unknown_due_to_occlusion",
                )
            }

            valid_pair_rows = [
                row for row in pair_rows if bool(row["view_valid"])
            ]
            valid_gaps = [
                float(row["projected_gap_median_px"])
                for row in valid_pair_rows
                if np.isfinite(float(row["projected_gap_median_px"]))
            ]
            valid_contacts = [
                float(row["contact_ratio_joint"])
                for row in valid_pair_rows
                if np.isfinite(float(row["contact_ratio_joint"]))
            ]

            temporal_metrics = temporal_pair_metrics(
                pair_frame_rows=pair_frame_rows,
                absolute_gap_threshold_px=args.pair_separation_min_gap_px,
                reference_frame_count=args.temporal_reference_frame_count,
                gap_ratio_threshold=args.temporal_gap_ratio_threshold,
                persistent_min_frames=args.temporal_persistent_min_frames,
                jump_threshold_px=args.temporal_jump_threshold_px,
                unstable_min_jumps=args.temporal_unstable_min_jumps,
            )

            summary_row = {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "pair_key": pair_key,
                "frame_count": len(pair_frame_rows),
                "view_count": len(pair_rows),
                "valid_view_count": len(valid_pair_rows),
                "valid_view_fraction": (
                    len(valid_pair_rows) / max(len(pair_rows), 1)
                ),
                "connected_frame_count": status_counts["connected"],
                "separated_frame_count": 0,
                "ambiguous_frame_count": status_counts["ambiguous"],
                "unknown_frame_count": status_counts[
                    "unknown_due_to_occlusion"
                ],
                "median_valid_view_gap_px": (
                    float(np.median(valid_gaps))
                    if valid_gaps else float("nan")
                ),
                "maximum_valid_view_gap_px": (
                    float(np.max(valid_gaps))
                    if valid_gaps else float("nan")
                ),
                "median_valid_view_contact_ratio": (
                    float(np.median(valid_contacts))
                    if valid_contacts else float("nan")
                ),
                "minimum_valid_view_contact_ratio": (
                    float(np.min(valid_contacts))
                    if valid_contacts else float("nan")
                ),
                "separated_frames": "",
                "unknown_frames": ",".join(
                    str(row["frame"]) for row in pair_frame_rows
                    if row["frame_status"] == "unknown_due_to_occlusion"
                ),
                **temporal_metrics,
            }
            pair_summary_rows.append(summary_row)

        # -------------------------------------------------------------
        # Global pair-level separation detection.
        #
        # Per-view/frame statuses above remain connected / ambiguous /
        # unknown. Here only, a whole cluster pair is marked "separated"
        # when its pair-level median valid-view gap is an unusually large
        # outlier relative to the other cluster pairs.
        # -------------------------------------------------------------
        finite_pair_scores = np.asarray(
            [
                float(row["median_valid_view_gap_px"])
                for row in pair_summary_rows
                if np.isfinite(float(row["median_valid_view_gap_px"]))
            ],
            dtype=np.float64,
        )

        if finite_pair_scores.size > 0:
            pair_gap_q1 = float(np.quantile(finite_pair_scores, 0.25))
            pair_gap_q3 = float(np.quantile(finite_pair_scores, 0.75))
            pair_gap_iqr = pair_gap_q3 - pair_gap_q1
            robust_pair_gap_threshold = (
                pair_gap_q3
                + args.pair_separation_iqr_multiplier * pair_gap_iqr
            )
            final_pair_gap_threshold = max(
                robust_pair_gap_threshold,
                args.pair_separation_min_gap_px,
            )
        else:
            pair_gap_q1 = float("nan")
            pair_gap_q3 = float("nan")
            pair_gap_iqr = float("nan")
            robust_pair_gap_threshold = float("nan")
            final_pair_gap_threshold = float("inf")

        separated_pair_rows: list[dict[str, Any]] = []
        retained_pair_summary_rows: list[dict[str, Any]] = []
        separated_pair_keys: set[tuple[int, int]] = set()

        for row in pair_summary_rows:
            score = float(row["median_valid_view_gap_px"])
            global_gap_outlier = bool(
                np.isfinite(score)
                and score > final_pair_gap_threshold
            )
            persistent_separation = bool(row["persistent_separation"])
            temporally_unstable = bool(row["temporally_unstable"])

            pair_is_separated = bool(
                global_gap_outlier
                or persistent_separation
                or temporally_unstable
            )

            separation_reasons: list[str] = []
            if global_gap_outlier:
                separation_reasons.append("global_gap_outlier")
            if persistent_separation:
                separation_reasons.append("persistent_separation")
            if temporally_unstable:
                separation_reasons.append("temporal_instability")

            row["pair_status"] = (
                "separated" if pair_is_separated else "retained"
            )
            row["separation_reason"] = ",".join(separation_reasons)
            row["global_gap_outlier"] = global_gap_outlier
            row["global_pair_gap_threshold_px"] = final_pair_gap_threshold

            if pair_is_separated:
                separated_pair_keys.add(
                    (int(row["cluster_a"]), int(row["cluster_b"]))
                )
                separated_pair_rows.append(row.copy())
            else:
                retained_pair_summary_rows.append(row)

        # Keep detailed metrics for traceability, but exclude globally
        # separated pairs from the main pair summary and all plot tables.
        plot_rows = [
            row for row in all_rows
            if (int(row["cluster_a"]), int(row["cluster_b"]))
            not in separated_pair_keys
        ]

        # -------------------------------------------------------------
        # Keep overlay/contribution-map images only for the top-10
        # ambiguous cluster pairs.
        #
        # Ranking priority:
        #   1. more ambiguous frames
        #   2. fewer connected frames
        #   3. larger pair-level median valid-view gap
        #
        # Metrics/CSV/report generation is otherwise unchanged.
        # -------------------------------------------------------------
        ambiguous_candidates = [
            row for row in retained_pair_summary_rows
            if int(row["ambiguous_frame_count"]) > 0
        ]
        ambiguous_candidates.sort(
            key=lambda row: (
                -int(row["ambiguous_frame_count"]),
                int(row["connected_frame_count"]),
                -(
                    float(row["median_valid_view_gap_px"])
                    if np.isfinite(float(row["median_valid_view_gap_px"]))
                    else float("-inf")
                ),
                int(row["cluster_a"]),
                int(row["cluster_b"]),
            )
        )
        top_ambiguous_pair_rows = ambiguous_candidates[:10]
        top_ambiguous_pair_keys = {
            (int(row["cluster_a"]), int(row["cluster_b"]))
            for row in top_ambiguous_pair_rows
        }

        if not args.metrics_only:
            allowed_prefixes = tuple(
                f"pair_{cluster_a}_{cluster_b}_"
                for cluster_a, cluster_b in sorted(top_ambiguous_pair_keys)
            )

            for image_dir in (overlay_dir, alpha_dir):
                for image_path in image_dir.glob("*.png"):
                    if not allowed_prefixes or not image_path.name.startswith(
                        allowed_prefixes
                    ):
                        image_path.unlink(missing_ok=True)

            # Avoid leaving CSV paths that point to deleted images.
            for row in all_rows:
                pair = (int(row["cluster_a"]), int(row["cluster_b"]))
                if pair not in top_ambiguous_pair_keys:
                    row["overlay_path"] = ""
                    row["contribution_a_path"] = ""
                    row["contribution_b_path"] = ""

        write_csv(
            output_dir / "contact_patch_view_metrics_all.csv",
            all_rows,
        )
        write_csv(
            output_dir / "contact_patch_frame_summary.csv",
            frame_summary_rows,
        )
        write_csv(
            output_dir / "contact_patch_pair_summary.csv",
            retained_pair_summary_rows,
        )
        write_csv(
            output_dir / "separated_cluster_pairs.csv",
            separated_pair_rows,
        )

        if not args.metrics_only:
            save_heatmap(
                plot_rows,
                plot_dir / "all_pairs_projected_gap_median.png",
                value_key="projected_gap_median_px",
                title="Visible 2D projected contact-patch median gap",
                colorbar_label="Median 2D nearest-neighbour distance (px)",
            )
            save_heatmap(
                plot_rows,
                plot_dir / "all_pairs_contact_ratio_joint.png",
                value_key="contact_ratio_joint",
                title="Visible 2D contact ratio under pixel threshold",
                colorbar_label="Visible-point contact ratio",
            )

        report = {
            "work_dir": str(work_dir),
            "checkpoint": str(checkpoint),
            "contact_patch_file": str(contact_patch_file),
            "output_dir": str(output_dir),
            "pair_count": len(pair_summary_rows),
            "retained_pair_count": len(retained_pair_summary_rows),
            "separated_pair_count": len(separated_pair_rows),
            "top_ambiguous_pair_count": len(top_ambiguous_pair_rows),
            "top_ambiguous_pairs": [
                {
                    "cluster_a": int(row["cluster_a"]),
                    "cluster_b": int(row["cluster_b"]),
                    "ambiguous_frame_count": int(row["ambiguous_frame_count"]),
                    "connected_frame_count": int(row["connected_frame_count"]),
                    "median_valid_view_gap_px": float(
                        row["median_valid_view_gap_px"]
                    ),
                }
                for row in top_ambiguous_pair_rows
            ],
            "frames": frames,
            "diva_dir": str(args.diva_dir),
            "ref_cam": args.ref_cam,
            "novel_cams": novel_cam_names,
            "novel_cam_angles_deg": novel_cam_angle_deg,
            "reference_camera": args.reference_camera,
            "reference_camera_source": reference_camera_source,
            "orbit_center": args.orbit_center,
            "orbit_radius_scale": args.orbit_radius_scale,
            "image_size": [width, height],
            "transform_source": transform_source,
            "render_result_keys": render_keys,
            "visibility_definition": {
                "projection": (
                    "Patch Gaussian centers must project inside the image and "
                    "in front of the novel camera."
                ),
                "full_scene_contribution": (
                    "Render the complete scene, the complete scene with patch A "
                    "removed, and the complete scene with patch B removed. "
                    "A patch contribution map is max(mean absolute RGB change, "
                    "positive accumulated-alpha change)."
                ),
                "gaussian_visibility_proxy": (
                    "A projected Gaussian is treated as visible when the sampled "
                    f"full-scene contribution map is >= {args.contribution_threshold}."
                ),
                "occlusion_scope": (
                    "Because contribution is measured in the final full-scene "
                    "render, occlusion by the opposite patch, other clusters, and "
                    "background geometry is respected."
                ),
            },
            "valid_view_definition": {
                "minimum_visible_gaussians_each_patch": (
                    args.min_visible_gaussians
                ),
                "minimum_visible_fraction_each_patch": (
                    args.min_visible_fraction
                ),
                "invalid_status": "unknown_due_to_occlusion",
            },
            "2d_gap_definition": {
                "correspondence": (
                    "No fixed Gaussian pairs. Visible projected A-to-B and "
                    "B-to-A nearest neighbours are recomputed independently "
                    "for every valid view."
                ),
                "main_metric": (
                    "Median of concatenated A-to-B and B-to-A 2D nearest-"
                    "neighbour distances in pixels."
                ),
                "contact_ratio": (
                    f"Fraction of visible patch points whose nearest opposite "
                    f"visible point is within {args.contact_threshold_px} pixels."
                ),
            },
            "view_classification": {
                "connected_if": (
                    f"median gap <= {args.connected_gap_threshold_px} px AND "
                    f"joint contact ratio >= "
                    f"{args.connected_contact_ratio_threshold}"
                ),
                "otherwise_valid_view": "ambiguous",
                "note": (
                    "Per-view separated is disabled. Views that would previously "
                    "have been separated are now ambiguous."
                ),
            },
            "frame_classification": {
                "unknown_due_to_occlusion_if": (
                    f"valid view count < {args.min_valid_views}"
                ),
                "connected_if": "connected valid-view fraction >= 0.5",
                "otherwise": "ambiguous",
                "note": "Frame-level separated is disabled.",
            },
            "global_pair_classification": {
                "score": "pair-level median valid-view gap in pixels",
                "robust_threshold": (
                    f"Q3 + {args.pair_separation_iqr_multiplier} * IQR"
                ),
                "minimum_threshold_px": args.pair_separation_min_gap_px,
                "q1_px": pair_gap_q1,
                "q3_px": pair_gap_q3,
                "iqr_px": pair_gap_iqr,
                "robust_threshold_px": robust_pair_gap_threshold,
                "final_threshold_px": final_pair_gap_threshold,
                "temporal_reference_frame_count": (
                    args.temporal_reference_frame_count
                ),
                "temporal_gap_ratio_threshold": (
                    args.temporal_gap_ratio_threshold
                ),
                "temporal_persistent_min_frames": (
                    args.temporal_persistent_min_frames
                ),
                "temporal_jump_threshold_px": (
                    args.temporal_jump_threshold_px
                ),
                "temporal_unstable_min_jumps": (
                    args.temporal_unstable_min_jumps
                ),
                "separated_if": (
                    "pair-level median valid-view gap > final threshold OR "
                    "the temporal separation threshold is exceeded for the "
                    "required number of consecutive valid frames OR the "
                    "frame-level gap has the required number of large jumps"
                ),
                "excluded_from": [
                    "contact_patch_pair_summary.csv",
                    "plots/all_pairs_projected_gap_median.png",
                    "plots/all_pairs_contact_ratio_joint.png",
                ],
            },
            "outputs": {
                "view_metrics": str(
                    output_dir / "contact_patch_view_metrics_all.csv"
                ),
                "frame_summary": str(
                    output_dir / "contact_patch_frame_summary.csv"
                ),
                "pair_summary": str(
                    output_dir / "contact_patch_pair_summary.csv"
                ),
                "separated_pairs": str(
                    output_dir / "separated_cluster_pairs.csv"
                ),
            },
            "frame_summary": frame_summary_rows,
            "pair_summary": retained_pair_summary_rows,
            "separated_pairs": separated_pair_rows,
        }
        with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        print()
        print("=" * 78)
        print("Occlusion-aware contact-patch distance analysis complete")
        print(f"Pairs   : {len(pair_summary_rows)} total")
        print(f"Retained: {len(retained_pair_summary_rows)}")
        print(f"Separated pairs: {len(separated_pair_rows)}")
        print(
            "Overlay/contribution images kept for top ambiguous pairs: "
            + (
                ", ".join(
                    f"{int(row['cluster_a'])}-{int(row['cluster_b'])}"
                    for row in top_ambiguous_pair_rows
                )
                if top_ambiguous_pair_rows
                else "none"
            )
        )
        if separated_pair_rows:
            separated_names = ", ".join(
                f"{int(row['cluster_a'])}-{int(row['cluster_b'])}"
                for row in separated_pair_rows
            )
            print(f"Separated pairs: {separated_names}")
        print(
            f"Pair-gap threshold: {final_pair_gap_threshold:.3f}px "
            f"(Q3={pair_gap_q3:.3f}, IQR={pair_gap_iqr:.3f})"
        )
        print(f"Frames  : {len(frames)}")
        print(f"Views   : {len(frames) * len(novel_cam_names)} per pair")
        print(f"Output  : {output_dir}")
        print("=" * 78)


if __name__ == "__main__":
    main()
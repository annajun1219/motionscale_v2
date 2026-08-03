#!/usr/bin/env python3
"""
Render a trained DAVIS-format scene from DyCheck-derived novel viewpoints.

Motivation
----------
Training only ever supervises one physical camera (the moving "camera0" of a
DyCheck iPhone capture), so the model has no ground truth for any other
viewpoint. This script renders the trained scene from:
  - the two DyCheck side cameras (camera1, camera2), which are real, static
    cameras captured alongside the main video but never used for training, and
  - up to three synthetic "novel" viewpoints, placed by (azimuth, elevation,
    radius) around the scene center, chosen to extrapolate only modestly
    beyond the range actually swept by camera0/camera1/camera2 (Gaussian
    Splatting degrades quickly far from any training view).

Coordinate-frame note
----------------------
DyCheck's `camera/*.json` poses live in their own SfM-gauge coordinate frame
(the first camera0 frame is fixed at position=0, orientation=identity by
bundle-adjustment gauge choice), using the OpenCV local camera-axis
convention (camera looks down +Z, Y is down) -- verified empirically by
checking that each camera's local +Z axis (transformed to world space)
aligns with the direction toward `scene.json`'s "center" field (dot product
~0.81-0.92 across camera0/camera1/camera2), and that local +Y aligns with
-`extra.json`'s "up" field. The trained scene lives in mega-sam's SLAM
coordinate frame (arbitrary origin/scale), which uses the *same* OpenCV
convention. Because both frames share axis conventions, no OpenGL<->OpenCV
flip is needed (contrast with the DiVa-360 rig case, which does need one).

Instead, this script:
  1. Takes the *relative* rotation from the reference camera (camera0's first
     frame) to each candidate view, in DyCheck's own frame. A relative
     rotation expressed in a camera's local axes is coordinate-frame
     independent, so it can be transplanted onto the reference camera's
     *trained* pose.
  2. Keeps the novel camera at the same distance from the scene as the
     reference camera (mega-sam's scale is unknown, so DyCheck's absolute
     translations aren't usable) and orbits it around the foreground
     centroid using that transplanted rotation.

This gives a physically-reasonable novel view for modest angular offsets from
the reference camera; it is not a substitute for real multi-view supervision,
and views far outside the observed camera0/camera1/camera2 cluster (roughly
azimuth -104..-74 deg, elevation -31..-23 deg, radius ~2.0-2.2 in DyCheck's
own units, measured about `scene.json`'s "center") should be expected to
degrade.

Example
-------
    python flow3d/analysis/2drender/render_output_novelview.py \\
        --ckpt outputs/davis/spin/2026_08_03_08_41_11__spin_run1/checkpoints/last.ckpt \\
        --seq_name spin \\
        --dycheck_dir data/DyCheck/spin
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v2 as iio
import imageio.v3 as iio3
import numpy as np
import torch
import yaml

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.renderer import Renderer
from flow3d.vis.utils import make_video_divisble

# data/DAVIS/JPEGImages/480p/<seq>/ was built by keeping every 3rd camera0
# frame from data/DyCheck/<seq>/rgb/1x/ (see preprocessing notes), so a
# training frame index maps back to a DyCheck camera0 frame number via *3.
DYCHECK_SUBSAMPLE_INTERVAL = 3

# Synthetic novel views: modest extrapolations beyond the observed
# camera0/camera1/camera2 cluster (azimuth -104..-74 deg, elevation
# -31..-23 deg, radius ~2.0-2.2), measured about scene.json's "center".
SYNTHETIC_VIEWS = {
    "novel1": {"azimuth_deg": -55.0, "elevation_deg": -27.0, "radius": 2.1},
    "novel2": {"azimuth_deg": -125.0, "elevation_deg": -27.0, "radius": 2.1},
    "novel3": {"azimuth_deg": -90.0, "elevation_deg": 10.0, "radius": 2.1},
}
REAL_DYCHECK_VIEWS = ["camera1", "camera2"]
ALL_VIEW_NAMES = REAL_DYCHECK_VIEWS + list(SYNTHETIC_VIEWS.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt", type=Path,
        default=Path("outputs/davis/spin/2026_08_03_08_41_11__spin_run1/checkpoints/last.ckpt"),
        help="Path to a trained SceneModel checkpoint (last.ckpt).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/davis/spin.yaml"))
    parser.add_argument("--seq_name", type=str, default="spin")
    parser.add_argument(
        "--dycheck_dir", type=Path, default=Path("data/DyCheck/spin"),
        help="DyCheck sequence dir containing camera/, scene.json, extra.json.",
    )
    parser.add_argument(
        "--views", type=str, default="all",
        help=f'Comma list from {ALL_VIEW_NAMES}, or "all" (default: all 5).',
    )
    parser.add_argument(
        "--frames", type=str, default="all",
        help='Comma list of training frame indices, or "all" (default: every training '
             "frame -- needed so the per-view videos cover the full sequence).",
    )
    parser.add_argument(
        "--image_stride", type=int, default=10,
        help="Also dump a standalone PNG every N frames (e.g. frame0000, frame0010, "
             "frame0020, ...). Does not affect which frames go into the videos.",
    )
    parser.add_argument(
        "--no_video", action="store_true",
        help="Skip stitching per-view videos; only dump the strided PNGs.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frame rate for the per-view output videos.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def load_scene_geometry(dycheck_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the DyCheck scene pivot (scene.json "center") and up vector (extra.json "up")."""
    scene = json.loads((dycheck_dir / "scene.json").read_text())
    extra = json.loads((dycheck_dir / "extra.json").read_text())
    center = np.array(scene["center"], dtype=np.float64)
    up = np.array(extra["up"], dtype=np.float64)
    up /= np.linalg.norm(up)
    return center, up


def load_dycheck_cam_json(dycheck_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one DyCheck camera/<name>.json; return (position, camera-to-world rotation)."""
    path = dycheck_dir / "camera" / filename
    if not path.is_file():
        raise FileNotFoundError(f"DyCheck camera pose not found: {path}")
    data = json.loads(path.read_text())
    position = np.array(data["position"], dtype=np.float64)
    r_w2c = np.array(data["orientation"], dtype=np.float64)
    return position, r_w2c.T


def load_dycheck_static_cam(dycheck_dir: Path, cam_id: str) -> tuple[np.ndarray, np.ndarray]:
    """camera1/camera2 are static across the whole capture; any of their frames works."""
    candidates = sorted((dycheck_dir / "camera").glob(f"{cam_id}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No camera/{cam_id}_*.json files found under {dycheck_dir}")
    return load_dycheck_cam_json(dycheck_dir, candidates[0].name)


def load_dycheck_ref_cam(dycheck_dir: Path, train_frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    orig_idx = train_frame_idx * DYCHECK_SUBSAMPLE_INTERVAL
    filename = f"0_{orig_idx:05d}.json"
    try:
        return load_dycheck_cam_json(dycheck_dir, filename)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\nTraining frame {train_frame_idx} maps to DyCheck camera0 frame "
            f"{orig_idx} (via *{DYCHECK_SUBSAMPLE_INTERVAL} subsampling), but that pose "
            "file is missing. Frame 0 (DyCheck frame 0000, the SfM gauge origin) is "
            "always available; other frames may not have a matching camera0 pose."
        ) from e


def lookat_c2w(pos: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation (OpenCV axes: x=right, y=down, z=forward) looking at target."""
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(-world_up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, z], axis=1)


def synth_novel_cam(
    center: np.ndarray, up: np.ndarray, azimuth_deg: float, elevation_deg: float, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a DyCheck-frame (position, c2w rotation) at (azimuth, elevation, radius)
    about `center`, looking back at `center`."""
    ref = np.array([1.0, 0.0, 0.0])
    tangent1 = ref - np.dot(ref, up) * up
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(up, tangent1)

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    direction = (
        np.cos(el) * np.cos(az) * tangent1
        + np.cos(el) * np.sin(az) * tangent2
        + np.sin(el) * up
    )
    position = center + radius * direction
    return position, lookat_c2w(position, center, up)


def azim_elev_radius(pos: np.ndarray, center: np.ndarray, up: np.ndarray) -> tuple[float, float, float]:
    ref = np.array([1.0, 0.0, 0.0])
    tangent1 = ref - np.dot(ref, up) * up
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(up, tangent1)
    v = pos - center
    r = np.linalg.norm(v)
    h = np.dot(v, up)
    proj = v - h * up
    az = np.degrees(np.arctan2(np.dot(proj, tangent2), np.dot(proj, tangent1)))
    el = np.degrees(np.arcsin(np.clip(h / r, -1.0, 1.0)))
    return float(az), float(el), float(r)


def build_candidate_views(
    dycheck_dir: Path, view_names: list[str]
) -> dict[str, dict]:
    """Load/synthesize each requested view in DyCheck's own coordinate frame."""
    center, up = load_scene_geometry(dycheck_dir)
    views = {}
    for name in view_names:
        if name in REAL_DYCHECK_VIEWS:
            cam_id = name.replace("camera", "")
            pos, r_c2w = load_dycheck_static_cam(dycheck_dir, cam_id)
        elif name in SYNTHETIC_VIEWS:
            spec = SYNTHETIC_VIEWS[name]
            pos, r_c2w = synth_novel_cam(
                center, up, spec["azimuth_deg"], spec["elevation_deg"], spec["radius"]
            )
        else:
            raise ValueError(f"Unknown view {name!r}; choose from {ALL_VIEW_NAMES}")
        az, el, r = azim_elev_radius(pos, center, up)
        views[name] = {"r_c2w": r_c2w, "azimuth_deg": az, "elevation_deg": el, "radius": r}
    return views


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


def parse_views(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(ALL_VIEW_NAMES)
    names = [x.strip() for x in text.split(",") if x.strip()]
    invalid = [n for n in names if n not in ALL_VIEW_NAMES]
    if invalid:
        raise ValueError(f"Unknown view(s) {invalid}; choose from {ALL_VIEW_NAMES}")
    return names


def load_dataset(config_path: Path, seq_name: str) -> CasualDataset:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(
        root_dir=scene_cfg["data_dir"], seq_name=seq_name, **scene_cfg.get("data", {}),
    )
    return CasualDataset(**asdict(data_cfg))


def main() -> None:
    args = build_parser().parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.dycheck_dir.is_dir():
        raise FileNotFoundError(f"DyCheck sequence dir not found: {args.dycheck_dir}")

    out_dir = args.out_dir if args.out_dir is not None else args.ckpt.parents[1] / "novel_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    view_names = parse_views(args.views)
    candidate_views = build_candidate_views(args.dycheck_dir, view_names)
    print(f"Selected {len(candidate_views)} novel viewpoint(s):")
    for name, v in candidate_views.items():
        print(f"  {name}: azimuth={v['azimuth_deg']:.1f} deg, elevation={v['elevation_deg']:.1f} deg, radius={v['radius']:.2f}")

    dataset = load_dataset(args.config, args.seq_name)
    frames = parse_frames(args.frames, dataset.num_frames)
    image_frames = {f for f in frames if f % args.image_stride == 0}
    print(f"Rendering {len(frames)} frame(s) total; dumping standalone PNGs for {sorted(image_frames)}")

    # Anchored once at DyCheck camera0's first frame (the SfM gauge origin, always
    # present); reused for every training frame rather than re-looked-up per frame,
    # since camera0's real pose only ever drifts ~25 deg over the whole capture and
    # most training frames don't have their own exact DyCheck camera0 pose file.
    pos_ref_dycheck, r_ref_dycheck = load_dycheck_ref_cam(args.dycheck_dir, 0)
    relative_rotations = {
        name: r_ref_dycheck.T @ v["r_c2w"] for name, v in candidate_views.items()
    }
    center, up = load_scene_geometry(args.dycheck_dir)
    ref_az, ref_el, ref_r = azim_elev_radius(pos_ref_dycheck, center, up)

    summary = []
    video_buffers: dict[str, list[np.ndarray]] = {name: [] for name in candidate_views}

    with torch.no_grad():
        renderer = Renderer.init_from_checkpoint(str(args.ckpt), device, work_dir=str(out_dir), port=None)
        model = renderer.model
        model.eval()

        for frame_idx in frames:
            w2c_ref = dataset.w2cs[frame_idx].to(device)
            K = dataset.Ks[frame_idx].to(device)
            img = dataset.get_image(frame_idx)
            H, W = img.shape[:2]
            img_wh = (W, H)

            c2w_ref = torch.linalg.inv(w2c_ref)
            r_ref_train = c2w_ref[:3, :3]
            c_ref_train = c2w_ref[:3, 3]

            # Scene pivot: foreground centroid at this frame, in the trained model's frame.
            fg_means = model.compute_poses_fg(torch.tensor([frame_idx], device=device))[0]
            pivot = fg_means[:, 0, :].mean(dim=0)
            radius = torch.linalg.norm(c_ref_train - pivot)

            if frame_idx in image_frames:
                ref_out = model.render(frame_idx, w2c_ref[None], K[None], img_wh, use_learned_poses=False)
                ref_img = (ref_out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                ref_fname = f"camera0_frame{frame_idx:04d}_ref.png"
                iio.imwrite(out_dir / ref_fname, ref_img)
                summary.append({
                    "view": "camera0",
                    "is_ref": True,
                    "frame_idx": frame_idx,
                    "azimuth_deg": ref_az,
                    "elevation_deg": ref_el,
                    "dycheck_radius": ref_r,
                    "file": ref_fname,
                })

            for name, v in candidate_views.items():
                r_rel_t = torch.from_numpy(relative_rotations[name]).to(
                    device=device, dtype=r_ref_train.dtype
                )

                r_novel = r_ref_train @ r_rel_t
                forward = r_novel[:, 2]
                forward = forward / torch.linalg.norm(forward)
                c_novel = pivot - radius * forward

                c2w_novel = torch.eye(4, device=device, dtype=r_ref_train.dtype)
                c2w_novel[:3, :3] = r_novel
                c2w_novel[:3, 3] = c_novel
                w2c_novel = torch.linalg.inv(c2w_novel)

                out = model.render(frame_idx, w2c_novel[None], K[None], img_wh, use_learned_poses=False)
                img_out = (out["img"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

                if not args.no_video:
                    video_buffers[name].append(img_out)

                if frame_idx in image_frames:
                    fname = f"{name}_frame{frame_idx:04d}.png"
                    iio.imwrite(out_dir / fname, img_out)
                    summary.append({
                        "view": name,
                        "is_ref": False,
                        "frame_idx": frame_idx,
                        "azimuth_deg": v["azimuth_deg"],
                        "elevation_deg": v["elevation_deg"],
                        "dycheck_radius": v["radius"],
                        "file": fname,
                    })
                    print(f"  saved {fname}")

        if not args.no_video:
            for name, frames_out in video_buffers.items():
                video = make_video_divisble(np.stack(frames_out, axis=0))
                video_path = out_dir / f"{name}.mp4"
                iio3.imwrite(video_path, video, fps=args.fps)
                print(f"  saved {video_path} ({len(frames_out)} frames @ {args.fps}fps)")

    (out_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "image_frames": sorted(image_frames),
                "video": (not args.no_video),
                "fps": args.fps,
                "views": summary,
            },
            indent=2,
        )
    )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

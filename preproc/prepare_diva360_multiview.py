"""
Prepare DAVIS-style image folders + static-rig camera files for a set of
DiVa-360 dome-rig cameras, so they can be loaded by CasualDataset with
camera_type="static_rig".

Image source note: transforms_{train,test,val}.json's fl_x/fl_y/cx/cy/w/h
describe an "undist/camXX/*.png" image set that isn't actually present on
disk (no distortion coefficients are provided anywhere in this dataset
either, so it can't be reconstructed). frames_1/camXX/*.png matches that
w,h exactly but turns out to be a pre-segmented (black background) variant
-- useless for our purpose, which specifically needs real background
content for novel-view supervision. image/camXX/*.jpg is the real capture
with the actual studio background, already at 1280x720 (matching dog_cam00
exactly, same 100 stride-4 frame names) -- we use this directly and derive
intrinsics fresh from camera_angle_x/camera_angle_y (also in the
calibration, resolution-independent) evaluated at 1280x720, rather than
scaling the (mismatched-resolution) fl_x/fl_y/cx/cy fields. This ignores
lens distortion (no coefficients available) -- a reasonable pinhole
approximation near image center, weakest near the frame edges.

For each requested camera:
  1. Load & merge calibration from transforms_{train,test,val}.json.
  2. Convert transform_matrix from OpenGL/NeRF convention (camera looks down
     -Z, Y-up) to the OpenCV/COLMAP convention CasualDataset expects.
  3. Derive fx,fy,cx,cy from camera_angle_x/camera_angle_y at the target
     canvas (1280x720), assuming a centered principal point.
  4. For cameras other than the reference (cam00, whose DAVIS images already
     exist and match): copy image/camXX/*.jpg (already 1280x720, real
     background, same 100 frame names as dog_cam00) to
     <davis_dir>/JPEGImages/<res>/dog_camXX/.
  5. Write <davis_dir>/flow3d_preprocessed/<res>/dog_camXX/static_rig.npy
     (the schema CasualDataset.load_cameras expects: traj_c2w, img_shape,
     intrinsics, tstamps) and a small rig_calib_manifest.json (fov_x_deg,
     used later to FOV-lock MoGe depth).

This script only ever writes NEW files (static_rig.npy, manifest, and new
per-camera image folders) — it never touches dog_cam00's existing
megasam/sam2_masks/cotracker3/cache outputs.
"""

import argparse
import json
import math
import os

import cv2
import numpy as np


def load_diva360_calibration(diva360_dir: str) -> dict:
    """Merge transforms_{train,test,val}.json, keyed by camera id."""
    frames = {}
    for split in ["train", "test", "val"]:
        path = os.path.join(diva360_dir, f"transforms_{split}.json")
        with open(path) as f:
            data = json.load(f)
        for frame in data["frames"]:
            cam_id = frame["file_path"].split("/")[1]
            frames[cam_id] = frame
    return frames


def opengl_to_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    """Flip Y and Z columns of the rotation (OpenGL: look down -Z, Y-up ->
    OpenCV: look down +Z, Y-down). Translation (camera position) is
    unaffected."""
    c2w_cv = c2w_gl.copy()
    c2w_cv[:3, 1] *= -1
    c2w_cv[:3, 2] *= -1
    return c2w_cv


def intrinsics_from_fov(camera_angle_x, camera_angle_y, dst_w, dst_h):
    """Derive a centered pinhole K from horizontal/vertical FOV (radians),
    evaluated directly at the target resolution -- resolution-independent,
    unlike scaling fl_x/fl_y/cx/cy from a mismatched source resolution."""
    fx = dst_w / (2 * math.tan(camera_angle_x / 2))
    fy = dst_h / (2 * math.tan(camera_angle_y / 2))
    return fx, fy, dst_w / 2, dst_h / 2


def prepare_camera(
    cam_id: str,
    frame_meta: dict,
    diva360_dir: str,
    davis_dir: str,
    res: str,
    target_w: int,
    target_h: int,
    reference_cam: str,
    reference_seq: str,
):
    seq_name = f"dog_{cam_id}" if cam_id != reference_cam else reference_seq
    img_dir = os.path.join(davis_dir, "JPEGImages", res, seq_name)
    preproc_dir = os.path.join(davis_dir, "flow3d_preprocessed", res, seq_name)
    os.makedirs(preproc_dir, exist_ok=True)

    c2w_gl = np.array(frame_meta["transform_matrix"], dtype=np.float64)
    c2w_cv = opengl_to_opencv(c2w_gl)

    fx, fy, cx, cy = intrinsics_from_fov(
        frame_meta["camera_angle_x"], frame_meta["camera_angle_y"], target_w, target_h,
    )
    fov_x_deg = math.degrees(frame_meta["camera_angle_x"])

    if cam_id == reference_cam:
        # Images already exist (data/DAVIS/JPEGImages/<res>/dog_cam00); just
        # read the existing frame list so tstamps/frame count match.
        frame_names = sorted(
            os.path.splitext(p)[0] for p in os.listdir(img_dir)
        )
    else:
        os.makedirs(img_dir, exist_ok=True)
        src_frame_dir = os.path.join(diva360_dir, "image", cam_id)
        assert os.path.isdir(src_frame_dir), f"{src_frame_dir} not found"
        # Use the reference camera's frame list as the canonical set of
        # (frame-synchronized, stride-4) timestamps.
        ref_dir = os.path.join(davis_dir, "JPEGImages", res, reference_seq)
        frame_names = sorted(os.path.splitext(p)[0] for p in os.listdir(ref_dir))
        for name in frame_names:
            src_path = os.path.join(src_frame_dir, f"{name}.jpg")
            assert os.path.exists(src_path), f"missing frame {src_path}"
            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            assert (w, h) == (target_w, target_h), (
                f"{src_path}: expected already-matching {(target_w, target_h)}, got {(w, h)}"
            )
            cv2.imwrite(os.path.join(img_dir, f"{name}.jpg"), img)

    n = len(frame_names)
    traj_c2w = np.tile(c2w_cv[None, :, :], (n, 1, 1)).astype(np.float32)
    static_rig = {
        "traj_c2w": traj_c2w,
        "img_shape": np.array([target_h, target_w]),
        "intrinsics": np.array([fx, fy, cx, cy], dtype=np.float32),
        "tstamps": np.arange(n),
    }
    np.save(os.path.join(preproc_dir, "static_rig.npy"), static_rig, allow_pickle=True)

    manifest = {
        "cam_id": cam_id,
        "seq_name": seq_name,
        "fov_x_deg": fov_x_deg,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "w": target_w, "h": target_h,
        "num_frames": n,
    }
    with open(os.path.join(preproc_dir, "rig_calib_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{cam_id}] seq_name={seq_name} frames={n} fov_x_deg={fov_x_deg:.2f} "
          f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} -> {preproc_dir}")
    return frame_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diva360_dir", required=True, help="data/DiVa360/processed_data/dog")
    parser.add_argument("--davis_dir", required=True, help="data/DAVIS")
    parser.add_argument("--cams", nargs="+", required=True, help="e.g. cam00 cam08 cam17 cam26")
    parser.add_argument("--res", default="480p")
    parser.add_argument("--target_w", type=int, default=1280)
    parser.add_argument("--target_h", type=int, default=720)
    parser.add_argument("--reference_cam", default="cam00")
    parser.add_argument("--reference_seq", default="dog_cam00",
                         help="existing seq_name for the reference camera's images")
    args = parser.parse_args()

    calib = load_diva360_calibration(args.diva360_dir)

    frame_name_sets = {}
    for cam_id in args.cams:
        assert cam_id in calib, f"no calibration found for {cam_id}"
        frame_name_sets[cam_id] = prepare_camera(
            cam_id, calib[cam_id], args.diva360_dir, args.davis_dir, args.res,
            args.target_w, args.target_h, args.reference_cam, args.reference_seq,
        )

    ref_names = frame_name_sets[args.reference_cam]
    for cam_id, names in frame_name_sets.items():
        assert names == ref_names, (
            f"frame-name mismatch: {cam_id} has {len(names)} frames, "
            f"{args.reference_cam} has {len(ref_names)}; per-index alignment "
            f"across views would be broken"
        )
    print(f"OK: all {len(args.cams)} cameras have {len(ref_names)} matching frame names.")


if __name__ == "__main__":
    main()

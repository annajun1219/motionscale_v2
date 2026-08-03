"""
Reproject the reference camera's (cam00) frame-0 object mask into other
static-rig cameras, to seed SAM2's video mask propagation without having to
manually click a prompt for every new camera.

Algorithm (per destination camera):
  1. Backproject every foreground pixel of the source frame-0 mask to 3D
     using the source camera's depth (Pi3, already computed for cam00) and
     its static-rig K/c2w.
  2. Project those 3D points into the destination camera using its own
     static-rig K/c2w.
  3. Rasterize the (sparse) projected points into a binary mask, close small
     gaps, and keep only the largest connected component.
  4. Save as a DAVIS-palette PNG at Annotations/<res>/<dst_seq>/00000000.png
     (same format/location process_davis.py expects for its SAM2 prompt),
     plus a debug overlay PNG for visual sanity-checking before trusting it.

This is a coarse seed, not a precise segmentation -- always inspect the debug
overlay. Cameras far from the source viewpoint (e.g. near-opposite side) may
get a sparse/empty seed and need a manual fallback via
preproc/interactive_image_annotator.py instead.
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

# DAVIS 2017 PNG palette (matches preproc/sam2/tools/vos_inference.py)
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0"


def load_ann_png(path):
    mask = Image.open(path)
    palette = mask.getpalette()
    return np.array(mask).astype(np.uint8), palette


def save_ann_png(path, mask, palette):
    assert mask.dtype == np.uint8 and mask.ndim == 2
    out = Image.fromarray(mask)
    out.putpalette(palette if palette is not None else DAVIS_PALETTE)
    out.save(path)


def load_static_rig_cam(preproc_dir):
    d = np.load(os.path.join(preproc_dir, "static_rig.npy"), allow_pickle=True).item()
    c2w = np.asarray(d["traj_c2w"][0], dtype=np.float64)
    fx, fy, cx, cy = d["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    h, w = d["img_shape"]
    return c2w, K, int(h), int(w)


def backproject(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    z = depth[ys, xs]
    valid = z > 1e-4
    xs, ys, z = xs[valid].astype(np.float64), ys[valid].astype(np.float64), z[valid].astype(np.float64)
    K_inv = np.linalg.inv(K)
    uv1 = np.stack([xs, ys, np.ones_like(xs)], axis=0)  # (3, N)
    pts_cam = (K_inv @ uv1) * z[None, :]  # (3, N)
    pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])  # (4, N)
    pts_world = (c2w @ pts_cam_h)[:3].T  # (N, 3)
    return pts_world


def project(pts_world: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    valid = pts_cam[:, 2] > 1e-4
    pts_cam = pts_cam[valid]
    proj = (K @ pts_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv


def rasterize(uv: np.ndarray, h: int, w: int, dilate: int = 9) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask[v[in_bounds], u[in_bounds]] = 255
    if mask.sum() == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, np.uint8(255), np.uint8(0))


def reproject_one(davis_dir, res, src_seq, dst_seq, src_depth_type, dst_frame0_img_path, out_ann_path, out_debug_path):
    src_preproc = os.path.join(davis_dir, "flow3d_preprocessed", res, src_seq)
    dst_preproc = os.path.join(davis_dir, "flow3d_preprocessed", res, dst_seq)

    src_ann_path = os.path.join(davis_dir, "Annotations", res, src_seq, "00000000.png")
    src_mask, palette = load_ann_png(src_ann_path)
    src_mask_bin = (src_mask > 0).astype(np.uint8)

    depth_path = os.path.join(src_preproc, src_depth_type, "depths", "00000000.npy")
    depth = np.load(depth_path)
    if depth.shape != src_mask_bin.shape:
        depth = cv2.resize(depth, (src_mask_bin.shape[1], src_mask_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    c2w_src, K_src, h_src, w_src = load_static_rig_cam(src_preproc)
    c2w_dst, K_dst, h_dst, w_dst = load_static_rig_cam(dst_preproc)

    pts_world = backproject(src_mask_bin, depth, K_src, c2w_src)
    if len(pts_world) == 0:
        print(f"  WARNING: source mask backprojected to 0 valid points")
        mask_out = np.zeros((h_dst, w_dst), dtype=np.uint8)
    else:
        uv = project(pts_world, K_dst, c2w_dst)
        mask_out = rasterize(uv, h_dst, w_dst)

    coverage = float((mask_out > 0).mean())
    print(f"  seed coverage: {coverage*100:.2f}% of frame ({int((mask_out>0).sum())} px)")

    os.makedirs(os.path.dirname(out_ann_path), exist_ok=True)
    save_ann_png(out_ann_path, (mask_out > 0).astype(np.uint8), palette)

    rgb = cv2.cvtColor(cv2.imread(dst_frame0_img_path), cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask_out > 0] = (0.4 * overlay[mask_out > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    os.makedirs(os.path.dirname(out_debug_path), exist_ok=True)
    cv2.imwrite(out_debug_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_dir", required=True)
    parser.add_argument("--res", default="480p")
    parser.add_argument("--src_cam", default="dog_cam00", help="source seq_name, e.g. dog_cam00")
    parser.add_argument("--src_depth_type", default="pi3")
    parser.add_argument("--dst_cams", nargs="+", required=True, help="e.g. dog_cam08 dog_cam17 dog_cam26")
    parser.add_argument("--debug_dir", default=None, help="defaults to <davis_dir>/mask_reprojection_debug")
    args = parser.parse_args()

    debug_dir = args.debug_dir or os.path.join(args.davis_dir, "mask_reprojection_debug")

    for dst_seq in args.dst_cams:
        print(f"[{dst_seq}]")
        dst_frame0_img = os.path.join(args.davis_dir, "JPEGImages", args.res, dst_seq, "00000000.jpg")
        out_ann_path = os.path.join(args.davis_dir, "Annotations", args.res, dst_seq, "00000000.png")
        out_debug_path = os.path.join(debug_dir, f"{dst_seq}_overlay.jpg")
        coverage = reproject_one(
            args.davis_dir, args.res, args.src_cam, dst_seq, args.src_depth_type,
            dst_frame0_img, out_ann_path, out_debug_path,
        )
        flag = "LIKELY BAD -- inspect / manual fallback" if coverage < 0.01 else "ok, but inspect anyway"
        print(f"  -> {out_ann_path}")
        print(f"  -> debug overlay: {out_debug_path}  [{flag}]")


if __name__ == "__main__":
    main()

import os
import sys

basedir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(basedir, "Pi3")
sys.path.extend([src_dir])

import numpy as np
import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from matplotlib import colormaps
import imageio.v2 as iio
import time


def apply_depth_colormap(depth, near_plane=None, far_plane=None, colormap="Spectral_r"):
    """
    Converts a depth map to colored depth image for visualization.

    Args:
        depth (np.ndarray): Depth map, shape (..., H, W) or (..., H, W, 1).
        near_plane (float | None): Closest depth to consider. Defaults to min(depth).
        far_plane (float | None): Furthest depth to consider. Defaults to max(depth).
        colormap (str): "gray" for grayscale, or any matplotlib colormap name ("Spectral_r", "turbo").

    Returns:
        np.ndarray: (..., H, W, 3) uint8 RGB image.
    """
    if depth.shape[-1] == 1:
        depth = depth[..., 0]

    # deal with nan, inf
    depth = np.nan_to_num(depth, nan=np.nan, posinf=np.nan, neginf=np.nan)
    if not np.isfinite(depth).any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # normalize
    near_plane = near_plane if near_plane is not None else float(np.nanmin(depth))
    far_plane = far_plane if far_plane is not None else float(np.nanmax(depth))
    if near_plane == far_plane:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    depth = (depth - near_plane) / (far_plane - near_plane)
    depth = np.clip(depth, 0.0, 1.0)

    # apply colormap to depth
    cmap = colormaps[colormap]
    depth_rgb = (cmap(depth)[..., :3] * 255).astype(np.uint8)

    return depth_rgb

# Example: python preproc/compute_depths_pi3.py --img_dir /path/to/DAVIS/JPEGImages/480p/horsejump-high --out_dir ./outputs/test_pi3/horsejump-high --ckpt preproc/checkpoints/model.safetensors --vis --apply_mask --save_confs --save_points
if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the input video folder (images directory)")
    parser.add_argument('--out_dir', type=str, default='./outputs/davis', help="Path to the output folder")
    parser.add_argument('--apply_mask', action='store_true', help='apply confidence masks to depths')
    parser.add_argument('--save_confs', action='store_true', help='save conf prediction')
    parser.add_argument('--save_points', action='store_true', help='save global point clouds')
    parser.add_argument("--interval", type=int, default=1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/model.safetensors",
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument('--vis', action='store_true', help='save depth/conf visualizations')

    args = parser.parse_args()

    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file

            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)

        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # Inference single video
    data_path = args.img_dir
    video_name = os.path.basename(os.path.normpath(data_path))

    # Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(data_path, interval=args.interval).to(device)  # (N, 3, H, W)
    frame_names = sorted([x for x in os.listdir(data_path) if x.endswith(('.png', '.jpg'))])

    # Infer
    print("=" * 20)
    print(f"Running model inference: {video_name}")
    print("=" * 20)
    start_time = time.time()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])  # Add batch dimension
    end_time = time.time()
    print(
        f"Inference time: {end_time - start_time:.4f}, "
        f"number of frames: {len(imgs)}, "
        f"FPS: {len(imgs) / (end_time - start_time):.4f}"
    )

    # process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0].cpu().numpy()

    # save depths
    out_dir = os.path.join(args.out_dir, "depths")
    os.makedirs(out_dir, exist_ok=True)
    if args.vis:
        vis_dir = os.path.join(args.out_dir, "depths", "vis")
        os.makedirs(vis_dir, exist_ok=True)

    depths = res['local_points'][..., 2].squeeze(0).detach().cpu().numpy()
    depths_rgb = apply_depth_colormap(depths) if args.vis else None
    for i, frame_name in enumerate(frame_names):
        np.save(os.path.join(out_dir, os.path.splitext(frame_name)[0] + ".npy"), depths[i])
        if args.vis:
            iio.imwrite(os.path.join(vis_dir, os.path.splitext(frame_name)[0] + ".jpg"), depths_rgb[i])

    # save masked depths and masks
    if args.apply_mask:
        masks_dir = os.path.join(args.out_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        if args.vis:
            vis_dir = os.path.join(args.out_dir, "depths", "vis_masked")
            os.makedirs(vis_dir, exist_ok=True)

        depths_masked = depths.copy()
        depths_masked[~masks] = np.nan
        depths_rgb = apply_depth_colormap(depths_masked) if args.vis else None
        for i, frame_name in enumerate(frame_names):
            iio.imwrite(os.path.join(masks_dir, os.path.splitext(frame_name)[0] + ".png"), (masks[i] * 255).astype(np.uint8))
            if args.vis:
                iio.imwrite(os.path.join(vis_dir, os.path.splitext(frame_name)[0] + ".jpg"), depths_rgb[i])

    # save conf
    if args.save_confs:
        conf_dir = os.path.join(args.out_dir, "conf")
        os.makedirs(conf_dir, exist_ok=True)
        if args.vis:
            conf_vis_dir = os.path.join(args.out_dir, "conf", "vis")
            os.makedirs(conf_vis_dir, exist_ok=True)

        confs = res['conf'][..., 0].squeeze(0).detach().cpu().numpy()
        confs_rgb = apply_depth_colormap(confs) if args.vis else None
        for i, frame_name in enumerate(frame_names):
            np.save(os.path.join(conf_dir, os.path.splitext(frame_name)[0] + ".npy"), confs[i])
            if args.vis:
                iio.imwrite(os.path.join(conf_vis_dir, os.path.splitext(frame_name)[0] + ".jpg"), confs_rgb[i])

    # save point clouds
    if args.save_points:
        points_dir = os.path.join(args.out_dir, "points")
        os.makedirs(points_dir, exist_ok=True)
        points = res["points"].squeeze(0).detach().cpu().numpy()  # (T, H, W, 3)
        for i, frame_name in enumerate(frame_names):
            write_ply(
                points[i],
                imgs[i].permute(1, 2, 0),
                os.path.join(points_dir, os.path.splitext(frame_name)[0] + ".ply"),
            )

        if args.apply_mask:
            points_dir = os.path.join(args.out_dir, "points_masked")
            os.makedirs(points_dir, exist_ok=True)
            for i, frame_name in enumerate(frame_names):
                write_ply(
                    points[i][masks[i]],
                    imgs[i].permute(1, 2, 0)[masks[i]],
                    os.path.join(points_dir, os.path.splitext(frame_name)[0] + ".ply"),
                )

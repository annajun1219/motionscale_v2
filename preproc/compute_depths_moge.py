import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import click
import numpy as np
from matplotlib import colormaps
from functools import partial
import imageio.v2 as iio


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


@click.command(help='Inference script')
@click.option('--input', '-i', 'input_path', default='path/to/images', help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
@click.option('--output', '-o', 'output_path', default='./outputs/moge', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='path/to/model.pt', help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2']), default='v2', help='Model version. Defaults to "v2"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for much faster inference.')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--threshold', type=float, default=0.04, help='Threshold for removing edges. Defaults to 0.01. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--save_disp', 'save_disp_', is_flag=True, help='Save disp map.')
@click.option('--save_masks', 'save_masks_', is_flag=True, help='Save masks.')
@click.option('--save_points', 'save_points_', is_flag=True, help='Save point maps (H, W, 3) as npy.')
@click.option('--vis', 'save_vis_', is_flag=True, help='Save depth/disp/normal visualizations.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    fov_x_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    threshold: float,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    save_disp_: bool,
    save_masks_: bool,
    save_points_: bool,
    save_vis_: bool,
    show: bool,
):  
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    import trimesh

    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
    import utils3d

    device = torch.device(device_name)

    # Load model
    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16:
        model.half()

    # Inference single video
    print(f"Processing {input_path}")
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    image_names = [os.path.basename(os.path.normpath(image_path)) for image_path in image_paths]

    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    # process images
    all_preds = []
    all_images = []
    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        # Inference
        output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else np.zeros_like(points)
        all_preds.append((points, depth, mask, intrinsics, normal))
        all_images.append(image)

    # collect outputs
    all_points, all_depths, all_masks, all_intrinsics, all_normals = map(partial(np.stack, axis=0), zip(*all_preds))

    # save depths
    out_dir = os.path.join(output_path, "depths")
    os.makedirs(out_dir, exist_ok=True)
    if save_vis_:
        vis_dir = os.path.join(output_path, "depths", "vis")
        os.makedirs(vis_dir, exist_ok=True)

    depths_rgb = apply_depth_colormap(all_depths) if save_vis_ else None
    for i in range(len(image_paths)):
        frame_name = image_names[i]
        out_path = os.path.join(out_dir, frame_name.replace('.jpg', '.npy').replace('.png', '.npy'))
        np.save(out_path, all_depths[i])
        if save_vis_:
            iio.imwrite(os.path.join(vis_dir, frame_name.replace(".png", ".jpg")), depths_rgb[i])

    # save disp
    if save_disp_:
        out_dir = os.path.join(output_path, "disp")
        os.makedirs(out_dir, exist_ok=True)
        if save_vis_:
            vis_dir = os.path.join(output_path, "disp", "vis")
            os.makedirs(vis_dir, exist_ok=True)

        all_disp = 1.0 / np.clip(all_depths, a_min=1e-6, a_max=None)
        disp_rgb = apply_depth_colormap(all_disp) if save_vis_ else None
        for i in range(len(image_paths)):
            frame_name = image_names[i]
            out_path = os.path.join(out_dir, frame_name.replace('.jpg', '.npy').replace('.png', '.npy'))
            np.save(out_path, all_disp[i])
            if save_vis_:
                iio.imwrite(os.path.join(vis_dir, frame_name.replace(".png", ".jpg")), disp_rgb[i])

    # save masks
    if save_masks_:
        mask_dir = os.path.join(output_path, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(len(image_paths)):
            frame_name = image_names[i]
            cv2.imwrite(os.path.join(mask_dir, frame_name.replace(".jpg", ".png")), (all_masks[i] * 255).astype(np.uint8))

    # save intrinsics
    np.save(os.path.join(output_path, "intrinsics.npy"), all_intrinsics)

    # save normals
    out_dir = os.path.join(output_path, "normals")
    os.makedirs(out_dir, exist_ok=True)
    if save_vis_:
        vis_dir = os.path.join(output_path, "normals", "vis")
        os.makedirs(vis_dir, exist_ok=True)

    for i in range(len(image_paths)):
        frame_name = image_names[i]
        out_path = os.path.join(out_dir, frame_name.replace('.jpg', '.npy').replace('.png', '.npy'))
        np.save(out_path, all_normals[i])
        if save_vis_:
            cv2.imwrite(os.path.join(vis_dir, frame_name.replace(".jpg", ".png")), cv2.cvtColor(colorize_normal(all_normals[i]), cv2.COLOR_RGB2BGR))

    # save points
    if save_points_:
        out_dir = os.path.join(output_path, "points")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(image_paths)):
            frame_name = image_names[i]
            out_path = os.path.join(out_dir, frame_name.replace('.jpg', '.ply').replace('.png', '.ply'))
            vertices = all_points[i][all_masks[i]]
            colors = all_images[i][all_masks[i]]
            trimesh.PointCloud(vertices, colors=colors).export(out_path)


# Example: python preproc/compute_depths_moge.py -i /path/to/DAVIS/JPEGImages/480p/horsejump-high -o ./outputs/test_moge/horsejump-high --pretrained preproc/checkpoints/moge-2-vitl-normal/model.pt --vis --save_masks --save_points
if __name__ == '__main__':
    main()

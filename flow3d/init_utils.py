import time
from typing import Literal

import cupy as cp
import imageio.v3 as iio
import numpy as np
import open3d as o3d

# from pytorch3d.ops import sample_farthest_points
import roma
import torch
import torch.nn.functional as F
from cuml import HDBSCAN, KMeans
from loguru import logger as guru
from matplotlib.pyplot import get_cmap
from tqdm import tqdm
from viser import ViserServer

from flow3d.loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    get_weights_for_procrustes,
    knn,
    knn_query,
    masked_l1_loss,
    compute_se3_reg_loss,
    center_to_cluster_mean_loss,
)
from flow3d.params import GaussianParams, MotionBases, ScalableMotionBases, CameraPoses
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.transforms import cont_6d_to_rmat, rt_to_mat4, solve_procrustes, rmat_to_cont_6d
from flow3d.vis.utils import draw_keypoints_video, get_server, project_2d_tracks
from flow3d.data import BaseDataset
from flow3d.data.utils import to_device


def init_trainable_poses(w2cs) -> CameraPoses:
    N = w2cs.shape[0]
    R = w2cs[:, :3, :3]
    t = w2cs[:, :3, 3:]
    R6d = R[:, :, :2].clone()
    return CameraPoses(R6d, t.clone())


def init_new_camera_poses(trainer, train_dataset, new_frames: int, k: int = 5) -> CameraPoses:
    """
    Initialize camera poses for new frames using the averaged relative correction
    from the last k existing frames. Rotations are averaged exactly via SVD projection
    onto SO(3); translations are averaged linearly.
    """
    num_existing = trainer.model.camera_poses.params["Rs"].shape[0]
    device = trainer.model.camera_poses.params["Rs"].device
    K = min(num_existing, k)

    all_w2cs = train_dataset.get_w2cs().to(device)

    # compute relative corrections for the last K frames, T_delta = inv(T_original) @ T_updated
    updated = trainer.model.camera_poses.get_camera_matrix()[num_existing - K:]  # (K, 4, 4)
    original = all_w2cs[num_existing - K: num_existing]  # (K, 4, 4)
    deltas = torch.linalg.inv(original) @ updated

    # average translation linearly
    t_avg = deltas[:, :3, 3].mean(dim=0)  # (3,)

    # average rotation via Lie algebra (rotvec)
    rotvecs = roma.rotmat_to_rotvec(deltas[:, :3, :3])  # (K, 3)
    rotvec_avg = rotvecs.mean(dim=0)  # (3,)
    R_avg = roma.rotvec_to_rotmat(rotvec_avg)  # (3, 3)

    # reconstruct average delta as 4x4
    delta = torch.eye(4, device=device)
    delta[:3, :3] = R_avg
    delta[:3, 3] = t_avg

    # apply to new frame original poses: T_new_corrected = T_new_original @ T_delta
    new_w2cs = all_w2cs[num_existing: num_existing + new_frames]
    new_w2cs_corrected = new_w2cs @ delta.unsqueeze(0)  # (new_frames, 4, 4)

    return init_trainable_poses(new_w2cs_corrected)


def init_new_bases(bases, new_frames):
    """
    Initialize new bases by taking the last frame transformation and repeating for new frames.
    """
    if isinstance(bases, MotionBases):
        args = {}
        for name, x in bases.params.items():
            x_new = torch.repeat_interleave(x[..., -1:, :].detach(), new_frames, dim=-2)
            args[name] = x_new
        new_bases = MotionBases(**args)
    elif isinstance(bases, ScalableMotionBases):
        args = {}
        for name, x in bases.params.items():
            if x.ndim > 2:
                x_new = torch.repeat_interleave(x[..., -1:, :].detach(), new_frames, dim=-2)
            else:
                x_new = x.detach().clone()
            args[name] = x_new
        new_bases = ScalableMotionBases(**args)
    else:
        raise ValueError(f"Motion base type {type(bases)} not supported")

    return new_bases


def optim_new_bases_by_velocity(bases, model, num_iters=100, damping_factor=0.85, win_size=5, use_rot_loss=False):
    """
    Optimize the new bases by estimating the velocity of gaussians.
    """
    num_frames = model.motion_bases.num_frames
    new_num_frames = bases.num_frames
    if num_frames < 2:
        raise ValueError(f"Old bases must have at least 2 frames")

    device = model.motion_bases.params["rots"].device
    win_size = min(num_frames, win_size)
    ts = torch.arange(num_frames - win_size, num_frames, device=device)

    # Estimate target positions from velocity
    with torch.no_grad():
        # get gaussian means from bases
        coefs = model.fg.get_coefs()
        cluster_ids = model.fg.get_cluster_ids()
        transfms = model.motion_bases.compute_transforms(ts, coefs, cluster_ids)  # (G, B, 3, 4)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(model.fg.params["means"], (0, 1), value=1.0),
        )  # (G, B, 3)

        ## Average velocity for target positions
        p_last = positions[:, -1]  # (G, 3)

        # calculate the average velocity from recent frames
        velocities = positions[:, 1:] - positions[:, :-1]  # (G, B-1, 3)
        smooth_vel = velocities.mean(dim=1)  # (G, 3)
        smooth_vel = smooth_vel * damping_factor

        # extrapolation for new frames, P_target = P_last + (Smooth_Velocity * step)
        steps = torch.arange(1, new_num_frames + 1, device=device).view(1, -1, 1)  # (1, B, 1)
        target_positions = p_last.unsqueeze(1) + smooth_vel.unsqueeze(1) * steps  # (G, B, 3)

        ## Average rotation velocity for target rotations
        if use_rot_loss:
            r_mats = transfms[..., :3, :3]  # (G, B, 3, 3)
            r_curr = r_mats[:, 1:]  # (G, B-1, 3, 3)
            r_prev = r_mats[:, :-1]  # (G, B-1, 3, 3)

            # calculate rotational velocities, R_delta = R_curr @ R_prev^T
            r_deltas = r_curr @ r_prev.transpose(-1, -2)  # (G, B-1, 3, 3)

            # linear average
            r_delta_avg = r_deltas.mean(dim=1)  # (G, 3, 3)

            # linear damping, blend with identity
            identity = torch.eye(3, device=device).unsqueeze(0).expand_as(r_delta_avg)
            r_delta_blended = (r_delta_avg * damping_factor) + (identity * (1.0 - damping_factor))

            # SVD to ensure rigid rotation
            u, _, vh = torch.linalg.svd(r_delta_blended)
            r_delta_damped = u @ vh

            # Ensure it's a rotation (det = 1), not a reflection (det = -1)
            det = torch.linalg.det(r_delta_damped)
            reflection_mask = det < 0
            if reflection_mask.any():
                u[reflection_mask, :, -1] *= -1
                r_delta_damped = u @ vh

            # extrapolate iteratively for multiple new frames
            target_rots_list = []
            current_rot = r_mats[:, -1]
            for _ in range(new_num_frames):
                current_rot = r_delta_damped @ current_rot
                target_rots_list.append(current_rot)

            target_rots = torch.stack(target_rots_list, dim=1)  # (G, B, 3, 3)

    # Define optimizer
    if isinstance(bases, MotionBases):
        params_list = [
            {"params": bases.params["rots"], "lr": 1e-2},
            {"params": bases.params["transls"], "lr": 3e-2}
        ]
    elif isinstance(bases, ScalableMotionBases):
        params_list = [
            {"params": bases.params["rots"], "lr": 1e-2},
            {"params": bases.params["transls"], "lr": 3e-2},
            {"params": bases.params["fine_rots"], "lr": 1e-2},
            {"params": bases.params["fine_transls"], "lr": 3e-2},
        ]
    else:
        raise ValueError(f"Bases must be a MotionBases or ScalableMotionBases, but got {bases}")

    optimizer = torch.optim.Adam(params_list)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / num_iters))

    # Optimization
    coefs = model.fg.get_coefs().detach()
    cluster_ids = model.fg.get_cluster_ids().detach()
    means_padded = F.pad(model.fg.params["means"].detach(), (0, 1), value=1.0)
    new_ts = torch.arange(new_num_frames, device=device)

    # freeze fine trans
    bases.params["fine_rots"].requires_grad_(False)
    bases.params["fine_transls"].requires_grad_(False)

    pbar = tqdm(range(num_iters), leave=False)
    for i in pbar:
        # forward pass for the new frames
        transfms = bases.compute_transforms(new_ts, coefs, cluster_ids)  # (G, B, 3, 4)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            means_padded,
        )  # (G, B, 3)

        # MSE Loss against the extrapolated momentum
        loss = torch.tensor(0.0, device=device)
        pos_loss = F.mse_loss(positions, target_positions.detach())
        loss += pos_loss

        # 2. Rotational Loss
        rot_loss = torch.tensor(0.0, device=device)
        if use_rot_loss:
            new_rots = transfms[..., :3, :3]  # (G, B, 3, 3)
            rot_loss = F.mse_loss(new_rots, target_rots.detach())
            loss += rot_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"{loss.item():.8f} "
            f"{pos_loss.item():.8f} "
            f"{rot_loss.item():.8f} "
        )
    
    # unfreeze fine trans
    bases.params["fine_rots"].requires_grad_(True)
    bases.params["fine_transls"].requires_grad_(True)


def init_fg_from_tracks_3d(
    cano_t: int, tracks_3d: TrackObservations, motion_coefs: torch.Tensor, cluster_ids=None
) -> GaussianParams:
    """
    using dataclasses individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_fg = tracks_3d.xyz.shape[0]

    # Initialize gaussian colors.
    colors = torch.logit(tracks_3d.colors)
    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(tracks_3d.xyz[:, cano_t], 3)
    dists = torch.from_numpy(dists)
    scales = dists.mean(dim=-1, keepdim=True)
    scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
    scales = torch.log(scales.repeat(1, 3))
    # Initialize gaussian means.
    means = tracks_3d.xyz[:, cano_t]
    # Initialize gaussian orientations as random.
    quats = torch.rand(num_fg, 4)
    # Initialize gaussian opacities.
    opacities = torch.logit(torch.full((num_fg,), 0.7))
    # cluster indices
    cluster_ids = cluster_ids.to(torch.int64)

    # Gaussian init
    gaussians = GaussianParams(
        means, quats, scales, colors, opacities, motion_coefs,
        cluster_ids=cluster_ids,
    )

    return gaussians


def init_bg(
    points: StaticObservations, clamp_scale=True, init_scales=None,
) -> GaussianParams:
    """
    using dataclasses instead of individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_init_bg_gaussians = points.xyz.shape[0]
    bg_scene_center = points.xyz.mean(0)
    bg_points_centered = points.xyz - bg_scene_center
    bg_min_scale = bg_points_centered.quantile(0.05, dim=0)
    bg_max_scale = bg_points_centered.quantile(0.95, dim=0)
    bg_scene_scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0
    bkdg_colors = torch.logit(points.colors)

    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    if init_scales is not None:
        bkdg_scales = init_scales.clamp(torch.quantile(init_scales, 0.05), torch.quantile(init_scales, 0.95))
    else:
        dists, _ = knn(points.xyz, 3)
        dists = torch.from_numpy(dists)
        bg_scales = dists.mean(dim=-1, keepdim=True)
        if clamp_scale:
            bg_scales = bg_scales.clamp(torch.quantile(bg_scales, 0.05), torch.quantile(bg_scales, 0.95))
        bkdg_scales = torch.log(bg_scales.repeat(1, 3))

    bg_means = points.xyz

    # Initialize gaussian orientations by normals
    local_normals = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(
        points.normals
    )
    bg_quats = roma.rotvec_to_unitquat(
        F.normalize(local_normals.cross(points.normals), dim=-1)
        * (local_normals * points.normals).sum(-1, keepdim=True).acos_()
    ).roll(1, dims=-1)
    bg_opacities = torch.logit(torch.full((num_init_bg_gaussians,), 0.7))

    gaussians = GaussianParams(
        bg_means,
        bg_quats,
        bkdg_scales,
        bkdg_colors,
        bg_opacities,
        scene_center=bg_scene_center,
        scene_scale=bg_scene_scale,
    )
    return gaussians


def init_shad_motion(
    points: StaticObservations,
    num_bases: int,
    rot_type: Literal["quat", "6d"],
    cano_t: int,
    cluster_init_type: str = "means",
    cluster_init_method: str = "kmeans",
    bases_type: str = "global",
    num_fine_bases: int = 5,
    coefs_type: str = "squared",
    coefs_sigma: float = 0.6,
):
    num_frames = len(points.xyz)
    num_points = points.xyz[cano_t].shape[0]
    device = points.xyz[cano_t].device

    ## Initialize transformation by ICP
    # estimate a scene scale for ICP distance
    all_points = torch.cat(points.xyz, dim=0)
    center = all_points.mean(0)
    dists = (all_points - center).norm(dim=1)
    scale = torch.quantile(dists, 0.98)
    max_corr_dist = 0.05 * scale

    # run ICP to get rough transformations
    pcd_cano = o3d.geometry.PointCloud()
    pcd_cano.points = o3d.utility.Vector3dVector(points.xyz[cano_t].numpy())
    transformations = []
    init_rots, init_ts = [], []

    for i in range(num_frames):
        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(points.xyz[i].numpy())

        if i == cano_t:
            trans = torch.eye(4)
        else:
            # Perform ICP registration
            result = o3d.pipelines.registration.registration_icp(
                source=pcd_cano,
                target=pcd_t,
                max_correspondence_distance=max_corr_dist,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            trans = torch.from_numpy(result.transformation).float()
            print(f"Frame {i}: fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.3f}")

        init_rots.append(rmat_to_cont_6d(trans[:3, :3]))
        init_ts.append(trans[:3, 3])

    init_rots = torch.stack(init_rots, dim=0).unsqueeze(0).repeat(num_bases, 1, 1)  # (N, T, 6)
    init_ts = torch.stack(init_ts, dim=0).unsqueeze(0).repeat(num_bases, 1, 1)  # (N, T, 3)

    ## Initialize motion bases
    # initialize clusters
    means_cano = points.xyz[cano_t]
    sampled_centers, num_bases, labels = sample_bases_centers_by_means(
        cluster_init_method, means_cano, num_bases
    )

    ids, counts = labels.unique(return_counts=True)
    print(f"{ids=}, {counts=}")

    # compute basis weights from the distance to the cluster centers
    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
    motion_coefs = initialize_coefs(dists2centers, coefs_type, coefs_sigma)

    # create motion bases
    if bases_type == "global":
        bases = MotionBases(init_rots, init_ts)
    elif bases_type == "scalable":
        # initialize motion coefficients
        motion_coefs = motion_coefs.new_zeros(motion_coefs.shape[0], num_fine_bases)
        for cluster_id in ids:
            mask_in_cluster = labels == cluster_id

            def normalize_pc(pc):
                centroid = pc.mean(dim=0)
                pc_centered = pc - centroid
                mean_dist = pc_centered.norm(dim=1).mean()
                pc_normalized = pc_centered / mean_dist
                return pc_normalized

            # initialize fine bases centers by the point cloud
            means_cluster = normalize_pc(means_cano[mask_in_cluster])
            if means_cluster.shape[0] < num_fine_bases:
                # too few points to subdivide; uniform coefs (after softmax -> 1/num_fine_bases)
                motion_coefs[mask_in_cluster] = 0.0
            else:
                fine_centers, _, fine_labels = sample_bases_centers_by_means(
                    cluster_init_method, means_cluster, num_fine_bases
                )
                dists2centers = torch.norm(means_cluster[:, None] - fine_centers, dim=-1)
                coefs = initialize_coefs(dists2centers, coefs_type, coefs_sigma)
                motion_coefs[mask_in_cluster] = coefs

        # initialize bases
        R_global = cont_6d_to_rmat(init_rots)  # (C, T, 3, 3)
        base_centers = sampled_centers.squeeze(0)  # (C, 3)
        t_local = init_ts - base_centers[:, None] + torch.einsum("ntij,nj->nti", R_global, base_centers)  # (C, T, 3)

        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        fine_rots = id_rot.reshape(1, 1, 1, -1).repeat(num_bases, num_fine_bases, num_frames, 1)
        fine_ts = torch.zeros(num_bases, num_fine_bases, num_frames, 3, device=device)
        bases = ScalableMotionBases(
            centers=base_centers,
            rots=init_rots,
            transls=t_local,
            fine_rots=fine_rots,
            fine_transls=fine_ts,
        )
    else:
        raise NotImplementedError

    # remap labels to 0->num_bases-1
    _, cluster_ids = labels.unique(return_inverse=True)

    return bases, motion_coefs, cluster_ids


def init_shad_params(
    cano_t: int, points: StaticObservations, motion_coefs: torch.Tensor, cluster_ids=None
) -> GaussianParams:
    num_points = points.xyz[cano_t].shape[0]

    # Initialize gaussian colors.
    colors = torch.logit(points.colors[cano_t])

    # Initialize gaussian scales
    dists, _ = knn(points.xyz[cano_t], 3)
    dists = torch.from_numpy(dists)
    scales = dists.mean(dim=-1, keepdim=True)
    scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
    scales = torch.log(scales.repeat(1, 3))

    # Initialize gaussian means.
    means = points.xyz[cano_t]

    # Initialize gaussian orientations by normals
    point_normals = points.normals[cano_t]
    local_normals = point_normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(point_normals)
    quats = roma.rotvec_to_unitquat(
        F.normalize(local_normals.cross(point_normals), dim=-1)
        * (local_normals * point_normals).sum(-1, keepdim=True).acos_()
    ).roll(1, dims=-1)

    # Initialize gaussian opacities.
    opacities = torch.logit(torch.full((num_points,), 0.7))

    # cluster indices
    cluster_ids = cluster_ids.to(torch.int64)

    # Gaussian init
    gaussians = GaussianParams(
        means, quats, scales, colors, opacities, motion_coefs, cluster_ids=cluster_ids,
    )

    return gaussians


def init_motion_params_with_procrustes(
    tracks_3d: TrackObservations,
    num_bases: int,
    rot_type: Literal["quat", "6d"],
    cano_t: int,
    cluster_init_type: str = "means",
    cluster_init_method: str = "kmeans",
    min_mean_weight: float = 0.1,
    bases_type: str = "global",
    num_fine_bases: int = 5,
    coefs_type: str = "squared",
    coefs_sigma: float = 0.6,
    vis: bool = False,
    port: int | None = None,
) -> tuple[MotionBases, torch.Tensor, TrackObservations]:
    """
    Sample centers and get initial se3 motion bases by solving procrustes
    """
    device = tracks_3d.xyz.device
    num_frames = tracks_3d.xyz.shape[1]

    # select, for all the tracks, their 3d positions on the cano frame
    means_cano = tracks_3d.xyz[:, cano_t].clone()  # (num_sampled_tracks, 3)

    # # remove outliers
    # scene_center = means_cano.median(dim=0).values
    # print(f"{scene_center=}")
    # dists = torch.norm(means_cano - scene_center, dim=-1)
    # dists_th = torch.quantile(dists, 0.95)
    # valid_mask = dists < dists_th
    #
    # # remove tracks that are not visible in any frame
    # valid_mask = valid_mask & tracks_3d.visibles.any(dim=1)

    # filter tracks again, use only track points that are visible on the cano frame
    valid_mask = tracks_3d.visibles[:, cano_t]
    # valid_mask = tracks_3d.visibles.any(dim=1)
    print(f"{valid_mask.sum()=}")

    # filter invalid tracks for all fields
    tracks_3d = tracks_3d.filter_valid(valid_mask)

    if vis and port is not None:
        server = get_server(port)
        try:
            pts = tracks_3d.xyz.cpu().numpy()
            clrs = tracks_3d.colors.cpu().numpy()
            while True:
                for t in range(num_frames):
                    server.scene.add_point_cloud("points", pts[:, t], clrs)
                    time.sleep(0.3)
        except KeyboardInterrupt:
            pass

    means_cano = means_cano[valid_mask]

    # initialize clusters and centers
    if cluster_init_type == "means":
        sampled_centers, num_bases, labels = sample_bases_centers_by_means(
            cluster_init_method, means_cano, num_bases
        )
    elif cluster_init_type == "velocities":
        sampled_centers, num_bases, labels = sample_initial_bases_centers(
            cluster_init_method, cano_t, tracks_3d, num_bases
        )
    else:
        raise ValueError(f"Invalid cluster_init_type: {cluster_init_type}")

    ids, counts = labels.unique(return_counts=True)
    print(f"{num_bases=}, {sampled_centers.shape=}")
    print(f"{ids=}, {counts=}")

    # compute basis weights from the distance to the cluster centers
    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
    motion_coefs = initialize_coefs(dists2centers, coefs_type, coefs_sigma)

    init_rots, init_ts = [], []

    if rot_type == "quat":
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        rot_dim = 4
    else:
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        rot_dim = 6

    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
    errs_before = np.full((num_bases, num_frames), -1.0)
    errs_after = np.full((num_bases, num_frames), -1.0)

    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    print(f"{tgt_ts=}")
    skipped_ts = {}
    for n, cluster_id in enumerate(ids):
        mask_in_cluster = labels == cluster_id
        cluster = tracks_3d.xyz[mask_in_cluster].transpose(
            0, 1
        )  # [num_frames, n_pts, 3]
        visibilities = tracks_3d.visibles[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        confidences = tracks_3d.confidences[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        weights = get_weights_for_procrustes(cluster, visibilities)
        prev_t = cano_t
        cluster_skip_ts = []
        for cur_t in tgt_ts:
            # compute pairwise transform from cano_t
            procrustes_weights = (
                weights[cano_t]
                * weights[cur_t]
                * (confidences[cano_t] + confidences[cur_t])
                / 2
            )
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
                cluster_skip_ts.append(cur_t)
            else:
                se3, (err, err_before) = solve_procrustes(
                    cluster[cano_t],
                    cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type,
                )
                init_rot, init_t, _ = se3
                assert init_rot.shape[-1] == rot_dim
                # double cover
                if rot_type == "quat" and torch.linalg.norm(
                    init_rot - init_rots[n][prev_t]
                ) > torch.linalg.norm(-init_rot - init_rots[n][prev_t]):
                    init_rot = -init_rot
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
                if err == np.nan:
                    print(f"{cur_t=} {err=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                if err_before == np.nan:
                    print(f"{cur_t=} {err_before=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                errs_after[n, cur_t] = err
                errs_before[n, cur_t] = err_before
            prev_t = cur_t
        skipped_ts[cluster_id.item()] = cluster_skip_ts

    guru.info(f"{skipped_ts=}")
    guru.info(
        "procrustes init median error: {:.5f} => {:.5f}".format(
            np.median(errs_before[errs_before > 0]),
            np.median(errs_after[errs_after > 0]),
        )
    )
    guru.info(
        "procrustes init mean error: {:.5f} => {:.5f}".format(
            np.mean(errs_before[errs_before > 0]), np.mean(errs_after[errs_after > 0])
        )
    )
    guru.info(f"{init_rots.shape=}, {init_ts.shape=}, {motion_coefs.shape=}")

    if vis:
        server = get_server(port)
        center_idcs = torch.argmin(dists2centers, dim=0)
        print(f"{dists2centers.shape=} {center_idcs.shape=}")
        vis_se3_init_3d(server, init_rots, init_ts, means_cano[center_idcs])
        vis_tracks_3d(server, tracks_3d.xyz[center_idcs].numpy(), name="center_tracks")

    # # initialize cluster indices by distance
    # # (this is inconsistent with the clusters in procrustes, and some cluster may have no points)
    # labels = torch.argmin(dists2centers, dim=-1)

    # create motion bases
    if bases_type == "global":
        bases = MotionBases(init_rots, init_ts)
    elif bases_type == "scalable":
        # initialize motion coefficients
        motion_coefs = motion_coefs.new_zeros(motion_coefs.shape[0], num_fine_bases)
        for cluster_id in ids:
            mask_in_cluster = labels == cluster_id

            def normalize_pc(pc):
                centroid = pc.mean(dim=0)
                pc_centered = pc - centroid
                mean_dist = pc_centered.norm(dim=1).mean()
                pc_normalized = pc_centered / mean_dist
                return pc_normalized

            # initialize fine bases centers by the point cloud
            means_cluster = normalize_pc(means_cano[mask_in_cluster])
            if means_cluster.shape[0] < num_fine_bases:
                # too few points to subdivide; uniform coefs (after softmax -> 1/num_fine_bases)
                motion_coefs[mask_in_cluster] = 0.0
            else:
                fine_centers, _, fine_labels = sample_bases_centers_by_means(
                    cluster_init_method, means_cluster, num_fine_bases
                )
                dists2centers = torch.norm(means_cluster[:, None] - fine_centers, dim=-1)

                # initialize coefs
                coefs = initialize_coefs(dists2centers, coefs_type, coefs_sigma)
                motion_coefs[mask_in_cluster] = coefs

        # initialize bases
        R_global = cont_6d_to_rmat(init_rots)  # (C, T, 3, 3)
        base_centers = sampled_centers.squeeze(0)  # (C, 3)
        t_local = init_ts - base_centers[:, None] + torch.einsum("ntij,nj->nti", R_global, base_centers)  # (C, T, 3)

        fine_rots = id_rot.reshape(1, 1, 1, rot_dim).repeat(num_bases, num_fine_bases, num_frames, 1)
        fine_ts = torch.zeros(num_bases, num_fine_bases, num_frames, 3, device=device)
        bases = ScalableMotionBases(
            centers=base_centers,
            rots=init_rots,
            transls=t_local,
            fine_rots=fine_rots,
            fine_transls=fine_ts,
        )
    else:
        raise NotImplementedError

    # remap labels to 0->num_bases-1
    _, cluster_ids = labels.unique(return_inverse=True)

    return bases, motion_coefs, tracks_3d, cluster_ids


def ridge_solve_bases(C: torch.Tensor, X: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """
    Ridge regression to solve bases B:
      argmin_B || C B - X ||^2 + lam ||B||^2
    C: (N,F)
    X: (N,D)
    returns B: (F,D)
    """
    F = C.shape[1]
    CtC = C.t() @ C
    CtX = C.t() @ X

    # Scale lambda relative to the actual matrix values
    max_diag = CtC.diag().max().clamp(min=1e-6)
    adaptive_lam = lam * max_diag

    return torch.linalg.solve(CtC + adaptive_lam * torch.eye(F, device=C.device, dtype=C.dtype), CtX)


def init_motion_params_for_split(
    model,
    new_clusters,
    cluster_ids_new,
    cluster_method: str = "kmeans",
    coefs_type: str = "squared",
    coefs_sigma: float = 0.6,
):
    """
    Initializes motion parameters when splitting clusters.
    Split one cluster into two and initialize parameters on the new set of gaussians.
    """
    means_cano = model.fg.params["means"].detach().cpu()
    cluster_ids_new = cluster_ids_new.cpu()
    device = means_cano.device
    num_frames = model.num_frames
    num_fine_bases = model.motion_bases.num_fine_bases
    num_new_clus = len(new_clusters)

    # existing gaussian transformations
    with torch.no_grad():
        means_fg = model.fg.params["means"]  # (G, 3)
        transfms = model.compute_transforms(torch.arange(0, num_frames, device=means_fg.device))  # (G, T, 3, 4)
        means_ts = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(means_fg, (0, 1), value=1.0),
        ).cpu()  # (G, T, 3)
        R_old = transfms[..., :3, :3].cpu()  # (G, T, 3, 3)
        t_old = transfms[..., :3, 3].cpu()  # (G, T, 3)

    ## Initialize parameters for new clusters
    id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    rot_dim = 6
    coarse_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_new_clus, num_frames, 1)
    coarse_ts = torch.zeros(num_new_clus, num_frames, 3, device=device)
    fine_rots = id_rot.reshape(1, 1, 1, rot_dim).repeat(num_new_clus, num_fine_bases, num_frames, 1)
    fine_ts = torch.zeros(num_new_clus, num_fine_bases, num_frames, 3, device=device)
    new_centers = torch.zeros(num_new_clus, 3, device=device)
    new_coefs = model.fg.params["motion_coefs"].detach().cpu().clone()

    for i, clus_id in enumerate(new_clusters):
        mask_in_cluster = cluster_ids_new == clus_id
        means_cluster = means_cano[mask_in_cluster]  # (G_c, 3)

        # 1) new center on cano frame for this new cluster
        new_center = means_cluster.median(dim=0).values  # (3,)
        new_centers[i] = new_center

        # 2) coarse transformation via procrustes
        proc_Rs, proc_ts = [], []
        src_points = means_cluster
        for t in range(num_frames):
            # solve rigid alignment
            dst_points = means_ts[mask_in_cluster, t]
            (R, t_vec, s), _ = solve_procrustes(
                src=src_points,
                dst=dst_points,
                weights=None,
                enforce_se3=True,
                rot_type="mat",
            )
            proc_Rs.append(R)
            proc_ts.append(t_vec)

        # Global procrustes R and t
        proc_Rs = torch.stack(proc_Rs, dim=0)  # (T, 3, 3)
        proc_ts = torch.stack(proc_ts, dim=0)  # (T, 3)

        # Convert to raw coarse R and t used in the motion bases
        # x_out = R_proc * x + T_proc = R_coarse * (x - center) + T_coarse + center
        # => R_coarse = R_proc, T_coarse = T_proc + R_coarse * center - center
        R_c = proc_Rs  # (T, 3, 3)
        t_c = proc_ts + torch.einsum("tij,j->ti", R_c, new_center) - new_center[None, :]  # (T, 3)
        coarse_rots[i] = rmat_to_cont_6d(R_c)
        coarse_ts[i] = t_c

        # 3) Estimate target fine R and t for solving
        # old per-gaussian transforms R and t
        R_total_old = R_old[mask_in_cluster]  # (N, T, 3, 3)
        t_total_old = t_old[mask_in_cluster]  # (N, T, 3)

        # Estimate target fine rotation
        # R_total_old = R_coarse * R_fine => R_fine = R_coarse^T * R_total_old
        R_c_T = R_c.transpose(-1, -2)  # (T, 3, 3)
        R_f = torch.einsum("tij,ntjk->ntik", R_c_T, R_total_old)  # (N, T, 3, 3)
        rot_f = rmat_to_cont_6d(R_f)  # (N, T, 6)

        # Fine translation forward pass:
        # T_total_old = -R_coarse * R_fine * center + R_coarse * T_fine + T_coarse + center
        # => T_fine = R_coarse^T * (T_total_old - T_coarse - center) + R_fine * center
        term1 = t_total_old - t_c[None, ...] - new_center[None, None, :]  # (N, T, 3)
        t_f = (
            torch.einsum("tij,ntj->nti", R_c_T, term1)
            + torch.einsum("ntij,j->nti", R_f, new_center)
        )  # (N, T, 3)

        # 4) Initialize motion coefs
        def normalize_pc(pc):
            centroid = pc.mean(dim=0)
            pc_centered = pc - centroid
            mean_dist = pc_centered.norm(dim=1).mean()
            pc_normalized = pc_centered / mean_dist
            return pc_normalized

        # assign coefs based on spatial distribution
        means_norm = normalize_pc(means_cluster)
        if means_norm.shape[0] < num_fine_bases:
            # too few points to subdivide; uniform coefs (after softmax -> 1/num_fine_bases)
            coefs = torch.zeros(means_norm.shape[0], num_fine_bases)
            new_coefs[mask_in_cluster] = 0.0
        else:
            fine_centers, _, _ = sample_bases_centers_by_means(cluster_method, means_norm, num_fine_bases)
            dists2centers = torch.norm(means_norm[:, None] - fine_centers, dim=-1)
            coefs = initialize_coefs(dists2centers, coefs_type, coefs_sigma)
            new_coefs[mask_in_cluster] = coefs

        # 5) solve bases: coefs @ bases ≈ target  (ridge)
        # Define rotation and translation targets, use the delta to identity
        delta_rot_f = rot_f - id_rot.view(1, 1, 6)  # (N, T, 6)
        target = torch.cat([delta_rot_f, t_f], dim=-1).reshape(rot_f.shape[0], num_frames * 9)  # (N, T*9)

        # ridge solve
        C = torch.softmax(coefs, dim=1)
        delta_B = ridge_solve_bases(C, target)  # (F, T*9)
        delta_B = delta_B.reshape(num_fine_bases, num_frames, 9)  # (F, T, 9)
        fine_rots[i] = delta_B[..., :6] + id_rot.view(1, 1, 6)
        fine_ts[i] = delta_B[..., 6:]

    new_bases = ScalableMotionBases(
        centers=new_centers,
        rots=coarse_rots,
        transls=coarse_ts,
        fine_rots=fine_rots,
        fine_transls=fine_ts,
    )

    return new_bases, new_coefs


def run_bg_optim(
    trainer,
    train_dataset,
    new_frames,
    num_iters: int = 300,
    optimize_poses: bool = False,
    new_poses: CameraPoses | None = None,
):
    """
    Optimize background for new frames.
    """
    bases = trainer.model.motion_bases
    num_frames = bases.num_frames
    device = bases.params["rots"].device
    collate_fn = BaseDataset.train_collate_fn

    # sample data range
    sample_end = num_frames + new_frames
    sample_ts = np.arange(num_frames, sample_end).tolist()

    # build batch data
    batch = []
    for t_ in sample_ts:
        batch.append(train_dataset[t_])
    batch = collate_fn(batch)
    batch = to_device(batch, device)

    # inject temp poses so get_camera_matrix() concatenates old.detach() + new
    if optimize_poses and new_poses is not None and trainer.model.camera_poses is not None:
        trainer.model.camera_poses.inject_temp_poses(new_poses)

    # save/restore requires_grad — freeze everything except bg, shad_bases, new poses
    orig_requires_grad = {}
    for name, param in trainer.model.named_parameters():
        orig_requires_grad[name] = param.requires_grad
        if (
            name.split(".")[0] in ("bg", "shad_bases")
            or ("temp_new_poses" in name and optimize_poses and new_poses is not None)
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # fresh optimizer for new frame poses only (no stale Adam state for old frames)
    poses_optimizer = None
    if optimize_poses and new_poses is not None and trainer.model.camera_poses is not None:
        poses_optimizer = torch.optim.Adam([
            {"params": new_poses.params["Rs"], "lr": 1e-4},
            {"params": new_poses.params["ts"], "lr": 1e-4},
        ])

    # optimizer
    optimizers = trainer.optimizers

    # run optim
    for i in (pbar := tqdm(range(0, num_iters))):
        # rgb on new frames
        loss, stats, _, _ = trainer.compute_bg_losses(batch)

        if loss.isnan():
            import ipdb
            ipdb.set_trace()

        loss.backward()

        for opt_name, opt in optimizers.items():
            if opt_name.split(".")[0] in ("bg", "shad_bases"):
                opt.step()
            opt.zero_grad(set_to_none=True)

        if poses_optimizer is not None:
            poses_optimizer.step()
            poses_optimizer.zero_grad(set_to_none=True)

        # for sched_name, sched in scheduler.items():
        #     if sched_name.split(".")[0] == "bg":
        #         sched.step()

        pbar.set_description(
            f"{loss.item():.4f} "
        )

    # restore requires_grad
    for name, param in trainer.model.named_parameters():
        param.requires_grad = orig_requires_grad[name]
    if trainer.model.camera_poses is not None:
        trainer.model.camera_poses.temp_new_poses = None


def run_motion_optim(
    trainer,
    train_dataset,
    old_frame_end,
    new_frames,
    win_size,
    num_iters: int = 100,
    fg_only: bool = True,
):
    """
    Optimize motion bases for new frames.
    """
    bases = trainer.model.motion_bases
    num_frames = bases.num_frames
    device = bases.params["rots"].device
    collate_fn = BaseDataset.train_collate_fn

    # sample data range
    sample_end = old_frame_end
    sample_start = max(0, sample_end - win_size)
    sample_ts = np.arange(sample_start, sample_end)
    sample_target_ts = np.arange(num_frames - new_frames, num_frames)

    # build batch data
    batch = []
    for sample_t in sample_ts:
        batch.append(train_dataset[(sample_t, sample_target_ts)])
    batch = collate_fn(batch)
    batch = to_device(batch, device)

    batch_new = []
    for t_ in sample_target_ts:
        batch_new.append(train_dataset[(t_, sample_ts)])
    batch_new = collate_fn(batch_new)
    batch_new = to_device(batch_new, device)

    # optimizer
    optimizers = trainer.optimizers
    scheduler = trainer.scheduler

    # freeze fine trans first
    trainer.model.motion_bases.params["fine_rots"].requires_grad_(False)
    trainer.model.motion_bases.params["fine_transls"].requires_grad_(False)

    # run optim
    for i in (pbar := tqdm(range(0, num_iters))):
        if i == num_iters // 2:
            trainer.model.motion_bases.params["fine_rots"].requires_grad_(True)
            trainer.model.motion_bases.params["fine_transls"].requires_grad_(True)

        # motion loss from old frames
        loss1, stats1, _, _ = trainer.compute_motion_losses(batch, fg_only=fg_only)

        # for k, v in stats1.items():
        #     trainer.writer.add_scalar(f"frame{num_frames}/{k}", v, i)

        # rgb, depth, mask loss on new frames
        loss2 = torch.tensor(0)
        if i >= num_iters // 2:
            # loss2, stats2, _, _ = trainer.compute_rgb_losses(batch_new)
            loss2, stats2, _, _ = trainer.compute_losses(batch_new)
            loss2 *= len(sample_target_ts) / len(sample_ts)

        loss = loss1 + loss2

        if loss.isnan():
            import ipdb
            ipdb.set_trace()

        loss.backward()

        for opt_name, opt in optimizers.items():
            if "motion_bases" in opt_name and "centers" not in opt_name:
                opt.step()
            opt.zero_grad(set_to_none=True)

        for sched_name, sched in scheduler.items():
            if "motion_bases" in sched_name:
                sched.step()

        pbar.set_description(
            f"{loss.item():.4f} "
            f"{loss1.item():.4f} "
            f"{loss2.item():.4f} "
        )

    trainer.model.motion_bases.params["fine_rots"].requires_grad_(True)
    trainer.model.motion_bases.params["fine_transls"].requires_grad_(True)

def run_initial_optim(
    fg: GaussianParams,
    bases: MotionBases,
    tracks_3d: TrackObservations,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    num_iters: int = 1000,
    use_depth_range_loss: bool = False,
):
    """
    :param motion_rots: [num_bases, num_frames, 4|6]
    :param motion_transls: [num_bases, num_frames, 3]
    :param motion_coefs: [num_bases, num_frames]
    :param means: [num_gaussians, 3]
    """
    # parameters to optimize
    if isinstance(bases, MotionBases):
        params_list = [
            {"params": bases.params["rots"], "lr": 1e-2},
            {"params": bases.params["transls"], "lr": 3e-2}
        ]
    elif isinstance(bases, ScalableMotionBases):
        params_list = [
            {"params": bases.params["centers"], "lr": 1e-3},
            {"params": bases.params["rots"], "lr": 1e-2},
            {"params": bases.params["transls"], "lr": 3e-2},
            {"params": bases.params["fine_rots"], "lr": 1e-2},
            {"params": bases.params["fine_transls"], "lr": 3e-2},
        ]
    else:
        raise ValueError(f"Bases must be a MotionBases or ScalableMotionBases, but got {bases}")
    params_list.extend([
        {"params": fg.params["motion_coefs"], "lr": 1e-2},
        {"params": fg.params["means"], "lr": 1e-3},
    ])

    optimizer = torch.optim.Adam(params_list)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_iters)
    )
    G = fg.params.means.shape[0]
    num_frames = bases.num_frames
    device = bases.params["rots"].device

    w_smooth_func = lambda i, min_v, max_v, th: (
        min_v if i <= th else (max_v - min_v) * (i - th) / (num_iters - th) + min_v
    )

    w_smooth_func2 = lambda i, start_v, end_v, s, e: (
        start_v if i <= s else
        end_v if i >= e else
        start_v + (end_v - start_v) * (i - s) / (e - s)
    )

    gt_2d, gt_depth = project_2d_tracks(
        tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs, return_depth=True
    )
    # (G, T, 2)
    gt_2d = gt_2d.swapaxes(0, 1)
    # (G, T)
    gt_depth = gt_depth.swapaxes(0, 1)

    ts = torch.arange(0, num_frames, device=device)
    ts_clamped = torch.clamp(ts, min=1, max=num_frames - 2)
    ts_neighbors = torch.cat((ts_clamped - 1, ts_clamped, ts_clamped + 1))  # i (3B,)

    # # freeze fine trans first
    # bases.params["fine_rots"].requires_grad_(False)
    # bases.params["fine_transls"].requires_grad_(False)

    pbar = tqdm(range(0, num_iters))
    for i in pbar:
        # if i == num_iters // 2:
        #     bases.params["fine_rots"].requires_grad_(True)
        #     bases.params["fine_transls"].requires_grad_(True)

        coefs = fg.get_coefs()
        cluster_ids = fg.get_cluster_ids()
        transfms = bases.compute_transforms(ts, coefs, cluster_ids)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(fg.params["means"], (0, 1), value=1.0),
        )

        transfms_coarse = bases.compute_transforms_coarse(ts, cluster_ids)
        pos_coarse = torch.einsum(
            "pnij,pj->pni",
            transfms_coarse,
            F.pad(fg.params["means"].detach(), (0, 1), value=1.0),
        )  # (G, T, 3)

        loss = 0.0
        track_3d_loss = masked_l1_loss(
            positions,
            tracks_3d.xyz,
            (tracks_3d.visibles.float() * tracks_3d.confidences)[..., None],
        )
        loss += track_3d_loss * 1.0

        pred_2d, pred_depth = project_2d_tracks(
            positions.swapaxes(0, 1), Ks, w2cs, return_depth=True
        )
        pred_2d = pred_2d.swapaxes(0, 1)
        pred_depth = pred_depth.swapaxes(0, 1)

        loss_2d = (
            masked_l1_loss(
                pred_2d,
                gt_2d,
                (tracks_3d.visibles.float() * tracks_3d.confidences)[..., None],
                quantile=0.95,
            )
            / Ks[0, 0, 0]
        )
        loss += 5.0 * loss_2d

        if use_depth_range_loss:
            near_depths = torch.quantile(gt_depth, 0.0, dim=0, keepdim=True)
            far_depths = torch.quantile(gt_depth, 0.98, dim=0, keepdim=True)
            loss_depth_in_range = 0
            if (pred_depth < near_depths).any():
                loss_depth_in_range += (near_depths - pred_depth)[
                    pred_depth < near_depths
                ].mean()
            if (pred_depth > far_depths).any():
                loss_depth_in_range += (pred_depth - far_depths)[
                    pred_depth > far_depths
                ].mean()

            loss += loss_depth_in_range * w_smooth_func(i, 0.05, 0.5, 400)

        motion_coef_sparse_loss = 1 - (coefs**2).sum(dim=-1).mean()
        loss += motion_coef_sparse_loss * 0.01

        # fine bases should be close to identity
        bases_reg_loss = torch.tensor(0.0)
        if "fine_rots" in bases.params:
            w_decay = w_smooth_func2(i, 0.1, 0.001, 0, 500)
            bases_reg_loss = compute_se3_reg_loss(bases.params["fine_rots"], bases.params["fine_transls"])
            loss += bases_reg_loss * w_decay

        # motion basis should be smooth.
        w_smooth = w_smooth_func(i, 0.01, 0.1, 400)
        small_acc_loss = compute_se3_smoothness_loss(bases.params["rots"], bases.params["transls"])
        if "fine_rots" in bases.params:
            small_acc_loss += compute_se3_smoothness_loss(bases.params["fine_rots"], bases.params["fine_transls"])
        loss += small_acc_loss * w_smooth

        small_acc_loss_tracks = compute_accel_loss(positions)
        loss += small_acc_loss_tracks * w_smooth * 0.5

        # regularize centers
        loss_center_cano = torch.tensor(0.0)
        loss_center_coarse = torch.tensor(0.0)
        if "centers" in bases.params:
            # align the centers in canonical frame
            loss_center_cano = center_to_cluster_mean_loss(
                fg.params["means"].detach(), bases.params["centers"], cluster_ids,
            )
            loss += loss_center_cano * 1.0

            # # align centers after coarse transformation
            # centers_ts = bases.get_centers(ts, freeze_centers=True)  # (C, T, 3)
            # loss_center_coarse = center_to_cluster_mean_loss(
            #     positions.detach(), centers_ts, cluster_ids, weights=None,
            # )
            # loss += loss_center_coarse * 1.0

        # coarse point cloud
        loss_coarse_align = masked_l1_loss(
            pos_coarse,
            positions.detach(),
            # (tracks_3d.visibles.float() * tracks_3d.confidences)[..., None],
            # quantile=0.95,
        )
        loss += loss_coarse_align * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"{loss.item():.3f} "
            f"{track_3d_loss.item():.3f} "
            f"{loss_2d.item():.3f} "
            f"{motion_coef_sparse_loss.item():.3f} "
            f"{small_acc_loss.item():.3f} "
            f"{small_acc_loss_tracks.item():.3f} "
            f"{bases_reg_loss.item():.3f} "
            f"{loss_center_cano.item():.3f} "
            f"{loss_center_coarse.item():.3f} "
            f"{loss_coarse_align.item():.3f} "
            # f"{z_accel_loss.item():.3f} "
        )


def random_quats(N: int) -> torch.Tensor:
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * np.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * np.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * np.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * np.pi * w),
        ],
        -1,
    )
    return quats


def compute_means(ts, fg: GaussianParams, bases: MotionBases):
    transfms = bases.compute_transforms(ts, fg.get_coefs(), fg.get_cluster_ids())
    means = torch.einsum(
        "pnij,pj->pni",
        transfms,
        F.pad(fg.params["means"], (0, 1), value=1.0),
    )
    return means


def vis_init_params(
    server,
    fg: GaussianParams,
    bases: MotionBases,
    name="init_params",
    num_vis: int = 100,
):
    idcs = np.random.choice(fg.num_gaussians, num_vis)
    labels = np.linspace(0, 1, num_vis)
    ts = torch.arange(bases.num_frames, device=bases.params["rots"].device)
    with torch.no_grad():
        pred_means = compute_means(ts, fg, bases)
        vis_means = pred_means[idcs].detach().cpu().numpy()
    vis_tracks_3d(server, vis_means, labels, name=name)


@torch.no_grad()
def vis_se3_init_3d(server, init_rots, init_ts, basis_centers):
    """
    :param init_rots: [num_bases, num_frames, 4|6]
    :param init_ts: [num_bases, num_frames, 3]
    :param basis_centers: [num_bases, 3]
    """
    # visualize the initial centers across time
    rot_dim = init_rots.shape[-1]
    assert rot_dim in [4, 6]
    num_bases = init_rots.shape[0]
    assert init_ts.shape[0] == num_bases
    assert basis_centers.shape[0] == num_bases
    labels = np.linspace(0, 1, num_bases)
    if rot_dim == 4:
        quats = F.normalize(init_rots, dim=-1, p=2)
        rmats = roma.unitquat_to_rotmat(quats.roll(-1, dims=-1))
    else:
        rmats = cont_6d_to_rmat(init_rots)
    transls = init_ts
    transfms = rt_to_mat4(rmats, transls)
    center_tracks3d = torch.einsum(
        "bnij,bj->bni", transfms, F.pad(basis_centers, (0, 1), value=1.0)
    )[..., :3]
    vis_tracks_3d(server, center_tracks3d.cpu().numpy(), labels, name="se3_centers")


@torch.no_grad()
def vis_tracks_2d_video(
    path,
    imgs: np.ndarray,
    tracks_3d: np.ndarray,
    Ks: np.ndarray,
    w2cs: np.ndarray,
    occs=None,
    radius: int = 3,
):
    num_tracks = tracks_3d.shape[0]
    labels = np.linspace(0, 1, num_tracks)
    cmap = get_cmap("gist_rainbow")
    colors = cmap(labels)[:, :3]
    tracks_2d = (
        project_2d_tracks(tracks_3d.swapaxes(0, 1), Ks, w2cs).cpu().numpy()  # type: ignore
    )
    frames = np.asarray(
        draw_keypoints_video(imgs, tracks_2d, colors, occs, radius=radius)
    )
    iio.imwrite(path, frames, fps=15)


def vis_tracks_3d(
    server: ViserServer,
    vis_tracks: np.ndarray,
    vis_label: np.ndarray | None = None,
    name: str = "tracks",
):
    """
    :param vis_tracks (np.ndarray): (N, T, 3)
    :param vis_label (np.ndarray): (N)
    """
    cmap = get_cmap("gist_rainbow")
    if vis_label is None:
        vis_label = np.linspace(0, 1, len(vis_tracks))
    colors = cmap(np.asarray(vis_label))[:, :3]
    guru.info(f"{colors.shape=}, {vis_tracks.shape=}")
    N, T = vis_tracks.shape[:2]
    vis_tracks = np.asarray(vis_tracks)
    for i in range(N):
        server.scene.add_spline_catmull_rom(
            f"/{name}/{i}/spline", vis_tracks[i], color=colors[i], segments=T - 1
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/start",
            vis_tracks[i, [0]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/end",
            vis_tracks[i, [-1]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="diamond",
        )


def cluster_by_velocities(coefs):
    # hdbscan clustering
    coefs_cp = cp.asarray(coefs.clone())
    model = HDBSCAN(min_cluster_size=20)
    model.fit(coefs_cp)
    labels = model.labels_
    num_clusters = labels.max().item() + 1
    cluster_persistence = model.cluster_persistence_

    return torch.tensor(labels), num_clusters, torch.tensor(cluster_persistence)


def clustering_by_coefs(coefs, num_clusters, mode="kmeans"):
    assert mode in ["hdbscan", "kmeans"]
    # clustering
    coefs_cp = cp.asarray(coefs.clone())
    if mode == "kmeans":
        model = KMeans(n_clusters=num_clusters)
    else:
        model = HDBSCAN(min_cluster_size=20)
    model.fit(coefs_cp)
    labels = model.labels_
    num_clusters = labels.max().item() + 1

    return torch.tensor(labels), num_clusters


def sample_bases_centers_by_means(mode, means, num_bases: int, pre_filter="kmeans", min_clus_thresh=0.005):
    """
    :param mode: "farthest" | "hdbscan" | "kmeans"
    """
    assert mode in ["hdbscan", "kmeans"]
    n_points = means.shape[0]

    # Pre-filter to remove noise
    if pre_filter == "kmeans":
        means_pre = cp.asarray(means.clone())
        pre_model = KMeans(n_clusters=num_bases)
        pre_model.fit(means_pre)
        pre_labels = torch.tensor(pre_model.labels_).cpu().long()

        # clusters smaller than threshold are noise
        ids, counts = pre_labels.unique(return_counts=True)
        noise_ids = ids[counts < n_points * min_clus_thresh]

        # mark small clusters as noise
        mask = torch.isin(pre_labels, noise_ids)
        pre_labels[mask] = -1
    elif pre_filter == "hdbscan":
        means_pre = cp.asarray(means.clone())

        pre_model = HDBSCAN(min_cluster_size=5)
        pre_model.fit(means_pre)
        pre_labels = torch.tensor(pre_model.labels_).cpu().long()
    elif pre_filter == "none":
        pre_labels = torch.zeros(n_points, dtype=torch.long)
    else:
        raise ValueError(f"pre_filter {pre_filter} not supported")

    # If too few inliers remain, fall back
    inliers_mask = pre_labels != -1
    n_in = int(inliers_mask.sum().item())
    min_required = int(0.5 * n_points)
    if n_in < min_required:
        print(f"[WARNING] Only {n_in} inliers left (< {min_required}); skipping pre-filter.")
        # treat all points as inliers
        inliers_mask = torch.ones_like(pre_labels, dtype=torch.bool)
        n_in = inliers_mask.numel()

    # cluster
    means_cp = cp.asarray(means[inliers_mask].clone())
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20)
    model.fit(means_cp)
    labels = torch.tensor(model.labels_).cpu().long()
    num_bases = labels.max().item() + 1

    # Add noise points back to nearest cluster
    if (~inliers_mask).any():
        # compute centroids of current clusters
        centroids = torch.stack(
            [means[inliers_mask][labels == i].mean(dim=0) for i in range(num_bases)]
        )  # (num_bases, 3)

        # assign each noise point to nearest centroid
        noise_points = means[~inliers_mask]  # (n_noise, 3)
        dists = torch.cdist(noise_points, centroids)  # (n_noise, num_bases)
        nearest = dists.argmin(dim=1)

        # build full label array
        full_labels = torch.full((n_points,), -1, dtype=torch.long)
        full_labels[inliers_mask] = labels
        full_labels[~inliers_mask] = nearest
    else:
        full_labels = labels.clone()

    sampled_centers = torch.stack(
        [
            means[full_labels == i].median(dim=0).values
            for i in range(num_bases)
        ]
    )[None]

    return sampled_centers, num_bases, full_labels

def sample_initial_bases_centers(
    mode: str, cano_t: int, tracks_3d: TrackObservations, num_bases: int
):
    """
    Perform clustering on all tracks, taking the entire track velocity vector as a sample.
    For each cluster, get the group of 3D positions of the track on the canonical frame.
    Use the median position of them as a sampled center.
    :param mode: "farthest" | "hdbscan" | "kmeans"
    :param tracks_3d: [G, T, 3]
    :param cano_t: canonical index
    :param num_bases: number of SE3 bases
    """
    assert mode in ["farthest", "hdbscan", "kmeans"]
    means_canonical = tracks_3d.xyz[:, cano_t].clone()
    # if mode == "farthest":
    #     vis_mask = tracks_3d.visibles[:, cano_t]
    #     sampled_centers, _ = sample_farthest_points(
    #         means_canonical[vis_mask][None],
    #         K=num_bases,
    #         random_start_point=True,
    #     )  # [1, num_bases, 3]
    #     dists2centers = torch.norm(means_canonical[:, None] - sampled_centers, dim=-1).T
    #     return sampled_centers, num_bases, dists2centers

    # linearly interpolate missing 3d points (visible=False)
    # points before first valid and after last valid value are also interpolated
    # velocity will be mostly zero outside the visible range
    xyz = cp.asarray(tracks_3d.xyz)
    print(f"{xyz.shape=}")
    visibles = cp.asarray(tracks_3d.visibles)

    num_tracks = xyz.shape[0]
    xyz_interp = batched_interp_masked(xyz, visibles)

    # num_vis = 50
    # server = get_server(port=8890)
    # idcs = np.random.choice(num_tracks, num_vis)
    # labels = np.linspace(0, 1, num_vis)
    # vis_tracks_3d(server, tracks_3d.xyz[idcs].get(), labels, name="raw_tracks")
    # vis_tracks_3d(server, xyz_interp[idcs].get(), labels, name="interp_tracks")

    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1]
    vel_dirs = (
        velocities / (cp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-5)
    ).reshape((num_tracks, -1))

    # [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    elif mode == "hdbscan":
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)
    else:
        raise ValueError(f"Invalid clustering function: {mode}")

    model.fit(vel_dirs)
    labels = model.labels_
    num_bases = labels.max().item() + 1
    sampled_centers = torch.stack(
        [
            means_canonical[torch.tensor(labels == i).cpu()].median(dim=0).values
            for i in range(num_bases)
        ]
    )[None]
    print("number of {} clusters: ".format(mode), num_bases)
    return sampled_centers, num_bases, torch.tensor(labels).cpu()


def interp_masked(vals: cp.ndarray, mask: cp.ndarray, pad: int = 1) -> cp.ndarray:
    """
    hacky way to interpolate batched with cupy
    by concatenating the batches and pad with dummy values
    :param vals: [B, M, *]
    :param mask: [B, M]
    """
    assert mask.ndim == 2
    assert vals.shape[:2] == mask.shape

    B, M = mask.shape

    # get the first and last valid values for each track
    sh = vals.shape[2:]
    vals = vals.reshape((B, M, -1))
    D = vals.shape[-1]
    first_val_idcs = cp.argmax(mask, axis=-1)
    last_val_idcs = M - 1 - cp.argmax(cp.flip(mask, axis=-1), axis=-1)
    bidcs = cp.arange(B)

    v0 = vals[bidcs, first_val_idcs][:, None]
    v1 = vals[bidcs, last_val_idcs][:, None]
    m0 = mask[bidcs, first_val_idcs][:, None]
    m1 = mask[bidcs, last_val_idcs][:, None]
    if pad > 1:
        v0 = cp.tile(v0, [1, pad, 1])
        v1 = cp.tile(v1, [1, pad, 1])
        m0 = cp.tile(m0, [1, pad])
        m1 = cp.tile(m1, [1, pad])

    vals_pad = cp.concatenate([v0, vals, v1], axis=1)
    mask_pad = cp.concatenate([m0, mask, m1], axis=1)

    M_pad = vals_pad.shape[1]
    vals_flat = vals_pad.reshape((B * M_pad, -1))
    mask_flat = mask_pad.reshape((B * M_pad,))
    idcs = cp.where(mask_flat)[0]

    cx = cp.arange(B * M_pad)
    out = cp.zeros((B * M_pad, D), dtype=vals_flat.dtype)
    for d in range(D):
        out[:, d] = cp.interp(cx, idcs, vals_flat[idcs, d])

    out = out.reshape((B, M_pad, *sh))[:, pad:-pad]
    return out


def batched_interp_masked(
    vals: cp.ndarray, mask: cp.ndarray, batch_num: int = 4096, batch_time: int = 64
):
    assert mask.ndim == 2
    B, M = mask.shape
    out = cp.zeros_like(vals)
    for b in tqdm(range(0, B, batch_num), leave=False):
        for m in tqdm(range(0, M, batch_time), leave=False):
            x = interp_masked(
                vals[b : b + batch_num, m : m + batch_time],
                mask[b : b + batch_num, m : m + batch_time],
            )  # (batch_num, batch_time, *)
            out[b : b + batch_num, m : m + batch_time] = x
    return out


def initialize_coefs(dists, coefs_type, sigma):
    if coefs_type == "linear":
        coefs = -dists / sigma
    elif coefs_type == "squared":
        coefs = -(dists ** 2) / (2 * sigma ** 2)
    else:
        raise ValueError(f"Invalid coefs_type: {coefs_type}")

    return coefs
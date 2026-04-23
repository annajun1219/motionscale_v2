import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.transforms import cont_6d_to_rmat


class CameraPoses(nn.Module):
    def __init__(
        self,
        Rs: torch.Tensor,
        ts: torch.Tensor,
    ):
        super().__init__()
        assert Rs.shape[-2:] == (3, 2) and ts.shape[-2:] == (3, 1) and Rs.shape[0] == ts.shape[0]
        self.params = nn.ParameterDict(
            {
                "Rs": nn.Parameter(Rs),  # (N, 3, 2)
                "ts": nn.Parameter(ts),  # (N, 3, 1)
            }
        )
        self.temp_new_poses = None

    @staticmethod
    def init_from_state_dict(
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ):
        param_keys = ["Rs", "ts"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return CameraPoses(**args)

    def inject_temp_poses(self, new_poses):
        self.temp_new_poses = new_poses

    def _get_active_params(self, name: str) -> torch.Tensor:
        base = self.params[name]
        if self.temp_new_poses is None:
            return base
        new = self.temp_new_poses.params[name]
        return torch.cat([base.detach(), new], dim=0)

    def get_rot_matrix(self):
        Rs = self._get_active_params("Rs")  # (N, 3, 2) for 6D rep
        r1 = Rs[:, :, 0]  # (N, 3)
        r2 = Rs[:, :, 1]  # (N, 3)

        r1 = r1 / torch.norm(r1, dim=-1, keepdim=True)
        r2 = r2 - (r1 * r2).sum(dim=-1, keepdim=True) * r1
        r2 = r2 / torch.norm(r2, dim=-1, keepdim=True)

        r3 = torch.cross(r1, r2, dim=-1)

        return torch.stack([r1, r2, r3], dim=-1)  # (N, 3, 3)

    def get_camera_matrix(self):
        """get the 3x4 camera pose"""
        rot_mats = self.get_rot_matrix()
        ts = self._get_active_params("ts")
        pose = torch.cat([rot_mats, ts], dim=-1)  # [..., 3, 4]
        homo_pad = torch.tensor([0., 0., 0., 1.], device=pose.device).repeat(pose.shape[0], 1, 1)
        pose = torch.cat([pose, homo_pad], dim=-2)  # (..., 4, 4)
        assert pose.shape[-2:] == (4, 4)
        return pose

    def add_new_frames(self, new_poses) -> dict:
        updated_params = {}
        device = self.params["Rs"].device
        for name, x in self.params.items():
            x_new = nn.Parameter(torch.cat([x, new_poses.params[name].to(device)], dim=0))
            self.params.update({name: x_new})
            updated_params[name] = x_new
        return updated_params


class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        motion_coefs: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
        cluster_ids: torch.Tensor | None = None,
    ):
        super().__init__()
        if not check_gaussian_sizes(
            means, quats, scales, colors, opacities, motion_coefs
        ):
            import ipdb
            ipdb.set_trace()

        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "colors": nn.Parameter(colors),
            "opacities": nn.Parameter(opacities),
        }
        if motion_coefs is not None:
            params_dict["motion_coefs"] = nn.Parameter(motion_coefs)
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))

        # cluster indices for scalable bases
        if cluster_ids is not None:
            self.register_buffer("cluster_ids", cluster_ids)

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["params.means", "params.quats", "params.scales", "params.colors", "params.opacities"]
        assert prefix is not None and prefix != ""
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        # retrieve all data in state_dict, use to initialize the new model
        init_args = {}
        for k in state_dict.keys():
            if k.startswith(prefix):
                arg_name = k.split(".")[-1]
                init_args[arg_name] = state_dict[k]

        return GaussianParams(**init_args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def get_colors(self) -> torch.Tensor:
        return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])

    def get_coefs(self) -> torch.Tensor:
        assert "motion_coefs" in self.params
        return self.motion_coef_activation(self.params["motion_coefs"])

    def get_cluster_ids(self) -> torch.Tensor:
        return self.cluster_ids.clone() if hasattr(self, "cluster_ids") else None

    def set_cluster_ids(self, new_ids):
        self.register_buffer("cluster_ids", new_ids)

    def add_params(self, new_params):
        """
        Add a given set of gaussians
        """
        updated_params = {}
        device = self.params["means"].device
        n_gaussians = new_params.num_gaussians
        for name, x in self.params.items():
            if name in new_params.params:
                x_new = nn.Parameter(torch.cat([x, new_params.params[name].to(device)], dim=0))
            else:
                x_new = nn.Parameter(torch.cat([x, x.new_zeros(n_gaussians, x.shape[-1])], dim=0))
            updated_params[name] = x_new
            self.params.update({name: x_new})  # this should be safer
        # densify cluster_ids
        if hasattr(self, "cluster_ids"):
            new_ids = torch.cat(
                [
                    self.cluster_ids,
                    new_params.cluster_ids.to(device),
                ],
                dim=0,
            )
            self.register_buffer("cluster_ids", new_ids)

        return updated_params

    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new
        # densify cluster_ids
        if hasattr(self, "cluster_ids"):
            new_ids = torch.cat(
                [
                    self.cluster_ids[~should_split],
                    self.cluster_ids[should_dup],
                    self.cluster_ids[should_split].repeat([2]),
                ],
                dim=0,
            )
            self.register_buffer("cluster_ids", new_ids)
        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new
        if hasattr(self, "cluster_ids"):
            new_ids = self.cluster_ids[~should_cull]
            self.register_buffer("cluster_ids", new_ids)
        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}
        return updated_params


class ScalableMotionBases(nn.Module):
    def __init__(self, centers, rots, transls, fine_rots, fine_transls):
        """
        :param centers: (C, 3) initial positions of cluster centers
        :param rots: (C, T, 6) coarse rotation bases per cluster
        :param transls: (C, T, 3) coarse translation bases per cluster
        :param fine_rots: (C, F, T, 6) fine rotation bases per cluster
        :param fine_transls: (C, F, T, 3) fine translation bases per cluster
        """
        super().__init__()
        assert self.check_sizes(centers, rots, transls, fine_rots, fine_transls)
        self.params = nn.ParameterDict(
            {
                "centers": nn.Parameter(centers),           # (C, 3)
                "rots": nn.Parameter(rots),                 # (C, T, 6)
                "transls": nn.Parameter(transls),           # (C, T, 3)
                "fine_rots": nn.Parameter(fine_rots),       # (C, F, T, 6)
                "fine_transls": nn.Parameter(fine_transls), # (C, F, T, 3)
            }
        )

    def check_sizes(self, centers, rots, transls, fine_rots, fine_transls) -> bool:
        check_num_clusters = (centers.shape[0] == rots.shape[0] == transls.shape[0] == fine_rots.shape[0] == fine_transls.shape[0])
        check_num_frames = (rots.shape[1] == transls.shape[1] == fine_rots.shape[2] == fine_transls.shape[2])
        check_num_fine = (fine_rots.shape[1] == fine_transls.shape[1])
        return (check_num_clusters and check_num_frames and check_num_fine)

    @property
    def num_frames(self):
        return self.params["rots"].shape[1]

    @property
    def num_clusters(self):
        return self.params["rots"].shape[0]

    @property
    def num_fine_bases(self):
        return self.params["fine_rots"].shape[1]

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["centers", "rots", "transls", "fine_rots", "fine_transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return ScalableMotionBases(**args)

    def dup_bases(self, should_dup):
        """
        Duplicate bases to densify.
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_new = nn.Parameter(torch.cat([x, x_dup], dim=0))
            updated_params[name] = x_new
            self.params.update({name: x_new})
        return updated_params

    def add_bases(self, new_bases):
        """
        Add new bases (new clusters).
        """
        updated_params = {}
        device = self.params["rots"].device
        for name, x in self.params.items():
            x_new = nn.Parameter(torch.cat([x, new_bases.params[name].to(device)], dim=0))  # cat along the cluster dim
            updated_params[name] = x_new
            self.params.update({name: x_new})

        return updated_params

    def cull_bases(self, should_cull):
        """
        cull bases
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params.update({name: x_new})
        return updated_params

    def add_new_frames(self, new_bases):
        """
        Add bases for additional frames.
        """
        updated_params = {}
        device = self.params["rots"].device
        for name, x in self.params.items():
            if name == "centers":
                continue
            x_new = nn.Parameter(torch.cat([x, new_bases.params[name].to(device)], dim=-2))
            updated_params[name] = x_new
            # self.params[name] = x_new
            self.params.update({name: x_new})  # this should be safer

        return updated_params

    def get_centers(self, ts: torch.Tensor, freeze_centers=False, freeze_coarse=False):
        """
        Transform canonical cluster centers to the current frames, using the coarse transformation only.

        Args:
            ts, (B): time
        Returns:
            centers_ts, (C, B, 3): transformed centers
        """
        centers = self.params["centers"]  # (C, 3)
        coarse_transls = self.params["transls"][:, ts]  # (C, B, 3)

        if freeze_centers:
            centers = centers.detach()
        if freeze_coarse:
            coarse_transls = coarse_transls.detach()

        centers_ts = centers[:, None] + coarse_transls  # (C, B, 3)

        return centers_ts

    def compute_transforms_coarse(self, ts: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the 3x4 transformation matrix of coarse transformation only.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        # coarse transformation
        coarse_transls = self.params["transls"][:, ts]  # (C, B, 3)
        coarse_rots = self.params["rots"][:, ts]  # (C, B, 6)
        coarse_rotmats = cont_6d_to_rmat(coarse_rots)  # (C, B, 3, 3)
        centers = self.params["centers"]  # (C, 3)

        # Compute effective translation: T_eff = -R_coarse * c + c + t_coarse
        transls_eff = (
            -torch.einsum("cbij,cj->cbi", coarse_rotmats, centers)
            + centers[:, None] + coarse_transls
        )  # (C, B, 3)

        return torch.cat([coarse_rotmats[cluster_ids], transls_eff[cluster_ids].unsqueeze(-1)], dim=-1)


    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the 3x4 transformation matrix via coarse-to-fine transforms.

        :param ts (B)
        :param coefs (G, F)
        :param cluster_ids (G) int
        returns transforms (G, B, 3, 4)
        """
        assert coefs.shape[0] == cluster_ids.shape[0]
        G, F_dim = coefs.shape
        C, _, B, _ = self.params["fine_transls"][:, :, ts].shape

        # coarse transformation
        coarse_transls = self.params["transls"][:, ts]  # (C, B, 3)
        coarse_rots = self.params["rots"][:, ts]  # (C, B, 6)
        coarse_rotmats = cont_6d_to_rmat(coarse_rots)  # (C, B, 3, 3)
        centers = self.params["centers"]  # (C, 3)
        
        # Fine transformation
        fine_transls = self.params["fine_transls"][:, :, ts]  # (C, F, B, 3)
        fine_rots = self.params["fine_rots"][:, :, ts]  # (C, F, B, 6)

        # Combine fine and coarse rotations
        # x_total = R_coarse * (R_fine * (x - center) + T_fine) + T_coarse + center
        # R_total = R_coarse * R_fine
        fine_rotmats = cont_6d_to_rmat(fine_rots) # (C, F, B, 3, 3)
        total_rotmats = torch.einsum("cbij,cfbjk->cfbik", coarse_rotmats, fine_rotmats)  # (C, F, B, 3, 3)
        total_6d = total_rotmats[..., :, :2].transpose(-1, -2).reshape(C, F_dim, B, 6)  # (C, F, B, 6)

        # Combine fine and coarse translations
        # T_total = -R_total * center + R_coarse * T_fine + T_coarse + center
        R_total_c = torch.einsum("cfbij,cj->cfbi", total_rotmats, centers)  # (C, F, B, 3)
        R_coarse_t_fine = torch.einsum("cbij,cfbj->cfbi", coarse_rotmats, fine_transls)  # (C, F, B, 3)
        total_transls = -R_total_c + R_coarse_t_fine + coarse_transls[:, None] + centers[:, None, None]  # (C, F, B, 3)

        # Flatten for embedding_bag
        transls_flat = total_transls.contiguous().view(C * F_dim, -1)  # (C*F, B*3)
        rots_flat = total_6d.contiguous().view(C * F_dim, -1)  # (C*F, B*6)

        # Create the lookup indices for the bags
        base_offsets = torch.arange(F_dim, device=cluster_ids.device)
        bag_indices = (cluster_ids.unsqueeze(1) * F_dim) + base_offsets  # (G, F)

        # Fused Gather + Weighted Sum
        transls = F.embedding_bag(
            weight=transls_flat,
            input=bag_indices,
            per_sample_weights=coefs,
            mode='sum',
        ).view(G, B, 3)  # (G, B*3) -> (G, B, 3)

        rots_blended = F.embedding_bag(
            weight=rots_flat,
            input=bag_indices,
            per_sample_weights=coefs,
            mode='sum',
        ).view(G, B, 6)  # (G, B*6) -> (G, B, 6)
        rotmats = cont_6d_to_rmat(rots_blended)  # (G, B, 3, 3)

        return torch.cat([rotmats, transls[..., None]], dim=-1)


class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        assert check_bases_sizes(rots, transls)
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )

    @property
    def num_frames(self):
        return self.params["rots"].shape[1]

    @property
    def num_bases(self):
        return self.params["rots"].shape[0]

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return MotionBases(**args)

    def add_new_frames(self, new_bases):
        """
        Add bases for additional frames.
        """
        updated_params = {}
        device = self.params["rots"].device
        for name, x in self.params.items():
            x_new = nn.Parameter(torch.cat([x, new_bases.params[name].to(device)], dim=1))
            updated_params[name] = x_new
            self.params.update({name: x_new})

        return updated_params

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        transls = self.params["transls"][:, ts]  # (K, B, 3)
        rots = self.params["rots"][:, ts]  # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", coefs, transls)
        rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)


def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    motion_coefs: torch.Tensor | None = None,
) -> bool:
    dims = means.shape[:-1]
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    if motion_coefs is not None and motion_coefs.numel() > 0:
        leading_dims_match &= motion_coefs.shape[:-1] == dims
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct


def check_bases_sizes(motion_rots: torch.Tensor, motion_transls: torch.Tensor) -> bool:
    return (
        motion_rots.shape[-1] == 6
        and motion_transls.shape[-1] == 3
        and motion_rots.shape[:-2] == motion_transls.shape[:-2]
    )

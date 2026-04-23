import os
import subprocess
from argparse import ArgumentParser


def process_sequence(
    gpu: int,
    img_dir: str,
    input_mask_dir: str,
    input_depth_dir: str,
    moge_depth_dir: str,
    sam2_mask_dir: str,
    track_dir: str,
    pi3_ckpt: str,
    moge_ckpt: str,
    sam2_ckpt: str,
    sam2_cfg: str,
    track_repo: str,
    track_ckpt: str = None,
    track_shuffle: bool = False,
):
    custom_env = os.environ.copy()
    custom_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ## Pi3 depth
    pi3_depth_cmd = [
        "python", "preproc/compute_depths_pi3.py",
        "--img_dir", img_dir,
        "--out_dir", input_depth_dir,
        "--ckpt", pi3_ckpt,
    ]
    subprocess.run(pi3_depth_cmd, env=custom_env, check=True)

    ## MoGe depth
    moge_depth_cmd = [
        "python", "preproc/compute_depths_moge.py",
        "-i", img_dir,
        "-o", moge_depth_dir,
        "--pretrained", moge_ckpt,
    ]
    subprocess.run(moge_depth_cmd, env=custom_env, check=True)

    ## SAM2 masks
    sam2_cmd = [
        "python", "preproc/compute_masks_sam2.py",
        "--img_dir", img_dir,
        "--input_mask_dir", input_mask_dir,
        "--out_dir", sam2_mask_dir,
        "--ckpt", sam2_ckpt,
        "--cfg", sam2_cfg,
    ]
    subprocess.run(sam2_cmd, env=custom_env, check=True)

    ## CoTracker tracks
    track_cmd = [
        "python", "cotracker/evaluation/compute_tracks_all.py",
        "--image_dir", os.path.abspath(img_dir),
        "--mask_dir", os.path.abspath(sam2_mask_dir),
        "--out_dir", os.path.abspath(track_dir),
    ]
    if track_ckpt is not None:
        track_cmd += ["--ckpt_path", os.path.abspath(track_ckpt)]
    if track_shuffle:
        track_cmd += ["--shuffle"]
    subprocess.run(track_cmd, env=custom_env, cwd=track_repo, check=True)


# Example:
# conda run --no-capture-output -n motion_scale_v2 python preproc/process_davis.py --base_dir /path/to/DAVIS/ && conda run --no-capture-output -n mega_sam python preproc/process_davis_megasam.py --base_dir /path/to/DAVIS/
if __name__ == "__main__":
    parser = ArgumentParser(description="Process davis data.")
    parser.add_argument("--base_dir", type=str, default="/path/to/DAVIS/")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory. Defaults to base_dir/flow3d_preprocessed.")
    parser.add_argument('--seqs', type=str, nargs='+', default=None)
    parser.add_argument("--res", type=str, default="480p")
    parser.add_argument("--pi3_ckpt", type=str, default="preproc/checkpoints/model.safetensors")
    parser.add_argument("--moge_ckpt", type=str, default="preproc/checkpoints/moge-2-vitl-normal/model.pt")
    parser.add_argument("--sam2_ckpt", type=str, default="preproc/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--input_depth_name", type=str, default="pi3")
    parser.add_argument("--moge_depth_name", type=str, default="moge")
    parser.add_argument("--mask_name", type=str, default="sam2_masks")
    parser.add_argument("--track_name", type=str, default="cotracker3")
    parser.add_argument("--track_repo", type=str, default="./preproc/co-tracker")
    parser.add_argument("--track_ckpt", type=str, default="preproc/checkpoints/scaled_offline.pth", help="override cotracker checkpoint path")
    parser.add_argument("--track_shuffle", action="store_true", help="shuffle points before chunking in cotracker")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # video seqs to process
    if args.seqs is not None:
        seq_names = args.seqs
    else:
        seq_names = [
            "bike-packing",
            "blackswan",
            "bmx-trees",
            "breakdance",
            "camel",
            "car-roundabout",
            "car-shadow",
            "cows",
            "dance-twirl",
            "dog",
            "dogs-jump",
            "drift-chicane",
            "drift-straight",
            "goat",
            "gold-fish",
            "horsejump-high",
            "india",
            "judo",
            "kite-surf",
            "lab-coat",
            "libby",
            "loading",
            "mbike-trick",
            "motocross-jump",
            "paragliding-launch",
            "parkour",
            "pigs",
            "scooter-black",
            "shooting",
            "soapbox",
        ]

    for seq_name in seq_names:
        ## DAVIS data
        # images input
        img_dir = os.path.join(args.base_dir, "JPEGImages", args.res, seq_name)
        # seed mask input (first-frame GT for SAM2)
        input_mask_dir = os.path.join(args.base_dir, "Annotations", args.res, seq_name)

        ## preprocessed data
        out_root = args.out_dir if args.out_dir is not None else os.path.join(args.base_dir, "flow3d_preprocessed")
        preproc_dir = os.path.join(out_root, args.res, seq_name)
        # if os.path.exists(preproc_dir):
        #     continue

        # pi3 depth output
        input_depth_dir = os.path.join(preproc_dir, args.input_depth_name)
        # moge depth output
        moge_depth_dir = os.path.join(preproc_dir, args.moge_depth_name)
        # sam2 mask output
        sam2_mask_dir = os.path.join(preproc_dir, args.mask_name)
        # cotracker track output
        track_dir = os.path.join(preproc_dir, args.track_name)

        process_sequence(
            gpu=args.gpu,
            img_dir=img_dir,
            input_mask_dir=input_mask_dir,
            input_depth_dir=input_depth_dir,
            moge_depth_dir=moge_depth_dir,
            sam2_mask_dir=sam2_mask_dir,
            track_dir=track_dir,
            pi3_ckpt=args.pi3_ckpt,
            moge_ckpt=args.moge_ckpt,
            sam2_ckpt=args.sam2_ckpt,
            sam2_cfg=args.sam2_cfg,
            track_repo=args.track_repo,
            track_ckpt=args.track_ckpt,
            track_shuffle=args.track_shuffle,
        )

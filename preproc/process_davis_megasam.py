import os
import subprocess
from argparse import ArgumentParser


def process_sequence(
    gpu: int,
    img_dir: str,
    input_depth_dir: str,
    moge_depth_dir: str,
    slam_dir: str,
    slam_repo: str,
    seq_name: str,
):
    custom_env = os.environ.copy()
    custom_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        "python", "run_davis_pi3.py",
        "--video_dir", os.path.abspath(img_dir),
        "--input_depth_dir", os.path.abspath(input_depth_dir),
        "--monodepth_dir", os.path.abspath(moge_depth_dir),
        "--output_dir", os.path.abspath(slam_dir),
        "--seq_name", seq_name,
        "--gpu", str(gpu),
    ]
    subprocess.run(cmd, env=custom_env, cwd=slam_repo, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process davis data with mega-sam (requires megasam env).")
    parser.add_argument("--base_dir", type=str, default="/path/to/DAVIS/")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory. Defaults to base_dir/flow3d_preprocessed.")
    parser.add_argument('--seqs', type=str, nargs='+', default=None)
    parser.add_argument("--res", type=str, default="480p")
    parser.add_argument("--input_depth_name", type=str, default="pi3")
    parser.add_argument("--moge_depth_name", type=str, default="moge")
    parser.add_argument("--slam_name", type=str, default="megasam")
    parser.add_argument("--slam_repo", type=str, default="./preproc/mega-sam")
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
        img_dir = os.path.join(args.base_dir, "JPEGImages", args.res, seq_name)

        ## preprocessed data (shared out_dir with process_davis.py)
        out_root = args.out_dir if args.out_dir is not None else os.path.join(args.base_dir, "flow3d_preprocessed")
        preproc_dir = os.path.join(out_root, args.res, seq_name)

        # pi3/moge depths (produced by process_davis.py)
        input_depth_dir = os.path.join(preproc_dir, args.input_depth_name, "depths")
        moge_depth_dir = os.path.join(preproc_dir, args.moge_depth_name, "depths")

        # mega-sam output
        slam_dir = os.path.join(preproc_dir, args.slam_name)

        process_sequence(
            gpu=args.gpu,
            img_dir=img_dir,
            input_depth_dir=input_depth_dir,
            moge_depth_dir=moge_depth_dir,
            slam_dir=slam_dir,
            slam_repo=args.slam_repo,
            seq_name=seq_name,
        )
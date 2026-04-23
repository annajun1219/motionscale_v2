import os
import subprocess
from argparse import ArgumentParser


# Example: python preproc/compute_shadows_all.py --img_base_dir /path/to/DAVIS/JPEGImages/480p/ --input_mask_dir ./outputs/annotator_masks/ --out_base_dir /path/to/DAVIS/flow3d_preprocessed/480p/
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_base_dir", type=str, required=True, help="Base dir with <seq>/ subdirs of video frames.")
    parser.add_argument("--input_mask_dir", type=str, required=True, help="Base dir with <seq>/ subdirs of annotated seed masks.")
    parser.add_argument("--out_base_dir", type=str, required=True, help="Base dir where output mask dirs will be written.")
    parser.add_argument("--mask_name", type=str, default="sam2_shadows", help="Subdirectory name for the output masks under <out_base_dir>/<seq>/<mask_name>.")
    parser.add_argument("--ckpt", type=str, default="preproc/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    seq_names = sorted([
        d for d in os.listdir(args.input_mask_dir)
        if os.path.isdir(os.path.join(args.input_mask_dir, d))
    ])
    if not seq_names:
        print(f"No sequence subdirectories found under {args.input_mask_dir}")
        exit(1)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for seq in seq_names:
        img_dir = os.path.join(args.img_base_dir, seq)
        mask_dir = os.path.join(args.input_mask_dir, seq)
        out_dir = os.path.join(args.out_base_dir, seq, args.mask_name)

        if not os.path.isdir(img_dir):
            print(f"[SKIP] {seq}: image dir not found at {img_dir}")
            continue

        if os.path.isdir(out_dir):
            print(f"[SKIP] {seq}: output already exists at {out_dir}")
            continue

        print(f"\n{'='*40}")
        print(f"Processing: {seq}")
        print(f"  img_dir: {img_dir}")
        print(f"  mask_dir: {mask_dir}")
        print(f"  out_dir: {out_dir}")
        print(f"{'='*40}")

        subprocess.run([
            "python", "preproc/compute_masks_sam2.py",
            "--img_dir", img_dir,
            "--input_mask_dir", mask_dir,
            "--out_dir", out_dir,
            "--ckpt", args.ckpt,
            "--cfg", args.cfg,
            "--score_thresh", str(args.score_thresh),
        ], env=env, check=True)

    print(f"\nDone. Processed {len(seq_names)} sequence(s).")
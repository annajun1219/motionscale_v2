import os
import sys

basedir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(basedir, "sam2")
sys.path.extend([src_dir])

import argparse

from sam2.build_sam import build_sam2_video_predictor
from tools.vos_inference import vos_inference


# Example: python preproc/compute_masks_sam2.py --img_dir /path/to/DAVIS/JPEGImages/480p/camel --input_mask_dir /path/to/DAVIS/Annotations/480p/camel --out_dir ./outputs/test_masks/camel
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2 VOS inference on a single sequence.")
    parser.add_argument("--img_dir", type=str, required=True, help="Flat directory of JPEG frames for this sequence.")
    parser.add_argument("--input_mask_dir", type=str, required=True, help="Flat directory holding the seed mask PNG(s) for this sequence.")
    parser.add_argument("--out_dir", type=str, required=True, help="Flat directory where output mask PNGs will be written.")
    parser.add_argument("--ckpt", type=str, default="preproc/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the SAM2 checkpoint")
    parser.add_argument("--cfg", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="SAM2 hydra config (resolved via installed sam2 package)")
    parser.add_argument("--score_thresh", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Loading SAM2 predictor from {args.ckpt}")
    predictor = build_sam2_video_predictor(
        config_file=args.cfg,
        ckpt_path=args.ckpt,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
        vos_optimized=False,
    )

    print("=" * 20)
    print(f"Running SAM2 VOS inference: {args.img_dir}")
    print("=" * 20)

    # vos_inference builds its paths as os.path.join(base, video_name, "<frame>.png").
    # Passing video_name="" collapses the subfolder component, so the three flat
    # dirs we pass here are read and written directly, with no <seq_name>/ nesting.
    vos_inference(
        predictor=predictor,
        base_video_dir=args.img_dir,
        input_mask_dir=args.input_mask_dir,
        output_mask_dir=args.out_dir,
        video_name="",
        score_thresh=args.score_thresh,
    )

    print(f"Saved SAM2 masks to {args.out_dir}")

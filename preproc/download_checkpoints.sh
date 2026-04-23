#!/bin/bash
# Download model checkpoints for preprocessing.
# Run from the repo root: bash preproc/download_checkpoints.sh

set -e

mkdir -p preproc/checkpoints

# Pi3 (metric depth)
wget -q --show-progress -P preproc/checkpoints https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors

# MoGe-2 (monocular depth + normals)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Ruicheng/moge-2-vitl-normal', local_dir='preproc/checkpoints/moge-2-vitl-normal')"

# SAM2.1 (mask propagation)
wget -q --show-progress -P preproc/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# CoTracker3 (2D point tracking)
wget -q --show-progress -P preproc/checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth

#!/bin/bash
# Install main dependencies
pip install -r requirements.txt

# Git-based installs with --no-build-isolation (setup.py may import torch at build time)
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

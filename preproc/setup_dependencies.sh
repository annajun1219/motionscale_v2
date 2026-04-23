#!/bin/bash
# Install preproc dependencies into the main conda environment (motion_scale).

set -e

# Python packages
pip install -r preproc/requirements_extra.txt

# Submodule packages
pip install -e preproc/sam2
pip install -e preproc/co-tracker

# MoGe
pip install --no-build-isolation git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6

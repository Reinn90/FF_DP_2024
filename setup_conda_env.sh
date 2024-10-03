#!/bin/bash

# Create conda environment.
conda env create -f environment.yml
conda activate FF

# Install additional packages.
# conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install hydra-core
pip install einops

# i added this
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
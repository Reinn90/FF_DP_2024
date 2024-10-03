#!/bin/bash

# Create conda environment.
# conda env create -f environment.yml
# conda activate FF

# Install additional packages. - Conda env
# conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch


# Create python venv for Wolffe HPC
# python3.9 -m venv FF
source FF/bin/activate

pip install --upgrade pip

# Insta HPC+Python3.9 torch appropriate packages
pip install -r requirements.txt

# deactivate venv - it will be sourced when the slurm script is ran
deactivate

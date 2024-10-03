#! /usr/bin/env bash
#
#SBATCH --job-name=ff_test
#SBATCH --output=logs/main.log
#SBATCH --error=logs/error.log
#
#SBATCH --ntasks=1
#SBATCH --time=10:00:00 # this sets the maximum time the job is allowed before killed


#SBATCH --nodelist=a100-101   # when using 80gb node
#SBATCH --partition=ampere80
##SBATCH --partition=cpu # the double hash means that SLURM won't read this line.

# load the python module
# module load PyTorch/Python3.10 # make sure to load the modules needed
echo "Job Starting..."

source FF/bin/activate
echo "venv activated"

python3.9 main.py

echo "job complete"


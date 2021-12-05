#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=ml_mri_train
#SBATCH -t 12:00:00
#SBATCH -C A100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill

echo "Starting to run SBATCH SCRIPT"

source /home/sskodate/py37_venv/bin/activate

python src/mri_networks.py
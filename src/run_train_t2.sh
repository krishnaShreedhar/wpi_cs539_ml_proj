#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=ml_t2_1
#SBATCH -t 12:00:00
#SBATCH -C P100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill

echo "Starting to run SBATCH SCRIPT"

source /home/sskodate/py37_venv/bin/activate

python3 mri_networks_raul_edits_t2w.py


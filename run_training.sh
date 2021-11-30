#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=ml_mri_train
#SBATCH -o script_output.txt
#SBATCH -e std_err_script.err
#SBATCH -t 12:00:00
#SBATCH -C P100
#SBATCH --mem 64G
#SBATCH -p short

echo "Starting to run SBATCH SCRIPT"

source /home/sskodate/py37_venv/bin/activate

python src/mri_networks.py
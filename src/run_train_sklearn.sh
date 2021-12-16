#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=ml_param
#SBATCH -t 54:00:00
#SBATCH -C A100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill

module load cuda11.1/toolkit/11.1.1
module load cudnn/8.1.1.33-11.2/3k5bbs63

source /home/sskodate/py37_venv/bin/activate

echo "Starting to run sklearn models"
python train_ensemble_models.py
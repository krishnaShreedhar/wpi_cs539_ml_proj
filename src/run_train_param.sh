#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=ml_param
#SBATCH -t 12:00:00
#SBATCH -C A100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill

echo "Starting to run SBATCH SCRIPT"

source /home/sskodate/py37_venv/bin/activate

python3 mri_nw_param.py --epochs=1 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=1 --mri_type=T1w --max_data=4 --batch_size=2


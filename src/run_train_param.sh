#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=ml_param
#SBATCH -t 48:00:00
#SBATCH -C A100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill

module load cuda11.1/toolkit/11.1.1
module load cudnn/8.1.1.33-11.2/3k5bbs63

echo "Starting to run SBATCH SCRIPT"

source /home/sskodate/py37_venv/bin/activate

python3 mri_nw_param.py --model_to_train=3dcnn --epochs=1 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=1 --mri_type=T1w --max_data=4 --batch_size=2

python3 mri_nw_param.py --model_to_train=resnet --epochs=1 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=1 --mri_type=T1w --max_data=4 --batch_size=2


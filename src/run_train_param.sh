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

echo "Starting to run SBATCH SCRIPT FOLD 3"

python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=T1w --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=T1w --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=T1wCE --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=T1wCE --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=T2w --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=T2w --max_data=1000 --batch_size=2


echo "Starting to run SBATCH SCRIPT FOLD 4"

python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=4 --mri_type=T1w --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=4 --mri_type=T1wCE --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=4 --mri_type=T2w --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=4 --mri_type=T2w --max_data=1000 --batch_size=2

echo "Starting to run SBATCH SCRIPT FLAIR FOLD 1234"

python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=1 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=1 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=2 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=2 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=3 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=3dcnn --epochs=50 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=4 --mri_type=FLAIR --max_data=1000 --batch_size=2
python3 mri_nw_param.py --model_to_train=resnet --epochs=50 --initial_lr=0.001 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=4 --mri_type=FLAIR --max_data=1000 --batch_size=2

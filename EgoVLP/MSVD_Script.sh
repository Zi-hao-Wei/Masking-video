#!/bin/bash
#SBATCH --job-name=single_job_test    # Job name
#SBATCH --mail-type=END       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuningc@umich.edu     # Where to send mail	
#SBATCH --nodes=1                    # Run on a single CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3g
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00               # Time limit hrs:min:sec
#SBATCH --output=MSVD_random_2.log
#SBATCH --account=eecs592s001w24_class 
#SBATCH --get-user-env



pwd; hostname; date

nvidia-smi

module load python3.10-anaconda/2023.03

module load cuda/11.8.0

source ~/.bashrc

conda activate egovlp

# python /home/yuningc/DB/datasets/jersey/convert.py


# CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/bbjr_resnet50.yaml --image_path /home/yuningc/jersey-2023/train/images/11/11_10.jpg --resume /home/yuningc/DB/outputs/workspace/L1BalanceCELoss/model/final --box_thresh 0.45 --visualize --image_short_side 128
# CUDA_VISIBLE_DEVICES=0 python util.py experiments/seg_detector/bbjr_resnet50.yaml --result_dir /home/yuningc/jersey-2023/test_light/images --image_path /home/yuningc/jersey-2023/test/images --resume /home/yuningc/DB/outputs/workspace/L1BalanceCELoss/model/final --box_thresh 0.61 --visualize --image_short_side 128 


CUDA_VISIBLE_DEVICES=0  python train_MSVD.py 
# CUDA_VISIBLE_DEVICES=0 python train.py experiments/seg_detector/CONVERT_ONLY.yaml --num_gpus 1 --validate --localization #--debug #jersey_test.yaml --num_gpus 1 --validate --debug
# python train.py /home/yuningc/DB/experiments/seg_detector/jersey_extra_dataset.yaml --num_gpus 1 --validate



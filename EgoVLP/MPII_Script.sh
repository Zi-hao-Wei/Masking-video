#!/bin/bash
#SBATCH --job-name=single_job_test    # Job name
#SBATCH --mail-type=END       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuningc@umich.edu     # Where to send mail	
#SBATCH --nodes=1                    # Run on a single CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3g
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00               # Time limit hrs:min:sec
#SBATCH --output=MPII_baseline.log
#SBATCH --account=eecs592s001w24_class 
#SBATCH --get-user-env



pwd; hostname; date

module load python3.10-anaconda/2023.03

module load cuda/11.8.0

source ~/.bashrc

conda activate egovlp

CUDA_VISIBLE_DEVICES=0  python train_MPII.py 

pwd; hostname; date

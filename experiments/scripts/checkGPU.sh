#!/bin/sh
#SBATCH -o checkGPU.out
#SBATCH -p titanx
#SBATCH --gres=gpu:1


srun --gres=gpu:1 ./deviceQuery
srun --gres=gpu:1 ./nvidia-smi -a
wait

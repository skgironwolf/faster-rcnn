#!/bin/sh
#SBATCH -o job3.out
#SBATCH -p titanx
#SBATCH --gres=gpu:1


srun --gres=gpu:1 python demo.py
wait

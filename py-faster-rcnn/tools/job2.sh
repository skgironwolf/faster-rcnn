#!/bin/sh
#SBATCH -o job2.out
#SBATCH -p titanx
#SBATCH --gres=gpu:1


srun --gres=gpu:1 python demo.py
wait

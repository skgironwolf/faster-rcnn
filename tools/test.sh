#!/bin/sh
#SBATCH -o test.out
#SBATCH -p m40-short
#SBATCH --gres=gpu:1

srun --gres=gpu:1 --net='zf' python roostDemo.py
wait

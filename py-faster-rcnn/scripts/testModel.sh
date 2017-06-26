#!/bin/sh
#SBATCH -o testModel.out
#SBATCH -p titanx-short
#SBATCH --gres=gpu:2


srun python model.py
wait

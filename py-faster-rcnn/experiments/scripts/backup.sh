#!/bin/sh
#SBATCH -o job3.out
#SBATCH -p titanx-short
#SBATCH --gres=gpu:1


srun --gres=gpu:1 ./roost_rcnn.sh 1 ZF cnn_data
wait

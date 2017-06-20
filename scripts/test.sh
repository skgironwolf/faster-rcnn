#!/bin/sh
#SBATCH -o test.out
#SBATCH -p m40-short
#SBATCH --gres=gpu:1


srun --gres=gpu:1 ./roost_test2.sh 1 ZF cnn_data
wait

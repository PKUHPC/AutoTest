#!/bin/bash

#########config of pku-math########
##SBATCH -o job_%j_%N.out
##SBATCH --partition=gpu
##SBATCH -J hice
##SBATCH --qos=debug
##SBATCH --get-user-env
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:1
##SBATCH --time=3:00:00
##
##module add cuda/9.0-cudnn7.1
##module add mkl/2017.0.098

######config of pku-hpc########
#SBATCH -o job_%j_%N.out
#SBATCH --partition=GPU
##SBATCH --partition=C032M0256G
#SBATCH -A hpc1806179325
#SBATCH --qos=high
#SBATCH -J hice
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

# module add cuda/9.0

$1

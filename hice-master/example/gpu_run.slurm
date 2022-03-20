#!/bin/bash

########config of pku-math########
##SBATCH -o job_%j_%N.out
##SBATCH --partition=gpu
##SBATCH -J hice 
##SBATCH --get-user-env
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
##SBATCH --time=120:00:00
#
#module add cuda/9.0-cudnn7.1   
#module add mkl/2017.0.098
#
#./matmul

#######config of pku-hpc########
#SBATCH -o job_%j_%N.out
#SBATCH --partition=GPU
##SBATCH --partition=C032M0256G
#SBATCH -A hpc1806179325
#SBATCH --qos=low
#SBATCH -J hice 
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00

module add cuda/8.0   
module add cudnn/8.0-v6.0  

./matmul

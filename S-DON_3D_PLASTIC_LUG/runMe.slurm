#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bblv-delta-gpu
#SBATCH --job-name=LUG_DeepXDE
#SBATCH --output=LUG_DeepXDE_GRU_ABS_SPLIT_H256_E300K_%j.out
#SBATCH --error=LUG_DeepXDE_GRU_ABS_SPLIT_H256_E300K_%j.err
#SBATCH --time=24:00:00      # hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
##SBATCH --gpu-bind=none 

#echo $MODULEPATH
module reset
#module load anaconda3_gpu
module load anaconda3_gpu/22.10.0
module list 
#pip3 list | grep tensorflow
#pip3 list | grep scikit-learn
#pip3 install scikit-image
#pip3 install deepxde
#pip3 install tensorflow-probability
which python

export DDE_BACKEND=tensorflow
srun python3 Lug_GRU_DeepXDE_rnd_split_ABS.py

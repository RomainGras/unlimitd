#!/bin/bash

# Slurm sbatch options
# SBATCH -o request.sh.log-%j
# SBATCH --gres=gpu:volta:1
# SBATCH -c 20

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the scriptcd
python train_regression.py --method="UnLiMiTDF" --subspacedim=500 --stop_epoch=10000 --seed=1
python test_regression.py  --method="UnLiMiTDF" --subspacedim=500 --seed=1

# Run the scriptcd
python train_regression.py --method="UnLiMiTDR" --subspacedim=500 --stop_epoch=10000 --seed=1
python test_regression.py  --method="UnLiMiTDR" --subspacedim=500 --seed=1

# Run the scriptcd
python train_regression.py --method="UnLiMiTDI" --stop_epoch=10000 --seed=1
python test_regression.py  --method="UnLiMiTDI" --seed=1

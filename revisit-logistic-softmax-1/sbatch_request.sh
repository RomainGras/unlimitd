#!/bin/bash

# Slurm sbatch options
# SBATCH -o request.sh.log-%j
# SBATCH --gres=gpu:volta:1
# SBATCH -c 20

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the scriptcd
# python train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO"
# python test.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO" --repeat=10

python train.py --dataset="CUB" --method="maml"

python test.py --dataset="CUB" --method="maml"
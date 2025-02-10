#!/bin/bash

# Slurm sbatch options
#SBATCH -o request.sh.log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the scriptcd
# python train_regression.py --method="UnLiMiTDF" --subspacedim=100 --stop_epoch=10000 --seed=1
# python test_regression.py  --method="UnLiMiTDF" --subspacedim=100 --seed=1

# Run the scriptcd
# python train_regression.py --method="UnLiMiTDR" --subspacedim=100 --stop_epoch=10000 --seed=1
# python test_regression.py  --method="UnLiMiTDR" --subspacedim=100 --seed=1

# Run the scriptcd
# python train_regression.py --method="UnLiMiTDI_w_conv_differentiated" --stop_epoch=100 --seed=1
# python test_regression.py  --method="UnLiMiTDI_w_conv_differentiated" --seed=1

python train_regression.py --method="DKT" --stop_epoch=100 --seed=1 --model="Conv3_net"
python test_regression.py  --method="DKT" --seed=1 --model="Conv3_net"

# python train_regression.py --method="DKT_w_net" --stop_epoch=600 --seed=1
# python test_regression.py  --method="DKT_w_net" --seed=1

# python train_regression.py --method="DKT" --stop_epoch=600 --seed=1
# python test_regression.py  --method="DKT" --seed=1

# python train_regression.py --method="UnLiMiTDI_w_conv_differentiated" --stop_epoch=600 --seed=1
# python test_regression.py --method="UnLiMiTDI_w_conv_differentiated" --seed=1 

# python train_regression.py --method="UnLiMiTDIX_w_conv_differentiated" --stop_epoch=600 --seed=1
# python test_regression.py --method="UnLiMiTDIX_w_conv_differentiated" --seed=1 

# python train_regression.py --method="UnLiMiTDFX_w_conv_differentiated" --stop_epoch=600 --seed=1
# python test_regression.py --method="UnLiMiTDFX_w_conv_differentiated" --seed=1 

python train_regression.py --method="DKT" --stop_epoch=100 --dataset="argus" --model="ThreeLayerMLP" --seed=1
python test_regression.py  --method="DKT" --dataset="argus" --model="ThreeLayerMLP" --seed=1

python train_regression.py --method="UnLiMiTDIX" --dataset="argus" --model="ThreeLayerMLP" --stop_epoch=100 --seed=1
python test_regression.py --method="UnLiMiTDIX" --dataset="argus" --model="ThreeLayerMLP" --seed=1


python train_regression.py --method="UnLiMiTDIX" --stop_epoch=100 --seed=1 --dataset="berkeley" --model="ThreeLayerMLP"
python test_regression.py --method="UnLiMiTDIX" --seed=1 --dataset="berkeley" --model="ThreeLayerMLP"
python test_regression.py --method="UnLiMiTDI" --seed=1 --ft

python train_regression.py --method="UnLiMiTDIX" --stop_epoch=100 --seed=1
python test_regression.py --method="UnLiMiTDIX" --seed=1
python test_regression.py --method="UnLiMiTDIX" --seed=1 --ft=True --task_update_num=50

python train_regression.py --method="UnLiMiTDI" --stop_epoch=100 --seed=1
python test_regression.py --method="UnLiMiTDI" --seed=1
python test_regression.py --method="UnLiMiTDI" --seed=1 --ft=True --task_update_num=50

python train_regression.py --method="MAML" --stop_epoch=800 --seed=1

python train_regression.py --method="MAML" --stop_epoch=500 --seed=1 --dataset="argus" --model="ThreeLayerMLP"
python test_regression.py --method="MAML" --task_update_num=50 --dataset="argus" --model="ThreeLayerMLP"
#!/bin/bash

#SBATCH -o %j.log
#SBATCH --gres=gpu:volta:1
#SBATCH --time=72:00:00


# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

python train.py --dataset="miniImagenet" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --stop_epoch=200 --train_aug --model="Conv4_diffDKTIX" --diff_net=no_norm

python test.py --dataset="miniImagenet" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4_diffDKTIX" --diff_net=no_norm 

python test.py --dataset="miniImagenet" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4_diffDKTIX" --diff_net=no_norm --n_ft=100 --lr=.1 --temp=.1 --optim_based_test=True

# python train.py --dataset="omniglot" --method="maml" --stop_epoch=200 --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4S"

# python test.py --dataset="omniglot" --method="maml" --stop_epoch=200 --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4S"

# python test.py --dataset="miniImagenet" --method="maml" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4"

# Script to test diffDKT

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNet10_custom"  

# python test.py --dataset="CUB" --method="differentialDKTIXnogpy" --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4_diffDKTIX" --diff_net=test_4 --n_ft=10 --lr=.01 --temp=.3 --optim_based_test=True

# python train.py --dataset="CUB" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --stop_epoch=200 --train_aug --model="Conv4_diffDKTIX" --diff_net=test_5

# python test.py --dataset="miniImagenet" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNet10_custom"

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNet10_custom"  --n_ft=100 --lr=.01 --temp=.04 --optim_based_test=True

# python test.py --dataset="miniImagenet" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNet10_custom"  --n_ft=500 --lr=.001 --temp=.04 --optim_based_test=True



# Script to test other
# python test.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="ResNet10"

# python test.py --dataset="miniImagenet" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="ResNet10"

# python test.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO" --model="ResNet10"

# python test.py --dataset="miniImagenet" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO" --model="ResNet10"


# python train.py --dataset="miniImagenet" --method="CDKT" --stop_epoch=300 --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO" --model="ResNet10" 

# python train.py --dataset="miniImagenet" --method="DKT" --stop_epoch=300 --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="ResNet10"

# python train.py --dataset="CUB" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --stop_epoch=300 --train_aug --model="identity" --diff_net="combined_Conv4_custom"  

# python train.py --dataset="CUB" --method="differentialDKTIXPL" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --stop_epoch=300 --train_aug --model="combined_Conv4_custom" 

# python train.py --dataset="miniImagenet" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --stop_epoch=300 --train_aug --model="identity" --diff_net="combined_ResNet10_custom"  

# python test.py --dataset="miniImagenet" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNet10_custom" 

# python train.py --dataset="CUB" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNet10_custom"  

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNetNoBN10"

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNetNoBN10" --n_ft=100 --lr=.01 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNetNoBN10" --n_ft=100 --lr=.001 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNetNoBN10" --n_ft=100 --lr=.1 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKTnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_ResNetNoBN10" --n_ft=1000 --lr=.001 --optim_based_test=True

# python train.py --dataset="CUB" --method="differentialDKTIXnogpy" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4_diffDKTIX"  


# Train MAML 1 inner upt, 10^-2 inner lr, 10^-3 outer lr
# python train.py --dataset="miniImagenet" --method="maml" --stop_epoch=200 --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4"

# python test.py --dataset="CUB" --method="maml" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4"

# python test.py --dataset="miniImagenet" --method="maml" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4"

# python maml_to_diffDKTIX.py --dataset="CUB" --method="maml" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4"

# python maml_jac-testing.py --dataset="CUB" --method="maml" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="ResNet10"

# python train.py --dataset="CUB" --method="differentialDKTIXPL" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --model="combined_Conv4NoBN"

# Run the scriptcd
# python train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO" -- model="Conv4NoBN" 

# python test.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=2 --tau=0.2 --loss="ELBO" --repeat=10

# python train.py --dataset="CUB" --method="maml"

# python test.py --dataset="CUB" --method="maml"

#python train.py --dataset="CUB" --method="maml" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="ResNet50"



# python train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4NoBN"

# python test.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4NoBN"

# python train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4NoBN" --kernel="linear" --loss="ELBO" --tau=1 --steps=2

# python test.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="Conv4NoBN" --kernel="linear" --loss="ELBO" --tau=1 --steps=2 



# python train.py --dataset="omniglot" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4S"

# python test.py --dataset="omniglot" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4S"

# python train.py --dataset="omniglot" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4S"

# python test.py --dataset="omniglot" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4S"



# python train.py --dataset="miniImagenet" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4"

# python test.py --dataset="miniImagenet" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4"

# python train.py --dataset="miniImagenet" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4"

# python test.py --dataset="miniImagenet" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="Conv4"




#python train.py --dataset="CUB" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

#python test.py --dataset="CUB" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

# python test.py --dataset="CUB" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=20 --lr=.001 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.01 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.001 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.0001 --optim_based_test=True



# python train.py --dataset="omniglot" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

# python test.py --dataset="omniglot" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

# python test.py --dataset="omniglot" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=20 --lr=.001 --optim_based_test=True

# python test.py --dataset="omniglot" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.01 --optim_based_test=True

# python test.py --dataset="omniglot" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.001 --optim_based_test=True

# python test.py --dataset="omniglot" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.0001 --optim_based_test=True





# python train.py --dataset="miniImagenet" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

# python test.py --dataset="miniImagenet" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

# python test.py --dataset="miniImagenet" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=20 --lr=.001 --optim_based_test=True

# python test.py --dataset="miniImagenet" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.01 --optim_based_test=True

# python test.py --dataset="miniImagenet" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.001 --optim_based_test=True

# python test.py --dataset="miniImagenet" --method="differentialDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.0001 --optim_based_test=True



# python train.py --dataset="omniglot" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="identity" --diff_net="combined_Conv4SNoBN"  

# python test.py --dataset="omniglot" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="identity" --diff_net="combined_Conv4SNoBN"  

# python test.py --dataset="omniglot" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="identity" --diff_net="combined_Conv4SNoBN"  --n_ft=1000 --lr=.001 --optim_based_test=True

# python test.py --dataset="omniglot" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="identity" --diff_net="combined_Conv4SNoBN"  --n_ft=100 --lr=.01 --optim_based_test=True

# python test.py --dataset="omniglot" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="identity" --diff_net="combined_Conv4SNoBN"  --n_ft=100 --lr=.1 --optim_based_test=True

# python test.py --dataset="omniglot" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --model="identity" --diff_net="combined_Conv4SNoBN" --n_ft=50 --lr=.1 --optim_based_test=True



# python train.py --dataset="CUB" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"  

# python test.py --dataset="CUB" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN"

# python test.py --dataset="CUB" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=1000 --lr=.001 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.1 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.001 --optim_based_test=True

# python test.py --dataset="CUB" --method="differentialDKTIX" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --model="identity" --diff_net="combined_Conv4NoBN" --n_ft=100 --lr=.0001 --optim_based_test=True
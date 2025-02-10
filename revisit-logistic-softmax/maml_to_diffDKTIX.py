import torch
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.maml import MAML
from methods.differentialDKTIXnogpytorch import differentialDKTIXnogpy
from io_utils import model_dict, get_resume_file, parse_args, get_best_file , get_assigned_file


def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        if(verbose): print("[INFO] Setting SEED: " + str(seed))   
    else:
        if(verbose): print("[INFO] Setting SEED: None")
        


type_rev_anneal = "inv_exp"
def rev_anneal_lr(x, final_lr, temp=1):
    """
    x (float) : (epoch - start_epoch)/(stop_epoch - start_epoch
    fina_lr (float) : final learning rate target
    """
    if type_rev_anneal == "linear":
        return final_lr * x
    elif type_rev_anneal == "inv_exp":
        alpha = 1/temp
        return final_lr * (1-exp(-alpha*x))/(1-exp(-alpha))
    else:
        return None

type_anneal = "exp"
def anneal_lr(x, init_lr, final_lr, temp=1):
    """
    x (float) : (epoch - start_epoch)/(stop_epoch - start_epoch
    fina_lr (float) : final learning rate target
    """
    if type_anneal == "exp":
        beta = 1/temp
        return (exp(-beta*x)-exp(-beta))/(1-exp(-beta))*(init_lr - final_lr) + final_lr
    else:
        return None

    
def total_l2_norm(tensor_dict):
    total_norm = 0.0
    n_tot = 0
    for tensor in tensor_dict.values():
        # Flatten the tensor and compute its L2 norm (sum of squares of all elements)
        total_norm += tensor.flatten().norm(p=2)**2
        n_tot += tensor.numel()
    # Return the square root of the total sum of squared norms
    return torch.sqrt(total_norm/n_tot)
    
    
def post_training_loop(params):
    
    # First define loaders
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    print(f"n_query : {n_query}")

    # Dataloader
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 224
        
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params) #n_eposide=100
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
    
    # WHERE WE NEED TO ADD BIAS
    # backbone.ConvBlock.maml = True
    # backbone.SimpleBlock.maml = True
    # backbone.BottleneckBlock.maml = True
    # backbone.ResNet.maml = True            
    model = differentialDKTIXnogpy(model_dict[params.model], **train_few_shot_params)
    if params.dataset in ['omniglot', 'cross_char']: # maml use different parameter in omniglot
        model.n_task     = 32
        model.task_update_num = 1
        model.train_lr = 0.1
        
    # Load state_dict
    model = model.cuda()
        
    checkpoint_dir = "./save/checkpoints/CUB/Conv4_maml_aug_5way_1shot"
    
    resume_file = get_resume_file(checkpoint_dir)
    # /!\ CAUTION : get_resume_file does not give the same results in testing that get_best_file, that is used in the test.py
    
    tmp = torch.load(resume_file)
    
    state_dict = tmp['state']
    # new_state_dict = remap_state_dict(old_state_dict)
    
    start_epoch = tmp['epoch'] + 1
    model.load_state_dict(state_dict, strict=False)  # /!\ cAUTION, VERY DANGEROUS STRICT = FALSE
    
    # To make sure that test works well
    print('Bayes Based test')
    model.eval()
    model.test_loop(val_loader)
    
    # seed = 1
    # iter_num = 600
    # repeat = 5
    
    # model.train()
    # accuracy_list = list()
    # for i in range(seed, seed+repeat):
        # if(seed!=0): _set_seed(i)
        # else: _set_seed(0)
        
        # accuracy_list.append(single_test(params, model, optim=True))
        
    # print("-----------------------------")
    # print('Optim : Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    # print("-----------------------------") 
    
    seed = 1
    iter_num = 600
    repeat = 5
    
    print('')
    print('Optim Based test')
    model.train()
    
        
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        
        accuracy_list.append(single_test(params, model, optim=True, n_ft=1))
        
    print("-----------------------------")
    print('Optim n_ft = 1: Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------") 
    
    
    
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        
        accuracy_list.append(single_test(params, model, optim=True, n_ft=10))
        
    print("-----------------------------")
    print('Optim n_ft = 10: Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------") 
    model.test_loop(val_loader, optim_based=True, n_ft=10, lr=0.01, temp=1, return_std = True)
    
    print('')
    print('=================================')
    print('')
    print('Begining post-training')
    print('')
    
    torch.cuda.empty_cache()
    
    # Scaling params initially won't require grad
    for value in model.scaling_params.values():
        value.requires_grad_(True)
    print(f'Scaling params require grad : {all([value.requires_grad for value in model.scaling_params.values()])}')

    # Optimizer :
    optimizer = torch.optim.Adam([{'params': model.scaling_params.values(), 'lr': 1e-3}, 
                                  {'params': model.feature.parameters(), 'lr': 0}])
    
    
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    
    final_sp_lr = optimizer.param_groups[0]['lr']
    max_acc = 0
    for epoch in range(start_epoch, stop_epoch):
        
        model.noise = .1
        
        # If reverse annealing, do reverse annealing for sigma
        reverse_annealing = False
        if reverse_annealing:
            optimizer.param_groups[0]['lr'] = rev_anneal_lr((epoch-start_epoch)/(stop_epoch-start_epoch), final_sp_lr, temp=1/1)
            
        annealing = False
        if annealing:
            optimizer.param_groups[1]['lr'] = anneal_lr((epoch-start_epoch)/(stop_epoch-start_epoch), .01, .005, temp=1/10)
            
        noise_annealing = False
        if noise_annealing:
            model.noise = anneal_lr((epoch-start_epoch)/(stop_epoch-start_epoch), .1, 0, temp=1/10)
        
        print(optimizer.param_groups[0]['lr'])
        print(optimizer.param_groups[1]['lr'])
        print(model.noise)
        
        model.train()
        model.train_loop(epoch, base_loader, optimizer)
        
        # total_l2_sp = total_l2_norm(model.scaling_params)
        
        # Simple Meta-Regularizer ?
        # model.scaling_params = {k: v/total_l2_sp for k, v in model.scaling_params.items()}
        
        print("scaling params")
        print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in model.scaling_params.items()})
        print({k: torch.min(v) for k, v in model.scaling_params.items()})
        
        model.eval()
        print('')
        print('Bayes test loop')
        model.test_loop(val_loader)
        print('')
        print('Optim test loop')
        model.train()
        model.test_loop( val_loader, optim_based=True, n_ft=10, lr=0.01, temp=1, return_std = True)
        
    seed = 1
    iter_num = 600
    repeat = 5
    
    # model.eval()
    
    # accuracy_list = list()
    # for i in range(seed, seed+repeat):
    #     if(seed!=0): _set_seed(i)
    #     else: _set_seed(0)
        
    #     accuracy_list.append(single_test(params, model, optim=False))
        
    # print("-----------------------------")
    # print('Bayes : Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    # print("-----------------------------") 
    
    
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        
        accuracy_list.append(single_test(params, model, optim=True, n_ft=1))
        
    print("-----------------------------")
    print('Optim n_ft = 1: Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------") 
    
    
    
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        
        accuracy_list.append(single_test(params, model, optim=True, n_ft=10))
        
    print("-----------------------------")
    print('Optim n_ft = 10: Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------") 
    
    
    
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        
        accuracy_list.append(single_test(params, model, optim=True, n_ft=100))
        
    print("-----------------------------")
    print('Optim n_ft = 100: Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------") 
    
    
    

def single_test(params, model, optim=False, n_ft=10, lr=.01, temp=1):
    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    
    split = 'novel'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 224

    datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)

    loadfile    = configs.data_dir[params.dataset] + split + '.json'

    novel_loader     = datamgr.get_data_loader( loadfile, aug = False)

    acc_mean, acc_std = model.test_loop( novel_loader, optim_based=optim, n_ft=n_ft, lr=lr, temp=temp, return_std = True)

    return acc_mean


def main():
    device = 'cuda:0'
    params = parse_args('maml_to_diffDKTIX')     
    
    print(f"Params : { params }")
    
    seed = params.seed
    _set_seed(seed)
    post_training_loop(params)
    
    
    repeat = params.repeat
    #repeat the test N times changing the seed in range [seed, seed+repeat]
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        # accuracy_list.append(single_test(parse_args('test')))
        accuracy_list.append(single_test(params))
    print("-----------------------------")
    print('Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------")        
if __name__ == '__main__':
    main()

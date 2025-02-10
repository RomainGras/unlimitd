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
        
        
        
def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc



def single_test(params):
    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    if params.dataset in ['omniglot', 'cross_char']:
        print(f"Model : {params.model}")
        print(f"Train aug : {params.train_aug}")
        assert ('Conv4S' in params.model or 'Conv4S' in params.diff_net) and not params.train_aug, 'omniglot only support Conv4S without augmentation'
        # params.model = 'Conv4S'   #Change of model for a Conv4S

    if params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: # maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    #checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    
    #if params.train_aug:
    #    checkpoint_dir += '_aug'
    #checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    #modelfile   = get_resume_file(checkpoint_dir)
        
        
    checkpoint_dir = "./save/saved_checkpoints/ResNet10_maml_aug_5way_1shot_nospetraining" # "./save/checkpoints/CUB/ResNet10_maml_aug_5way_1shot"
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    else:
        modelfile   = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))
    

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 84 #224

    datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)

    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = configs.data_dir['CUB'] + split +'.json'
    elif params.dataset == 'cross_char':
        if split == 'base':
            loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
        else:
            loadfile  = configs.data_dir['emnist'] + split +'.json' 
    else: 
        loadfile    = configs.data_dir[params.dataset] + split + '.json'

    novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
    if params.adaptation:
        model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
    model.eval()

    acc_mean, acc_std = model.test_loop( novel_loader, return_std = True) #, jac_test=False

    return acc_mean


def main():
    device = 'cuda:0'
    params = parse_args('test')     
    
    print(f"Params : { params }")
    
    seed = params.seed
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

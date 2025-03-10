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
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.DKT import DKT
from methods.CDKT import CDKT
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.differentialCDKT import differentialCDKT
from methods.differentialDKT import differentialDKT
from methods.differentialDKTIXnogpytorch import differentialDKTIXnogpy
from methods.differentialDKTnogpytorch import differentialDKTnogpy
from methods.differentialDKTIX import differentialDKTIX
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
        print(f"Diff net : {params.diff_net}")
        print(f"Train aug : {params.train_aug}")
        assert ('Conv4S' in params.model or 'Conv4S' in params.diff_net) and not params.train_aug, 'omniglot only support Conv4S without augmentation'
        # params.model = 'Conv4S'   #Change of model for a Conv4S

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'DKT':
        model           = DKT(model_dict[params.model], **few_shot_params)
    elif params.method == 'CDKT':
        model           = CDKT(model_dict[params.model], **few_shot_params)
    elif params.method == 'differentialCDKT':
        model           = differentialCDKT(model_dict[params.model], model_dict[params.diff_net], **few_shot_params)
    elif params.method == 'differentialDKT':
        model           = differentialDKT(model_dict[params.model], model_dict[params.diff_net], **few_shot_params)
    elif params.method == 'differentialDKTIX':
        model           = differentialDKTIX(model_dict[params.model], model_dict[params.diff_net], **few_shot_params)
        print("scaling params")
        print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in model.scaling_params.items()})
        print({k: torch.min(v) for k, v in model.scaling_params.items()})
        
    elif(params.method == 'differentialDKTIXnogpy'):
        model           = differentialDKTIXnogpy(model_dict[params.model], **few_shot_params)
        
    elif(params.method == 'differentialDKTnogpy'):
        # print(type(model_dict[params.diff_net]))
        model           = differentialDKTnogpy(model_dict[params.model], model_dict[params.diff_net], **few_shot_params)

    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6': 
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S': 
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    
    if 'differential' in params.method:
        checkpoint_dir += f'_{params.diff_net}'
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if params.method in ['CDKT', 'differentialCDKT']:
        tau = str(params.tau).replace('.', 'dot')
        checkpoint_dir += '_%s_%stau_%dsteps' % (params.loss, tau, params.steps)
        if params.mean < 0:
            checkpoint_dir += '_negmean'
        if params.mean > 0:
            mean = str(params.mean).replace('.', 'dot')
            checkpoint_dir += '_%smean' % (mean)
        checkpoint_dir += '_%s' % (params.kernel)

    #modelfile   = get_resume_file(checkpoint_dir)
    if params.method in ['CDKT']:
        model.get_steps(params.steps)
        model.get_temperature(params.tau)
        model.get_loss(params.loss)
        model.get_negmean(params.mean)
        model.get_kernel_type(params.kernel)
        # shift
        model.get_negmean(-5)
    
    if params.method in ['differentialCDKT']:
        model.get_steps(params.steps)
        model.get_temperature(params.tau)
        model.get_loss(params.loss)
        model.get_negmean(params.mean)
        model.get_kernel_type(params.kernel)
        # shift
        model.get_negmean(-5)
        
    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            print(checkpoint_dir)
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            # tmp_test = torch.load("./save/checkpoints/CUB/identity_differentialDKTIX_aug_5way_1shot/test.tar")
            # print(any(['scaling_params' in str for str in tmp_test['state'].keys()]))
            if params.method in ['differentialDKTIX', 'differentialDKTIXnogpy']:
                model.scaling_params = tmp['sp']
                
                print("scaling params")
                print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in model.scaling_params.items()})
                print({k: torch.min(v) for k, v in model.scaling_params.items()})
            model.load_state_dict(tmp['state'])
        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))
    
    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    if params.method in ['maml', 'maml_approx', 'DKT', 'CDKT', 'differentialCDKT', 'differentialDKT', 'differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTnogpy']: #maml do not support testing with feature
        if 'Conv' in params.model or (params.diff_net and 'Conv' in params.diff_net):
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84 
        else:
            image_size = 224

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
        
        if params.method in ['differentialDKT', 'differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTnogpy'] and params.optim_based_test:
            acc_mean, acc_std = model.test_loop( novel_loader, optim_based=True, n_ft=params.n_ft, lr=params.lr, temp=params.temp, return_std = True) #, jac_test=False
        else:
            acc_mean, acc_std = model.test_loop( novel_loader, return_std = True) #, jac_test=False
            

    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++'] :
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
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

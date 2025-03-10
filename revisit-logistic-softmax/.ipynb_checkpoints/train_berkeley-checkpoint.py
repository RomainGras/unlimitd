from regression_datasets import provide_data
import numpy as np
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, CyclicLR
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.DKT import DKT
from methods.CDKT import CDKT
from methods.differentialCDKT import differentialCDKT
from methods.differentialCDKTIX import differentialCDKTIX
from methods.differentialDKT import differentialDKT
from methods.differentialDKTIX import differentialDKTIX
from methods.differentialDKTIXnogpytorch import differentialDKTIXnogpy
from methods.differentialDKTnogpytorch import differentialDKTnogpy
from methods.differentialDKTIXPL import differentialDKTIXPL
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.Imaml import iMAML
from io_utils import model_dict, parse_args, get_resume_file

from math import exp

meta_train_data, meta_valid_data, meta_test_data = provide_data(dataset='berkeley', data_dir="./filelists/sensor_data/")


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


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    print("Tot epochs: " + str(stop_epoch))
    if optimization == 'Adam':
        if params.method in ['CDKT', 'differentialDKT', 'differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTnogpy', 'differentialCDKT', 'differentialCDKTIX', 'differentialDKTIXPL']:
            if params.method in ['CDKT', 'differentialCDKT', 'differentialCDKTIX'] : 
                flag = model.get_negmean(params.mean)
                model.get_kernel_type(model.kernel_type)
            for name, param in model.named_parameters():
                print(f"Parameter name: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Parameter leaf: {param.is_leaf}\n")
        
            # for single_model in model.model.kernels:
            #     print(f"Parameter of kernel name: {single_model.covar_module.base_kernel.parameters()}")
            #     print(f"sp : {single_model.covar_module.base_kernel.sp}")
            #    for name, param in single_model.covar_module.base_kernel.named_parameters():
            #         print(f"Parameter name: {name}")
            #         print(f"Parameter shape: {param.shape}\n")
                
                
            if params.method in ['CDKT', 'differentialCDKT', 'differentialCDKTIX'] and flag:
                optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 1e-4},
                                              {'params': model.feature_extractor.parameters(), 'lr': 1e-3},
                                              {'params': model.NEGMEAN, 'lr': 1e-4}])
            
            
            elif params.method in ['differentialDKTIXPL']:  ## TODO REMOVE THIS WHEN DONE WITH TESTING THE PARAMETERS
                for value in model.scaling_params.values():
                    value.requires_grad_(True)
                    print(value.is_leaf)
                    
                outer_lr = 1e-3
                outer_lr_sp = 1e-3
                optimizer_sp = torch.optim.Adam([{'params': model.scaling_params.values(), 'lr': outer_lr}])
                optimizer_feat_extr = torch.optim.Adam([{'params': model.feature.parameters(), 'lr': outer_lr_sp}])
                
                print(f"Outer LR {outer_lr}")
                print(f"Outer LR SP {outer_lr_sp}")
                
                
            elif params.method in ['differentialDKTIXnogpy']:
                # optimizer_params = []
                # optimizer_params.append({'params': model.parameters(), 'lr':1e-4})
                # for param in model.scaling_params.values():
                #     optimizer_params.append({'params': param, 'lr':1e10})
                # optimizer = torch.optim.Adam(optimizer_params +
                #                               [{'params': model.feature_extractor.parameters(), 'lr': 1e-3}])
                for value in model.scaling_params.values():
                    value.requires_grad_(True)
                    print(value.is_leaf)
                    
                optimizer_sp = torch.optim.Adam([{'params': model.scaling_params.values(), 'lr': 1e-2}])
                optimizer_feat_extr = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 1e-2}])
            
            elif params.method in ['differentialDKTnogpy']:
                optimizer = torch.optim.Adam([{'params': model.diff_net.parameters(), 'lr': 1e-4},
                                              {'params': model.feature_extractor.parameters(), 'lr': 1e-3}])         
            elif 'IX' in params.method:
                # optimizer_params = []
                # optimizer_params.append({'params': model.parameters(), 'lr':1e-4})
                # for param in model.scaling_params.values():
                #     optimizer_params.append({'params': param, 'lr':1e10})
                # optimizer = torch.optim.Adam(optimizer_params +
                #                               [{'params': model.feature_extractor.parameters(), 'lr': 1e-3}])
                for value in model.scaling_params.values():
                    value.requires_grad_(True)
                    print(value.is_leaf)
                    
                optimizer = torch.optim.Adam([{'params': model.scaling_params.values(), 'lr': 1e-4},
                                              {'params': model.model.parameters(), 'lr': 1e-4},
                                              {'params': model.feature_extractor.parameters(), 'lr': 1e-3}])
                
            else:
                optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 1e-4},
                                              {'params': model.feature_extractor.parameters(), 'lr': 1e-3}])
            
            if params.method in ['CDKT', 'differentialCDKT', 'differentialCDKTIX'] : 
                model.get_steps(params.steps)
                model.get_temperature(params.tau)
                model.get_loss(params.loss)
                # model.get_kernel_type(params.kernel)

        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            print(f"Meta-lr : {optimizer.defaults['lr']}")
            if params.method=='imaml':
                inner_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                model.set_inner_optimizer(inner_optimizer)
    else:
        raise ValueError('Unknown optimization, please define by yourself')
    
    max_acc = 0
    
    init_sp_lr = optimizer_sp.param_groups[0]['lr']
    final_sp_lr = 1e-4
    for epoch in range(start_epoch, stop_epoch):
        
        # If reverse annealing, do reverse annealing for sigma
        sp_annealing = True
        if params.method in ['differentialDKTIXnogpy', 'differentialDKTIXPL'] and sp_annealing:
            optimizer_sp.param_groups[0]['lr'] = anneal_lr((epoch-start_epoch)/(stop_epoch-start_epoch), init_sp_lr, final_sp_lr, temp=1/10)
            
        params_annealing = True
        if params.method in ['differentialDKTIXnogpy', 'differentialDKTIXPL'] and params_annealing:
            optimizer_feat_extr.param_groups[0]['lr'] = anneal_lr((epoch-start_epoch)/(stop_epoch-start_epoch), .01, .005, temp=1/10)
            
        noise_annealing = True
        if params.method in ['differentialDKTIXnogpy', 'differentialDKTIXPL'] and noise_annealing:
            model.noise = anneal_lr((epoch-start_epoch)/(stop_epoch-start_epoch), .1, 0, temp=1/10)
        
        print("sp lr: ", optimizer_sp.param_groups[0]['lr'])
        print("optim lr: ", optimizer_feat_extr.param_groups[0]['lr'])
        print("noise: ", model.noise)
        if params.method in ['differentialDKTIXnogpy', 'differentialDKTIXPL']:
            print("scaling params")
            print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in model.scaling_params.items()})
            print({k: torch.min(v) for k, v in model.scaling_params.items()})
            
            optimizer = (optimizer_sp, optimizer_feat_extr)
            
        model.train()
        model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        if epoch > params.stop_epoch - 20:          
            acc = model.test_loop(val_loader)
        elif params.dataset in ['cross_char']:
            acc = model.test_loop(val_loader)
        else:
            acc = 0.
        
        if epoch%10==0:
            acc = model.test_loop(val_loader)
            outfile = os.path.join(params.checkpoint_dir, f'{epoch}.tar')
            if params.method in ['differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTIXPL']:
                torch.save({'epoch': epoch, 'state': model.state_dict(), 'sp': model.scaling_params}, outfile)
            else:
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            
        # if epoch%50==0:
        #     acc = model.test_loop(val_loader, jac_test=True)
            
        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("--> Best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            if params.method in ['differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTIXPL']:
                torch.save({'epoch': epoch, 'state': model.state_dict(), 'sp': model.scaling_params}, outfile)
            else:
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            if params.method in ['differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTIXPL']:
                torch.save({'epoch': epoch, 'state': model.state_dict(), 'sp': model.scaling_params}, outfile)
            else:
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    device = 'cuda:0'
    params = parse_args('train')
    # train classification configuration
    params.train_n_way = 5
    params.test_n_way = 5
    print(params)
    #
    _set_seed(parse_args('train').seed)
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'
    
    if 'Conv' in params.model or (params.diff_net and 'Conv' in params.diff_net):
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224     # Originally 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert ('Conv4S' in params.model or 'Conv4S' in params.diff_net) and not params.train_aug, 'omniglot only support Conv4S without augmentation'
        # params.model = 'Conv4S'   #Change of model for a Conv4S

    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 600  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 # default
    
    if params.loss == "PLL":
        params.stop_epoch = 800
        print("Update stop_epoch: {}".format(params.stop_epoch))
    
    if params.dataset == "miniImagenet":
        if params.n_shot == 5:
            params.stop_epoch = 800
            print("Update stop_epoch: {}".format(params.stop_epoch))
            
    if params.dataset == "cross":
        if params.n_shot == 1:
            if params.loss == "ELBO":
                params.stop_epoch = 800
                print("Update stop_epoch: {}".format(params.stop_epoch))

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

    elif params.method in ['differentialDKT', 'differentialDKTIX', 'differentialDKTIXnogpy', 'differentialDKTnogpy', 'differentialCDKT', 'differentialCDKTIX', 'differentialDKTIXPL', 'CDKT', 'DKT', 'protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx', 'imaml']:
        n_query = max(1, int(
            16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print(f"n_query : {n_query}")
            
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params) #n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
        
        if(params.method == 'DKT'):
            model = DKT(model_dict[params.model], **train_few_shot_params)
            model.init_summary()
        elif(params.method == 'CDKT'):
            model = CDKT(model_dict[params.model], **train_few_shot_params)
            model.init_summary()
        elif(params.method == 'differentialDKT'):
            # print(type(model_dict[params.diff_net]))
            model = differentialDKT(model_dict[params.model], model_dict[params.diff_net], **train_few_shot_params)
        elif(params.method == 'differentialDKTIX'):
            # print(type(model_dict[params.diff_net]))
            model = differentialDKTIX(model_dict[params.model], model_dict[params.diff_net], **train_few_shot_params)
        elif(params.method == 'differentialDKTIXnogpy'):
            model = differentialDKTIXnogpy(model_dict[params.model], **train_few_shot_params)
        elif(params.method == 'differentialDKTnogpy'):
            # print(type(model_dict[params.diff_net]))
            model = differentialDKTnogpy(model_dict[params.model], model_dict[params.diff_net], **train_few_shot_params)
        elif(params.method == 'differentialCDKT'):
            # print(type(model_dict[params.diff_net]))
            model = differentialCDKT(model_dict[params.model], model_dict[params.diff_net], **train_few_shot_params)
            model.init_summary()
        elif(params.method == 'differentialCDKTIX'):
            model = differentialCDKTIX(model_dict[params.model], model_dict[params.diff_net], **train_few_shot_params)
            model.init_summary()
        elif params.method == 'differentialDKTIXPL':
            model = differentialDKTIXPL(model_dict[params.model], **train_few_shot_params)  # /!\ DIFF NET = PARAMS.MODEL
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
            # model.init_summary()   # TODO INVESTIGATE
            
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
        elif params.method in ['imaml']:
            backbone.ConvBlock.maml = False
            backbone.SimpleBlock.maml = False
            backbone.BottleneckBlock.maml = False
            backbone.ResNet.maml = False
            model = iMAML(model_dict[params.model], approx=(params.method == 'imaml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.to(device)

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if 'differential' in params.method:
        params.checkpoint_dir += f'_{params.diff_net}'
        
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if params.method in ['CDKT', 'differentialCDKT', 'differentialCDKTIX']:
        tau = str(params.tau).replace('.', 'dot')
        params.checkpoint_dir += '_%s_%stau_%dsteps' % (params.loss, tau, params.steps)
        if params.mean < 0:
            params.checkpoint_dir += '_negmean'
        if params.mean > 0:
            mean = str(params.mean).replace('.', 'dot')
            params.checkpoint_dir += '_%smean' % (mean)
        params.checkpoint_dir += '_%s' % (params.kernel)
    
    print(params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method in ['maml', 'maml_approx', 'imaml', 'imaml_approx', 'differentialDKTIXPL']:
        stop_epoch = params.stop_epoch * model.n_task  # maml-like use multiple tasks in one update

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
            if 'IX' in params.method:
                model.scaling_params = tmp['sp']
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.",
                                         "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    # model = torch.compile(model)
    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)


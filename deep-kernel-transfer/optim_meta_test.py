import torch
import torch.nn as nn
import torch.optim as optim
import configs

# from data.qmul_loader import get_batch, train_people, test_people
from data.regression_data_loader import data_provider

from io_utils import parse_args_regression, get_resume_file
from methods.UnLiMiTDI_regression import UnLiMiTDI
from methods.UnLiMiTDR_regression import UnLiMiTDR
from methods.UnLiMiTDproj_regression import UnLiMiTDproj
from methods.UnLiMiTDIX_regression import UnLiMiTDIX
from methods.UnLiMiTDprojX_regression import UnLiMiTDprojX
from methods.maml import MAML
from projection import create_random_projection_matrix, proj_sketch
import backbone
import os
import numpy as np

import matplotlib.pyplot as plt

params = parse_args_regression('train_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

provider = data_provider(params.dataset)

    
print(f"Using already trained models of MAML and Ours to perform optimization meta-testing")



print_every = 10  # Collect data every 10 steps of adaptation
total_adapt_steps = 100  # Total meta-testing-steps
for inn_steps in [1, 3]:
    bb = select_model(dataset=params.dataset, method=params.method, model=params.model)
    print(f"Meta-testing MAML {inn_steps} inner steps")
    model = MAML(bb, -1, problem="regression").cuda()   # n_support is -1 because it's directly implemented in loader
    params.checkpoint_dir += f"_{inn_steps}_inn_steps"
    
    model.task_update_num = total_adapt_steps
    optimizer = None
    
    mses_per_task=[]
    for epoch in range(params.n_test_epochs):
        mse_per_step = model.test_loop(params.n_support, provider, optimizer = optimizer, print_every=print_every).cpu().detach().numpy()
        mse_per_step = [float(mse.cpu().detach().numpy()) for mse in mse_per_step]
        mses_per_task.append(mse_per_step)
        
    # Need to compute the mse_mean and mse_std per adaptation step, wrt the tasks : put everything into a notebook to test it
        

# if params.method=='MAML':
#     model = MAML(bb, n_support=9, approx=(params.method == 'maml_approx'), problem = "regression").cuda()
#     optimizer = torch.optim.Adam([{'params': bb.parameters(), 'lr': 0.001}])
#     for epoch in range(params.stop_epoch):
#         model.train_loop_regression(epoch, provider, optimizer)
#         
#     params.checkpoint_dir += f"_{model.task_update_num}_inn_steps"
#         
# elif params.method=='UnLiMiTDI_w_conv_differentiated':
#     model = UnLiMiTDI(None, bb).cuda()
#     optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001}])
#     for epoch in range(params.stop_epoch):
#         model.train_loop(epoch, provider, optimizer)
#         
# elif params.method=='UnLiMiTDIX_w_conv_differentiated':
#     model = UnLiMiTDIX(None, bb).cuda()
#     optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
#     for epoch in range(params.stop_epoch):
#         model.train_loop(epoch, provider, optimizer)
# else:
#     ValueError('Unrecognised method')
    
    

model.save_checkpoint(params.checkpoint_dir)





def select_model(dataset, method, model="Conv3"):
    if dataset == "QMUL":
        if method == "DKT":
            bb               = backbone.Conv3().cuda()
            if model=="Conv3_net":
                simple_net_multi = backbone.simple_net_multi_output()        
                bb = backbone.CombinedNetwork(bb, simple_net_multi).cuda()
        elif model == "Conv3" and method == "MAML":
            backbone.Conv3.maml = True
            backbone.simple_net.maml = True
            bb               = backbone.Conv3().cuda()
            bb               = backbone.CombinedNetwork(bb, backbone.simple_net()).cuda()  # nn.Linear(2916, 1)
        elif model == "Conv3" and "UnLiMiTD" in method:
            bb               = backbone.Conv3().cuda()
            bb               = backbone.CombinedNetwork(bb, backbone.simple_net()).cuda()  # nn.Linear(2916, 1)
        else:
            raise ValueError("Model not recognized")

    elif dataset in ("berkeley", "argus"):

        if dataset == "berkeley":
            input_dim=11
        else:
            input_dim=3

        if model == "ThreeLayerMLP" and method in ("DKT"):
            bb = backbone.ThreeLayerMLP(input_dim=input_dim, output_dim=32)
        elif model == "ThreeLayerMLP" and method in ("MAML"):
            backbone.ThreeLayerMLP.maml = True
            bb = backbone.ThreeLayerMLP(input_dim=input_dim, output_dim=32)
        elif model == "ThreeLayerMLP":
            bb = backbone.ThreeLayerMLP(input_dim=input_dim, output_dim=1)
        elif model == "SteinwartMLP" and method in ("DKT"):
            bb = backbone.SteinwartMLP(input_dim=input_dim, output_dim=32)
        elif model == "SteinwartMLP" and method in ("MAML"):
            backbone.SteinwartMLP.maml = True
            bb = backbone.SteinwartMLP(input_dim=input_dim, output_dim=32)
        elif model == "SteinwartMLP":
            bb = backbone.SteinwartMLP(input_dim=input_dim, output_dim=1)
        else:
            raise ValueError("Model not recognized")

    else:
        raise ValueError("Dataset not recognized")
        
    return bb

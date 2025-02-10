import torch
import torch.nn as nn
import torch.optim as optim
import configs

# from data.qmul_loader import get_batch, train_people, test_people
from data.regression_data_loader import data_provider

from io_utils import parse_args_regression, get_resume_file
from methods.DKT_regression import DKT
from methods.feature_transfer_regression import FeatureTransfer
from methods.UnLiMiTDR_regression import UnLiMiTDR
from methods.UnLiMiTDproj_regression import UnLiMiTDproj
from methods.UnLiMiTDIX_regression import UnLiMiTDIX
from methods.UnLiMiTDprojX_regression import UnLiMiTDprojX
from methods.maml import MAML
from projection import create_random_projection_matrix, proj_sketch
import backbone
import os
import numpy as np

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

if params.dataset == "QMUL":
    if params.method == "DKT":
        bb               = backbone.Conv3().cuda()
        if params.model=="Conv3_net":
            simple_net_multi = backbone.simple_net_multi_output()        
            bb = backbone.CombinedNetwork(bb, simple_net_multi).cuda()
    elif params.model == "Conv3" and params.method == "MAML":
        backbone.Conv3.maml = True
        backbone.simple_net.maml = True
        bb               = backbone.Conv3().cuda()
        bb               = backbone.CombinedNetwork(bb, backbone.simple_net()).cuda()  # nn.Linear(2916, 1)
    elif params.model == "Conv3" and "UnLiMiTD" in params.method:
        bb               = backbone.Conv3().cuda()
        if not params.conv_net_not_differentiated: # Conv net is differentiated
            bb               = backbone.CombinedNetwork(bb, backbone.simple_net()).cuda()  # nn.Linear(2916, 1)
    else:
        raise ValueError("Model not recognized")

elif params.dataset in ("berkeley", "argus"):
    
    if params.dataset == "berkeley":
        input_dim=11
    else:
        input_dim=3
        
    if params.model == "ThreeLayerMLP" and params.method in ("DKT"):
        bb = backbone.ThreeLayerMLP(input_dim=input_dim, output_dim=32)
    elif params.model == "ThreeLayerMLP" and params.method in ("MAML"):
        backbone.ThreeLayerMLP.maml = True
        bb = backbone.ThreeLayerMLP(input_dim=input_dim, output_dim=32)
    elif params.model == "ThreeLayerMLP":
        bb = backbone.ThreeLayerMLP(input_dim=input_dim, output_dim=1)
    elif params.model == "SteinwartMLP" and params.method in ("DKT"):
        bb = backbone.SteinwartMLP(input_dim=input_dim, output_dim=32)
    elif params.model == "SteinwartMLP" and params.method in ("MAML"):
        backbone.SteinwartMLP.maml = True
        bb = backbone.SteinwartMLP(input_dim=input_dim, output_dim=32)
    elif params.model == "SteinwartMLP":
        bb = backbone.SteinwartMLP(input_dim=input_dim, output_dim=1)
    else:
        raise ValueError("Model not recognized")

else:
    raise ValueError("Dataset not recognized")
    
if params.method == "MAML":
    if params.dataset=="QMUL":
        n_support=9
        n_task=8
        train_lr=0.01
        outer_lr=0.001
        stop_epoch=n_task*100 # =800
    elif params.dataset=="berkeley":
        n_support=10
        n_task=12
        train_lr=1e-4
        outer_lr=1e-4
        stop_epoch=n_task*100 # =1200
    elif params.dataset=="argus":
        n_support=50
        n_task=5
        train_lr=1e-5
        outer_lr=1e-5
        stop_epoch=n_task*100 # =500
    else:
        raise ValueError("Dataset not recognized")
        

print(f"This is {params.method}, with {params.stop_epoch} epochs, and kernel {configs.kernel_type}")

if params.method=='MAML':
    model = MAML(bb, n_support=n_support, approx=(params.method == 'maml_approx'), problem = "regression").cuda()
    optimizer = torch.optim.Adam([{'params': bb.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop_regression(epoch, provider, optimizer)
    params.checkpoint_dir += f"_{model.task_update_num}_inn_steps"
    print(params.checkpoint_dir)

if params.method=='DKT':
    model = DKT(bb).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, provider, optimizer)
        
elif params.method=='transfer':
    model = FeatureTransfer(bb).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)
        
elif params.method=='UnLiMiTDI':
    if params.conv_net_not_differentiated:
        model = UnLiMiTDIX(bb, backbone.simple_net(), has_scaling_params=False).cuda()
    else:
        model = UnLiMiTDIX(None, bb, has_scaling_params=False).cuda()    
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, provider, optimizer)
        
elif params.method=='UnLiMiTDIX':
    if params.conv_net_not_differentiated:
        model = UnLiMiTDIX(bb, backbone.simple_net(), has_scaling_params=True).cuda()
    else:
        model = UnLiMiTDIX(None, bb, has_scaling_params=True).cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': model.model.covar_module.scaling_params.values(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, provider, optimizer)

elif params.method=='UnLiMiTDR':
    input_dimension = sum(p.numel() for p in simple_net.parameters())
    P = create_random_projection_matrix(input_dimension, params.subspacedim).cuda()
    model = UnLiMiTDproj(bb, simple_net, P).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)
        
elif params.method=='UnLiMiTDF':
    # Unlimitd-I training
    model = UnLiMiTDI(bb, simple_net).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch//2):
        model.train_loop(epoch, optimizer)
    model.save_checkpoint(params.checkpoint_dir)
        
    # FIM proj search needs no gradient
    for param in model.model.parameters():
        param.requires_grad_(False)
    for param in model.feature_extractor.parameters():
        param.requires_grad_(False)
    optimizer = None
    # Batch preparation
    nb_batch_proj = 10
    
    batches = []
    for _ in range(nb_batch_proj):
        batch, batch_labels = get_batch(train_people)
        for person_task in batch :
            person_conv = model.feature_extractor(person_task.cuda()).detach()
            batches.append(person_conv)  
    batches = torch.stack(batches)
    # FIM projection computation
    input_dimension = sum(p.numel() for p in simple_net.parameters())
    P = proj_sketch(model.diff_net, batches, params.subspacedim).cuda()
    # Gradients back to training mode
    for param in model.model.parameters():
        param.requires_grad_(True)
    for param in model.feature_extractor.parameters():
        param.requires_grad_(True)
    
    # Unlimitd-F training
    model = UnLiMiTDproj(bb, simple_net, P).cuda()
    model.load_checkpoint(params.checkpoint_dir)
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])

elif params.method=='UnLiMiTDFX':
    print("Not Implemented YET")
    
elif params.method=='UnLiMiTDFX_w_conv_differentiated':
    model = UnLiMiTDI(None, combined_network, subspace_dim = params.subspacedim).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch//2):
        model.train_loop(epoch, optimizer)
    model.save_checkpoint(params.checkpoint_dir)
        
    # FIM proj search needs no gradient
    for param in model.model.parameters():
        param.requires_grad_(False)
    optimizer = None
    # Batch preparation
    nb_batch_proj = 10
    
    batches = []
    for _ in range(nb_batch_proj):
        batch, batch_labels = get_batch(train_people)
        for person_task in batch :
            person_task = person_task.cuda().detach()
            batches.append(person_task)  
    batches = torch.stack(batches)
    # FIM projection computation
    P = proj_sketch(model.diff_net, batches, params.subspacedim).cuda()
    # Gradients back to training mode
    for param in model.model.parameters():
        param.requires_grad_(True)
    
    # Unlimitd-F training
    model = UnLiMiTDprojX(None, combined_network, P).cuda()
    model.load_checkpoint(params.checkpoint_dir)
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch//2):
        model.train_loop(epoch, optimizer)

#elif params.method=='UnLiMiTDF_zero_order':
#    model = UnLiMiTDF_zero_order(bb).cuda()
#elif params.method=='UnLiMiTDF_zero_order_CosSim':
#    model = UnLiMiTDF_zero_order_CosSim(bb).cuda()
else:
    ValueError('Unrecognised method')

model.save_checkpoint(params.checkpoint_dir)

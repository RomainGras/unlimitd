import torch
import torch.nn as nn
import torch.optim as optim
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.DKT_regression import DKT
from methods.feature_transfer_regression import FeatureTransfer
from methods.UnLiMiTDI_regression import UnLiMiTDI
from methods.UnLiMiTDR_regression import UnLiMiTDR
from methods.UnLiMiTDproj_regression import UnLiMiTDproj
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

bb               = backbone.Conv3().cuda()
simple_net       = backbone.simple_net().cuda()
simple_net_multi = backbone.simple_net_multi_output().cuda()

combined_network       = backbone.CombinedNetwork(bb, simple_net).cuda()
combined_network_multi = backbone.CombinedNetwork(bb, simple_net_multi).cuda()


if params.method=='DKT':
    model = DKT(bb).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)

elif params.method=='transfer':
    model = FeatureTransfer(bb).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)


elif params.method=='DKT_w_net':
    model = DKT(combined_network).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)

elif params.method=='DKT_w_net_multi':
    model = DKT(combined_network_multi).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)

elif params.method=='UnLiMiTDI':
    model = UnLiMiTDI(bb, simple_net).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)
#elif params.method=='UnLiMiTDF':
#    model = UnLiMiTDF(bb).cuda()

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
    for epoch in range(params.stop_epoch//2):
        model.train_loop(epoch, optimizer)
#elif params.method=='UnLiMiTDF_zero_order':
#    model = UnLiMiTDF_zero_order(bb).cuda()
#elif params.method=='UnLiMiTDF_zero_order_CosSim':
#    model = UnLiMiTDF_zero_order_CosSim(bb).cuda()
else:
    ValueError('Unrecognised method')

model.save_checkpoint(params.checkpoint_dir)

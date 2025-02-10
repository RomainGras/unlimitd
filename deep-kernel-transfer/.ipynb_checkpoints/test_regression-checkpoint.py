import torch
import torch.nn as nn
import torch.optim as optim
import configs

# from data.qmul_loader import get_batch, train_people, test_people
from data.regression_data_loader import data_provider

from io_utils import parse_args_regression, get_resume_file
from methods.maml import MAML
from methods.DKT_regression import DKT
from methods.feature_transfer_regression import FeatureTransfer
from methods.UnLiMiTDI_regression import UnLiMiTDI
from methods.UnLiMiTDR_regression import UnLiMiTDR
from methods.UnLiMiTDproj_regression import UnLiMiTDproj
from methods.UnLiMiTDIX_regression import UnLiMiTDIX
from methods.UnLiMiTDprojX_regression import UnLiMiTDprojX
import backbone
import numpy as np
import os

params = parse_args_regression('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)


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


provider = data_provider(params.dataset)

print(f"This is {params.method} testing, with kernel {configs.kernel_type}")

def create_random_projection_matrix(n, subspace_dimension):
    """
    Create a projection matrix from R^n to a subspace of dimension `subspace_dimension`.
    
    Args:
    n (int): Dimension of the original space.
    subspace_dimension (int): Dimension of the target subspace.

    Returns:
    torch.Tensor: A (n x subspace_dimension) projection matrix.
    """
    # Check if subspace_dimension is not greater than n
    if subspace_dimension > n:
        raise ValueError("subspace_dimension must be less than or equal to n")

    # Generate a random n x subspace_dimension matrix
    random_matrix = torch.randn(n, subspace_dimension)

    # Perform QR decomposition to orthonormalize the columns
    q, _ = torch.linalg.qr(random_matrix)

    # Return the first 'subspace_dimension' columns of Q, which form an orthonormal basis
    return q[:, :subspace_dimension].T


if params.method=='DKT':
    model = DKT(bb).cuda()
    optimizer = None
elif params.method=='transfer':
    model = FeatureTransfer(bb).cuda()
    optimizer = optim.Adam([{'params':model.parameters(),'lr':0.001}])

elif params.method=='MAML':
    model = MAML(bb, -1, problem="regression").cuda()   # n_support is -1 because it's directly implemented in loader
    params.checkpoint_dir += f"_{model.task_update_num}_inn_steps"
    
    model.task_update_num = params.task_update_num
    optimizer = None
elif params.method=='DKT_w_net':
    model = DKT(combined_network).cuda()
    optimizer = None
elif params.method=='DKT_w_net_multi':
    model = DKT(combined_network_multi).cuda()
    optimizer = None
elif params.method=='UnLiMiTDI':
    if params.conv_net_not_differentiated:
        model = UnLiMiTDIX(bb, backbone.simple_net(), has_scaling_params=False).cuda()
    else:
        model = UnLiMiTDIX(None, bb, has_scaling_params=False).cuda()    
    optimizer = None
elif params.method=='UnLiMiTDR':
    input_dimension = sum(p.numel() for p in simple_net.parameters())
    # Dummy projection matrix to initialize the model, replaced afterwards with model.load_checkpoint
    P = create_random_projection_matrix(input_dimension, params.subspacedim).cuda()
    model = UnLiMiTDproj(bb, simple_net, P).cuda()
    optimizer = None
elif params.method=='UnLiMiTDF':
    input_dimension = sum(p.numel() for p in simple_net.parameters())
    # Dummy projection matrix to initialize the model, replaced afterwards with model.load_checkpoint
    P = create_random_projection_matrix(input_dimension, params.subspacedim).cuda()
    model = UnLiMiTDproj(bb, simple_net, P).cuda()
    optimizer = None
elif params.method=='UnLiMiTDIX':
    if params.conv_net_not_differentiated:
        model = UnLiMiTDIX(bb, backbone.simple_net(), has_scaling_params=True).cuda()
    else:
        model = UnLiMiTDIX(None, bb, has_scaling_params=True).cuda()
    optimizer = None
    
elif params.method=='UnLiMiTDFX':
    print("Not implemented")
        
elif params.method=='UnLiMiTDFX_w_conv_differentiated':
    N = sum(p.numel() for p in combined_network.parameters())
    # Dummy projection matrix to initialize the model, replaced afterwards with model.load_checkpoint
    P = create_random_projection_matrix(N, params.subspacedim).cuda()
    model = UnLiMiTDprojX(None, combined_network, P).cuda()
    optimizer = None

else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)
    
mse_list = []
for epoch in range(params.n_test_epochs):
    if params.ft and 'UnLiMiTD' in params.method:
        mse = float(model.test_loop_ft(params.n_support, params.task_update_num, provider, optimizer, shrinking_factor=.01).cpu().detach().numpy())
        # In the case of OursI, shrinking_factor acts only as a learning rate
    else:
        mse = float(model.test_loop(params.n_support, provider, optimizer = optimizer).cpu().detach().numpy())
    mse_list.append(mse)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("Average RMSE: " + str(np.mean(np.sqrt(mse_list))) + " +- " + str(np.std(np.sqrt(mse_list))))
print("-------------------")

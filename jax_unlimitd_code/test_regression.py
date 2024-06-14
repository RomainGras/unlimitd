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
import backbone
import numpy as np

params = parse_args_regression('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
bb               = backbone.Conv3().cuda()
simple_net       = backbone.simple_net().cuda()
simple_net_multi = backbone.simple_net_multi_output().cuda()

combined_network       = backbone.CombinedNetwork(bb, simple_net).cuda()
combined_network_multi = backbone.CombinedNetwork(bb, simple_net_multi).cuda()

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

elif params.method=='DKT_w_net':
    model = DKT(combined_network).cuda()
    optimizer = None
elif params.method=='DKT_w_net_multi':
    model = DKT(combined_network_multi).cuda()
    optimizer = None
elif params.method=='UnLiMiTDI':
    model = UnLiMiTDI(bb, simple_net).cuda()
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
else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)

mse_list = []
for epoch in range(params.n_test_epochs):
    mse = float(model.test_loop(params.n_support, optimizer).cpu().detach().numpy())
    mse_list.append(mse)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")

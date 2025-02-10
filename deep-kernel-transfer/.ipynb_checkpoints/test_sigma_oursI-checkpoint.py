# Plots the graph of ours wrt sigma's hyper parameter

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
from methods.UnLiMiTDIX_regression import UnLiMiTDIX
from methods.UnLiMiTDprojX_regression import UnLiMiTDprojX
import backbone
import numpy as np
import matplotlib.pyplot as plt

params = parse_args_regression('test_sigma_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


bb               = backbone.Conv3().cuda()
simple_net       = backbone.simple_net().cuda()
simple_net_multi = backbone.simple_net_multi_output().cuda()

combined_network       = backbone.CombinedNetwork(bb, simple_net).cuda()
combined_network_multi = backbone.CombinedNetwork(bb, simple_net_multi).cuda()

print(f"This is sigma hyper parameter testing, of ours, with kernel {configs.kernel_type}")

model = UnLiMiTDI(None, combined_network).cuda()
optimizer = None
    
def test_for_sigma(model, sigma):
    mse_list = []
    for epoch in range(params.n_test_epochs):
        if params.ft:
            bb               = backbone.Conv3().cuda()
            simple_net       = backbone.simple_net().cuda()
            combined_network       = backbone.CombinedNetwork(bb, simple_net).cuda()
            mse = float(model.test_loop_ft(params.n_support, combined_network, optimizer).cpu().detach().numpy())
        else:
            mse = float(model.test_loop(params.n_support, optimizer).cpu().detach().numpy())
        mse_list.append(mse)

    print("-------------------")
    print(f"Average MSE for sigma = {sigma}: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")
    return np.mean(mse_list), np.std(mse_list)


sigmas = np.logspace(-4, 4, num=20)
mses_avg = []
mses_std = []

for sigma in sigmas:
    print("===================")
    print(f"Training for sigma = {sigma}")
    print("===================")
    model = UnLiMiTDI(None, combined_network, sigma=sigma).cuda()
    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001}])
    for epoch in range(params.stop_epoch):
        model.train_loop(epoch, optimizer)
    
    mean_mse, std_mse = test_for_sigma(model, sigma)
    mses_avg.append(mean_mse)
    mses_std.append(std_mse)
    

# Norm plot
plt.errorbar(sigmas, mses_avg, yerr=mses_std, fmt='o', capsize=5, capthick=2, ecolor='red', label='Average MSEs varying $$\sigma$$')

# Set x-axis to logarithmic scale
plt.xscale('log')
# plt.yscale('log')

# Add labels and title if needed
plt.xlabel(r'$\sigma$')
plt.ylabel('MSE')
plt.title(r'MSE vs $\sigma$ QMUL')

plt.savefig("influence_of_sigma_QMUL", dpi=300)
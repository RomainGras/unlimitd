## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, jvp, jacrev

## Our packages
import gpytorch
from time import gmtime, strftime
import random
from statistics import mean
from data.qmul_loader import get_batch, train_people, test_people
from configs import kernel_type

import copy
import torch.optim as optim

class UnLiMiTDIX(nn.Module):
    def __init__(self, conv_net, diff_net, has_scaling_params=True, sigma=1):
        super(UnLiMiTDIX, self).__init__()
        if conv_net is None:
            # All network is differentiated, convolution layers included
            self.feature_extractor = nn.Identity()
            self.diff_net = diff_net
        else:
            self.feature_extractor = conv_net
            self.diff_net = diff_net  #Diff
        
        self.sigma = sigma
        self.has_scaling_params = has_scaling_params
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(19, 30000).cuda()  # QMUL (19, 30000), QMUL with conv net not differentiated (19, 2916), berkeley (30, 11), argus (100, 3)
        if(train_y is None): train_y=torch.ones(19).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, diff_net = self.diff_net, kernel=kernel_type, has_scaling_params=self.has_scaling_params, sigma=self.sigma)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def train_loop(self, epoch, provider, optimizer):
        batch, batch_labels = provider.get_train_batch()
        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()

            inputs_conv = self.feature_extractor(inputs)  #If convolution is not differentiated, else, it's just identity
            
            inputs_conv_flat = inputs_conv.view(inputs_conv.size(0), -1)
            self.model.set_train_data(inputs=inputs_conv_flat, targets=labels - self.diff_net(inputs_conv).reshape(-1)) 
            
            predictions = self.model(inputs_conv_flat)
            
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

            if (epoch%5==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
                # print(self.model.covar_module.scaling_params.values())
                # print(all([torch.equal(param_covar_module, param_net) for (param_covar_module, param_net) in zip(self.model.covar_module.params.values(), dict(self.diff_net.named_parameters()).values())]))  # Shows that params in covar module are the same that params in diff_net

    def test_loop(self, n_support, provider, optimizer=None): # no optimizer needed for GP
        (x_support, y_support), (x_query, y_query) = provider.get_test_batch()
        
        # choose a random test person
        n = np.random.randint(0, x_support.size(0)-1)
    
        x_conv_support = self.feature_extractor(x_support[n]).detach()
        x_conv_support_flat = x_conv_support.view(x_conv_support.size(0), -1)
        self.model.set_train_data(inputs=x_conv_support_flat, targets=y_support[n] - self.diff_net(x_conv_support).reshape(-1), strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            x_conv_query = self.feature_extractor(x_query[n]).detach()
            x_conv_query_flat = x_conv_query.view(x_conv_query.size(0), -1)
            pred    = self.likelihood(self.model(x_conv_query_flat))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean
            lower += self.diff_net(x_conv_query).reshape(-1)
            upper += self.diff_net(x_conv_query).reshape(-1)
        mse = self.mse(pred.mean + self.diff_net(self.feature_extractor(x_query[n])).reshape(-1), y_query[n])

        return mse
    
    
    def test_loop_ft(self, n_support, task_update_num, provider, optimizer=None, shrinking_factor=1, print_every=None): # no optimizer needed for GP
        n_ft = task_update_num
        mse_per_step = []
        (x_support, y_support), (x_query, y_query) = provider.get_test_batch()

        # choose a random test person
        n = np.random.randint(0, x_support.size(0)-1)
    
        x_conv_support = self.feature_extractor(x_support[n]).detach()

        # Create a new model instance and load the original model's state
        ft_net = copy.deepcopy(self.diff_net)  # Deep copy the original model's weights        
        # Set up an optimizer for fine-tuning
        optimizer = optim.Adam(ft_net.parameters(), lr=0)  # lr=0 because we manually apply updates

        # Fine-tuning loop
        print(f"Beggining adaptation with n_support {x_support.size(0)}")
        for i_ft in range(n_ft):
            # Forward pass
            train_logit = ft_net(x_conv_support).reshape(-1)
            inner_loss = F.mse_loss(train_logit, y_support[n])

            # Backward pass
            inner_loss.backward()

            # Manually update each parameter using the custom learning rates
            with torch.no_grad():
                for name, param in ft_net.named_parameters():
                    param_update = shrinking_factor * self.model.covar_module.scaling_params[name] * param.grad
                    param -= param_update

            # Clear the gradients after the update
            optimizer.zero_grad()

            if print_every is not None and i_ft%print_every==0:
                with torch.no_grad():
                    self.model.eval()
                    self.feature_extractor.eval()
                    self.likelihood.eval()
                    pred = ft_net(x_query[n]).reshape(-1)
                    mse = self.mse(pred, y_query[n])
                    mse_per_step.append(mse)
                    self.model.train()
                    self.feature_extractor.train()
                    self.likelihood.train()

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            x_conv_query = self.feature_extractor(x_query[n]).detach()
            pred = ft_net(x_conv_query).reshape(-1)
            mse = self.mse(pred, y_query[n])
            print(f"Final MSE : {mse.item()}")
            # mse_list.append(mse.item())
            # mse = self.mse(mse.item(), y_query[n])
        
        mse_per_step.append(mse)
        if print_every is None:
            return mse
        else:
            return mse_per_step

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        conv_net_state_dict = self.feature_extractor.state_dict()
        diff_net_state_dict   = self.diff_net.state_dict()
        torch.save({'gp': gp_state_dict, 'sp': self.model.covar_module.scaling_params, 'likelihood': likelihood_state_dict, 'conv_net':conv_net_state_dict, 'diff_net':diff_net_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.model.covar_module.scaling_params = ckpt['sp']
        self.feature_extractor.load_state_dict(ckpt['conv_net'])
        self.diff_net.load_state_dict(ckpt['diff_net'])


# ##################
# NTKernel
# ##################

class NTKernel(gpytorch.kernels.Kernel):
    # Also try autodiff by using output as loss function (out_dim = 1)
    def __init__(self, net, has_scaling_params, sigma=1, **kwargs):
        super(NTKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.sigma = sigma
        self.jac_func = create_jac_func(net)
        self.params = dict(net.named_parameters())
        if has_scaling_params:
            self.scaling_params = {k: torch.ones_like(v, device='cuda:0', requires_grad=True) for k, v in self.params.items()}
        else:
            self.scaling_params = {k: torch.ones_like(v, device='cuda:0', requires_grad=False) for k, v in self.params.items()}
        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1 = x1.reshape(x1.size(0), 3, 100, 100)
        x2 = x2.reshape(x2.size(0), 3, 100, 100)
        
        jac1 = self.jac_func(self.params, x1)
        sp_jac1 = [(self.scaling_params[k]*j).flatten(2) for (k, j) in jac1.items()]  
        if torch.equal(x1, x2):
            jac2=jac1
            sp_jac2=sp_jac1
        else:
            jac2 = self.jac_func(self.params, x2)
            sp_jac2 = [(self.scaling_params[k]*j).flatten(2) for (k, j) in jac2.items()]  
            
        ntk_list = []
        for j1, j2 in zip(sp_jac1, sp_jac2):
            ntk_list.append(torch.einsum('Naf,Maf->aNM', j1, j2))
        
        ntk = self.sigma * torch.sum(torch.stack(ntk_list), dim=0).squeeze(0)
        
        if diag:
            return ntk.diag()
        return ntk

    

###################
#NTKernel CosSim (TODO)
###################
class CosSimNTKernel(gpytorch.kernels.Kernel):
    def __init__(self, net, **kwargs):
        super(CosSimNTKernel, self).__init__(**kwargs)
        self.net = net
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        N = sum(p.numel() for p in net.parameters()) # Number of params in the network
        self.N = N
        
        # Add N scaling parameters, initializing them as one
        self.scaling_params = nn.Parameter(torch.ones(N)).cuda()
        self.sp = self.scaling_params.view(1, self.N).cuda()

    def forward(self, x1, x2, diag=False, **params):
        r1T = (self.compute_jacobian(x1) * self.sp).T
        r1T_norm = r1T.norm(dim=0, keepdim=True)
        r1T_normalized = r1T/r1T_norm
        
        r2T = (self.compute_jacobian(x2) * self.sp).T if x1 is not x2 else jac1T
        r2T_norm = r2T.norm(dim=0, keepdim=True)
        r2T_normalized = r2T/r2T_norm
        
        result = self.alpha * r1T_normalized.T@r2T_normalized
        
        if diag:
            return result.diag()
        return result
    
    def compute_jacobian(self, inputs):
        """
        Return the jacobian of a batch of inputs, thanks to the vmap functionality
        """
        self.zero_grad()
        params = {k: v for k, v in self.net.named_parameters()}
        def fnet_single(params, x):
            return functional_call(self.net, params, (x.unsqueeze(0),)).squeeze(0)
        
        jac = vmap(jacrev(fnet_single), (None, 0))(params, inputs)
        jac = jac.values()
        # jac1 of dimensions [Nb Layers, Nb input / Batch, dim(y), Nb param/layer left, Nb param/layer right]
        reshaped_tensors = [
            j.flatten(2)                # Flatten starting from the 3rd dimension to acount for weights and biases layers
                .permute(2, 0, 1)         # Permute to align dimensions correctly for reshaping
                .reshape(-1, j.shape[0] * j.shape[1])  # Reshape to (c, a*b) using dynamic sizing
            for j in jac
        ]
        return torch.cat(reshaped_tensors, dim=0).T

###################
#GP
###################    
class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, diff_net, kernel='NTK', sigma=1, has_scaling_params=True):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## NTKernel
        if(kernel=='NTK'):
            self.covar_module = NTKernel(diff_net, has_scaling_params, sigma)
        elif(kernel=='NTKcossim'):
            self.covar_module = CosSimNTKernel(diff_net)        
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def create_jac_func(net):
    """
    Computes the functional call of a single input, to be differentiated in parallel using vmap
    """
    def fnet_single(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

    jac_func = vmap(jacrev(fnet_single), (None, 0))
    return jac_func

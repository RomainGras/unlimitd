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

class UnLiMiTDIX(nn.Module):
    def __init__(self, conv_net, diff_net):
        super(UnLiMiTDIX, self).__init__()
        if conv_net is None:
            # All network is differentiated, convolution layers included
            self.feature_extractor = lambda x : x #identity
            self.diff_net = diff_net
            self.is_conv_diff = False
        else:
            self.feature_extractor = conv_net
            self.diff_net = diff_net  #Differentiable network
            self.is_conv_diff = True
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(19, 30000).cuda()
        if(train_y is None): train_y=torch.ones(19).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, diff_net = self.diff_net, kernel=kernel_type)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def train_loop(self, epoch, optimizer):
        batch, batch_labels = get_batch(train_people)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
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

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
                print(self.model.sigma)

    def test_loop(self, n_support, optimizer=None): # no optimizer needed for GP
        inputs, targets = get_batch(test_people)

        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind   = [i for i in range(19) if i not in support_ind]

        x_all = inputs.cuda()
        y_all = targets.cuda()

        x_support = inputs[:,support_ind,:,:,:].cuda()
        y_support = targets[:,support_ind].cuda()

        # choose a random test person
        n = np.random.randint(0, len(test_people)-1)
    
        x_conv_support = self.feature_extractor(x_support[n]).detach()
        x_conv_support_flat = x_conv_support.view(x_conv_support.size(0), -1)
        self.model.set_train_data(inputs=x_conv_support_flat, targets=y_support[n] - self.diff_net(x_conv_support).reshape(-1), strict=False)

        self.model.eval()
        if self.is_conv_diff:
            self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            x_conv_query = self.feature_extractor(x_all[n]).detach()
            x_conv_query_flat = x_conv_query.view(x_conv_query.size(0), -1)
            pred    = self.likelihood(self.model(x_conv_query_flat))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean
            lower += self.diff_net(x_conv_query).reshape(-1)
            upper += self.diff_net(x_conv_query).reshape(-1)
        mse = self.mse(pred.mean + self.diff_net(self.feature_extractor(x_all[n])).reshape(-1), y_all[n])

        return mse

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        if self.is_conv_diff:
            conv_net_state_dict = self.feature_extractor.state_dict()
        else:
            conv_net_state_dict = None
        diff_net_state_dict   = self.diff_net.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'conv_net':conv_net_state_dict, 'diff_net':diff_net_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        if self.is_conv_diff:
            self.feature_extractor.load_state_dict(ckpt['conv_net'])
        self.diff_net.load_state_dict(ckpt['diff_net'])


# ##################
# NTKernel
# ##################

class NTKernel(gpytorch.kernels.Kernel):
    def __init__(self, net, **kwargs):
        super(NTKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.net = net
        
        N = sum(p.numel() for p in net.parameters()) # Number of params in the network
        self.N = N
        
        # Add N scaling parameters, initializing them as one
        self.scaling_param = nn.Parameter(torch.ones(N)).cuda()
        self.sp = self.scaling_param.view(1, self.N).cuda()
        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1 = x1.reshape(x1.size(0), 3, 100, 100)
        x2 = x2.reshape(x2.size(0), 3, 100, 100)
        jac1 = self.compute_jacobian(x1)
        jac2 = self.compute_jacobian(x2) if x1 is not x2 else jac1
        
        r1 = jac1*self.sp
        r2 = jac2*self.sp
        
        result = r1 @ r2.T
        
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
        self.scaling_param = nn.Parameter(torch.ones(N)).cuda()
        self.sp = self.scaling_param.view(1, self.N).cuda()

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
    def __init__(self, train_x, train_y, likelihood, diff_net, kernel='NTK'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## NTKernel
        if(kernel=='NTK'):
            self.covar_module = NTKernel(diff_net)
        elif(kernel=='NTKcossim'):
            self.covar_module = CosSimNTKernel(diff_net)        
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

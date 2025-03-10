## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F

## Our packages
import gpytorch
from time import gmtime, strftime
import random
from statistics import mean
# from data.data_loader import get_batch, train_people, test_people
from configs import kernel_type

# QMUL (19, 2916), berkeley (30, 32), argus (100, 32), QMUL with net (19, 40)
batch_dim, feature_dim = 30, 15

class DKT(nn.Module):
    def __init__(self, backbone):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(batch_dim, feature_dim).cuda() 
        if(train_y is None): train_y=torch.ones(batch_dim).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel_type)
        # train_x, train_y = tasks.sample_task().sample_data(n_shot_train, noise=.1)

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
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        for X, Y in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(X)
            
            self.model.set_train_data(inputs=z, targets=Y)
            predictions = self.model(z)
            
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, Y)

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    def test_loop(self, n_support, provider, optimizer=None): # no optimizer needed for GP
        (x_support, y_support), (x_query, y_query) = provider.get_test_batch()
        
        # choose a random test person
        n = np.random.randint(0, x_support.size(0)-1)
        
        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)
            
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query[n]).detach()
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query[n])

        return mse

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        if kernel_type != "linear":
            self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        #ckpt_net = dict()
        #for name, weight in ckpt['net'].items():
        #    new_key = name.replace("0.layer", "0.conv")
        #    ckpt_net[new_key] = weight
        self.feature_extractor.load_state_dict(ckpt['net'])

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=feature_dim)
        elif(kernel=='linear'):
            self.covar_module = linear_kernel()
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        

class linear_kernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super(linear_kernel, self).__init__(has_lengthscale=False, **kwargs)
        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if diag:
            return (x1 * x2).sum(dim=-1).view(-1)
        else:
            return (x1 @ x2.transpose(-2, -1))

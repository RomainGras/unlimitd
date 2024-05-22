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

class UnLiMiTDR(nn.Module):
    def __init__(self, conv_net, diff_net, subspace_dimension):
        super(UnLiMiTDR, self).__init__()
        ## GP parameters
        self.feature_extractor = conv_net
        self.diff_net = diff_net  #Differentiable network
        
        input_dimension = sum(p.numel() for p in diff_net.parameters())
        self.subspace_dimension = subspace_dimension
        self.P = create_projection_matrix(input_dimension, subspace_dimension).cuda()
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(19, 2916).cuda()
        if(train_y is None): train_y=torch.ones(19).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, diff_net = self.diff_net, kernel=kernel_type, subspace_dimension = self.subspace_dimension, P = self.P)

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

            inputs_conv = self.feature_extractor(inputs)
            self.model.set_train_data(inputs=inputs_conv, targets=labels - self.diff_net(inputs_conv).reshape(-1))  
            predictions = self.model(inputs_conv)
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

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
        self.model.set_train_data(inputs=x_conv_support, targets=y_support[n] - self.diff_net(x_conv_support).reshape(-1), strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            x_conv_query = self.feature_extractor(x_all[n]).detach()
            pred    = self.likelihood(self.model(x_conv_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean
            lower += self.diff_net(x_conv_query).reshape(-1)
            upper += self.diff_net(x_conv_query).reshape(-1)
        mse = self.mse(pred.mean + self.diff_net(self.feature_extractor(x_all[n])).reshape(-1), y_all[n])

        return mse

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        conv_net_state_dict   = self.feature_extractor.state_dict()
        diff_net_state_dict   = self.diff_net.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'conv_net':conv_net_state_dict, 'diff_net':diff_net_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['conv_net'])
        self.diff_net.load_state_dict(ckpt['diff_net'])


# ##################
# NTKernel
# ##################

class NTKernel_proj(gpytorch.kernels.Kernel):
    def __init__(self, net, P, subspace_dimension, **kwargs):
        super(NTKernel_proj, self).__init__(**kwargs)
        self.net = net
        
        self.sub_dim = subspace_dimension
        self.P = P # Projection matrix
        
        # Add 10 scaling parameters, initializing them as one
        self.scaling_param = nn.Parameter(torch.ones(subspace_dimension))
        
    def forward(self, x1, x2, diag=False, **params):
        jac1 = self.compute_jacobian(x1)
        jac2 = self.compute_jacobian(x2) if x1 is not x2 else jac1
        D = torch.diag(torch.pow(self.scaling_param, 2))
        
        result = torch.chain_matmul(jac1.T, self.P.T, D, self.P, jac2)
        
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
        return torch.cat(reshaped_tensors, dim=0)

    


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, diff_net, subspace_dimension, P, kernel='NTK'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## NTKernel
        if(kernel=='NTK'):
            self.covar_module = NTKernel_proj(diff_net, P, subspace_dimension)
        elif(kernel=='cossim' or kernel=='bncossim'):
            # NOT IMPLEMENTED
            raise ValueError("NOT IMPLEMENTED")            
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def create_projection_matrix(n, subspace_dimension):
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
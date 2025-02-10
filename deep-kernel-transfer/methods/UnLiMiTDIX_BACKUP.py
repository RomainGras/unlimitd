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
    def __init__(self, conv_net, diff_net):
        super(UnLiMiTDIX, self).__init__()
        if conv_net is None:
            # All network is differentiated, convolution layers included
            self.feature_extractor = lambda x : x #identity
            self.diff_net = diff_net
            self.is_conv_diff = False
        else:
            self.feature_extractor = conv_net
            self.diff_net = diff_net  #Diff
                
        N = sum(p.numel() for p in diff_net.parameters()) # Number of params in the network
        self.scaling_param = nn.Parameter(torch.ones(N, device="cuda"))
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(100, 3).cuda()  # QMUL (19, 30000), berkeley (30, 11), argus (100, 3)
        if(train_y is None): train_y=torch.ones(100).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, diff_net = self.diff_net, scaling_param=self.scaling_param, kernel=kernel_type)

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

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    def test_loop(self, n_support, provider, optimizer=None): # no optimizer needed for GP
        (x_support, y_support), (x_query, y_query) = provider.get_test_batch()
        
        # choose a random test person
        n = np.random.randint(0, x_support.size(0)-1)
    
        x_conv_support = self.feature_extractor(x_support[n]).detach()
        x_conv_support_flat = x_conv_support.view(x_conv_support.size(0), -1)
        self.model.set_train_data(inputs=x_conv_support_flat, targets=y_support[n] - self.diff_net(x_conv_support).reshape(-1), strict=False)

        self.model.eval()
        if self.is_conv_diff:
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
    
    def generate_and_print_sigma(self, shrinking_factor):
        
        def inverse_transformation(transformed_tensor, original_shapes, shrinking_factor):
            # Step 1: Transpose the tensor back
            transposed_tensor = transformed_tensor.T

            # Convert to PyTorch tensor if it's a NumPy array
            if isinstance(transposed_tensor, np.ndarray):
                transposed_tensor = torch.from_numpy(transposed_tensor)

            # Step 2: Calculate the correct split sizes
            split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in original_shapes]

            # Determine the dimension to split along
            split_dim = 1 if transposed_tensor.dim() > 1 else 0

            # Step 3: Split the tensor back into the original number of reshaped tensors
            split_tensors = torch.split(transposed_tensor, split_sizes, dim=split_dim)

            original_tensors = []

            for split_tensor, shape in zip(split_tensors, original_shapes):
                # Step 4: Reshape back to the original flattened shape
                reshaped_tensor = split_tensor.reshape(-1, *shape[:2]) if len(shape) > 1 else split_tensor.reshape(shape[0], -1)

                # Step 5: Permute dimensions back
                if len(shape) > 2:  # Only permute if the original tensor had more than 2 dimensions
                    permuted_tensor = reshaped_tensor.permute(1, 2, 0)
                else:
                    permuted_tensor = reshaped_tensor

                # Step 6: Reshape back to the original shape
                original_tensor = shrinking_factor * permuted_tensor.reshape(shape)

                original_tensors.append(original_tensor)


            return original_tensors

        def name_tensor(named_params, inverse_transformed):
            named_tensor = {name : sigma for name,sigma in zip(named_params.keys(), inverse_transformed)}
            return named_tensor

        named_params = dict(self.model.covar_module.net.named_parameters())
        shape_list = list([param.shape for param in named_params.values()])

        # Adapts the form of Sigma such that it is suitable to be used as a learning rate
        Sigma = self.model.covar_module.sp[0,:]
        Sigma_adapted_lr = inverse_transformation(Sigma.T, shape_list, shrinking_factor=shrinking_factor)
        Sigma_adapted_lr = name_tensor(named_params, Sigma_adapted_lr)
        
        self.Sigma_adapted_lr = Sigma_adapted_lr
        
        sigma_np = Sigma.detach().cpu().numpy()
        print(f"Ten first elements of Sigma : {sigma_np[:10]}")
        print(f"Ten last elements of Sigma : {sigma_np[-10:]}")
        print(f"Max and Min of Sigma values : {max(sigma_np)} ; {min(sigma_np)}")
        
        
    def test_loop_ft(self, n_support, task_update_num, ft_net, provider, optimizer=None, print_every=None): # no optimizer needed for GP
        n_ft = task_update_num
        mse_per_step=[]
        (x_support, y_support), (x_query, y_query) = provider.get_test_batch()

        # choose a random test person
        n = np.random.randint(0, x_support.size(0)-1)
    
        x_conv_support = self.feature_extractor(x_support[n]).detach()

        # Create a new model instance and load the original model's state
        ft_net.load_state_dict(copy.deepcopy(self.diff_net.state_dict()))  # Deep copy the original model's weights

        # Set up an optimizer for fine-tuning
        optimizer = optim.Adam(ft_net.parameters(), lr=0)  # lr=0 because we manually apply updates

        # Fine-tuning loop
        print(f"Beggining adaptation with n_support {x_support.size(0)}")
        for i_ft in range(n_ft):
            # Forward pass
            train_logit = ft_net(x_conv_support).reshape(-1)
            inner_loss = F.mse_loss(train_logit, y_support)

            # Backward pass
            inner_loss.backward()

            # Manually update each parameter using the custom learning rates
            with torch.no_grad():
                for name, param in ft_net.named_parameters():
                    if name in self.Sigma_adapted_lr:
                        print(self.Sigma_adapted_lr[name])
                        param_update = self.Sigma_adapted_lr[name] * param.grad
                        param -= param_update

            # Clear the gradients after the update
            optimizer.zero_grad()

            if print_every is not None and i_ft%print_every==0:
                with torch.no_grad():
                    self.model.eval()
                    if self.is_conv_diff:
                        self.feature_extractor.eval()
                    self.likelihood.eval()
                    pred = ft_net(x_query[n]).reshape(-1)
                    mse = self.mse(pred, y_query[n])
                    mse_per_step.append(mse)
                    self.model.train()
                    if self.is_conv_diff:
                        self.feature_extractor.train()
                    self.likelihood.train()

        self.model.eval()
        if self.is_conv_diff:
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
        if self.is_conv_diff:
            conv_net_state_dict = self.feature_extractor.state_dict()
        else:
            conv_net_state_dict = None
        diff_net_state_dict   = self.diff_net.state_dict()
        torch.save({'gp': gp_state_dict, 'sp': self.scaling_param, 'likelihood': likelihood_state_dict, 'conv_net':conv_net_state_dict, 'diff_net':diff_net_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        print(ckpt['sp'])
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.scaling_param = ckpt['sp']
        if self.is_conv_diff:
            self.feature_extractor.load_state_dict(ckpt['conv_net'])
        self.diff_net.load_state_dict(ckpt['diff_net'])


# ##################
# NTKernel
# ##################

class NTKernel(gpytorch.kernels.Kernel):
    def __init__(self, net, scaling_param, **kwargs):
        super(NTKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.net = net
        
        # Add N scaling parameters, initializing them as one
        self.sp = scaling_param.reshape(1, -1)
        
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
    def __init__(self, train_x, train_y, likelihood, diff_net, scaling_param, kernel='NTK'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## NTKernel
        if(kernel=='NTK'):
            self.covar_module = NTKernel(diff_net, scaling_param)
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
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

class multistep_unlimited(nn.Module):
    def __init__(self, conv_net, diff_net, has_scaling_params=True, sigma=1, projection_matrix=None, method="UnLiMiTDI++"):
        super(unlimited_plus, self).__init__()
        if conv_net is None:
            # All network is differentiated, convolution layers included
            self.feature_extractor = nn.Identity()
            self.diff_net = diff_net
        else:
            self.feature_extractor = conv_net
            self.diff_net = diff_net  #Diff
        
        self.method = method
        self.sigma = sigma
        self.loss_fn = nn.MSELoss()
        self.scaling_params = {k: 1*torch.ones_like(v, device="cpu") for (k, v) in params.items()}        

    def inner_ponderated_L2_regularization(self, params, meta_params):
        regularization = 0
        for name, sp in self.scaling_params.items():
            regularization += (((sp + ellipse_jitter).pow(-2)) * (params[name] - meta_params[name]).pow(2)).sum()
        return regularization

    def inner_minimizer(self, X, Y):
        meta_params = {k: v for k, v in net.named_parameters()}

        inner_net = copy.deepcopy(self.diff_net)

        in_optimizer = optim.Adam([{'params':inner_net.parameters(), 'lr':0}])
        in_optimizer.zero_grad()
        avg_loss=0.0

        for i_ft in range(n_inner_steps):
            # Forward pass
            train_logit = inner_net(X)
            params = {k: v for k, v in inner_net.named_parameters()}
            inner_loss = self.loss_fn(train_logit.squeeze(1), Y) + lamb / 2 * self.inner_ponderated_L2_regularization(params, meta_params)

            # Backward pass
            inner_loss.backward()

            # Manually update each parameter using the custom learning rates
            with torch.no_grad():
                for name, param in inner_net.named_parameters():
                    param_update = inner_lr * param.grad
                    param -= param_update

            # Clear the gradients after the update
            in_optimizer.zero_grad()

        params = {k: v.detach() for k, v in inner_net.named_parameters()}
        return params

    def fnet_single(self, params, x):
        return functional_call(self.diff_net, params, (x.unsqueeze(0),)).squeeze(0)

    def task_param_upt(self, x_query, y_query, params):
        # Forward pass
        self.diff_net.train()
        phi = functional_call(self.diff_net, params, (x_query,))
        self.diff_net.eval()

        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(params, x_query)
        # TODO add ellipse jitter to scaling param before this step
        s_jac = {k : self.scaling_params[k]*j for (k, j) in jac1.items()}

        s_jac_flat = [s_j.flatten(2) for s_j in s_jac.values()]

        # Compute J(x1) @ J(x2).T
        ntk = torch.stack([torch.einsum('Naf,Maf->aNM', j1, j2) for j1, j2 in zip(s_jac_flat, s_jac_flat)])
        ntk = ntk.sum(0).squeeze(0)

        # Compute solutions to (NTK_c + lambda I)^-1 (Y_c - phi_c)
        inverse_term = ntk + Lamb
        residual = y_query - phi.squeeze(1)  # phi is of shape [n_way*n_support, n_way]

        # Solve the system (NTK_c + epsilon I_k) * result = residual
        # sols.append(torch.linalg.solve(inverse_term, residual))  # IF ONE WANTS TO USE FIXED JITTER
        sols = safe_cholesky_solve(inverse_term, residual)

        # Update parameters 
        tensor_update = {k : self.scaling_params[k] * torch.tensordot(s_jac[k][:,0], sols, dims=([0], [0])) for k in params.keys()}
        # tensor_update = {k : scaling_params[k] * torch.sum(s_jac[k][:,0], dim=0) for k in params.keys()}

        return tensor_update
    
    def train_loop(self, epoch, provider, optimizer):
        
        n_support = 9
        n_task = 8
        batch, batch_labels = provider.get_train_batch()
        tasks_updates = []
        for task_idx, (inputs, labels) in enumerate(zip(batch, batch_labels)):
            optimizer.zero_grad()
            
            x_support, x_query = inputs[:n_support], inputs[n_support:]
            y_support, y_query = labels[:n_support], labels[n_support:]
            meta_params = dict(self.diff_net.named_parameters())

            params = self.inner_minimizer(x_support, y_support)
            task_upt = self.task_param_upt(x_query, y_query, params)
            tasks_updates.append(task_upt)
            
            if task_idx%n_task==0:
                with torch.no_grad():
                    for name, w in meta_params.items():
                        w += outer_lr / n_tasks * sum([task_upt[name] for task_upt in tasks_updates])
                task_updates = []

            if i%200==0:
                print(f"Epoch {i}")

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass
    
    def test_loop(self, n_support, provider, optimizer=None): # no optimizer needed for GP
        (x_support, y_support), (x_query, y_query) = provider.get_test_batch()
        
        # choose a random test person
        n = np.random.randint(0, x_support.size(0)-1)
    
        x_conv_support = self.feature_extractor(x_support[n]).detach()
        x_conv_support_flat = x_conv_support.view(x_conv_support.size(0), -1)
        
        x_conv_query = self.feature_extractor(x_query[n]).detach()
        x_conv_query_flat = x_conv_query.view(x_conv_query.size(0), -1)
        
        if self.has_mean:
            jac = compute_jacobian(self.diff_net, x_conv_support)
            support_mean = jac @ self.mean
            jac = compute_jacobian(self.diff_net, x_conv_query)
            query_mean = jac @ self.mean
        else:
            support_mean = self.diff_net(x_conv_support).reshape(-1)
            query_mean = self.diff_net(x_conv_query).reshape(-1)
            
        self.model.set_train_data(inputs=x_conv_support_flat, targets=y_support[n] - support_mean, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            pred    = self.likelihood(self.model(x_conv_query_flat))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean
            lower += query_mean
            upper += query_mean
        mse = self.mse(pred.mean + query_mean, y_query[n])

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
        torch.save({
            'gp': gp_state_dict, 
            'sp': self.model.covar_module.scaling_params, 
            'mean': self.mean,
            'projection_matrix': self.projection_matrix,
            'likelihood': likelihood_state_dict, 
            'conv_net':conv_net_state_dict, 
            'diff_net':diff_net_state_dict
        }, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['conv_net'])
        self.diff_net.load_state_dict(ckpt['diff_net'])
        self.mean = ckpt['mean']
        if '++' not in self.method and 'projection_m)atrix' in ckpt.keys() and ckpt['projection_matrix'] is not None:
            self.projection_matrix = ckpt['projection_matrix']
            self.model.covar_module.scaling_params = ckpt['sp']
        if '++' in self.method:
            self.model.covar_module.scaling_params = ckpt['sp']

            
            
    
    
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        if A.dim() == 2:
            L = torch.linalg.cholesky(A, upper=upper, out=out)
            return L
        else:
            L_list = []
            for idx in range(A.shape[0]):
                L = torch.linalg.cholesky(A[idx], upper=upper, out=out)
                L_list.append(L)
            return torch.stack(L_list, dim=0)
    except:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(8):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                if Aprime.dim() == 2:
                    L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new} to the diagonal",
                        RuntimeWarning,
                    )
                    return L
                else:
                    L_list = []
                    for idx in range(Aprime.shape[0]):
                        L = torch.linalg.cholesky(Aprime[idx], upper=upper, out=out)
                        L_list.append(L)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new} to the diagonal",
                        RuntimeWarning,
                    )
                    return torch.stack(L_list, dim=0)
            except:
                continue

                
                
def safe_cholesky_solve(A, X):
    """
    Solve the inverse problem A^{-1} X, using psd_safe_cholesky to have good values for jitter
    """
    # Step 1: Perform Cholesky decomposition to get L
    L = psd_safe_cholesky(A)
    X = X.unsqueeze(1)
    
    # Step 2: Solve L * Z = X for Z using forward substitution
    Z = torch.linalg.solve_triangular(L, X, upper=False)  # 'upper=False' indicates L is lower triangular
    
    # Step 3: Solve L^T * Y = Z for Y using backward substitution
    Y = torch.linalg.solve_triangular(L.T, Z, upper=True)  # 'upper=True' indicates L^T is upper triangular

    return Y.squeeze()
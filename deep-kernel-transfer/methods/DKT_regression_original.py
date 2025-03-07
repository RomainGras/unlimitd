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
from data.qmul_loader import get_batch, train_people, test_people
from configs import kernel_type

class DKT_original(nn.Module):
    def __init__(self, backbone):
        super(DKT_original, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(19, 2916).cuda()
        if(train_y is None): train_y=torch.ones(19).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel_type)

        self.model      = model.cuda()
        self.batch_norm = nn.BatchNorm1d(train_x.shape[1]).cuda()
        
        if kernel_type=='spectral':
            inps, labs = [], []
            for _ in range(5):
                batch, batch_labels = get_batch(train_people)
                batch, batch_labels = batch.cuda(), batch_labels.cuda()
                inputs, labels = batch.view(-1, 3, 100, 100), batch_labels.view(-1)
                inps.append(inputs)
                labs.append(labels)
            inputs = torch.cat(inps, dim=0)
            labels = torch.cat(labs, dim=0)
            
            z = self.batch_norm(self.feature_extractor(inputs))
            self.model.covar_module.initialize_from_data(z, labels)
            
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
            z = self.batch_norm(self.feature_extractor(inputs))

            # self.model.set_train_data(inputs=z, targets=labels)
            # predictions = self.model(z)
            # loss = -self.mll(predictions, self.model.train_targets)
            K = self.model.covar_module(z).evaluate()
            # print(K)
            loss = NMLL(torch.zeros_like(labels), K, labels, noise=self.model.likelihood.noise.item())

            loss.backward()
            optimizer.step()
            # mse = self.mse(predictions.mean, labels)

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  noise: %.3f' % (
                    epoch, loss.item(),
                    self.model.likelihood.noise.item()
                ))

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
def NMLL(phi, K, y_support, noise=.1):
    
    X = y_support - phi
    
    # Perform Cholesky decomposition to get L
    # Noise
    n = K.size(0)
    device = f'cuda:{K.get_device()}'
    I = torch.eye(n, device=device)
    L = psd_safe_cholesky(K + noise*I)
    X = X.unsqueeze(1)
    
    # Solve L * Z = X for Z using forward substitution
    Z = torch.linalg.solve_triangular(L, X, upper=False)
    
    # Computes Z^T * Z = (Y-phi)^T K^-1 (Y-phi)
    sol1 = Z.T @ Z
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    
    return sol1 + logdet
    
# To handle Cholesky decompositions

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

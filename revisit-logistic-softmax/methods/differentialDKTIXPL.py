# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import time
from gp_utils import square_ntk, NMLL, support_query_ntk, out_distr
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.func import functional_call, vmap, vjp, jvp, jacrev

import copy
import torch.optim as optim
device = "cuda:0"

class differentialDKTIXPL(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False):
        super(differentialDKTIXPL, self).__init__( model_func,  n_way, n_support, change_way = False)
        
        self.diff_net = self.feature
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        # self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        # self.classifier.bias.data.fill_(0)
        
        self.n_task     = 6
        self.approx = approx #first order approx.        
        self.n_way = 5

        self.task_update_num = 1  # Number of inner updates

        self.noise = .1   # Usual value for jitter in gpytorch for float numbers
        self.Noise = self.noise * torch.eye(self.n_way * self.n_support, device=device)  # If one wants to use solve with fixed jitter
        
        self.temp = 1 # Temperature for cross entropy outer loss
        
        self.train_lr = 1e-2  # Inner learning rate
        print(f"Inner LR : {self.train_lr}")
        
        # Labels "Y" will always be the same for each task, we directly construct the labels:
        def contruct_target_list(N, n_way):  # N = n_support or n_query
            target_list = list()
            samples_per_model = int(N)
            for c in range(n_way):
                target = torch.ones(N * n_way, dtype=torch.float32) * - 1.0
                start_index = c * samples_per_model
                stop_index = start_index+samples_per_model
                target[start_index:stop_index] = 1.0
                target_list.append(target.cuda())
            return target_list
        
        self.target_list_support = contruct_target_list(self.n_support, self.n_way)
        
        self.n_query = 17 - self.n_support
        self.target_list_query = contruct_target_list(self.n_query, self.n_way)
        
        # Construct diagonal elements of Sigma as a dict of same the shapes as parameters for easier computations
        params = dict(self.diff_net.named_parameters())
        self.scaling_params = {k: 1*torch.ones_like(v, device=device) for (k, v) in params.items()}
        

    def fnet_single(self, params, x):
        return functional_call(self.diff_net, params, (x.unsqueeze(0),)).squeeze(0)

    def forward(self,x):
        scores = self.diff_net(x)
        return scores

    def inner_loop(self, x_support):
        
        self.zero_grad()
        
        # Create a param dict
        params = {k: v for k, v in self.diff_net.named_parameters()}

        for inner_epoch in range(self.task_update_num):
            # Forward pass
            self.diff_net.train()
            phi = functional_call(self.diff_net, params, (x_support,))
            self.diff_net.eval()

            # Compute J(x1)
            jac1 = vmap(jacrev(self.fnet_single), (None, 0))(params, x_support)
            s_jac = {k : self.scaling_params[k]*j for (k, j) in jac1.items()}   # Useful for later
            ntk_jac1 = [s_j.flatten(2) for s_j in s_jac.values()]   # Useful for the NTK computation

            # Compute J(x1) @ J(x2).T
            ntks = torch.stack([torch.einsum('Naf,Maf->aNM', j1, j2) for j1, j2 in zip(ntk_jac1, ntk_jac1)])
            ntks = ntks.sum(0)

            # Compute solutions to (NTK_c + eps I)^-1 (Y_c - phi_c)
            sols = []
            for c in range(self.n_way):
                inverse_term = ntks[c] + self.Noise # IF ONE WANTS TO USE FIXED JITTER
                residual = self.target_list_support[c] - phi[:, c]  # phi is of shape [n_way*n_support, n_way]

                # Solve the system (NTK_c + epsilon I_k) * result = residual
                # sols.append(torch.linalg.solve(inverse_term, residual))  # IF ONE WANTS TO USE FIXED JITTER
                sols.append(safe_cholesky_solve(inverse_term, residual))

            # Update parameters 
            tensor_update = {k : self.scaling_params[k] * sum(torch.tensordot(s_jac[k][:,c], sols[c], dims=([0], [0])) for c in range(self.n_way)) for k in params.keys()}
            params = {k: param + self.train_lr * tensor_update[k] for k, param in params.items()}

        return params


    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        self.n_query = x.size(1) - self.n_support
        assert self.n_way  ==  x.size(0), "differentialDKTIXPL do not support way change"
        
        x = x.cuda()
        x_var = Variable(x)
        x_support = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_query = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        
        # Inner updates
        inner_params = self.inner_loop(x_support)

        # outer optimization by only softmax, MAP estimate
        # scores = functional_call(self.diff_net, inner_params, (x_query,))
        # loss = self.loss_fn( scores / self.temp, y_query )
        
        ###############################################
        
        # outer optimization using LSC with NMLL for outer update
        
        # Forward pass
        phi = functional_call(self.diff_net, inner_params, (x_query,))

        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(inner_params, x_query)
        ntks = square_ntk(jac1, self.scaling_params)
        
        loss = 0
        for c in range(self.n_way):
                loss += NMLL(phi[:, c], ntks[c], self.target_list_query[c], noise=self.noise)
                
                
        ###############################################
        
        # outer optimization using LSC with NPLL for outer update
        
        # # Forward pass
        # phi_support = functional_call(self.diff_net, inner_params, (x_support,))
        # phi_query = functional_call(self.diff_net, inner_params, (x_query,))

        # # Compute J(x1), J(x2)
        # jac1 = vmap(jacrev(self.fnet_single), (None, 0))(inner_params, x_support)
        # jac2 = vmap(jacrev(self.fnet_single), (None, 0))(inner_params, x_query)

        # ntks_ss = square_ntk(jac1, self.scaling_params)
        # ntks_qs = support_query_ntk(jac1, jac2, self.scaling_params)
        # ntks_qq = square_ntk(jac2, self.scaling_params)
        
        # loss = 0
        # for c in range(self.n_way):
        #     out_mean = out_distr(phi_support[:, c], phi_query[:, c], ntks_ss[c], ntks_qs[c], ntks_qq[c], self.target_list_support[c])[0]
        #     loss += self.mse(out_mean, self.target_list_query)
        
        
        return loss

    
    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        #train
        for i, (x,_) in enumerate(train_loader):        
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            
            loss = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()#.data[0]
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
      
    
    def test_loop(self, test_loader, return_std = False, optim_based = True, n_ft=5, lr=.1): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        start_time = time.time()
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
           
            if optim_based:
                if n_ft == -1:
                    n_ft = self.task_update_num
                if lr == -1:
                    lr = self.train_lr
                correct_this, count_this = self.optim_correct(x, n_ft, lr)
            else:
                correct_this, count_this = self.correct(x)
            
            acc_all.append(correct_this/ count_this *100 )
        
        print(f"One iteration : {(time.time() - start_time)/1e0}s")
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
        
        
    def optim_correct(self, x, n_ft, lr):
        temp = .3
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        x_train = x_support
        y_train = y_support
        
        ft_diff_net = copy.deepcopy(self.diff_net).cuda()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([{'params':ft_diff_net.parameters(), 'lr':0}])
        optimizer.zero_grad()
        avg_loss=0.0
        # print("scaling params")
        # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
        # print({k: torch.min(v) for k, v in self.scaling_params.items()})
        for i_ft in range(n_ft):
            # Forward pass
            train_logit = ft_diff_net(x_train)
            inner_loss = criterion(train_logit/temp, y_train)

            # Backward pass
            inner_loss.backward()

            # Manually update each parameter using the custom learning rates
            with torch.no_grad():
                for name, param in ft_diff_net.named_parameters():
                    if name in self.scaling_params.keys():
                        if param.grad is not None:
                            param_update = lr * self.scaling_params[name]**2 * param.grad
                            param -= param_update
                        else:
                            print(f"Warning: Gradient for {name} is None. Skipping update.")

            # Clear the gradients after the update
            optimizer.zero_grad()
            
        with torch.no_grad():
            ft_diff_net.eval()
            
            output_query = ft_diff_net(x_query)
            
            _, y_pred = torch.max(output_query, 1)
            top1_correct = np.sum(y_pred.detach().cpu().numpy() == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this
    
    def correct(self, x):
        raise ValueError("Bayesian meta-testing not implemented")
     
        
    def montecarlo(self, mean_list, cov_matrix_list, times=1000, temperature=1, return_logits=False):
        samples_list = []
        for mean, cov_matrix in zip(mean_list, cov_matrix_list):
            samples = self.sample_gaussian(mean, cov_matrix, torch.Size((times, )))
            samples_list.append(samples)
        # classes, times, query points
        all_samples = torch.stack(samples_list)
        # times, classes, query points
        all_samples = all_samples.permute(1, 0, 2)
        if return_logits: return all_samples
        # compute logits
        C = all_samples.shape[1]
        all_samples = torch.sigmoid(all_samples / temperature)
        all_samples = all_samples / all_samples.sum(dim=1, keepdim=True).repeat(1, C, 1)
        # classes, query points
        avg = all_samples.mean(dim=0)
        
        return torch.argmax(avg, dim=0)     
    
    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits = self.set_forward(x)
        return logits



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
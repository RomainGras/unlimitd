# This version of differential CDKT utilizes the UnlimitdIX kernel without using the mean prior

## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, jvp, jacrev
from methods.meta_template import MetaTemplate
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import warnings
from torch.distributions import MultivariateNormal
import warnings
# warnings.filterwarnings("error", category=UserWarning)

## Our packages
import gpytorch
from time import gmtime, strftime
import random
from configs import kernel_type, autodiff
#Check if tensorboardx is installed
try:
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

# # Training CMD
# ATTENTION: to test each method use exaclty the same command but replace 'train.py' with 'test.py' or 'calibrate.py'
# CUB + data augmentation
# python3 train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --train_aug --tau=1 --loss='ELBO' --steps=2 --seed=1
# python3 train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --tau=1 --loss='ELBO' --steps=2 --seed=1

class differentialCDKTIX(MetaTemplate):
    """
    This one has one conv network and one diff network per class. No network is shared between classes here.
    The NTK is only used on the head of the network for now (TODO)
    """
    def __init__(self, model_func, diff_net, n_way, n_support):
        super(differentialCDKTIX, self).__init__(model_func, n_way, n_support)
        self.device = 'cuda:0'
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.diff_net = diff_net()
        dummy_z = torch.randn(1,3,84,84)
        # else:
        #     self.diff_net = diff_net()
        #     dummy_z = torch.randn(1, 1600)  #Conv4 dummy_z
        self.get_model_likelihood_mll() #Init model, likelihood, and mll
        if(kernel_type=="cossim"):
            self.normalize=True
        elif(kernel_type=="bncossim"):
            self.normalize=True
            latent_size = np.prod(self.feature_extractor.final_feat_dim)
            self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        else:
            self.normalize=False
        
        self.kernel_type = kernel_type
        self.get_model_likelihood_mll() #Init model, likelihood
        print(self.kernel_type)
        self.normalize = True
        self.normalize_mean = True
        self.mu_q = []
        self.sigma_q = []
        
        # if(kernel_type=="cossim"):
        #     self.normalize=True
        # elif(kernel_type=="bncossim"):
        #     self.normalize=True
        #     latent_size = np.prod(self.feature_extractor.final_feat_dim)
        #     self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        # else:
        #     # self.normalize=False
        #     pass
    
    def get_steps(self, steps):
        if steps == -1:
            self.STEPS = 'Annealing'
        else:
            self.STEPS = steps
    
    def get_temperature(self, temperature=1.):
        self.TEMPERATURE = temperature

    def get_negmean(self, mean=0.):
        if mean == 999:
            self.register_parameter("NEGMEAN", nn.Parameter(torch.zeros(1, device=self.device)))
            return True
        else:
            self.NEGMEAN = mean
            return False

    def get_loss(self, loss='ELBO'):
        self.LOSS = loss
        
    def get_kernel_type(self, kernel_type='NTK'):
        self.kernel_type = kernel_type
        self.get_model_likelihood_mll()
        
    def init_summary(self):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M%S", gmtime())
            writer_path = "./log/" + time_string
            self.writer = SummaryWriter(log_dir=writer_path)
        
    def get_model_likelihood_mll(self, train_x_list=None, train_y_list=None):
        if(train_x_list is None): train_x_list=[torch.ones(100, 64).to(self.device)]*self.n_way
        if(train_y_list is None): train_y_list=[torch.ones(100).to(self.device)]*self.n_way
        model_list = list()
        for c, (train_x, train_y) in enumerate(zip(train_x_list, train_y_list)):
            model = Kernel(device=self.device, net=self.diff_net, c=c, kernel=self.kernel_type)
            model_list.append(model)
        self.model = CombinedKernels(model_list)
        return self.model
    
    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).to(self.device)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).to(self.device)
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).to(self.device)
        y_query = np.repeat(range(self.n_way), self.n_query)

        with torch.no_grad():
            self.model.eval()
            self.feature_extractor.eval()
            self.diff_net.eval()
            
            sub_x_support = self.feature_extractor.forward(x_support).detach()
            z_support = self.compute_jacobian(sub_x_support)
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            support_outputs = self.model(z_support.reshape(z_support.size(0), -1))
            
            # to be optimized (steps should not be fixed)
            support_mu, support_sigma = self.predict_mean_field(y_support, support_outputs, steps=30)
            
            sub_x_query = self.feature_extractor.forward(x_query).detach()
            z_query = self.compute_jacobian(sub_x_query)
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            
            q_posterior_list = []
            for c in range(len(self.model.kernels)):
                posterior = self.model.kernels[c].predict(z_query, z_support, support_mu[c], support_sigma[c])
                q_posterior_list.append(posterior)

            y_pred = self.montecarlo(q_posterior_list, times=1000, temperature=self.TEMPERATURE, return_logits=True) 
        return y_pred
    
    @torch.no_grad()
    def fiveoutputs_3rd_spe(self, z_batch, model):
        sorted_z_batch = torch.empty(z_batch.shape).to(self.device)
        n_elemts = z_batch.size(0)//self.n_way

        # specialization matrix
        spe = torch.empty(self.n_way, self.n_way).to(self.device)
        for c in range(self.n_way):
            for cp in range(self.n_way):
                matrix = model[cp][c*n_elemts:(c+1)*n_elemts]

                upper_triangular = torch.triu(matrix, diagonal=0)
                # Sum all the elements above the diagonal
                sum_upper_triangular = upper_triangular.sum()
                spe[c][cp]

        flattened_spe = spe.flatten()
        transformations = dict()
        for _ in range(self.n_way):
            #Take the softmax of all the elements in the matrix
            # print(flattened_spe.reshape(5,5))
            with torch.no_grad():
                softmax_matrix = F.softmax(flattened_spe, dim=0)
            
            if torch.isnan(softmax_matrix).any():
                print("NaN found, skipping specialization")
                return z_batch, {i: i for i in range(self.n_way)}

            # print(softmax_matrix.reshape(5,5))
            rd_element_idx = torch.multinomial(softmax_matrix, num_samples=1, replacement=True)
            rd_class = rd_element_idx // self.n_way # Indice of the row
            rd_elemt = rd_element_idx % self.n_way # Indice of the column
            indices_1 = torch.tensor([self.n_way * i + rd_elemt for i in range(self.n_way)]).to(self.device)
            indices_2 = torch.tensor([i + rd_class * self.n_way for i in range(self.n_way)]).to(self.device)

            # Combine indices from both calculations, ensuring uniqueness if necessary
            all_indices = torch.cat((indices_1, indices_2)).unique()
            flattened_spe[all_indices] = float('-inf')
            # print(f"Input number {rd_elemt[0]} will have class number {rd_class[0]}")
            
            transformations[rd_class] = rd_elemt
            sorted_z_batch[n_elemts*rd_class:n_elemts*(rd_class+1)] = z_batch[n_elemts*rd_elemt:n_elemts*(rd_elemt+1)]
        
        # keys = torch.arange(self.n_way).tolist()  # For random transformations
        # values = torch.randperm(self.n_way).tolist() 
        return sorted_z_batch.to(self.device), transformations # z_batch, {k: v for k, v in zip(keys, values)} 


    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        if self.STEPS == 'Annealing':
            STEPS = 1 + epoch // 50
        else:
            STEPS = self.STEPS
        
        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).to(self.device)
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).to(self.device))
            x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).to(self.device)
            y_support = np.repeat(range(self.n_way), self.n_support)
            x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).to(self.device)
            y_query = np.repeat(range(self.n_way), self.n_query)
            x_train = x_all
            # print(f"x_train shape : {x_train.shape}")
            # print(f"y_train shape : {y_all.shape}")
            # print(f"y_train : {y_all}")
            
            y_train = y_all
            # print(x_train.shape, y_train.shape)

            self.model.train()
            self.feature_extractor.train()
            
            z_train = self.feature_extractor.forward(x_train)  # First CNN or Transformer layer gives an embedding
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            
            # print(f"z shape : {z_train.shape}")
            output = self.model(z_train.reshape(z_train.size(0),-1))
            # print(f"output shape : {output.shape}")
            
            # Comment this for non specialized cdkt :
            # _, transformations = self.fiveoutputs_3rd_spe(z_batch = z_train, model = output)
            # sorted_z_train = z_train.clone()
            # for (rd_class, rd_elemt) in transformations.items():
            #     z_train[(self.n_support + self.n_query)*rd_class:(self.n_support + self.n_query)*(rd_class+1)] = sorted_z_train[(self.n_support + self.n_query)*rd_elemt:(self.n_support + self.n_query)*(rd_elemt+1)]
            # output = self.model(z_train.reshape(z_train.size(0),-1))
            
            lenghtscale = 0.0
            outputscales = []
            meanscale = 0.0
            for idx, single_model in enumerate(self.model.kernels):
                if(single_model.covar_module.base_kernel.lengthscale is not None):
                    lenghtscale+=single_model.covar_module.base_kernel.lengthscale.mean().cpu().detach().numpy().squeeze()
                if(single_model.covar_module.outputscale is not None):
                    outputscales.append(single_model.covar_module.outputscale.cpu().detach().numpy().squeeze())

            if(single_model.covar_module.base_kernel.lengthscale is not None): lenghtscale /= float(len(self.model.kernels))
            # if(single_model.covar_module.outputscale is not None): outputscale /= float(len(self.model.kernels))

        
            ## Optimize
            optimizer.zero_grad()
            
            if self.LOSS == 'ELBO':
                loss = self.MeanFieldELBO(y=y_train, output=output, steps=STEPS, REQUIRES_GRAD=False, temperature=self.TEMPERATURE)
            else:
                loss = self.MeanFieldPredictiveLoglikelihood(y_train[:self.n_support * self.n_way], z_train[:self.n_support * self.n_way], y_train, z_train, steps=STEPS, REQUIRES_GRAD=False, times=1000, tau=self.TEMPERATURE)

            # Uncomment this line if want to add the condition of scalar product of scaling params for each class to be low : this add 50*sum <normalized_spi, normalized_spj> on all i<j to the loss
            # scaling_params = []
            # for single_model in self.model.kernels:
            #     scaling_params.append(single_model.covar_module.base_kernel.sp)
            # scaling_params = torch.cat(scaling_params, dim=0)
            # print(f"scaling_params : {scaling_params}")
            # norm_scaling_params = F.normalize(scaling_params, p=2, dim=1)
            # print(norm_scaling_params.shape)
            # inner_prod_sp = norm_scaling_params @ norm_scaling_params.T
            # print(f"Inner prod sp : {inner_prod_sp}")
            # upper_triangular = torch.triu(inner_prod_sp, diagonal=1)
            # above_diagonal_elements = upper_triangular[upper_triangular != 0]
            # print(f"above_diagonal_elements : {above_diagonal_elements}")
            # average_sp = upper_triangular.mean()*2
            # print(f"average_sp : {average_sp}")
            # loss += 50 * average_sp
            
            try:
                torch.nan_to_num(loss).backward() 
                if not all([torch.isfinite(p.grad).all() for p in self.feature_extractor.parameters()]):
                    print("Nan in the gradients, skipping this iteration.")
                else:
                    optimizer.step()
            except:
                pass

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss.item(), self.iteration)

            # Eval on the query (validation set)
            # if i % print_freq==0:
            #     with torch.no_grad():
            #         self.model.eval()
            #         self.feature_extractor.eval()
            #         z_support = self.feature_extractor.forward(x_support).detach()
            #         if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            #         support_outputs = self.model(z_support)
                    
            #         # to be optimized (steps should not be fixed)
            #         support_mu, support_sigma = self.predict_mean_field(y_support, support_outputs, steps=30)
                    
            #         z_query = self.feature_extractor.forward(x_query).detach()
            #         if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                    
            #         q_posterior_list = []
            #         for c in range(len(self.model.kernels)):
            #             posterior = self.model.kernels[c].predict(z_query, z_support, support_mu[c], support_sigma[c])
            #             q_posterior_list.append(posterior)
                    
            #         y_pred = self.montecarlo(q_posterior_list, times=1000, temperature=self.TEMPERATURE)
            #         # y_pred = self.expectation(q_posterior_list)     
            #         y_pred = y_pred.cpu().numpy()         
                    
            #         accuracy_query = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
            #         if(self.writer is not None): self.writer.add_scalar('GP_query_accuracy', accuracy_query, self.iteration)

            #         # C * N
            #         train_mu = self.mu
            #         train_pred = torch.argmax(train_mu, dim=0).cpu().numpy()
            #         if len(train_pred) == len(y_train):
            #             train_acc = (np.sum(train_pred==y_train.cpu().numpy()) / float(len(y_train))) * 100.0
            #         else:
            #             train_acc = (np.sum(train_pred==y_support) / float(len(y_support))) * 100.0

            # if i % print_freq==0:
            #     # if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
            #     print('Epoch [{:d}] [{:d}/{:d}] | Mean {:f} | Outscale {:f} | Lenghtscale {:f} | Loss {:f} | Query {:f} | Train {:f}'.format(epoch, i, len(train_loader), meanscale, outputscale, lenghtscale, loss.item(), accuracy_query, train_acc))

            if i % print_freq==0:
                # if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                outputscales_str = ", ".join("{:.6f}".format(scale) for scale in outputscales)
                print('Epoch [{:d}] [{:d}/{:d}] | Outscale {} | Average sp {:f} | Loss {:f}'.format(epoch, i, len(train_loader), outputscales_str, average_sp, loss.item()))
                for single_model in self.model.kernels:
                    print(single_model.covar_module.base_kernel.sp.detach().reshape(-1).cpu().numpy()[0:10])

    def correct(self, x):
        ##Dividing input x in query and support set
        
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).to(self.device)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).to(self.device)
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).to(self.device)
        y_query = np.repeat(range(self.n_way), self.n_query)

        with torch.no_grad():
            self.model.eval()
            self.feature_extractor.eval()
            
            z_support = self.feature_extractor.forward(x_support).detach()  # First CNN or Transformer layer gives an embedding
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            support_outputs = self.model(z_support.reshape(z_support.size(0),-1))
            
            # Comment this for non specialized cdkt :
            # _, transformations = self.fiveoutputs_3rd_spe(z_batch = z_support, model = support_outputs)
            # sorted_z_support = z_support.clone()
            # for (rd_class, rd_elemt) in transformations.items():
            #     z_support[self.n_support*rd_class:self.n_support*(rd_class+1)] = sorted_z_support[self.n_support*rd_elemt:self.n_support*(rd_elemt+1)]
            # support_outputs = self.model(z_support.reshape(z_support.size(0), -1))
            
            # to be optimized (steps should not be fixed)
            support_mu, support_sigma = self.predict_mean_field(y_support, support_outputs, steps=30)
            
            z_query = self.feature_extractor.forward(x_query).detach()  # First CNN or Transformer layer gives an embedding
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            
            # Comment this for non specialized cdkt :
            # Sort z_query in the same way as z_support
            # sorted_z_query = z_query.clone()
            # for (rd_class, rd_elemt) in transformations.items():
            #     z_query[self.n_query*rd_class:self.n_query*(rd_class+1)] = sorted_z_query[self.n_query*rd_elemt:self.n_query*(rd_elemt+1)]
            
            q_posterior_list = []
            for c in range(len(self.model.kernels)):
                # print(f"z_query : {z_query.shape}")
                # print(f"diff_net_query : {self.diff_net(z_query).shape}")
                # print(f"z_support : {z_support.shape}")
                # print(f"diff_net_support : {self.diff_net(z_support).shape}")
                posterior = self.model.kernels[c].predict(z_query.reshape(z_query.size(0), -1), z_support.reshape(z_support.size(0), -1), support_mu[c], support_sigma[c])
                # print(f"posterior mean : {posterior.mean.shape}")
                # posterior.mean += self.diff_net(z_query)[:, c]
                q_posterior_list.append(posterior)
            
            y_pred = self.montecarlo(q_posterior_list, times=10000, temperature=self.TEMPERATURE)     
            y_pred = y_pred.cpu().numpy() 
            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this

    def test_loop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            if(i % 100==0):
                acc_mean = np.mean(np.asarray(acc_all))
                print('Test | Batch {:d}/{:d} | Acc {:f}'.format(i, len(test_loader), acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        if(return_std): return acc_mean, acc_std
        else: return acc_mean
            
    def predict_mean_field(self, y, output, steps=10):
        temperature = self.TEMPERATURE
        with torch.no_grad():
            y = torch.tensor(y).long().detach().to(self.device)
            # initiate params
            # N = self.n_support * self.n_way
            N = len(y) 
            C = self.n_way
            tilde_f = torch.empty(C, N).to(self.device)
            nn.init.normal_(tilde_f, mean=0, std=0.01)
            gamma = torch.empty(C, N).to(self.device)
            nn.init.uniform_(gamma, a=0, b=1)
            alpha = torch.empty(N).to(self.device)
            nn.init.uniform_(alpha, a=5, b=10)
            sigma = torch.cat([0.01 * torch.eye(N).unsqueeze(0)] * C, dim=0).to(self.device)
            mu = torch.empty(C, N).to(self.device)
            nn.init.normal_(mu, mean=0, std=0.01)
            omega = torch.empty(C, N).to(self.device)
            nn.init.uniform_(omega, a=0, b=1)

            # mean_field
            Y = F.one_hot(y, num_classes = C).T # C, N
            mean_vec = torch.ones((output.shape[0], output.shape[1])).to(self.device) * self.NEGMEAN

            for step in range(steps):
                # 4.4a
                tilde_f = torch.sqrt(mu ** 2 + torch.diagonal(sigma, dim1=1, dim2=2)) / temperature
                # 4.4b
                psi = torch.digamma(alpha)
                psi = psi.repeat(C, 1) # make it C, N
                gamma = torch.exp(psi - 0.5 * mu  / temperature) / (2 * C * torch.cosh(0.5 * tilde_f).clamp(min=1e-6))
                # 4.4c
                alpha = gamma.sum(axis=0) + 1
                # 4.4d
                try:
                    sigma = torch.linalg.inv(torch.linalg.inv(output) + torch.diag_embed(omega  / temperature ** 2))
                except:
                    sigma = output
                # 4.4e
                try:
                    mu = 0.5 / temperature * torch.bmm(sigma, (Y - gamma).unsqueeze(2)).squeeze(2) + torch.bmm(torch.linalg.inv(torch.bmm(output, torch.diag_embed(omega  / temperature ** 2)) + torch.eye(output.shape[1], device=self.device).unsqueeze(0).repeat(output.shape[0], 1, 1)), mean_vec.unsqueeze(2)).squeeze(2)
                except:
                    mu = 0.5 / temperature * torch.bmm(sigma, (Y - gamma).unsqueeze(2)).squeeze(2) + mean_vec
                # 4.4f
                omega = (gamma + Y) * torch.tanh(0.5 * tilde_f) * 0.5 / tilde_f.clamp(min=1e-6)  
                omega = omega.clamp(min=1e-6)
            return mu, sigma  
    
    def montecarlo(self, q_posterior_list, times=1000, temperature=1, return_logits=False):
        samples_list = []
        for posterior in q_posterior_list:
            samples = posterior.rsample(torch.Size((times, )))
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
    
    def MeanFieldELBO(self, y, output, steps=2, REQUIRES_GRAD=False, temperature=1):
        y = torch.tensor(y).long()
        N = (self.n_support + self.n_query) * self.n_way
        C = self.n_way
        tilde_f = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.normal_(tilde_f, mean=0, std=1)
        gamma = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.uniform_(gamma, a=0, b=1)
        alpha = torch.empty(N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.uniform_(alpha, a=5, b=10)
        sigma = torch.cat([1e-2 * torch.eye(N, requires_grad=REQUIRES_GRAD).unsqueeze(0)] * C, dim=0).reshape(C, N, N).to(self.device)
        mu = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.normal_(mu, mean=0, std=1)
        omega = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.uniform_(omega, a=0, b=1)
        psi = torch.empty_like(omega, requires_grad=REQUIRES_GRAD).to(self.device)
        # output = output.double()

        try:
            if torch.isnan(self.sigma.detach()).any() or torch.isnan(self.mu.detach()).any():
                self.sigma = sigma.detach()
                self.mu = mu.detach()
            sigma = self.sigma.detach()
            mu = self.mu.detach()
        except:
            self.sigma = sigma
            self.mu = mu
            
        Y = F.one_hot(y, num_classes = C).T # C, N
        mean_vec = torch.ones((output.shape[0], output.shape[1])).to(self.device) * self.NEGMEAN

        tilde_f_ls = [tilde_f]
        gamma_ls = [gamma]
        alpha_ls = [alpha]
        sigma_ls = [sigma]
        mu_ls = [mu]
        omega_ls = [omega]
        psi_ls = [psi]
        
        for step in range(steps):
            # 4.4a
            tilde_f_ls.append(torch.sqrt(mu_ls[-1].data ** 2 + torch.diagonal(sigma_ls[-1].data, dim1=1, dim2=2)) / temperature)
            # 4.4b
            # psi_ls.append(torch.polygamma(1, alpha_ls[-1]).repeat(C, 1)) # make it C, N
            psi_ls.append(torch.digamma(alpha_ls[-1]).repeat(C, 1))
            gamma_ls.append((torch.exp(psi_ls[-1] - 0.5 * mu_ls[-1] / temperature) / (2 * C * torch.cosh(0.5 * tilde_f_ls[-1]))).nan_to_num(nan=0., posinf=0., neginf=0.).clamp(min=1e-6))
            # 4.4c
            alpha_ls.append(gamma_ls[-1].sum(axis=0) + 1)
            # 4.4d
            try:
                sigma_ls.append(torch.linalg.inv(torch.linalg.inv(output) + torch.diag_embed(omega_ls[-1] / temperature ** 2)))
            except:
                sigma_ls.append(output)

            # 4.4e
            # mu_ls.append(0.5 / temperature * torch.bmm(sigma_ls[-1], (Y - gamma_ls[-1]).unsqueeze(2)).squeeze(2))
            try:
                mu_ls.append(0.5 / temperature * torch.bmm(sigma_ls[-1], (Y - gamma_ls[-1]).unsqueeze(2)).squeeze(2) + torch.bmm(torch.linalg.inv(torch.bmm(output, torch.diag_embed(omega_ls[-1]  / temperature ** 2)) + torch.eye(output.shape[1], device=self.device).unsqueeze(0).repeat(output.shape[0], 1, 1)), mean_vec.unsqueeze(2)).squeeze(2))
            except:
                mu_ls.append(0.5 / temperature * torch.bmm(sigma_ls[-1], (Y - gamma_ls[-1]).unsqueeze(2)).squeeze(2) + mean_vec)
            # 4.4f
            omega_ls.append(((gamma_ls[-1] + Y) * torch.tanh(0.5 * tilde_f_ls[-1]) * 0.5 / tilde_f_ls[-1]).clamp(min=1e-6))
        
        eps = 1e-6
        ELBO = 0.
        # 0.5 * (omega[-1] * tilde_f[-1] ** 2).sum() appears in line 1 and last line so addition is 0
        ELBO = ELBO - math.log(2) * (Y + gamma_ls[-1]).sum() + 0.5 * ((Y - gamma_ls[-1]) * mu_ls[-1] / temperature).sum()
        L = psd_safe_cholesky(output)
        ELBO = ELBO - 0.5 * (torch.logdet(output).sum() - torch.logdet(sigma_ls[-1]).sum() + torch.cholesky_solve((sigma_ls[-1] + torch.bmm((self.NEGMEAN - mu_ls[-1].reshape(C, N, 1)), (self.NEGMEAN - mu_ls[-1].reshape(C, 1, N)))), L).diagonal(dim1=-1, dim2=-2).sum())
        ELBO = ELBO + alpha_ls[-1].sum() + torch.lgamma(alpha_ls[-1]).sum() + ((1 - alpha_ls[-1]) * torch.digamma(alpha_ls[-1])).sum()
        ELBO = ELBO - (gamma_ls[-1] * (torch.log(gamma_ls[-1] + eps) - 1)).sum() + (gamma_ls[-1] * (torch.digamma(alpha_ls[-1]) - math.log(C)).unsqueeze(0).repeat(C, 1)).sum() - (alpha_ls[-1] / C).sum()
        ELBO = ELBO - ((Y + gamma_ls[-1]) * torch.log((torch.cosh(0.5 * tilde_f_ls[-1]) + eps).nan_to_num())).sum()
        self.mu = mu_ls[-1].detach()
        self.sigma = sigma_ls[-1].detach()
    
        return - ELBO
    
    # UNCHANGED :
    def MeanFieldPredictiveLoglikelihood(self, y_support, z_support, y_query, z_query, steps=2, REQUIRES_GRAD=False, times=32, tau=1):
        # with torch.no_grad():
        temperature = tau
        output = self.model(z_support)
        y = torch.tensor(y_support).long()
        y_query = torch.tensor(y_query).long().to(self.device)
        N = len(y_support)
        C = self.n_way
        tilde_f = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.normal_(tilde_f, mean=0, std=1)
        gamma = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.uniform_(gamma, a=0, b=1)
        alpha = torch.empty(N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.uniform_(alpha, a=5, b=10)
        sigma = torch.cat([1e-2 * torch.eye(N, requires_grad=REQUIRES_GRAD).unsqueeze(0)] * C, dim=0).reshape(C, N, N).to(self.device)
        mu = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.normal_(mu, mean=0, std=1)
        omega = torch.empty(C, N, requires_grad=REQUIRES_GRAD).to(self.device)
        nn.init.uniform_(omega, a=0, b=1)
        psi = torch.empty_like(omega, requires_grad=REQUIRES_GRAD).to(self.device)
        try:
            sigma = self.sigma.detach()
            mu = self.mu.detach()
        except:
            self.sigma = sigma
            self.mu = mu

        Y = F.one_hot(y, num_classes = C).T.to(self.device) # C, N
        mean_vec = torch.ones((output.shape[0], output.shape[1])).to(self.device) * self.NEGMEAN

        tilde_f_ls = [tilde_f]
        gamma_ls = [gamma]
        alpha_ls = [alpha]
        sigma_ls = [sigma]
        mu_ls = [mu]
        omega_ls = [omega]
        psi_ls = [psi]
        for step in range(steps):
            # 4.4a
            tilde_f_ls.append(torch.sqrt(mu_ls[-1].data ** 2 + torch.diagonal(sigma_ls[-1].data, dim1=1, dim2=2)) / temperature)
            # 4.4b
            # psi_ls.append(torch.polygamma(1, alpha_ls[-1]).repeat(C, 1)) # make it C, N
            psi_ls.append(torch.digamma(alpha_ls[-1]).repeat(C, 1))
            gamma_ls.append((torch.exp(psi_ls[-1] - 0.5 * mu_ls[-1] / temperature) / (2 * C * torch.cosh(0.5 * tilde_f_ls[-1]))).nan_to_num(nan=0., posinf=0., neginf=0.).clamp(min=1e-6))
            # 4.4c
            alpha_ls.append(gamma_ls[-1].sum(axis=0) + 1)
            # 4.4d
            try:
                sigma_ls.append(torch.linalg.inv(torch.linalg.inv(output) + torch.diag_embed(omega_ls[-1] / temperature ** 2)))
            except:
                sigma_ls.append(output)

            # 4.4e
            # mu_ls.append(0.5 / temperature * torch.bmm(sigma_ls[-1], (Y - gamma_ls[-1]).unsqueeze(2)).squeeze(2))
            try:
                mu_ls.append(0.5 / temperature * torch.bmm(sigma_ls[-1], (Y - gamma_ls[-1]).unsqueeze(2)).squeeze(2) + torch.bmm(torch.linalg.inv(torch.bmm(output, torch.diag_embed(omega_ls[-1]  / temperature ** 2)) + torch.eye(output.shape[1], device=self.device).unsqueeze(0).repeat(output.shape[0], 1, 1)), mean_vec.unsqueeze(2)).squeeze(2))
            except:
                mu_ls.append(0.5 / temperature * torch.bmm(sigma_ls[-1], (Y - gamma_ls[-1]).unsqueeze(2)).squeeze(2) + mean_vec)
            # 4.4f
            omega_ls.append(((gamma_ls[-1] + Y) * torch.tanh(0.5 * tilde_f_ls[-1]) * 0.5 / tilde_f_ls[-1]).clamp(min=1e-6))
                
        self.mu = mu_ls[-1].detach()
        self.sigma = sigma_ls[-1].detach()
        q_posterior_list = []
        for c in range(len(self.model.kernels)):
            posterior = self.model.kernels[c].predict(z_query, z_support, mu_ls[-1][c], sigma_ls[-1][c])
            q_posterior_list.append(posterior)
        samples_list = []
        for posterior in q_posterior_list:
            samples = posterior.rsample(torch.Size((times, )))
            samples_list.append(samples)
        # classes, times, query points
        all_samples = torch.stack(samples_list).to(self.device)
        # times, classes, query points
        all_samples = all_samples.permute(1, 0, 2)
        # compute logits
        # classes, query points
        logits = F.log_softmax(F.logsigmoid(all_samples / temperature).mean(0), 0)
        return nn.CrossEntropyLoss()(logits.T, y_query)

class NTKernel(gpytorch.kernels.Kernel):
    def __init__(self, net, scaling_params, c, normalize, **kwargs):  # i is the output index. Each index or the output has its own kernel that is sigma * grad(NN_i(x))^T @ grad(NN_i(x))
        super(NTKernel, self).__init__(**kwargs)
        self.net = net
        self.params = {k: v for k, v in net.named_parameters()}
        self.c = c
        self.init_f_net_single()
        self.normalize = normalize
        self.scaling_params = scaling_params

    def init_f_net_single(self):
        self.zero_grad()
        def fnet_single(params, x):
            return functional_call(self.net, params, (x.unsqueeze(0),)).squeeze(0)
        self.fnet_single = fnet_single
        
    def forward(self, x1, x2, diag=False, **params):
        x1 = x1.reshape(x1.size(0), 3, 84, 84)
        x2 = x2.reshape(x2.size(0), 3, 84, 84)
        
        self.zero_grad()
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(self.params, x1)
        jac1 = [(self.scaling_params[k] * j).flatten(2) for k, j in jac1.items()] 

        jac2 = vmap(jacrev(self.fnet_single), (None, 0))(self.params, x2)
        jac2 = [(self.scaling_params[k] * j).flatten(2) for k, j in jac2.items()] 

        result = torch.stack([torch.einsum('Naf,Maf->NMa', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        result = result[:, :, self.c]  # Only interested by the value of c'output
        if diag:
            return result.diag()
        return result


class Kernel(nn.Module):
    '''
    Parameters learned by the model:
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, device, net, c, kernel='rbf'):
        super().__init__()
        self.device = device
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = None
        
        if(kernel=='NTK'):
             self.covar_module = gpytorch.kernels.ScaleKernel(NTKernel(net, scaling_params, c, normalize=False))
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")
        self.covar_module = self.covar_module.to(self.device)

    def forward(self, x):
        covar_x = self.covar_module(x).evaluate()
        while not torch.all(torch.linalg.eigvals(covar_x).real > 1e-6).item():
            covar_x += 1e-6 * torch.eye(covar_x.shape[0], device=self.device)
        return covar_x
    
    def predict(self, z_query, z_support, support_mu, support_sigma, noise=0.1):
        K_lt = self.covar_module(z_support, z_query).evaluate()
        K_tt = self.covar_module(z_query).evaluate()
        covar_x = self.covar_module(z_support).evaluate()

        L = psd_safe_cholesky(covar_x)
        mean = K_lt.T @ torch.cholesky_solve(support_mu.unsqueeze(1), L).squeeze()
        covar = K_tt - K_lt.T @ torch.cholesky_solve(K_lt, L) + K_lt.T @ torch.cholesky_solve(support_sigma, L) @ torch.cholesky_solve(K_lt, L)
        
        return MultivariateNormal(mean, scale_tril=psd_safe_cholesky(covar))

class CombinedKernels(nn.Module):
    def __init__(self, kernel_list) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(kernel_list)
    
    def forward(self, x):
        covar = []
        mean = []
        for kernel in self.kernels:
            covar_x = kernel(x)
            # mean.append(mean_x)
            covar.append(covar_x)
        return torch.stack(covar, dim=0)


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

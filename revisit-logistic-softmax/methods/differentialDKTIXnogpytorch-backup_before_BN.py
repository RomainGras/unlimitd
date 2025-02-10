## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, jvp, jacrev
from methods.meta_template import MetaTemplate

from gp_utils import square_ntk, NMLL, support_query_ntk, out_distr
import time
from time import gmtime, strftime
import random
from configs import kernel_type, autodiff

import copy
import torch.optim as optim
#Check if tensorboardx is installed
try:
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

# # Training CMD
# ATTENTION: to test each method use exaclty the same command but replace 'train.py' with 'test.py'
# Omniglot->EMNIST without data augmentation
# python3 train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1
# python3 train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=5
# CUB + data augmentation
# python3 train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --train_aug
# python3 train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug

class differentialDKTIXnogpy(MetaTemplate):
    def __init__(self, model_func, diff_net, n_way, n_support):
        super(differentialDKTIXnogpy, self).__init__(model_func, n_way, n_support)
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.diff_net = diff_net()
        
        # Add N scaling parameters, initializing them as one
        net_params = {k: v for k, v in self.diff_net.named_parameters()}
        self.scaling_params = {k: torch.ones_like(v).cuda() for k, v in net_params.items()}
        
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
        
        self.n_train = 17
        self.self_support = 1
        self.n_way =5
        self.noise = 0
        self.target_list_train = contruct_target_list(self.n_train, self.n_way)
        self.target_list_support = contruct_target_list(self.n_support, self.n_way)
        
        self.init_jac_func()
        
        if(kernel_type=="cossim"):
            self.normalize=True
        elif(kernel_type=="bncossim"):
            self.normalize=True
            latent_size = np.prod(self.feature_extractor.final_feat_dim)
            self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        else:
            self.normalize=False
        
        print(f"Normalization : {self.normalize}")

    def init_summary(self):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M%S", gmtime())
            writer_path = "./log/" + time_string
            self.writer = SummaryWriter(log_dir=writer_path)

            
    def init_jac_func(self):
        """
        Computes the functional call of a single input, to be differentiated in parallel using vmap
        """
        def fnet_single(params, x):
            return functional_call(self.diff_net, params, (x.unsqueeze(0),)).squeeze(0)
        
        self.jac_func = vmap(jacrev(fnet_single), (None, 0))
        
        
    def set_forward(self, x, is_feature=False):
        pass

    
    def set_forward_loss(self, x):
        pass 
    
    def empirical_ntk_ntk_vps(self, params, x1):
        """
        In construction still
        """
        def get_ntk(x1, x2):
            def func_x1(params):
                return self.fnet_single(params, x1)

            output, vjp_fn = vjp(func_x1, params)

            def get_ntk_slice(vec):
                # This computes ``vec @ J(x2).T``
                # `vec` is some unit vector (a single slice of the Identity matrix)
                vjps = vjp_fn(vec)
                # This computes ``J(X1) @ vjps``
                _, jvps = jvp(func_x1, (params,), vjps)
                return jvps

            # Here's our identity matrix
            basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
            return vmap(get_ntk_slice)(basis)

        # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
        # Since the x1, x2 inputs to ``empirical_ntk_ntk_vps`` are batched,
        # we actually wish to compute the NTK between every pair of data points
        # between {x1} and {x2}. That's what the ``vmaps`` here do.
        result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x1)
        return torch.einsum('NMaa->aNM', result)


    def jacobian_autodiff(self, params, z):
        f_net = functional_call(self.diff_net, params, (z,))
        return torch.autograd.functional.jacobian(f_net, params)
        
    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        # optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-4},
        #                               {'params': self.feature_extractor.parameters(), 'lr': 1e-3}])
        
        
        for i, (x,_) in enumerate(train_loader):
            # starting_time = time.time()
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).cuda()
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).cuda())
            x_train = x_all
            y_train = y_all

            self.feature_extractor.train()
            self.diff_net.train()
            
            z_train = self.feature_extractor.forward(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            
            # TODO IMPLEMENT LENGTHSCALE & OUTPUTSCALE
            
            #lenghtscale = 0.0
            #noise = 0.0
            #outputscale = 0.0
            #for idx, single_model in enumerate(self.model.models):
                #print(f"target_list[idx] : {target_list[idx].shape}") # [85]
                #print(f"self.diff_net(z_train) : {self.diff_net(z_train).shape}")  # [85,5]
                #single_model.set_train_data(inputs=z_train.reshape(z_train.size(0), -1), targets=target_list[idx]-self.diff_net(z_train)[:, idx], strict=False)
                #if hasattr(single_model.covar_module, 'lengthscale') and (single_model.covar_module.lengthscale is not None): #Originally if(single_model.covar_module.base_kernel.lengthscale is not None):
                #    lenghtscale+=single_model.covar_module.lengthscale.mean().cpu().detach().numpy().squeeze()
                #noise+=single_model.likelihood.noise.cpu().detach().numpy().squeeze()
                #if hasattr(single_model.covar_module, 'outputscale') and (single_model.covar_module.outputscale is not None): #Originally if(single_model.covar_module.outputscale is not None):
                #    outputscale+=single_model.covar_module.outputscale.cpu().detach().numpy().squeeze()
            #if hasattr(single_model.covar_module, 'lengthscale') and (single_model.covar_module.lengthscale is not None):# Originally if(single_model.covar_module.base_kernel.lengthscale is not None):
            #    lenghtscale /= float(len(self.model.models))
            #noise /= float(len(self.model.models))
            #if hasattr(single_model.covar_module, 'outputscale') and (single_model.covar_module.outputscale is not None): # Originally if(single_model.covar_module.outputscale is not None):
            #    outputscale /= float(len(self.model.models))

            ## Optimize
            optimizer.zero_grad()
            
            # Create a param dict
            params = {k: v for k, v in self.diff_net.named_parameters()}
            
            # Forward pass
            phi = functional_call(self.diff_net, params, (z_train,))
            
            ###############
            
            # Compute J(x1)
            jac1 = self.jac_func(params, z_train)
            
            del params
            
            # Compute the NTK
            ntks = square_ntk(jac1, self.scaling_params)
            
            ###############
            # TODO
            # ntks = self.empirical_ntk_ntk_vps(params, z_train)
            
            loss = 0
            for c in range(self.n_way):
                loss+=NMLL(phi[:, c], ntks[c], self.target_list_train[c], noise=self.noise)
            
            loss.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()
            # Print the current memory allocated on the GPU (in bytes)
            # print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2} MB")

            # Print the maximum memory allocated on the GPU (in bytes)
            # print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB")

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss, self.iteration)
            # print("scaling params")
            # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
            # print("diff_net params")
            # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.diff_net.named_parameters()})
            

            if i % print_freq==0:
                
                #Eval on the query (validation set)
                # with torch.no_grad():
                    # x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
                    # y_support = np.repeat(range(self.n_way), self.n_support)
                    # x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
                    # y_query = np.repeat(range(self.n_way), self.n_query)

                    # self.feature_extractor.eval()
                    # self.diff_net.eval()
                    
                    # z_support = self.feature_extractor.forward(x_support).detach()
                    # if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
                    # z_query = self.feature_extractor.forward(x_query).detach()
                    # if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                    
                    # Create a param dict
                    # params = {k: v for k, v in self.diff_net.named_parameters()}
                    # Forward pass
                    # phi_support = functional_call(self.diff_net, params, (z_support,))
                    # phi_query = functional_call(self.diff_net, params, (z_query,))
                    # Compute Jacobians
                    # jac1 = vmap(jacrev(self.fnet_single), (None, 0))(params, z_support)
                    # jac2 = vmap(jacrev(self.fnet_single), (None, 0))(params, z_query)
                    # Compute the NTKs
                    # ntks_ss = square_ntk(jac1, self.scaling_params)
                    # ntks_qs = support_query_ntk(jac1, jac2, self.scaling_params)
                    # ntks_qq = square_ntk(jac2, self.scaling_params)
                    
                    # predictions_list = list()
                    # for c in range(self.n_way):
                    #     out_mean = out_distr(phi_support[:, c], phi_query[:, c], ntks_ss[c], ntks_qs[c], ntks_qq[c], self.target_list_support[c])[0]
                    #     predictions_list.append(torch.sigmoid(out_mean).cpu().detach().numpy())
                        
                    # y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
                    # accuracy_query = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
                    # if(self.writer is not None): self.writer.add_scalar('GP_query_accuracy', accuracy_query, self.iteration)
                    
                # if(self.writer is not None): self.writer.add_histogram('phi_support', phi_support, self.iteration)
                mean_vec_avg = torch.mean(self.diff_net(z_train).detach(), dim=0)
                mean_vec_avg_str = ", ".join("{:.6f}".format(avg) for avg in mean_vec_avg)
                # print('Epoch [{:d}] [{:d}/{:d}] | Mean functions {} | Noise {:f} | Loss {:f} | Query acc {:f}'.format(epoch, i, len(train_loader), mean_vec_avg_str, self.noise, loss.item(), accuracy_query))
                
                print('Epoch [{:d}] [{:d}/{:d}] | Mean functions {} | Noise {:f} | Loss {:f}'.format(epoch, i, len(train_loader), mean_vec_avg_str, self.noise, loss.item()))
                # print("Scaling Params : ")
                # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
            
            
            # print(f"{i} iteration : {(time.time()-starting_time)}s")
    
    @torch.no_grad()
    def correct(self, x, N=0, laplace=False):
        ##Dividing input x in query and support set
        # print(f"x shape {x.shape}")
        torch.cuda.empty_cache()
        
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        x_train = x_support
        y_train = y_support
        
        z_support = self.feature_extractor.forward(x_support).detach()
        if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
        z_query = self.feature_extractor.forward(x_query).detach()
        if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
        
        # Comment these lines for no specialization
        #_, transformations = self.fiveoutputs_3rd_spe(z_batch = z_train)
        #sorted_z_train = z_train.clone()
        #for (rd_class, rd_elemt) in transformations.items():
        #    z_train[self.n_support*rd_class:self.n_support*(rd_class+1)] = sorted_z_train[self.n_support*rd_elemt:self.n_support*(rd_elemt+1)]

        # Create a param dict
        params = {k: v for k, v in self.diff_net.named_parameters()}
        # Forward pass
        phi_support = functional_call(self.diff_net, params, (z_support,))
        phi_query = functional_call(self.diff_net, params, (z_query,))
        # Compute Jacobians
        jac1 = self.jac_func(params, z_support)
        jac2 = self.jac_func(params, z_query)
        
        del params
        # Compute the NTKs
        ntks_ss = square_ntk(jac1, self.scaling_params)
        ntks_qs = support_query_ntk(jac1, jac2, self.scaling_params)
        ntks_qq = square_ntk(jac2, self.scaling_params)

        predictions_list = list()
        for c in range(self.n_way):
            out_mean = out_distr(phi_support[:, c], phi_query[:, c], ntks_ss[c], ntks_qs[c], ntks_qq[c], self.target_list_support[c])[0]
            predictions_list.append(torch.sigmoid(out_mean).cpu().detach().numpy())
        y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
        accuracy_query = (np.sum(y_pred==y_query) / float(len(y_support))) * 100.0
        y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
        top1_correct = np.sum(y_pred == y_query)
        count_this = len(y_query)
        return float(top1_correct), count_this, 0 # originally : avg_loss/float(N+1e-10)

    
    def optim_correct(self, x, n_ft, lr):
        temp = .3
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        x_train = x_support
        y_train = y_support
        
        ft_feature_extr = copy.deepcopy(self.feature_extractor).cuda()
        ft_diff_net = copy.deepcopy(self.diff_net).cuda()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([# {'params':ft_feature_extr.parameters(), 'lr':lr},
                               {'params':ft_diff_net.parameters(), 'lr':0}])
        optimizer.zero_grad()
        avg_loss=0.0
        # print("scaling params")
        # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
        # print({k: torch.min(v) for k, v in self.scaling_params.items()})
        for i_ft in range(n_ft):
            # Forward pass
            z_train = ft_feature_extr(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            train_logit = ft_diff_net(z_train)
            inner_loss = criterion(train_logit/temp, y_train)

            # Backward pass
            inner_loss.backward()

            # Manually update each parameter using the custom learning rates
            with torch.no_grad():
                for name, param in ft_diff_net.named_parameters():
                    if name in self.scaling_params.keys():
                        if param.grad is not None:
                            param_update = lr * self.scaling_params[name] * param.grad
                            param -= param_update
                        else:
                            print(f"Warning: Gradient for {name} is None. Skipping update.")

            # Clear the gradients after the update
            optimizer.zero_grad()
            
        with torch.no_grad():
            ft_feature_extr.eval()
            ft_diff_net.eval()
            
            z_query = ft_feature_extr(x_query)
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            output_query = ft_diff_net(z_query)
            
            _, y_pred = torch.max(output_query, 1)
            top1_correct = np.sum(y_pred.detach().cpu().numpy() == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this, 0
    

    def test_loop(self, test_loader, optim_based=False, n_ft=0, lr=0, record=None, return_std=False):
        print_freq = 10
        correct =0
        count = 0
        acc_all = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            
            if optim_based:
                correct_this, count_this, loss_value = self.optim_correct(x, n_ft, lr)
            else:
                correct_this, count_this, loss_value = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            if(i % 100==0):
                acc_mean = np.mean(np.asarray(acc_all))
                print('Test | Batch {:d}/{:d} | Loss {:f} | Acc {:f}'.format(i, len(test_loader), loss_value, acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        if(return_std): return acc_mean, acc_std
        else: return acc_mean    
    

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        # Init to dummy values
        x_train = x_support
        y_train = y_support
        target_list = list()
        samples_per_model = int(len(y_train) / self.n_way)
        for way in range(self.n_way):
            target = torch.ones(len(y_train), dtype=torch.float32) * -1.0
            start_index = way * samples_per_model
            stop_index = start_index+samples_per_model
            target[start_index:stop_index] = 1.0
            target_list.append(target.cuda())
        z_train = self.feature_extractor.forward(x_train).detach() #[340, 64]
        if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
        train_list = [z_train]*self.n_way
        for idx, single_model in enumerate(self.model.models):
            single_model.set_train_data(inputs=z_train, targets=target_list[idx], strict=False)


        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            z_query_list = [z_query]*len(y_query)
            predictions = self.likelihood(*self.model(*z_query_list)) #return n_way MultiGaussians
            predictions_list = list()
            for gaussian in predictions:
                predictions_list.append(gaussian.mean) #.cpu().detach().numpy())
            y_pred = torch.stack(predictions_list, 1)
        return y_pred


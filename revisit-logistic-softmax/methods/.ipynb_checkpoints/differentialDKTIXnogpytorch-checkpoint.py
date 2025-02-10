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

device = "cuda:0"
sec_device = "cuda:0"

class L2Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(L2Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)

class differentialDKTIXnogpy(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(differentialDKTIXnogpy, self).__init__(model_func, n_way, n_support)
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        
        self.normalize_output = False
        self.normalize_jac_per_layer = False
        
        print("NORMALIZE OUTPUT :", self.normalize_output)
        print("NORMALIZE JACOBIAN PER LAYER :", self.normalize_jac_per_layer)
        
        self.classifier = nn.Linear(self.feat_dim, n_way)  # Or 2048
        self.classifier.bias.data.fill_(0)
        if self.normalize_output:
            self.feature_extractor = nn.Sequential(self.feature, self.classifier, L2Normalize(p=2, dim=1))
        else:
            self.feature_extractor = nn.Sequential(self.feature, self.classifier)
            
        
        # Add N scaling parameters, initializing them as one
        net_params = {k: v for k, v in self.feature_extractor.named_parameters()}
        self.scaling_params = {k: torch.ones_like(v).to(sec_device) for k, v in net_params.items()}
        
        def construct_target_list(N, n_way):  # N = n_support or n_query
            target_list = list()
            samples_per_model = int(N)
            for c in range(n_way):
                target = torch.ones(N * n_way, dtype=torch.float32) * - 1.0
                start_index = c * samples_per_model
                stop_index = start_index+samples_per_model
                target[start_index:stop_index] = 1.0
                target_list.append(target.cuda())
            return target_list
        
        self.construct_target_list = construct_target_list
        
        self.n_train = 17
        self.self_support = 1
        self.n_way = 5
        self.noise = 0.1
        self.target_list_train = construct_target_list(self.n_train, self.n_way)
        self.target_list_support = construct_target_list(self.n_support, self.n_way)
        
        print(f"Normalization : {self.normalize_output}")
        
        
    def init_summary(self):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M%S", gmtime())
            writer_path = "./log/" + time_string
            self.writer = SummaryWriter(log_dir=writer_path)
        
        
    def set_forward(self, x, is_feature=False):
        pass

    
    def set_forward_loss(self, x):
        pass 
        
        
    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        
        rate_sp_updates = 4 
        n_inner = 0
        inn_lr = 1e-3
        
        optimizer_sp = optimizer[0]
        optimizer_feat_extr = optimizer[1]
        
        for i, (x,_) in enumerate(train_loader):
            # starting_time = time.time()
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).cuda()
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).cuda())
            z_train = x_all
            y_train = y_all

            self.feature_extractor.train()
            
            ## Optimize
            optimizer_feat_extr.zero_grad()
            if epoch % rate_sp_updates == 0:
                optimizer_sp.zero_grad()
            
            # Create a param dict
            params = {k: v for k, v in self.feature_extractor.named_parameters()}
            
            # Pre-loop
            criterion = nn.MSELoss()
            for inn_iter in range(n_inner):
                phi = functional_call(self.feature_extractor, params, (z_train,))
                inner_loss = 0
                for c in range(self.n_way):
                    inner_loss += criterion(phi[:, c], self.target_list_train[c])
                grad = torch.autograd.grad(inner_loss, list(params.values()), create_graph=True, retain_graph=True)
                #grad = [ g.detach()  for g in grad ]  # If perform FO approx
                for k, (name, param) in enumerate(params.items()):
                    params[name] = param - inn_lr * self.scaling_params[name] * grad[k]
            
            # Forward pass
            phi = functional_call(self.feature_extractor, params, (z_train,))
            #for i in range(phi.size(0)):
            #    print(phi[i, :], y_train[i])
            #    print('')
            self.feature_extractor.eval()

            # Need to init jac_func after freesing batch norm statistics
            jac_func = create_jac_func(self.feature_extractor)

            # Compute J(x1)
            jac1 = jac_func(params, z_train)
            if self.normalize_jac_per_layer:
                for param, jac in jac1.items():
                    jac1[param] = F.normalize(jac)

            # Compute the NTK
            if sec_device != device:
                deplace_jac_dict_cuda(jac1, new_device = sec_device)

            ntks = square_ntk(jac1, self.scaling_params)

            loss = 0
            for c in range(self.n_way):
                loss+=NMLL(phi[:, c].to(sec_device), ntks[c], self.target_list_train[c].to(sec_device), noise=self.noise)
            
            loss.backward()
            optimizer_feat_extr.step()
            if epoch % rate_sp_updates == 0:
                optimizer_sp.step()
            
            torch.cuda.empty_cache()
            # Print the current memory allocated on the GPU (in bytes)
            # print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2} MB")

            # Print the maximum memory allocated on the GPU (in bytes)
            # print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB")
                
            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss, self.iteration)
            # print("scaling params")
            # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
            # print("feature_extractor params")
            # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.feature_extractor.named_parameters()})
            

            if i % print_freq==0:
                    
                # if(self.writer is not None): self.writer.add_histogram('phi_support', phi_support, self.iteration)
                mean_vec_avg = torch.mean(self.feature_extractor(z_train).detach(), dim=0)
                mean_vec_avg_str = ", ".join("{:.6f}".format(avg) for avg in mean_vec_avg)
                # print('Epoch [{:d}] [{:d}/{:d}] | Mean functions {} | Noise {:f} | Loss {:f} | Query acc {:f}'.format(epoch, i, len(train_loader), mean_vec_avg_str, self.noise, loss.item(), accuracy_query))
                
                print('Epoch [{:d}] [{:d}/{:d}] | Mean functions {} | Noise {:f} | Loss {:f}'.format(epoch, i, len(train_loader), mean_vec_avg_str, self.noise, loss.item()))
                # print("Scaling Params : ")
                # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
            
            
            # print(f"{i} iteration : {(time.time()-starting_time)}s")}
    
    @torch.no_grad()
    def correct(self, x, sc_no_opt, N=0, laplace=False):
        ##Dividing input x in query and support set
        # print(f"x shape {x.shape}")
        torch.cuda.empty_cache()
        
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        x_train = x_support
        y_train = y_support
        
        z_support = x_support
        z_query = x_query
        
        # Comment these lines for no specialization
        #_, transformations = self.fiveoutputs_3rd_spe(z_batch = z_train)
        #sorted_z_train = z_train.clone()
        #for (rd_class, rd_elemt) in transformations.items():
        #    z_train[self.n_support*rd_class:self.n_support*(rd_class+1)] = sorted_z_train[self.n_support*rd_elemt:self.n_support*(rd_elemt+1)]

        # Create a param dict
        params = {k: v for k, v in self.feature_extractor.named_parameters()}
        # Forward pass
        phi_support = functional_call(self.feature_extractor, params, (z_support,))
        phi_query = functional_call(self.feature_extractor, params, (z_query,))
        # Compute Jacobians
        jac_func = create_jac_func(self.feature_extractor)
        jac1 = jac_func(params, z_support)
        jac2 = jac_func(params, z_query)
        
        if self.normalize_jac_per_layer:
            for param, jac in jac1.items():
                jac1[param] = F.normalize(jac)
            for param, jac in jac2.items():
                jac2[param] = F.normalize(jac)
        
        del params
        
        # Compute the NTKs
        ntks_ss = square_ntk(jac1, sc_no_opt)
        ntks_qs = support_query_ntk(jac1, jac2, sc_no_opt)
        ntks_qq = square_ntk(jac2, sc_no_opt)
            
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

    
    def optim_correct(self, x, sc_no_opt, n_ft, lr, temp):
        criterion_type = "cross-entropy"  # cross-entropy or LR
        
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        x_train = x_support
        y_train = y_support
        
        if criterion_type=="LR":
            Y_support = self.construct_target_list(self.n_support, self.n_way)
            Y_query = self.construct_target_list(self.n_query, self.n_way)
            criterion = nn.MSELoss()
            
        else:
            criterion = nn.CrossEntropyLoss()
            
        
        ft_feature_extractor = copy.deepcopy(self.feature_extractor).cuda()
        ft_feature_extractor.train()
        
        optimizer = optim.Adam([{'params':ft_feature_extractor.parameters(), 'lr':0}])
        optimizer.zero_grad()
        avg_loss=0.0
        # print("scaling params")
        # print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
        # print({k: torch.min(v) for k, v in self.scaling_params.items()})
        for i_ft in range(n_ft):
            # Forward pass
            train_logit = ft_feature_extractor(x_train)
            if criterion_type=="LR":
                inner_loss = 0
                for c in range(self.n_way):
                    inner_loss += criterion(train_logit[:, c], Y_support[c])
            else:
                inner_loss = criterion(train_logit/temp, y_train)

            # Backward pass
            inner_loss.backward()

            # Manually update each parameter using the custom learning rates
            with torch.no_grad():
                for name, param in ft_feature_extractor.named_parameters():
                    if param.grad is not None:
                        param_update = lr * sc_no_opt[name] * param.grad
                        param -= param_update
                    else:
                        print(f"Warning: Gradient for {name} is None. Skipping update.")

            # Clear the gradients after the update
            optimizer.zero_grad()
            
        with torch.no_grad():
            # ft_feature_extractor.eval()
            
            output_query = ft_feature_extractor(x_query)
            
            _, y_pred = torch.max(output_query, 1)
            top1_correct = np.sum(y_pred.detach().cpu().numpy() == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this, 0
    

    def test_loop(self, test_loader, optim_based=False, n_ft=0, lr=0, temp=.3, record=None, return_std=False):
        print_freq = 10
        correct =0
        count = 0
        acc_all = []
        iter_num = len(test_loader)
        
        # When training, moving the scaling params from one GPU to another makes a copy of the original scaling params, without the optimizer retaining it's location in memory : would need to redefine the optimizer
        sc_no_opt = copy.deepcopy(self.scaling_params)
        deplace_jac_dict_cuda(sc_no_opt, new_device = "cuda:0")
        
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            
            if optim_based:
                correct_this, count_this, loss_value = self.optim_correct(x, sc_no_opt, n_ft, lr, temp)
            else:
                correct_this, count_this, loss_value = self.correct(x, sc_no_opt)
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
   


def create_jac_func(net):
    """
    Computes the functional call of a single input, to be differentiated in parallel using vmap
    """
    def fnet_single(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

    jac_func = vmap(jacrev(fnet_single), (None, 0))
    return jac_func


def deplace_jac_dict_cuda(jac, new_device):
    for k, j in jac.items():
        jac[k] = j.to(new_device)
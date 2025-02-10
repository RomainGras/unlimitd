## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, jvp, jacrev
from methods.meta_template import MetaTemplate

## Our packages
import gpytorch
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

class differentialDKTIX(MetaTemplate):
    def __init__(self, model_func, diff_net, n_way, n_support):
        super(differentialDKTIX, self).__init__(model_func, n_way, n_support)
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.diff_net = diff_net()
        dummy_z = torch.randn(85, 33433)
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
        
        print(f"Normalization : {self.normalize}")

    def init_summary(self):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M%S", gmtime())
            writer_path = "./log/" + time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def get_model_likelihood_mll(self, train_x_list=None, train_y_list=None):
        if(train_x_list is None): train_x_list=[torch.ones(100, 64).cuda()]*self.n_way
        if(train_y_list is None): train_y_list=[torch.ones(100).cuda()]*self.n_way
        
        # Add N scaling parameters, initializing them as one
        self.params = {k: v for k, v in self.diff_net.named_parameters()}
        self.scaling_params = dict()
        for k, v in self.diff_net.named_parameters():
            self.scaling_params[k] = torch.ones_like(v).cuda()
        
        def fnet_single(params, x):
            return functional_call(self.diff_net, self.params, (x.unsqueeze(0),)).squeeze(0)
        self.fnet_single = fnet_single
        
        model_list = list()
        likelihood_list = list()
        for c, (train_x, train_y) in enumerate(zip(train_x_list, train_y_list)):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, net=self.diff_net, kernel=kernel_type, scaling_params=self.scaling_params, c=c)
            model_list.append(model)
            likelihood_list.append(model.likelihood)
        self.model = gpytorch.models.IndependentModelList(*model_list).cuda()
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list).cuda()
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model).cuda()
        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def _reset_likelihood(self, debug=False):
        for param in self.likelihood.parameters():
           param.data.normal_(0.0, 0.01)

    def _print_weights(self):
        for k, v in self.feature_extractor.state_dict().items():
            print("Layer {}".format(k))
            print(v)

    def _reset_variational(self):
        mean_init = torch.zeros(128) #num_inducing_points
        covar_init = torch.eye(128, 128) #num_inducing_points
        mean_init = mean_init.repeat(64, 1) #batch_shape
        covar_init = covar_init.repeat(64, 1, 1) #batch_shape
        for idx, param in enumerate(self.gp_layer.variational_parameters()):
            if(idx==0): param.data.copy_(mean_init) #"variational_mean"
            elif(idx==1): param.data.copy_(covar_init) #"chol_variational_covar"
            else: raise ValueError('[ERROR] DKT the variational_parameters at index>1 should not exist!')

    def _reset_parameters(self):
        if(self.leghtscale_list is None):
            self.leghtscale_list = list()
            self.noise_list = list()
            self.outputscale_list = list()
            for idx, single_model in enumerate(self.model.models):
                self.leghtscale_list.append(single_model.covar_module.base_kernel.lengthscale.clone().detach())
                self.noise_list.append(single_model.likelihood.noise.clone().detach())
                self.outputscale_list.append(single_model.covar_module.outputscale.clone().detach())
        else:
            for idx, single_model in enumerate(self.model.models):
                single_model.covar_module.base_kernel.lengthscale=self.leghtscale_list[idx].clone().detach()#.requires_grad_(True)
                single_model.likelihood.noise=self.noise_list[idx].clone().detach()
                single_model.covar_module.outputscale=self.outputscale_list[idx].clone().detach()

                
    @torch.no_grad()
    def fiveoutputs_3rd_spe(self, z_batch):
        sorted_z_batch = torch.empty(z_batch.shape).cuda()
        n_shot = z_batch.size(0)//self.n_way

        # specialization matrix
        spe = self.diff_net(z_batch)
        
        reshape_spe = spe.view(self.n_way, n_shot, self.n_way)
        # Compute the mean along the middle dimension
        spe = reshape_spe.mean(dim=1)
        
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
            indices_1 = torch.tensor([self.n_way * i + rd_elemt for i in range(self.n_way)]).cuda()
            indices_2 = torch.tensor([i + rd_class * self.n_way for i in range(self.n_way)]).cuda()

            # Combine indices from both calculations, ensuring uniqueness if necessary
            all_indices = torch.cat((indices_1, indices_2)).unique()
            flattened_spe[all_indices] = float('-inf')
            # print(f"Input number {rd_elemt[0]} will have class number {rd_class[0]}")
            
            transformations[rd_class] = rd_elemt
            sorted_z_batch[n_shot*rd_class:n_shot*(rd_class+1)] = z_batch[n_shot*rd_elemt:n_shot*(rd_elemt+1)]
        
        # keys = torch.arange(self.n_way).tolist()  # For random transformations
        # values = torch.randperm(self.n_way).tolist() 
        return sorted_z_batch.cuda(), transformations # z_batch, {k: v for k, v in zip(keys, values)} 
    
    
    def train_loop(self, epoch, train_loader, optimizer, print_freq=33):
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

            target_list = list()
            samples_per_model = int(len(y_train) / self.n_way) #25 / 5 = 5
            for way in range(self.n_way):
                target = torch.ones(len(y_train), dtype=torch.float32) * -1.0
                start_index = way * samples_per_model
                stop_index = start_index+samples_per_model
                target[start_index:stop_index] = 1.0
                target_list.append(target.cuda())

            self.model.train()
            self.likelihood.train()
            self.feature_extractor.train()
            z_train = self.feature_extractor.forward(x_train)
            
            # Compute jacobian :
            z_train = vmap(jacrev(self.fnet_single), (None, 0))(self.params, z_train)
            z_train = [(self.scaling_params[k] * j).flatten(2) for k, j in z_train.items()] 
            
            z_train = torch.cat(z_train, dim=2)
            # print(z_train.shape)
            
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            
            # Comment this for non specialized cdkt :
            #_, transformations = self.fiveoutputs_3rd_spe(z_batch = z_train)
            #sorted_z_train = z_train.clone()
            #sorted_x = x.clone()
            #for (rd_class, rd_elemt) in transformations.items():
            #    z_train[(self.n_support + self.n_query)*rd_class:(self.n_support + self.n_query)*(rd_class+1)] = sorted_z_train[(self.n_support + self.n_query)*rd_elemt:(self.n_support + self.n_query)*(rd_elemt+1)]
            #    x[rd_class:rd_class+1] = sorted_x[rd_elemt:rd_elemt+1]


            lenghtscale = 0.0
            noise = 0.0
            outputscale = 0.0
            for idx, single_model in enumerate(self.model.models):
                #print(f"target_list[idx] : {target_list[idx].shape}") # [85]
                #print(f"self.diff_net(z_train) : {self.diff_net(z_train).shape}")  # [85,5]
                single_model.set_train_data(inputs=z_train[:, idx, :], targets=target_list[idx]-self.diff_net(self.feature_extractor(x_train))[:, idx], strict=False)
                if hasattr(single_model.covar_module, 'lengthscale') and (single_model.covar_module.lengthscale is not None): #Originally if(single_model.covar_module.base_kernel.lengthscale is not None):
                    lenghtscale+=single_model.covar_module.lengthscale.mean().cpu().detach().numpy().squeeze()
                noise+=single_model.likelihood.noise.cpu().detach().numpy().squeeze()
                if hasattr(single_model.covar_module, 'outputscale') and (single_model.covar_module.outputscale is not None): #Originally if(single_model.covar_module.outputscale is not None):
                    outputscale+=single_model.covar_module.outputscale.cpu().detach().numpy().squeeze()
            if hasattr(single_model.covar_module, 'lengthscale') and (single_model.covar_module.lengthscale is not None):# Originally if(single_model.covar_module.base_kernel.lengthscale is not None):
                lenghtscale /= float(len(self.model.models))
            noise /= float(len(self.model.models))
            if hasattr(single_model.covar_module, 'outputscale') and (single_model.covar_module.outputscale is not None): # Originally if(single_model.covar_module.outputscale is not None):
                outputscale /= float(len(self.model.models))

            ## Optimize
            optimizer.zero_grad()
            #self.model.eval()   #In case of eventual batch normalization layers 
            output = self.model(*self.model.train_inputs)
            #self.model.train()
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()

            print("scaling params")
            print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.scaling_params.items()})
            print("diff_net params")
            print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in self.diff_net.named_parameters()})
            
            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss, self.iteration)

            if i % print_freq==0:
                
                #Eval on the query (validation set)
                with torch.no_grad():
                    x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
                    y_support = np.repeat(range(self.n_way), self.n_support)
                    x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
                    y_query = np.repeat(range(self.n_way), self.n_query)
                    
                    self.model.eval()
                    self.likelihood.eval()
                    self.feature_extractor.eval()
                    z_support = self.feature_extractor.forward(x_support).detach()
                   
                    # Compute jacobian :
                    z_support = vmap(jacrev(self.fnet_single), (None, 0))(self.params, z_support)
                    z_support = [(self.scaling_params[k] * j).flatten(2) for k, j in z_support.items()] 

                    z_support = torch.cat(z_support, dim=2)
                    if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
                    z_support_list = z_support.permute(1,0,2) #originally [z_support]*len(y_support)
                    z_support_list = torch.unbind(z_support_list, dim=0)
                    
                    #print(f"self.model : {self.model}")
                    #print(f"z_support shape : {z_support.shape}")
                    #print(f"y_support length : {len(y_support)}")
                    #print(f"z_support_list length : {len(z_support_list)}")
                    #print(f"Length of z_support_list: {len(z_support_list)}")  # Should be 5 to match the number of models
                    #for i, tensor in enumerate(z_support_list):
                    #    print(f"Shape of z_support_list[{i}]: {tensor.shape}")
                    #print(f"self.model(*z_support_list) : {len(self.model(*z_support_list))}")

                    predictions = self.likelihood(*self.model(*z_support_list)) #return 20 MultiGaussian Distributions
                    predictions_list = list()
                    for c, gaussian in enumerate(predictions):
                        predictions_list.append(torch.sigmoid(gaussian.mean+self.diff_net(self.feature_extractor(x_support))[:,c]).cpu().detach().numpy())
                    y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
                    accuracy_support = (np.sum(y_pred==y_support) / float(len(y_support))) * 100.0
                    if(self.writer is not None): self.writer.add_scalar('GP_support_accuracy', accuracy_support, self.iteration)
                    z_query = self.feature_extractor.forward(x_query).detach()
                    z_query = vmap(jacrev(self.fnet_single), (None, 0))(self.params, z_query)
                    z_query = [(self.scaling_params[k] * j).flatten(2) for k, j in z_query.items()] 

                    z_query = torch.cat(z_query, dim=2)
                    if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                    z_query_list = z_query.permute(1, 0, 2) # Originally [z_query]*len(y_query)
                    z_query_list = torch.unbind(z_query_list, dim=0)
                    predictions = self.likelihood(*self.model(*z_query_list)) #return 20 MultiGaussian Distributions
                    predictions_list = list()
                    for c, gaussian in enumerate(predictions):
                        predictions_list.append(torch.sigmoid(gaussian.mean+self.diff_net(self.feature_extractor(x_query))[:,c]).cpu().detach().numpy())
                    y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
                    accuracy_query = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
                    if(self.writer is not None): self.writer.add_scalar('GP_query_accuracy', accuracy_query, self.iteration)
                    
                if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                mean_vec_avg = torch.mean(self.diff_net(self.feature_extractor(x_train)).detach(), dim=0)
                mean_vec_avg_str = ", ".join("{:.6f}".format(avg) for avg in mean_vec_avg)
                print('Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Mean functions {} | Noise {:f} | Loss {:f} | Supp. {:f} | Query {:f}'.format(epoch, i, len(train_loader), outputscale, mean_vec_avg_str, noise, loss.item(), accuracy_support, accuracy_query))
            
            # print(f"{i} iteration : {(time.time()-starting_time)}s")

    def correct(self, x, N=0, laplace=False):
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        ## Laplace approximation of the posterior
        if(laplace):
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF, Matern
            from sklearn.gaussian_process.kernels import ConstantKernel as C
            kernel = 1.0 * RBF(length_scale=0.1 , length_scale_bounds=(0.1, 10.0))
            gp = GaussianProcessClassifier(kernel=kernel, optimizer=None)
            z_support = self.feature_extractor.forward(x_support).detach()
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            gp.fit(z_support.cpu().detach().numpy(), y_support.cpu().detach().numpy())
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            y_pred = gp.predict(z_query.cpu().detach().numpy())
            accuracy = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
            top1_correct = np.sum(y_pred==y_query)
            count_this = len(y_query)
            return float(top1_correct), count_this, 0.0

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

        z_train = self.feature_extractor.forward(x_train)

        # Compute jacobian :
        z_train = vmap(jacrev(self.fnet_single), (None, 0))(self.params, z_train)
        z_train = [(self.scaling_params[k] * j).flatten(2) for k, j in z_train.items()] 

        z_train = torch.cat(z_train, dim=2)
        # print(z_train.shape)

        if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
        
        # Comment these lines for no specialization
        #_, transformations = self.fiveoutputs_3rd_spe(z_batch = z_train)
        #sorted_z_train = z_train.clone()
        #for (rd_class, rd_elemt) in transformations.items():
        #    z_train[self.n_support*rd_class:self.n_support*(rd_class+1)] = sorted_z_train[self.n_support*rd_elemt:self.n_support*(rd_elemt+1)]
            
        z_train_list = torch.unbind(z_train.permute(1,0,2), dim=0)
        for idx, single_model in enumerate(self.model.models):
            single_model.set_train_data(inputs=z_train_list[idx], targets=target_list[idx]-self.diff_net(self.feature_extractor(x_train))[:,idx], strict=False)

        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=1e-3)

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.eval()

        avg_loss=0.0
        for i in range(0, N):
            ## Optimize
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            z_query = self.feature_extractor.forward(x_query)
            
            # Compute jacobian :
            z_query = vmap(jacrev(self.fnet_single), (None, 0))(self.params, z_query)
            z_query = [(self.scaling_params[k] * j).flatten(2) for k, j in z_query.items()] 
            
            z_query = torch.cat(z_query, dim=2)
            # print(z_train.shape)
            
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            
            # Comment these lines for no specialization
            #sorted_z_query = z_query.clone()
            #for (rd_class, rd_elemt) in transformations.items():
            #    z_query[self.n_query*rd_class:self.n_query*(rd_class+1)] = sorted_z_query[self.n_query*rd_elemt:self.n_query*(rd_elemt+1)]
                
            z_query_list = torch.unbind(z_query.permute(1,0,2), dim=0) # Originally [z_query]*len(y_query)
            # print(z_query_list[0].reshape(z_query_list[0].size(0), -1).shape)
            # print(self.model(*z_query_list))
            
            predictions = self.likelihood(*self.model(*z_query_list)) #return n_way MultiGaussians
            predictions_list = list()
            # Getting the sigmoid of each binary classifier, and the max is the prediction
            for c, gaussian in enumerate(predictions):
                predictions_list.append(torch.sigmoid(gaussian.mean+self.diff_net(self.feature(x_query))[:,c]).cpu().detach().numpy())
            y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this, avg_loss/float(N+1e-10)

    
    def optim_correct(self, x, n_ft, lr):
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
        for i_ft in range(n_ft):
            # Forward pass
            z_train = ft_feature_extr(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            train_logit = ft_diff_net(z_train)
            inner_loss = criterion(train_logit, y_train)

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
            
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
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

class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, net, scaling_params, c=0, kernel='linear'):
        #Set the likelihood noise and enable/disable learning
        likelihood.noise_covar.raw_noise.requires_grad = False
        likelihood.noise_covar.noise = torch.tensor(0.1)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ## Linear kernel
        if(kernel=='linear'):
            self.covar_module = gpytorch.kernels.LinearKernel()
        elif(kernel=='cossim' or kernel=='bncossim'):
        ## Cosine distance and BatchNorm Cosine distance
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


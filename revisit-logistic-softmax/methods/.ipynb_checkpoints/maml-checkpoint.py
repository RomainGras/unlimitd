# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import time
import gpytorch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.func import functional_call, vmap, vjp, jvp, jacrev

class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False):
        super(MAML, self).__init__( model_func,  n_way, n_support, change_way = False)
        
        print(self.feature.state_dict().keys())
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)  # Or 2048
        self.classifier.bias.data.fill_(0)
        
        self.n_task     = 4
        self.task_update_num = 1  # 1
        self.train_lr = 0.01  # 0.01
        print(f"Inner lr : {self.train_lr}")
        self.approx = approx #first order approx.        

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()
        
        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, y_b_i)

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
            
            # Specialization :
            # x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
            # z_batch = self.feature.forward(x_support)
            # _, transformations =  self.fiveoutputs_3rd_spe(z_batch)
            # sorted_x = x.clone()
            # for (rd_class, rd_elemt) in transformations.items():
                # x[rd_class:rd_class+1] = sorted_x[rd_elemt:rd_elemt+1]
                
                
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
      
    
    def test_loop(self, test_loader, return_std = False, jac_test = False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        start_time = time.time()
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
           
            # Specialization :
            # x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
            # z_batch = self.feature.forward(x_support)
            # _, transformations =  self.fiveoutputs_3rd_spe(z_batch)
            #transformations = self.random_shuffle()
            # sorted_x = x.clone()
            # for (rd_class, rd_elemt) in transformations.items():
                # x[rd_class:rd_class+1] = sorted_x[rd_elemt:rd_elemt+1]
            
            if jac_test:
                correct_this, count_this = self.jac_correct(x)
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
  
    
    def jac_correct(self,x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        # with torch.no_grad():
        self.feature.eval()
        self.classifier.eval()

        z_support = self.feature.forward(x_a_i).detach()
        support_mean_vec = self.classifier(z_support).T  #Size 5,85
        support_target_list = self.construct_target_list(support_mean_vec)
        z_query = self.feature.forward(x_b_i).detach()
        query_mean_vec = self.classifier(z_query).T
        
        # print(support_target_list[0])

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_list = []
        cov_matrix_list = []
        for c in range(self.n_way):
            # print(f"x_a_i.shape : {x_a_i.shape}")
            # print(f"x_b_i.shape : {x_b_i.shape}")
            # print(f"support_mean_vec[c, :].shape : {support_mean_vec[c, :].shape}")
            # print(f"support_target_list[c].shape : {support_target_list[c].shape}")
            # print(f"support_mean_vec[c, :].shape : {support_mean_vec[c, :].shape}")
            x_a_i = x_a_i.reshape(x_a_i.size(0), -1)
            x_b_i = x_b_i.reshape(x_b_i.size(0), -1)
            #gp = ExactGPModel(x_a_i, support_mean_vec[c, :].reshape(-1), likelihood, nn.Sequential(self.feature, self.classifier), c).to("cuda")
            #gp.train()
            #gp.set_train_data(inputs=x_a_i, targets= (support_target_list[c] - support_mean_vec[c, :].reshape(-1)), strict=False)  
            #gp.eval()
            #posterior = gp(x_b_i)
            # Shift the mean by the shift vector
            #shifted_mean = posterior.mean + query_mean_vec[c, :].reshape(-1)
            #max_posterior = torch.norm(posterior.covariance_matrix, p=float('inf'))
            
            #print(max_posterior)
            #print(torch.linalg.det(posterior.covariance_matrix))
            #print(posterior.covariance_matrix)
            #print(posterior.mean)
            #print(shifted_mean)
            # Create a new multivariate normal distribution with the shifted mean and the same covariance matrix
            # shifted_mvn = MultivariateNormal(shifted_mean, posterior.covariance_matrix )#+ 1e1*torch.eye(posterior.covariance_matrix.size(0), device = "cuda"))
            posterior_mean, posterior_cov = self.gaussian_posterior(x_a_i, x_b_i, support_target_list[c], support_mean_vec[c, :].reshape(-1), query_mean_vec[c, :].reshape(-1), c)
            # print(f"posterior mean shape {posterior_mean.shape}")
            mean_list.append(posterior_mean)
            # We don't want communication between all the query inputs so we only take the diagonal of the covariance matrix
            diagonal_post_cov = torch.diag(posterior_cov)
            cov_matrix_list.append(torch.diag(diagonal_post_cov))
            # q_posterior_list.append(shifted_mvn)
        
        y_query = np.repeat(range( self.n_way ), self.n_query )
        y_pred = self.montecarlo(mean_list, cov_matrix_list, times=10000)     
        y_pred = y_pred.cpu().numpy() 
        top1_correct = np.sum(y_pred == y_query)
        count_this = len(y_query)
        print(float(top1_correct)/ count_this *100)
        return float(top1_correct), count_this

    def construct_target_list(self, support_output):
        """Construct a target for regression type adaptation"""
        target_list = list()
        samples_per_model = int(support_output.size(1) / self.n_way) #25 / 5 = 5
        for way in range(self.n_way):
            target = torch.ones(support_output.size(1), dtype=torch.float32, device="cuda") * -1.0
            start_index = way * samples_per_model
            stop_index = start_index+samples_per_model
            target[start_index:stop_index] = 1.0
            for i in range(self.n_way):
                #mean = torch.mean(support_output[way,i * samples_per_model:(i+1) * samples_per_model])
                #std = torch.std(support_output[way,i * samples_per_model:(i+1) * samples_per_model])
                target[i * samples_per_model:(i+1) * samples_per_model] = 1 * target[i * samples_per_model:(i+1) * samples_per_model]  #TODO
            target_list.append(target.cuda())
        return target_list

    
    def gaussian_posterior(self, x_a_i, x_b_i, target, support_mean_vec, query_mean_vec, c):
        """
        Computes the posterior of the gaussian process corresponding to one class
        """
        NTKernel = vmap_ntk(nn.Sequential(self.feature, self.classifier), c)
        
        K_a_a = NTKernel(x_a_i, x_a_i).evaluate()
        K_b_b = NTKernel(x_b_i, x_b_i).evaluate()
        K_b_a = NTKernel(x_b_i, x_a_i).evaluate()
        sigma_I = 1e-2 * torch.eye(K_a_a.size(0), device="cuda")

        # Compute the Cholesky decomposition of (K(X, X) + sigma_n^2 * I)
        L = torch.linalg.cholesky(K_a_a + sigma_I)

        # Solve for alpha: L * L.T * alpha = y
        # Step 1: Solve L * z = y
        z = torch.linalg.solve(L, target - support_mean_vec)

        # Step 2: Solve L.T * alpha = z
        alpha = torch.linalg.solve(L.T, z)

        # Compute the mean prediction f_star
        f_star = torch.matmul(K_b_a, alpha) + query_mean_vec

        # Solve for v: L * v = K(X, X*)
        v = torch.linalg.solve(L, K_b_a.T)

        # Compute the covariance prediction
        cov_f_star = K_b_b - torch.matmul(v.T, v)
        
        # print(f"posterior mean : {v}")
        # print(f"posterior covariance : {cov_f_star}")
        # print(f"posterior covariance det : {torch.linalg.det(cov_f_star/torch.norm(cov_f_star, p=float('inf')))}")
        
        return f_star, cov_f_star
        
        
        
        
    def sample_gaussian(self, mu, Sigma, num_samples=1):
        """
        Sample from a multivariate Gaussian distribution.

        Parameters:
        mu (torch.Tensor): Mean vector of the Gaussian distribution (D,).
        Sigma (torch.Tensor): Covariance matrix of the Gaussian distribution (D, D).
        num_samples (int): Number of samples to generate.

        Returns:
        torch.Tensor: Samples from the Gaussian distribution (num_samples, D).
        """
        # Check dimensions
        D = mu.size(0)

        # Generate samples from a standard normal distribution
        z = torch.randn((num_samples[0], D), device = "cuda")
        # print(mu.shape)
        # print(z.shape)
        # print(num_samples)

        # Perform Cholesky decomposition of the covariance matrix
        # print(f"Sigma : {Sigma}")
        L = torch.linalg.cholesky(Sigma + 1e-2*torch.eye(Sigma.size(0), device = "cuda"))

        # Transform the standard normal samples to match the desired mean and covariance
        samples = mu + z @ L.t()

        return samples       
        
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

    
class vmap_ntk(gpytorch.kernels.Kernel):
    def __init__(self, net, c, **kwargs):
        super(vmap_ntk, self).__init__(**kwargs)
        self.net = net
        self.c = c
        
    def forward(self, x1, x2, diag=False, **params):
        """Calculates the Components of the NTK and places into a dictionary whose keys are the named parameters of the self.net. 

        While torch.vmap function is still in development, there appears to me to be an issue with how
        greedy torch.vmap allocates reserved memory. Batching the calls to vmap seems to help. Just like
        with training a self.net: you can go very fast with high batch size, but it requires an absurd amount 
        of memory. Unlike training a self.net, there is no regularization, so you should make batch size as high
        as possible

        We suggest clearing the cache after running this operation.

            parameters:
                self.net: a torch.nn.Module object that terminates to a single neuron output
                xloader: a torch.data.utils.DataLoader object whose first value is the input data to the self.net
                device: a string, either 'cpu' or 'cuda' where the self.net will be run on

            returns:
                NTKs: a dictionary whose keys are the names parameters and values are said parameters additive contribution to the NTK
        """
        x1 = x1.reshape(x1.size(0), 3, 84, 84)
        x2 = x2.reshape(x2.size(0), 3, 84, 84)
        # print(x1.shape)
        # print(x2.shape)
        device='cuda'
        NTK = torch.zeros(x1.size(0), x2.size(0), device = device)
        params_that_need_grad = []
        for param in self.net.parameters():
            if param.requires_grad:
                params_that_need_grad.append(param.requires_grad)
        
        # Compute NTK one layer at a time to not overload GPU memory
        for i,z in enumerate(self.net.named_parameters()):
            if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
                continue
            name, param = z
            # print(name)
            # print(param.shape)
            
            J_layer_1 = self.compute_J_layer_vector(x1, param, device)
            J_layer_2 = self.compute_J_layer_vector(x2, param, device) if x1 is not x2 else J_layer_1
            # print(J_layer_1.device)
            # print(J_layer_2.device)
             
            NTK += J_layer_1 @ J_layer_2.T
            if device=='cuda':
                torch.cuda.empty_cache()
            # print(NTK - NTK.T)
            # time.sleep(1)
            
        if diag:
            return NTK.diag()
        return NTK
    
    def compute_J_layer_vector(self, x1, param, device):
        J_layer_1=[]
        basis_vectors_1 = torch.eye(x1.size(0),device=device,dtype=torch.bool)
        # print(basis_vectors_1.shape)
        y1 = self.net(x1)[:,self.c]
        # print(f"y1 shape : {y1.shape}")
        for i in range(x1.size(0)):
            grad = torch.autograd.grad(y1[i],param, retain_graph=True, create_graph=False)[0].reshape(-1)
            # print(grad.shape)
            J_layer_1.append(grad.detach().to(device))
        # def compute_grad(y_i, param):
        #     grad = torch.autograd.grad(y_i, param, retain_graph=True, create_graph=False)[0]
        #     return grad
        # compute_grad_vmap = torch.vmap(compute_grad, in_dims=(0, None), out_dims=0)
        # J_layer_1 = compute_grad_vmap(y1, param)
        # J_layer_1 = J_layer_1.detach().to(device)
        return torch.stack(J_layer_1)


class NTKernel(gpytorch.kernels.Kernel):
    def __init__(self, net, c, normalize, **kwargs):  # i is the output index. Each index or the output has its own kernel that is sigma * grad(NN_i(x))^T @ grad(NN_i(x))
        super(NTKernel, self).__init__(**kwargs)
        self.net = net
        self.c = c
        self.normalize = normalize

    def forward(self, x1, x2, diag=False, **params):
        x1 = x1.reshape(x1.size(0), 3, 84, 84)
        x2 = x2.reshape(x2.size(0), 3, 84, 84)
        jac1T = self.compute_jacobian(x1).T
        jac2T = self.compute_jacobian(x2).T if x1 is not x2 else jac1T
            
        if self.normalize :
            jac1T_norm = jac1T.norm(dim=0, keepdim=True)
            jac1T = jac1T/jac1T_norm
            jac2T_norm = jac2T.norm(dim=0, keepdim=True)
            jac2T = jac2T/jac2T_norm
        
        result = jac1T.T@jac2T
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
            # Make sure output has the right dimensions
            return functional_call(self.net, params, (x.unsqueeze(0),)).squeeze(0)[self.c]
        
        jac = vmap(jacrev(fnet_single), (None, 0))(params, inputs)
        jac_values = jac.values()

        reshaped_tensors = []
        for j in jac_values:
            if len(j.shape) >= 3:  # For layers with weights
                # Flatten parameters dimensions and then reshape
                flattened = j.flatten(start_dim=1)  # Flattens to [batch, params]
                reshaped = flattened.T  # Transpose to align dimensions as [params, batch]
                reshaped_tensors.append(reshaped)
            elif len(j.shape) == 2:  # For biases or single parameter components
                reshaped_tensors.append(j.T)  # Simply transpose

        # Concatenate all the reshaped tensors into one large matrix
        return torch.cat(reshaped_tensors, dim=0).T

    def compute_jacobian_autodiff(self, inputs):
        """
        Return the jacobian of a batch of inputs, using autodifferentiation
        Useful for when dealing with models using batch normalization or other kind of running statistics
        """
        inputs.requires_grad_(True)
        self.net.parameters()
        outputs = self.net(inputs).requires_grad_(True)
        N = sum(p.numel() for p in self.net.parameters())
        jac = torch.empty(outputs.size(0), N).to("cuda:0")
        for j in range(outputs.size(0)):
            # print(j)
            grad_y1 = torch.autograd.grad(outputs[j, self.c], self.net.parameters()) # We need to create and retain every single graph for the gradient to be able to run through during backprop
            # print_memory_usage()
            flattened_tensors = [t.flatten() for t in grad_y1]
            jac[j] = torch.cat(flattened_tensors)
            # print_memory_usage()
            # if device == "cuda":
            #     torch.cuda.empty_cache()
            #     print_memory_usage()
        return jac

    def compute_jacobian_vmap_autodiff(self, inputs):
        """
        Return the jacobian of a batch of inputs, thanks to the vmap functionality
        """
        device = "cuda"
        if device=='cuda':
            torch.cuda.empty_cache()
        params_that_need_grad = []
        for param in self.net.parameters():
            if param.requires_grad:
                params_that_need_grad.append(param.requires_grad)

        inputs = inputs.to(device, non_blocking=True)
        inputs.requires_grad_(True)
        outputs = self.net(inputs)
        basis_vectors = torch.eye(len(inputs),device=device,dtype=torch.bool)
        J_layer = []
        for i,z in enumerate(self.net.named_parameters()):
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
            if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
                continue
            name, param = z
            outputsc = outputs[:, self.c]   
            #Seems like for retain_graph=False, you might need to do multiple forward passes.

            def torch_row_Jacobian(v): #y would have to be a single piece of the batch
                return torch.autograd.grad(outputsc,param,v, retain_graph=False)[0].reshape(-1)
            J_layer.append(vmap(torch_row_Jacobian)(basis_vectors).detach())

            del outputsc
            if device=='cuda':
                torch.cuda.empty_cache()
            #print(name)
        #for layer in J_layer:
        #    print(layer.shape)
        del params_that_need_grad
        del inputs
        del outputs
        del basis_vectors
        if device=='cuda':
            torch.cuda.empty_cache()
        J_layer = torch.cat(J_layer, axis=1)
        return J_layer

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, net, c):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            vmap_ntk(net, c)
        )
        #self.covar_module = CosSimNTKernel(net)
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        #self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=40)
        #self.feature_extractor = feature_extractor
        
    def forward(self, x):
        #z = self.feature_extractor(x)
        #z_normalized = z - z.min(0)[0]
        #z_normalized = 2 * (z_normalized / z_normalized.max(0)[0]) - 1
        #x_normalized = x - x.min(0)[0]
        #x_normalized = 2 * (x_normalized / x_normalized.max(0)[0]) - 1
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
 
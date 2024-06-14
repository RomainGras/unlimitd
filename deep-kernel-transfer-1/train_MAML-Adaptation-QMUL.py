# Code adapted from: 
# https://github.com/vmikulik/maml-pytorch
# https://github.com/cbfinn/maml

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from data.qmul_loader import get_batch, train_people, test_people
import gpytorch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.func import functional_call, vmap, vjp, jvp, jacrev

torch.manual_seed(42)


class QMULMAMLBigModel(nn.Module):
    def __init__(self):
        super(QMULMAMLBigModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, 3,stride=2,dilation=2)
        self.conv2 = nn.Conv2d(36,36, 3,stride=2,dilation=2)
        self.conv3 = nn.Conv2d(36,36, 3,stride=2,dilation=2)
        self.l4 = nn.Linear(2916, 40)
        self.l5 = nn.Linear(40,40)
        self.l6 = nn.Linear(40,1)

    def return_clones(self):
        conv1_w = self.conv1.weight.data.clone().detach()
        conv2_w = self.conv2.weight.data.clone().detach()
        conv3_w = self.conv3.weight.data.clone().detach()
        l4_w = self.l4.weight.data.clone().detach()
        l5_w = self.l5.weight.data.clone().detach()
        l6_w = self.l6.weight.data.clone().detach()
        return [conv1_w, conv2_w, conv3_w, l4_w, l5_w, l6_w]

    def assign_clones(self, weights_list):
        self.conv1.weight.data.copy_(weights_list[0])
        self.conv2.weight.data.copy_(weights_list[1])
        self.conv3.weight.data.copy_(weights_list[2])

    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = nn.functional.relu(self.l4(x))
        out = nn.functional.relu(self.l5(out))
        return out
    
    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        x = nn.functional.conv2d(x, weights[0], weights[1], stride=2, dilation=2)
        x = nn.functional.relu(x)
        x = nn.functional.conv2d(x, weights[2], weights[3], stride=2, dilation=2)
        x = nn.functional.relu(x)
        x = nn.functional.conv2d(x, weights[4], weights[5], stride=2, dilation=2)
        x = nn.functional.relu(x)
        
        x = x.view(x.size(0), -1)  # Flatten the output for linear layers
        
        
        x = nn.functional.linear(x, weights[6], weights[7])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[8], weights[9])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[10], weights[11])
        return x

def get_support_query_batches(people, n_support):
    """
    Gives support and query batches for one task by randomly shuffling the images
    """
    inputs, targets = get_batch(people)
    
    #choose a random person
    n = np.random.randint(0, len(test_people)-1)
    support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
    query_ind   = [i for i in range(19) if i not in support_ind]
    x_all = inputs[n].cuda()
    y_all = targets[n].cuda()
        
    x_support = x_all[support_ind,:,:,:].cuda()
    y_support = y_all[support_ind].cuda()
    x_query = x_all[query_ind,:,:,:].cuda()
    y_query = y_all[query_ind].cuda()
    
    return x_support, y_support, x_query, y_query

class MAML():
    def __init__(self, model, inner_lr, meta_lr, n_support=10, inner_steps=1, tasks_per_meta_batch=1000):
        
        # important objects
        self.model = model
        self.weights = list(model.parameters()) # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)
        
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_support = n_support
        self.n_query = 19 - n_support # For QMUL data loader
        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well 
        # self.tasks_per_meta_batch = tasks_per_meta_batch 
        
        # metrics
        self.plot_every = 10
        self.print_every = 100
        self.meta_losses = []
    
    def inner_loop(self, support_batch, support_batch_label, query_batch, query_batch_label):   # All from one task, meaning x tensor of rank 4 and y rank 1
        # reset inner model to current maml weights
        temp_weights = [w.clone().cuda() for w in self.weights]
        
        # perform training on data sampled from task (in sine task, same size query and support)
        X, y = support_batch, support_batch_label
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y[:,None]) / self.n_support
            
            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
        
        # sample new data for meta-update and compute loss
        X, y = query_batch, query_batch_label
        loss = self.criterion(self.model.parameterised(X, temp_weights), y[:,None]) / self.n_query
        
        return loss
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
        
        for iteration in range(1, num_iterations+1):
            
            # compute meta loss
            meta_loss = 0
            batch, batch_labels = get_batch(train_people)
            batch, batch_labels = batch.cuda(), batch_labels.cuda()
            indices = np.arange(batch.shape[1]) # Indices for the shuffling of datapoints for each task
            
            for inputs, labels in zip(batch, batch_labels):
                np.random.shuffle(indices)
                support_indices = np.sort(indices[0:self.n_support])

                query_indices = np.sort(indices[self.n_support:])
                support_inputs, query_inputs = inputs[support_indices, :, :, :], inputs[query_indices, :, :, :]
                support_labels, query_labels = labels[support_indices], labels[query_indices]
                meta_loss += self.inner_loop(support_inputs, support_labels, query_inputs, query_labels)
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            
            # log metrics
            epoch_loss += meta_loss.item() / batch.shape[0]
            
            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
            
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0


def loss_on_task(initial_model, n_support, num_steps, optim=torch.optim.SGD):
    """
    trains the model on a random sine task and measures the loss curve.
    
    for each n in num_steps_measured, records the model function after n gradient updates.
    """

    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 36, 3, stride=2, dilation=2)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(36, 36, 3, stride=2, dilation=2)),
        ('relu2', nn.ReLU()),
        ('conv3', nn.Conv2d(36, 36, 3, stride=2, dilation=2)),
        ('relu3', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('l4', nn.Linear(2916, 40)),
        ('relu4', nn.ReLU()),
        ('l5', nn.Linear(40, 40)),
        ('relu5', nn.ReLU()),
        ('l6', nn.Linear(40, 1))
    ])).cuda()
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    
    x_support, y_support, x_query, y_query = get_support_query_batches(test_people, n_support)
    
    optimiser = optim(model.parameters(), 0.01)

        
    for step in range(1, num_steps+1):
        loss = criterion(model(x_support), y_support[:,None]) / n_support
        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

    #Evaluate on query set
    loss = criterion(model(x_query), y_query[:,None]) / (19-n_support)    
    return loss               

def average_losses(initial_model, num_steps, n_support, nb_test_iterations, optim=torch.optim.SGD):
    """
    returns the average learning trajectory of the model trained for ``n_iterations`` over ``n_samples`` tasks
    """

    #x = np.linspace(-5, 5, 2) # dummy input for test_on_new_task
    avg_losses = list()
    for _ in range(nb_test_iterations):
        loss = loss_on_task(initial_model, n_support, num_steps, optim)
        avg_losses.append(loss.item())    
    return avg_losses







class NTKernel(gpytorch.kernels.Kernel):
    def __init__(self, net, sigma, **kwargs):
        super(NTKernel, self).__init__(**kwargs)
        self.net = net
        self.sigma = sigma

    def forward(self, x1, x2, diag=False, **params):
        x1 = x1.reshape(x1.size(0), 3, 100, 100)
        x2 = x2.reshape(x2.size(0), 3, 100, 100)
        jac1 = self.compute_jacobian(x1)
        jac2 = self.compute_jacobian(x2) if x1 is not x2 else jac1
        result = self.sigma * jac1@jac2.T
        
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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, net, sigma):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = NTKernel(net, sigma)
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




def loss_on_task_jacobian(model, gp, likelihood, n_support): # no optimizer needed for GP
    criterion = nn.MSELoss()
    
    x_support, y_support, x_query, y_query = get_support_query_batches(test_people, n_support)
    
    x_support_flat = x_support.view(x_support.size(0), -1) # Erase when not differentiating the whole network
    gp.train()
    gp.set_train_data(inputs=x_support_flat, targets=y_support - model(x_support).reshape(-1), strict=False)
    gp.eval()
        
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        x_query_flat = x_query.view(x_query.size(0), -1)
        pred    = likelihood(gp(x_query_flat))
        lower, upper = pred.confidence_region() #2 standard deviations above and below the mean
        lower += model(x_query).reshape(-1)
        upper += model(x_query).reshape(-1)
    mse = criterion(pred.mean + model(x_query).reshape(-1), y_query)

    return mse


def average_losses_jacobian(model, gp, likelihood, n_support, nb_test_iterations):
    """
    returns the average learning trajectory of the model trained for ``n_iterations`` over ``n_samples`` tasks
    """

    avg_losses = list()
    for _ in range(nb_test_iterations):

        loss = loss_on_task_jacobian(model, gp, likelihood, n_support)
        avg_losses.append(loss.item())    
    return avg_losses



def main():     
    ## Simulation Parameters
    train_iterations = 10000
    inner_steps = 1 
    
    maml = MAML(QMULMAMLBigModel().cuda(), inner_lr=0.01, meta_lr=0.001)
    
    # For training new model, uncomment these lines :
    #maml.main_loop(num_iterations=train_iterations)
    #model_params = maml.model.state_dict()
    #torch.save(model_params, f'maml_params_QMUL_{train_iterations}.pth')
    
    model_params = torch.load(f'maml_params_QMUL_{train_iterations}.pth')
    maml.model.load_state_dict(model_params)
    
    # Testing
    
    maml.model.eval()
    n_support=10
    num_steps_fine_tuning = [1, 3, 5, 10, 100]
    nb_test_iterations = 100
    
    print("Test fine-tuning, please wait...")
    for num_steps in num_steps_fine_tuning:
        mse_list = average_losses(maml.model, num_steps, n_support, nb_test_iterations, optim=torch.optim.Adam)
        print("-------------------")
        print(f"For {num_steps} steps of adaptation")
        print(f"Average MSE: {np.mean(mse_list)} +- {np.std(mse_list)}")
        print("-------------------")
    
    
    sigmas = [1, 5, 10, 50, 100]
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    dummy_inputs = torch.ones(19, 30000).cuda()
    dummy_labels = torch.ones(19).cuda()
    
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 36, 3, stride=2, dilation=2)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(36, 36, 3, stride=2, dilation=2)),
        ('relu2', nn.ReLU()),
        ('conv3', nn.Conv2d(36, 36, 3, stride=2, dilation=2)),
        ('relu3', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('l4', nn.Linear(2916, 40)),
        ('relu4', nn.ReLU()),
        ('l5', nn.Linear(40, 40)),
        ('relu5', nn.ReLU()),
        ('l6', nn.Linear(40, 1))
    ])).cuda()
    model.load_state_dict(maml.model.state_dict())
    
    print("Test Jacobian adaptation, please wait...")
    for sigma in sigmas:
        gp = ExactGPModel(dummy_inputs, dummy_labels, likelihood, model, sigma).cuda()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
        
        likelihood.eval()
    
        mse_list = average_losses_jacobian(model, gp, likelihood, n_support, nb_test_iterations)
        print("-------------------")
        print(f"For sigma = {sigma}")
        print(f"Average MSE: {np.mean(mse_list)} +- {np.std(mse_list)}")
        print("-------------------")



if __name__ == '__main__':
    main()       

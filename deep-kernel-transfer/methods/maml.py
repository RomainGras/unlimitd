# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from data.qmul_loader import get_batch, train_people, test_people

from torch.func import functional_call

class MAML(MetaTemplate):
    def __init__(self, model_func, n_support,  n_way = 1, approx = False, problem = "classification"):
        super(MAML, self).__init__( model_func,  n_way, n_support, change_way = False)
        self.problem = problem
        if problem == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        if problem == "regression": 
            self.loss_fn = nn.MSELoss()
            # In regression problems, n_way doesn't appear, because the output of the model is already dimension of the output space
        if problem=="classification" or (problem=="regression" and self.feature.feat_dim > 1):
            self.last_layer = backbone.Linear_fw(self.feature.feat_dim, n_way)
            self.last_layer.bias.data.fill_(0)
        else:
            self.last_layer = nn.Identity()
        
        self.n_task     = 8
        self.task_update_num = 3
        self.train_lr = 0.01
        self.approx = approx #first order approx.        

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.last_layer.forward(out)
        return scores

    def set_forward(self,x, y_a_i, is_feature = False, print_every=0):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x = x.cuda()
        x_var = Variable(x)
        
        if self.problem == "classification":
            x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
            x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        elif self.problem == "regression":
            x_a_i = x_var[:self.n_support].contiguous() #support data 
            x_b_i = x_var[self.n_support:].contiguous() #query data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()
        
        scores = []

        for task_step in range(self.task_update_num):
            score_at_step = self.forward(x_a_i).squeeze(1)
            set_loss = self.loss_fn( score_at_step, y_a_i)
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
            if print_every and task_step%print_every==0:
                scores.append(self.forward(x_b_i).squeeze(1))

        scores.append(self.forward(x_b_i).squeeze(1))
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x, y = None, print_every=0): # y is None for classification
        if self.problem == "classification":
            y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
            y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
            
        elif self.problem == "regression":
            y_var = Variable(y)
            y_a_i = y_var[:self.n_support].contiguous().cuda()
            y_b_i = y_var[self.n_support:].contiguous().cuda()
            
        scores = self.set_forward(x, y_a_i, is_feature = False, print_every=print_every)
        
        losses = [self.loss_fn(scores_step, y_b_i) for scores_step in scores]
        return losses

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

            loss = self.set_forward_loss(x)[0]
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
    
    
    def train_loop_regression(self, epoch, provider, optimizer): #overwrite parrent function
        print_freq = 8
        avg_loss=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()
        
        batch, batch_labels = provider.get_train_batch()
        #train
        for i, (x,y) in enumerate(zip(batch, batch_labels)):
            self.n_query = x.size(1) - self.n_support
            if self.problem == "classification" :
                assert self.n_way  ==  x.size(0), "MAML do not support way change"
            
            loss = self.set_forward_loss(x, y)[0]
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
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(batch), avg_loss/float(i+1)))
                
    def save_checkpoint(self, checkpoint):
        # save state
        nn_state_dict         = self.feature.state_dict()
        last_layer_state_dict = self.last_layer.state_dict()
        torch.save({'net':nn_state_dict, 'last_layer':last_layer_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        #ckpt_net = dict()
        #for name, weight in ckpt['net'].items():
        #    new_key = name.replace("0.layer", "0.conv")
        #    ckpt_net[new_key] = weight
        self.feature.load_state_dict(ckpt['net'])
        self.last_layer.load_state_dict(ckpt['last_layer'])
                      
    def test_loop(self, n_support, test_loader, return_std = False, optimizer = None, print_every=0): #overwrite parrent function
        if self.problem=="classification":
            correct =0
            count = 0
            acc_all = []

            iter_num = len(test_loader) 
            for i, (x,_) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way  ==  x.size(0), "MAML do not support way change"
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this/ count_this *100 )

            acc_all  = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
            if return_std:
                return acc_mean, acc_std
            else:
                return acc_mean
        
        else: #Regression
            # TODO : Actually implement different n_support
            (x_support, y_support), (x_query, y_query) = test_loader.get_test_batch()

            # choose a random test person
            n = np.random.randint(0, x_support.size(0)-1)
            self.n_support = x_support.size(0)
            print(f"Beggining adaptation with n_support {self.n_support}")
            self.feature.eval()
            
            # set_forward_loss splits x_support and x_query according to _support. Concatenating them is more practical, but the split will be the same
            x = torch.cat((x_support[n], x_query[n]), 0)
            y = torch.cat((y_support[n], y_query[n]), 0)
            
            mse = self.set_forward_loss(x, y, print_every=print_every) # If print_every is bigger than 0, then returns a list of float mse for every epoch of adaptation
            if print_every:
                return mse
            else:
                return mse[0]
            

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits = self.set_forward(x)
        return logits


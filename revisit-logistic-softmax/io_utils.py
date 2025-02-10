import numpy as np
import os
import glob
import argparse
import backbone
import torch.nn as nn


import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

#Redefine Conv4 here :

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)    
    
class ConvNet(nn.Module):
    maml = False # Default

    def __init__(self, depth, n_way=-1, flatten=True, padding=1, bn=False):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            if self.maml:
                conv_layer = Conv2d_fw(indim, outdim, 3, padding=padding)
                if bn:
                    BN     = BatchNorm2d_fw(outdim)
            else:
                conv_layer = nn.Conv2d(indim, outdim, 3, stride=1, padding=padding, bias=False)
                if bn:
                    BN     = nn.BatchNorm2d(outdim)
            
            relu = nn.ReLU(inplace=True)
            layers.append(conv_layer)
            if bn:
                layers.append(BN)
            layers.append(relu)

            if i < 4:  # Pooling only for the first 4 layers
                pool = nn.MaxPool2d(2)
                layers.append(pool)

            # Initialize the layers
            init_layer(conv_layer)
            if bn:
                init_layer(BN)

        if flatten:
            layers.append(Flatten())
        
        if n_way>0:
            layers.append(nn.Linear(1600,n_way))
            self.final_feat_dim = n_way
        else:
            self.final_feat_dim = 1600
            
        self.trunk = nn.Sequential(*layers)
        

    def forward(self, x):
        out = self.trunk(x)
        return out

def Conv4NoBN():
    print("Conv4 No Batch Normalization")
    return ConvNet(4, bn=False)

def Conv4NoBN_class(n_way=5):
    print("Conv4 No Batch Normalization with final classifier layer of 5 way")
    return ConvNet(4, n_way=n_way, bn=False)

def Conv4():
    print("Conv4 No Batch Normalization")
    return ConvNet(4, bn=True)

def Conv4_class(n_way=5):
    print("Conv4 No Batch Normalization with final classifier layer of 5 way")
    return ConvNet(4, n_way=n_way, bn=True)


def combined_conv3():
    print("Conv3")
    return backbone.CombinedNetwork(backbone.Conv3(), nn.Linear(1764,5))


# Redefine Resnet here:

class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res, bn = False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        if bn:
            self.BN1 = nn.BatchNorm2d(outdim)
        else: # We need to define a layer such as this one for forwarding
            self.BN1 = nn.Identity()
            
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        if bn:
            self.BN2 = nn.BatchNorm2d(outdim)
        else: # We need to define a layer such as this one for forwarding
            self.BN2 = nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            if bn:
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            if bn:
                self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = out + short_out
        out = self.relu2(out)
        return out
    
class ResNet_custom(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True, n_way = -1, bn = False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet_custom,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if bn:
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        if bn:
            init_layer(bn1)

        if bn:
            trunk = [conv1, bn1, relu, pool1]
        else:
            trunk = [conv1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)  # If the start of a new type of layers
                B = block(indim, list_of_out_dims[i], half_res, bn=bn)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(3) # originally avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]
            
        if n_way>0:
            trunk.append(nn.Linear(2048,n_way))
            self.final_feat_dim = n_way
        else:
            self.final_feat_dim = 2048

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out
    
    

def ResNetNoBN10( flatten = True):
    return ResNet_custom(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNetNoBN10_classifier( flatten = True):
    return ResNet_custom(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, n_way = 5)

def ResNetNoBN18_classifier( flatten = True):
    return ResNet_custom(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, n_way = 5)

def ResNet10( flatten = True):
    return ResNet_custom(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, bn = True)

def ResNet10_classifier( flatten = True):
    return ResNet_custom(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, n_way = 5, bn = True)

def ResNet18_classifier( flatten = True):
    return ResNet_custom(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, n_way = 5, bn = True)


# Recreating Conv4SNoBN
    
class ConvNetNoBN(nn.Module):
    maml = False # Default

    def __init__(self, depth, n_way=-1, flatten=True, padding=1):
        super(ConvNetNoBN, self).__init__()
        layers = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            if self.maml:
                conv_layer = Conv2d_fw(indim, outdim, 3, padding=padding)
                # BN     = BatchNorm2d_fw(outdim)
            else:
                conv_layer = nn.Conv2d(indim, outdim, 3, stride=1, padding=padding, bias=False)
                # BN     = nn.BatchNorm2d(outdim)
            
            relu = nn.ReLU(inplace=True)
            layers.append(conv_layer)
            # layers.append(BN)
            layers.append(relu)

            if i < 4:  # Pooling only for the first 4 layers
                pool = nn.MaxPool2d(2)
                layers.append(pool)

            # Initialize the layers
            init_layer(conv_layer)
            # init_layer(BN)

        if flatten:
            layers.append(Flatten())
        
        if n_way>0:
            layers.append(nn.Linear(1600,n_way))
            self.final_feat_dim = n_way
        else:
            self.final_feat_dim = 1600
            
        self.trunk = nn.Sequential(*layers)
        

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet4SNoBN(nn.Module): #For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, n_way=-1, flatten = True, padding=1):
        super(ConvNet4SNoBN,self).__init__()
        layers = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            conv_layer = nn.Conv2d(indim, outdim, 3, stride=1, padding=padding, bias=False)
            # BN     = nn.BatchNorm2d(outdim)
            relu = nn.ReLU(inplace=True)
            layers.append(conv_layer)
            # layers.append(BN)
            layers.append(relu)
            
            if i < 4:  # Pooling only for the first 4 layers
                pool = nn.MaxPool2d(2)
                layers.append(pool)
            
            init_layer(conv_layer)

        if flatten:
            layers.append(Flatten())

        #trunk.append(nn.BatchNorm1d(64))    #TODO remove
        #trunk.append(nn.ReLU(inplace=True)) #TODO remove
        if n_way>0:
            layers.append(nn.Linear(64, n_way))
            self.final_feat_dim = n_way
        else:
            self.final_feat_dim = 64
        self.trunk = nn.Sequential(*layers)

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        #out = torch.tanh(out) #TODO remove
        return out
    
    
def Conv4SNoBN():
    print("Conv4S")
    return ConvNet4SNoBN(4, n_way=-1)

def Conv4SNoBN_class():
    print("Conv4S")
    return ConvNet4SNoBN(4, n_way=5)


##############################
# Sort this out
##############################


    
    
class BatchNorm2d_hack(nn.BatchNorm2d):  # Used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_hack, self).__init__(num_features)
    
    def forward(self, x):
        # Determine if the model is in training or eval mode
        running_mean = torch.zeros(x.data.size(1), device=x.device)
        running_var = torch.ones(x.data.size(1), device=x.device)
        out = F.batch_norm(
            x, 
            running_mean, 
            running_var, 
            self.weight, 
            self.bias, 
            training=self.training,  # Ensures batch statistics are used
            momentum=1,   # Using momentum=1 disables running stats update : originally ; momentum=int(self.training)
            eps=1e-5
        )
        return out
    
# Simple Conv Block
class ConvBlock_DIFF(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock_DIFF, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, stride = 1, padding = padding, bias = True) # Default : bias=False
            self.BN     = BatchNorm2d_hack(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)  # Originally : self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self,x):
        x = self.C(x)
        x = self.BN(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x


class Conv4_DIFF(nn.Module):
    def __init__(self, depth, flatten = True):
        super(Conv4_DIFF,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock_DIFF(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out
    
    
def Conv4_diffDKTIX():
    print("Conv4_diffDKTIX")
    return Conv4_DIFF(4)
##############################
##############################



model_dict = dict(
            Conv3 = backbone.Conv3,
            Conv4 = backbone.Conv4,
            Conv4NoBN = Conv4NoBN,
            Conv4_custom = Conv4,
            Conv4S = backbone.Conv4S,
            Conv4SNoBN = Conv4SNoBN,
            Conv4_diffDKTIX = Conv4_diffDKTIX,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNetNoBN10 = ResNetNoBN10,
            combined_ResNetNoBN10 = ResNetNoBN10_classifier,
            combined_ResNet10_custom = ResNet10_classifier,
            ResNet18 = backbone.ResNet18,
            combined_ResNetNoBN18 = ResNetNoBN18_classifier,
            combined_ResNet18_custom = ResNet18_classifier,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101,
            simple_netC_0hl = backbone.simple_netC_0hl,
            simple_netC_0hl_1o = backbone.simple_netC_0hl_1o,
            simple_netC_1hl = backbone.simple_netC_1hl,
            simple_netC_1hl_1o = backbone.simple_netC_1hl_1o,
            simple_netC_2hl = backbone.simple_netC_2hl,
            simple_netC_2hl_1o = backbone.simple_netC_2hl_1o,
            identity = backbone.Identity,
            combined_resnet10 = backbone.CombinedNetwork(backbone.ResNet10(), nn.Linear(512,5)),
            combined_resnet34 = backbone.CombinedNetwork(backbone.ResNet34(), nn.Linear(512,5)),
            combined_Conv3 = combined_conv3,
            combined_Conv4NoBN = Conv4NoBN_class,
            combined_Conv4_custom = Conv4_class,
            combined_Conv4S = backbone.CombinedNetwork(backbone.Conv4S(), nn.Linear(1600,5)),
            combined_Conv6 = backbone.CombinedNetwork(backbone.Conv6(), nn.Linear(1600,5)),
            combined_Conv3_1o = backbone.CombinedNetwork(backbone.Conv3(), nn.Linear(1764,1)),
            combined_Conv4SNoBN = Conv4SNoBN_class
)


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--seed' , default=0, type=int,  help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--steps'   , default=2, type=int,  help='mean field steps for cdkt') 
    parser.add_argument('--tau'   , default=1., type=float,  help='temperature for cdkt')
    parser.add_argument('--loss'   , default='ELBO', type=str,  help='loss for cdkt')
    parser.add_argument('--mean'   , default=0., type=float,  help='prior mean for cdkt, should be negative if we want to be like softmax')
    parser.add_argument('--kernel'   , default='bncossim', type=str,  help='kernel for GP, choices include: linear, rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim')
    parser.add_argument('--batch_size'   , default=1, type=int,  help='task numbers in parallel for cdkt') 
    parser.add_argument('--diff_net'   , default=None, type=str,  help='for UNLIMITD-CDKT, the differentiated head network') 
    
    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
        
    elif script == 'maml_to_diffDKTIX':
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=10, type=int, help ='Stopping epoch')
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--repeat', default=5, type=int, help ='Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]')
        parser.add_argument('--optim_based_test', default=False, type=bool, help ='Using optimization-based testing for ours or oursX')
        parser.add_argument('--n_ft', default=0, type=int, help ='Using optimization-based testing for ours or oursX, this is the number of fine-tuning steps')
        parser.add_argument('--lr', default=0, type=float, help ='Using optimization-based testing for ours, this is the learning rate of the optimizer. Using optimization-based testing for oursX, this is the shrinking factor. They are both into the same variable for convinience reasons.')
        parser.add_argument('--temp', default=0.3, type=float, help ='Using optimization-based testing for ours or oursX, this is the temperature of softmax')
        
        
        
    else:
       raise ValueError('Unknown script')


    return parser.parse_args()

def parse_args_regression(script):
        parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
        parser.add_argument('--seed' , default=0, type=int,  help='Seed for Numpy and pyTorch. Default: 0 (None)')
        parser.add_argument('--model'       , default='Conv3',   help='model: Conv{3} / MLP{2}')
        parser.add_argument('--method'      , default='DKT',   help='DKT / transfer')
        parser.add_argument('--dataset'     , default='QMUL',    help='QMUL / sines')
        parser.add_argument('--spectral', action='store_true', help='Use a spectral covariance kernel function')

        if script == 'train_regression':
            parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
            parser.add_argument('--stop_epoch'  , default=100, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
            parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        elif script == 'test_regression':
            parser.add_argument('--n_support', default=5, type=int, help='Number of points on trajectory to be given as support points')
            parser.add_argument('--n_test_epochs', default=10, type=int, help='How many test people?')
        return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding, dilation=self.dilation)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding, dilation=self.dilation)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
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
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetS(nn.Module): #For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten = True):
        super(ConvNetS,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        #trunk.append(nn.BatchNorm1d(64))    #TODO remove
        #trunk.append(nn.ReLU(inplace=True)) #TODO remove
        #trunk.append(nn.Linear(64, 64))     #TODO remove
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        #out = torch.tanh(out) #TODO remove
        return out

class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,5,5]

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out



    
# Backbone for QMUL regression
class Conv3(nn.Module):
    maml=False
    def __init__(self):
        super(Conv3, self).__init__()
        if self.maml:
            self.conv1 = Conv2d_fw(3, 36, 3,stride=2,dilation=2)
            self.conv2 = Conv2d_fw(36,36, 3,stride=2,dilation=2)
            self.conv3 = Conv2d_fw(36,36, 3,stride=2,dilation=2)
            
        else:
            self.conv1 = nn.Conv2d(3, 36, 3,stride=2,dilation=2)
            self.conv2 = nn.Conv2d(36,36, 3,stride=2,dilation=2)
            self.conv3 = nn.Conv2d(36,36, 3,stride=2,dilation=2)
        
        self.feat_dim = 2916

    def return_clones(self):
        layer1_w = self.conv1.weight.data.clone().detach()
        layer2_w = self.conv2.weight.data.clone().detach()
        layer3_w = self.conv3.weight.data.clone().detach()
        return [layer1_w, layer2_w, layer3_w]

    def assign_clones(self, weights_list):
        self.conv1.weight.data.copy_(weights_list[0])
        self.conv2.weight.data.copy_(weights_list[1])
        self.conv3.weight.data.copy_(weights_list[2])

    def forward(self, x):
        out = F.relu(self.conv1.forward(x))
        out = F.relu(self.conv2.forward(out))
        out = F.relu(self.conv3.forward(out))
        out = out.view(out.size(0), -1)
        return out

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)

def Conv4S():
    return ConvNetS(4)

def Conv4SNP():
    return ConvNetSNopool(4)

def ResNet10( flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34( flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)

#===================
# simple net
#===================

class simple_net(nn.Module):
    maml=False
    def __init__(self):
        super(simple_net, self).__init__()
        if self.maml:
            self.layer1 = Linear_fw(2916, 40)
            self.layer2 = Linear_fw(40, 40)
            self.layer3 = Linear_fw(40, 1)
        else:
            self.layer1 = nn.Linear(2916, 40)
            self.layer2 = nn.Linear(40,40)
            self.layer3 = nn.Linear(40,1)
        self.feat_dim=1
        
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out

    
class simple_net_multi_output(nn.Module):
    def __init__(self):
        super(simple_net_multi_output, self).__init__()
        self.layer1 = nn.Linear(2916, 40)
        self.layer2 = nn.Linear(40,40)
        self.feat_dim=40
        
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        return out


    
class CombinedNetwork(nn.Module):
    def __init__(self, net1, net2):
        super(CombinedNetwork, self).__init__()
        self.networks = nn.Sequential(
            net1,
            net2
        )
        self.feat_dim = net2.feat_dim
    
    def forward(self, x):
        return self.networks(x)
    
    
#========================
# MLPs
#========================

class ThreeLayerMLP(nn.Module):
    maml = False
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if self.maml:
            self.hidden1 = Linear_fw(input_dim, 32)
            self.hidden2 = Linear_fw(32, 32)
            self.hidden3 = Linear_fw(32, 32)
            self.output = Linear_fw(32, output_dim)
        else:
            self.hidden1 = nn.Linear(input_dim, 32)
            self.hidden2 = nn.Linear(32, 32)
            self.hidden3 = nn.Linear(32, 32)
            self.output = nn.Linear(32, output_dim)
        
        self.feat_dim = output_dim

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))

        x = self.output(x)
        return x
    

class SimpleMLP(nn.Module):
    """
    Simple 3-layer MLP with 32 hidden units each, ReLU activations, and default pytorch layer initialization
    """
    def __init__(self, input_dim, output_dim, hidden_size=32):
        super().__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)
        
        self.relu = nn.ReLU()
    
    def return_clones(self):
        layer1_w = self.layer1.weight.data.clone().detach()
        layer2_w = self.layer2.weight.data.clone().detach()
        layer3_w = self.layer3.weight.data.clone().detach()
        output_w = self.output.weight.data.clone().detach()
        return [layer1_w, layer2_w, layer3_w, output_w]

    def assign_clones(self, weights_list):
        self.layer1.weight.data.copy_(weights_list[0])
        self.layer2.weight.data.copy_(weights_list[1])
        self.layer3.weight.data.copy_(weights_list[2])
        self.output.weight.data.copy_(weights_list[3])
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output(x)
        
        

class SteinwartMLP(nn.Module):
    """
    3-layer MLP with 32 hidden units each, Leaky ReLU activations,
    He (Kaiming) initialization for weights, and a simplified Steinwart (2019) initialization for biases.
    """
    maml=False
    def __init__(self, input_dim, output_dim, hidden_size=32, leaky_relu_negative_slope=0.01):
        super().__init__()
        if self.maml:
            self.hidden1 = Linear_fw(input_dim, hidden_size)
            self.hidden2 = Linear_fw(hidden_size, hidden_size)
            self.hidden3 = Linear_fw(hidden_size, hidden_size)
            self.output = Linear_fw(hidden_size, output_dim)
        else:
            self.hidden1 = nn.Linear(input_dim, hidden_size)
            self.hidden2 = nn.Linear(hidden_size, hidden_size)
            self.hidden3 = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, output_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        
        self.feat_dim = output_dim
        
        
        # Initialize weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies:
          - He (Kaiming) initialization for the weight tensors.
          - Steinwart-style bias initialization:
              bias_i = - <w_i, x*_i>,
            where w_i is a normalized random vector, and x*_i is sampled from an assumed domain.
        """
        # Domain boundaries for Steinwart bias initialization (adjust if needed)
        min_val = -1.0
        max_val =  1.0

        def steinwart_bias_init(layer: nn.Linear):
            """
            Computes a bias = - <w_i, x*_i>, where:
              - w_i is a normalized random vector (per output neuron).
              - x*_i is sampled uniformly in [min_val, max_val]^in_features.
            """
            out_features, in_features = layer.weight.shape

            # Sample random vectors w_i in R^{in_features} and normalize to unit sphere
            w = torch.randn(out_features, in_features)
            w = w / (w.norm(dim=1, keepdim=True) + 1e-8)  # Avoid division by zero

            # Sample x*_i uniformly
            x_star = min_val + (max_val - min_val) * torch.rand_like(w)

            # bias_i = - <w_i, x*_i>
            b = -(w * x_star).sum(dim=1)

            with torch.no_grad():
                layer.bias.copy_(b)

        # Apply to each layer
        for layer in [self.hidden1, self.hidden2, self.hidden3, self.output]:
            # He (Kaiming) initialization for weights (suitable for Leaky ReLU)
            nn.init.kaiming_normal_(layer.weight, a=0.01)  # a = negative slope of LeakyReLU

            # Steinwart bias initialization
            steinwart_bias_init(layer)

    def forward(self, x):
        # Pass through 3 hidden layers with Leaky ReLU
        x = self.leaky_relu(self.hidden1(x))
        x = self.leaky_relu(self.hidden2(x))
        x = self.leaky_relu(self.hidden3(x))

        # Output layer (no activation by default)
        return self.output(x)


if __name__ == "__main__":
    # Example usage
    model = SteinwartInitMLP(input_dim=10, output_dim=1)
    example_input = torch.randn(4, 10)  # batch of size 4, dimension 10
    output = model(example_input)
    print("Output shape:", output.shape)

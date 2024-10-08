import numpy as np
import os
import glob
import argparse
import backbone
import torch.nn as nn

model_dict = dict(
            Conv3 = backbone.Conv3,
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
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
            combined_Conv3 = backbone.CombinedNetwork(backbone.Conv3(), nn.Linear(1764,5)),
            combined_Conv4 = backbone.CombinedNetwork(backbone.Conv4(), nn.Linear(1600,5)),
            combined_Conv4S = backbone.CombinedNetwork(backbone.Conv4S(), nn.Linear(1600,5)),
            combined_Conv6 = backbone.CombinedNetwork(backbone.Conv6(), nn.Linear(1600,5)),
            combined_Conv3_1o = backbone.CombinedNetwork(backbone.Conv3(), nn.Linear(1764,1)),
            Conv4Net = backbone.CombinedNetwork(backbone.Conv4Net(), nn.Linear(1600,5))
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
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--repeat', default=5, type=int, help ='Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]')
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

#importing modules
import os
import argparse

import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from load_sat_data import *

from ResNet import *
from DenseNet import *
from BranchedMobileNet import *

from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument("target",help="path to model", type=str)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

if args.data == 'MNIST':
    input_shape = (1,28,28)
    output_classes = 10
    n_channels = 1

if args.data == 'CIFAR10':
    input_shape = (3,32,32)
    output_classes = 10
    n_channels = 3

if args.data == 'CIFAR100':
    input_shape = (3,32,32)
    output_classes = 100
    n_channels = 3

if args.data == 'sat-6':
    input_shape = (4,28,28)
    output_classes = 6
    n_channels = 4

if args.data == 'eurosat-rgb':
    input_shape = (3,64,64)
    output_classes = 10
    n_channels = 3

if args.data == 'eurosat-full':
    input_shape = (13,64,64)
    output_classes = 10
    n_channels = 13

def get_model(model_directory,run,device):

    model_name = model_directory.split('/')[1]

    if model_name == 'ResNet18':
        model = ResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif model_name == 'ResNet34':
        model = ResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif model_name == 'ResNet50':
        model = ResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    elif model_name == 'BranchedResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
        important = [1,2,3,16,31,46,61,62,63,64]
    elif model_name == 'BranchedResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif model_name == 'BranchedResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    elif model_name == 'BranchedMobileNet':
        model = BranchedMobileNet([0.25,0.5,0.75],num_classes=output_classes,input_channels=n_channels, in_size = 32)
        important = [2,6,10,16,20,36,48,61,79,98,117,135,154,172,191,210,214,218]
    else:
        model = ResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18

    #putting to GPU and loading weights
    model = model.to(device)

    weights_directory = "saved-models/best-"+model_name+'-CIFAR-10-'+str(run)+'.pth'
    model.load_state_dict(torch.load(model_directory + weights_directory, map_location=device))

    print("loaded weights at: ",model_directory + weights_directory)

    return(model,important)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_directory = args.target

print(model_directory)

model,important = get_model(model_directory,0,device)

macs, params, layerwise = get_model_complexity_info(model, input_shape,print_per_layer_stat=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

layerwise_macs, layerwise_params = np.array(layerwise[0]),np.array(layerwise[1])
np.save((model_directory+"/metrics/layerwise_parameters.npy"),layerwise_params)
np.save((model_directory+"/metrics/layerwise_macs.npy"),layerwise_macs)

print(layerwise_macs.shape,layerwise_params.shape)

important_macs = np.take(layerwise_macs,important)
important_params = np.take(layerwise_params,important)

print(important_macs)

print(np.sum(important_macs))
print(np.sum(important_params))

np.save((model_directory+"/metrics/layerwise_parameters.npy"),layerwise_params)
np.save((model_directory+"/metrics/layerwise_macs.npy"),layerwise_macs)

#importing modules
import os
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision as tv

from ResNet import *
from DenseNet import *

#some helper functions
def get_label(branch_weights):
    string = "("
    for weight in branch_weights:
        weight = np.round(weight,decimals=3)
        string = string + str(weight) + " : "
    string = string[:-3]+')'
    return(string)

def get_save_name(directory,info):
    model_name = info[1]
    save_name = model_name
    if 'branched' in model_name.lower():
        weights = np.load(directory+"/metrics/branch-weights-"+model_name+".npy")
        save_name = save_name + " " + get_label(weights)
    return(save_name)

def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
    elif model_name == 'BranchedResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif model_name == 'BranchedResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    else:
        model = ResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18

    #putting to GPU and loading weights
    model = model.to(device)

    weights_directory = "saved-models/best-"+model_name+'-CIFAR-10-'+str(run)+'.pth'
    model.load_state_dict(torch.load(model_directory + weights_directory, map_location=device))

    print("loaded weights at: ",model_directory + weights_directory)

    return(model)

def activation_hook(inst, inp, out):
    """Run activation hook.
    Parameters
    ----------
    inst : torch.nn.Module
        The layer we want to attach the hook to.
    inp : tuple of torch.Tensor
        The input to the `forward` method.
    out : torch.Tensor
        The output of the `forward` method.
    """

    acts = torch.sum(out,dim=(2,3))
    acts_shp = acts.shape

    max_activations = np.zeros(acts_shp[1])
    activation_indices = np.zeros(acts_shp[1],dtype=int)

    print("In:",inst)
    print("Output shape: ", out.shape)
    print("Sum along channels: ", acts.shape)

    for i in range(acts_shp[1]):
        max_activations[i] = torch.max(acts[:,i])
        activation_indices[i] = torch.argmax(acts[:,i])

    print("Max activations: ", max_activations)
    print("Activation indices: ", activation_indices)

    np.savetxt(activations,max_activations)
    np.savetxt(indices,activation_indices)


#Arguments for running
parser = argparse.ArgumentParser()

parser.add_argument("target",help="path to model;", type=str)
parser.add_argument("-r","--runs", help="Must be less than or equal to number of trained models.",type=int,default=3)
parser.add_argument("-b","--batch_size", help="Batch size for cka analysis.",type=int,default=128)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

model_directory  = args.target

target_info = model_directory.split('/')

save_name = get_save_name(model_directory,target_info)

print('arguments passed:')
print("Target: " + args.target)
print("Dataset: " + args.data)
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(model_directory+"activation-analysis")
    print("Directory " , model_directory+"/activation-analysis/" ,  " Created ")
except FileExistsError:
    print("Directory " ,model_directory+"/activation-analysis/" ,  " already exists")

save_directory = model_directory+"/activation-analysis/"

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(save_directory+"/figures/")
    print("Directory " , save_directory+"/figures/" ,  " Created ")
except FileExistsError:
    print("Directory " ,save_directory+"/figures/" ,  " already exists")

#Preparing data

#normalising data
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.data == 'MNIST':
    train_data = datasets.MNIST('data', train=True, download=True, transform = transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform = transforms.ToTensor())
    output_classes = 10
    n_channels = 1
    label_names = list('0123456789')
    n_values = 10000
    annotate = True

if args.data == 'CIFAR10':
    train_data = datasets.CIFAR10('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform = transform_test)
    output_classes = 10
    n_channels = 3
    meta_data = np.load('data/cifar-10-batches-py/batches.meta',allow_pickle=True)
    label_names = meta_data['label_names']
    n_values = meta_data['num_cases_per_batch']
    annotate = True

if args.data == 'CIFAR100':
    train_data = datasets.CIFAR100('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR100('data', train=False, download=True, transform = transform_test)
    output_classes = 100
    n_channels = 3
    meta_data = np.load('data/cifar-100-python/meta',allow_pickle=True)
    label_names = meta_data['fine_label_names']
    annotate = False
    n_values = 10000

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)
test_loader =   DataLoader(test_data, batch_size=args.batch_size)

#Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

matrices = list()

run = 0

print("Run: " + str(run))
#defining models

model = get_model(model_directory,run,device)

input,target = next(iter(train_loader))
input,target = input.to(device),target.to(device)

with torch.no_grad():
    output = model(input)[0]
output_shape = len(output)

for name,param in model.named_modules():
    if name == "layer1.0.conv1":
        print("Attaching forward hook to: ", name)
        layer1 = param.register_forward_hook(activation_hook)

activations_file = save_directory+"activations.txt"
activations = open(activations_file,'w')  # write in text mode

indices_file = save_directory+"indices.txt"
indices = open(indices_file,'w')  # write in text mode

meta = np.save((save_directory+"meta"), np.array([args.target,args.data,args.runs, args.batch_size],dtype=object))

batch_n = 0
for batch in test_loader:

    x,y = batch

    # for i in range(args.batch_size):  # show just the frogs
    #     if i == 6:  # 6 = frog
    #         imshow(tv.utils.make_grid(x[i]))

    x,y = x.to(device),y.to(device)
    #1-forward pass - get logites

    with torch.no_grad():
        acts = model(x)[0]

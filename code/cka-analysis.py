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

import CKA.cca_core
from CKA.alternative_CKA import *
from CKA.CKA import linear_CKA, kernel_CKA

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

#Arguments for running
parser = argparse.ArgumentParser()

parser.add_argument("target_a",help="path to model a", type=str)
parser.add_argument("target_b",help="path to model b", type=str)
parser.add_argument("-r","--runs", help="Must be less than or equal to number of trained models.",type=int,default=3)
parser.add_argument("-b","--batch_size", help="Batch size for cka analysis.",type=int,default=128)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

model_a_directory  = args.target_a
model_b_directory = args.target_b

target_a_info = model_a_directory.split('/')
target_b_info = model_b_directory.split('/')

save_name_a = get_save_name(model_a_directory,target_a_info)
save_name_b = get_save_name(model_b_directory,target_b_info)

directory = target_a_info[0]+'/'+target_a_info[1]+'/'+target_a_info[2]+'/'

print('arguments passed:')
print("Target a: " + args.target_a)
print("Target b: " + args.target_b)
print("Dataset: " + args.data)
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(directory+"cka-analysis")
    print("Directory " , directory+"/cka-analysis/" ,  " Created ")
except FileExistsError:
    print("Directory " ,directory+"/cka-analysis/" ,  " already exists")

save_directory = directory+"/cka-analysis/"

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(save_directory+"/figures/")
    print("Directory " , save_directory+"/figures/" ,  " Created ")
except FileExistsError:
    print("Directory " ,save_directory+"/figures/" ,  " already exists")

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(save_directory+"/matrices/")
    print("Directory " , save_directory+"/matrices/" ,  " Created ")
except FileExistsError:
    print("Directory " ,save_directory+"/matrices/" ,  " already exists")

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

for run in range(args.runs):
    print("Run: " + str(run))
    #defining models

    model_a = get_model(model_a_directory,run,device)
    model_b = get_model(model_b_directory,run,device)

    for name,module in model_a.named_modules():
        print(module)

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    with torch.no_grad():
        output_a = model_a(input)[0]
        output_b = model_b(input)[0]
    output_shape_a = len(output_a)
    output_shape_b = len(output_b)

    CKA_matrix = np.zeros((output_shape_a,output_shape_b))
    for layer_a in range(output_shape_a):
        avg_acts_a = output_a[layer_a][:,:,:,:].flatten(start_dim=1).detach().numpy()
        print(output_a[layer_a][:,:,:,:].shape)
        # print(avg_acts_a.shape)
        for layer_b in range(output_shape_b):
            avg_acts_b = output_b[layer_b][:,:,:,:].flatten(start_dim=1).detach().numpy()

            CKA_matrix[layer_a,layer_b] = alternative_CKA(avg_acts_a,avg_acts_b)

    batch_n = 0
    for batch in test_loader:

        x,y = batch
        x,y = x.to(device),y.to(device)
        #1-forward pass - get logites

        with torch.no_grad():
            acts_a = model_a(x)[0]
            acts_b = model_b(x)[0]

        for layer_a in range(len(acts_a)):
            avg_acts_a = acts_a[layer_a][:,:,:,:].flatten(start_dim=1).detach().numpy()
            for layer_b in range(len(acts_b)):
                avg_acts_b = acts_b[layer_b][:,:,:,:].flatten(start_dim=1).detach().numpy()
                CKA_matrix[layer_a,layer_b] = np.mean((alternative_CKA(avg_acts_a,avg_acts_b),CKA_matrix[layer_a,layer_b]))

        batch_n = batch_n + 1
        if batch_n % 5 == 0:
            print("Batch ",batch_n," of ",len(test_loader))
            break

    matrices.append(CKA_matrix)

avg_CKA = sum(matrices)/len(matrices)
np.save((save_directory+"/matrices/"+save_name_a+"-"+save_name_b+".npy"),avg_CKA)



plt.rcParams['font.size'] = 20
plt.figure(figsize = (10,7))
sns.heatmap(avg_CKA)#, vmin=0, vmax=1)
plt.ylim(1,avg_CKA.shape[1]+1)
plt.xlim(1,avg_CKA.shape[0]+1)
# plt.title("CKA metric")
plt.xlabel(save_name_b)
plt.ylabel(save_name_a)
plt.savefig(save_directory+"/figures/"+save_name_a+"-"+save_name_b+".pdf", bbox_inches = 'tight')

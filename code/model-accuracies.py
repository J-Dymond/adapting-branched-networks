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
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from load_sat_data import *

#models
from ResNet import *
from DenseNet import *
from BranchedMobileNet import *

#some helper functions
def get_label(branch_weights):
    string = "("
    for weight in branch_weights:
        weight = np.round(weight,decimals=3)
        string = string + str(weight) + " : "
    string = string[:-3]+')'
    return(string)

# def get_save_name(directory,info):
#     model_name = info[1]
#     save_name = model_name
#     if 'branched' in model_name.lower():
#         weights = np.load(directory+"/metrics/branch-weights-"+model_name+".npy")
#         save_name = save_name + " " + get_label(weights)
#     return(save_name)

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
    elif model_name == 'BranchedMobileNet':
        model = BranchedMobileNet([0.25,0.5,0.75],num_classes=output_classes,input_channels=n_channels, in_size = 32)
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

parser.add_argument("target",help="path to model", type=str)
parser.add_argument("-r","--runs", help="Must be less than or equal to number of trained models.",type=int,default=3)
parser.add_argument("-b","--batch_size", help="Batch size for exit analysis.",type=int,default=128)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
parser.add_argument("-p","--partition", help="Which partition of the dataset to use: train,test,val",type=str,default='test')
# parser.add_argument("-t","--threshold", help="Confidence threshold for classification (max = 1.0)",type=float,default=0.8)
args = parser.parse_args()

model_directory  = args.target

# entropy_threshold = (1 - args.threshold)
entropy_thresholds = [0.0]#np.linspace(2.25,0,10)

target_info = model_directory.split('/')

# save_name = get_save_name(model_directory,target_info)

directory = target_info[0]+'/'+target_info[1]+'/'+target_info[2]+'/'

print('arguments passed:')
print("Target a: " + args.target)
print("Dataset: " + args.data)
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))
print("Entropy thresholds: " + str(entropy_thresholds))

save_directory = directory+"/metrics/"

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
    n_vals = {'train':50000,'val':10000,'test':10000}
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
    n_vals = {'train':50000,'val':10000,'test':10000}

if args.data == 'sat-6':

    # train_target = pd.read_csv("data/sat-6/y_train.csv")
    val_target = pd.read_csv("data/sat-6/y_test.csv")

    # train_len = train_target.shape[0]
    n_values = val_target.shape[0]

    # train_data = "data/sat-6/X_train_sat6.csv"
    val_data = "data/sat-6/X_test_sat6.csv"
    # train_data = "data/sat-6/train.csv"
    # val_data = "data/sat-6/test.csv"

    # Split training and validation data ===========================================
    # trainSet, valSet = train_test_split(trainval, test_size=0.20, random_state=42)

    transformTrain = transforms.Compose([
        transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.3), torchvision.transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),])
    # Use Dataset class ==========================================================
    # train_data = iterable_csv_dataset(train_data,train_len,transform=transformTrain)
    # test_data = iterable_csv_dataset(val_data,val_len,transform=None)

    # train_data = csv_loaded_data(train_target,train_data,train_len,transform=transformTrain)
    test_data = csv_loaded_data(val_target,val_data,n_values,transform=None)
    output_classes = 6
    n_channels = 4

if args.data == 'eurosat-rgb':
    transform = {
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

    data = ImageFolder('data/eurosat/jpeg/', transform=transform['test'])
    n_data = len(data)
    n_test = 5000

    train_data, test_data = torch.utils.data.random_split(data, [n_data-n_test, n_test])
    n_values = n_test
    output_classes = 10
    n_channels = 3

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size)
test_loader =  DataLoader(test_data, batch_size=args.batch_size)
val_loader =  DataLoader(test_data, batch_size=args.batch_size)

data_loaders = {
    'train':train_loader,
    'val':val_loader,
    'test':test_loader
}

dataloader = data_loaders[args.partition]
n_values = n_vals[args.partition]

#Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

softmax = nn.Softmax(dim=1)

try:
    # Create target Directory
    os.mkdir(model_directory+"/metrics/threshold-test/")
    print("Directory: " , model_directory+"/metrics/threshold-test/" ,  " Created ")
except FileExistsError:
    print("Directory: " , model_directory+"/metrics/threshold-test/" ,  " Created ")

for entropy_threshold in entropy_thresholds:

    accuracies = np.zeros(args.runs)
    threshold_exits = np.zeros((args.runs,n_values))

    for run in range(args.runs):
        print("Run: " + str(run))
        #defining models

        exits = np.zeros(n_values)
        predictions = np.zeros((4,n_values,output_classes))
        ground_truth = np.zeros(n_values)
        print('n_values:',n_values)
        model = get_model(model_directory,run,device)
        model.eval()

        input,target = next(iter(dataloader))
        input,target = input.to(device),target.to(device)
        with torch.no_grad():
            output = model(input)[0]
        output_shape = len(output)

        run_acc = list()
        run_exits = list()

        for batch_idx, (x, y) in enumerate(tqdm(dataloader,desc = args.partition+' batch')):
            x,y = x.to(device),y.to(device)
            #1-forward pass - get logites

            with torch.no_grad():
                l = model(x)[-1]

            #get softmaxed logits
            soft = []
            for out in l:
                soft.append(softmax(out).cpu().numpy())

            soft = np.array(soft)
            # print('soft',soft)

            #entropy on softmax to get conficenc
            confidence = Categorical(probs = torch.from_numpy(soft)).entropy().numpy()
            # print('confidence', confidence)
            n_outputs = confidence.shape[1]

            #exits
            ex_indices = np.zeros(n_outputs)

            for i in range(confidence.T.shape[0]):
                inp = confidence.T[i,:]
                possible_exits = np.where(inp<entropy_threshold,inp,inp[-1])
                ex = np.min(np.argwhere(inp==possible_exits))
                ex_indices[i] = ex

            #getting predictions using exits
            preds = torch.zeros(n_outputs)
            for input,exit in enumerate(ex_indices):
                preds[input] = np.argmax(soft[int(exit),input,:])

            # print(y)
            #recording accuracy using predictions
            batch_acc = sum(y.eq(preds))/preds.shape[0]
            run_acc.append(batch_acc.item())
            # print('batch acc',batch_acc)
            exits[batch_idx*args.batch_size:batch_idx*args.batch_size+n_outputs] = ex_indices
            run_exits.append(ex_indices)
            # print('exits',exits)
            predictions[:,batch_idx*args.batch_size:batch_idx*args.batch_size+n_outputs,:] = soft
            ground_truth[batch_idx*args.batch_size:batch_idx*args.batch_size+n_outputs] = y

        np.save((model_directory+"/metrics/exits-run-"+str(run)+".npy"),exits)
        # run_exits = np.concatenate(run_exits)
        # print('run_exits',run_exits)
        # print('run_acc:',run_acc)
        accuracies[run] = np.sum(run_acc)/len(run_acc)
        threshold_exits[run,:] = exits
        # print('threshold exits:',threshold_exits)

        np.save((model_directory+"/metrics/"+args.partition+"-raw-softmax-run-"+str(run)+".npy"),predictions)
        np.save((model_directory+"/metrics/"+args.partition+"-ground-truth-run-"+str(run)+".npy"),ground_truth)
        print("raw softmax: ",predictions.shape)
        print("ground truth: ",ground_truth.shape)
    entropy_threshold = round(entropy_threshold,5)
    print("Threshold: ", entropy_threshold)
    print("Accuracies: ", accuracies)

    np.save((model_directory+"/metrics/threshold-test/accuracies-threshold-"+str(entropy_threshold).strip('.')+".npy"),accuracies)
    np.save((model_directory+"/metrics/threshold-test/exits-threshold-"+str(entropy_threshold).strip('.')+".npy"),threshold_exits)

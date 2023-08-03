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

from ResNet import *
from DenseNet import *

#Arguments for running
parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Directory to run class separation code in. Format: '../trained-models/*model*/Runx/'", type=str)
parser.add_argument("-m","--model", help="Backbone architecture to be used",type=str,default='ResNet18')
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do. Must be less than or equal to number of trained models.",type=int,default=3)
parser.add_argument("-b","--batch_size", help="Batch size for class separation.",type=int,default=128)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

print('arguments passed:'+args.target_directory)
print("Target")
print("Architecture: " + args.model)
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))

directory = args.target_directory

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(directory+"confusion-analysis")
    print("Directory " , directory+"/confusion-analysis/" ,  " Created ")
except FileExistsError:
    print("Directory " ,directory+"/confusion-analysis/" ,  " already exists")

if 'branched' in args.model.lower():
    branch_weights = np.load(directory+"metrics/branch-weights-"+args.model+".npy")
    print(branch_weights)



save_directory = directory+"/confusion-analysis/"
model_directory = directory+"/saved-models/"

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

print("Labels: ",label_names)

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)
test_loader =   DataLoader(test_data, batch_size=args.batch_size)

matrices = list()

for run in range(args.runs):
    print("Run: " + str(run))

    #defining model

    if args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'ResNet50':
        model = ResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    elif args.model == 'BranchedResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'BranchedResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'BranchedResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    else:
        model = ResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18

    #putting to GPU and loading weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_name = "best-"+args.model+'-CIFAR-10-'+str(run)+'.pth'
    model.load_state_dict(torch.load(model_directory + model_name, map_location=device))

    #getting embeddings
    n_epochs = 1

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    output = model(input)

    n_classifiers = len(output[-1])

    n_targets = output[-1][-1][0].shape[0] #Gets the output of the model, since it is returned as a list to account for branched networks


    print("Number of classifiers: " + str(n_classifiers))
    print("Number of targets: " + str(n_targets))

    loss = nn.CrossEntropyLoss()

    model.eval()
    for classifier in range(n_classifiers):
        matrices.append(np.zeros((n_targets, n_targets)))

    # Initialize the prediction and label lists(tensors)
    predlist=np.zeros(0)
    lbllist=np.zeros(0)

    for batch in test_loader:

        x,y = batch
        x,y = x.to(device),y.to(device)
        #1-forward pass - get logites
        with torch.no_grad():
            l = model(x)[-1]
        for classifier in range(n_classifiers):
            _, preds = torch.max(l[classifier], 1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                matrices[run*n_classifiers + classifier][t.long(), p.long()] += 1

#get average confusion matrix across all runs
averages_matrices = np.zeros((n_targets,n_targets,n_classifiers))
for run in range(args.runs):
    for classifier in range(n_classifiers):
        matrix = matrices[run*n_classifiers + classifier]
        averages_matrices[:,:,classifier] = averages_matrices[:,:,classifier] + matrix
averages_matrices = np.rint(averages_matrices/args.runs)
averages_matrices = averages_matrices.astype(int)

diff = averages_matrices[:,:,-1] - averages_matrices[:,:,0]
min = np.min(diff)
max = np.max(diff)
#plot confusion matrices and save them
for classifier in range(n_classifiers):
    final_cm = pd.DataFrame(averages_matrices[:,:,-1], index = [label for label in label_names],
          columns = [label for label in label_names])
    df_cm = pd.DataFrame(averages_matrices[:,:,classifier], index = [label for label in label_names],
          columns = [label for label in label_names])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, vmin=0, vmax=1.1*(n_values/n_targets), annot=annotate, fmt='d')
    plt.title("Classification confusion matrix at exit:"+str(classifier+1))
    plt.savefig(save_directory+'/exit'+str(classifier+1), bbox_inches = 'tight')

    plt.figure(figsize = (10,7))
    sns.heatmap(final_cm-df_cm, vmin=min, vmax=max, annot=annotate, fmt='d')
    plt.title("Difference to final exit confusion matrix at exit:"+str(classifier+1))
    plt.savefig(save_directory+'/diff-exit'+str(classifier+1), bbox_inches = 'tight')

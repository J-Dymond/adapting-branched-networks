#Importing modules
import os
import argparse
import sys

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from ResNet import *
from DenseNet import *
from load_sat_data import *

#Arguments for running
parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Directory to run class separation code in. Format: 'trained-models/*model*/Runx/'", type=str)
parser.add_argument("-m","--model", help="Backbone architecture to be used",type=str,default='ResNet18')
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do. Must be less than or equal to number of trained models.",type=int,default=3)
parser.add_argument("-b","--batch_size", help="Batch size for training.",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for probe training",type=float,default=1e-2)
parser.add_argument("-e","--epochs",help="Number of epochs for linear probe training",type=int,default=10)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

print('arguments passed:'+args.target_directory)
print("Target")
print("Architecture: " + args.model)
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))
print("Learning Rate: "+str(args.learning_rate))

directory = args.target_directory

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(directory+"/linear-probe-values/")
    print("Directory: " , directory+"/linear-probe-values/" ,  " Created ")
    save_directory = directory+"/linear-probe-values/"
except FileExistsError:
    print("Directory: " ,directory+"/linear-probe-values/" ,  " already exists")
    save_directory = directory+"/linear-probe-values/"

if args.data == "CIFAR100-H":
    try:
        # Create target Directory
        os.mkdir(directory+"/linear-probe-values/hierarchical/")
        print("Directory: " , directory+"/linear-probe-values/hierarchical/" ,  " Created ")
        save_directory = directory+"/linear-probe-values/hierarchical/"
    except FileExistsError:
        print("Directory: " ,directory+"/linear-probe-values/hierarchical/" ,  " already exists")
        save_directory = directory+"/linear-probe-values/hierarchical/"



try:
    # Create target Directory
    os.mkdir(save_directory+"/accuracy/")
    print("Directory: " , save_directory+"/accuracy/" ,  " Created ")
except FileExistsError:
    print("Directory: " , save_directory+"/accuracy/" ,  " already exists")

try:
    # Create target Directory
    os.mkdir(save_directory+"/loss/")
    print("Directory: " , save_directory+"/loss/" ,  " Created ")
except FileExistsError:
    print("Directory: " , save_directory+"/loss/" ,  " already exists")


model_directory = directory+"/saved-models/"
metric_directory = directory+"/metrics/"

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
    probe_classes = 10
    n_channels = 1
    hierarchical = False

if args.data == 'CIFAR10':
    train_data = datasets.CIFAR10('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform = transform_test)
    output_classes = 10
    probe_classes = 10
    n_channels = 3
    hierarchical = False

if args.data == 'CIFAR100':
    train_data = datasets.CIFAR100('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR100('data', train=False, download=True, transform = transform_test)
    output_classes = 100
    coarse_classes = 20
    probe_classes = 100
    n_channels = 3
    hierarchical = False

if args.data == 'sat-6':

    train_target = pd.read_csv("data/sat-6/y_train.csv")
    val_target = pd.read_csv("data/sat-6/y_test.csv")

    train_len = train_target.shape[0]
    val_len = val_target.shape[0]

    train_data = "data/sat-6/X_train_sat6.csv"
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

    train_data = csv_loaded_data(train_target,train_data,train_len,transform=transformTrain)
    test_data = csv_loaded_data(val_target,val_data,val_len,transform=None)
    output_classes = 6
    probe_classes = 6
    n_channels = 4
    hierarchical = False

if args.data == 'CIFAR100-H':
    train_data = datasets.CIFAR100('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR100('data', train=False, download=True, transform = transform_test)
    meta_data = np.load('data/cifar-100-python/meta',allow_pickle=True)
    fine_label_names = meta_data['fine_label_names']
    coarse_label_names = meta_data['coarse_label_names']
    output_classes = 100
    coarse_classes = 20
    probe_classes = 20
    n_channels = 3
    #Get jierarchical classes
    class Dictlist(dict):
        def __setitem__(self, key, value):
            try:
                self[key]
            except KeyError:
                super(Dictlist, self).__setitem__(key, [])
            self[key].append(value)
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_coarse(fine_labels,assiginments):
        coarse_targets = torch.zeros(fine_labels.shape,dtype=int)
        for i in range(fine_labels.shape[0]):
            for key,item in assiginments.items():
                if fine_labels[i] in item:
                    coarse_targets[i] = key
        return(coarse_targets)

    #Set hierarchical flag to true and obtain label label_assiginments
    hierarchical = True
    #Getting coarse label dictionary
    x=unpickle('data/cifar-100-python/test')
    fine_to_coarse=Dictlist()
    for i in range(0,len(x[b'coarse_labels'])):
        fine_to_coarse[x[b'coarse_labels'][i]]=x[ b'fine_labels'][i]
    label_assiginments=dict(fine_to_coarse)
    for i in label_assiginments.keys():
        label_assiginments[i]=list(dict.fromkeys(label_assiginments[i]))

    #printing label assignments
    print('\nLabel names:\n')
    for index in label_assiginments:
        print(coarse_label_names[index],':',[fine_label_names[i] for i in label_assiginments[index]])
    print("\n\nTargets:\n")
    for index in label_assiginments:
        print(index,':\t',label_assiginments[index])

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)

for run in range(args.runs):
    print("Run: " + str(run))

    #defining model

    if args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'ResNet10':
        model = ResNet(BasicBlock, [1,1,1,1],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'ResNet50':
        model = ResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    elif args.model == 'DenseNet18':
            model = DenseNet3(depth=18,num_classes=output_classes,input_channels=n_channels) # DenseNet121
    elif args.model == 'DenseNet50':
            model = DenseNet3(depth=50,num_classes=output_classes,input_channels=n_channels) # DenseNet121
    elif args.model == 'DenseNet121':
            model = DenseNet3(depth=121,num_classes=output_classes,input_channels=n_channels) # DenseNet121

    elif args.model == 'BranchedResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # BranchedResnet18
    elif args.model == 'BranchedResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # BranchedResnet34
    elif args.model == 'BranchedResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # BranchedResnet50
    elif args.model == 'BranchedResNet10':
        model = BranchedResNet(BasicBlock, [1,1,1,1],num_classes=output_classes,input_channels=n_channels) # BranchedResnet11
    elif args.model == 'BranchedDenseNet121':
        model = model = BranchedDenseNet3(depth=121,num_classes=output_classes,input_channels=n_channels)
    elif args.model == 'BranchedDenseNet50':
        model = model = BranchedDenseNet3(depth=50,num_classes=output_classes,input_channels=n_channels)
    elif args.model == 'BranchedDenseNet18':
        model = model = BranchedDenseNet3(depth=18,num_classes=output_classes,input_channels=n_channels)
    else:
        sys.exit('please specify a valid model')

    if os.path.exists(metric_directory+"hierarchical_branches.npy") == True:
        print("Hierarchical branches detected..")
        hierarchical_branches = np.load(metric_directory+"hierarchical_branches.npy")
        print("Changing branches: ")
        if hierarchical_branches[0] == 1:
            print("Branch 1")
            input_features = model.branch_layer1.in_features
            model.branch_layer1 = nn.Linear(input_features, coarse_classes)
        if hierarchical_branches[1] == 1:
            print("Branch 2")
            input_features = model.branch_layer2.in_features
            model.branch_layer2 = nn.Linear(input_features, coarse_classes)
        if hierarchical_branches[2] == 1:
            print("Branch 3")
            input_features = model.branch_layer3.in_features
            model.branch_layer3 = nn.Linear(input_features, coarse_classes)
        if hierarchical_branches[3] == 1:
            print("Final Exit")
            input_features = model.linear.in_features
            model.linear = nn.Linear(input_features, coarse_classes)

        print("Output layers:\nExit 1 :",model.branch_layer1,"\nExit 2 :",model.branch_layer2,"\nExit 3 :",model.branch_layer3,"\nExit 4 :",model.linear)

    #putting to GPU and loading weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_name = "best-"+args.model+'-CIFAR-10-'+str(run)+'.pth'
    model.load_state_dict(torch.load(model_directory + model_name, map_location=device))


    #Defining linear probe class
    class LinearClassifier(torch.nn.Module):
      def __init__(self, input_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, probe_classes)
      def forward(self,x):
        return F.dropout(self.linear(x),p=0.15)

    #Get model size
    n_epochs = 1

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    embeddings = model(input)[0][:]
    output_shape = len(embeddings)

    n_layers = output_shape

    #Generate classifiers
    classifiers = []

    for i in range (0, n_layers):
        classifier_dimension = torch.flatten(embeddings[i][0]).shape[0]
        classifiers.append(LinearClassifier(classifier_dimension))


    #Train classifiers

    linear_probes = []
    linear_probes_loss = []

    n_epochs = args.epochs
    loss = nn.CrossEntropyLoss()

    for layer in range(n_layers):

      print('\nLayer: '+str(layer+1))

      classifier = classifiers[layer]
      params = classifier.parameters()
      classifier.to(device)
      optimiser = optim.SGD(params,lr=args.learning_rate,momentum=0.9)

      for epoch in range(n_epochs):
        print('\nEpoch: '+str(epoch+1))

        losses = list()
        accuracies = list()

        classifier.train()
        for batch in train_loader:

            x,y = batch
            if hierarchical:
                y = get_coarse(y,label_assiginments)
            x,y = x.to(device),y.to(device)

            with torch.no_grad():
              batch_embeddings = model(x)[0]

            layer_input = torch.flatten(batch_embeddings[layer],start_dim=1)
            l = classifier(layer_input)

            #2-objective function
            J = loss(l,y)

            #3-clean gradients
            classifier.zero_grad()

            #4-accumulate partial derivatives of J
            J.backward()

            #5-step in opposite direction of gradient
            optimiser.step()
            break

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
        print('Training:')
        print(f'Loss: {torch.tensor(losses).mean():.2f}', end='\n')
        print(f'Accuracy: {torch.tensor(accuracies).mean():.2f}')

        #Reset losses and accuracies for validation
        val_losses = list()
        val_accuracies = list()

        classifier.eval()

        for batch in val_loader:

            x,y = batch
            if hierarchical:
                y = get_coarse(y,label_assiginments)
            x,y = x.to(device),y.to(device)

            with torch.no_grad():
              batch_embeddings = model(x)[0]

            layer_input = torch.flatten(batch_embeddings[layer],start_dim=1)
            with torch.no_grad():
              l = classifier(layer_input)

            #2-objective function
            J = loss(l,y)

            val_losses.append(J.item())
            val_accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

        val_loss = torch.tensor(val_losses).mean()
        val_acc = torch.tensor(val_accuracies).mean()

        print('Validation:')
        print(f'Loss: {val_loss:.2f}', end='\n')
        print(f'Accuracy: {val_acc:.2f}')

      print()

      linear_probes.append(val_acc.detach().cpu().numpy())
      linear_probes_loss.append(val_loss.detach().cpu().numpy())

    linear_probes = np.array(linear_probes)
    linear_probes_loss = np.array(linear_probes_loss)

    save_string = save_directory+'/accuracy/'+model_name+'.npy'
    np.save(save_string, linear_probes)

    save_string = save_directory+'/loss/'+model_name+'.npy'
    np.save(save_string, linear_probes_loss)

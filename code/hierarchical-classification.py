#Importing modules
import os
import torch
from torch import nn
# from torch import linalg
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from ResNet import *
from DenseNet import *

#arguments for running

parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Name of directory to save data", type=str)
parser.add_argument("-m","--model", help="Architecture to be used",type=str,default='BranchedResNet18')
parser.add_argument("-w","--branch_weightings", nargs="+",
    help="How to weight the branch losses. Format: a b c d -> a+b+c+d should equal 1.0. Default:[0.2,0.0,0.3,0.5]. Use 0.0 to deactivate branch.",
    type=float,default=[0.2,0.0,0.3,0.5])
parser.add_argument("-hb","--hierarchical_branches", nargs="+",
    help="Which branches to use hierarchical labels with, 1 for coarse, 0 for fine labels. Format: a b c d Default:[1,1,0,0].",
    type=int,default=[1,1,0,0])
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do",type=int,default=3)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR100')
parser.add_argument("-e","--epochs", help="Number of epochs to run experiment for",type=int,default=200)
parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=float,default=1e-2)
args = parser.parse_args()

print('Arguments passed:')
print("target_directory: " + args.target_directory)
print("Architecture: " + args.model)
print("Weights: ", args.branch_weightings)
print("Hierarchical branches: ", args.hierarchical_branches)
print("Dataset: " + args.data)
print("Epochs: ", args.epochs)
print("Runs: ",args.runs)
print("Batch size: ", args.batch_size)
print("Learning Rate: ", args.learning_rate)

try:
    # Create target Directory
    os.mkdir("trained-models/"+args.model)
    print("Directory: " , "trained-models/"+args.model ,  " Created ")
except FileExistsError:
    print("Directory: " , "trained-models/"+args.model ,  " already exists")

try:
    # Create dataset Directory
    os.mkdir("trained-models/"+args.model+"/"+args.data+"/hierarchical/")
    print("Directory: " , "trained-models/"+args.model+"/"+args.data+"/hierarchical/" ,  " Created ")
except FileExistsError:
    print("Directory: " , "trained-models/"+args.model+"/"+args.data+"/hierarchical/" ,  " already exists")

try:
    save_directory = "trained-models/"+args.model+"/"+args.data+"/hierarchical/"+args.target_directory+"/"
    os.mkdir(save_directory)
    print("Saving data to: " , save_directory)

except FileExistsError:
    print(save_directory, "Already exists...")
    for run in range(100):
        try:
            save_directory = "trained-models/"+args.model+"/"+args.data+"/hierarchical/"+args.target_directory+'_'+str(run)+"/"
            os.mkdir(save_directory)
            print("Instead saving data to: " , save_directory)
            break

        except FileExistsError:
            continue

metric_directory = save_directory+"/metrics/"
os.mkdir(metric_directory)

model_directory = save_directory+"/saved-models/"
os.mkdir(model_directory)

gradient_directory = save_directory+"/gradient-values/"
os.mkdir(gradient_directory)

for run in range(args.runs):
    gradient_run_directory = save_directory+"/gradient-values/Run-"+str(run)
    os.mkdir(gradient_run_directory)


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

if args.data == 'CIFAR100':
    train_data = datasets.CIFAR100('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR100('data', train=False, download=True, transform = transform_test)
    coarse_classes = 20
    output_classes = 100
    n_channels = 3
    meta_data = np.load('data/cifar-100-python/meta',allow_pickle=True)
    fine_label_names = meta_data['fine_label_names']
    coarse_label_names = meta_data['coarse_label_names']

#Getting coarse label dictionary
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

def get_coarse(fine_labels):
    coarse_targets = torch.zeros(fine_labels.shape,dtype=int)
    for i in range(fine_labels.shape[0]):
        for key,item in label_assiginments.items():
            if fine_labels[i] in item:
                coarse_targets[i] = key
    return(coarse_targets)


# print("\nIn trainloader:\n")
# # train, val = random_split(train_data, [45000,5000])
#
# x,y = next(iter(train_loader))
# print("Fine grained targets:", y)
# coarse_targets = get_coarse(y)
# print("Coarse targets:", coarse_targets)

train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)

branch_weights = args.branch_weightings
hierarchical_branches = args.hierarchical_branches

np.save(metric_directory+'hierarchical_branches.npy',hierarchical_branches)

train_losses = np.zeros((args.runs,args.epochs))
val_losses = np.zeros((args.runs,args.epochs))

branch_train_accs = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_val_accs = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_train_losses = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_val_losses = np.zeros((args.runs,len(branch_weights),args.epochs))

checkpoint_metrics = np.zeros((args.runs,3))

for run in range(args.runs):

    print("\nRun:" + str(run+1))

    if args.model == 'ResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'ResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'ResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    else:
        model = BranchedResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18

    if hierarchical_branches[0] == 1:
        input_features = model.branch_layer1.in_features
        model.branch_layer1 = nn.Linear(input_features, coarse_classes)
    if hierarchical_branches[1] == 1:
        input_features = model.branch_layer2.in_features
        model.branch_layer2 = nn.Linear(input_features, coarse_classes)
    if hierarchical_branches[2] == 1:
        input_features = model.branch_layer3.in_features
        model.branch_layer3 = nn.Linear(input_features, coarse_classes)
    if hierarchical_branches[3] == 1:
        input_features = model.linear.in_features
        model.linear = nn.Linear(input_features, coarse_classes)

    print("Output layers:\nExit 1 :",model.branch_layer1,"\nExit 2 :",model.branch_layer2,"\nExit 3 :",model.branch_layer3,"\nExit 4 :",model.linear)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    output = model(input)
    output_shape = len(output[0])
    n_layers = output_shape
    n_targets = output[-1][0][0].shape[0]

    #Number of convolutional layers to be tracked
    n_conv_layers = 0
    for name, param in model.named_parameters():
      if 'conv' in name:
        n_conv_layers = n_conv_layers + 1

    #Defining optimiser and scheduler
    params = model.parameters()
    optimiser = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimiser = optim.Adam(params, lr=args.learning_rate, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=200)
    loss = nn.CrossEntropyLoss()

    #Training

    best_accuracy = -1.0
    nb_epochs = args.epochs

    epoch_gradients_std = list()
    epoch_gradients_mean = list()
    epoch_gradients_abs_mean = list()

    epoch_weights_std = list()
    epoch_weights_mean = list()
    epoch_weights_abs_mean = list()

    for epoch in range(nb_epochs):
        print("\n\nEpoch:",epoch)
        #track loss and accuracy
        losses = list()
        accuracies = list()

        #and for the branches
        branch_losses = list()
        for i in range(len(branch_weights)):
          branch_losses.append(list())
        branch_accuracies = list()
        for i in range(len(branch_weights)):
          branch_accuracies.append(list())

        model.train() # because of dropout

        #Values for tracking
        batch_gradients_std = torch.zeros(n_conv_layers)
        batch_gradients_mean = torch.zeros(n_conv_layers)
        batch_gradients_abs_mean = torch.zeros(n_conv_layers)

        batch_weights_std = torch.zeros(n_conv_layers)
        batch_weights_mean = torch.zeros(n_conv_layers)
        batch_weights_abs_mean = torch.zeros(n_conv_layers)
        print("Training")
        for batch in train_loader:
            x,y = batch
            y_coarse = get_coarse(y)
            x,y,y_coarse = x.to(device),y.to(device),y_coarse.to(device)

            target = [y,y_coarse]

            #1-forward pass - get logits from all exit branches
            l = model(x)[-1]

            E1 = l[0]
            E2 = l[1]
            E3 = l[2]
            E4 = l[3]

            #losses, all exits have same target
            L1 = loss(E1,target[hierarchical_branches[0]])
            L2 = loss(E2,target[hierarchical_branches[1]])
            L3 = loss(E3,target[hierarchical_branches[2]])
            L4 = loss(E4,target[hierarchical_branches[3]])

            #2-objective function
            J = branch_weights[0]*L1 + branch_weights[1]*L2 + branch_weights[2]*L3 + branch_weights[3]*L4

            #3-clean gradients
            model.zero_grad()

            #4-accumulate partial derivatives of J
            J.backward()

            layer = 0
            for name, param in model.named_parameters():
              if "conv" in name:
                flattened_gradients = torch.flatten(param.grad)
                flattened_weights = torch.flatten(param)

                batch_gradients_std[layer] = batch_gradients_std[layer] + torch.std(flattened_gradients)
                batch_gradients_mean[layer] = batch_gradients_mean[layer] + torch.mean(flattened_gradients)
                batch_gradients_abs_mean[layer] = batch_gradients_abs_mean[layer] + torch.mean(torch.abs(flattened_gradients))

                batch_weights_std[layer] = batch_weights_std[layer] + torch.std(flattened_weights)
                batch_weights_mean[layer] = batch_weights_mean[layer] + torch.mean(flattened_weights)
                batch_weights_abs_mean[layer] = batch_weights_abs_mean[layer] + torch.mean(torch.abs(flattened_weights))

            #5-step in opposite direction of gradient
            optimiser.step()

            #6-record losses
            losses.append(J.item())
            accuracies.append(y.eq(E3.detach().argmax(dim=1)).float().mean())

            #also for branches
            branch_losses[0].append(L1.item())
            branch_losses[1].append(L2.item())
            branch_losses[2].append(L3.item())
            branch_losses[3].append(L4.item())

            branch_accuracies[0].append(y.eq(E1.detach().argmax(dim=1)).float().mean())
            branch_accuracies[1].append(y.eq(E2.detach().argmax(dim=1)).float().mean())
            branch_accuracies[2].append(y.eq(E3.detach().argmax(dim=1)).float().mean())
            branch_accuracies[3].append(y.eq(E4.detach().argmax(dim=1)).float().mean())
            break


        #Recording gradients
        epoch_gradient_std = batch_gradients_std/len(train_loader)
        epoch_gradients_std.append(np.array(epoch_gradient_std.detach().cpu().numpy()))

        epoch_gradient_mean = batch_gradients_mean/len(train_loader)
        epoch_gradients_mean.append(np.array(epoch_gradient_mean.detach().cpu().numpy()))

        epoch_gradient_abs_mean = batch_gradients_std/len(train_loader)
        epoch_gradients_abs_mean.append(np.array(epoch_gradient_abs_mean.detach().cpu().numpy()))

        #Recording weights
        epoch_weight_std = batch_weights_std/len(train_loader)
        epoch_weights_std.append(np.array(epoch_weight_std.detach().cpu().numpy()))

        epoch_weight_mean = batch_weights_mean/len(train_loader)
        epoch_weights_mean.append(np.array(epoch_weight_mean.detach().cpu().numpy()))

        epoch_weight_abs_mean = batch_weights_std/len(train_loader)
        epoch_weights_abs_mean.append(np.array(epoch_weight_abs_mean.detach().cpu().numpy()))


        train_losses[run,epoch] = torch.tensor(losses).mean()

        print(f'Total Loss: {train_losses[run,epoch]:.2f}', end='\n')
        print('Exit Losses: ')
        for i in range(len(branch_weights)):
            branch_train_losses[run,i,epoch] = torch.tensor(branch_losses[i]).mean()
            print(f'Exit {(i+1)}: {branch_train_losses[run,i,epoch]:.2f}', end=', ')
        print('\nExit accuracies: ')
        for i in range(len(branch_weights)):
            branch_train_accs[run,i,epoch] = torch.tensor(branch_accuracies[i]).mean()
            print(f'Exit {(i+1)}: {branch_train_accs[run,i,epoch]:.2f}', end=', ')

        #Reset losses
        losses = list()
        accuracies = list()
        #and for the branches
        branch_losses = list()
        for i in range(len(branch_weights)):
          branch_losses.append(list())
        branch_accuracies = list()
        for i in range(len(branch_weights)):
          branch_accuracies.append(list())
        model.eval()

        print('\nValidation:')
        for batch in val_loader:

            x,y = batch
            y_coarse = get_coarse(y)
            x,y,y_coarse = x.to(device),y.to(device),y_coarse.to(device)

            target = [y,y_coarse]

            #1-forward pass - get logits from all exit branches
            with torch.no_grad():
                l = model(x)[-1]

                E1 = l[0]
                E2 = l[1]
                E3 = l[2]
                E4 = l[3]

            #losses, all exits have same target
            L1 = loss(E1,target[hierarchical_branches[0]])
            L2 = loss(E2,target[hierarchical_branches[1]])
            L3 = loss(E3,target[hierarchical_branches[2]])
            L4 = loss(E4,target[hierarchical_branches[3]])

            #2-objective function
            J = branch_weights[0]*L1 + branch_weights[1]*L2 + branch_weights[2]*L3 + branch_weights[3]*L4

            losses.append(J.item())

            #also for branches
            branch_losses[0].append(L1.item())
            branch_losses[1].append(L2.item())
            branch_losses[2].append(L3.item())
            branch_losses[3].append(L4.item())

            branch_accuracies[0].append(y.eq(E1.detach().argmax(dim=1)).float().mean())
            branch_accuracies[1].append(y.eq(E2.detach().argmax(dim=1)).float().mean())
            branch_accuracies[2].append(y.eq(E3.detach().argmax(dim=1)).float().mean())
            branch_accuracies[3].append(y.eq(E4.detach().argmax(dim=1)).float().mean())
            break


        val_losses[run,epoch] = torch.tensor(losses).mean()

        print(f'Total Loss: {val_losses[run,epoch]:.2f}', end='\n')
        print('Exit Losses: ')
        for i in range(len(branch_weights)):
            branch_val_losses[run,i,epoch] = torch.tensor(branch_losses[i]).mean()
            print(f'Exit {(i+1)}: {branch_val_losses[run,i,epoch]:.2f}', end=', ')
        print('\nExit accuracies: ')
        for i in range(len(branch_weights)):
            branch_val_accs[run,i,epoch] = torch.tensor(branch_accuracies[i]).mean()
            print(f'Exit {(i+1)}: {branch_val_accs[run,i,epoch]:.2f}', end=', ')

        val_acc = branch_val_accs[run,-1,epoch]

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_directory + 'best-BranchedResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')
            best_accuracy = val_acc

        # scheduler.step()

    #Saving model
    n_epochs = 1

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    output = model(input)
    output_shape = len(output[0])
    n_layers = output_shape
    n_targets = output[-1][0][0].shape[0]

    #Saving gradient values
    numpy_epoch_gradients_std = np.array(epoch_gradients_std)
    save_string = gradient_directory+'Run-'+str(run)+'/std-gradients.npy'
    np.save(save_string, numpy_epoch_gradients_std)

    numpy_epoch_gradients_mean = np.array(epoch_gradients_mean)
    save_string = gradient_directory+'Run-'+str(run)+'/mean-gradients.npy'
    np.save(save_string, numpy_epoch_gradients_mean)

    numpy_epoch_gradients_abs_mean = np.array(epoch_gradients_abs_mean)
    save_string = gradient_directory+'Run-'+str(run)+'/abs-mean-gradients.npy'
    np.save(save_string, numpy_epoch_gradients_abs_mean)

    #Saving Weight Values
    numpy_epoch_weights_std = np.array(epoch_weights_std)
    save_string = gradient_directory+'Run-'+str(run)+'/std-weights.npy'
    np.save(save_string, numpy_epoch_weights_std)

    numpy_epoch_weights_mean = np.array(epoch_weights_mean)
    save_string = gradient_directory+'Run-'+str(run)+'/mean-weights.npy'
    np.save(save_string, numpy_epoch_weights_mean)

    numpy_epoch_weights_abs_mean = np.array(epoch_weights_abs_mean)
    save_string = gradient_directory+'Run-'+str(run)+'/abs-mean-weights.npy'
    np.save(save_string, numpy_epoch_weights_abs_mean)

    #Saving model
    print("\n\nSaving model..")
    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))

    torch.save(model.state_dict(), model_directory + 'epoch'+str(epoch)+'-BranchedResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')

    checkpoint_metrics[run,:] = np.array([epoch,best_accuracy,optimiser.param_groups[0]['lr']])

#Saving all tracked metrics for analysis

save_string = metric_directory+'val-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, val_losses)

save_string = metric_directory+'train-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, train_losses)

save_string = metric_directory+'branch-val-accuracies-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_val_accs)

save_string = metric_directory+'branch-train-accuracies-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_train_accs)

save_string = metric_directory+'branch-train-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_train_losses)

save_string = metric_directory+'branch-val-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_val_losses)

save_string = metric_directory+'checkpoint-metrics-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, checkpoint_metrics)

save_string = metric_directory+'branch-weights-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_weights)

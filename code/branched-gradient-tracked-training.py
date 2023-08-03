#Importing modules

import os
import argparse
import sys

import torch
from torch import nn
from torch import optim
# from torch import linalg
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from ResNet import *
from DenseNet import *
from load_sat_data import *
from BranchedMobileNet import *
from label_smoothing import *
from diversity_loss import diversity_loss

#Arguments for running
parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Name of directory to save data", type=str)
parser.add_argument("-m","--model", help="Backbone architecture to be used",type=str,default='BranchedResNet18')
parser.add_argument("-w","--branch_weightings", nargs="+",
    help="How to weight the branch losses. Format: a b c d -> a+b+c+d should equal 1.0. Default:[0.2,0.2,0.2,0.4]. Use 0.0 to deactivate branch.",
    type=float,default=[0.2,0.2,0.2,0.4])
parser.add_argument("-cw","--class_weightings",help="file path to numpy array for class weightings")
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do",type=int,default=3)
parser.add_argument("-e","--epochs", help="Number of epochs to run experiment for",type=int,default=200)
parser.add_argument("-s","--scheduler_epochs", help="How many epochs to loop the scheduler over",type=int,default=50)
parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=float,default=1e-2)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

print('arguments passed:')
print("target_directory: " + args.target_directory)
print("Architecture: " + args.model)
print("Branch Weightings: " + str(args.branch_weightings))
print("Epochs: " + str(args.epochs))
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))

#Prepare directories for saving

try:
    # Create dataset Directory
    os.mkdir("trained-models/"+args.model+"/"+args.data)
    print("Directory: " , "trained-models/"+args.model+"/"+args.data ,  " Created ")
except FileExistsError:
    print("Directory: " , "trained-models/"+args.model+"/"+args.data ,  " already exists")

try:
    save_directory = "trained-models/"+args.model+"/"+args.data+"/"+args.target_directory+"/"
    os.mkdir(save_directory)
    print("Saving data to: " , save_directory)

except FileExistsError:
    print(save_directory, "Already exists...")
    for run in range(1,100):
        try:
            save_directory = "trained-models/"+args.model+"/"+args.data+"/"+args.target_directory+'_'+str(run)+"/"
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
if args.data == 'CIFAR10':
    train_data = datasets.CIFAR10('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform = transform_test)
    output_classes = 10
    n_channels = 3
if args.data == 'CIFAR100':
    train_data = datasets.CIFAR100('data', train=True, download=True, transform = transform_train)
    test_data = datasets.CIFAR100('data', train=False, download=True, transform = transform_test)
    output_classes = 100
    n_channels = 3

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
    output_classes = 10
    n_channels = 3

if args.data == 'eurosat-full':
    transform = {
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

    data = ImageFolder('data/eurosat/tiff/', transform=transform['test'])
    n_data = len(data)
    n_test = 5000

    train_data, test_data = torch.utils.data.random_split(data, [n_data-n_test, n_test])
    output_classes = 10
    n_channels = 13

if args.class_weightings is None:
    class_weights = torch.ones(output_classes)
else:
    class_weights_numpy = np.float32(np.load(args.class_weightings))
    class_weights = torch.from_numpy(class_weights_numpy)

print("Class weighting: ",class_weights)

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size,drop_last=True)
val_loader =   DataLoader(test_data, batch_size=args.batch_size,drop_last=True)

branch_weights = args.branch_weightings

train_losses = np.zeros((args.runs,args.epochs))
val_losses = np.zeros((args.runs,args.epochs))

branch_train_accs = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_val_accs = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_train_losses = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_val_losses = np.zeros((args.runs,len(branch_weights),args.epochs))

checkpoint_metrics = np.zeros((args.runs,3))

for run in range(args.runs):

    print("\nRun:" + str(run+1))

    if args.model == 'BranchedResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'BranchedResNet10':
        model = BranchedResNet(BasicBlock, [1,1,1,1],num_classes=output_classes,input_channels=n_channels) # Resnet10
    elif args.model == 'BranchedResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'BranchedResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    elif args.model == 'BranchedDenseNet121':
        model = BranchedDenseNet3(depth=121,num_classes=output_classes,input_channels=n_channels)
    elif args.model == 'BranchedDenseNet50':
        model = BranchedDenseNet3(depth=50,num_classes=output_classes,input_channels=n_channels)
    elif args.model == 'BranchedDenseNet18':
        model = BranchedDenseNet3(depth=18,num_classes=output_classes,input_channels=n_channels)
    elif args.model == 'BranchedMobileNet':
        model = BranchedMobileNet([0.25,0.5,0.75],num_classes=output_classes,input_channels=n_channels, in_size = 32)
    else:
        sys.exit("Please specify a valid architecture")     # Resnet18

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

    print("Number of convolutional layers: " + str(n_conv_layers))

    #Defining optimiser and scheduler
    params = model.parameters()
    optimiser = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimiser = optim.Adam(params, lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=args.scheduler_epochs)
    # loss = nn.CrossEntropyLoss()
    # branch_loss = nn.CrossEntropyLoss(class_weights).to(device)

    loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    branch_loss = LabelSmoothingCrossEntropy(epsilon=0.3).to(device)


    class_weights.to(device)

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

        for batch_idx, (x, y) in enumerate(train_loader):
            # if batch_idx%100 == 0:
            #     print("\n"+str(batch_idx)+"/"+str(len(train_loader)))
            x,y = x.to(device),y.to(device)
            #x: b x 1 x 28 x 28
            # b = x.size(0)
            # x = x.view(b,-1)
            #1-forward pass - get logits from all exit branches
            l = model(x)[-1]

            E1 = l[0]
            E2 = l[1]
            E3 = l[2]
            E4 = l[3]

            #losses, all exits have same target
            L1 = branch_loss(E1,y) 
            L2 = branch_loss(E2,y) + 0.2*diversity_loss(l[:2])
            L3 = loss(E3,y) + 0.2*diversity_loss(l[:3])
            L4 = loss(E4,y) + 0.2*diversity_loss(l[:])

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
                layer = layer + 1

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

        print(f'\n\nEpoch {epoch+1}', end = '\n')
        print('Training:')
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

        for batch_idx, (x, y) in enumerate(val_loader):
            # if batch_idx%200 == 0:
            #     print(str(batch_idx)+"/"+str(len(val_loader)))
            # x,y = batch
            x,y = x.to(device),y.to(device)
            #x: b x 1 x 28 x 28
            b = x.size(0)
            # x = x.view(b,-1)
            #1-forward pass - get logits
            with torch.no_grad():
                l = model(x)[-1]

                E1 = l[0]
                E2 = l[1]
                E3 = l[2]
                E4 = l[3]

            #losses, all exits have same target
            L1 = branch_loss(E1,y) 
            L2 = branch_loss(E2,y) + 0.2*diversity_loss(l[:2])
            L3 = loss(E3,y) + 0.2*diversity_loss(l[:3])
            L4 = loss(E4,y) + 0.2*diversity_loss(l[:])

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

        print('\nValidation:')
        print(f'Total Loss: {val_losses[run,epoch]:.2f}', end='\n')
        print('Exit Losses: ')
        for i in range(len(branch_weights)):
            branch_val_losses[run,i,epoch] = torch.tensor(branch_losses[i]).mean()
            print(f'Exit {(i+1)}: {branch_val_losses[run,i,epoch]:.2f}', end=', ')
        print('\nExit accuracies: ')
        for i in range(len(branch_weights)):
            branch_val_accs[run,i,epoch] = torch.tensor(branch_accuracies[i]).mean()
            print(f'Exit {(i+1)}: {branch_val_accs[run,i,epoch]:.2f}', end=', ')

        val_acc = max(branch_val_accs[run,:,epoch])

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_directory + 'best-'+args.model+'-CIFAR-10-'+str(run)+'.pth')
            best_accuracy = val_acc

        scheduler.step()

    # #Saving model
    # n_epochs = 1
    #
    # input,target = next(iter(train_loader))
    # input,target = input.to(device),target.to(device)
    # output = model(input)
    # output_shape = len(output[0])
    # n_layers = output_shape
    # n_targets = output[-1][0][0].shape[0]

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

    torch.save(model.state_dict(), model_directory + 'epoch'+str(epoch)+'-'+args.model+'-CIFAR-10-'+str(run)+'.pth')

    checkpoint_metrics[run,:] = np.array([epoch,best_accuracy,optimiser.param_groups[0]['lr']])

#Saving all tracked metrics for analysis

save_string = metric_directory+'val-losses-'+args.model+'.npy'
np.save(save_string, val_losses)

save_string = metric_directory+'train-losses-'+args.model+'.npy'
np.save(save_string, train_losses)

save_string = metric_directory+'branch-val-accuracies-'+args.model+'.npy'
np.save(save_string, branch_val_accs)

save_string = metric_directory+'branch-train-accuracies-'+args.model+'.npy'
np.save(save_string, branch_train_accs)

save_string = metric_directory+'branch-train-losses-'+args.model+'.npy'
np.save(save_string, branch_train_losses)

save_string = metric_directory+'branch-val-losses-'+args.model+'.npy'
np.save(save_string, branch_val_losses)

save_string = metric_directory+'checkpoint-metrics-'+args.model+'.npy'
np.save(save_string, checkpoint_metrics)

save_string = metric_directory+'branch-weights-'+args.model+'.npy'
np.save(save_string, branch_weights)

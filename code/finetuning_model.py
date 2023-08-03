#Importing modules
import os
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from ResNet import *
from DenseNet import *
#Arguments for running

parser = argparse.ArgumentParser()
parser.add_argument("load_directory",help="Directory to load model from. Format: 'trained-models/*model*/*run_name*/'", type=str)
parser.add_argument("save_directory",help="Directory to save model at. Format: '*new_run_name*'", type=str)
parser.add_argument("-lm","--load_model", help="Backbone architecture to be loaded",type=str,default='BranchedResNet18')
parser.add_argument("-sm","--save_model", help="Architecture to be trained (must be of equal length)",type=str,default='ResNet18')
parser.add_argument("-r","--runs", help="Number of runs with trained models.",type=int,default=3)
parser.add_argument("-b","--batch_size", help="Batch size for training.",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=float,default=1e-2)
parser.add_argument("-e","--epochs",help="Number of epochs for training",type=int,default=200)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
args = parser.parse_args()

print('\nArguments passed:')
print("Load directory: "+args.load_directory)
print("Save directory: "+args.save_directory)
print("Loaded architecture: " + args.load_model)
print("New Architecture: " + args.save_model)
print("Dataset: " + args.data)
print("Runs: ",args.runs)
print("Batch size: ",args.batch_size)
print("Learning Rate: ",args.learning_rate,'\n')



try:
    save_directory = "trained-models/"+args.save_model+"/"+args.data+"/"+args.save_directory+"/"
    os.mkdir(save_directory)
    print("Saving data to: " , save_directory,"\n")

except FileExistsError:
    print(save_directory, "Already exists...")
    for run in range(100):
        try:
            save_directory = "trained-models/"+args.save_model+"/"+args.data+"/"+args.save_directory+'_'+str(run)+"/"
            os.mkdir(save_directory)
            print("Instead saving data to: " , save_directory,"\n")
            break

        except FileExistsError:
            continue

#Create directories in target folder
metric_directory = save_directory+"/metrics/"
os.mkdir(metric_directory)

model_directory = save_directory+"/saved-models/"
os.mkdir(model_directory)

gradient_directory = save_directory+"/gradient-values/"
os.mkdir(gradient_directory)

for run in range(args.runs):
    gradient_run_directory = save_directory+"/gradient-values/Run-"+str(run)
    os.mkdir(gradient_run_directory)

#Load relevant data from source folder
pretrain_metrics_directory = args.load_directory+"/metrics/"
pretrain_metrics = np.load(pretrain_metrics_directory+"checkpoint-metrics-"+args.load_model+".npy")
pretrained_weights = np.load(pretrain_metrics_directory+"branch-weights-"+args.load_model+".npy")

load_model_directory = args.load_directory+"/saved-models/"

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


def get_model(model_name):
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

    return model

print('\nBeginning experiment:\n')

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)

train_losses = np.zeros((args.runs,args.epochs))
train_accs = np.zeros((args.runs,args.epochs))

val_losses = np.zeros((args.runs,args.epochs))
val_accs = np.zeros((args.runs,args.epochs))

checkpoint_metrics = np.zeros((args.runs,3))

for run in range(args.runs):
    print("Run: " + str(run))

    #defining model
    model = get_model(args.save_model)

    #putting to GPU and loading weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_name = "best-"+args.load_model+'-CIFAR-10-'+str(run)+'.pth'

    pretrained_dict = torch.load(load_model_directory + model_name, map_location=device)

    key_counter = 0
    for name, param in pretrained_dict.items():
        if name in model.state_dict().keys():
            model.state_dict()[name].copy_(param)
            key_counter = key_counter+1

    print("\nLoaded " + str(key_counter) + " target keys.\n")
    print("Randomly initialising missing layers:")

    for name, param in model.state_dict().items():
        if name in pretrained_dict.keys():
            continue
        else:
            print(name)

    param_count = list()
    #Number of convolutional layers to be tracked
    n_conv_layers = 0
    for name, param in model.named_parameters():
      if 'conv' in name:
        n_conv_layers = n_conv_layers + 1
        param_count.append(torch.flatten(param).shape[0])

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

    print("\nTracking ",n_conv_layers," convolutional layers\n")

    #Defining optimiser and scheduler
    params = model.parameters()
    optimiser = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimiser = optim.Adam(params, lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=200)
    loss = nn.CrossEntropyLoss()

    #Training
    print("Training:")
    best_accuracy = -1
    nb_epochs = args.epochs

    epoch_gradients_std = list()
    epoch_gradients_mean = list()
    epoch_gradients_abs_mean = list()

    epoch_weights_std = list()
    epoch_weights_mean = list()
    epoch_weights_abs_mean = list()

    train_accuracy =  np.zeros(args.epochs)
    val_accuracy = np.zeros(args.epochs)
    train_loss= np.zeros(args.epochs)
    val_loss = np.zeros(args.epochs)

    for epoch in range(nb_epochs):
        #track loss and accuracy
        losses = list()
        accuracies = list()
        model.train() # because of dropout

        #Values for tracking
        batch_gradients_std = torch.zeros(n_conv_layers)
        batch_gradients_mean = torch.zeros(n_conv_layers)
        batch_gradients_abs_mean = torch.zeros(n_conv_layers)

        batch_weights_std = torch.zeros(n_conv_layers)
        batch_weights_mean = torch.zeros(n_conv_layers)
        batch_weights_abs_mean = torch.zeros(n_conv_layers)

        for batch in train_loader:
            x,y = batch
            x,y = x.to(device),y.to(device)
            #x: b x 1 x 28 x 28
            b = x.size(0)
            # x = x.view(b,-1)
            #1-forward pass - get logits
            l = model(x)[-1][0]
            #2-objective function
            J = loss(l,y)
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

            losses.append(J.item())
            accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

        train_accuracy[epoch] = torch.tensor(accuracies).mean()
        train_loss[epoch] = torch.tensor(losses).mean()

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


        print(f'Epoch {epoch+1}', end = ', ')
        print(f'Training Loss: {train_loss[epoch]:.2f}', end=', ')
        print(f'Training Accuracy: {train_accuracy[epoch]:.2f}')

        losses = list()
        accuracies = list()
        model.eval()

        for batch in val_loader:

            x,y = batch
            x,y = x.to(device),y.to(device)
            #x: b x 1 x 28 x 28
            b = x.size(0)
            # x = x.view(b,-1)
            #1-forward pass - get logites
            with torch.no_grad():
                l = model(x)[-1][0]
            #2-objective function
            J = loss(l,y)

            losses.append(J.item())
            accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())


        val_loss[epoch] = torch.tensor(losses).mean()
        val_accuracy[epoch] = torch.tensor(accuracies).mean()

        print(f'Epoch {epoch+1}', end = ', ')
        print(f'Validation Loss: {torch.tensor(losses).mean():.2f}', end=', ')
        print(f'Validation Accuracy: {torch.tensor(accuracies).mean():.2f}')

        if val_accuracy[epoch] > best_accuracy:
            torch.save(model.state_dict(), model_directory + 'best-'+args.save_model+'-CIFAR-10-'+str(run)+'.pth')
            best_accuracy = val_accuracy[epoch]

        scheduler.step()

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
    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))


    torch.save(model.state_dict(), model_directory + 'epoch'+str(epoch)+'-'+args.save_model+'-CIFAR-10-'+str(run)+'.pth')

    val_accs[run,:] = val_accuracy
    val_losses[run,:] = val_loss
    train_accs[run,:] = train_accuracy
    train_losses[run,:] = train_loss
    checkpoint_metrics[run,:] = np.array([epoch,best_accuracy,optimiser.param_groups[0]['lr']])

save_string = metric_directory+'parameter-counts-'+args.save_model+'.npy'
np.save(save_string, param_count)

save_string = metric_directory+'val-accuracy-'+args.save_model+'.npy'
np.save(save_string, val_accs)

save_string = metric_directory+'train-accuracy-'+args.save_model+'.npy'
np.save(save_string, train_accs)

save_string = metric_directory+'val-losses-'+args.save_model+'.npy'
np.save(save_string, val_losses)

save_string = metric_directory+'train-losses-'+args.save_model+'.npy'
np.save(save_string, train_losses)

save_string = metric_directory+'checkpoint-metrics-'+args.save_model+'.npy'
np.save(save_string, checkpoint_metrics)

save_string = metric_directory+'pretrained-branch-weights-'+args.save_model+'.npy'
np.save(save_string, pretrained_weights)

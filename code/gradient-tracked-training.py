#Importing modules
import os
import torch
import sys
from torch import nn
# from torch import linalg
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from ResNet import *
from DenseNet import *
from load_sat_data import *
#arguments for running

parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Name of directory to save data", type=str)
parser.add_argument("-m","--model", help="Architecture to be used",type=str,default='ResNet18')
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do",type=int,default=3)
parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
parser.add_argument("-e","--epochs", help="Number of epochs to run experiment for",type=int,default=200)
parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=float,default=1e-2)
args = parser.parse_args()

print('arguments passed:')
print("target_directory: " + args.target_directory)
print("Architecture: " + args.model)
print("Epochs: " + str(args.epochs))
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))
print("Learning Rate: "+str(args.learning_rate))

try:
    # Create target Directory
    os.mkdir("trained-models/"+args.model)
    print("Directory: " , "trained-models/"+args.model ,  " Created ")
except FileExistsError:
    print("Directory: " , "trained-models/"+args.model ,  " already exists")

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
    for run in range(100):
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

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)

train_losses = np.zeros((args.runs,args.epochs))
train_accs = np.zeros((args.runs,args.epochs))

val_losses = np.zeros((args.runs,args.epochs))
val_accs = np.zeros((args.runs,args.epochs))

checkpoint_metrics = np.zeros((args.runs,3))

for run in range(args.runs):

    print("Run:" + str(run+1))

    if args.model == 'ResNet18':
        model = ResNet(BasicBlock,[2,2,2,2],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'ResNet10':
        model = ResNet(BasicBlock,[1,1,1,1],num_classes=output_classes,input_channels=n_channels) # Resnet18
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet34
    elif args.model == 'ResNet50':
        model = ResNet(Bottleneck, [3,4,6,3],num_classes=output_classes,input_channels=n_channels) # Resnet50
    elif args.model == 'DenseNet121':
            model = DenseNet3(depth=121,num_classes=output_classes,input_channels=n_channels) # DenseNet121
    elif args.model == 'DenseNet50':
            model = DenseNet3(depth=50,num_classes=output_classes,input_channels=n_channels) # DenseNet121
    elif args.model == 'DenseNet18':
            model = DenseNet3(depth=18,num_classes=output_classes,input_channels=n_channels) # DenseNet121
    else:
        sys.exit('please specify a valid model')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    output = model(input)
    output_shape = len(output[0])
    n_layers = output_shape
    n_targets = output[-1][0][0].shape[0]

    param_count = list()

    #Number of convolutional layers to be tracked
    n_conv_layers = 0
    for name, param in model.named_parameters():
      if 'conv' in name:
        n_conv_layers = n_conv_layers + 1
        param_count.append(torch.flatten(param).shape[0])


    print("Number of convolutional layers: " + str(n_conv_layers))

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

        for batch_idx, (x, y) in enumerate(train_loader):
            # if batch_idx%20 == 0:
            #     print("\n"+str(batch_idx)+"/"+str(len(train_loader)))
            # x,y = batch
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
            break

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

        for batch_idx, (x, y) in enumerate(val_loader):
            # if batch_idx%200 == 0:
            #     print(str(batch_idx)+"/"+str(len(val_loader)))
            # x,y = batch
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
            break


        val_loss[epoch] = torch.tensor(losses).mean()
        val_accuracy[epoch] = torch.tensor(accuracies).mean()

        print(f'Epoch {epoch+1}', end = ', ')
        print(f'Validation Loss: {torch.tensor(losses).mean():.2f}', end=', ')
        print(f'Validation Accuracy: {torch.tensor(accuracies).mean():.2f}')

        if val_accuracy[epoch] > best_accuracy:
            torch.save(model.state_dict(), model_directory + 'best-'+args.model+'-CIFAR-10-'+str(run)+'.pth')
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


    torch.save(model.state_dict(), model_directory + 'epoch'+str(epoch)+'-'+args.model+'-CIFAR-10-'+str(run)+'.pth')

    val_accs[run,:] = val_accuracy
    val_losses[run,:] = val_loss
    train_accs[run,:] = train_accuracy
    train_losses[run,:] = train_loss
    checkpoint_metrics[run,:] = np.array([epoch,best_accuracy,optimiser.param_groups[0]['lr']])

save_string = metric_directory+'parameter-counts-'+args.model+'.npy'
np.save(save_string, param_count)

save_string = metric_directory+'val-accuracy-'+args.model+'.npy'
np.save(save_string, val_accs)

save_string = metric_directory+'train-accuracy-'+args.model+'.npy'
np.save(save_string, train_accs)

save_string = metric_directory+'val-losses-'+args.model+'.npy'
np.save(save_string, val_losses)

save_string = metric_directory+'train-losses-'+args.model+'.npy'
np.save(save_string, train_losses)

save_string = metric_directory+'checkpoint-metrics-'+args.model+'.npy'
np.save(save_string, checkpoint_metrics)

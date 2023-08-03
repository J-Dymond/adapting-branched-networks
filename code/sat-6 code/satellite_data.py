import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms
import rasterio

# Check if GPU is available ===================================
if torch.cuda.is_available():
    devCnt = torch.cuda.device_count()
    devName = torch.cuda.get_device_name(0)
    print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read in CSV data =============================================================
trainval = pd.read_csv("data/sat-6/y_train.csv")
testSet = pd.read_csv("data/sat-6/y_test.csv")

# Split training and validation data ===========================================
trainSet, valSet = train_test_split(trainval, test_size=0.20, random_state=42)

# Check lengths
print(len(trainSet))
print(len(valSet))
print(len(testSet))

# Subclass Dataset  ========================================================

class cnnDS(Dataset):

    def __init__(self, df, directory, transform=None):
        self.df = df
        self.directory = directory
        self.transform = transform

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 1]
        source = rasterio.open(self.directory + str(self.df.iloc[idx, 0]) + ".png")
        image = source.read()
        source.close()
        image = image.astype('uint8')
        image = torch.from_numpy(image)
        image = image.float()/255
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)

class csv_data(Dataset):

    def __init__(self, target, data_dir, len, batch_size, transform=None):
        self.target = target
        self.data_dir = data_dir
        self.transform = transform
        self.len = len
        self.bs = batch_size

    def __getitem__(self, idx):
        label = self.target.iloc[idx, 1]
        data = pd.read_csv(self.data_dir, nrows = 1, skiprows = idx-1).to_numpy()
        image = np.reshape(data, (28,28,4))
        image = np.moveaxis(image,2,0)
        image = image.astype('uint8')
        image = torch.from_numpy(image)
        image = image.float()/255
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return(int(self.len/self.bs))


class iterable_csv_dataset(IterableDataset):
    def __init__(self, data_dir, len, batch_size, transform=None):
        self.data = csv.reader(open(data_dir),delimiter=',')
        self.len = len
        self.bs = batch_size
        self.transform = transform

    def preprocess(self, image):
        image = np.array(image)
        image = np.reshape(image, (28,28,4))
        image = np.moveaxis(image,2,0)
        image = image.astype('uint8')
        image = torch.from_numpy(image)
        image = image.float()/255
        if self.transform is not None:
            image = self.transform(image)

        return image

    def line_mapper(self, data):
        data = np.array(data, dtype='float32')
        label = torch.from_numpy(np.array(data[-1])).type(torch.LongTensor)
        input = data[:-1]
        image = self.preprocess(input)


        return image,label

    def __iter__(self):
        data =  self.data
        mapped_itr = map(self.line_mapper, data)

        return mapped_itr

    def __len__(self):
        return(int(self.len/self.bs))

# Define data augmentations for training ========================================
transformTrain = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.3), torchvision.transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
])

train_target = pd.read_csv("data/sat-6/y_train.csv")
val_target = pd.read_csv("data/sat-6/y_test.csv")

train_len = train_target.shape[0]
val_len = train_target.shape[0]

train_data = "data/sat-6/X_train_sat6.csv"
val_data = "data/sat-6/X_test_sat6.csv"

batch_size=15
# Split training and validation data ===========================================
# trainSet, valSet = train_test_split(trainval, test_size=0.20, random_state=42)

transformTrain = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.3), torchvision.transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),])
# Use Dataset class ==========================================================
# train_data = iterable_csv_dataset(train_data,train_len,args.batch_size,transform=transformTrain)
# test_data = iterable_csv_dataset(val_data,test_len,args.batch_size,transform=None)

trainDS = csv_data(train_target,train_data,train_len,batch_size,transform=None)
valDS = csv_data(val_target,val_data,val_len,batch_size,transform=None)

# trainDS = cnnDS(train_target, "data/sat-6/train_x/", transform=None)
# valDS = cnnDS(val_target, "data/sat-6/test_x/", transform=None)

# Define data loaders ========================================================
trainDL = torch.utils.data.DataLoader(trainDS, batch_size=15, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)
valDL =  torch.utils.data.DataLoader(valDS, batch_size=15, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

# Check Tensor shapes ======================================================
batch = next(iter(trainDL))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

# Check first sample from batch =================================================
exImg = images[1]
exMsk = labels[1]
print(exImg.shape, exImg.dtype, type(exImg), exMsk.shape,
exMsk.dtype, type(exMsk), exImg.min(),
exImg.max(), exMsk.min(), exMsk.max())

# Visualize image=======================================
def img_display(img):     # unnormalize
    image_vis = img.permute(1,2,0)
    image_vis = image_vis.numpy()*255
    image_vis = image_vis.astype('uint8')
    return image_vis[:,:,:3]

# Get some random training images=========================
dataiter = iter(trainDL)
images, labels = dataiter.next()
cover_types = {0: 'Building', 1: 'Barren', 2: 'Trees', 3: 'Grasslands', 4: 'Road', 5: 'Water'}

# Viewing data examples used for training=========================
fig, axis = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = images[i], labels[i]
        ax.imshow(img_display(image)) # add image
        ax.set(title = f"{cover_types[label.item()]}") # add label
plt.show()
exit()
input("Press Enter to continue...")

# Define CNN model=======================================
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(16 * 7 * 7, 6)
        )
    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# Initiate the model =========================
model = Net()

# Defining the optimizer =======================
optimizer = Adam(model.parameters(), lr=0.07)

# Defining the loss function =======================
criterion = CrossEntropyLoss()

# checking if GPU is available and push model to GPU =======================
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# Sumarize model ===========================================
print(model)

# Define accuracy metric to monitor ==============================
def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

# Train and validate for 10 epochs ===================================
n_epochs = 10
print_every = 2
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainDL)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(trainDL):
        data_, target_ = data_.to(device), target_.to(device)# on GPU
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data_t, target_t in (valDL):
            data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss/len(valDL))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        # Saving the best weight
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'sat6_model.pt')
            print('Detected network improvement, saving current model')
    model.train()

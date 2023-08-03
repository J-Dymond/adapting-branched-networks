import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms
# import rasterio

# # Subclass Dataset  ========================================================
# class cnnDS(Dataset):
#
#     def __init__(self, df, directory, transform=None):
#         self.df = df
#         self.directory = directory
#         self.transform = transform
#
#     def __getitem__(self, idx):
#         label = self.df.iloc[idx, 1]
#         source = rasterio.open(self.directory + str(self.df.iloc[idx, 0]) + ".png")
#         image = source.read()
#         source.close()
#         image = image.astype('uint8')
#         image = torch.from_numpy(image)
#         image = image.float()/255
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, label
#
#     def __len__(self):
#         return len(self.df)

class csv_loaded_data(Dataset):

    def __init__(self, target, data_dir, len, transform=None):
        self.target = target
        self.data = pd.read_csv(data_dir,header=None).to_numpy()
        self.transform = transform
        self.len = self.data.shape[0]
        print("Target: ", self.target.to_numpy().shape[0],"\t Data: ", self.data.shape[0])

    def __getitem__(self, idx):
        label = self.target.iloc[idx, 1]
        data =  self.data[idx,:]
        image = np.reshape(data, (28,28,4))
        image = np.moveaxis(image,2,0)
        image = image.astype('uint8')
        image = torch.from_numpy(image)
        image = image.float()/255
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return(self.len)

class csv_data(Dataset):

    def __init__(self, target, data_dir, len, transform=None):
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
        return(self.len)


class iterable_csv_dataset(IterableDataset):
    def __init__(self, data_dir, len, transform=None):
        self.data = csv.reader(open(data_dir),delimiter=',')
        self.len = len
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
        return(self.len)

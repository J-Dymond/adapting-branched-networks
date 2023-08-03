import numpy as np
import pandas as pd
import cv2
import csv
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms


transform = {
    'test': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
}

data = ImageFolder('data/EuroSat/jpeg/', transform=transform['test'])
n_data = len(data)
n_test = 5000
train_set, val_set = torch.utils.data.random_split(data, [n_data-n_test, n_test])

print(len(train_set),len(val_set))

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = 40, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size = 40, shuffle=True)

for idx, (x,y) in enumerate(train_loader):
    print(idx)
    print(x.shape,y)
    break

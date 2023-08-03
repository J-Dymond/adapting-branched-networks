import torch
from torch import nn

def div_cross(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def diversity_loss(branches):

    loss = 0

    for i in range(1,len(branches)):
        for j in range(0,i):
            t1 = branches[i].clone()
            t2 = branches[j].clone()
            loss += div_cross(t1,t2)

    return -1*loss
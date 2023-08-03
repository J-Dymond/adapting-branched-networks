import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(DenseNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        global embedding

        out = self.relu(self.bn1(x))
        embedding.append(torch.flatten(out,start_dim=1))

        out = self.conv1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = torch.cat([x, out], 1)

        return out

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        global embedding

        out = self.relu(self.bn1(x))
        embedding.append(torch.flatten(out,start_dim=1))

        out = self.conv1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.relu(self.bn2(out))
        embedding.append(torch.flatten(out,start_dim=1))

        out = self.conv2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = torch.cat([x, out], 1)

        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        global embedding

        out = self.relu(self.bn1(x))
        embedding.append(torch.flatten(out,start_dim=1))

        out = self.conv1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = F.avg_pool2d(out, 2)
        embedding.append(torch.flatten(out,start_dim=1))

        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5,input_channels = 3, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = DenseNetBasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(input_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        global embedding
        embedding = []

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))

        out = self.block3(out)
        out = self.relu(self.bn1(out))
        embedding.append(torch.flatten(out,start_dim=1))

        out = F.avg_pool2d(out, 8)
        embedding.append(torch.flatten(out,start_dim=1))

        out = out.view(-1, self.in_planes)
        out = self.fc(out)

        output = [embedding,[out]]
        return output

class BranchedDenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5,input_channels = 3, bottleneck=True, dropRate=0.0):
        super(BranchedDenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = DenseNetBasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(input_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        self.branch1 = nn.Linear(in_planes*8*8, num_classes)

        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        self.branch2 = nn.Linear(in_planes*4*4, num_classes)
        # 3rd block

        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.branch3 = nn.Linear(in_planes*2*2, num_classes)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        global embedding
        embedding = []
        outputs = []

        out = self.conv1(x)
        out = self.trans1(self.block1(out))

        flat_out = F.adaptive_avg_pool2d(out,(8,8))
        flat_out = flat_out.view(flat_out.size(0), -1)
        outputs.append(self.branch1(flat_out))

        out = self.trans2(self.block2(out))

        flat_out = F.adaptive_avg_pool2d(out,(4,4))
        flat_out = flat_out.view(flat_out.size(0), -1)
        outputs.append(self.branch2(flat_out))

        out = self.block3(out)
        out = self.relu(self.bn1(out))

        flat_out = F.adaptive_avg_pool2d(out,(2,2))
        flat_out = flat_out.view(flat_out.size(0), -1)
        outputs.append(self.branch3(flat_out))

        embedding.append(torch.flatten(out,start_dim=1))
        out = F.avg_pool2d(out, 8)
        embedding.append(torch.flatten(out,start_dim=1))
        out = out.view(-1, self.in_planes)
        outputs.append(self.fc(out))

        output = [embedding,outputs]
        return output

import torch
import cv2
import numpy as np
import re

import torch.nn as nn
from torchvision import models
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict
from mobileNetV3 import *

class BranchedMobileNet(nn.Module):
    def __init__(self, output_layers, num_classes=10, input_channels = 3, in_size = 32, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.output_classes = num_classes
        self.selected_out = OrderedDict()
        self.fhooks = list()
        self.fhook_identifiers = list()
        #module list dynamically creates branch modules
        self.branches = nn.ModuleList()

        #mobilenet MODEL
        self.mobilenet = MobileNetV3(classes_num=self.output_classes, input_size=in_size, input_channels=input_channels)

        #get possible layers to add branch
        possible_branches=list()
        for idx, (name,module) in enumerate(list(self.mobilenet.named_modules())):
            if re.match(r"featureList.[0-9]*$", name):
                item = re.sub('\D', '', name)
                possible_branches.append(int(item))

        n_branches = len(possible_branches)
        selected_layers = np.round(np.array(output_layers)*n_branches).astype('int').astype('str')

        #Attach forward hooks to pass to branches
        output_filters = list() #list to keep output filters sizes

        for idx, (name,module) in enumerate(list(self.mobilenet.named_modules())):
            if re.match(r"featureList.[0-9]*$", name):
                layer_no = re.sub('\D', '', name) #if name matches selected layers attach hook
                if layer_no in selected_layers:
                    param_shapes = list() #get shapes of the paramaters in the layer
                    for in_idx, (param_name,param) in enumerate(module.named_parameters()):
                        param_shapes.append(param.shape)
                    output_filters.append(param_shapes[-1][0]) #final paramater = bias = number of filters
                    #attach hooks and keep hook name
                    self.fhooks.append(module.register_forward_hook(self.branch_hook(module)))
                    self.fhook_identifiers.append(name)

        #create branches for each selected layer
        for idx, layer in enumerate(selected_layers):
            #use output filters to create appropriately sized branches
            layer_channels = output_filters[idx]
            self.branches.append(
                nn.Sequential(
                nn.AdaptiveAvgPool2d((8,8)),
                nn.Flatten(start_dim=1),
                nn.Linear((layer_channels*8*8),self.output_classes)
                )
            )

    #forward hook will take output from selected layer, put it in dictionary
    def branch_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        outputs = list()
        final_out = self.mobilenet(x) #run forward pass to obtain output and retrieve forward hooks
        sub_out = self.selected_out.items()

        #pass forward hooks to approriate branch
        for idx, (output) in enumerate(sub_out):
            outputs.append(self.branches[idx](output[1]))

        #final output should be last output
        outputs.append(final_out)
        return [outputs] #, self.selected_out, self.fhook_identifiers

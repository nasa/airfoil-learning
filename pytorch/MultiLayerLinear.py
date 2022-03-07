import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Module, Dropout
from typing import List
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiLayerLinear(Module):
    """
    Model aims to predict both cp and y with spatial convolutions (MFConv).
    """
    # Multi-layer Perceptron 
    # https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124/2

    # https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict

    def __init__(self,in_channels:int,out_channels:int,h_sizes:List[int]=None):
        """This class joins a bunch of linear layers together to predict the output size

        Args:
            in_channels (int): number of inputs channels
            out_channels (int): number of output channels
            h_sizes (List[int], optional): Any additional internal linear layers. Defaults to None.
        """        
        super(MultiLayerLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.ModuleList()
        if (h_sizes!=None):
            self.layers.append(nn.Linear(in_channels,h_sizes[0],bias=True))
            self.layer_sizes = h_sizes
            for k in range(len(h_sizes)-1):
                self.layers.append(nn.Linear(h_sizes[k], h_sizes[k+1],bias=True))
            self.layers.append(nn.Linear(h_sizes[len(h_sizes)-1], out_channels,bias=True))
        else:
            self.layers.append(nn.Linear(in_channels,out_channels))
            self.layer_sizes = None
   
    def forward(self,x):
        out = x
        for i in range(len(self.layers)-1):
            out = F.relu6(self.layers[i](out))        
        return self.layers[-1](out) # dont activate the last laye

    def __str__(self):
        n_layers = len(self.layers)
        if (self.layer_sizes):
            layer_sizes = '-'.join([str(x) for x in self.layer_sizes])
            return f"MLL_IN-{self.in_channels}_OUT-{self.out_channels}_{layer_sizes}"
        else:
            return f"MLL_IN-{self.in_channels}_OUT-{self.out_channels}"

    def __repr__(self):
        return self.__str__()

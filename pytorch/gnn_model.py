import torch
from typing import List
import torch.nn.functional as F
from torch.nn import Linear, Module, Dropout, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm
# from torch_geometric.nn import SplineConv
from SplineConv import SplineConv
from MultiLayerLinear import MultiLayerLinear

class GnnModel(Module):
    """Defines the graph neural network structure for prediction 
        Example of encoder decoder with cnn and batchnorms https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
    """
    def __init__(self,linear_input_size:int, neurons:List[int]=[16,32,64],linear_layers:MultiLayerLinear = None,batch_norm_encoder=False, batch_norm_decoder=False):
        """Constructor for GnnModel. User passes the number of neurons they would like the graph network to use. The graph network then constructs and encoder with those neurons and corresponding decoder. 

        Args:
            linear_input_size (int): 
            neurons (List[int], optional): Number of neurons to use for each graph neural network. Defaults to [16,32,64].
            use_batch_norm (bool, optional): Whether to use batch norm in between encoder networks. Defaults to False.
            dropout (float, optional): [description]. Defaults to 0.
            linear_layers (MultiLayerLinear, optional): [description]. Defaults to None.
        """
        super(GnnModel, self).__init__()
        self.linear_input_size = linear_input_size
        self.batch_norm_encoder = batch_norm_encoder
        self.batch_norm_decoder = batch_norm_decoder
        self.encoder = ModuleList()
        self.encoder_bn = ModuleList()
        self.neurons = neurons
        # Encoder 
        for n in range(1,len(neurons)): 
            self.encoder.append(SplineConv(neurons[n-1], neurons[n], dim=2,degree=2,kernel_size=3))
            self.encoder_bn.append(BatchNorm(neurons[n]))
        self.encoder.append(SplineConv(neurons[n], neurons[n], dim=2,degree=2,kernel_size=3))
        self.encoder_bn.append(BatchNorm(neurons[n]))

        # Decoder 
        self.decoder = ModuleList()
        self.decoder_bn = ModuleList()
        i = 0
        n_encoder = len(self.encoder)-1
        for n in range(len(neurons)-1,0,-1):
            out_channels = self.encoder[n_encoder-i].out_channels 
            self.decoder.append(SplineConv(out_channels+neurons[n], neurons[n], dim=2,degree=2,kernel_size=3))
            self.decoder_bn.append(BatchNorm(neurons[n]))
            i+=1
        self.linear_layers = linear_layers
    
    def forward(self,data:Data):
        """Sends torch geometric data through the neural network. 

        Args:   
            data (torch_geometric.data.Data): https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html 

        Returns:
            (torch.tensor): tensor describing the results of the network. This depends on what is predicted by MultiLayerLinear
        """
        x, edge_index, edge_attr, batch, pos = data.x,data.edge_index, data.edge_attr, data.batch, data.pos
        batch_size = max(batch)+1
        # Run through encoder and store results 
        out = list()
        out.append(x)
        for encoder in self.encoder:
            out.append(F.relu6(encoder(out[-1],edge_index,edge_attr=pos)))
        # Run through decoder and apply results from encoder 
        n = len(out)-3
        dout_out = torch.cat([out[-1], out[-2]],dim=1) 
        for decoder in self.decoder:
            dout = F.relu6(decoder(dout_out, edge_index, edge_attr=pos))     # https://arxiv.org/pdf/1711.08920.pdf From SplineConv Paper G = (V, E, U) where U is the position and also edge attributes
            dout_out = torch.cat([dout, out[n]],dim=1)                            # data = spline_basis(edge_attr, self.kernel_size, self.is_open_spline, self.degree) Spline basis
            n -= 1

        # conditions = conditions.reshape((batch_size,3))
        out = dout.reshape((batch_size,self.linear_input_size*dout.shape[1]))
        # out8 = torch.cat((out7,conditions),dim=1)
        return self.linear_layers(out)
    
    def __str__(self):
        n_encoders = len(self.encoder)
        n_decoders = len(self.decoder)
        neurons = '-'.join([str(n) for n in self.neurons])
        linear = str(self.linear_layers)
        return f"GnnModel_{neurons}_{n_encoders}_{n_decoders}_{str(self.batch_norm_encoder)}_{str(self.batch_norm_decoder)}_MLinear_{linear}"
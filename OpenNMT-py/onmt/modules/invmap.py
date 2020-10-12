# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        super(ReLUNet, self).__init__()

        input_layer = nn.Linear(input_dim, hidden_dim)
        output_layer = nn.Linear(hidden_dim, output_dim)

        layers = [input_layer]
        for i in range(hidden_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, input_):
        return self.network(input_)


class InvertibleMap(nn.Module):
    """
    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, emb_dim, hidden_dim, coupling_layers, cell_layers, device):
        super(InvertibleMap, self).__init__()

        self.device = device
        self.coupling_layers = []
        for i in range(coupling_layers):
            self.coupling_layers.append(ReLUNet(emb_dim//2, emb_dim//2, hidden_dim, cell_layers))  
        self.coupling_layers = nn.ModuleList(self.coupling_layers)      

    def forward(self, input_):
        inpsize = input_.size()
        emb_dim = inpsize[-1]
        h = input_
        for i, layer in enumerate(self.coupling_layers):
            h1, h2 = torch.split(h, emb_dim//2, dim=-1)
            if i%2 == 0:
                h = torch.cat((h1, h2 + layer(h1)), dim=-1)
            else:
                h = torch.cat((h1 + layer(h2), h2), dim=-1)
        
        return h
    
    def reverse(self, output):
        emb_dim = output.size()[-1]
        h = output
        i = len(self.coupling_layers)-1
        while i>=0:
            h1, h2 = torch.split(h, emb_dim//2, dim=-1)
            if i%2 == 0:
                h = torch.cat((h1, h2-self.coupling_layers[i](h1)), dim=-1)
            else:
                h = torch.cat((h1-self.coupling_layers[i](h2), h2), dim=-1)
            i-=1
        
        return h

    def orthogonalize(self, beta):
        pass
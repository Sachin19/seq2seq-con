# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class JLMap(nn.Module):
    """
    Average Attention module from
    "Accelerating Neural Transformer via an Average Attention Network"
    :cite:`DBLP:journals/corr/abs-1805-00631`.

    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, dim, map_id_init=False, trainable=False, use_noise=False, mu=0., sigma=1., device=torch.device("cpu")):
        self.dim = dim
        super(JLMap, self).__init__()

        # self.map = nn.Linear(dim, dim, bias=False)
        self.map = nn.Parameter(torch.Tensor(dim, dim)).to(device)
        self.register_buffer('noise', torch.zeros(dim, dim))
        self.dim = dim
        self.use_noise = use_noise
        self.mu = 0.
        self.sigma = 0.
        self.device = device
        self.trainable = trainable

        if map_id_init or not trainable:
            self.map.data.copy_(torch.diag(torch.ones(dim)))
            # if not trainable:
            #     self.map.requires_grad=False
        

    def forward(self, inputs):
        if self.use_noise and self.training:
            self.noise = torch.randn(self.dim, self.dim).to(self.device)
            out = F.linear(inputs, self.map * (1+self.noise), bias=None)
        else:
            out = F.linear(inputs, self.map)
        if not self.trainable:
            return out.detach()
        return out

    def orthogonalize(self, beta):
        if beta > 0.0:
            W = self.map.data
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
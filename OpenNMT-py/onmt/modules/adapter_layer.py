"""Position feed-forward network from "Attention is All You Need"."""

import torch.nn as nn


class AdapterLayer(nn.Module):
    """A one/two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN, 0 is no hidden layer.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff=0, dropout=0.1):
        super(AdapterLayer, self).__init__()
        self.d_ff = d_ff
        self.adapter_w_1 = nn.Linear(d_model, d_ff if d_ff>0 else d_model)
        self.adapter_w_2 = nn.Identity()
        if d_ff > 0:
            self.adapter_w_2 = nn.Linear(d_ff, d_model)
        self.adapter_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.adapter_dropout_1 = nn.Dropout(dropout)
        self.adapter_relu = nn.ReLU()
        self.adapter_dropout_2 = nn.Identity()
        if d_ff > 0:
            self.adapter_dropout_2 = nn.Dropout(dropout)
        
        self.initialize()

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.adapter_dropout_1(
            self.adapter_relu(self.adapter_w_1(self.adapter_layer_norm(x)))
        )
        output = self.adapter_dropout_2(self.adapter_w_2(inter))
        return output + x

    def update_dropout(self, dropout):
        self.adapter_dropout_1.p = adapter_dropout
        if self.d_ff > 0:
            self.adapter_dropout_2.p = adapter_dropout
    
    def initialize(self):
        pass


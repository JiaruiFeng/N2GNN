"""
GNN pooling layer.
"""

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn.dense import Linear


class Pooling(nn.Module):
    r"""Pooling layer for set. Pool on the input tensor given input index.
    Args:
        hidden_channels (int): Hidden size.
        pooling_method (str): Graph pooling method, choosing from (sum, mean, max, attention).
    """

    def __init__(self,
                 hidden_channels: int,
                 pooling_method: str):
        super(Pooling, self).__init__()
        self.hidden_channels = hidden_channels
        self.pooling_method = pooling_method

        # Different kind of graph pooling
        if pooling_method == "sum":
            self.pool = global_add_pool
        elif pooling_method == "mean":
            self.pool = global_mean_pool
        elif pooling_method == "max":
            self.pool = global_max_pool
        elif pooling_method == "attention":
            self.pool = GlobalAttention(gate_nn=Linear(self.hidden_channels, 1))
        else:
            raise ValueError("The pooling method not implemented")

        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_method == "attention":
            self.pool.reset_parameters()

    def forward(self,
                x: Tensor,
                batch: Tensor) -> Tensor:
        return self.pool(x, batch)

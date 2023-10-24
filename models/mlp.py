"""
Multi layer perceptron.
"""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense import Linear

from .norms import Normalization


class MLP(nn.Module):
    r"""Multi-layer perceptron.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        norm_type (str, optional): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_type: Optional[str] = "Batch"):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.linear1 = Linear(self.in_channels, self.out_channels)
        self.linear2 = Linear(self.out_channels, self.out_channels)
        self.norm = Normalization(self.out_channels, self.norm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.norm(x)
        x = F.relu(x)
        return self.linear2(x)

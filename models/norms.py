"""
Normalization layer.
"""

import torch.nn as nn
from torch import Tensor
from torch.nn import Identity
from torch_geometric.nn import BatchNorm, LayerNorm, InstanceNorm, PairNorm, GraphSizeNorm


class Normalization(nn.Module):
    r"""Model normalization layer.
    Args:
        hidden_channels (int): Hidden size.
        norm_type (str): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
    """

    def __init__(self,
                 hidden_channels: int,
                 norm_type: str = "Batch"):
        super(Normalization, self).__init__()
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type

        # norm list
        if self.norm_type == "Batch":
            self.norm = BatchNorm(self.hidden_channels)
        elif self.norm_type == "Layer":
            self.norm = LayerNorm(self.hidden_channels)
        elif self.norm_type == "Instance":
            self.norm = InstanceNorm(self.hidden_channels)
        elif self.norm_type == "GraphSize":
            self.norm = GraphSizeNorm()
        elif self.norm_type == "Pair":
            self.norm = PairNorm()
        elif self.norm_type == "None":
            self.norm = Identity()
        else:
            raise ValueError("Not supported norm method")

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.norm.apply(self.weights_init)

    def forward(self,
                x: Tensor) -> Tensor:
        return self.norm(x)

"""
GNN conv layers.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_scatter import scatter_add

from .mlp import MLP


class GINETupleMultiplyConv(MessagePassing):
    r"""Graph isomorphism network layer with tuple aggregation.
        The message of tuple is computed by projection(down) -> multiply -> activation -> projection(up) -> concatenation -> projection.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        tuple_size (int): The length of tuple in update message.
        inner_channels (int, optional): Inner feature size during the tuple message passing.
        initial_eps (float, optional): Epsilon for center node information in GIN.
        train_eps (bool, optional): If true, the epsilon is trainable.
        norm_type (str, optional): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        HP (bool, optional): If true, add hierarchical pooling in tuple aggregation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 tuple_size: int,
                 inner_channels: int = 32,
                 initial_eps: Optional[float] = 0.,
                 train_eps: Optional[bool] = False,
                 norm_type: Optional[str] = "Batch",
                 HP: Optional[bool] = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.tuple_size = tuple_size
        self.initial_eps = initial_eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.HP = HP
        if self.train_eps:
            self.eps = nn.Parameter(torch.Tensor([self.initial_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([self.initial_eps]))

        self.tuple_multipler = nn.Parameter(torch.randn(self.tuple_size, 1, self.inner_channels))
        self.inner_proj = Linear(self.in_channels, self.inner_channels)
        self.tuple_proj = Linear(self.inner_channels * self.tuple_size, self.in_channels)
        if self.HP:
            self.hp_mlp = MLP(self.inner_channels, self.inner_channels, self.norm_type)
        self.mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        nn.init.kaiming_normal_(self.tuple_multipler.data)
        self.inner_proj.reset_parameters()
        self.tuple_proj.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.HP:
            self.hp_mlp.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: LongTensor,
                edge_attr: Optional[list] = None,
                first2second: Optional[LongTensor] = None,
                second2tuple: Optional[LongTensor] = None,
                num_first: Optional[int] = None
                ) -> Tensor:
        source = edge_index[0]

        x_ = self.inner_proj(x)
        x_tuple = x_[edge_index[1:]]

        if edge_attr is not None:
            edge_attr_size = edge_attr.size(0)
            x_tuple[:edge_attr_size] += edge_attr

        multipler = F.tanh(self.tuple_multipler)
        x_tuple = x_tuple * multipler
        x_tuple = x_tuple.transpose(0, 1).contiguous().view(-1, self.tuple_size * self.inner_channels)
        x_j = F.relu(self.tuple_proj(x_tuple))

        if self.HP:
            num_first = torch.sum(num_first)
            out = scatter_add(x_j, first2second, dim=self.node_dim, dim_size=num_first)
            out = self.hp_mlp(out)
            out = scatter_add(out, second2tuple, dim=self.node_dim, out=torch.zeros_like(x))
        else:
            out = scatter_add(x_j, source, dim=self.node_dim, out=torch.zeros_like(x))

        out = out + (1 + self.eps) * x
        return self.mlp(out)

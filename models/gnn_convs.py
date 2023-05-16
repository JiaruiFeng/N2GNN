"""
GNN conv layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_scatter import scatter_add

from .mlp import MLP


class GINETupleConcatConv(MessagePassing):
    r"""Graph isomorphism network layer with tuple aggregation.
        The message for tuple is computed by concatenation-> projection.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        tuple_size (int): The length of tuple in update message.
        initial_eps (float): Epsilon for center node information in GIN.
        train_eps (bool): If true, the epsilon is trainable.
        norm_type (str): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        HP (bool): If true, add hierarchical pooling in tuple aggregation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 tuple_size: int,
                 initial_eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "Batch",
                 HP: bool = False):
        super(GINETupleConcatConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tuple_size = tuple_size
        self.initial_eps = initial_eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.HP = HP
        if self.train_eps:
            self.eps = nn.Parameter(torch.Tensor([self.initial_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([self.initial_eps]))

        self.tuple_proj = Linear(self.tuple_size * self.in_channels, self.in_channels)
        if self.HP:
            self.hp_mlp = MLP(self.in_channels, self.in_channels, self.norm_type)
        self.mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.tuple_proj.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.HP:
            self.hp_mlp.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: LongTensor,
                edge_attr: list = None,
                first2second: LongTensor = None,
                second2tuple: LongTensor = None,
                num_first: int = None
                ) -> Tensor:
        source = edge_index[0]
        x_tuple_list = []

        for i in range(self.tuple_size):
            x_j = x[edge_index[i + 1]]
            x_tuple_list.append(x_j)

        if edge_attr is not None:
            for i, edge_emb in enumerate(edge_attr):
                x_tuple_list[i] = x_tuple_list[i] + edge_emb
        x_j = F.relu(self.tuple_proj(torch.cat(x_tuple_list, dim=-1)))

        if self.HP:
            num_first = torch.sum(num_first)
            out = scatter_add(x_j, first2second, dim=self.node_dim, dim_size=num_first)
            out = self.hp_mlp(out)
            out = scatter_add(out, second2tuple, dim=self.node_dim, out=torch.zeros_like(x))
        else:
            out = scatter_add(x_j, source, dim=self.node_dim, out=torch.zeros_like(x))

        out = out + (1 + self.eps) * x
        return self.mlp(out)


class GINETupleMultiplyConv(MessagePassing):
    r"""Graph isomorphism network layer with tuple aggregation.
        The message of tuple is computed by multiply-> activation -> summation -> projection.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        tuple_size (int): The length of tuple in update message.
        initial_eps (float): Epsilon for center node information in GIN.
        train_eps (bool): If true, the epsilon is trainable.
        norm_type (str): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        HP (bool): If true, add hierarchical pooling in tuple aggregation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 tuple_size: int,
                 initial_eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "Batch",
                 HP: bool = False):
        super(GINETupleMultiplyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tuple_size = tuple_size
        self.initial_eps = initial_eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.HP = HP
        if self.train_eps:
            self.eps = nn.Parameter(torch.Tensor([self.initial_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([self.initial_eps]))

        self.tuple_multipler = nn.Parameter(torch.randn(self.tuple_size, 1, self.in_channels))
        self.tuple_proj = Linear(self.in_channels, self.in_channels)
        if self.HP:
            self.hp_mlp = MLP(self.in_channels, self.in_channels, self.norm_type)
        self.mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        nn.init.kaiming_normal_(self.tuple_multipler.data)
        self.tuple_proj.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.HP:
            self.hp_mlp.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: LongTensor,
                edge_attr: list = None,
                first2second: LongTensor = None,
                second2tuple: LongTensor = None,
                num_first: int = None
                ) -> Tensor:
        source = edge_index[0]
        x_tuple_list = []

        for i in range(self.tuple_size):
            x_j = x[edge_index[i + 1]]
            x_tuple_list.append(x_j)

        if edge_attr is not None:
            for i, edge_emb in enumerate(edge_attr):
                x_tuple_list[i] = x_tuple_list[i] + edge_emb

        x_tuple_list = torch.vstack([x_j.unsqueeze(0) for x_j in x_tuple_list])
        x_tuple_list = F.elu(x_tuple_list * F.tanh(self.tuple_multipler))
        x_j = F.relu(self.tuple_proj(torch.sum(x_tuple_list, dim=0)))

        if self.HP:
            num_first = torch.sum(num_first)
            out = scatter_add(x_j, first2second, dim=self.node_dim, dim_size=num_first)
            out = self.hp_mlp(out)
            out = scatter_add(out, second2tuple, dim=self.node_dim, out=torch.zeros_like(x))
        else:
            out = scatter_add(x_j, source, dim=self.node_dim, out=torch.zeros_like(x))

        out = out + (1 + self.eps) * x
        return self.mlp(out)

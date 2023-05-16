"""
GNN framework for N2GNN.
"""

import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch import Tensor
from models.mlp import MLP
from .jumping_knowledge import JumpingKnowledge
from .norms import Normalization
from .utils import *
import torch


class N2GNN(nn.Module):
    r"""An implementation of N2GNN.
    Args:
        num_layers (int): the total number of GNN layer.
        gnn_layer (nn.Module): gnn layer used in GNN model.
        init_encoder (nn.Module): initial node feature encoding.
        feature_encoders (list): Additional feature encoder.
        edge_encoder (nn.Module): Edge feature encoder.
        JK (str): Method of jumping knowledge, last,concat,max or sum.
        norm_type (str): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        residual (bool): If ture, add residual connection.
        initial_eps (float): Epsilon for center node information in aggregation.
        train_eps (bool): If true, the epsilon is trainable.
        drop_prob (float): dropout rate.
        add_root (bool): If true, add root embedding at each layer.
    """

    def __init__(self,
                 num_layers: int,
                 gnn_layer: nn.Module,
                 init_encoder: nn.Module,
                 feature_encoders: list,
                 edge_encoder: nn.Module = None,
                 JK: str = "last",
                 norm_type: bool = "Batch",
                 residual: bool = False,
                 initial_eps: float = 0.0,
                 train_eps: bool = False,
                 drop_prob: float = 0.0,
                 add_root: bool = True):
        super(N2GNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = gnn_layer.out_channels
        self.JK = JK
        self.norm_type = norm_type
        self.residual = residual
        self.initial_eps = initial_eps
        self.train_eps = train_eps
        self.drop_prob = drop_prob
        self.add_root = add_root
        self.dropout = nn.Dropout(drop_prob)
        self.init_encoder = init_encoder
        self.feature_encoders = nn.ModuleList(feature_encoders)

        self.jk_decoder = JumpingKnowledge(self.hidden_channels,
                                           self.JK,
                                           self.num_layers,
                                           self.drop_prob)

        # norm list
        norm = Normalization(self.hidden_channels, self.norm_type)
        norms = clones(norm, self.num_layers)

        self.gnns = clones(gnn_layer, self.num_layers)
        self.norms = c(norms)
        self.edge_encoders = clones(edge_encoder, self.num_layers)

        # node feature enhancement
        if self.add_root:
            mlp = MLP(self.hidden_channels, self.hidden_channels, self.norm_type)
            # proj = Linear(self.hidden_channels, self.hidden_channels)
            self.root_mlps = clones(mlp, self.num_layers)
            self.root_norms = c(norms)
            if self.train_eps:
                self.root_eps = torch.nn.Parameter(torch.Tensor([self.initial_eps for _ in range(self.num_layers)]))
            else:
                self.register_buffer('root_eps', torch.Tensor([self.initial_eps for _ in range(self.num_layers)]))

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.init_encoder.reset_parameters()
        self.jk_decoder.reset_parameters()

        for f in self.feature_encoders:
            f.reset_parameters()

        for g in self.gnns:
            g.reset_parameters()
        for n in self.norms:
            n.reset_parameters()
        if self.edge_encoders is not None:
            for e in self.edge_encoders:
                e.reset_parameters()
        if self.add_root:
            for m in self.root_mlps:
                m.reset_parameters()
            for n in self.root_norms:
                n.reset_parameters()
            self.root_eps.data.fill_(self.initial_eps)

    def forward(self,
                data: Data) -> Tensor:
        x, edge_index, root_idx, node_idx = data.x, data.edge_index, data.root_index, data.tuple2second
        edge_attr = get_pyg_attr(data, "edge_attr")
        if edge_attr is not None:
            if len(edge_attr.size()) == 3:
                edge_attr = edge_attr.permute(2, 0, 1).contiguous()
            else:
                edge_attr = edge_attr.transpose(0, 1)
        first2second = get_pyg_attr(data, "first2second")
        second2tuple = get_pyg_attr(data, "second2tuple")
        num_first = get_pyg_attr(data, "num_first")

        # initial projection
        x = self.init_encoder(x).squeeze()

        # additional feature augmentation
        for i, f in enumerate(self.feature_encoders):
            x += f(get_pyg_attr(data, f"z{str(i)}"))

        # forward in gnn layer
        h_list = [x]
        for l in range(self.num_layers):
            h = h_list[l]

            if edge_attr is not None:
                # tuple with edge attr must be placed before tuple without edge attr, and the order of edge attr should
                # be corresponded to the order of tuple.
                edge_emb = self.edge_encoders[l](edge_attr)
            else:
                edge_emb = edge_attr
            out = self.norms[l](self.gnns[l](h, edge_index, edge_emb, first2second, second2tuple, num_first))

            if self.add_root:
                # root augmentation
                h_root = h[root_idx]
                h_root = self.root_norms[l](self.root_mlps[l]((1 + self.root_eps[l]) * h + h_root[node_idx]))
                out += h_root

            out = self.dropout(out)
            if self.residual:
                out = out + h_list[l]

            h_list.append(out)

        # pooling
        h_list = [global_add_pool(h, node_idx) for h in h_list]
        return self.jk_decoder(h_list)

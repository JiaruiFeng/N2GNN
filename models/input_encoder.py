"""
Different feature input encoder for different dataset and feature type.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.dense import Linear


class EmbeddingEncoder(nn.Module):
    r"""Input encoder with embedding layer. Used for categorical feature.
    Args:
        in_channels (int): Input feature size.
        hidden_channels (int): Hidden size.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int):
        super(EmbeddingEncoder, self).__init__()
        self.init_proj = nn.Embedding(in_channels, hidden_channels)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.init_proj(x)


class LinearEncoder(nn.Module):
    r"""Input encoder with linear projection layer. Used for one-hot or numerical feature.
    Args:
        in_channels (int): Input feature size.
        hidden_channels (int): Hidden size.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int):
        super(LinearEncoder, self).__init__()
        self.init_proj = Linear(in_channels, hidden_channels)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.init_proj(x)


class QM9InputEncoder(nn.Module):
    r"""Input encoder for QM9 dataset.
    Args:
        hidden_channels (int): Hidden size.
        use_pos (bool, optional): If True, add position feature to embedding.
    """

    def __init__(self,
                 hidden_channels: int,
                 use_pos: Optional[bool] = False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            in_channels = 22
        else:
            in_channels = 19
        self.init_proj = Linear(in_channels, hidden_channels)
        self.z_embedding = nn.Embedding(10, 8)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        z = x[:, 0].squeeze().long()
        x = x[:, 1:]
        z_emb = self.z_embedding(z)
        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)
        x = self.init_proj(x)

        return x


@torch.jit.script
def gaussian(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    r"""Gaussian basis.
    Args:
        x (Tensor): Input value tensor.
        mean (Tensor): Mean value tensor for gaussian distribution.
        std (Tensor): Std value tensor for gaussian distribution.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class RDEncoder(nn.Module):
    r"""Encoder for resistance distance with gaussian basis kernel.
    Args:
        hidden_channels (int): Hidden size of the model.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.means = nn.Embedding(1, self.hidden_channels)
        self.stds = nn.Embedding(1, self.hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 1).expand(-1, self.hidden_channels)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class AtomEncoder(torch.nn.Module):
    r"""Atom encoder for OGBG molecular prediction datasets.
    Args:
        hidden_channels (int): Hidden size of the model.
    """

    def __init__(self, hidden_channels: int):
        super(AtomEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate([119, 4, 12, 12, 10, 6, 6, 2, 2]):
            emb = torch.nn.Embedding(dim, self.hidden_channels)
            self.atom_embedding_list.append(emb)
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.atom_embedding_list:
            l.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""Bond encoder for OGBG molecular prediction datasets.
    Args:
        hidden_channels (int): Hidden size of the model.
    """

    def __init__(self, hidden_channels: int):
        super(BondEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate([5, 6, 2]):
            emb = torch.nn.Embedding(dim, self.hidden_channels)
            self.bond_embedding_list.append(emb)
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.bond_embedding_list:
            l.reset_parameters()

    def forward(self, edge_attr: Tensor) -> Tensor:
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[..., i])

        return bond_embedding

"""
Different output decoders for different datasets/tasks.
"""

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.dense import Linear

from .pooling import Pooling


class GraphClassification(nn.Module):
    r"""Framework for graph classification.
    Args:
        embedding_model (nn.Module):  GNN embedding model.
        pooling_method (str): Graph pooling method.
        out_channels (int): Output size, equal to the number of class for classification.
    """

    def __init__(self,
                 embedding_model: nn.Module,
                 pooling_method: str,
                 out_channels: int):
        super(GraphClassification, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_channels = embedding_model.hidden_channels
        self.pooling_method = pooling_method
        self.out_channels = out_channels

        self.pool = Pooling(self.hidden_channels, self.pooling_method)

        # classifier
        self.classifier = nn.Sequential(Linear(self.hidden_channels, self.hidden_channels // 2),
                                        nn.ELU(),
                                        Linear(self.hidden_channels // 2, self.out_channels))

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.classifier.apply(self.weights_init)
        self.pool.reset_parameters()

    def forward(self,
                data: Data) -> Tensor:
        batch = data.node2graph
        # node representation
        x = self.embedding_model(data)
        pool_x = self.pool(x, batch)
        return self.classifier(pool_x)


class GraphRegression(nn.Module):
    r"""Framework for graph regression.
    Args:
        embedding_model (nn.Module): GNN embedding model.
        pooling_method (str): Graph pooling method.
    """

    def __init__(self,
                 embedding_model: nn.Module,
                 pooling_method: str):
        super(GraphRegression, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_channels = embedding_model.hidden_channels
        self.pooling_method = pooling_method

        self.pool = Pooling(self.hidden_channels, self.pooling_method)

        # regressor
        self.regressor = nn.Sequential(Linear(self.hidden_channels, self.hidden_channels // 2),
                                       nn.ELU(),
                                       Linear(self.hidden_channels // 2, 1))

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.regressor.apply(self.weights_init)
        self.pool.reset_parameters()

    def forward(self,
                data: Data) -> Tensor:
        batch = data.node2graph
        # node representation
        x = self.embedding_model(data)
        pool_x = self.pool(x, batch)
        return self.regressor(pool_x).squeeze()


class NodeClassification(nn.Module):
    r"""Framework for node classification.
    Args:
        embedding_model (nn.Module):  GNN embedding model.
        out_channels (int): Output size, equal to the number of class for classification.
    """

    def __init__(self,
                 embedding_model: nn.Module,
                 out_channels: int):
        super(NodeClassification, self).__init__()
        self.out_channels = out_channels
        self.embedding_model = embedding_model
        self.hidden_channels = embedding_model.hidden_channels
        self.classifier = nn.Sequential(Linear(self.hidden_channels, self.hidden_channels // 2),
                                        nn.ELU(),
                                        Linear(self.hidden_channels // 2, self.out_channels))

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.classifier.apply(self.weights_init)

    def forward(self,
                data: Data) -> Tensor:
        # node representation
        x = self.embedding_model(data)
        return self.classifier(x)


class NodeRegression(nn.Module):
    r"""Framework for node regression.
    Args:
        embedding_model (nn.Module): GNN embedding model.
    """

    def __init__(self,
                 embedding_model: nn.Module):
        super(NodeRegression, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_channels = embedding_model.hidden_channels
        self.regressor = nn.Sequential(Linear(self.hidden_channels, self.hidden_channels // 2),
                                       nn.ELU(),
                                       Linear(self.hidden_channels // 2, 1))

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.regressor.apply(self.weights_init)

    def forward(self,
                data: Data) -> Tensor:
        # node representation
        x = self.embedding_model(data)
        return self.regressor(x).squeeze()

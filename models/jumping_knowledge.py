"""
Jumping knowledge method for GNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense import Linear


class JumpingKnowledge(nn.Module):
    r"""Jumping Knowledge method for combining result of GNN model from multiple layers.
    Args:
        hidden_channels (int): Hidden size.
        JK (str): Method of jumping knowledge, choose from (last, concat, max, sum, attention).
        num_layers (int): The number of layer in the GNN model.
        drop_prob (float): Dropout probability.
    """

    def __init__(self,
                 hidden_channels: int,
                 JK: str,
                 num_layers: int,
                 drop_prob: float = 0.1):
        super(JumpingKnowledge, self).__init__()
        self.hidden_channels = hidden_channels
        self.JK = JK
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        if self.JK == "concat":
            self.output_decoder = nn.Sequential(Linear((self.num_layers + 1) * self.hidden_channels,
                                                       self.hidden_channels),
                                                nn.ELU(), nn.Dropout(drop_prob))
        else:
            self.output_decoder = nn.Sequential(Linear(self.hidden_channels, self.hidden_channels),
                                                nn.ELU(),
                                                nn.Dropout(drop_prob))

        if self.JK == "attention":
            self.attention_lstm = nn.LSTM(self.hidden_channels,
                                          self.num_layers,
                                          num_layers=1,
                                          batch_first=True,
                                          bidirectional=True,
                                          dropout=0.)

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        if self.JK == "attention":
            self.attention_lstm.reset_parameters()

        self.output_decoder.apply(self.weights_init)

    def forward(self,
                h_list: list) -> Tensor:
        # JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim=-1),
                                               kernel_size=self.num_layers + 1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list = torch.cat(h_list, dim=0).transpose(0, 1)
            self.attention_lstm.flatten_parameters()
            attention_score, _ = self.attention_lstm(h_list)
            attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(-1)
            node_representation = torch.sum(h_list * attention_score, dim=1)

        return self.output_decoder(node_representation)

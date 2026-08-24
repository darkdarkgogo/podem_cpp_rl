import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax


class AGNNConv(MessagePassing):
    def __init__(self, in_channels, output_channels=None, wea=False, mlp=None, reverse=False):
        super().__init__(aggr="add", flow="target_to_source" if reverse else "source_to_target")
        if output_channels is None:
            output_channels = in_channels
        self.wea = wea
        if self.wea:
            self.edge_encoder = nn.Linear(16, output_channels)
        self.msg = nn.Linear(in_channels, output_channels)
        self.attn_lin = nn.Linear(output_channels + output_channels, 1)

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index: torch.Tensor, ptr: OptTensor, size_i):
        h_attn_q_i = self.msg(x_i)
        h_attn = self.msg(x_j)
        if self.wea:
            edge_embedding = self.edge_encoder(edge_attr)
            h_attn = h_attn + edge_embedding
        attention = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        attention = softmax(attention, index, ptr, size_i)
        return h_attn * attention

    def update(self, aggr_out):
        return aggr_out


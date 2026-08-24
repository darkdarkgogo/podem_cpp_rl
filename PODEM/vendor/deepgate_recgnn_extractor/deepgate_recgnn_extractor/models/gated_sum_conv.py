import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GatedSumConv(MessagePassing):
    def __init__(self, in_channels, output_channels=None, wea=False, mlp=None, reverse=False, mapper=None, gate=None):
        super().__init__(aggr="add", flow="target_to_source" if reverse else "source_to_target")
        if output_channels is None:
            output_channels = in_channels
        self.wea = wea
        self.mapper = nn.Linear(in_channels, output_channels) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(in_channels, output_channels), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        h_j = torch.cat((x_j, edge_attr), dim=1) if self.wea else x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out


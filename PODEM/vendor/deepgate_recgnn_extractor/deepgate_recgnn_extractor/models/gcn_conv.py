import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class AggConv(MessagePassing):
    def __init__(self, in_channels, output_channels=None, wea=False, mlp=None, reverse=False):
        super().__init__(aggr="add", flow="target_to_source" if reverse else "source_to_target")
        if output_channels is None:
            output_channels = in_channels
        self.wea = wea
        self.msg = nn.Linear(in_channels, output_channels) if mlp is None else mlp

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        if self.wea:
            return self.msg(torch.cat((x_j, edge_attr), dim=1))
        return self.msg(x_j)

    def update(self, aggr_out):
        return aggr_out


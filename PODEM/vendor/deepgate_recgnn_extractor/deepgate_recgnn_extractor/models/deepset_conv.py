import torch
from torch_geometric.nn import MessagePassing

from .mlp import MLP


class DeepSetConv(MessagePassing):
    def __init__(self, in_channels, output_channels=None, wea=False, mlp=None, reverse=False, mlp_post=None):
        super().__init__(aggr="add", flow="target_to_source" if reverse else "source_to_target")
        if output_channels is None:
            output_channels = in_channels
        self.wea = wea
        self.msg = MLP(in_channels, output_channels, output_channels, num_layer=3, p_drop=0.2) if mlp is None else mlp
        self.msg_post = mlp_post

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        if self.wea:
            return self.msg(torch.cat((x_j, edge_attr), dim=1))
        return self.msg(x_j)

    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        return aggr_out


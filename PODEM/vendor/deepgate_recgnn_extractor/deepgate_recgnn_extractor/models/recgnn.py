import torch
from torch import nn
from torch.nn import GRU, LSTM

from ..dag_utils import custom_backward_subgraph, subgraph
from .deepset_conv import DeepSetConv
from .gat_conv import AGNNConv
from .gated_sum_conv import GatedSumConv
from .gcn_conv import AggConv
from .mlp import MLP


_AGGR_FUNCTION_FACTORY = {
    "aggnconv": AGNNConv,
    "deepset": DeepSetConv,
    "gated_sum": GatedSumConv,
    "conv_sum": AggConv,
}

_UPDATE_FUNCTION_FACTORY = {
    "lstm": LSTM,
    "gru": GRU,
}


class RecGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.intermediate_supervision = args.intermediate_supervision
        self.reverse = args.reverse
        self.custom_backward = args.custom_backward
        self.use_edge_attr = args.use_edge_attr
        self.mask = args.mask

        self.num_aggr = args.num_aggr
        self.num_gate_types = args.num_gate_types
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        dim_aggr = self.dim_hidden
        if args.aggr_function not in _AGGR_FUNCTION_FACTORY:
            raise KeyError(f"Unsupported aggr function: {args.aggr_function}")

        aggr_forward_pre = nn.Linear(dim_aggr, self.dim_hidden)
        if args.aggr_function == "deepset":
            aggr_forward_post = nn.Linear(self.dim_hidden, self.dim_hidden)
            self.aggr_forward = _AGGR_FUNCTION_FACTORY[args.aggr_function](
                dim_aggr,
                self.dim_hidden,
                mlp=aggr_forward_pre,
                mlp_post=aggr_forward_post,
                wea=self.use_edge_attr,
            )
        else:
            self.aggr_forward = _AGGR_FUNCTION_FACTORY[args.aggr_function](
                dim_aggr,
                self.dim_hidden,
                mlp=aggr_forward_pre,
                wea=self.use_edge_attr,
            )

        if self.reverse:
            aggr_backward_pre = nn.Linear(dim_aggr, self.dim_hidden)
            if args.aggr_function == "deepset":
                aggr_backward_post = nn.Linear(self.dim_hidden, self.dim_hidden)
                self.aggr_backward = _AGGR_FUNCTION_FACTORY[args.aggr_function](
                    dim_aggr,
                    self.dim_hidden,
                    mlp=aggr_backward_pre,
                    mlp_post=aggr_backward_post,
                    wea=self.use_edge_attr,
                )
            else:
                self.aggr_backward = _AGGR_FUNCTION_FACTORY[args.aggr_function](
                    dim_aggr,
                    self.dim_hidden,
                    mlp=aggr_backward_pre,
                    reverse=True,
                    wea=self.use_edge_attr,
                )

        if args.update_function not in _UPDATE_FUNCTION_FACTORY:
            raise KeyError(f"Unsupported update function: {args.update_function}")

        if self.wx_update:
            self.update_forward = _UPDATE_FUNCTION_FACTORY[args.update_function](self.dim_node_feature + self.dim_hidden, self.dim_hidden)
            if self.reverse:
                self.update_backward = _UPDATE_FUNCTION_FACTORY[args.update_function](self.dim_node_feature + self.dim_hidden, self.dim_hidden)
        else:
            self.update_forward = _UPDATE_FUNCTION_FACTORY[args.update_function](self.dim_hidden, self.dim_hidden)
            if self.reverse:
                self.update_backward = _UPDATE_FUNCTION_FACTORY[args.update_function](self.dim_hidden, self.dim_hidden)

        self.gate_type_embedding = nn.Embedding(self.num_gate_types, self.dim_hidden)

        if self.wx_mlp:
            self.predictor = MLP(
                self.dim_hidden + self.dim_node_feature,
                self.dim_mlp,
                self.dim_pred,
                num_layer=self.num_fc,
                norm_layer=args.norm_layer,
                act_layer=args.activation_layer,
                sigmoid=False,
                tanh=False,
            )
        else:
            self.predictor = MLP(
                self.dim_hidden,
                self.dim_mlp,
                self.dim_pred,
                num_layer=self.num_fc,
                norm_layer=args.norm_layer,
                act_layer=args.activation_layer,
                sigmoid=False,
                tanh=False,
            )

        self.last_node_embedding = None

    def _log(self, message):
        if getattr(self.args, "verbose", False):
            print(message)

    def forward(self, graph):
        num_nodes = graph.num_nodes
        num_layers_f = max(graph.forward_level).item() + 1
        num_layers_b = max(graph.backward_level).item() + 1
        h_init = self.gate_type_embedding(graph.gate_type.to(self.device)).unsqueeze(0)
        self._log(
            "[DeepGate] Forward start: "
            f"num_nodes={num_nodes} forward_levels={num_layers_f} backward_levels={num_layers_b}"
        )

        if self.mask:
            h_true = torch.ones_like(h_init).to(self.device)
            h_false = -torch.ones_like(h_init).to(self.device)
            h_true.requires_grad = False
            h_false.requires_grad = False
            h_init = self.imply_mask(graph, h_init, h_true, h_false)
        else:
            h_true = None
            h_false = None

        if self.args.update_function == "lstm":
            preds = self._lstm_forward(graph, h_init, num_layers_f, num_layers_b, num_nodes)
        elif self.args.update_function == "gru":
            preds = self._gru_forward(graph, h_init, num_layers_f, num_layers_b, h_true, h_false)
        else:
            raise NotImplementedError("update_function must be lstm or gru")

        return preds

    def _lstm_forward(self, graph, h_init, num_layers_f, num_layers_b, num_nodes):
        x, edge_index = graph.x, graph.edge_index
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        node_state = (h_init, torch.zeros(1, num_nodes, self.dim_hidden).to(self.device))
        preds = []

        for round_idx in range(self.num_rounds):
            if getattr(self.args, "log_rounds", False):
                self._log(f"[DeepGate] Round {round_idx + 1}/{self.num_rounds}: forward propagation")
            for level_idx in range(1, num_layers_f):
                layer_mask = graph.forward_level == level_idx
                layer_nodes = graph.forward_index[layer_mask]
                layer_state = (
                    torch.index_select(node_state[0], dim=1, index=layer_nodes),
                    torch.index_select(node_state[1], dim=1, index=layer_nodes),
                )
                layer_edge_index, layer_edge_attr = subgraph(layer_nodes, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state[0].squeeze(0), layer_edge_index, layer_edge_attr)
                layer_msg = torch.index_select(msg, dim=0, index=layer_nodes)
                layer_x = torch.index_select(x, dim=0, index=layer_nodes)

                if self.wx_update:
                    _, layer_state = self.update_forward(torch.cat([layer_msg, layer_x], dim=1).unsqueeze(0), layer_state)
                else:
                    _, layer_state = self.update_forward(layer_msg.unsqueeze(0), layer_state)

                node_state[0][:, layer_nodes, :] = layer_state[0]
                node_state[1][:, layer_nodes, :] = layer_state[1]

            if self.reverse:
                if getattr(self.args, "log_rounds", False):
                    self._log(f"[DeepGate] Round {round_idx + 1}/{self.num_rounds}: backward propagation")
                for level_idx in range(1, num_layers_b):
                    layer_mask = graph.backward_level == level_idx
                    layer_nodes = graph.backward_index[layer_mask]
                    layer_state = (
                        torch.index_select(node_state[0], dim=1, index=layer_nodes),
                        torch.index_select(node_state[1], dim=1, index=layer_nodes),
                    )
                    if self.custom_backward:
                        layer_edge_index = custom_backward_subgraph(layer_nodes, edge_index, device=self.device, dim=0)
                        layer_edge_attr = None
                    else:
                        layer_edge_index, layer_edge_attr = subgraph(layer_nodes, edge_index, edge_attr, dim=0)

                    msg = self.aggr_backward(node_state[0].squeeze(0), layer_edge_index, layer_edge_attr)
                    layer_msg = torch.index_select(msg, dim=0, index=layer_nodes)
                    layer_x = torch.index_select(x, dim=0, index=layer_nodes)

                    if self.wx_update:
                        _, layer_state = self.update_backward(torch.cat([layer_msg, layer_x], dim=1).unsqueeze(0), layer_state)
                    else:
                        _, layer_state = self.update_backward(layer_msg.unsqueeze(0), layer_state)

                    node_state[0][:, layer_nodes, :] = layer_state[0]
                    node_state[1][:, layer_nodes, :] = layer_state[1]

            if self.intermediate_supervision:
                preds.append(self.predictor(node_state[0].squeeze(0)))

        node_embedding = node_state[0].squeeze(0)
        self.last_node_embedding = node_embedding
        pred = self.predictor(torch.cat([node_embedding, x], dim=1)) if self.wx_mlp else self.predictor(node_embedding)
        preds.append(pred)
        return preds

    def _gru_forward(self, graph, h_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        x, edge_index = graph.x, graph.edge_index
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        node_state = h_init
        preds = []

        for round_idx in range(self.num_rounds):
            if getattr(self.args, "log_rounds", False):
                self._log(f"[DeepGate] Round {round_idx + 1}/{self.num_rounds}: forward propagation")
            for level_idx in range(1, num_layers_f):
                layer_mask = graph.forward_level == level_idx
                layer_nodes = graph.forward_index[layer_mask]
                layer_state = torch.index_select(node_state, dim=1, index=layer_nodes)
                layer_edge_index, layer_edge_attr = subgraph(layer_nodes, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state.squeeze(0), layer_edge_index, layer_edge_attr)
                layer_msg = torch.index_select(msg, dim=0, index=layer_nodes)
                layer_x = torch.index_select(x, dim=0, index=layer_nodes)

                if self.wx_update:
                    _, layer_state = self.update_forward(torch.cat([layer_msg, layer_x], dim=1).unsqueeze(0), layer_state)
                else:
                    _, layer_state = self.update_forward(layer_msg.unsqueeze(0), layer_state)
                node_state[:, layer_nodes, :] = layer_state

                if self.mask:
                    node_state = self.imply_mask(graph, node_state, h_true, h_false)

            if self.reverse:
                if getattr(self.args, "log_rounds", False):
                    self._log(f"[DeepGate] Round {round_idx + 1}/{self.num_rounds}: backward propagation")
                for level_idx in range(1, num_layers_b):
                    layer_mask = graph.backward_level == level_idx
                    layer_nodes = graph.backward_index[layer_mask]
                    layer_state = torch.index_select(node_state, dim=1, index=layer_nodes)

                    if self.custom_backward:
                        layer_edge_index = custom_backward_subgraph(layer_nodes, edge_index, device=self.device, dim=0)
                        layer_edge_attr = None
                    else:
                        layer_edge_index, layer_edge_attr = subgraph(layer_nodes, edge_index, edge_attr, dim=0)

                    msg = self.aggr_backward(node_state.squeeze(0), layer_edge_index, layer_edge_attr)
                    layer_msg = torch.index_select(msg, dim=0, index=layer_nodes)
                    layer_x = torch.index_select(x, dim=0, index=layer_nodes)

                    if self.wx_update:
                        _, layer_state = self.update_backward(torch.cat([layer_msg, layer_x], dim=1).unsqueeze(0), layer_state)
                    else:
                        _, layer_state = self.update_backward(layer_msg.unsqueeze(0), layer_state)

                    node_state[:, layer_nodes, :] = layer_state

                    if self.mask:
                        node_state = self.imply_mask(graph, node_state, h_true, h_false)

            if self.intermediate_supervision:
                preds.append(self.predictor(node_state.squeeze(0)))

        node_embedding = node_state.squeeze(0)
        self.last_node_embedding = node_embedding
        pred = self.predictor(torch.cat([node_embedding, x], dim=1)) if self.wx_mlp else self.predictor(node_embedding)
        preds.append(pred)
        return preds

    def imply_mask(self, graph, h, h_true, h_false):
        true_mask = (graph.mask == 1.0).unsqueeze(0)
        false_mask = (graph.mask == 0.0).unsqueeze(0)
        normal_mask = (graph.mask == -1.0).unsqueeze(0)
        return h * normal_mask + h_true * true_mask + h_false * false_mask


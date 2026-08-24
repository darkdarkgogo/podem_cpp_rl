import random

import numpy
import torch


def top_sort(edge_index, graph_size):   ##用作给所有gate区分level
    node_ids = numpy.arange(graph_size, dtype=int)
    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    level = 0
    while unevaluated_nodes.any():
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        unready_children = child_nodes[unevaluated_mask]
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)
        if not nodes_to_evaluate.any():
            raise ValueError("Input graph is not a DAG")
        node_order[nodes_to_evaluate] = level
        unevaluated_nodes[nodes_to_evaluate] = False
        level += 1

    return torch.from_numpy(node_order).long()


def return_order_info(edge_index, num_nodes):
    node_indices = torch.LongTensor([index for index in range(num_nodes)])
    forward_level = top_sort(edge_index, num_nodes)
    reverse_edge_index = torch.LongTensor([list(edge_index[1]), list(edge_index[0])])
    backward_level = top_sort(reverse_edge_index, num_nodes)
    forward_index = node_indices
    backward_index = torch.LongTensor([index for index in range(num_nodes)])
    return forward_level, forward_index, backward_level, backward_index


def subgraph(target_idx, edge_index, edge_attr=None, dim=0):  #找出终点是当前节点的边
    edge_indices = []
    for node in target_idx:
        node_edges = edge_index[dim] == node
        edge_indices += [node_edges.nonzero().squeeze(-1)]
    edge_indices = torch.cat(edge_indices, dim=-1)
    local_edge_index = edge_index[:, edge_indices]
    local_edge_attr = edge_attr[edge_indices, :] if edge_attr is not None else None
    return local_edge_index, local_edge_attr


def custom_backward_subgraph(l_node, edge_index, device, dim=0):
    local_edge_index = torch.Tensor().to(device=device)
    for node in l_node:
        node_edges = edge_index[dim] == node
        subset_edges = torch.masked_select(edge_index, node_edges).reshape(edge_index.shape[0], -1)
        pos_count = torch.count_nonzero(node_edges)
        random_predecessor = random.randint(0, pos_count - 1)
        indices = torch.tensor([random_predecessor], device=device)
        subset_edges = torch.index_select(subset_edges, 1, indices)
        local_edge_index = torch.cat((local_edge_index, subset_edges), dim=1)

    local_edge_index = local_edge_index.to(torch.long)
    updated_edges = local_edge_index
    for node in l_node:
        node_vec = torch.tensor([node], device=device)
        node_edges = local_edge_index[0] == node
        predecessor = local_edge_index[1][node_edges]
        successor_edges = edge_index[1] == predecessor
        successors = edge_index[0][successor_edges]
        for successor in successors:
            if successor != node:
                successor_vec = torch.tensor([successor], device=device)
                new_edge = torch.stack((node_vec, successor_vec), dim=0)
                updated_edges = torch.cat((updated_edges, new_edge), dim=1)

    return updated_edges.to(torch.long)


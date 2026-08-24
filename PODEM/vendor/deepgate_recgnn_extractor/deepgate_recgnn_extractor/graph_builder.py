from dataclasses import dataclass

import torch

from .bench_parser import ParsedBench
from .circuit_utils import generate_prob_cont, generate_prob_obs, identify_reconvergence


@dataclass
class BuiltCircuit:
    x_data: list
    edge_index: list
    level_list: list
    fanin_list: list
    fanout_list: list
    gate_meta: list
    po_indices: list


def _new_node(name_to_index, x_data, node_name, gate_type):
    x_data.append([node_name, gate_type])
    name_to_index[node_name] = len(name_to_index)


def build_circuit(parsed: ParsedBench, gate_to_index: dict) -> BuiltCircuit:
    name_to_index = {}
    x_data = []
    edge_index = []

    for node_name in parsed.primary_inputs:
        _new_node(name_to_index, x_data, node_name, gate_to_index["input_pin"])

    for gate in parsed.gates:
        _new_node(name_to_index, x_data, gate["name"], gate_to_index[gate["gate_type"]])

    for output_name in parsed.primary_outputs:
        synthetic_output_name = f"output_pin_{output_name}"
        _new_node(name_to_index, x_data, synthetic_output_name, gate_to_index["output_pin"])

    for gate in parsed.gates:
        dst_idx = name_to_index[gate["name"]]
        for src_name in gate["fanins"]:
            if src_name not in name_to_index:
                raise ValueError(f"Unknown fanin '{src_name}' for gate '{gate['name']}'")
            edge_index.append([name_to_index[src_name], dst_idx])

    for output_name in parsed.primary_outputs:
        if output_name not in name_to_index:
            raise ValueError(f"Unknown primary output source '{output_name}'")
        output_idx = name_to_index[f"output_pin_{output_name}"]
        edge_index.append([name_to_index[output_name], output_idx])

    fanout_list = [[] for _ in x_data]
    fanin_list = [[] for _ in x_data]
    bfs_queue = []
    levels = [-1] * len(x_data)
    max_level = 0

    for index, node in enumerate(x_data):
        if node[1] == gate_to_index["input_pin"]:
            bfs_queue.append(index)
            levels[index] = 0

    for src_idx, dst_idx in edge_index:
        fanout_list[src_idx].append(dst_idx)
        fanin_list[dst_idx].append(src_idx)

    while bfs_queue:
        index = bfs_queue.pop()
        next_level = levels[index] + 1
        for next_node in fanout_list[index]:
            if levels[next_node] < next_level:
                levels[next_node] = next_level
                bfs_queue.insert(0, next_node)
                if levels[next_node] > max_level:
                    max_level = levels[next_node]

    if -1 in levels:
        raise ValueError("Cycle detected or unreachable nodes found in the circuit")

    level_list = [[] for _ in range(max_level + 1)]
    for index, level in enumerate(levels):
        x_data[index].append(level)
        level_list[level].append(index)

    pi_indices = level_list[0]
    x_data = generate_prob_cont(x_data, pi_indices, level_list, fanin_list)
    x_data = generate_prob_obs(x_data, level_list, fanin_list, fanout_list)
    x_data, _ = identify_reconvergence(x_data, level_list, fanin_list, fanout_list)

    po_indices = []
    primary_output_set = {f"output_pin_{name}" for name in parsed.primary_outputs}
    gate_meta = []
    for index, node in enumerate(x_data):
        name = node[0]
        is_pi = node[1] == gate_to_index["input_pin"]
        is_po = node[1] == gate_to_index["output_pin"] or (name in primary_output_set)
        if is_po:
            po_indices.append(index)
        gate_meta.append(
            {
                "index": index,
                "name": name,
                "gate_type_id": node[1],
                "level": node[2],
                "c1": node[3],
                "c0": node[4],
                "co": node[5],
                "fanout_flag": bool(node[6]),
                "fanout_count": len(fanout_list[index]),
                "is_reconvergent": bool(node[7]),
                "reconv_source_index": int(node[8]),
                "is_pi": is_pi,
                "is_po": is_po,
            }
        )

    return BuiltCircuit(
        x_data=x_data,
        edge_index=edge_index,
        level_list=level_list,
        fanin_list=fanin_list,
        fanout_list=fanout_list,
        gate_meta=gate_meta,
        po_indices=po_indices,
    )


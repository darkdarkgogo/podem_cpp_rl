from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rl_podem.cpp_bridge import _load_cpp_embedding_artifact
from rl_podem.gat_gru import DirectionalGATGRU
from rl_podem.smartatpg_features import load_circuit_graph
from smartatpg_portable import compute_embeddings, export_embeddings, load_graph, load_model


class LegacyModelTests(unittest.TestCase):
    def test_v6_v7_11d_models_keep_original_features_and_actor_inputs(self):
        import cpp_podem
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bench = root / "legacy.bench"
            bench.write_text("INPUT(a)\nINPUT(b)\nn=NOT(a)\ny=AND(n,b)\nOUTPUT(y)\n", encoding="utf-8")
            graph = load_circuit_graph(bench)
            portable_graph = load_graph(bench)
            for version in (6, 7):
                for variant in ("fanin_mean", "level_gat_gru"):
                    with self.subTest(version=version, variant=variant):
                        graph_state = {}
                        with torch.no_grad():
                            hidden = graph.x[:, :11]
                            if variant == "fanin_mean":
                                layer = nn.Linear(22, 11)
                                graph_state = {f"graph_encoder.layer.{key}": value for key, value in layer.state_dict().items()}
                                neighbors = torch.zeros_like(hidden)
                                for i, inputs in enumerate(graph.fanins):
                                    if inputs:
                                        neighbors[i] = hidden[list(inputs)].mean(0)
                                hidden = layer(torch.cat((hidden, neighbors), dim=1)).relu()
                            else:
                                for name, edges in (("forward_pass", graph.forward_level_edges[1:]),
                                                    ("reverse_pass", reversed(graph.reverse_level_edges))):
                                    direction = DirectionalGATGRU()
                                    direction.projection = nn.Linear(11, 11, bias=False)
                                    direction.attention = nn.Parameter(torch.randn(22) * 0.1)
                                    direction.gru = nn.GRUCell(11, 11)
                                    graph_state.update({f"graph_encoder.{name}.{key}": value for key, value in direction.state_dict().items()})
                                    for edge in edges:
                                        hidden = direction.update_level(hidden, edge)
                        actor_width = 12 if version == 7 and variant == "level_gat_gru" else 11
                        actor = nn.Sequential(nn.Linear(actor_width if version == 7 else 32, 32), nn.Tanh(), nn.Linear(32, 2))
                        state = {**graph_state, **{f"backtrace_actor.{key}": value for key, value in actor.state_dict().items()}}
                        if version == 6:
                            preencoder = nn.Sequential(nn.Linear(11, 32), nn.Tanh())
                            objective = nn.Embedding(2, 32)
                            state.update({f"gate_encoder.{key}": value for key, value in preencoder.state_dict().items()})
                            state["objective_value_embedding.weight"] = objective.weight
                        config = "fanin_mean_1x22x11" if variant == "fanin_mean" else "level_gat_gru_fwd_rev_11d_v1"
                        lines = [f"SMARTATPG_MODEL_V{version}", "backend smartatpg",
                                 "feature_schema SMARTATPG_FEATURES_V2_11D", f"encoder_variant {variant}",
                                 f"graph_config {config}", "gate_embedding_dim 11",
                                 f"actor_input_dim {actor_width}", "action_mask_dim 2",
                                 f"decision_state_dim {actor_width + 2}", f"snapshot {'1' * 64}",
                                 "best_round 1", "best_score -1,2,3,-4,1", "hidden_dim 32"]
                        for name, tensor in state.items():
                            rows, cols = (1, tensor.numel()) if tensor.ndim == 1 else tensor.shape
                            lines.append(f"tensor {name} {rows} {cols}")
                            lines.append(" ".join(format(float(v), ".9g") for v in tensor.flatten()))
                        model_path = root / "legacy.txt"
                        model_path.write_text("\n".join([*lines, "end"]) + "\n", encoding="utf-8")
                        model = load_model(model_path)
                        self.assertEqual(model.gate_embedding_dim, 11)
                        torch.testing.assert_close(torch.tensor(compute_embeddings(model, portable_graph)), hidden, atol=2e-6, rtol=2e-5)
                        emb_path = root / "legacy.emb"
                        export_embeddings(model, portable_graph, emb_path)
                        _, table, metadata = _load_cpp_embedding_artifact(emb_path, include_metadata=True)
                        self.assertEqual(metadata["gate_embedding_dim"], "11")
                        cpp_podem.validate_actor_artifacts(str(emb_path), str(model_path), graph.circuit_hash, list(graph.names), "smartatpg")
                        for value in (0, 1):
                            vector = table[graph.names[-1]]
                            with torch.no_grad():
                                actor_input = vector
                                if version == 6:
                                    actor_input = preencoder(vector) + objective(torch.tensor(value))
                                elif variant == "level_gat_gru":
                                    actor_input = torch.cat((vector, torch.tensor([float(value)])))
                                expected = actor(actor_input)
                            actual = torch.tensor(cpp_podem.score_actor_v2(str(model_path), vector.tolist(), value))
                            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

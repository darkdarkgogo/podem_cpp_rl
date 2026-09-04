import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rl_podem.smartatpg_features import COST_CAP, load_circuit_graph
from rl_podem.smartatpg import SmartATPGPolicy
from smartatpg_portable import load_graph


class SCOAPFeatureTests(unittest.TestCase):
    def check_graph(self, text, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scoap.bench"
            path.write_text(text, encoding="utf-8")
            graph = load_circuit_graph(path)
            portable = load_graph(path)
        self.assertEqual(graph.names, portable.names)
        torch.testing.assert_close(graph.x, torch.tensor(portable.features))
        self.assertTrue(torch.isfinite(graph.x).all())
        self.assertTrue(((graph.x >= 0) & (graph.x <= 1)).all())
        for name, values in expected.items():
            i = graph.name_to_index[name]
            self.assertEqual((graph.cc0[i], graph.cc1[i], graph.co[i]), values)
        return graph

    def test_all_gate_types_use_noncontrolling_side_costs(self):
        prefix = "INPUT(a)\nINPUT(b)\nINPUT(c)\nn=AND(b,c)\n"
        for kind, cc, side_co in (
            ("AND", (2, 5), 4), ("NAND", (5, 2), 4),
            ("OR", (4, 2), 3), ("NOR", (2, 4), 3),
        ):
            with self.subTest(gate=kind):
                self.check_graph(prefix + f"y={kind}(a,n)\nOUTPUT(y)\n", {
                    "y": (*cc, 0), "a": (1, 1, side_co),
                    "n": (2, 3, 2), "b": (1, 1, 4), "c": (1, 1, 4),
                })
        for kind, cc in (("BUF", (3, 4)), ("NOT", (4, 3))):
            with self.subTest(gate=kind):
                self.check_graph(prefix + f"y={kind}(n)\nOUTPUT(y)\n", {
                    "y": (*cc, 0), "n": (2, 3, 1),
                    "b": (1, 1, 3), "a": (1, 1, math.inf),
                })

    def test_fanout_minimum_output_boundary_and_dead_logic(self):
        self.check_graph(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\nn=AND(b,c)\n"
            "y=AND(a,n)\nz=OR(a,n)\ndead=BUF(y)\n"
            "OUTPUT(y)\nOUTPUT(z)\nOUTPUT(n)\nOUTPUT(c)\n",
            {"a": (1, 1, 3), "n": (2, 3, 0), "b": (1, 1, 2),
             "c": (1, 1, 0), "y": (2, 5, 0), "dead": (3, 6, math.inf)},
        )

    def test_repeated_pins_are_counted_by_position_and_costs_are_capped(self):
        self.check_graph("INPUT(a)\ny=AND(a,a)\nOUTPUT(y)\n", {
            "a": (1, 1, 2), "y": (2, 3, 0),
        })
        rows = ["INPUT(g0)"] + [f"g{i}=AND(g{i-1},g{i-1})" for i in range(1, 45)] + ["OUTPUT(g44)"]
        graph = self.check_graph("\n".join(rows), {})
        self.assertEqual(max(graph.cc1), COST_CAP)
        self.assertEqual(max(graph.co), COST_CAP)

    def test_co_column_reaches_trainable_encoder(self):
        graph = self.check_graph("INPUT(a)\ny=NOT(a)\nOUTPUT(y)\n", {})
        policy = SmartATPGPolicy()
        with torch.no_grad():
            policy.graph_encoder.layer.weight.zero_()
            policy.graph_encoder.layer.bias.fill_(1)
            policy.graph_encoder.layer.weight[0, 11] = 1
        embedding = policy.graph_embeddings(graph)
        torch.testing.assert_close(embedding[:, 0], 1 + graph.x[:, 11])
        embedding.sum().backward()
        self.assertGreater(float(policy.graph_encoder.layer.weight.grad[0, 11]), 0)


if __name__ == "__main__":
    unittest.main()

"""Focused checks for the NN-All PODEM baseline."""

from pathlib import Path
import tempfile
import unittest

import torch

from rl_podem.nn_all_features import (
    candidate_vector, load_nn_all_features, write_native_features,
)
from rl_podem.nn_all_training import (
    NnAllClassifier, PathSampleCollector, _circuit_split, _fit,
    export_native_model,
)


BENCH = """INPUT(a)
INPUT(b)
x = AND(a, b)
y = NAND(a, b)
z = OR(x, y)
q = NOR(a, b)
r = NOT(q)
OUTPUT(z)
OUTPUT(r)
"""


class NnAllFeaturesTest(unittest.TestCase):
    def test_feature_order_and_cop(self):
        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            features = load_nn_all_features(bench)
            index = features.indices
            self.assertEqual(features.cop[index["a"]], 0.5)
            self.assertEqual(features.cop[index["x"]], 0.25)
            self.assertEqual(features.cop[index["y"]], 0.75)
            self.assertEqual(features.cop[index["z"]], 0.8125)
            self.assertEqual(features.cop[index["q"]], 0.25)
            self.assertEqual(features.cop[index["r"]], 0.75)
            self.assertEqual(
                candidate_vector(features, "x", "a"),
                (0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            self.assertEqual(len(candidate_vector(features, "z", "x")), 11)
            artifact = Path(temporary) / "features.txt"
            write_native_features(artifact, features)
            self.assertIn("gate x AND 1", artifact.read_text(encoding="utf-8"))

    def test_collector_labels_backtracked_and_detected_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            collector = PathSampleCollector(load_nn_all_features(bench), 1)
            collector.event_callback({"event": "episode_start", "fault_id": "fault"})
            request = {
                "mode": "backtrace", "objective_name": "x",
                "candidate_names": ["a", "b"], "action_mask": [True, True],
                "heuristic_action": 0, "sequence": 1,
            }
            collector.decision_callback(request)
            collector.event_callback({"event": "backtrack", "decision_sequence": 1})
            request["sequence"] = 2
            collector.decision_callback(request)
            collector.event_callback({"event": "episode_end", "outcome": 1})
            self.assertEqual([sample[3] for sample in collector.samples], [0, 1])
            self.assertEqual([sample[2] for sample in collector.samples], ["fault", "fault"])

    def test_split_is_by_whole_circuit_and_model_export(self):
        train, holdout = _circuit_split([Path(f"c{index}.bench") for index in range(10)], 2026)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(holdout), 2)
        self.assertFalse(set(train) & set(holdout))
        positive = ((1.0,) * 11, 1, "p", 1)
        negative = ((0.0,) * 11, 1, "n", 0)
        model, epoch, loss = _fit([positive, negative] * 4, [positive, negative], 2026, 2)
        self.assertIsInstance(model, NnAllClassifier)
        self.assertIn(epoch, (1, 2))
        self.assertGreaterEqual(loss, 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "model.txt"
            export_native_model(artifact, model, "a" * 64)
            content = artifact.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("NN_ALL_PODEM_MODEL_V1\n"))
            self.assertIn("w1 660 ", content)

    def test_native_score_and_fault_order(self):
        import cpp_podem

        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            features = load_nn_all_features(bench)
            table = Path(temporary) / "features.txt"
            write_native_features(table, features)
            model = NnAllClassifier()
            artifact = Path(temporary) / "model.txt"
            export_native_model(artifact, model, "a" * 64)
            vector = candidate_vector(features, "x", "a")
            with torch.no_grad():
                expected = float(model(torch.tensor([vector], dtype=torch.float32))[0])
            actual = cpp_podem.score_nn_all_candidate(str(artifact), vector)
            self.assertAlmostEqual(actual, expected, places=5)
            fault_ids = [
                entry["fault_id"]
                for entry in cpp_podem.catalog_stuck_at(str(bench))["faults"][:3]
            ]
            records = cpp_podem.run_native_nn_all_validation(
                str(bench), str(table), str(artifact), 100, 2026,
                fault_ids, "small",
            )
            self.assertEqual([item["fault_id"] for item in records], fault_ids)
            self.assertTrue(all(item["atpg_seconds"] >= 0 for item in records))
            with self.assertRaisesRegex(Exception, "Cannot open NN-All model"):
                cpp_podem.run_native_nn_all_validation(
                    str(bench), str(table), str(artifact.with_name("missing.txt")),
                    100, 2026, fault_ids, "small",
                )


if __name__ == "__main__":
    unittest.main()

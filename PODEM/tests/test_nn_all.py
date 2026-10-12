"""Focused checks for the NN-All PODEM baseline."""

from pathlib import Path
import tempfile
import unittest

from rl_podem.nn_all_features import (
    candidate_vector, load_nn_all_features, write_native_features,
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


if __name__ == "__main__":
    unittest.main()

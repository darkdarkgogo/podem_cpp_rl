"""Check reporting with synthetic data; never touch live experiment outputs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment as experiment


class ReportTests(unittest.TestCase):
    def test_elapsed_time_migration_preserves_measurements_and_is_idempotent(self):
        checkpoint = {
            "best_label": "medium_sweep_1", "best_score": [-49, 1, 1, 1, 100],
            "config": {}, "progress": [], "agent": {"update_count": 0},
        }
        rows = []
        for index, model in enumerate(("heuristic", "old_v6", "gae")):
            row = {key: 10 for key in experiment.PATTERNS}
            row.update(circuit="c432", model=model, cpu_seconds=1.0 + index,
                       total_cpu_seconds=2.0 + index, wall_seconds=3.0 + index,
                       cpu_seconds_samples=[1.0 + index],
                       total_cpu_seconds_samples=[2.0 + index],
                       wall_seconds_samples=[3.0 + index])
            rows.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "benchmark"
            directory.mkdir()
            for name in ("raw_results.json", "summary.json"):
                experiment.save_json(directory / name, rows)
            experiment.save_json(directory / "protocol.json", {
                "cpu_time": "legacy label", "warmups_per_circuit_model": 1,
                "measured_repeats": 1,
            })
            original_raw = (directory / "raw_results.json").read_bytes()
            with patch.object(experiment, "RUN", root), patch(
                "torch.load", return_value=checkpoint
            ), contextlib.redirect_stdout(io.StringIO()):
                experiment.refresh_report(directory)
                first_raw = (directory / "raw_results.json").read_bytes()
                experiment.refresh_report(directory)
            self.assertEqual(first_raw, (directory / "raw_results.json").read_bytes())
            self.assertEqual(original_raw, (directory / "raw_results.json.original").read_bytes())
            results = json.loads(first_raw)
            for before, after in zip(rows, results):
                self.assertEqual(before["cpu_seconds"], after["atpg_seconds"])
                self.assertEqual(before["total_cpu_seconds"], after["native_total_seconds"])
                self.assertEqual(before["cpu_seconds_samples"], after["atpg_seconds_samples"])
                self.assertEqual(before["detected"], after["detected"])
                self.assertNotIn("cpu_seconds", after)
            report = (root / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("1 interleaved measured runs", report)
            self.assertIn("not process CPU time", report)
            self.assertNotIn("V6 CPU s", report)
            self.assertEqual(report, (directory / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

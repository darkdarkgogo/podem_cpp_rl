import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_smartatpg import _stage_circuit_copy, _summarize, percentage_change
from run_smartatpg_linux import relocate_manifest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LinuxManifestTests(unittest.TestCase):
    def test_windows_paths_are_relocated_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            data = root / "data"
            data.mkdir(parents=True)
            files = {}
            for name in (
                "teacher_training.json",
                "teacher_validation.json",
                "test.bench",
                "test.faultmap",
                "test_profile.json",
            ):
                path = data / name
                path.write_text(name, encoding="utf-8")
                files[name] = path
            windows = lambda name: f"C:\\old\\PODEM\\data\\{name}"
            manifest = {
                "teacher_training": windows("teacher_training.json"),
                "teacher_validation": windows("teacher_validation.json"),
                "teacher_sha256": {
                    "training": sha256(files["teacher_training.json"]),
                    "validation": sha256(files["teacher_validation.json"]),
                },
                "circuits": [{
                    "name": "test",
                    "circuit": windows("test.bench"),
                    "fault_map": windows("test.faultmap"),
                    "profile": windows("test_profile.json"),
                    "artifact_sha256": {
                        "circuit": sha256(files["test.bench"]),
                        "fault_map": sha256(files["test.faultmap"]),
                        "profile": sha256(files["test_profile.json"]),
                    },
                }],
            }
            source = root / "source.json"
            output = root / "run" / "manifest.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            relocated = relocate_manifest(source, output, root)
            self.assertEqual(
                Path(relocated["circuits"][0]["circuit"]),
                files["test.bench"].resolve(),
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), relocated)

            files["test.bench"].write_text("changed", encoding="utf-8")
            output.unlink()
            with self.assertRaisesRegex(ValueError, "hash changed"):
                relocate_manifest(source, output, root)


class BenchmarkSummaryTests(unittest.TestCase):
    def test_benchmark_stages_an_immutable_circuit_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "test.bench"
            source.parent.mkdir()
            source.write_text("INPUT(a)\n", encoding="utf-8")
            item = {
                "name": "test",
                "circuit": str(source),
                "artifact_sha256": {"circuit": sha256(source)},
            }
            staged = _stage_circuit_copy(item, root / "results")
            self.assertEqual(staged.read_bytes(), source.read_bytes())
            staged.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Staged benchmark circuit"):
                _stage_circuit_copy(item, root / "results")

    def test_percentage_change(self):
        self.assertEqual(percentage_change(100, 80), -20.0)
        self.assertEqual(percentage_change(0, 0), 0.0)
        self.assertIsNone(percentage_change(0, 1))

    def test_summary_uses_timing_medians_and_reports_reduction(self):
        models = ("heuristic", "rl_best", "rl_final")
        records = []
        for model, backtracks, seconds in (
            ("heuristic", 100, (2.0, 4.0)),
            ("rl_best", 90, (1.5, 2.5)),
            ("rl_final", 80, (1.0, 2.0)),
        ):
            for repeat, atpg_seconds in enumerate(seconds, 1):
                records.append({
                    "repeat": repeat,
                    "circuit": "test",
                    "model": model,
                    "detected": 10,
                    "total_faults": 10,
                    "equivalent_detected": 5,
                    "equivalent_faults": 5,
                    "aborted": 0,
                    "redundant": 0,
                    "backtracks": backtracks,
                    "backtrace_steps": backtracks * 10,
                    "test_vectors": 2,
                    "atpg_seconds": atpg_seconds,
                    "native_total_seconds": atpg_seconds + 0.1,
                    "wall_seconds": atpg_seconds + 0.2,
                })
        _, totals, comparisons = _summarize(
            records, {"circuits": [{"name": "test"}]}, models, 2
        )
        self.assertEqual(totals["heuristic"]["atpg_seconds"], 3.0)
        self.assertEqual(
            comparisons["rl_final"]["backtracks"]["reduction_percent"], 20.0
        )


if __name__ == "__main__":
    unittest.main()

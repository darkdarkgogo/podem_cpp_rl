import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_smartatpg import _stage_circuit_copy, _summarize, percentage_change
from run_smartatpg_benchmark_linux import main as run_benchmark_main
from run_smartatpg_training_linux import main as run_training_main
from smartatpg_portable import CIRCUITS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SplitLauncherTests(unittest.TestCase):
    def test_scripts_directory_contains_only_current_workflow(self):
        self.assertEqual(
            {path.name for path in SCRIPTS.glob("*.py")},
            {
                "benchmark_smartatpg.py",
                "build_native.py",
                "convert_binary_bench.py",
                "convert_full_scan_bench.py",
                "prepare_smartatpg_benchmark.py",
                "prepare_smartatpg_training.py",
                "run_smartatpg_benchmark_linux.py",
                "run_smartatpg_training_linux.py",
                "smartatpg_portable.py",
                "train_smartatpg.py",
            },
        )

    def test_training_launcher_only_trains_and_exports_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training"

            def fake_command(command, log_path, environment):
                if str(command[2]).endswith("train_smartatpg.py"):
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "model_best.txt").write_text("model", encoding="utf-8")
                return 0

            with (
                patch("run_smartatpg_training_linux.sys.platform", "linux"),
                patch("run_smartatpg_training_linux._check_cpp_extension"),
                patch(
                    "run_smartatpg_training_linux._tee_command",
                    side_effect=fake_command,
                ) as tee,
            ):
                run_training_main(["--output-dir", str(output)])
            commands = [call.args[0] for call in tee.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertTrue(str(commands[0][2]).endswith("prepare_smartatpg_training.py"))
            self.assertTrue(str(commands[1][2]).endswith("train_smartatpg.py"))
            self.assertTrue(str(commands[2][2]).endswith("prepare_smartatpg_benchmark.py"))
            flattened = " ".join(" ".join(map(str, command)) for command in commands)
            self.assertNotIn("build_native.py", flattened)
            self.assertNotIn("benchmark_smartatpg.py", flattened)
            self.assertEqual(commands[1][commands[1].index("--rounds") + 1], "20")

    def test_benchmark_launcher_only_builds_and_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
            output = root / "results"
            with (
                patch("run_smartatpg_benchmark_linux.sys.platform", "linux"),
                patch(
                    "run_smartatpg_benchmark_linux._tee_command", return_value=0
                ) as tee,
            ):
                run_benchmark_main([
                    str(bundle), "--output-dir", str(output), "--repeats", "2"
                ])
            commands = [call.args[0] for call in tee.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertTrue(str(commands[0][1]).endswith("build_native.py"))
            self.assertTrue(str(commands[1][2]).endswith("benchmark_smartatpg.py"))
            flattened = " ".join(" ".join(map(str, command)) for command in commands)
            self.assertNotIn("train_smartatpg.py", flattened)
            self.assertNotIn(".pth", flattened)

    def test_benchmark_runtime_has_no_torch_dependency(self):
        for name in (
            "run_smartatpg_benchmark_linux.py",
            "benchmark_smartatpg.py",
            "smartatpg_portable.py",
        ):
            source = (SCRIPTS / name).read_text(encoding="utf-8")
            self.assertNotIn("import torch", source)
            self.assertNotIn("from torch", source)

    def test_final_benchmark_lists_all_paper_circuits(self):
        self.assertEqual(len(CIRCUITS), 16)
        self.assertIn("s13207", CIRCUITS)
        self.assertIn("s15850", CIRCUITS)


class BenchmarkSummaryTests(unittest.TestCase):
    def test_benchmark_stages_an_immutable_circuit_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "test.bench"
            source.parent.mkdir()
            source.write_text("INPUT(a)\nOUTPUT(a)\n", encoding="utf-8")
            fault_map = root / "source" / "test.faultmap"
            fault_map.write_text("fault map\n", encoding="utf-8")
            item = {
                "name": "test",
                "circuit": str(source),
                "fault_map": str(fault_map),
                "artifact_sha256": {
                    "circuit": sha256(source),
                    "fault_map": sha256(fault_map),
                },
            }
            staged = _stage_circuit_copy(item, root / "results")
            self.assertEqual(staged.read_bytes(), source.read_bytes())
            staged.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Staged benchmark circuit"):
                _stage_circuit_copy(item, root / "results")

    def test_percentage_change(self):
        self.assertEqual(percentage_change(100, 80), 20.0)
        self.assertEqual(percentage_change(0, 0), 0.0)
        self.assertIsNone(percentage_change(0, 1))

    def test_summary_compares_atpg_time_only(self):
        models = ("heuristic", "rl_best")
        records = []
        for model, backtracks, seconds in (
            ("heuristic", 100, (2.0, 4.0)),
            ("rl_best", 90, (1.5, 2.5)),
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
                    "native_total_seconds": atpg_seconds + 100,
                    "wall_seconds": atpg_seconds + 200,
                })
        rows, totals, comparisons = _summarize(
            records, {"circuits": [{"name": "test"}]}, models, 2
        )
        self.assertEqual(totals["heuristic"]["atpg_seconds"], 3.0)
        self.assertNotIn("wall_seconds", totals["heuristic"])
        self.assertNotIn("native_total_seconds", rows[0])
        self.assertEqual(set(comparisons["rl_best"]), {
            "backtracks", "backtrace_steps", "atpg_seconds"
        })
        self.assertEqual(
            comparisons["rl_best"]["backtracks"]["reduction_percent"], 10.0
        )


if __name__ == "__main__":
    unittest.main()

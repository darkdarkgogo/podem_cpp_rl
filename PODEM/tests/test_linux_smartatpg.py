import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_smartatpg import (
    _parse_native_output, _stage_circuit_copy, _summarize,
    _summarize_preprocessing, _write_reports, percentage_change, run_benchmark,
)
from run_smartatpg_benchmark_linux import main as run_benchmark_main
from run_smartatpg_training_linux import main as run_training_main
from smartatpg_portable import CIRCUITS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SplitLauncherTests(unittest.TestCase):
    def test_benchmark_defaults_to_2000_backtracks(self):
        self.assertEqual(run_benchmark.__defaults__, (5, 14, 2000))

    def test_launchers_reject_500_backtracks(self):
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "requires backtrack limit 2000"),
        ):
            run_training_main(["--backtrack-limit", "500"])
        with (
            patch("run_smartatpg_benchmark_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "requires backtrack limit 2000"),
        ):
            run_benchmark_main(["unused", "--backtrack-limit", "500"])

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
                    model_dir = Path(command[4])
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "model_best.txt").write_text("model", encoding="utf-8")
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
            self.assertEqual(len(commands), 4)
            self.assertTrue(str(commands[0][2]).endswith("prepare_smartatpg_training.py"))
            self.assertTrue(str(commands[1][2]).endswith("train_smartatpg.py"))
            self.assertTrue(str(commands[2][2]).endswith("train_smartatpg.py"))
            self.assertTrue(str(commands[3][2]).endswith("prepare_smartatpg_benchmark.py"))
            flattened = " ".join(" ".join(map(str, command)) for command in commands)
            self.assertNotIn("build_native.py", flattened)
            self.assertNotIn("benchmark_smartatpg.py", flattened)
            for command in commands[1:3]:
                self.assertEqual(command[command.index("--rounds") + 1], "30")
            prepare = commands[0]
            self.assertEqual(
                prepare[prepare.index("--backtrack-limit") + 1], "2000"
            )

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
            benchmark = commands[1]
            self.assertEqual(
                benchmark[benchmark.index("--backtrack-limit") + 1], "2000"
            )

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

    def test_benchmark_rejects_any_other_backtrack_limit(self):
        with self.assertRaisesRegex(ValueError, "requires backtrack limit 2000"):
            run_benchmark("unused", "unused", "unused", backtrack_limit=500)

    def test_summary_compares_atpg_time_only(self):
        models = ("heuristic", "smartatpg_mean", "smartatpg_gat_gru")
        records = []
        for model, backtracks, seconds in (
            ("heuristic", 100, (2.0, 4.0)),
            ("smartatpg_mean", 90, (1.5, 2.5)),
            ("smartatpg_gat_gru", 80, (1.0, 2.0)),
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
                    "rl_select_calls": 0 if model == "heuristic" else backtracks,
                    "actor_forward_calls": (
                        0 if model == "heuristic"
                        else backtracks if model == "smartatpg_mean" else 10
                    ),
                    "rl_select_seconds": (
                        0.0 if model == "heuristic" else atpg_seconds / 1000
                    ),
                    "actor_forward_seconds": (
                        0.0 if model == "heuristic" else atpg_seconds / 2000
                    ),
                    "native_total_seconds": atpg_seconds + 100,
                    "wall_seconds": atpg_seconds + 200,
                })
        rows, totals, comparisons = _summarize(
            records, {"circuits": [{"name": "test"}]}, models, 2
        )
        self.assertEqual(totals["heuristic"]["atpg_seconds"], 3.0)
        self.assertNotIn("wall_seconds", totals["heuristic"])
        self.assertNotIn("native_total_seconds", rows[0])
        self.assertEqual(set(comparisons["smartatpg_mean"]), {
            "backtracks", "backtrace_steps", "atpg_seconds"
        })
        self.assertEqual(
            comparisons["smartatpg_mean"]["backtracks"]["reduction_percent"], 10.0
        )
        self.assertEqual(totals["smartatpg_mean"]["actor_forward_calls"], 90)
        self.assertEqual(totals["smartatpg_gat_gru"]["actor_forward_calls"], 10)
        self.assertAlmostEqual(
            totals["smartatpg_mean"]["average_rl_select_microseconds"],
            2.0 / 1000 * 1.0e6 / 90,
        )

    def test_native_actor_timing_is_parsed(self):
        output = """
#total number of detected faults = 10
#total number of gate faults (uncollapsed) = 12
#number of equivalent detected faults = 5
#number of equivalent gate faults (collapsed) = 6
#number of aborted faults = 1
#number of redundant faults = 1
#total number of backtracks = 20
#total number of backtrace steps = 30
#number of test vectors = 4
#number of RL backtrace policy selections = 40
#number of Actor forward evaluations = 10
#total RL backtrace policy selection time = 0.004000000s
#total Actor forward time = 0.003000000s
cputime for test pattern generation (one circuit): 1.250000s 1.500000s
"""
        parsed = _parse_native_output(output, Path("native.log"))
        self.assertEqual(parsed["rl_select_calls"], 40)
        self.assertEqual(parsed["actor_forward_calls"], 10)
        self.assertEqual(parsed["rl_select_seconds"], 0.004)
        self.assertEqual(parsed["actor_forward_seconds"], 0.003)

    def test_preprocessing_summary_separates_both_embeddings(self):
        summary = _summarize_preprocessing([
            {"operation": "graph_feature_build", "seconds": 1.0},
            {"operation": "graph_feature_build", "seconds": 2.0},
            {"operation": "graph_embedding", "model": "smartatpg_mean", "seconds": 3.0},
            {"operation": "graph_embedding", "model": "smartatpg_mean", "seconds": 4.0},
            {"operation": "graph_embedding", "model": "smartatpg_gat_gru", "seconds": 5.0},
        ])
        self.assertEqual(summary["graph_feature_build_seconds"], 3.0)
        self.assertEqual(summary["embedding_seconds"]["smartatpg_mean"], 7.0)
        self.assertEqual(summary["embedding_seconds"]["smartatpg_gat_gru"], 5.0)

    def test_final_report_contains_actor_and_embedding_timing(self):
        base = {
            "detected": 10, "total_faults": 12, "aborted": 1,
            "redundant": 1, "test_vectors": 2, "backtracks": 20,
            "backtrace_steps": 30, "atpg_seconds": 1.0,
            "fault_coverage": 10 / 12, "rl_select_calls": 0,
            "actor_forward_calls": 0, "average_rl_select_microseconds": 0.0,
            "average_actor_forward_microseconds": 0.0,
        }
        totals = {
            "heuristic": dict(base),
            "smartatpg_mean": dict(
                base, rl_select_calls=100, actor_forward_calls=100,
                average_rl_select_microseconds=1.5,
                average_actor_forward_microseconds=1.0,
            ),
            "smartatpg_gat_gru": dict(
                base, rl_select_calls=80, actor_forward_calls=20,
                average_rl_select_microseconds=0.5,
                average_actor_forward_microseconds=1.1,
            ),
        }
        comparisons = {
            name: {
                key: {"reduction_percent": 0.0, "improvement_percent": 0.0}
                for key in ("backtracks", "backtrace_steps", "atpg_seconds")
            }
            for name in ("smartatpg_mean", "smartatpg_gat_gru")
        }
        model = SimpleNamespace(
            encoder_variant="test", actor_input_dim=12, best_round=1,
            best_score=(1.0,), tensors={
                "backtrace_actor.0.weight": SimpleNamespace(rows=2, cols=3),
                "graph_encoder.weight": SimpleNamespace(rows=4, cols=5),
            },
        )
        preprocessing = [
            {"operation": "graph_feature_build", "seconds": 1.0},
            {"operation": "graph_embedding", "model": "smartatpg_mean", "seconds": 2.0},
            {"operation": "graph_embedding", "model": "smartatpg_gat_gru", "seconds": 3.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = _write_reports(
                Path(directory), [{"circuit": "test", "model": "heuristic"}],
                totals, comparisons,
                {"smartatpg_mean": model, "smartatpg_gat_gru": model},
                preprocessing,
            )
            report = (Path(directory) / "FINAL_RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("| smartatpg_mean | 100 | 100 | 0 |", report)
        self.assertIn("| smartatpg_gat_gru | 80 | 20 | 60 |", report)
        self.assertIn("| smartatpg_gat_gru embedding | 3.000000 |", report)
        self.assertEqual(
            result["preprocessing"]["embedding_seconds"]["smartatpg_mean"], 2.0
        )
        self.assertEqual(result["models"]["smartatpg_mean"]["parameter_count"], 26)
        self.assertEqual(
            result["models"]["smartatpg_mean"]["actor_parameter_count"], 6
        )


if __name__ == "__main__":
    unittest.main()

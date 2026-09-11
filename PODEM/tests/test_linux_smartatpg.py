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
from train_smartatpg import _validate_resume_config, _validate_round_target


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SplitLauncherTests(unittest.TestCase):
    def test_native_linux_build_enables_local_cpu_optimization(self):
        build_script = (SCRIPTS / "build_native.py").read_text(encoding="utf-8")
        setup_script = (SCRIPTS.parent / "setup.py").read_text(encoding="utf-8")
        for source in (build_script, setup_script):
            self.assertIn('"-O3"', source)
            self.assertIn('"-march=native"', source)
            self.assertNotIn('"-Ofast"', source)

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
                "plot_final_comparison.py",
                "run_smartatpg_benchmark_linux.py",
                "run_smartatpg_training_linux.py",
                "smartatpg_portable.py",
                "train_smartatpg.py",
            },
        )

    def test_training_launcher_only_trains_and_exports_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training"

            def fake_command(command, log_path, environment, prefix="", on_start=None):
                if str(command[2]).endswith("train_smartatpg.py"):
                    model_dir = Path(command[4])
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "model_best.txt").write_text("model", encoding="utf-8")
                    if "level_gat_gru" in command:
                        (model_dir / "model_best_reinforced.txt").write_text(
                            "model", encoding="utf-8"
                        )
                return 0

            with (
                patch("run_smartatpg_training_linux.sys.platform", "linux"),
                patch("run_smartatpg_training_linux._check_cpp_extension"),
                patch("run_smartatpg_training_linux.torch.cuda.device_count", return_value=4),
                patch(
                    "run_smartatpg_training_linux._tee_command",
                    side_effect=fake_command,
                ) as tee,
            ):
                run_training_main(["--output-dir", str(output)])
            commands = [call.args[0] for call in tee.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertTrue(str(commands[0][2]).endswith("prepare_smartatpg_training.py"))
            self.assertTrue(str(commands[2][2]).endswith("prepare_smartatpg_benchmark.py"))
            train_calls = [
                call for call in tee.call_args_list
                if str(call.args[0][2]).endswith("train_smartatpg.py")
            ]
            self.assertEqual(len(train_calls), 1)
            gpu_by_encoder = {
                call.args[0][call.args[0].index("--encoder") + 1]:
                call.args[2]["CUDA_VISIBLE_DEVICES"]
                for call in train_calls
            }
            self.assertEqual(gpu_by_encoder, {"level_gat_gru": "0"})
            flattened = " ".join(" ".join(map(str, command)) for command in commands)
            self.assertNotIn("build_native.py", flattened)
            self.assertNotIn("benchmark_smartatpg.py", flattened)
            for command in (call.args[0] for call in train_calls):
                self.assertEqual(command[command.index("--rounds") + 1], "8")
            gat_command = next(
                command for command in (call.args[0] for call in train_calls)
                if "level_gat_gru" in command
            )
            self.assertEqual(
                gat_command[gat_command.index("--reinforcement-rounds") + 1], "5"
            )
            self.assertTrue(
                str(commands[2][4]).endswith("model_best_reinforced.txt")
            )
            prepare = commands[0]
            self.assertEqual(prepare[prepare.index("--count") + 1], "50")
            self.assertEqual(
                prepare[prepare.index("--backtrack-limit") + 1], "2000"
            )

    def test_training_launcher_requires_one_selected_gpu(self):
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            patch("run_smartatpg_training_linux._check_cpp_extension"),
            patch("run_smartatpg_training_linux.torch.cuda.device_count", return_value=0),
            self.assertRaisesRegex(RuntimeError, "only 0 CUDA device"),
        ):
            run_training_main([])
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            patch("run_smartatpg_training_linux._check_cpp_extension"),
            patch("run_smartatpg_training_linux.torch.cuda.device_count", return_value=1),
            self.assertRaisesRegex(RuntimeError, "only 1 CUDA device"),
        ):
            run_training_main(["--gpu", "1"])
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "between 1 and 5"),
        ):
            run_training_main(["--gat-reinforcement-rounds", "6"])
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "exactly 8"),
        ):
            run_training_main(["--rounds", "7"])

    def test_training_round_target_cannot_change_on_resume(self):
        saved = {"rounds": 20, "seed": 2026, "encoder_variant": "level_gat_gru"}
        current = {"rounds": 8, "seed": 2026, "encoder_variant": "level_gat_gru"}
        with self.assertRaisesRegex(ValueError, "Training configuration changed"):
            _validate_resume_config(saved, current)
        saved = dict(current)
        self.assertEqual(_validate_resume_config(saved, current), 8)
        current["seed"] = 14
        with self.assertRaisesRegex(ValueError, "Training configuration changed"):
            _validate_resume_config(saved, current)
        _validate_round_target(8, 99, 8)
        _validate_round_target(9, 0, 8)
        with self.assertRaisesRegex(ValueError, "already started round 9"):
            _validate_round_target(9, 1, 8)
        with self.assertRaisesRegex(ValueError, "already started round 9"):
            _validate_round_target(10, 0, 8)

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
        models = ("scoap_heuristic", "smartatpg_gat_gru")
        records = []
        for model, backtracks, seconds in (
            ("scoap_heuristic", 100, (2.0, 4.0)),
            ("smartatpg_gat_gru", 80, (1.0, 2.0)),
        ):
            for repeat, atpg_seconds in enumerate(seconds, 1):
                records.append({
                    "repeat": repeat,
                    "circuit": "test",
                    "model": model,
                    "detected": 10,
                    "redundant_uncollapsed": 0,
                    "successful_faults": 10,
                    "total_faults": 10,
                    "equivalent_detected": 5,
                    "equivalent_faults": 5,
                    "aborted": 0,
                    "redundant": 0,
                    "backtracks": backtracks,
                    "backtrace_steps": backtracks * 10,
                    "test_vectors": 2,
                    "atpg_seconds": atpg_seconds,
                    "rl_select_calls": 0 if model == "scoap_heuristic" else backtracks,
                    "actor_forward_calls": (
                        0 if model == "scoap_heuristic" else 10
                    ),
                    "rl_select_seconds": (
                        0.0 if model == "scoap_heuristic" else atpg_seconds / 1000
                    ),
                    "actor_forward_seconds": (
                        0.0 if model == "scoap_heuristic" else atpg_seconds / 2000
                    ),
                    "native_total_seconds": atpg_seconds + 100,
                    "wall_seconds": atpg_seconds + 200,
                })
        rows, totals, comparisons = _summarize(
            records, {"circuits": [{"name": "test"}]}, models, 2
        )
        self.assertEqual(totals["scoap_heuristic"]["atpg_seconds"], 3.0)
        self.assertNotIn("wall_seconds", totals["scoap_heuristic"])
        self.assertNotIn("native_total_seconds", rows[0])
        self.assertEqual(set(comparisons["smartatpg_gat_gru"]), {
            "backtracks", "backtrace_steps", "atpg_seconds"
        })
        self.assertEqual(
            comparisons["smartatpg_gat_gru"]["backtracks"]["reduction_percent"], 20.0
        )
        self.assertEqual(totals["smartatpg_gat_gru"]["actor_forward_calls"], 10)
        self.assertAlmostEqual(
            totals["smartatpg_gat_gru"]["average_rl_select_microseconds"],
            1.5 / 1000 * 1.0e6 / 80,
        )

    def test_native_actor_timing_is_parsed(self):
        output = """
#total number of detected faults = 10
#total number of redundant faults (uncollapsed) = 1
#total number of successful faults = 11
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
        self.assertEqual(parsed["redundant_uncollapsed"], 1)
        self.assertEqual(parsed["successful_faults"], 11)
        self.assertEqual(parsed["rl_select_calls"], 40)
        self.assertEqual(parsed["actor_forward_calls"], 10)
        self.assertEqual(parsed["rl_select_seconds"], 0.004)
        self.assertEqual(parsed["actor_forward_seconds"], 0.003)

    def test_preprocessing_summary_contains_only_gat_embedding(self):
        summary = _summarize_preprocessing([
            {"operation": "graph_feature_build", "seconds": 1.0},
            {"operation": "graph_feature_build", "seconds": 2.0},
            {"operation": "graph_embedding", "model": "smartatpg_gat_gru", "seconds": 5.0},
        ])
        self.assertEqual(summary["graph_feature_build_seconds"], 3.0)
        self.assertEqual(summary["embedding_seconds"]["smartatpg_gat_gru"], 5.0)

    def test_final_report_contains_actor_and_embedding_timing(self):
        base = {
            "detected": 10, "total_faults": 12, "aborted": 1,
            "redundant": 1, "redundant_uncollapsed": 1,
            "successful_faults": 11, "test_vectors": 2, "backtracks": 20,
            "backtrace_steps": 30, "atpg_seconds": 1.0,
            "fault_coverage": 11 / 12, "rl_select_calls": 0,
            "actor_forward_calls": 0, "average_rl_select_microseconds": 0.0,
            "average_actor_forward_microseconds": 0.0,
        }
        totals = {
            "scoap_heuristic": dict(base),
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
            for name in ("smartatpg_gat_gru",)
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
            {"operation": "graph_embedding", "model": "smartatpg_gat_gru", "seconds": 3.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = _write_reports(
                Path(directory), [{"circuit": "test", "model": "scoap_heuristic"}],
                totals, comparisons,
                {"smartatpg_gat_gru": model},
                preprocessing,
            )
            report = (Path(directory) / "FINAL_RESULTS.md").read_text(encoding="utf-8")
        self.assertNotIn("smartatpg_mean", report)
        self.assertIn("scoap_heuristic", report)
        self.assertIn("| smartatpg_gat_gru | 80 | 20 | 60 |", report)
        self.assertIn("| smartatpg_gat_gru embedding | 3.000000 |", report)
        self.assertEqual(result["models"]["smartatpg_gat_gru"]["parameter_count"], 26)
        self.assertEqual(
            result["models"]["smartatpg_gat_gru"]["actor_parameter_count"], 6
        )


if __name__ == "__main__":
    unittest.main()

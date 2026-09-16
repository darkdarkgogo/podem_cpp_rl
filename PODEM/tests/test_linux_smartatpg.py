import hashlib
import json
import sys
import tempfile
import threading
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
from compare_smartatpg_validation import (
    ScoapValidationEvaluator, build_validation_comparison,
)
from run_smartatpg_benchmark_linux import main as run_benchmark_main
from run_smartatpg_training_linux import main as run_training_main
from smartatpg_portable import CIRCUITS

from run_dual_smartatpg_training_linux import (
    _run_parallel_training, main as run_dual_training_main,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _SplitLauncherInventoryTests:
    def test_native_linux_build_enables_local_cpu_optimization(self):
        build_script = (SCRIPTS / "build_native.py").read_text(encoding="utf-8")
        setup_script = (SCRIPTS.parent / "setup.py").read_text(encoding="utf-8")
        for source in (build_script, setup_script):
            self.assertIn('"-O3"', source)
            self.assertIn('"-march=native"', source)
            self.assertNotIn('"-Ofast"', source)

    def test_benchmark_defaults_to_200_backtracks(self):
        self.assertEqual(run_benchmark.__defaults__, (5, 14, 200))

    def test_launchers_reject_500_backtracks(self):
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "requires backtrack limit 200"),
        ):
            run_training_main(["--backtrack-limit", "500"])
        with (
            patch("run_smartatpg_benchmark_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "requires backtrack limit 200"),
        ):
            run_benchmark_main(["unused", "--backtrack-limit", "500"])

    def test_scripts_directory_contains_only_current_workflow(self):
        self.assertEqual(
            {path.name for path in SCRIPTS.glob("*.py")},
            {
                "benchmark_smartatpg.py",
                "build_native.py",
                "compare_smartatpg_validation.py",
                "convert_binary_bench.py",
                "convert_full_scan_bench.py",
                "generate_normal_bench_dataset.py",
                "prepare_smartatpg_benchmark.py",
                "prepare_smartatpg_training.py",
                "plot_final_comparison.py",
                "run_smartatpg_benchmark_linux.py",
                "run_smartatpg_training_linux.py",
                "run_dual_smartatpg_training_linux.py",
                "smartatpg_portable.py",
                "train_smartatpg.py",
            },
        )


class ValidationComparisonTests(unittest.TestCase):
    IDENTITY_KEYS = {
        "format", "manifest_hash", "encoder_variant", "normal_rounds",
        "faults_per_update", "k_epochs", "backtrack_limit",
        "validation_catalog_hash", "validation_circuits",
    }

    @staticmethod
    def _summary(round_number, backtracks, is_best, seconds=1.0):
        return {
            "round": round_number,
            "episodes": 2,
            "detected_faults": 1,
            "redundant_faults": 1,
            "aborted_faults": 0,
            "backtracks_total": backtracks,
            "backtrace_steps_total": backtracks * 2,
            "return_total": 0.0,
            "test_vectors": 1,
            "atpg_seconds": seconds,
            "fault_coverage": 0.5,
            "backtracks_mean": backtracks / 2,
            "backtrace_steps_mean": backtracks,
            "return_mean": 0.0,
            "circuits": [{
                "circuit": "v",
                "episodes": 2,
                "detected_faults": 1,
                "redundant_faults": 1,
                "aborted_faults": 0,
                "backtracks_total": backtracks,
                "backtrace_steps_total": backtracks * 2,
                "return_total": 0.0,
                "test_vectors": 1,
                "atpg_seconds": seconds,
                "fault_coverage": 0.5,
                "backtracks_mean": backtracks / 2,
                "backtrace_steps_mean": backtracks,
                "return_mean": 0.0,
            }],
            "is_best": is_best,
        }

    def _write_run(
        self, root, name, encoder, best_round, manifest_hash, catalog_hash
    ):
        directory = root / name
        directory.mkdir()
        identity = {
            "format": "SMARTATPG_VALIDATION_IDENTITY_V1",
            "manifest_hash": manifest_hash,
            "encoder_variant": encoder,
            "normal_rounds": 2,
            "faults_per_update": 8,
            "k_epochs": 1,
            "backtrack_limit": 200,
            "validation_catalog_hash": catalog_hash,
            "validation_circuits": ["v"],
        }
        self.assertEqual(set(identity), self.IDENTITY_KEYS)
        (directory / "validation_identity.json").write_text(
            json.dumps(identity), encoding="utf-8"
        )
        rounds = [
            self._summary(1, 8, best_round == 1),
            self._summary(2, 4, best_round == 2),
        ]
        (directory / "validation_metrics.json").write_text(
            json.dumps(rounds), encoding="utf-8"
        )
        return directory

    @staticmethod
    def _baseline_record(_evaluator, item, fault_id, _limit, _seed):
        detected = int(fault_id == "f0")
        return {
            "circuit": item["name"], "fault_id": fault_id,
            "outcome": 1 if detected else 0, "detected": detected,
            "redundant": 1 - detected, "aborted": 0,
            "backtracks": 10, "backtrace_steps": 20, "return": 0.0,
            "test_vectors": detected, "atpg_seconds": 2.0,
        }

    def _build(self, root, **identity_changes):
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"format": "test"}), encoding="utf-8")
        circuits = [{
            "name": "v", "circuit": str(root / "v.bench"),
            "episode_fault_ids": ["f0", "f1"],
        }]
        catalog_payload = [{"name": "v", "fault_ids": ["f0", "f1"]}]
        catalog_hash = hashlib.sha256(json.dumps(
            catalog_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        gat = self._write_run(
            root, "gat", "level_gat_gru", 1, sha256(manifest), catalog_hash
        )
        mean = self._write_run(
            root, "mean", "fanin_mean", 2, sha256(manifest), catalog_hash
        )
        for run_name, changes in identity_changes.items():
            path = {"gat": gat, "mean": mean}[run_name] / "validation_identity.json"
            identity = json.loads(path.read_text(encoding="utf-8"))
            identity.update(changes)
            path.write_text(json.dumps(identity), encoding="utf-8")
        output = root / "comparison"
        patches = (
            patch(
                "compare_smartatpg_validation._resolve_circuit_records",
                return_value=([], circuits),
            ),
            patch(
                "compare_smartatpg_validation._load_validation_catalogs",
                side_effect=lambda _manifest, items: items,
            ),
            patch(
                "compare_smartatpg_validation._evaluate_fault",
                side_effect=self._baseline_record,
            ),
        )
        return manifest, gat, mean, output, patches

    def test_builds_two_run_comparison_with_one_cached_scoap_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, gat, mean, output, patches = self._build(root)
            with patches[0], patches[1], patches[2] as evaluate:
                result = build_validation_comparison(
                    manifest, gat, mean, output
                )
                resumed = build_validation_comparison(
                    manifest, gat, mean, output
                )
            self.assertEqual(result, resumed)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(
                result["format"], "SMARTATPG_DUAL_VALIDATION_COMPARISON_V1"
            )
            self.assertEqual(result["models"]["smartatpg_gat_gru"]["best_round"], 1)
            self.assertEqual(result["models"]["smartatpg_mean"]["best_round"], 2)
            self.assertEqual(len(result["comparisons"]), 4)
            self.assertEqual(len(result["direct_comparisons"]), 2)
            self.assertEqual(
                {row["scope"] for row in result["comparisons"][0]["rows"]},
                {"total", "circuit"},
            )
            self.assertTrue((output / "scoap_validation.json").is_file())
            self.assertTrue((output / "validation_comparison.json").is_file())
            self.assertTrue((output / "validation_comparison.csv").is_file())
            csv_text = (output / "validation_comparison.csv").read_text("utf-8")
            for field in (
                "detected_faults", "redundant_faults", "aborted_faults",
                "test_vectors", "backtracks_total", "backtrace_steps_total",
                "return_total", "atpg_seconds", "scoap_backtracks_total",
                "gat_minus_mean",
            ):
                self.assertIn(field, csv_text)

    def test_scoap_evaluator_uses_native_backtrace_heuristic_protocol(self):
        captured = {}

        def run_stuck_at(*args):
            captured["args"] = args
            return {"episodes": 1, "atpg_seconds": 0.25}

        event_callback = object()
        with patch.dict(sys.modules, {
            "cpp_podem": SimpleNamespace(run_stuck_at=run_stuck_at),
        }):
            result = ScoapValidationEvaluator().run(
                "v.bench", backtrack_limit=200, seed=2026,
                fault_ids=["f0"], use_scoap=False,
                event_callback=event_callback,
            )
        args = captured["args"]
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(args[2], event_callback)
        self.assertEqual(args[3:6], (200, 2026, ["f0"]))
        self.assertEqual(args[7], "backtrace_rl")
        self.assertIs(args[9], True)
        self.assertEqual(args[1]({"heuristic_action": 1}), 1)

    def test_rejects_incompatible_run_identity_and_rounds(self):
        cases = (
            ("manifest_hash", {"mean": {"manifest_hash": "c" * 64}}, "manifest_hash"),
            ("catalog_hash", {"mean": {"validation_catalog_hash": "c" * 64}},
             "validation_catalog_hash"),
            ("wrong_encoder", {"gat": {"encoder_variant": "fanin_mean"}},
             "Wrong encoder variant"),
            ("wrong_protocol", {"mean": {"faults_per_update": 4}},
             "Wrong V8 training protocol"),
        )
        for label, changes, message in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, gat, mean, output, patches = self._build(root, **changes)
                with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                    ValueError, message
                ):
                    build_validation_comparison(manifest, gat, mean, output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, gat, mean, output, patches = self._build(root)
            metrics_path = mean / "validation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_path.write_text(json.dumps(metrics[:1]), encoding="utf-8")
            with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                ValueError, "Incomplete validation rounds"
            ):
                build_validation_comparison(manifest, gat, mean, output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, gat, mean, output, patches = self._build(root)
            metrics_path = gat / "validation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics[1]["is_best"] = True
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                ValueError, "exactly one best"
            ):
                build_validation_comparison(manifest, gat, mean, output)

    def test_rejects_incomplete_or_inconsistent_round_metrics(self):
        def missing_fault_episode(round_summary):
            round_summary["circuits"][0]["episodes"] = 1

        def inconsistent_total(round_summary):
            round_summary["backtracks_total"] += 2
            round_summary["backtracks_mean"] = (
                round_summary["backtracks_total"] / round_summary["episodes"]
            )

        def inconsistent_coverage(round_summary):
            round_summary["circuits"][0]["fault_coverage"] = 0.75

        def inconsistent_mean(round_summary):
            round_summary["circuits"][0]["backtracks_mean"] += 1.0

        cases = (
            ("missing_fault_episode", missing_fault_episode),
            ("inconsistent_total", inconsistent_total),
            ("inconsistent_coverage", inconsistent_coverage),
            ("inconsistent_mean", inconsistent_mean),
        )
        for label, mutate in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, gat, mean, output, patches = self._build(root)
                metrics_path = gat / "validation_metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                mutate(metrics[0])
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                    ValueError, "validation metrics"
                ):
                    build_validation_comparison(manifest, gat, mean, output)

    def test_rejects_invalid_metric_types_and_ranges(self):
        def set_both(round_summary, key, value):
            round_summary[key] = value
            round_summary["circuits"][0][key] = value

        def bool_count(round_summary):
            round_summary["circuits"][0]["detected_faults"] = True
            round_summary["circuits"][0]["test_vectors"] = True

        def float_counts(round_summary):
            set_both(round_summary, "detected_faults", 1.5)
            set_both(round_summary, "redundant_faults", 0.5)
            set_both(round_summary, "test_vectors", 1.5)
            set_both(round_summary, "fault_coverage", 0.75)

        def negative_counts(round_summary):
            set_both(round_summary, "detected_faults", -1)
            set_both(round_summary, "redundant_faults", 3)
            set_both(round_summary, "test_vectors", -1)
            set_both(round_summary, "fault_coverage", -0.5)

        def float_work(round_summary):
            set_both(round_summary, "backtracks_total", 1.5)
            set_both(round_summary, "backtracks_mean", 0.75)

        def negative_work(round_summary):
            set_both(round_summary, "backtrace_steps_total", -2)
            set_both(round_summary, "backtrace_steps_mean", -1.0)

        def negative_time(round_summary):
            set_both(round_summary, "atpg_seconds", -1.0)

        def infinite_time(round_summary):
            set_both(round_summary, "atpg_seconds", float("inf"))

        def infinite_return(round_summary):
            set_both(round_summary, "return_total", float("inf"))
            set_both(round_summary, "return_mean", float("inf"))

        cases = (
            ("bool_count", bool_count),
            ("float_counts", float_counts),
            ("negative_counts", negative_counts),
            ("float_work", float_work),
            ("negative_work", negative_work),
            ("negative_time", negative_time),
            ("infinite_time", infinite_time),
            ("infinite_return", infinite_return),
        )
        for label, mutate in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, gat, mean, output, patches = self._build(root)
                metrics_path = gat / "validation_metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                mutate(metrics[0])
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                    ValueError, "validation metrics"
                ):
                    build_validation_comparison(manifest, gat, mean, output)

class SplitLauncherTests(_SplitLauncherInventoryTests, unittest.TestCase):
    def test_training_launcher_only_trains_and_exports_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training"

            def fake_command(command, log_path, environment, prefix="", on_start=None):
                if str(command[2]).endswith("prepare_smartatpg_training.py"):
                    preparation = Path(command[4])
                    preparation.mkdir(parents=True, exist_ok=True)
                    (preparation / "training_manifest.json").write_text(
                        json.dumps({
                            "backtrack_limit": 200, "normal_rounds": 2,
                            "faults_per_update": 8, "k_epochs": 1,
                            "train": [], "validation": [],
                            "train_circuits": [{"name": "t"}],
                            "validation_circuits": [{"name": "v"}],
                        }),
                        encoding="utf-8",
                    )
                if str(command[2]).endswith("train_smartatpg.py"):
                    model_dir = Path(command[4])
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "model_best.txt").write_text("model", encoding="utf-8")
                    (model_dir / "model_latest.txt").write_text(
                        "model", encoding="utf-8"
                    )
                return 0

            with (
                patch("run_smartatpg_training_linux.sys.platform", "linux"),
                patch("run_smartatpg_training_linux._check_cpp_extension"),
                patch("run_smartatpg_training_linux.torch", SimpleNamespace(
                    __version__="test",
                    cuda=SimpleNamespace(
                        device_count=lambda: 4, is_available=lambda: True
                    ),
                )),
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
                self.assertEqual(command[command.index("--rounds") + 1], "2")
                self.assertEqual(command[command.index("--k-epochs") + 1], "1")
            gat_command = next(
                command for command in (call.args[0] for call in train_calls)
                if "level_gat_gru" in command
            )
            self.assertNotIn("--reinforcement-rounds", gat_command)
            self.assertIn("--resume", gat_command)
            train_log = next(
                call.args[1] for call in tee.call_args_list
                if str(call.args[0][2]).endswith("train_smartatpg.py")
            )
            self.assertEqual(Path(train_log), output / "train_gat_gru.log")
            self.assertTrue(
                str(commands[2][4]).endswith("model_best.txt")
            )
            prepare = commands[0]
            self.assertTrue(str(prepare[3]).endswith("data"))
            self.assertNotIn("--count", prepare)
            self.assertNotIn("--backtrack-limit", prepare)
            metadata = json.loads(
                (output / "training_run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["training_protocol"]["backtrack_limit"], 200)
            self.assertEqual(metadata["training_protocol"]["normal_rounds"], 2)
            self.assertEqual(metadata["training_protocol"]["faults_per_update"], 8)
            self.assertEqual(metadata["training_protocol"]["k_epochs"], 1)
            self.assertEqual(metadata["training_protocol"]["training_circuit_count"], 1)
            self.assertEqual(metadata["training_protocol"]["validation_circuit_count"], 1)

    def test_training_launcher_requires_one_selected_gpu(self):
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            patch("run_smartatpg_training_linux._check_cpp_extension"),
            patch("run_smartatpg_training_linux.torch", SimpleNamespace(
                cuda=SimpleNamespace(device_count=lambda: 0)
            )),
            self.assertRaisesRegex(RuntimeError, "only 0 CUDA device"),
        ):
            run_training_main([])
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            patch("run_smartatpg_training_linux._check_cpp_extension"),
            patch("run_smartatpg_training_linux.torch", SimpleNamespace(
                cuda=SimpleNamespace(device_count=lambda: 1)
            )),
            self.assertRaisesRegex(RuntimeError, "only 1 CUDA device"),
        ):
            run_training_main(["--gpu", "1"])
        with (
            patch("run_smartatpg_training_linux.sys.platform", "linux"),
            self.assertRaisesRegex(ValueError, "exactly 2"),
        ):
            run_training_main(["--rounds", "4"])

    def test_training_launcher_passes_cross_manifest_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "training"
            source = root / "source.pth"
            source.write_bytes(b"checkpoint")

            def fake_command(command, _log_path, _environment, **_kwargs):
                if str(command[2]).endswith("prepare_smartatpg_training.py"):
                    preparation = Path(command[4])
                    preparation.mkdir(parents=True, exist_ok=True)
                    (preparation / "training_manifest.json").write_text(
                        json.dumps({
                            "backtrack_limit": 200, "normal_rounds": 2,
                            "faults_per_update": 8, "k_epochs": 1,
                            "train_circuits": [{"name": "t"}],
                            "validation_circuits": [{"name": "v"}],
                        }),
                        encoding="utf-8",
                    )
                elif str(command[2]).endswith("train_smartatpg.py"):
                    model_dir = Path(command[4])
                    model_dir.mkdir(parents=True, exist_ok=True)
                    for name in ("model_best.txt", "model_latest.txt"):
                        (model_dir / name).write_text("model", encoding="utf-8")
                return 0

            fake_torch = SimpleNamespace(
                __version__="test",
                cuda=SimpleNamespace(
                    device_count=lambda: 1, is_available=lambda: True
                ),
            )
            with (
                patch("run_smartatpg_training_linux.sys.platform", "linux"),
                patch("run_smartatpg_training_linux.torch", fake_torch),
                patch("run_smartatpg_training_linux._check_cpp_extension"),
                patch(
                    "run_smartatpg_training_linux._tee_command",
                    side_effect=fake_command,
                ) as tee,
            ):
                run_training_main([
                    "--output-dir", str(output),
                    "--continue-from", str(source),
                ])
            train = next(
                call.args[0] for call in tee.call_args_list
                if str(call.args[0][2]).endswith("train_smartatpg.py")
            )
            self.assertNotIn("--resume", train)
            self.assertEqual(
                Path(train[train.index("--continue-from") + 1]), source.resolve()
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
                benchmark[benchmark.index("--backtrack-limit") + 1], "200"
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
        with self.assertRaisesRegex(ValueError, "requires backtrack limit 200"):
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


class DualTrainingLauncherTests(unittest.TestCase):
    @staticmethod
    def _fake_torch(device_count=2):
        return SimpleNamespace(
            __version__="test",
            cuda=SimpleNamespace(
                device_count=lambda: device_count, is_available=lambda: True
            ),
        )

    def test_dual_launcher_prepares_once_trains_both_and_then_compares(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training"
            commands = []

            def fake_command(command, _log_path, _environment, **_kwargs):
                commands.append(command)
                if str(command[2]).endswith("prepare_smartatpg_training.py"):
                    preparation = Path(command[4])
                    preparation.mkdir(parents=True, exist_ok=True)
                    (preparation / "training_manifest.json").write_text(
                        json.dumps({
                            "backtrack_limit": 200, "normal_rounds": 2,
                            "faults_per_update": 8, "k_epochs": 1,
                            "train_circuits": [{"name": "t"}],
                            "validation_circuits": [{"name": "v"}],
                        }),
                        encoding="utf-8",
                    )
                if str(command[2]).endswith("compare_smartatpg_validation.py"):
                    for run_name in ("smartatpg_gat_gru", "smartatpg_mean"):
                        self.assertTrue((output / run_name / "model_best.txt").is_file())
                return 0

            def fake_parallel(jobs):
                for job in jobs:
                    directory = Path(job["output_prefix"])
                    directory.mkdir(parents=True, exist_ok=True)
                    for name in (
                        "model_best.txt", "model_latest.txt",
                        "validation_identity.json", "validation_metrics.json",
                    ):
                        (directory / name).write_text("{}", encoding="utf-8")
                return {job["name"]: 1.0 for job in jobs}

            with (
                patch("run_dual_smartatpg_training_linux.sys.platform", "linux"),
                patch("run_dual_smartatpg_training_linux.torch", self._fake_torch(4)),
                patch("run_dual_smartatpg_training_linux._check_cpp_extension"),
                patch(
                    "run_dual_smartatpg_training_linux._tee_command",
                    side_effect=fake_command,
                ),
                patch(
                    "run_dual_smartatpg_training_linux._run_parallel_training",
                    side_effect=fake_parallel,
                ) as parallel,
            ):
                run_dual_training_main(["--output-dir", str(output)])

            self.assertEqual(sum(
                str(command[2]).endswith("prepare_smartatpg_training.py")
                for command in commands
            ), 1)
            self.assertEqual(sum(
                str(command[2]).endswith("compare_smartatpg_validation.py")
                for command in commands
            ), 1)
            self.assertEqual(parallel.call_count, 1)
            jobs = parallel.call_args.args[0]
            train_commands = [job["command"] for job in jobs]
            gpu_by_encoder = {
                command[command.index("--encoder") + 1]: job["environment"]["CUDA_VISIBLE_DEVICES"]
                for command, job in zip(train_commands, jobs)
            }
            self.assertEqual(gpu_by_encoder, {
                "level_gat_gru": "0", "fanin_mean": "1",
            })
            for command in train_commands:
                self.assertEqual(command[command.index("--rounds") + 1], "2")
                self.assertEqual(command[command.index("--k-epochs") + 1], "1")
                self.assertEqual(
                    Path(command[3]), output / "preparation" / "training_manifest.json"
                )
            flattened = " ".join(" ".join(map(str, command)) for command in commands)
            self.assertNotIn("prepare_smartatpg_benchmark.py", flattened)
            self.assertNotIn("benchmark_smartatpg.py", flattened)
            self.assertNotIn("benchmark_bundle", flattened)
            metadata = json.loads((output / "training_run_metadata.json").read_text("utf-8"))
            self.assertIn("validation_comparison", metadata)
            self.assertNotIn("benchmark_bundle", metadata)

    def test_dual_launcher_rejects_invalid_gpu_and_platform_configurations(self):
        cases = (
            ("linux", 1, [], RuntimeError, "only 1 CUDA device"),
            ("linux", 2, ["--gat-gpu", "0", "--mean-gpu", "0"], ValueError,
             "must be distinct"),
            ("linux", 2, ["--gat-gpu", "-1"], ValueError, "non-negative"),
            ("linux", 2, ["--mean-gpu", "2"], RuntimeError, "Requested GPU 2"),
            ("win32", 2, [], RuntimeError, "intended for Linux"),
        )
        for platform, devices, arguments, error, message in cases:
            with self.subTest(platform=platform, devices=devices, arguments=arguments):
                with (
                    patch("run_dual_smartatpg_training_linux.sys.platform", platform),
                    patch("run_dual_smartatpg_training_linux.torch", self._fake_torch(devices)),
                    patch("run_dual_smartatpg_training_linux._check_cpp_extension"),
                    self.assertRaisesRegex(error, message),
                ):
                    run_dual_training_main(arguments)

    def test_parallel_training_terminates_active_peer_after_failure(self):
        started = threading.Event()
        terminated = threading.Event()

        class Process:
            def __init__(self, active=True):
                self.active = active
                self.terminate_calls = 0

            def poll(self):
                return None if self.active else 0

            def terminate(self):
                self.terminate_calls += 1
                self.active = False
                terminated.set()

        gat = Process(active=False)
        mean = Process()

        def fake_command(command, _log, _environment, on_start=None, **_kwargs):
            process = gat if command[0] == "gat" else mean
            on_start(process)
            if command[0] == "gat":
                started.wait(1)
                return 9
            started.set()
            terminated.wait(2)
            return 0

        jobs = [
            {"name": "gat", "command": ["gat"], "log_path": "gat.log", "environment": {}, "output_prefix": "gat"},
            {"name": "mean", "command": ["mean"], "log_path": "mean.log", "environment": {}, "output_prefix": "mean"},
        ]
        with patch(
            "run_dual_smartatpg_training_linux._tee_command", side_effect=fake_command
        ):
            with self.assertRaisesRegex(SystemExit, "9"):
                _run_parallel_training(jobs)
        self.assertEqual(mean.terminate_calls, 1)

    def test_parallel_training_returns_timings_after_both_jobs_succeed(self):
        class Process:
            def poll(self):
                return 0

        def fake_command(_command, _log, _environment, on_start=None, **_kwargs):
            on_start(Process())
            return 0

        jobs = [
            {"name": "gat", "command": ["gat"], "log_path": "gat.log", "environment": {}, "output_prefix": "gat"},
            {"name": "mean", "command": ["mean"], "log_path": "mean.log", "environment": {}, "output_prefix": "mean"},
        ]
        with patch(
            "run_dual_smartatpg_training_linux._tee_command", side_effect=fake_command
        ):
            timings = _run_parallel_training(jobs)
        self.assertEqual(set(timings), {"gat", "mean"})
        self.assertTrue(all(value >= 0 for value in timings.values()))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from rl_podem.validation_tables import build_three_way_row
from rl_podem.artifact_io import (
    INFERENCE_CHECKPOINT_FORMAT,
    atomic_json,
    atomic_json_lines,
    manifest_hash,
)

try:
    import torch
    from rl_podem import validation
except ModuleNotFoundError:
    torch = None
    validation = None


def _load_train_entry():
    spec = importlib.util.spec_from_file_location(
        "smartatpg_train_entry", SCRIPTS / "train_smartatpg.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmartATPGEntryPointTests(unittest.TestCase):
    def test_shared_artifact_contract_and_atomic_writers(self):
        self.assertEqual(
            INFERENCE_CHECKPOINT_FORMAT, "SMARTATPG_INFERENCE_ROUND_V1",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_bytes(b"manifest\n")
            self.assertEqual(
                manifest_hash(manifest), hashlib.sha256(b"manifest\n").hexdigest(),
            )
            json_path = root / "value.json"
            lines_path = root / "records.jsonl"
            atomic_json(json_path, {"value": 1})
            atomic_json_lines(lines_path, [{"value": 1}, {"value": 2}])
            self.assertTrue(json_path.read_bytes().endswith(b"\n"))
            self.assertEqual(lines_path.read_bytes().count(b"\n"), 2)

    def test_training_and_validation_modules_have_one_way_boundaries(self):
        training_source = (
            PYTHON / "rl_podem" / "training.py"
        ).read_text(encoding="utf-8")
        validation_source = (
            PYTHON / "rl_podem" / "validation.py"
        ).read_text(encoding="utf-8")
        validation_core_source = (
            PYTHON / "rl_podem" / "validation_core.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from .training import", validation_source)
        self.assertNotIn("validation_core", training_source)
        self.assertNotIn("training", validation_core_source)
        self.assertNotIn("import torch", validation_core_source)
        self.assertNotIn("cpp_bridge", validation_core_source)
        for name in (
            "_native_validation_batch", "_load_validation_catalogs",
            "_summarize_validation", "validation_score",
        ):
            self.assertNotIn(f"def {name}(", training_source)

    def test_scripts_directory_has_only_two_experiment_entries(self):
        self.assertEqual(
            {path.name for path in SCRIPTS.glob("*.py")},
            {"train_smartatpg.py", "validate_smartatpg.py"},
        )

    def test_train_rejects_same_gpu_before_running_commands(self):
        entry = _load_train_entry()
        with self.assertRaisesRegex(ValueError, "distinct"):
            entry.main(["--gat-gpu", "0", "--mean-gpu", "0"])

    def test_required_training_artifacts_include_each_round_and_final(self):
        entry = _load_train_entry()
        names = {path.name for path in entry._required_artifacts(Path("model"), 2)}
        self.assertEqual(names, {
            "training_state.pth",
            "inference_round_01.pth", "model_round_01.txt",
            "inference_round_02.pth", "model_round_02.txt",
            "inference_final.pth", "model_final.txt",
        })

    def test_training_main_contains_no_validation_phase(self):
        source = (PYTHON / "rl_podem" / "training.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main(argv=None):"):]
        for forbidden in (
            "VALIDATE_START", "SCOAP_VALIDATE", "validation_state.json",
            "run_native_validation", "model_best.txt",
        ):
            self.assertNotIn(forbidden, main_source)

    def test_native_timers_start_after_setup_and_before_atpg_test(self):
        source = (ROOT / "src" / "python_bindings.cpp").read_text(encoding="utf-8")
        for function in ("run_native_validation", "run_native_scoap_validation"):
            start = source.index(f"py::list {function}")
            end = source.find("\npy::", start + 1)
            body = source[start:] if end < 0 else source[start:end]
            self.assertLess(body.index("atpg.input"), body.index("policy->start_run()"))
            self.assertLess(body.index("atpg.retain_faults"), body.index("policy->start_run()"))
            self.assertLess(body.index("policy->start_run()"), body.index("atpg.test()"))

    def test_native_timer_excludes_previous_fault_reporting_io(self):
        source = (ROOT / "src" / "python_bindings.cpp").read_text(encoding="utf-8")
        start = source.index("void on_episode_complete() override")
        end = source.index("\nprivate:", start)
        body = source[start:end]
        self.assertLess(body.index("write_record(records_.back())"), body.rindex("interval_started_"))
        self.assertLess(body.index("std::fflush(stdout)"), body.rindex("interval_started_"))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FreshValidationTests(unittest.TestCase):
    @staticmethod
    def _record(circuit, fault_id, *, detected, backtracks):
        outcome = 1 if detected else 0
        return {
            "circuit": circuit,
            "fault_id": fault_id,
            "outcome": outcome,
            "detected": int(detected),
            "redundant": int(not detected),
            "aborted": 0,
            "backtracks": backtracks,
            "backtrace_steps": backtracks + 1,
            "return": 100.0 if detected else -100.0,
            "test_vectors": int(detected),
            "atpg_seconds": 0.01,
        }

    def test_checkpoint_rejects_training_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "training_manifest.json"
            manifest.write_text('{"identity": 1}\n', encoding="utf-8")
            checkpoint = root / "inference_round_01.pth"
            torch.save({
                "format": validation.INFERENCE_CHECKPOINT_FORMAT,
                "manifest_hash": "0" * 64,
                "encoder_variant": "level_gat_gru",
                "reward_scheme": "cubic_backtrack_v1",
                "backtrack_limit": 100,
                "round": 1,
                "policy_old": {"weight": torch.tensor([1.0])},
            }, checkpoint)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validation._load_inference_checkpoint(
                    checkpoint, "level_gat_gru", manifest, 1,
                )

    def test_fresh_validation_evaluates_all_rounds_and_scoap_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            dataset = root / "dataset"
            paths = []
            for name in ("b12_C", "b15_C", "b17_C", "b20_C", "b21_C", "b22_C"):
                path = dataset / "validation" / f"{name}.bench"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("INPUT(a)\nOUTPUT(a)\n", encoding="utf-8")
                paths.append(path)
            manifests = {}
            model_dirs = {}
            for name in ("gat", "mean"):
                manifests[name] = root / f"{name}_manifest.json"
                manifests[name].write_text("{}\n", encoding="utf-8")
                model_dirs[name] = root / f"{name}_models"
            (run_dir / "train_summary.json").write_text(json.dumps({
                "format": "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION",
                "dataset_root": str(dataset),
                "seed": 2026,
                "rounds": 2,
                "manifests": {key: str(value) for key, value in manifests.items()},
                "training_dirs": {key: str(value) for key, value in model_dirs.items()},
            }), encoding="utf-8")

            def add_catalog(_manifest, circuits):
                for item in circuits:
                    item["episode_fault_ids"] = [f"{item['name']}:f0"]
                return circuits

            evaluated = []

            def evaluate(name, _encoder, _manifest, _model_dir, circuits,
                         _output_dir, round_number, _seed):
                evaluated.append((name, round_number, id(circuits)))
                detected = round_number == 2 if name == "gat" else True
                backtracks = round_number if name == "gat" else round_number * 10
                records = [self._record(
                    item["name"], item["episode_fault_ids"][0],
                    detected=detected, backtracks=backtracks,
                ) for item in circuits]
                return records, validation._summarize_validation(
                    records, circuits, round_number,
                )

            scoap_calls = []

            def scoap(item, fault_ids, _seed):
                scoap_calls.append(item["name"])
                return [self._record(
                    item["name"], fault_ids[0], detected=True, backtracks=30,
                )]

            with (
                patch.object(validation, "discover_validation_dataset", return_value=tuple(paths)),
                patch.object(validation, "_load_validation_catalogs", side_effect=add_catalog),
                patch.object(validation, "_evaluate_model_round", side_effect=evaluate),
                patch.object(validation, "_evaluate_scoap_batch", side_effect=scoap),
            ):
                result = validation.run_fresh_validation(run_dir)

            self.assertEqual([(name, round_) for name, round_, _ in evaluated], [
                ("gat", 1), ("gat", 2), ("mean", 1), ("mean", 2),
            ])
            self.assertEqual(len({identity for _, _, identity in evaluated}), 1)
            self.assertEqual(scoap_calls, [path.stem for path in paths])
            self.assertEqual(result["model_selection"]["gat"]["best_round"], 2)
            self.assertEqual(result["model_selection"]["mean"]["best_round"], 1)
            self.assertEqual(len(result["rows"]), 7)
            self.assertEqual(result["rows"][-1]["circuit"], "TOTAL")


class ThreeWayTableTests(unittest.TestCase):
    @staticmethod
    def _summary(backtracks, backtrace, seconds, detected, episodes):
        return {
            "backtracks_total": backtracks,
            "backtrace_steps_total": backtrace,
            "atpg_seconds": seconds,
            "detected_faults": detected,
            "episodes": episodes,
            "fault_coverage": detected / episodes,
            "circuits": [],
        }

    def test_total_row_uses_atpg_seconds_and_weighted_coverage(self):
        scoap = self._summary(100, 200, 4.0, 8, 10)
        gat = self._summary(60, 100, 3.0, 9, 10)
        mean = self._summary(80, 160, 5.0, 7, 10)
        row = build_three_way_row("TOTAL", scoap, gat, mean)
        self.assertEqual(row["scoap_runtime_atpg_s"], 4.0)
        self.assertEqual(row["gat_fault_coverage_pct"], 90.0)
        self.assertEqual(row["gat_backtracks_reduction_pct_vs_scoap"], 40.0)
        self.assertAlmostEqual(row["mean_fault_coverage_delta_pp_vs_scoap"], -10.0)

    def test_zero_baseline_reduction_is_explicit(self):
        scoap = self._summary(0, 0, 0.0, 1, 1)
        gat = self._summary(1, 0, 0.0, 1, 1)
        mean = self._summary(0, 0, 0.0, 1, 1)
        row = build_three_way_row("TOTAL", scoap, gat, mean)
        self.assertIsNone(row["gat_backtracks_reduction_pct_vs_scoap"])
        self.assertEqual(row["mean_backtracks_reduction_pct_vs_scoap"], 0.0)


if __name__ == "__main__":
    unittest.main()

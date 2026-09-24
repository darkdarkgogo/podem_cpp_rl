import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from rl_podem.validation_tables import build_three_way_row


def _load_train_entry():
    spec = importlib.util.spec_from_file_location(
        "smartatpg_train_entry", SCRIPTS / "train_smartatpg.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmartATPGEntryPointTests(unittest.TestCase):
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

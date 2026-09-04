import sys
import json
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_smartatpg_training import (
    FAULT_FILTER, MANIFEST_FORMAT, _validate_resume, select_hard_faults, sha256_file,
)
from rl_podem.backends import smartatpg_metadata
from train_smartatpg import _episode_order, _validate_manifest, validation_score


class SmartATPGTrainingTests(unittest.TestCase):
    def test_hard_fault_ranking_uses_all_three_keys(self):
        profiles = [
            {"fault_id": "z", "backtracks": 9, "backtrace_steps": 10, "outcome": 1},
            {"fault_id": "b", "backtracks": 10, "backtrace_steps": 5, "outcome": 1},
            {"fault_id": "a", "backtracks": 10, "backtrace_steps": 5, "outcome": 1},
            {"fault_id": "c", "backtracks": 10, "backtrace_steps": 7, "outcome": 1},
        ]
        selected = select_hard_faults(profiles, 4)
        self.assertEqual([item["fault_id"] for item in selected], ["c", "a", "b", "z"])

    def test_hard_faults_exclude_aborted_and_untestable_results(self):
        profiles = [
            {"fault_id": "aborted", "backtracks": 500, "outcome": 2},
            {"fault_id": "untestable", "backtracks": 499, "outcome": 0},
            {"fault_id": "easy", "backtracks": 1, "outcome": 1},
            {"fault_id": "hard", "backtracks": 400, "outcome": 1},
        ]
        self.assertEqual(
            [row["fault_id"] for row in select_hard_faults(profiles, 2)],
            ["hard", "easy"],
        )
        with self.assertRaisesRegex(RuntimeError, "Only 2 baseline-detected"):
            select_hard_faults(profiles, 3)
        with self.assertRaisesRegex(RuntimeError, "Only 0 baseline-detected"):
            select_hard_faults(profiles[:2], 1)
        with self.assertRaises(ValueError):
            select_hard_faults(profiles, 0)

    def test_detected_manifest_validation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                **smartatpg_metadata(), "format": MANIFEST_FORMAT,
                "fault_filter": FAULT_FILTER, "fault_count_per_circuit": 100,
                "backtrack_limit": 500, "profile_seed": 14, "circuits": [],
            }
            for name in ("c6288", "s38417"):
                profiles = [
                    {"fault_id": f"{name}_{i}", "backtracks": i, "outcome": 1}
                    for i in range(105)
                ] + [{"fault_id": f"{name}_aborted", "backtracks": 500, "outcome": 2}]
                selected = select_hard_faults(profiles, 100)
                item = {
                    "name": name, "training_faults": selected,
                    "training_fault_ids": [row["fault_id"] for row in selected],
                    "artifact_sha256": {},
                }
                keys = ["source_circuit", "circuit", "fault_map", "profile"]
                if name == "s38417":
                    keys.append("scan_circuit")
                for key in keys:
                    path = root / f"{name}_{key}"
                    path.write_text(json.dumps(profiles) if key == "profile" else "fixture", encoding="utf-8")
                    item[key] = str(path)
                    item["artifact_sha256"][key] = sha256_file(path)
                manifest["circuits"].append(item)
            manifest_path = root / "training_manifest.json"

            def check_resume():
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                return _validate_resume(manifest_path, 100, 500, 14)

            self.assertEqual(_validate_manifest(manifest), manifest["circuits"])
            self.assertEqual(check_resume(), manifest)
            manifest["circuits"][0]["training_fault_ids"][0] = "c6288_aborted"
            with self.assertRaisesRegex(ValueError, "baseline detected top 100"):
                _validate_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "baseline detected top 100"):
                check_resume()
            manifest["format"] = "SMARTATPG_PAPER_TRAINING_V1"
            with self.assertRaisesRegex(ValueError, "detected-only"):
                _validate_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                check_resume()

    def test_round_order_is_deterministic_and_contains_200_faults(self):
        circuits = [
            {"name": "c6288", "training_fault_ids": [f"c{i}" for i in range(100)]},
            {"name": "s38417", "training_fault_ids": [f"s{i}" for i in range(100)]},
        ]
        first = _episode_order(circuits, 2026, 3)
        self.assertEqual(first, _episode_order(circuits, 2026, 3))
        self.assertNotEqual(first, _episode_order(circuits, 2026, 4))
        self.assertEqual(len(first), 200)
        self.assertEqual(len(set(first)), 200)

    def test_best_score_prioritizes_detection_then_search_cost(self):
        baseline = {
            "detected_faults": 200,
            "backtracks_total": 100,
            "backtrace_steps_total": 1000,
            "return_total": 50.0,
        }
        fewer_detected = dict(baseline, detected_faults=199, backtracks_total=0)
        fewer_backtracks = dict(baseline, backtracks_total=90)
        self.assertLess(validation_score(baseline, 1), validation_score(fewer_detected, 2))
        self.assertLess(validation_score(fewer_backtracks, 2), validation_score(baseline, 1))


if __name__ == "__main__":
    unittest.main()

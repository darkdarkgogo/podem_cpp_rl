import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_smartatpg_training import select_hard_faults
from train_smartatpg import _episode_order, validation_score


class SmartATPGTrainingTests(unittest.TestCase):
    def test_hard_fault_ranking_uses_all_three_keys(self):
        profiles = [
            {"fault_id": "z", "backtracks": 9, "backtrace_steps": 10, "outcome": 0},
            {"fault_id": "b", "backtracks": 10, "backtrace_steps": 5, "outcome": 1},
            {"fault_id": "a", "backtracks": 10, "backtrace_steps": 5, "outcome": 2},
            {"fault_id": "c", "backtracks": 10, "backtrace_steps": 7, "outcome": 1},
        ]
        selected = select_hard_faults(profiles, 4)
        self.assertEqual([item["fault_id"] for item in selected], ["c", "a", "b", "z"])

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

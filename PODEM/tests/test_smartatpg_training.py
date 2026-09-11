import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_smartatpg_training import (
    FAULTS_PER_CIRCUIT, FAULT_FILTER, HEURISTIC, MANIFEST_FORMAT, SELECTION,
    _validate_resume, prepare, select_hard_faults, sha256_file,
)
from smartatpg_portable import CIRCUITS
from rl_podem.backends import smartatpg_metadata
from train_smartatpg import (
    BEST_CHECKPOINT_FORMAT, REINFORCEMENT_BEST_FORMAT, _episode_order,
    _reinforcement_order, _run_gat_reinforcement,
    _validate_manifest, _validate_reinforcement_evaluation, main as train_main,
    device, unresolved_faults, validation_score,
)


class _FakeAgent:
    def __init__(self):
        self.version = 0
        self.policy_old = {"version": 0}

    def training_state_dict(self):
        return {
            "policy_old": {"version": self.version},
            "policy": {"actor": self.version, "critic": self.version},
            "optimizer": {"step": self.version},
            "rnd": {"predictor": self.version, "target": 0},
            "rnd_optimizer": {"step": self.version},
            "rnd_normalization": {"count": self.version},
            "version": self.version,
        }

    def load_training_state_dict(self, state):
        self.version = int(state["version"])
        self.policy_old = {"version": self.version}


class _FakeTrainer:
    def __init__(self, circuit_name, agent, calls):
        self.circuit_name = circuit_name
        self.agent = agent
        self.calls = calls
        self.episode_metrics = []

    def run(self, _circuit, *, fault_ids, **_kwargs):
        fault_id = fault_ids[0]
        self.calls.append((self.circuit_name, fault_id))
        self.agent.version += 1
        self.agent.policy_old = {"version": self.agent.version}
        self.episode_metrics = [{
            "detected": 0,
            "backtracks": 10,
            "backtrace_steps": 20,
            "combined_reward_sum": -1.0,
        }]


class _FakeWriter:
    def add_scalar(self, *_args, **_kwargs):
        pass

    def flush(self):
        pass


def _reinforcement_circuits():
    return [
        {
            "name": name,
            "circuit": f"{name}.bench",
            "fault_map": f"{name}.faultmap",
            "training_fault_ids": [f"{prefix}{index}" for index in range(50)],
        }
        for name, prefix in (("c6288", "c"), ("s38417", "s"))
    ]


def _reinforcement_evaluation(circuits, unresolved, backtracks):
    unresolved = set(unresolved)
    circuit_records = []
    detected = 0
    for item in circuits:
        episodes = []
        for fault_id in item["training_fault_ids"]:
            is_detected = int((item["name"], fault_id) not in unresolved)
            detected += is_detected
            episodes.append({"fault_id": fault_id, "detected": is_detected})
        circuit_records.append({"circuit": item["name"], "episodes": episodes})
    episode_count = sum(len(item["training_fault_ids"]) for item in circuits)
    return {
        "round": 0,
        "episodes": episode_count,
        "detected_faults": detected,
        "backtracks_total": backtracks,
        "backtrace_steps_total": backtracks * 2,
        "return_total": float(detected),
        "fault_coverage": detected / episode_count,
        "backtracks_mean": backtracks / episode_count,
        "backtrace_steps_mean": backtracks * 2 / episode_count,
        "return_mean": detected / episode_count,
        "circuits": circuit_records,
    }


def _source_config(circuits, reinforcement_rounds):
    return {
        "encoder_variant": "level_gat_gru",
        "device": str(device),
        "heuristic": HEURISTIC,
        "circuit_order": [item["name"] for item in circuits],
        "faults_per_circuit": FAULTS_PER_CIRCUIT,
        "normal_rounds": 8,
        "reinforcement_rounds": reinforcement_rounds,
    }


class SmartATPGTrainingTests(unittest.TestCase):
    def test_training_preparation_defaults_to_2000_backtracks(self):
        self.assertEqual(prepare.__defaults__, (50, 2000, 14, 8, 5, False))
        with self.assertRaisesRegex(ValueError, "exactly 50 faults"):
            prepare("unused", count=49)
        with self.assertRaisesRegex(ValueError, "requires backtrack limit 2000"):
            prepare("unused", backtrack_limit=500)

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
                "fault_filter": FAULT_FILTER,
                "fault_count_per_circuit": FAULTS_PER_CIRCUIT,
                "backtrack_limit": 2000, "profile_seed": 14,
                "heuristic": HEURISTIC,
                "circuit_order": list(CIRCUITS),
                "normal_rounds": 8, "reinforcement_rounds": 5,
                "selection": SELECTION,
                "circuits": [],
            }
            for name in CIRCUITS:
                profiles = [
                    {"fault_id": f"{name}_{i}", "backtracks": i, "outcome": 1}
                    for i in range(55)
                ] + [{"fault_id": f"{name}_aborted", "backtracks": 2000, "outcome": 2}]
                selected = select_hard_faults(profiles, FAULTS_PER_CIRCUIT)
                item = {
                    "name": name, "training_faults": selected,
                    "training_fault_ids": [row["fault_id"] for row in selected],
                    "artifact_sha256": {},
                }
                keys = ["source_circuit", "circuit", "fault_map", "profile"]
                if name.startswith("s"):
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
                return _validate_resume(
                    manifest_path, FAULTS_PER_CIRCUIT, 2000, 14, 8, 5
                )

            self.assertEqual(_validate_manifest(manifest), manifest["circuits"])
            self.assertEqual(check_resume(), manifest)
            manifest["backtrack_limit"] = 500
            with self.assertRaisesRegex(ValueError, "2000-backtrack manifest"):
                _validate_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                check_resume()
            manifest["backtrack_limit"] = 2000
            manifest["circuits"][0]["training_fault_ids"][0] = "c432_aborted"
            with self.assertRaisesRegex(ValueError, "baseline detected top 50"):
                _validate_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "baseline detected top 50"):
                check_resume()
            manifest["format"] = "SMARTATPG_PAPER_TRAINING_V1"
            with self.assertRaisesRegex(ValueError, "detected-only"):
                _validate_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                check_resume()

    def test_manifest_rejects_duplicate_faults(self):
        circuits = [
            {
                "name": name,
                "training_fault_ids": [f"{name}_{i}" for i in range(50)],
            }
            for name in CIRCUITS
        ]
        circuits[0]["training_fault_ids"][-1] = circuits[0]["training_fault_ids"][0]
        with self.assertRaisesRegex(ValueError, "800 unique"):
            from prepare_smartatpg_training import _validate_fault_selection
            _validate_fault_selection(circuits, FAULTS_PER_CIRCUIT)

    def test_round_order_is_deterministic_and_contains_800_faults(self):
        circuits = [
            {
                "name": name,
                "training_fault_ids": [f"{name}_{i}" for i in range(50)],
            }
            for name in CIRCUITS
        ]
        first = _episode_order(circuits, 2026, 3)
        self.assertEqual(first, _episode_order(circuits, 2026, 3))
        self.assertNotEqual(first, _episode_order(circuits, 2026, 4))
        self.assertEqual(len(first), 800)
        self.assertEqual(len(set(first)), 800)

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

    def test_reinforcement_uses_only_current_unresolved_faults(self):
        evaluation = {
            "circuits": [
                {
                    "circuit": "c6288",
                    "episodes": [
                        {"fault_id": "c0", "detected": 1},
                        {"fault_id": "c1", "detected": 0},
                    ],
                },
                {
                    "circuit": "s38417",
                    "episodes": [
                        {"fault_id": "s0", "detected": 0},
                        {"fault_id": "s1", "detected": 1},
                    ],
                },
            ]
        }
        self.assertEqual(
            unresolved_faults(evaluation),
            [("c6288", "c1"), ("s38417", "s0")],
        )

    def test_reinforcement_trains_each_unresolved_fault_once_per_round(self):
        faults = [("c6288", "c1"), ("s38417", "s0"), ("c6288", "c2")]
        first = _reinforcement_order(faults, 2026, 1)
        self.assertEqual(first, _reinforcement_order(faults, 2026, 1))
        self.assertEqual(len(first), len(faults))
        self.assertEqual(set(first), set(faults))

    def test_reinforcement_rejects_incomplete_or_non_training_evaluation(self):
        circuits = _reinforcement_circuits()
        evaluation = _reinforcement_evaluation(circuits, [], 10)
        evaluation["circuits"][0]["episodes"][0]["fault_id"] = "test-only"
        with self.assertRaisesRegex(ValueError, "every training fault exactly once"):
            _validate_reinforcement_evaluation(evaluation, circuits)

    def test_reinforcement_shrinks_failures_selects_best_and_stops_early(self):
        circuits = _reinforcement_circuits()
        initial_failures = {("c6288", "c0"), ("s38417", "s0"), ("s38417", "s1")}
        evaluations = [
            _reinforcement_evaluation(circuits, initial_failures, 300),
            _reinforcement_evaluation(circuits, {("s38417", "s1")}, 250),
            _reinforcement_evaluation(circuits, set(), 200),
        ]
        agent = _FakeAgent()
        calls = []
        trainers = {
            item["name"]: _FakeTrainer(item["name"], agent, calls)
            for item in circuits
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            best_checkpoint = output_dir / "best_training_state.pth"
            torch.save({
                "format": BEST_CHECKPOINT_FORMAT,
                "manifest_hash": "manifest",
                "config": _source_config(circuits, 5),
                "round": 8,
                "agent": agent.training_state_dict(),
            }, best_checkpoint)
            with (
                patch("train_smartatpg.evaluate_round", side_effect=evaluations) as evaluate,
                patch("train_smartatpg.export_actor"),
            ):
                state = _run_gat_reinforcement(
                    agent=agent,
                    circuits=circuits,
                    trainers=trainers,
                    evaluators={},
                    output_dir=output_dir,
                    best_checkpoint_path=best_checkpoint,
                    manifest_digest="manifest",
                    normal_rounds=8,
                    reinforcement_rounds=5,
                    backtrack_limit=2000,
                    seed=2026,
                    writer=_FakeWriter(),
                )

            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(set(calls[:3]), initial_failures)
            self.assertEqual(calls[3:], [("s38417", "s1")])
            self.assertEqual(state["current_round"], 3)
            self.assertEqual(state["current_unresolved"], [])
            self.assertEqual(state["best_round"], 2)
            self.assertEqual(len(state["round_metrics"]), 3)
            best = torch.load(
                output_dir / "best_reinforced_training_state.pth",
                map_location="cpu",
            )
            self.assertEqual(best["format"], REINFORCEMENT_BEST_FORMAT)
            self.assertEqual(best["reinforcement_round"], 2)

    def test_reinforcement_keeps_source_best_when_candidate_is_worse(self):
        circuits = _reinforcement_circuits()
        initial_failures = {("c6288", "c0")}
        evaluations = [
            _reinforcement_evaluation(circuits, initial_failures, 100),
            _reinforcement_evaluation(
                circuits, initial_failures | {("c6288", "c1")}, 50
            ),
        ]
        agent = _FakeAgent()
        agent.version = 7
        agent.policy_old = {"version": 7}
        calls = []
        trainers = {
            item["name"]: _FakeTrainer(item["name"], agent, calls)
            for item in circuits
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            best_checkpoint = output_dir / "best_training_state.pth"
            torch.save({
                "format": BEST_CHECKPOINT_FORMAT,
                "manifest_hash": "manifest",
                "config": _source_config(circuits, 1),
                "round": 8,
                "agent": agent.training_state_dict(),
            }, best_checkpoint)
            with (
                patch("train_smartatpg.evaluate_round", side_effect=evaluations),
                patch("train_smartatpg.export_actor"),
            ):
                state = _run_gat_reinforcement(
                    agent=agent,
                    circuits=circuits,
                    trainers=trainers,
                    evaluators={},
                    output_dir=output_dir,
                    best_checkpoint_path=best_checkpoint,
                    manifest_digest="manifest",
                    normal_rounds=8,
                    reinforcement_rounds=1,
                    backtrack_limit=2000,
                    seed=2026,
                    writer=_FakeWriter(),
                )

            self.assertEqual(state["best_round"], 0)
            self.assertEqual(state["best_agent"]["version"], 7)
            best = torch.load(
                output_dir / "best_reinforced_training_state.pth",
                map_location="cpu",
            )
            self.assertEqual(best["agent"]["version"], 7)

    def test_reinforcement_resumes_at_fault_boundary_and_rebuilds_artifacts(self):
        circuits = _reinforcement_circuits()
        initial_failures = {("c6288", "c0"), ("c6288", "c1")}
        initial_evaluation = _reinforcement_evaluation(
            circuits, initial_failures, 100
        )
        final_evaluation = _reinforcement_evaluation(circuits, set(), 80)

        def fake_export(_policy, path, **_kwargs):
            Path(path).write_text("model", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            best_checkpoint = output_dir / "best_training_state.pth"
            first_agent = _FakeAgent()
            torch.save({
                "format": BEST_CHECKPOINT_FORMAT,
                "manifest_hash": "manifest",
                "config": _source_config(circuits, 1),
                "round": 8,
                "agent": first_agent.training_state_dict(),
            }, best_checkpoint)
            first_calls = []
            first_trainers = {
                item["name"]: _FakeTrainer(item["name"], first_agent, first_calls)
                for item in circuits
            }
            original_run = first_trainers["c6288"].run

            def interrupt_after_first(*args, **kwargs):
                if first_calls:
                    raise RuntimeError("simulated interruption")
                return original_run(*args, **kwargs)

            first_trainers["c6288"].run = interrupt_after_first
            with (
                patch("train_smartatpg.evaluate_round", return_value=initial_evaluation),
                patch("train_smartatpg.export_actor", side_effect=fake_export),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                _run_gat_reinforcement(
                    agent=first_agent,
                    circuits=circuits,
                    trainers=first_trainers,
                    evaluators={},
                    output_dir=output_dir,
                    best_checkpoint_path=best_checkpoint,
                    manifest_digest="manifest",
                    normal_rounds=8,
                    reinforcement_rounds=1,
                    backtrack_limit=2000,
                    seed=2026,
                    writer=_FakeWriter(),
                )

            self.assertEqual(len(first_calls), 1)
            for name in (
                "best_reinforced_training_state.pth",
                "model_best_reinforced.txt",
                "reinforcement_metrics.json",
                "reinforcement_unresolved_faults.json",
            ):
                (output_dir / name).unlink(missing_ok=True)

            resumed_agent = _FakeAgent()
            resumed_calls = []
            resumed_trainers = {
                item["name"]: _FakeTrainer(
                    item["name"], resumed_agent, resumed_calls
                )
                for item in circuits
            }
            with (
                patch("train_smartatpg.evaluate_round", return_value=final_evaluation),
                patch("train_smartatpg.export_actor", side_effect=fake_export),
            ):
                state = _run_gat_reinforcement(
                    agent=resumed_agent,
                    circuits=circuits,
                    trainers=resumed_trainers,
                    evaluators={},
                    output_dir=output_dir,
                    best_checkpoint_path=best_checkpoint,
                    manifest_digest="manifest",
                    normal_rounds=8,
                    reinforcement_rounds=1,
                    backtrack_limit=2000,
                    seed=2026,
                    writer=_FakeWriter(),
                )

            self.assertEqual(len(resumed_calls), 1)
            self.assertEqual(set(first_calls + resumed_calls), initial_failures)
            self.assertEqual(state["completed_episodes"], 2)
            for name in (
                "best_reinforced_training_state.pth",
                "model_best_reinforced.txt",
                "reinforcement_metrics.json",
                "reinforcement_unresolved_faults.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_only_gat_accepts_at_most_five_reinforcement_rounds(self):
        with self.assertRaises(SystemExit):
            train_main([
                "missing.json", "unused", "--encoder", "fanin_mean",
                "--reinforcement-rounds", "1",
            ])
        with self.assertRaisesRegex(ValueError, "cannot exceed 5"):
            train_main([
                "missing.json", "unused", "--encoder", "level_gat_gru",
                "--reinforcement-rounds", "6",
            ])
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            train_main([
                "missing.json", "unused", "--rounds", "7",
                "--encoder", "level_gat_gru", "--reinforcement-rounds", "1",
            ])
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            train_main([
                "missing.json", "unused", "--rounds", "7",
                "--encoder", "level_gat_gru",
            ])


if __name__ == "__main__":
    unittest.main()

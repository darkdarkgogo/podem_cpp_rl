import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_smartatpg_training as preparation
from prepare_smartatpg_training import (
    BACKTRACK_LIMIT,
    FAULT_FILTER,
    MANIFEST_FORMAT,
    NORMAL_TRAINING_ROUNDS,
    TRAIN_FAULTS_PER_CIRCUIT,
    discover_dataset,
    prepare,
    select_training_faults,
    select_validation_faults,
)
if torch is not None:
    from train_smartatpg import (
        BEST_CHECKPOINT_FORMAT,
        _episode_order,
        _evaluate_fault,
        _initial_state,
        _append_json_line,
        _load_continuation,
        _load_validation_state,
        _summarize_validation,
        _training_protocol,
        _validate_resume,
        _validation_order,
        validation_score,
    )


class _FakeEvaluator:
    def run(self, _circuit, *, fault_ids, event_callback, **_kwargs):
        fault_id = fault_ids[0]
        event_callback({"event": "backtrace_step", "decision_sequence": 1})
        event_callback({
            "event": "pi_not_done", "decision_sequence": 1,
            "backtracks": 2, "pi_visits": 1,
        })
        event_callback({
            "event": "episode_end", "fault_id": fault_id, "outcome": 1,
            "backtracks": 2, "backtrace_steps": 3, "pi_visits": 1,
        })
        return {
            "episodes": 1, "detected": 1, "redundant": 0, "aborted": 0,
            "backtracks": 2, "backtrace_steps": 3,
        }


class _FakeAgent:
    def __init__(self):
        self.loaded = None

    def load_training_state_dict(self, state):
        self.loaded = state


class SmartATPGPreparationTests(unittest.TestCase):
    def test_fixed_training_contract(self):
        self.assertEqual(BACKTRACK_LIMIT, 200)
        self.assertEqual(NORMAL_TRAINING_ROUNDS, 5)
        self.assertEqual(TRAIN_FAULTS_PER_CIRCUIT, 30)
        self.assertEqual(
            MANIFEST_FORMAT,
            "SMARTATPG_DATA_SPLIT_MANIFEST_V6_TOP30_11D_CO_NO_BUF",
        )
        self.assertEqual(prepare.__defaults__, (14, False))
        self.assertEqual(
            FAULT_FILTER, "train_top30_hard_detected_validation_full_catalog"
        )

    def test_dataset_discovery_uses_train_and_validation_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "validation").mkdir()
            for name in ("z", "a", "m"):
                (root / "train" / f"{name}.bench").write_text(
                    "INPUT(a)\nOUTPUT(a)\n", encoding="utf-8"
                )
            for name in ("v2", "v1"):
                (root / "validation" / f"{name}.bench").write_text(
                    "INPUT(a)\nOUTPUT(a)\n", encoding="utf-8"
                )
            result = discover_dataset(
                root, expected_train_count=3,
                expected_validation_names=("v1", "v2"),
            )
            self.assertEqual(
                [path.stem for path in result["train"]], ["a", "m", "z"]
            )
            self.assertEqual(
                [path.stem for path in result["validation"]], ["v1", "v2"]
            )

    def test_training_selects_detectable_faults_by_heuristic_difficulty(self):
        profiles = [
            {"fault_id": "d3", "outcome": 1, "backtracks": 1,
             "backtrace_steps": 9},
            {"fault_id": "aborted", "outcome": 2, "backtracks": 200,
             "backtrace_steps": 999},
            {"fault_id": "redundant", "outcome": 0, "backtracks": 3,
             "backtrace_steps": 999},
            {"fault_id": "d2", "outcome": 1, "backtracks": 1,
             "backtrace_steps": 10},
            {"fault_id": "d1", "outcome": 1, "backtracks": 1,
             "backtrace_steps": 10},
            {"fault_id": "hardest", "outcome": 1, "backtracks": 2,
             "backtrace_steps": 1},
        ]
        selected = select_training_faults(profiles)
        self.assertEqual(
            [item["fault_id"] for item in selected],
            ["hardest", "d1", "d2", "d3"],
        )
        self.assertEqual(select_validation_faults(profiles), profiles)
        self.assertIsNot(select_validation_faults(profiles)[0], profiles[0])

    def test_training_limits_each_circuit_to_thirty_faults(self):
        profiles = [
            {
                "fault_id": f"f{index:02d}",
                "outcome": 1,
                "backtracks": index,
                "backtrace_steps": index * 2,
            }
            for index in range(35)
        ]
        selected = select_training_faults(profiles)
        self.assertEqual(len(selected), 30)
        self.assertEqual(
            [item["fault_id"] for item in selected],
            [f"f{index:02d}" for index in range(34, 4, -1)],
        )

    def test_training_rejects_a_circuit_without_detectable_faults(self):
        with self.assertRaisesRegex(RuntimeError, "train circuit empty"):
            select_training_faults([
                {"fault_id": "a", "outcome": 2, "backtracks": 200,
                 "backtrace_steps": 10},
                {"fault_id": "b", "outcome": 0, "backtracks": 0,
                 "backtrace_steps": 1},
            ], circuit_name="empty")

    def test_fresh_preparation_refuses_a_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preparation"
            output.mkdir()
            (output / "foreign.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                prepare(Path(directory) / "data", output)

    def test_preparation_is_relocatable_and_rejects_changed_completed_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"
            output = root / "artifacts" / "preparation"
            (dataset / "train").mkdir(parents=True)
            (dataset / "validation").mkdir()
            for split, name in (("train", "t"), ("validation", "v")):
                (dataset / split / f"{name}.bench").write_text(
                    "INPUT(a)\nOUTPUT(a)\n", encoding="utf-8"
                )

            original_discover = discover_dataset

            def small_discover(path):
                return original_discover(
                    path, expected_train_count=1,
                    expected_validation_names=("v",),
                )

            def fake_profile(source, split, seed, graph_identity):
                profiles = [
                    {"fault_id": f"{source.stem}:sa0", "outcome": 1,
                     "backtracks": 1, "backtrace_steps": 2},
                    {"fault_id": f"{source.stem}:hard", "outcome": 1,
                     "backtracks": 2, "backtrace_steps": 1},
                    {"fault_id": f"{source.stem}:sa1", "outcome": 2,
                     "backtracks": 200, "backtrace_steps": 3},
                ]
                return {
                    "format": preparation.PROFILE_FORMAT,
                    "split": split,
                    "name": source.stem,
                    "source_sha256": preparation.sha256_file(source),
                    "circuit_hash": graph_identity[0],
                    "gate_count": graph_identity[1],
                    "backtrack_limit": 200,
                    "profile_seed": seed,
                    "profiles": profiles,
                }

            with (
                patch.object(preparation, "TRAIN_CIRCUIT_COUNT", 1),
                patch.object(preparation, "VALIDATION_NAMES", ("v",)),
                patch.object(preparation, "discover_dataset", small_discover),
                patch.object(preparation, "_profile_payload", fake_profile),
            ):
                manifest = preparation.prepare(dataset, output)
                self.assertEqual(manifest["train_faults_per_circuit"], 30)
                self.assertFalse(Path(manifest["train_circuits"][0]["circuit"]).is_absolute())
                self.assertEqual(
                    manifest["train_circuits"][0]["episode_fault_ids"],
                    ["t:hard", "t:sa0"],
                )
                self.assertEqual(
                    manifest["validation_circuits"][0]["episode_fault_ids"],
                    ["v:sa0", "v:hard", "v:sa1"],
                )
                self.assertEqual(preparation.prepare(dataset, output, resume=True), manifest)
                manifest_path = output / "training_manifest.json"
                legacy = {
                    **manifest,
                    "format": "SMARTATPG_DATA_SPLIT_TRAINING_V5_11D_CO_NO_BUF",
                }
                self.assertNotEqual(legacy["format"], MANIFEST_FORMAT)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    preparation._validate_manifest(legacy, manifest_path)
                tampered = json.loads(json.dumps(manifest))
                tampered["train_circuits"][0]["episode_faults"].reverse()
                tampered["train_circuits"][0]["episode_fault_ids"].reverse()
                with self.assertRaisesRegex(ValueError, "fault list changed"):
                    preparation._validate_manifest(tampered, manifest_path)
                added = dataset / "train" / "added.bench"
                added.write_text("INPUT(a)\nOUTPUT(a)\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exactly 1"):
                    preparation.prepare(dataset, output, resume=True)
                added.unlink()
                profile = output / "profiles" / "train" / "t.json"
                profile.write_text(profile.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "profile|Profile"):
                    preparation.prepare(dataset, output, resume=True)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SmartATPGTrainingStateTests(unittest.TestCase):

    def test_episode_order_is_deterministic_and_complete(self):
        circuits = [
            {"name": "a", "episode_fault_ids": ["a0", "a1"]},
            {"name": "b", "episode_fault_ids": ["b0"]},
        ]
        first = _episode_order(circuits, 2026, 1)
        self.assertEqual(first, _episode_order(circuits, 2026, 1))
        self.assertCountEqual(first, [("a", "a0"), ("a", "a1"), ("b", "b0")])
        self.assertEqual(
            _validation_order(circuits),
            [("a", "a0"), ("a", "a1"), ("b", "b0")],
        )

    def test_validation_reads_outcome_from_terminal_event(self):
        record = _evaluate_fault(
            _FakeEvaluator(), {"name": "v", "circuit": "v.bench"},
            "v:GO:sa0", 200, 14,
        )
        self.assertEqual(record["outcome"], 1)
        self.assertEqual(record["detected"], 1)
        self.assertEqual(record["backtracks"], 2)
        self.assertEqual(record["backtrace_steps"], 3)

    def test_validation_summary_requires_the_full_catalog(self):
        circuits = [{"name": "v", "episode_fault_ids": ["f0", "f1"]}]
        records = [
            {"circuit": "v", "fault_id": "f0", "outcome": 1,
             "detected": 1, "backtracks": 2, "backtrace_steps": 4,
             "return": 100.0},
            {"circuit": "v", "fault_id": "f1", "outcome": 2,
             "detected": 0, "backtracks": 200, "backtrace_steps": 20,
             "return": -100.0},
        ]
        summary = _summarize_validation(records, circuits, 3)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["detected_faults"], 1)
        self.assertEqual(summary["backtracks_total"], 202)
        with self.assertRaisesRegex(ValueError, "full fault catalog"):
            _summarize_validation(records[:1], circuits, 3)

    def test_validation_jsonl_recovers_an_appended_record_before_state_update(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "validation_state.json"
            records_path = Path(directory) / "validation_records.jsonl"
            state, records = _load_validation_state(
                state_path, records_path, "a" * 64, 1
            )
            self.assertEqual((state["next_index"], records), (0, []))
            _append_json_line(records_path, {"circuit": "v", "fault_id": "f0"})
            state, records = _load_validation_state(
                state_path, records_path, "a" * 64, 1
            )
            self.assertEqual(state["next_index"], 1)
            self.assertEqual(records[0]["fault_id"], "f0")

    def test_best_score_uses_the_approved_lexicographic_order(self):
        base = {
            "detected_faults": 10, "backtracks_total": 20,
            "backtrace_steps_total": 30, "return_total": 40.0,
        }
        self.assertLess(
            validation_score({**base, "detected_faults": 11}, 2),
            validation_score(base, 1),
        )
        self.assertLess(
            validation_score({**base, "backtracks_total": 19}, 2),
            validation_score(base, 1),
        )
        self.assertLess(validation_score(base, 1), validation_score(base, 2))

    def test_same_task_resume_requires_exact_manifest_and_config(self):
        config = {
            "rounds": 5, "manifest_hash": "a" * 64,
            "backtrack_limit": 200, "training_episode_count": 3,
        }
        state = _initial_state("a" * 64, config)
        self.assertIs(_validate_resume(state, "a" * 64, config), state)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            _validate_resume(state, "b" * 64, config)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            _validate_resume(state, "a" * 64, {**config, "rounds": 4})

    def test_cross_manifest_continuation_loads_the_complete_agent_state(self):
        full_agent_state = {
            "policy": {"actor": 1, "critic": 2, "encoder": 3},
            "policy_old": {"actor": 1},
            "optimizer": {"state": 4},
            "rnd": {"predictor": 5, "target": 6},
            "rnd_optimizer": {"state": 7},
            "rnd_error_stats": {"count": 8},
        }
        saved = {
            "format": BEST_CHECKPOINT_FORMAT,
            "manifest_hash": "a" * 64,
            "agent": full_agent_state,
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": None,
        }
        agent = _FakeAgent()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            with patch("train_smartatpg.torch.load", return_value=saved):
                lineage = _load_continuation(checkpoint, agent)
        self.assertIs(agent.loaded, full_agent_state)
        self.assertEqual(lineage["source_manifest_hash"], "a" * 64)
        self.assertEqual(lineage["source_format"], BEST_CHECKPOINT_FORMAT)
        reset = _initial_state("b" * 64, {"rounds": 5}, continuation=lineage)
        self.assertEqual(reset["current_round"], 1)
        self.assertEqual(reset["episode_index"], 0)
        self.assertEqual(reset["validation_metrics"], [])
        self.assertIsNone(reset["best_score"])
        self.assertIsNone(reset["best_agent"])

    def test_export_protocol_contains_only_current_data_split_identity(self):
        config = {
            "manifest_hash": "a" * 64, "backtrack_limit": 200,
            "rounds": 5, "training_circuit_count": 1024,
            "validation_circuit_count": 6,
        }
        self.assertEqual(_training_protocol(config), {
            "manifest_hash": "a" * 64,
            "backtrack_limit": 200,
            "normal_rounds": 5,
            "training_circuit_count": 1024,
            "validation_circuit_count": 6,
        })

if __name__ == "__main__":
    unittest.main()

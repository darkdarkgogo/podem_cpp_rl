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
    FAULTS_PER_UPDATE,
    FAULT_FILTER,
    LEGACY_MANIFEST_FORMAT,
    LAZY_VALIDATION_MANIFEST_FORMAT,
    MANIFEST_FORMAT,
    NORMAL_TRAINING_ROUNDS,
    PPO_EPOCHS_PER_UPDATE,
    TRAIN_FAULTS_PER_CIRCUIT,
    discover_dataset,
    prepare,
    select_training_faults,
    select_validation_faults,
    validation_fault_ids,
)
if torch is not None:
    import train_smartatpg as training
    from train_smartatpg import (
        AGENT_TYPES,
        BEST_CHECKPOINT_FORMAT,
        _episode_order,
        _evaluate_fault,
        _fault_update_boundary,
        _initial_state,
        _append_json_line,
        _catalog_fault_ids,
        _load_continuation,
        _load_validation_catalogs,
        _load_validation_state,
        _record_validation_metric,
        _summarize_fault_records,
        _summarize_validation,
        _training_protocol,
        _validate_resume,
        _validation_catalog_hash,
        _validation_identity,
        _validation_order,
        validation_score,
    )
    from rl_podem.gat_gru import GATGRUSmartATPGPPOAgent
    from rl_podem.smartatpg_rewards import MEAN_REWARD_SCHEME
    from rl_podem.smartatpg import SmartATPGPPOAgent


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
            "atpg_seconds": 0.125,
        }


class _FakeAgent:
    def __init__(self):
        self.loaded = None

    def load_training_state_dict(self, state):
        self.loaded = state


class SmartATPGPreparationTests(unittest.TestCase):
    def test_fixed_training_contract(self):
        self.assertEqual(BACKTRACK_LIMIT, 100)
        self.assertEqual(NORMAL_TRAINING_ROUNDS, 2)
        self.assertEqual(FAULTS_PER_UPDATE, 8)
        self.assertEqual(PPO_EPOCHS_PER_UPDATE, 1)
        self.assertEqual(TRAIN_FAULTS_PER_CIRCUIT, 30)
        self.assertEqual(
            MANIFEST_FORMAT,
            "SMARTATPG_DATA_SPLIT_MANIFEST_V8_TWO_ROUND_BATCH8_EPOCH1_11D_CO_NO_BUF",
        )
        self.assertEqual(
            LEGACY_MANIFEST_FORMAT,
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
            {"fault_id": "aborted", "outcome": 2, "backtracks": 100,
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
                {"fault_id": "a", "outcome": 2, "backtracks": 100,
                 "backtrace_steps": 10},
                {"fault_id": "b", "outcome": 0, "backtracks": 0,
                 "backtrace_steps": 1},
            ], circuit_name="empty")

    def test_validation_catalog_rejects_empty_invalid_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validation_fault_ids({"faults": []}, "v.bench")
        with self.assertRaisesRegex(ValueError, "invalid"):
            validation_fault_ids({"faults": [{}]}, "v.bench")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validation_fault_ids(
                {"faults": [{"fault_id": "f0"}, {"fault_id": "f0"}]},
                "v.bench",
            )

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

            profiled_splits = []

            def fake_profile(source, split, seed, graph_identity):
                profiled_splits.append(split)
                profiles = [
                    {"fault_id": f"{source.stem}:sa0", "outcome": 1,
                     "backtracks": 1, "backtrace_steps": 2},
                    {"fault_id": f"{source.stem}:hard", "outcome": 1,
                     "backtracks": 2, "backtrace_steps": 1},
                    {"fault_id": f"{source.stem}:sa1", "outcome": 2,
                     "backtracks": 100, "backtrace_steps": 3},
                ]
                return {
                    "format": preparation.PROFILE_FORMAT,
                    "split": split,
                    "name": source.stem,
                    "source_sha256": preparation.sha256_file(source),
                    "circuit_hash": graph_identity[0],
                    "gate_count": graph_identity[1],
                    "backtrack_limit": 100,
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
                self.assertEqual(profiled_splits, ["train"])
                validation = manifest["validation_circuits"][0]
                self.assertEqual(manifest["normal_rounds"], 2)
                self.assertEqual(manifest["faults_per_update"], 8)
                self.assertEqual(manifest["k_epochs"], 1)
                self.assertNotIn("profile", validation)
                self.assertNotIn("episode_fault_ids", validation)
                self.assertNotIn("validation_episode_count", manifest)
                self.assertFalse((output / "profiles" / "validation").exists())
                self.assertEqual(preparation.prepare(dataset, output, resume=True), manifest)
                manifest_path = output / "training_manifest.json"
                unsupported = {
                    **manifest,
                    "format": "SMARTATPG_DATA_SPLIT_TRAINING_V5_11D_CO_NO_BUF",
                }
                self.assertNotEqual(unsupported["format"], MANIFEST_FORMAT)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    preparation._validate_manifest(unsupported, manifest_path)
                tampered = json.loads(json.dumps(manifest))
                tampered["train_circuits"][0]["episode_faults"].reverse()
                tampered["train_circuits"][0]["episode_fault_ids"].reverse()
                with self.assertRaisesRegex(ValueError, "fault list changed"):
                    preparation._validate_manifest(tampered, manifest_path)
                tampered_validation = json.loads(json.dumps(manifest))
                tampered_validation["validation_circuits"][0][
                    "circuit_hash"
                ] = "changed"
                with self.assertRaisesRegex(ValueError, "graph identity changed"):
                    preparation._validate_manifest(
                        tampered_validation, manifest_path
                    )

                validation_source = dataset / "validation" / "v.bench"
                validation_text = validation_source.read_text(encoding="utf-8")
                validation_source.write_text(
                    validation_text + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    preparation._validate_manifest(manifest, manifest_path)
                validation_source.write_text(validation_text, encoding="utf-8")
                validation_identity = (
                    validation["circuit_hash"], validation["gate_count"]
                )
                validation_payload = fake_profile(
                    validation_source, "validation", 14, validation_identity
                )
                validation_profile = output / "profiles" / "validation" / "v.json"
                preparation._atomic_json(validation_profile, validation_payload)
                legacy = json.loads(json.dumps(manifest))
                legacy["format"] = LEGACY_MANIFEST_FORMAT
                legacy["normal_rounds"] = 5
                legacy.pop("faults_per_update")
                legacy.pop("k_epochs")
                legacy["validation_circuits"] = [preparation._record(
                    manifest_path, validation_profile, validation_source,
                    "validation", validation_payload,
                )]
                legacy["validation_episode_count"] = 3
                self.assertIs(
                    preparation._validate_manifest(legacy, manifest_path), legacy
                )
                legacy_lazy = json.loads(json.dumps(manifest))
                legacy_lazy["format"] = LAZY_VALIDATION_MANIFEST_FORMAT
                legacy_lazy["normal_rounds"] = 5
                legacy_lazy.pop("faults_per_update")
                legacy_lazy.pop("k_epochs")
                self.assertIs(
                    preparation._validate_manifest(legacy_lazy, manifest_path),
                    legacy_lazy,
                )
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

    def test_trainer_exposes_both_encoder_agents(self):
        self.assertEqual(set(AGENT_TYPES), {"fanin_mean", "level_gat_gru"})
        self.assertIs(AGENT_TYPES["fanin_mean"], SmartATPGPPOAgent)
        self.assertIs(AGENT_TYPES["level_gat_gru"], GATGRUSmartATPGPPOAgent)

    def test_fault_update_boundary_batches_eight_and_flushes_remainder(self):
        boundaries = [
            index for index in range(1, 11)
            if _fault_update_boundary(index, 10, 8)
        ]
        self.assertEqual(boundaries, [8, 10])

    def test_new_manifest_loads_validation_fault_catalog_at_runtime(self):
        circuits = [{"name": "v", "circuit": "v.bench"}]
        catalog = {"faults": [{"fault_id": "f0"}, {"fault_id": "f1"}]}
        with patch("train_smartatpg.catalog_cpp_podem", return_value=catalog) as load:
            result = _load_validation_catalogs(
                {"format": MANIFEST_FORMAT}, circuits
            )
        load.assert_called_once_with("v.bench")
        self.assertIs(result, circuits)
        self.assertEqual(circuits[0]["episode_fault_ids"], ["f0", "f1"])
        self.assertEqual(len(_validation_catalog_hash(circuits)), 64)

    def test_legacy_manifest_keeps_stored_validation_fault_order(self):
        circuits = [{
            "name": "v", "circuit": "v.bench",
            "episode_fault_ids": ["f1", "f0"],
        }]
        with patch("train_smartatpg.catalog_cpp_podem") as load:
            result = _load_validation_catalogs(
                {"format": LEGACY_MANIFEST_FORMAT}, circuits
            )
        load.assert_not_called()
        self.assertEqual(result[0]["episode_fault_ids"], ["f1", "f0"])

    def test_runtime_catalog_rejects_duplicate_fault_ids(self):
        with patch("train_smartatpg.catalog_cpp_podem", return_value={
            "faults": [{"fault_id": "f0"}, {"fault_id": "f0"}],
        }):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _catalog_fault_ids("v.bench")

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
            "v:GO:sa0", 100, 14, MEAN_REWARD_SCHEME,
        )
        self.assertEqual(record["outcome"], 1)
        self.assertEqual(record["detected"], 1)
        self.assertEqual(record["redundant"], 0)
        self.assertEqual(record["aborted"], 0)
        self.assertEqual(record["test_vectors"], 1)
        self.assertEqual(record["atpg_seconds"], 0.125)
        self.assertEqual(record["backtracks"], 2)
        self.assertEqual(record["backtrace_steps"], 3)

    def test_validation_rejects_mismatched_reward_protocol(self):
        evaluator = _FakeEvaluator()
        evaluator.agent = type(
            "Agent", (), {"encoder_variant": "level_gat_gru"},
        )()
        item = {"name": "v", "circuit": "v.bench"}
        with self.assertRaisesRegex(ValueError, "evaluator encoder"):
            _evaluate_fault(
                evaluator, item, "f0", 100, 14, MEAN_REWARD_SCHEME,
            )
        with self.assertRaisesRegex(ValueError, "backtrack_limit=100"):
            _evaluate_fault(
                evaluator, item, "f0", 200, 14, "cubic_backtrack_v1",
            )

    def test_validation_summary_rejects_nonfinite_record_values(self):
        record = {
            "circuit": "v", "fault_id": "f0", "outcome": 1,
            "detected": 1, "redundant": 0, "aborted": 0,
            "backtracks": 0, "backtrace_steps": 1, "return": 100.0,
            "test_vectors": 1, "atpg_seconds": 0.1,
        }
        for key in ("return", "atpg_seconds"):
            invalid = {**record, key: float("inf")}
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "Non-finite validation",
            ):
                _summarize_fault_records([invalid])

    def test_validation_summary_requires_the_full_catalog(self):
        circuits = [{"name": "v", "episode_fault_ids": ["f0", "f1"]}]
        records = [
            {"circuit": "v", "fault_id": "f0", "outcome": 1,
             "detected": 1, "redundant": 0, "aborted": 0,
             "backtracks": 2, "backtrace_steps": 4, "return": 100.0,
             "test_vectors": 1, "atpg_seconds": 0.1},
            {"circuit": "v", "fault_id": "f1", "outcome": 2,
             "detected": 0, "redundant": 0, "aborted": 1,
             "backtracks": 100, "backtrace_steps": 20, "return": -100.0,
             "test_vectors": 0, "atpg_seconds": 0.2},
        ]
        summary = _summarize_validation(records, circuits, 3)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["detected_faults"], 1)
        self.assertEqual(summary["backtracks_total"], 102)
        with self.assertRaisesRegex(ValueError, "full fault catalog"):
            _summarize_validation(records[:1], circuits, 3)

    def test_validation_summary_preserves_each_circuit_and_all_outcomes(self):
        circuits = [
            {"name": "a", "episode_fault_ids": ["a0", "a1"]},
            {"name": "b", "episode_fault_ids": ["b0"]},
        ]
        records = [
            {"circuit": "a", "fault_id": "a0", "outcome": 1,
             "detected": 1, "redundant": 0, "aborted": 0,
             "backtracks": 2, "backtrace_steps": 3, "return": 100.0,
             "test_vectors": 1, "atpg_seconds": 0.1},
            {"circuit": "a", "fault_id": "a1", "outcome": 0,
             "detected": 0, "redundant": 1, "aborted": 0,
             "backtracks": 4, "backtrace_steps": 5, "return": -100.0,
             "test_vectors": 0, "atpg_seconds": 0.2},
            {"circuit": "b", "fault_id": "b0", "outcome": 2,
             "detected": 0, "redundant": 0, "aborted": 1,
             "backtracks": 6, "backtrace_steps": 7, "return": -100.0,
             "test_vectors": 0, "atpg_seconds": 0.3},
        ]
        result = _summarize_validation(records, circuits, 1)
        self.assertEqual(result["detected_faults"], 1)
        self.assertEqual(result["redundant_faults"], 1)
        self.assertEqual(result["aborted_faults"], 1)
        self.assertEqual([row["circuit"] for row in result["circuits"]], ["a", "b"])
        self.assertEqual(result["circuits"][0]["test_vectors"], 1)
        self.assertAlmostEqual(result["atpg_seconds"], 0.6)

    def test_validation_identity_declares_protocol_and_catalog(self):
        config = {
            "manifest_hash": "a" * 64,
            "seed": 2026,
            "encoder_variant": "fanin_mean",
            "reward_scheme": MEAN_REWARD_SCHEME,
            "rounds": 2,
            "faults_per_update": 8,
            "k_epochs": 1,
            "backtrack_limit": 100,
            "validation_catalog_hash": "b" * 64,
        }
        identity = _validation_identity(config, [
            {"name": "b12_C"}, {"name": "b15_C"},
        ])
        self.assertEqual(identity, {
            "format": "SMARTATPG_VALIDATION_IDENTITY_V1",
            "manifest_hash": "a" * 64,
            "seed": 2026,
            "encoder_variant": "fanin_mean",
            "reward_scheme": MEAN_REWARD_SCHEME,
            "normal_rounds": 2,
            "faults_per_update": 8,
            "k_epochs": 1,
            "backtrack_limit": 100,
            "validation_catalog_hash": "b" * 64,
            "validation_circuits": ["b12_C", "b15_C"],
        })
        self.assertEqual(identity["encoder_variant"], "fanin_mean")
        self.assertEqual(identity["validation_circuits"], ["b12_C", "b15_C"])
        self.assertEqual(identity["faults_per_update"], 8)

    def test_v8_fresh_resume_creates_identity_for_both_encoders(self):
        train = [{"name": "t", "circuit": "t.bench", "episode_fault_ids": ["t0"]}]
        validation = [{"name": "v", "circuit": "v.bench", "episode_fault_ids": ["v0"]}]
        for encoder in ("level_gat_gru", "fanin_mean"):
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({
                    "format": MANIFEST_FORMAT, "normal_rounds": 2,
                    "backtrack_limit": BACKTRACK_LIMIT,
                }), encoding="utf-8")
                output = root / "training"
                with (
                    patch.object(training, "_resolve_circuit_records", return_value=(train, validation)),
                    patch.object(training, "_load_validation_catalogs", return_value=validation),
                    patch.object(training, "load_circuit_graph", return_value=object()),
                    patch.object(training, "AGENT_TYPES", {encoder: lambda *_a, **_k: object()}),
                    patch.object(training, "CppPodemBacktraceV2Trainer", return_value=object()),
                    patch.object(training, "_initial_state", side_effect=RuntimeError("reached initial state")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "reached initial state"):
                        training.main([str(manifest), str(output), "--encoder", encoder,
                                       "--seed", "77", "--resume"])
                identity = json.loads((output / "validation_identity.json").read_text("utf-8"))
                self.assertEqual(identity["encoder_variant"], encoder)
                self.assertEqual(identity["seed"], 77)

    def test_v8_real_resume_requires_matching_identity(self):
        train = [{"name": "t", "circuit": "t.bench", "episode_fault_ids": ["t0"]}]
        validation = [{"name": "v", "circuit": "v.bench", "episode_fault_ids": ["v0"]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"format": MANIFEST_FORMAT,
                                            "normal_rounds": 2, "backtrack_limit": 100}),
                                encoding="utf-8")
            output = root / "training"
            output.mkdir()
            (output / "training_state.pth").write_bytes(b"checkpoint")
            with (
                patch.object(training, "_resolve_circuit_records", return_value=(train, validation)),
                patch.object(training, "_load_validation_catalogs", return_value=validation),
                patch.object(training, "load_circuit_graph", return_value=object()),
                patch.object(training, "AGENT_TYPES", {"level_gat_gru": lambda *_a, **_k: object()}),
                patch.object(training, "CppPodemBacktraceV2Trainer", return_value=object()),
            ):
                args = [str(manifest), str(output), "--resume"]
                with self.assertRaisesRegex(FileNotFoundError, "validation_identity"):
                    training.main(args)
                identity = _validation_identity({
                    "manifest_hash": training._manifest_hash(manifest), "seed": 2026,
                    "encoder_variant": "level_gat_gru", "rounds": 2,
                    "reward_scheme": "cubic_backtrack_v1",
                    "faults_per_update": 8, "k_epochs": 1, "backtrack_limit": 100,
                    "validation_catalog_hash": training._validation_catalog_hash(validation),
                }, validation)
                (output / "validation_identity.json").write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Validation identity changed"):
                    training.main(args + ["--seed", "2027"])

    def test_legacy_main_config_assembly_skips_v8_validation_identity(self):
        train_circuits = [{
            "name": "t", "circuit": "t.bench", "episode_fault_ids": ["t0"],
        }]
        validation_circuits = [{
            "name": "v", "circuit": "v.bench", "episode_fault_ids": ["v0"],
        }]
        for manifest_format in (
            LEGACY_MANIFEST_FORMAT, LAZY_VALIDATION_MANIFEST_FORMAT,
        ):
            with self.subTest(manifest_format=manifest_format):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest_path = root / "manifest.json"
                    manifest_path.write_text(json.dumps({
                        "format": manifest_format,
                        "normal_rounds": 5,
                        "backtrack_limit": BACKTRACK_LIMIT,
                    }), encoding="utf-8")
                    output_dir = root / "training"
                    with (
                        patch.object(
                            training, "_resolve_circuit_records",
                            return_value=(train_circuits, validation_circuits),
                        ),
                        patch.object(
                            training, "_load_validation_catalogs",
                            return_value=validation_circuits,
                        ),
                        patch.object(training, "load_circuit_graph", return_value=object()),
                        patch.object(training, "AGENT_TYPES", {
                            "level_gat_gru": lambda *_args, **_kwargs: object(),
                        }),
                        patch.object(
                            training, "CppPodemBacktraceV2Trainer",
                            return_value=object(),
                        ),
                        patch.object(
                            training, "_initial_state",
                            side_effect=RuntimeError("reached initial state"),
                        ) as initial_state,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "reached initial state"):
                            training.main([str(manifest_path), str(output_dir)])
                    config = initial_state.call_args.args[1]
                    self.assertNotIn("faults_per_update", config)
                    self.assertFalse((output_dir / "validation_identity.json").exists())

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

    def test_new_best_validation_round_clears_the_previous_best_marker(self):
        state = _initial_state("a" * 64, {"rounds": 2})
        round_one = {
            "round": 1,
            "detected_faults": 10,
            "backtracks_total": 20,
            "backtrace_steps_total": 30,
            "return_total": 40.0,
        }
        round_two = {
            "round": 2,
            "detected_faults": 11,
            "backtracks_total": 20,
            "backtrace_steps_total": 30,
            "return_total": 40.0,
        }

        self.assertTrue(_record_validation_metric(
            state, round_one, validation_score(round_one, 1)
        ))
        self.assertTrue(_record_validation_metric(
            state, round_two, validation_score(round_two, 2)
        ))

        self.assertEqual(state["best_round"], 2)
        self.assertEqual(
            [item["is_best"] for item in state["validation_metrics"]],
            [False, True],
        )

    def test_non_best_validation_round_preserves_the_previous_best_marker(self):
        state = _initial_state("a" * 64, {"rounds": 2})
        round_one = {
            "round": 1,
            "detected_faults": 11,
            "backtracks_total": 20,
            "backtrace_steps_total": 30,
            "return_total": 40.0,
        }
        round_two = {
            "round": 2,
            "detected_faults": 10,
            "backtracks_total": 20,
            "backtrace_steps_total": 30,
            "return_total": 40.0,
        }

        self.assertTrue(_record_validation_metric(
            state, round_one, validation_score(round_one, 1)
        ))
        self.assertFalse(_record_validation_metric(
            state, round_two, validation_score(round_two, 2)
        ))

        self.assertEqual(state["best_round"], 1)
        self.assertEqual(
            [item["is_best"] for item in state["validation_metrics"]],
            [True, False],
        )

    def test_same_task_resume_requires_exact_manifest_and_config(self):
        config = {
            "rounds": 5, "manifest_hash": "a" * 64,
            "backtrack_limit": 100, "training_episode_count": 3,
        }
        state = _initial_state("a" * 64, config)
        self.assertIs(_validate_resume(state, "a" * 64, config), state)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            _validate_resume(state, "b" * 64, config)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            _validate_resume(state, "a" * 64, {**config, "rounds": 4})

    def test_batched_resume_requires_an_update_boundary(self):
        config = {
            "rounds": 2, "manifest_hash": "a" * 64,
            "backtrack_limit": 100, "training_episode_count": 10,
            "faults_per_update": 8,
        }
        state = _initial_state("a" * 64, config)
        state.update(episode_index=8, completed_episodes=8)
        self.assertIs(_validate_resume(state, "a" * 64, config), state)
        state.update(episode_index=7, completed_episodes=7)
        with self.assertRaisesRegex(ValueError, "update boundary"):
            _validate_resume(state, "a" * 64, config)

    def test_cross_manifest_continuation_loads_the_complete_agent_state(self):
        full_agent_state = {
            "policy": {"actor": 1, "critic": 2, "encoder": 3},
            "policy_old": {"actor": 1},
            "optimizer": {"state": 4},
            "rnd": {"predictor": 5, "target": 6},
            "rnd_optimizer": {"state": 7},
            "rnd_error_stats": {"count": 8},
        }
        protocol = {
            "encoder_variant": "fanin_mean",
            "reward_scheme": MEAN_REWARD_SCHEME,
            "backtrack_limit": 100,
        }
        saved = {
            "format": BEST_CHECKPOINT_FORMAT,
            "manifest_hash": "a" * 64,
            "config": protocol,
            "agent": full_agent_state,
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": None,
        }
        agent = _FakeAgent()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            with patch("train_smartatpg.torch.load", return_value=saved):
                lineage = _load_continuation(checkpoint, agent, protocol)
        self.assertIs(agent.loaded, full_agent_state)
        self.assertEqual(lineage["source_manifest_hash"], "a" * 64)
        self.assertEqual(lineage["source_format"], BEST_CHECKPOINT_FORMAT)
        reset = _initial_state("b" * 64, {"rounds": 5}, continuation=lineage)
        self.assertEqual(reset["current_round"], 1)
        self.assertEqual(reset["episode_index"], 0)
        self.assertEqual(reset["validation_metrics"], [])
        self.assertIsNone(reset["best_score"])
        self.assertIsNone(reset["best_agent"])

        invalid_configs = (
            {},
            {**protocol, "backtrack_limit": 200},
            {**protocol, "reward_scheme": "cubic_backtrack_v1"},
            {**protocol, "encoder_variant": "level_gat_gru"},
        )
        for invalid in invalid_configs:
            with self.subTest(config=invalid), patch(
                "train_smartatpg.torch.load",
                return_value={**saved, "config": invalid},
            ), self.assertRaisesRegex(ValueError, "incompatible SmartATPG protocol"):
                _load_continuation(checkpoint, _FakeAgent(), protocol)

    def test_export_protocol_contains_only_current_data_split_identity(self):
        config = {
            "manifest_hash": "a" * 64, "backtrack_limit": 100,
            "reward_scheme": MEAN_REWARD_SCHEME,
            "rounds": 5, "training_circuit_count": 1024,
            "validation_circuit_count": 6,
        }
        self.assertEqual(_training_protocol(config), {
            "manifest_hash": "a" * 64,
            "backtrack_limit": 100,
            "reward_scheme": MEAN_REWARD_SCHEME,
            "normal_rounds": 5,
            "training_circuit_count": 1024,
            "validation_circuit_count": 6,
        })
        batched = {
            **config, "rounds": 2, "faults_per_update": 8, "k_epochs": 1,
        }
        self.assertEqual(_training_protocol(batched), {
            "manifest_hash": "a" * 64,
            "backtrack_limit": 100,
            "reward_scheme": MEAN_REWARD_SCHEME,
            "normal_rounds": 2,
            "training_circuit_count": 1024,
            "validation_circuit_count": 6,
            "faults_per_update": 8,
            "k_epochs": 1,
        })

if __name__ == "__main__":
    unittest.main()

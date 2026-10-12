"""Focused checks for the NN-All PODEM baseline."""

from pathlib import Path
import hashlib
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl_podem.nn_all_features import (
    candidate_vector, load_nn_all_features, write_native_features,
)
from rl_podem.nn_all_training import (
    NnAllClassifier, PathSampleCollector, _circuit_split, _fit,
    export_native_model,
    _reservoir_add,
)
from rl_podem import validation
from rl_podem.validation import _evaluate_nn_all, _load_nn_all_model
from rl_podem.validation_tables import build_four_way_row


BENCH = """INPUT(a)
INPUT(b)
x = AND(a, b)
y = NAND(a, b)
z = OR(x, y)
q = NOR(a, b)
r = NOT(q)
OUTPUT(z)
OUTPUT(r)
"""


class NnAllFeaturesTest(unittest.TestCase):
    def test_feature_order_and_cop(self):
        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            features = load_nn_all_features(bench)
            index = features.indices
            self.assertEqual(features.cop[index["a"]], 0.5)
            self.assertEqual(features.cop[index["x"]], 0.25)
            self.assertEqual(features.cop[index["y"]], 0.75)
            self.assertEqual(features.cop[index["z"]], 0.8125)
            self.assertEqual(features.cop[index["q"]], 0.25)
            self.assertEqual(features.cop[index["r"]], 0.75)
            self.assertEqual(
                candidate_vector(features, "x", "a"),
                (0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            self.assertEqual(len(candidate_vector(features, "z", "x")), 11)
            artifact = Path(temporary) / "features.txt"
            write_native_features(artifact, features)
            self.assertIn("gate x AND 1", artifact.read_text(encoding="utf-8"))

    def test_collector_labels_backtracked_and_detected_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            collector = PathSampleCollector(load_nn_all_features(bench), 1)
            collector.event_callback({"event": "episode_start", "fault_id": "fault"})
            request = {
                "mode": "backtrace", "objective_name": "x",
                "candidate_names": ["a", "b"], "action_mask": [True, True],
                "heuristic_action": 0, "sequence": 1,
            }
            collector.decision_callback(request)
            collector.event_callback({
                "event": "pi_assignment", "pi_name": "a",
                "first_sequence": 1, "last_sequence": 1,
            })
            collector.event_callback({"event": "backtrack", "decision_sequence": 1})
            collector.event_callback({"event": "backtrack_pi", "pi_name": "a"})
            request["sequence"] = 2
            collector.decision_callback(request)
            collector.event_callback({
                "event": "pi_assignment", "pi_name": "b",
                "first_sequence": 2, "last_sequence": 2,
            })
            collector.event_callback({"event": "episode_end", "outcome": 1})
            self.assertEqual([sample[3] for sample in collector.samples], [0, 1])
            self.assertEqual([sample[2] for sample in collector.samples], ["fault", "fault"])

    def test_collector_attributes_whole_pi_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            collector = PathSampleCollector(load_nn_all_features(bench), 1)
            collector.event_callback({"event": "episode_start", "fault_id": "fault"})
            request = {
                "mode": "backtrace", "objective_name": "x",
                "candidate_names": ["a", "b"], "action_mask": [True, True],
                "heuristic_action": 0,
            }
            for sequence in (1, 2):
                collector.decision_callback({**request, "sequence": sequence})
            collector.event_callback({
                "event": "pi_assignment", "pi_name": "a",
                "first_sequence": 1, "last_sequence": 2,
            })
            collector.event_callback({"event": "backtrack", "decision_sequence": 2})
            collector.event_callback({"event": "backtrack_pi", "pi_name": "a"})
            collector.decision_callback({**request, "sequence": 3})
            collector.event_callback({
                "event": "pi_assignment", "pi_name": "b",
                "first_sequence": 3, "last_sequence": 3,
            })
            collector.event_callback({"event": "backtrack_pi", "pi_name": "unused"})
            collector.event_callback({"event": "episode_end", "outcome": 1})
            self.assertEqual([sample[3] for sample in collector.samples], [0, 0, 1])

    def test_collector_skips_forced_backtrace_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            collector = PathSampleCollector(load_nn_all_features(bench), 1)
            collector.event_callback({"event": "episode_start", "fault_id": "fault"})
            action = collector.decision_callback({
                "mode": "backtrace", "objective_name": "x",
                "candidate_names": ["a", "b"], "action_mask": [False, True],
                "heuristic_action": 1, "sequence": 1,
            })
            self.assertEqual(action, 1)
            collector.event_callback({
                "event": "pi_assignment", "pi_name": "b",
                "first_sequence": 1, "last_sequence": 1,
            })
            collector.event_callback({"event": "episode_end", "outcome": 1})
            self.assertEqual(collector.samples, [])

    def test_split_is_by_whole_circuit_and_model_export(self):
        train, holdout = _circuit_split([Path(f"c{index}.bench") for index in range(10)], 2026)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(holdout), 2)
        self.assertFalse(set(train) & set(holdout))
        positive = ((1.0,) * 11, 1, "p", 1)
        negative = ((0.0,) * 11, 1, "n", 0)
        model, epoch, loss = _fit([positive, negative] * 4, [positive, negative], 2026, 2)
        self.assertIsInstance(model, NnAllClassifier)
        self.assertIn(epoch, (1, 2))
        self.assertGreaterEqual(loss, 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "model.txt"
            export_native_model(artifact, model, "a" * 64)
            content = artifact.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("NN_ALL_PODEM_MODEL_V1\n"))
            self.assertIn("w1 660 ", content)

    def test_per_class_reservoir_is_bounded_and_deterministic(self):
        def collect():
            reservoir = []
            generator = random.Random(2026)
            for seen in range(1, 101):
                _reservoir_add(reservoir, seen, seen, 10, generator)
            return reservoir
        self.assertEqual(collect(), collect())
        self.assertEqual(len(collect()), 10)
        self.assertNotEqual(collect(), list(range(1, 11)))

    def test_native_score_and_fault_order(self):
        import cpp_podem

        with tempfile.TemporaryDirectory() as temporary:
            bench = Path(temporary) / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            features = load_nn_all_features(bench)
            table = Path(temporary) / "features.txt"
            write_native_features(table, features)
            model = NnAllClassifier()
            artifact = Path(temporary) / "model.txt"
            export_native_model(artifact, model, "a" * 64)
            vectors = [candidate_vector(features, objective, candidate)
                       for objective, candidate in (("x", "a"), ("z", "x"), ("r", "q"))]
            with torch.no_grad():
                expected_scores = model(torch.tensor(vectors, dtype=torch.float32)).tolist()
            for vector, expected in zip(vectors, expected_scores):
                actual = cpp_podem.score_nn_all_candidate(str(artifact), vector)
                self.assertAlmostEqual(actual, expected, places=5)
            fault_ids = [
                entry["fault_id"]
                for entry in cpp_podem.catalog_stuck_at(str(bench))["faults"][:3]
            ]
            records = cpp_podem.run_native_nn_all_validation(
                str(bench), str(table), str(artifact), 100, 2026,
                fault_ids, "small",
            )
            self.assertEqual([item["fault_id"] for item in records], fault_ids)
            self.assertTrue(all(item["atpg_seconds"] >= 0 for item in records))
            table.write_text(table.read_text(encoding="utf-8").replace(
                "gate x AND", "gate x NOR",
            ), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "feature row hash mismatch"):
                cpp_podem.run_native_nn_all_validation(
                    str(bench), str(table), str(artifact), 100, 2026,
                    fault_ids, "small",
                )
            with self.assertRaisesRegex(Exception, "Cannot open NN-All model"):
                cpp_podem.run_native_nn_all_validation(
                    str(bench), str(table), str(artifact.with_name("missing.txt")),
                    100, 2026, fault_ids, "small",
                )

    def test_validation_summary_and_model_identity(self):
        import cpp_podem

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = root / "small.bench"
            bench.write_text(BENCH, encoding="utf-8")
            manifest = [{"split": "train", "circuit": "small.bench", "fault_ids": []}]
            (root / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            digest = hashlib.sha256(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            export_native_model(root / "model_best.txt", NnAllClassifier(), digest)
            model_hash = hashlib.sha256((root / "model_best.txt").read_bytes()).hexdigest()
            (root / "train_summary.json").write_text(json.dumps({
                "format": "NN_ALL_PODEM_TRAINING_V1",
                "feature_schema": "NN_ALL_PODEM_11D_V1",
                "dataset_root": str(root), "seed": 2026,
                "manifest_sha256": digest, "model_sha256": model_hash,
                "best_epoch": 1,
            }), encoding="utf-8")
            model, _ = _load_nn_all_model(root, root, 2026)
            fault_ids = [
                entry["fault_id"]
                for entry in cpp_podem.catalog_stuck_at(str(bench))["faults"][:3]
            ]
            circuits = [{"name": "small", "circuit": str(bench),
                         "episode_fault_ids": fault_ids}]
            records, summary = _evaluate_nn_all(circuits, model, root / "validation", 2026)
            self.assertEqual([item["fault_id"] for item in records], fault_ids)
            self.assertEqual(summary["episodes"], len(fault_ids))
            self.assertEqual(summary["detected_faults"] + summary["redundant_faults"] +
                             summary["aborted_faults"], len(fault_ids))
            self.assertAlmostEqual(summary["atpg_seconds"],
                                   sum(item["atpg_seconds"] for item in records))
            (root / "model_best.txt").write_text(
                (root / "model_best.txt").read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model SHA-256 mismatch"):
                _load_nn_all_model(root, root, 2026)
            (root / "train_summary.json").write_text(json.dumps({
                "format": "NN_ALL_PODEM_TRAINING_V1",
                "feature_schema": "NN_ALL_PODEM_11D_V1", "dataset_root": str(root),
                "seed": 2026, "manifest_sha256": "0" * 64,
                "model_sha256": model_hash, "best_epoch": 1,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                _load_nn_all_model(root, root, 2026)

    def test_four_way_total_metrics(self):
        def summary(backtracks, steps, seconds, detected):
            return {
                "backtracks_total": backtracks, "backtrace_steps_total": steps,
                "atpg_seconds": seconds, "fault_coverage": detected / 10,
                "detected_faults": detected, "redundant_faults": 10 - detected,
                "aborted_faults": 0, "circuits": [],
            }
        row = build_four_way_row(
            "TOTAL", summary(20, 40, 2, 8), summary(15, 30, 1, 9),
            summary(10, 20, 3, 7), summary(5, 10, 0.5, 10),
        )
        self.assertEqual(row["nn_all_detected_faults"], 10)
        self.assertEqual(row["nn_all_backtracks_reduction_pct_vs_scoap"], 75)
        self.assertEqual(row["nn_all_runtime_atpg_s"], 0.5)
        self.assertAlmostEqual(row["nn_all_fault_coverage_delta_pp_vs_scoap"], 20)

    def test_four_way_validation_is_opt_in_and_keeps_three_way_files(self):
        import cpp_podem

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            bench = dataset / "validation" / "small.bench"
            bench.parent.mkdir(parents=True)
            bench.write_text(BENCH, encoding="utf-8")
            nn_dir = root / "nn"
            nn_dir.mkdir()
            manifest = [{"split": "train", "circuit": "sample.bench", "fault_ids": []}]
            (nn_dir / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            digest = hashlib.sha256(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            export_native_model(nn_dir / "model_best.txt", NnAllClassifier(), digest)
            model_hash = hashlib.sha256((nn_dir / "model_best.txt").read_bytes()).hexdigest()
            (nn_dir / "train_summary.json").write_text(json.dumps({
                "format": "NN_ALL_PODEM_TRAINING_V1", "feature_schema": "NN_ALL_PODEM_11D_V1",
                "dataset_root": str(dataset), "seed": 2026,
                "manifest_sha256": digest, "model_sha256": model_hash,
                "best_epoch": 1,
            }), encoding="utf-8")
            run_dir = root / "smart"
            run_dir.mkdir()
            manifests = {}
            for name in ("gat", "mean"):
                path = root / f"{name}.json"
                path.write_text("{}", encoding="utf-8")
                manifests[name] = str(path)
            (run_dir / "train_summary.json").write_text(json.dumps({
                "format": "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION",
                "dataset_root": str(dataset), "seed": 2026, "rounds": 1,
                "manifests": manifests,
                "training_dirs": {name: str(root / name) for name in manifests},
            }), encoding="utf-8")
            fault_ids = [entry["fault_id"] for entry in
                         cpp_podem.catalog_stuck_at(str(bench))["faults"][:2]]

            def catalogs(_manifest, circuits):
                circuits[0]["episode_fault_ids"] = fault_ids
                return circuits

            def records(item, ids, _seed):
                return [{
                    "circuit": item["name"], "fault_id": fault,
                    "outcome": 1, "detected": 1, "redundant": 0, "aborted": 0,
                    "backtracks": 1, "backtrace_steps": 2,
                    "return": 1.0, "test_vectors": 1, "atpg_seconds": 0.1,
                } for fault in ids]

            def model_round(_name, _encoder, _manifest, _model_dir, circuits,
                            _output_dir, round_number, seed, _lock):
                items = records(circuits[0], circuits[0]["episode_fault_ids"], seed)
                return items, validation._summarize_validation(items, circuits, round_number)

            with (patch.object(validation, "discover_validation_dataset", return_value=(bench,)),
                  patch.object(validation, "_load_validation_catalogs", side_effect=catalogs),
                  patch.object(validation, "_evaluate_model_round", side_effect=model_round),
                  patch.object(validation, "_evaluate_scoap_batch", side_effect=records)):
                three = validation.run_fresh_validation(run_dir)
                self.assertEqual(three["format"], "SMARTATPG_THREE_WAY_VALIDATION_V1")
                self.assertFalse((run_dir / "validation" /
                                  "validation_four_way_comparison.csv").exists())
                four = validation.run_fresh_validation(run_dir, nn_all_run_dir=nn_dir)
            self.assertEqual(four["format"], "SMARTATPG_FOUR_WAY_VALIDATION_V1")
            self.assertEqual(four["nn_all_training"]["manifest_sha256"], digest)
            self.assertEqual(four["rows"][-1]["circuit"], "TOTAL")
            self.assertTrue((run_dir / "validation" /
                             "validation_three_way_comparison.csv").is_file())
            detailed = run_dir / "validation" / "detailed" / "nn_all_podem_records.jsonl"
            self.assertEqual([json.loads(line)["fault_id"] for line in
                              detailed.read_text(encoding="utf-8").splitlines()], fault_ids)

    def test_nn_all_cli_option_is_passed_to_validation(self):
        with patch.object(validation, "run_fresh_validation") as run:
            validation.main(["--nn-all-run-dir", "nn-run"])
        self.assertEqual(run.call_args.kwargs["nn_all_run_dir"], Path("nn-run"))


if __name__ == "__main__":
    unittest.main()

"""Structural and artifact checks for the three trained GAT ablations."""

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

from rl_podem.gat_ablation import GROUPS, comparison_row, run_validation
from rl_podem.gat_gru import (
    GATGRUSmartATPGPolicy, GATGRUSmartATPGPPOAgent, GATNoValuePolicy,
)
from rl_podem.training import ABLATION_AGENTS
from rl_podem.smartatpg_artifacts import (
    encoder_variant, export_actor, export_descriptors, policy_from_state,
)
from rl_podem.smartatpg_features import load_circuit_graph
from rl_podem.smartatpg_portable import load_model
from rl_podem.validation import _load_inference_checkpoint
from rl_podem.artifact_io import INFERENCE_CHECKPOINT_FORMAT, manifest_hash
import train_gat_ablation


BENCH = "INPUT(a)\nINPUT(b)\ny = AND(a, b)\nOUTPUT(y)\n"
PROTOCOL = {
    "manifest_hash": "a" * 64,
    "backtrack_limit": 100,
    "reward_scheme": "cubic_backtrack_depthnorm_v2",
    "normal_rounds": 1,
    "training_circuit_count": 1,
    "validation_circuit_count": 1,
    "faults_per_update": 4,
    "minibatch_size": 512,
    "k_epochs": 4,
}


class GATAblationTests(unittest.TestCase):
    def test_no_cache_keeps_gru_and_python_embedding_cache(self):
        torch.manual_seed(11)
        self.assertIs(ABLATION_AGENTS["no_cache"], GATGRUSmartATPGPPOAgent)
        self.assertIs(GATGRUSmartATPGPPOAgent.policy_class, GATGRUSmartATPGPolicy)
        policy = GATGRUSmartATPGPolicy()
        state = policy.state_dict()
        self.assertTrue(any(".gru." in name for name in state))
        self.assertEqual(encoder_variant(state), "level_gat_gru")
        with tempfile.TemporaryDirectory() as directory:
            circuit = Path(directory) / "tiny.bench"
            circuit.write_text(BENCH, encoding="utf-8")
            graph = load_circuit_graph(circuit)
            with patch.object(policy.graph_encoder, "forward", wraps=policy.graph_encoder.forward) as forward:
                with torch.no_grad():
                    first = policy.graph_embeddings(graph, cached=True)
                    second = policy.graph_embeddings(graph, cached=True)
                self.assertEqual(forward.call_count, 1)
                self.assertTrue(torch.equal(first, second))

    def test_no_object_value_removes_actor_and_critic_signal(self):
        torch.manual_seed(12)
        policy = GATNoValuePolicy()
        state = policy.state_dict()
        self.assertEqual(encoder_variant(state), "level_gat_gru_no_object_value")
        self.assertEqual(tuple(state["backtrace_actor.0.weight"].shape), (32, 22))
        self.assertEqual(tuple(state["critic.0.weight"].shape), (32, 11))
        objective = torch.randn(11)
        candidates = torch.randn(2, 11)
        logits_zero, critic_zero = policy.backtrace_logits(objective, candidates, 0)
        logits_one, critic_one = policy.backtrace_logits(objective, candidates, 1)
        self.assertTrue(torch.equal(logits_zero, logits_one))
        self.assertTrue(torch.equal(critic_zero, critic_one))
        with tempfile.TemporaryDirectory() as directory:
            actor = Path(directory) / "actor.txt"
            export_actor(state, actor, training_protocol=PROTOCOL)
            portable = load_model(actor)
            self.assertEqual(portable.actor_input_dim, 22)

    def test_native_actor_accepts_both_new_variants(self):
        try:
            import cpp_podem
        except ImportError:
            self.skipTest("native cpp_podem extension is not installed")
        with tempfile.TemporaryDirectory() as directory:
            circuit = Path(directory) / "tiny.bench"
            circuit.write_text(BENCH, encoding="utf-8")
            graph = load_circuit_graph(circuit)
            for policy_type in (GATGRUSmartATPGPolicy, GATNoValuePolicy):
                with self.subTest(policy=policy_type.__name__):
                    policy = policy_type()
                    state = policy.state_dict()
                    actor = Path(directory) / f"{policy_type.__name__}.txt"
                    embeddings = Path(directory) / f"{policy_type.__name__}.emb"
                    export_actor(state, actor, training_protocol=PROTOCOL)
                    export_descriptors(state, graph, embeddings)
                    cpp_podem.validate_actor_artifacts(
                        str(embeddings), str(actor), graph.circuit_hash,
                        list(graph.names), "smartatpg",
                    )
                    values = policy.graph_embeddings(graph)
                    objective, candidate = values[-1], values[0]
                    expected = policy.batch_candidate_logits(
                        objective.unsqueeze(0),
                        torch.stack((candidate, values[1])).unsqueeze(0),
                        [1],
                    )[0, 0]
                    actual = cpp_podem.score_actor_candidate_v2(
                        str(actor), objective.tolist(), candidate.tolist(), 1,
                    )
                    self.assertAlmostEqual(float(expected), actual, places=4)

    def test_comparison_contains_only_the_three_ablations(self):
        results = {}
        for index, group in enumerate(GROUPS):
            values = {
                "detected_faults": 10 - index,
                "redundant_faults": 0,
                "aborted_faults": index,
                "backtracks_total": 20 + index,
                "backtrace_steps_total": 30 + index,
                "atpg_seconds": 1.0 + index,
                "fault_coverage": 1.0 - index / 10,
            }
            results[group] = {"circuits": [{"circuit": "tiny", **values}], **values}
        row = comparison_row("tiny", results)
        self.assertEqual(row["no_cache_detected_faults"], 10)
        self.assertEqual(row["no_cache_fault_coverage_pct"], 100)
        self.assertEqual(row["no_lock_aborted_faults"], 1)
        self.assertFalse(any("full" in key for key in row))

    def test_validation_does_not_read_old_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "encoder_variant": "level_gat_gru", "normal_rounds": 1,
            }), encoding="utf-8")
            dataset = root / "dataset" / "validation"
            dataset.mkdir(parents=True)
            (dataset / "tiny.bench").write_text(BENCH, encoding="utf-8")
            groups = {
                group: {
                    "encoder_variant": "level_gat_gru_no_object_value"
                    if group == "no_object_value" else "level_gat_gru",
                    "backtrace_lock": group != "no_lock",
                    "model_dir": str(root / group),
                }
                for group in GROUPS
            }
            (root / "train_summary.json").write_text(json.dumps({
                "format": "SMARTATPG_GAT_ABLATION_TRAINING_V2",
                "groups": groups, "rounds": 1, "backtrack_limit": 100,
                "seed": 2026, "manifest": str(manifest),
                "dataset_root": str(root / "dataset"),
                "full_run_dir": str(root / "missing_full"),
            }), encoding="utf-8")
            circuits = [{"name": "tiny", "circuit": str(dataset / "tiny.bench"),
                         "episode_fault_ids": ["f0"]}]
            calls = []

            def evaluate(group, _variant, _manifest, _model_dir, _circuits,
                         _output_dir, number, _seed, lock, ablation, *, logits_cache):
                calls.append((group, number, lock, ablation, logits_cache))
                record = {
                    "circuit": "tiny", "fault_id": "f0", "outcome": 1,
                    "detected": 1, "redundant": 0, "aborted": 0,
                    "backtracks": 0, "backtrace_steps": 1,
                    "return": 1.0, "test_vectors": 1, "atpg_seconds": 0.1,
                }
                metrics = {
                    "detected_faults": 1, "redundant_faults": 0,
                    "aborted_faults": 0, "backtracks_total": 0,
                    "backtrace_steps_total": 1, "atpg_seconds": 0.1,
                    "fault_coverage": 1.0, "return_total": 1.0,
                }
                return [record], {"round": 1, "circuits": [
                    {"circuit": "tiny", **metrics}], **metrics}

            with patch("rl_podem.gat_ablation._load_validation_catalogs", return_value=circuits), \
                    patch("rl_podem.gat_ablation._evaluate_model_round", side_effect=evaluate), \
                    patch("rl_podem.gat_ablation._validate_summary"):
                report = run_validation(root)
            self.assertEqual([call[0] for call in calls], list(GROUPS))
            self.assertEqual([call[-1] for call in calls], [False, True, True])
            self.assertEqual(set(report["model_selection"]), set(GROUPS))
            self.assertFalse(any("full" in key for key in report["rows"][0]))

    def test_training_resume_uses_old_ablation_summary_without_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            dataset = root / "dataset"
            manifest.write_text(json.dumps({
                "encoder_variant": "level_gat_gru", "normal_rounds": 1,
                "backtrack_limit": 100, "dataset_root": str(dataset),
                "profile_seed": 14,
            }), encoding="utf-8")
            output = root / "ablation"
            output.mkdir()
            (output / "train_summary.json").write_text(json.dumps({
                "format": "SMARTATPG_GAT_ABLATION_TRAINING_V2",
                "manifest": str(manifest), "dataset_root": str(dataset),
                "seed": 2026, "full_run_dir": str(root / "missing_full"),
            }), encoding="utf-8")
            calls = []

            def fake_run(command, _environment, _log_path, _prefix):
                calls.append(command)
                model_dir = Path(command[command.index("rl_podem.training") + 2])
                model_dir.mkdir(parents=True, exist_ok=True)
                for name in ("training_state.pth", "inference_round_01.pth",
                             "model_round_01.txt"):
                    (model_dir / name).touch()
                return 0

            with patch.object(train_gat_ablation, "_run", side_effect=fake_run):
                train_gat_ablation.main(["--output-dir", str(output)])
            self.assertEqual(len(calls), 3)
            summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary["groups"]), set(GROUPS))
            self.assertNotIn("full_run_dir", summary)

    def test_validation_rejects_different_training_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            checkpoint = Path(directory) / "inference.pth"
            payload = {
                "format": INFERENCE_CHECKPOINT_FORMAT,
                "manifest_hash": manifest_hash(manifest),
                "encoder_variant": "level_gat_gru",
                "reward_scheme": "cubic_backtrack_depthnorm_v2",
                "backtrack_limit": 100,
                "round": 1,
                "ablation": "full",
                "backtrace_lock": True,
                "seed": 7,
                "policy_old": {"weight": torch.ones(1)},
            }
            with patch("rl_podem.validation.torch.load", return_value=payload):
                with self.assertRaisesRegex(ValueError, "seed mismatch"):
                    _load_inference_checkpoint(
                        checkpoint, "level_gat_gru", manifest, 1,
                        expected_ablation="full", expected_backtrace_lock=True,
                        expected_seed=2026,
                    )


if __name__ == "__main__":
    unittest.main()

"""Structural and artifact checks for the four GAT ablation groups."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rl_podem.gat_ablation import GROUPS, comparison_row
from rl_podem.gat_gru import GATNoGRUPolicy, GATNoValuePolicy
from rl_podem.smartatpg_artifacts import (
    encoder_variant, export_actor, export_descriptors, policy_from_state,
)
from rl_podem.smartatpg_features import load_circuit_graph
from rl_podem.smartatpg_portable import (
    compute_embeddings, load_graph, load_model,
)
from rl_podem.validation import _load_inference_checkpoint
from rl_podem.artifact_io import INFERENCE_CHECKPOINT_FORMAT, manifest_hash


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
    def test_no_gru_has_no_gru_tensors_and_portable_embeddings_match(self):
        torch.manual_seed(11)
        policy = GATNoGRUPolicy()
        state = policy.state_dict()
        self.assertFalse(any(".gru." in name for name in state))
        self.assertEqual(encoder_variant(state), "level_gat_no_gru")
        with tempfile.TemporaryDirectory() as directory:
            circuit = Path(directory) / "tiny.bench"
            actor = Path(directory) / "actor.txt"
            circuit.write_text(BENCH, encoding="utf-8")
            export_actor(state, actor, training_protocol=PROTOCOL)
            portable = load_model(actor)
            expected = policy_from_state(state).graph_embeddings(load_circuit_graph(circuit))
            actual = compute_embeddings(portable, load_graph(circuit))
            self.assertEqual(portable.encoder_variant, "level_gat_no_gru")
            self.assertTrue(torch.allclose(expected, torch.tensor(actual), atol=1e-5, rtol=1e-5))

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
            for policy_type in (GATNoGRUPolicy, GATNoValuePolicy):
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

    def test_comparison_reports_coverage_before_search_cost(self):
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
        self.assertAlmostEqual(row["no_gru_coverage_delta_pp_vs_full"], -10.0)
        self.assertEqual(row["no_gru_backtracks_delta_vs_full"], 1)
        self.assertEqual(row["no_lock_aborted_faults"], 2)
        self.assertEqual(row["no_lock_aborted_faults_delta_vs_full"], 2)

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

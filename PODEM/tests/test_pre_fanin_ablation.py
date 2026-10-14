"""Focused contract checks for the two-output pre-fanin GAT variants."""

from pathlib import Path
import tempfile
import unittest

import torch

from rl_podem.artifact_io import INFERENCE_CHECKPOINT_FORMAT, manifest_hash
from rl_podem.gat_gru import GATGRUSmartATPGPolicy, GATNoValuePolicy, GATNoValuePPOAgent
from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer, catalog_cpp_podem
from rl_podem.pre_fanin_ablation import _check_group, _validate_record_order, comparison_row
from rl_podem.smartatpg_artifacts import export_actor, export_descriptors, policy_from_state
from rl_podem.smartatpg_features import load_circuit_graph
from rl_podem.smartatpg_portable import load_model
from rl_podem.smartatpg_rewards import (
    GAT_REWARD_SCHEME, smartatpg_backtrace_step_reward, smartatpg_circuit_depth,
)
from rl_podem.validation_core import _native_validation_batch


class PreFaninAblationTests(unittest.TestCase):
    PROTOCOL = {
        "manifest_hash": "0" * 64,
        "backtrack_limit": 100,
        "reward_scheme": "cubic_backtrack_depthnorm_v2",
        "normal_rounds": 2,
        "faults_per_update": 4,
        "minibatch_size": 512,
        "k_epochs": 4,
        "training_circuit_count": 1,
        "validation_circuit_count": 1,
    }

    def test_no_value_actor_and_critic_ignore_objective_value(self):
        torch.manual_seed(11)
        policy = GATNoValuePolicy().eval()
        objective = torch.arange(11, dtype=torch.float32) / 10
        logits_zero, critic_zero = policy.backtrace_logits(objective, 0)
        logits_one, critic_one = policy.backtrace_logits(objective, 1)
        self.assertEqual(policy.backtrace_actor[0].in_features, 11)
        self.assertEqual(policy.backtrace_actor[2].out_features, 2)
        torch.testing.assert_close(logits_zero, logits_one)
        torch.testing.assert_close(critic_zero, critic_one)
        self.assertEqual(GATGRUSmartATPGPolicy().backtrace_actor[0].in_features, 12)

    def test_no_value_actor_exports_and_loads_as_distinct_gat_variant(self):
        policy = GATNoValuePolicy().eval()
        state = policy.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.txt"
            export_actor(state, path, training_protocol=self.PROTOCOL)
            model = load_model(path)
            self.assertEqual(model.encoder_variant, "level_gat_gru_no_object_value")
            self.assertEqual(model.actor_input_dim, 11)
            self.assertEqual(policy_from_state(state).backtrace_actor[0].in_features, 11)
            try:
                import cpp_podem
            except ImportError:
                return
            objective = torch.arange(11, dtype=torch.float32) / 10
            expected = policy.backtrace_logits(objective, 0)[0]
            native_zero = torch.tensor(cpp_podem.score_actor_v2(str(path), objective.tolist(), 0))
            native_one = torch.tensor(cpp_podem.score_actor_v2(str(path), objective.tolist(), 1))
            torch.testing.assert_close(native_zero, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(native_zero, native_one)

    def test_native_cache_switch_preserves_fault_results(self):
        bench = "INPUT(a)\nINPUT(b)\ny = AND(a, b)\nOUTPUT(y)\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            circuit = root / "small.bench"
            circuit.write_text(bench, encoding="utf-8")
            graph = load_circuit_graph(circuit)
            policy = GATGRUSmartATPGPolicy().eval()
            actor = root / "actor.txt"
            embeddings = root / "embeddings.txt"
            export_actor(policy.state_dict(), actor, training_protocol=self.PROTOCOL)
            export_descriptors(policy.state_dict(), graph, embeddings)
            fault_ids = [item["fault_id"] for item in catalog_cpp_podem(circuit)["faults"]][:4]
            item = {"name": "small", "circuit": str(circuit)}
            step_reward = smartatpg_backtrace_step_reward(
                GAT_REWARD_SCHEME, smartatpg_circuit_depth(graph.levels),
            )
            cached = _native_validation_batch(
                item, fault_ids, embeddings, actor, root / "cached.jsonl",
                14, GAT_REWARD_SCHEME, step_reward, True, True,
            )
            uncached = _native_validation_batch(
                item, fault_ids, embeddings, actor, root / "uncached.jsonl",
                14, GAT_REWARD_SCHEME, step_reward, True, False,
            )
            self.assertEqual(
                [{k: v for k, v in record.items() if k != "atpg_seconds"} for record in cached],
                [{k: v for k, v in record.items() if k != "atpg_seconds"} for record in uncached],
            )

            no_value_policy = GATNoValuePolicy().eval()
            no_value_actor = root / "no_value_actor.txt"
            no_value_embeddings = root / "no_value_embeddings.txt"
            export_actor(no_value_policy.state_dict(), no_value_actor, training_protocol=self.PROTOCOL)
            export_descriptors(no_value_policy.state_dict(), graph, no_value_embeddings)
            no_value_records = _native_validation_batch(
                item, fault_ids, no_value_embeddings, no_value_actor,
                root / "no_value.jsonl", 14, GAT_REWARD_SCHEME,
                step_reward, True, True,
            )
            self.assertEqual(len(no_value_records), len(fault_ids))

            agent = GATNoValuePPOAgent({"small": graph}, rnd_beta=0)
            trainer = CppPodemBacktraceV2Trainer(
                graph, agent=agent, auto_update=False,
                reward_scheme=GAT_REWARD_SCHEME, backtrace_lock=True,
            )
            trainer.run(circuit, backtrack_limit=100, seed=14,
                        fault_ids=fault_ids[:1], use_scoap=True)
            self.assertEqual(len(trainer.episode_metrics), 1)

    def test_comparison_uses_full_as_reference(self):
        def sample(detected, time):
            return {
                "detected_faults": detected, "redundant_faults": 1,
                "aborted_faults": 0, "backtracks_total": 2,
                "backtrace_steps_total": 3, "atpg_seconds": time,
                "fault_coverage": detected / 10, "circuits": [],
            }
        results = {
            "full": sample(8, 2), "no_cache": sample(8, 3),
            "no_lock": sample(7, 2), "no_object_value": sample(6, 4),
        }
        row = comparison_row("TOTAL", results)
        self.assertEqual(row["no_cache_detected_faults_delta_vs_full"], 0)
        self.assertEqual(row["no_cache_runtime_delta_pct_vs_full"], 50)
        self.assertAlmostEqual(row["no_object_value_coverage_delta_pp_vs_full"], -20)

    def test_existing_full_checkpoint_needs_no_ablation_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            model_dir = root / "full"
            model_dir.mkdir()
            state = GATGRUSmartATPGPolicy().state_dict()
            for number in (1, 2):
                torch.save({
                    "format": INFERENCE_CHECKPOINT_FORMAT,
                    "manifest_hash": manifest_hash(manifest),
                    "encoder_variant": "level_gat_gru",
                    "reward_scheme": GAT_REWARD_SCHEME,
                    "backtrack_limit": 100,
                    "seed": 2026,
                    "round": number,
                    "policy_old": state,
                }, model_dir / f"inference_round_{number:02d}.pth")
            summary = {
                "rounds": 2,
                "groups": {
                    "full": {
                        "encoder_variant": "level_gat_gru",
                        "backtrace_lock": True,
                        "model_dir": str(model_dir),
                    },
                    "no_cache": {
                        "encoder_variant": "level_gat_gru",
                        "backtrace_lock": True,
                        "model_dir": str(model_dir),
                        "source_group": "full",
                    },
                },
            }
            full_dir, full_snapshots = _check_group(summary, "full", manifest, 2026)
            no_cache_dir, no_cache_snapshots = _check_group(
                summary, "no_cache", manifest, 2026,
            )
            self.assertEqual(full_dir, no_cache_dir)
            self.assertEqual(full_snapshots, no_cache_snapshots)

    def test_existing_full_fault_order_must_match_catalog(self):
        circuits = [{"name": "c1", "episode_fault_ids": ["f1", "f2"]}]
        records = [
            {"circuit": "c1", "fault_id": "f1"},
            {"circuit": "c1", "fault_id": "f2"},
        ]
        _validate_record_order(records, circuits)
        with self.assertRaisesRegex(ValueError, "fault catalog"):
            _validate_record_order(list(reversed(records)), circuits)


if __name__ == "__main__":
    unittest.main()

"""Focused contract checks for the two-output pre-fanin GAT variants."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl_podem.artifact_io import INFERENCE_CHECKPOINT_FORMAT, manifest_hash
from rl_podem.gat_gru import GATGRUSmartATPGPolicy, GATNoValuePolicy, GATNoValuePPOAgent
from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer, catalog_cpp_podem
from rl_podem.pre_fanin_ablation import (
    _check_group, comparison_row, run_validation,
)
from rl_podem.smartatpg_artifacts import export_actor, export_descriptors, policy_from_state
from rl_podem.smartatpg_features import load_circuit_graph
from rl_podem.smartatpg_portable import load_model
from rl_podem.smartatpg_rewards import (
    GAT_REWARD_SCHEME, smartatpg_backtrace_step_reward, smartatpg_circuit_depth,
)
from rl_podem.validation_core import _native_validation_batch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import train_pre_fanin_ablation


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

    def test_comparison_contains_only_three_ablation_groups(self):
        def sample(detected, time):
            return {
                "detected_faults": detected, "redundant_faults": 1,
                "aborted_faults": 0, "backtracks_total": 2,
                "backtrace_steps_total": 3, "atpg_seconds": time,
                "fault_coverage": detected / 10, "circuits": [],
            }
        results = {
            "no_cache": sample(8, 3),
            "no_lock": sample(7, 2), "no_object_value": sample(6, 4),
        }
        row = comparison_row("TOTAL", results)
        self.assertEqual(row["no_cache_detected_faults"], 8)
        self.assertEqual(row["no_lock_runtime_atpg_s"], 2)
        self.assertEqual(row["no_object_value_fault_coverage_pct"], 60)
        self.assertFalse(any("full" in key for key in row))

    def test_no_cache_requires_its_own_ablation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            state = GATGRUSmartATPGPolicy().state_dict()
            groups = {}
            for group, lock in (("no_cache", True), ("no_lock", False)):
                model_dir = root / group
                model_dir.mkdir()
                for number in (1, 2):
                    torch.save({
                        "format": INFERENCE_CHECKPOINT_FORMAT,
                        "manifest_hash": manifest_hash(manifest),
                        "encoder_variant": "level_gat_gru",
                        "reward_scheme": GAT_REWARD_SCHEME,
                        "backtrack_limit": 100, "backtrace_lock": lock,
                        "ablation": group, "seed": 2026, "round": number,
                        "policy_old": state,
                    }, model_dir / f"inference_round_{number:02d}.pth")
                groups[group] = {
                    "encoder_variant": "level_gat_gru",
                    "backtrace_lock": lock,
                    "model_dir": str(model_dir),
                }
            summary = {"rounds": 2, "groups": groups}
            no_cache_dir, _ = _check_group(summary, "no_cache", manifest, 2026)
            self.assertEqual(no_cache_dir, root / "no_cache")
            summary["groups"]["no_lock"]["model_dir"] = str(no_cache_dir)
            with self.assertRaisesRegex(ValueError, "own trained model"):
                _check_group(summary, "no_cache", manifest, 2026)

    def test_three_training_jobs_use_distinct_gpus_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ablation"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "encoder_variant": "level_gat_gru", "normal_rounds": 2,
                "backtrack_limit": 100, "dataset_root": str(root / "data"),
                "profile_seed": 14,
            }), encoding="utf-8")
            captured = []

            def fake_parallel(jobs):
                captured.extend(jobs)
                for group, _, _, _ in jobs:
                    model_dir = output / "training" / group
                    model_dir.mkdir(parents=True)
                    for filename in (
                        "training_state.pth", "inference_round_02.pth",
                        "model_round_02.txt",
                    ):
                        (model_dir / filename).touch()

            with patch.object(train_pre_fanin_ablation, "_run"), \
                    patch.object(train_pre_fanin_ablation, "_run_parallel",
                                 side_effect=fake_parallel):
                train_pre_fanin_ablation.main([
                    "--manifest", str(manifest), "--output-dir", str(output),
                    "--no-cache-gpu", "2", "--no-lock-gpu", "0",
                    "--no-object-value-gpu", "1",
                ])
            self.assertEqual([job[0] for job in captured], [
                "no_cache", "no_lock", "no_object_value",
            ])
            self.assertEqual([
                job[2]["CUDA_VISIBLE_DEVICES"] for job in captured
            ], ["2", "0", "1"])
            self.assertEqual([
                job[1][job[1].index("--ablation") + 1] for job in captured
            ], ["no_cache", "no_lock", "no_object_value"])
            summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["format"], "SMARTATPG_PRE_FANIN_ABLATION_TRAINING_V3_THREE_GROUPS")
            self.assertEqual(set(summary["groups"]), {
                "no_cache", "no_lock", "no_object_value",
            })

    def test_validation_compares_only_independent_ablations(self):
        try:
            import cpp_podem
        except ImportError:
            self.skipTest("Native extension is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "data" / "validation"
            validation.mkdir(parents=True)
            (validation / "small.bench").write_text(
                "INPUT(a)\nINPUT(b)\ny = AND(a, b)\nOUTPUT(y)\n", encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "encoder_variant": "level_gat_gru", "normal_rounds": 2,
            }), encoding="utf-8")
            groups = {}
            for group in ("no_cache", "no_lock", "no_object_value"):
                model_dir = root / "training" / group
                model_dir.mkdir(parents=True)
                variant = (
                    "level_gat_gru_no_object_value" if group == "no_object_value"
                    else "level_gat_gru"
                )
                state = (
                    GATNoValuePolicy() if group == "no_object_value"
                    else GATGRUSmartATPGPolicy()
                ).state_dict()
                for number in (1, 2):
                    torch.save({
                        "format": INFERENCE_CHECKPOINT_FORMAT,
                        "manifest_hash": manifest_hash(manifest),
                        "encoder_variant": variant,
                        "reward_scheme": GAT_REWARD_SCHEME,
                        "backtrack_limit": 100,
                        "backtrace_lock": group != "no_lock",
                        "ablation": group, "seed": 2026,
                        "round": number, "policy_old": state,
                    }, model_dir / f"inference_round_{number:02d}.pth")
                    export_actor(
                        state, model_dir / f"model_round_{number:02d}.txt",
                        training_protocol=self.PROTOCOL,
                    )
                groups[group] = {
                    "encoder_variant": variant,
                    "backtrace_lock": group != "no_lock",
                    "model_dir": str(model_dir),
                }
            (root / "train_summary.json").write_text(json.dumps({
                "format": "SMARTATPG_PRE_FANIN_ABLATION_TRAINING_V3_THREE_GROUPS",
                "rounds": 2, "backtrack_limit": 100, "seed": 2026,
                "manifest": str(manifest), "dataset_root": str(root / "data"),
                "groups": groups,
            }), encoding="utf-8")
            report = run_validation(root)
            self.assertEqual(set(report["model_selection"]), set(groups))
            self.assertEqual(set(report["logits_cache"]), set(groups))
            self.assertFalse(report["logits_cache"]["no_cache"])
            self.assertTrue(report["logits_cache"]["no_lock"])
            self.assertFalse(any("full" in key for key in report["rows"][-1]))
            self.assertTrue((root / "validation" / "validation_pre_fanin_ablation.csv").is_file())


if __name__ == "__main__":
    unittest.main()

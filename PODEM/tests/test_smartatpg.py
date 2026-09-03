import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rl_podem.smartatpg_features import (
    FEATURE_DIM, GATE_TYPES, GRAPH_CONFIG, load_circuit_graph,
)
from rl_podem.smartatpg import (
    GATE_EMBEDDING_DIM, POLICY_STATE_DIM, GraphGate, SmartATPGPPOAgent,
    SmartATPGPolicy,
)
from rl_podem.curriculum import CppPodemCurriculumEvaluator
from rl_podem.cpp_bridge import _load_cpp_embedding_artifact, export_actor_v2_state_dict
from rl_podem.smartatpg_artifacts import export_actor, export_descriptors, snapshot_id, policy_from_state
from rl_podem.artifact_paths import training_output_paths
from smartatpg_portable import (
    compute_embeddings as compute_portable_embeddings,
    load_graph as load_portable_graph,
    load_model as load_portable_model,
)


BENCH = """INPUT(a)
INPUT(b)
n = NOT(a)
y = AND(n, b)
q = NOR(n, b)
unused = BUF(a)
OUTPUT(y)
OUTPUT(q)
"""


class SmartATPGTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.bench"
        self.path.write_text(BENCH, encoding="utf-8")
        self.graph = load_circuit_graph(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def agent(self, method="gae"):
        return SmartATPGPPOAgent({"test": self.graph}, advantage_method=method,
                                normalize_returns=False, normalize_advantages=True,
                                return_scale=100, rnd_beta=0, k_epochs=2)

    def gates(self):
        return {name: GraphGate(name, self.graph.circuit_hash, index)
                for index, name in enumerate(self.graph.names)}

    def test_features_and_controllability(self):
        g = self.graph
        i = g.name_to_index
        self.assertEqual(GATE_TYPES, ("PI", "AND", "NAND", "OR", "NOR", "NOT", "BUF"))
        self.assertEqual(FEATURE_DIM, 11)
        self.assertEqual(g.x.shape, (6, FEATURE_DIM))
        self.assertEqual((g.cc0[i["y"]], g.cc1[i["y"]]), (2, 4))
        self.assertEqual((g.cc0[i["q"]], g.cc1[i["q"]]), (2, 4))
        self.assertFalse(hasattr(g, "co"))
        self.assertTrue(torch.isfinite(g.x).all())
        self.assertEqual(g.fanouts[i["a"]], 2)
        self.assertEqual(GRAPH_CONFIG["layers"], 1)

    def test_invalid_inputs(self):
        for text in ("INPUT(a)\nOUTPUT(missing)",
                     "a=NOT(b)\nb=NOT(a)\nOUTPUT(a)",
                     "INPUT(a)\ny=AND(a,a,a)\nOUTPUT(y)",
                     "INPUT(a)\na=NOT(a)\nOUTPUT(a)",
                     "INPUT(a)\nINPUT(b)\ny=XOR(a,b)\nOUTPUT(y)",
                     "INPUT(a)\nINPUT(b)\ny=XNOR(a,b)\nOUTPUT(y)"):
            self.path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_circuit_graph(self.path)

    def test_graph_permutation_and_gradients(self):
        policy = SmartATPGPolicy()
        g = self.graph
        permutation = torch.randperm(len(g.names))
        inverse = permutation.argsort()
        permuted = dataclasses.replace(g, x=g.x[permutation], edge_index=inverse[g.edge_index])
        a = policy.graph_encoder(g)
        b = policy.graph_encoder(permuted)
        torch.testing.assert_close(a[permutation], b)
        a.sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.graph_encoder.parameters()), 0)

    def test_directed_mean_and_empty_neighbors(self):
        policy = SmartATPGPolicy()
        with torch.no_grad():
            for parameter in policy.graph_encoder.parameters():
                parameter.zero_()
            layer = policy.graph_encoder.layer
            layer.weight[0, 7] = 1
            layer.weight[1, FEATURE_DIM + 7] = 1
            hidden = policy.graph_encoder(self.graph)
        i = self.graph.name_to_index
        torch.testing.assert_close(hidden[i["y"], :2], torch.tensor([1.0, 0.25]))
        torch.testing.assert_close(hidden[i["a"], :2], torch.zeros(2))

    def test_inverted_gates_repeated_pins_and_cost_cap(self):
        self.path.write_text("INPUT(a)\nINPUT(b)\nn=NOT(a)\nx=OR(n,b)\ny=NOR(n,b)\nz=NAND(n,b)\nt=AND(a,a)\nOUTPUT(x)\nOUTPUT(y)\nOUTPUT(z)\nOUTPUT(t)\nOUTPUT(a)", encoding="utf-8")
        g = load_circuit_graph(self.path)
        i = g.name_to_index
        self.assertEqual((g.cc0[i["x"]], g.cc1[i["x"]]), (4, 2))
        self.assertEqual((g.cc0[i["y"]], g.cc1[i["y"]]), (2, 4))
        self.assertEqual((g.cc0[i["z"]], g.cc1[i["z"]]), (4, 2))
        self.assertEqual(g.fanouts[i["a"]], 3)
        rows = ["INPUT(g0)"] + [f"g{i}=AND(g{i-1},g{i-1})" for i in range(1, 45)] + ["OUTPUT(g44)"]
        self.path.write_text("\n".join(rows), encoding="utf-8")
        capped = load_circuit_graph(self.path)
        self.assertEqual(max(capped.cc1), 10**9)
        self.assertTrue(torch.isfinite(capped.x).all())

    def test_mc_gae_update_encoder_and_freeze_old(self):
        for method in ("mc", "gae"):
            agent = self.agent(method)
            gates = self.gates()
            before = {k: v.clone() for k, v in agent.policy_old.state_dict().items()}
            for value in (0, 1):
                agent.select_backtrace_action(gates["y"], value, [gates["n"], gates["b"]])
                agent.add_reward(-1 if value == 0 else 2)
            agent.finish_episode(100)
            for k, v in before.items():
                torch.testing.assert_close(agent.policy_old.state_dict()[k], v, rtol=0, atol=0)
            metrics = agent.update()
            self.assertEqual(metrics["steps"], 2)
            self.assertTrue(any(not torch.equal(before[k], v)
                                for k, v in agent.policy.state_dict().items() if k.startswith("graph_encoder.")))
            self.assertFalse(agent.policy_old._embedding_cache)
            restored = self.agent(method)
            restored.load_training_state_dict(agent.training_state_dict())
            for key, value in agent.policy.state_dict().items():
                torch.testing.assert_close(value, restored.policy.state_dict()[key])

    def test_cache_invalidates_on_load(self):
        policy = SmartATPGPolicy()
        with torch.no_grad():
            before = policy.graph_embeddings(self.graph, cached=True).clone()
            state = {k: v.clone() for k, v in policy.state_dict().items()}
            state["graph_encoder.layer.bias"].add_(2)
            policy.load_state_dict(state)
            after = policy.graph_embeddings(self.graph, cached=True)
        self.assertFalse(torch.equal(before, after))

    def test_rollout_mask_is_owned(self):
        agent = self.agent()
        gates = self.gates()
        mask = torch.tensor([True, True])
        agent.select_backtrace_action(gates["y"], 0, [gates["n"], gates["b"]], mask)
        mask.fill_(False)
        self.assertEqual(agent.buffer.steps[0].action_mask.tolist(), [True, True])
        self.assertEqual(agent.buffer.steps[0].objective_embedding.numel(), 0)

    def test_embedding_and_policy_state_dimensions(self):
        policy = SmartATPGPolicy()
        embeddings = policy.graph_embeddings(self.graph)
        descriptors = policy.descriptors(
            self.graph, [self.graph.name_to_index["y"]], [[1, 0]], embeddings
        )
        self.assertEqual(embeddings.shape[1], GATE_EMBEDDING_DIM)
        self.assertEqual(GATE_EMBEDDING_DIM, 11)
        self.assertEqual(descriptors.shape, (1, POLICY_STATE_DIM))
        self.assertEqual(POLICY_STATE_DIM, 13)
        self.assertEqual(descriptors[0, -2:].tolist(), [1.0, 0.0])

    def test_smartatpg_default_learning_rates(self):
        agent = SmartATPGPPOAgent({"test": self.graph}, rnd_beta=0)
        self.assertEqual(agent.lr_actor, 0.001)
        self.assertEqual(agent.lr_critic, 0.01)

    def test_curriculum_evaluator_is_deterministic_and_read_only(self):
        agent = self.agent()
        evaluator = CppPodemCurriculumEvaluator(
            self.graph,
            {"fault": {"backtracks": 10, "backtrace_steps": 100}},
            agent,
        )
        before = {key: value.detach().clone() for key, value in agent.policy_old.state_dict().items()}
        request = {
            "mode": "backtrace",
            "objective_name": "y",
            "objective_value": 1,
            "candidate_names": ["n", "b"],
        }
        first = evaluator.decision_callback(request)
        second = evaluator.decision_callback(request)
        self.assertEqual(first, second)
        self.assertFalse(agent.buffer.steps)

        evaluator.event_callback({"event": "episode_start", "fault_id": "fault"})
        evaluator.event_callback({
            "event": "episode_end",
            "fault_id": "fault",
            "outcome": 1,
            "backtracks": 5,
            "backtrace_steps": 50,
        })
        self.assertAlmostEqual(evaluator.episode_metrics[0]["extrinsic_reward"], 115.0)
        self.assertEqual(agent.update_count, 0)
        self.assertFalse(agent.buffer.steps)
        for key, value in before.items():
            torch.testing.assert_close(value, agent.policy_old.state_dict()[key], rtol=0, atol=0)

    def test_generated_sidecars_cannot_overwrite_checkpoints(self):
        root = Path(self.temp.name)
        for checkpoint in (root / "best.txt.json", root / "best.txt.tmp",
                           root / "best_snapshots" / "weights.pth"):
            with self.assertRaises(ValueError):
                training_output_paths(checkpoint, root / "best.txt", root / "latest.txt", "smartatpg")
        training_output_paths(root / "state.pth", root / "best.txt", root / "latest.txt", "smartatpg")

    def test_deepgate_export_rejects_ppo_before_import(self):
        from rl_podem.deepgate_bridge import export_cpp_embeddings
        checkpoint = Path(self.temp.name) / "ppo.pth"
        torch.save({"format": "RL_PODEM_CURRICULUM_TRAINING_V4", "agent": {"policy": {}}}, checkpoint)
        with patch("rl_podem.deepgate_bridge._ensure_deepgate_importable") as importer:
            with self.assertRaisesRegex(ValueError, "DeepGate encoder checkpoint"):
                export_cpp_embeddings(self.path, checkpoint, Path(self.temp.name) / "wrong.emb")
        importer.assert_not_called()

    def test_paired_export_and_native_logits(self):
        import cpp_podem
        agent = self.agent()
        state = agent.policy_old.state_dict()
        actor = Path(self.temp.name) / "actor.txt"
        embeddings = Path(self.temp.name) / "graph.emb"
        export_actor(state, actor)
        export_descriptors(state, self.graph, embeddings)
        _, table, metadata = _load_cpp_embedding_artifact(
            embeddings, expected_backend="smartatpg", include_metadata=True)
        self.assertEqual(metadata["snapshot"], snapshot_id(state))
        self.assertEqual(metadata["gate_embedding_dim"], "11")
        self.assertEqual(metadata["policy_state_dim"], "13")
        self.assertTrue(all(vector.numel() == 11 for vector in table.values()))
        cpp_podem.validate_actor_artifacts(str(embeddings), str(actor), self.graph.circuit_hash,
                                          list(self.graph.names), "smartatpg")
        duplicate_names = list(self.graph.names)
        duplicate_names[-1] = duplicate_names[0]
        with self.assertRaisesRegex(RuntimeError, "duplicate wire"):
            cpp_podem.validate_actor_artifacts(str(embeddings), str(actor), self.graph.circuit_hash,
                                              duplicate_names, "smartatpg")
        unicode_directory = Path(self.temp.name) / "\u4e2d\u6587"
        unicode_directory.mkdir()
        unicode_actor = unicode_directory / "actor.txt"
        unicode_embeddings = unicode_directory / "graph.emb"
        export_actor(state, unicode_actor)
        export_descriptors(state, self.graph, unicode_embeddings)
        cpp_podem.validate_actor_artifacts(str(unicode_embeddings), str(unicode_actor), self.graph.circuit_hash,
                                          list(self.graph.names), "smartatpg")
        policy = policy_from_state(state)
        for name, vector in table.items():
            for value in (0, 1):
                descriptor = torch.cat((vector, torch.ones(2)))
                with torch.no_grad():
                    expected = policy.batch_logits(descriptor.unsqueeze(0), [value])[0][0]
                    single, _ = policy.backtrace_logits(descriptor, value)
                    torch.testing.assert_close(single, expected)
                actual = torch.tensor(cpp_podem.score_actor_v2(str(actor), descriptor.tolist(), value))
                torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)
                self.assertEqual(int(expected.argmax()), int(actual.argmax()))
        with self.assertRaises(ValueError):
            _load_cpp_embedding_artifact(embeddings)
        with self.assertRaises(ValueError):
            export_actor_v2_state_dict(state, actor)
        with self.assertRaises(RuntimeError):
            cpp_podem.validate_actor_artifacts(str(embeddings), str(actor), self.graph.circuit_hash,
                                              list(self.graph.names), "deepgate")
        different = {key: value.clone() for key, value in state.items()}
        different["graph_encoder.layer.bias"].add_(1)
        export_actor(different, actor)
        with self.assertRaisesRegex(RuntimeError, "snapshot mismatch"):
            cpp_podem.validate_actor_artifacts(str(embeddings), str(actor), self.graph.circuit_hash,
                                              list(self.graph.names), "smartatpg")

    def test_v5_contains_graphsage_and_portable_inference_matches_torch(self):
        state = self.agent().policy_old.state_dict()
        model_path = Path(self.temp.name) / "model_v5.txt"
        export_actor(state, model_path, best_round=4, best_score=(-200, 3, 40, -5, 4))
        model = load_portable_model(model_path)
        self.assertEqual(model.best_round, 4)
        self.assertEqual(model.best_score, (-200.0, 3.0, 40.0, -5.0, 4.0))
        self.assertIn("graph_encoder.layer.weight", model.tensors)
        portable_graph = load_portable_graph(self.path)
        portable = torch.tensor(compute_portable_embeddings(model, portable_graph))
        with torch.no_grad():
            expected = policy_from_state(state).graph_embeddings(self.graph)
        torch.testing.assert_close(portable, expected, atol=1e-6, rtol=1e-5)

        changed = {key: value.clone() for key, value in state.items()}
        changed["graph_encoder.layer.bias"].add_(0.5)
        changed_path = Path(self.temp.name) / "changed_v5.txt"
        export_actor(changed, changed_path)
        changed_model = load_portable_model(changed_path)
        changed_embedding = compute_portable_embeddings(changed_model, portable_graph)
        self.assertNotEqual(changed_embedding, tuple(map(tuple, portable.tolist())))

    def test_legacy_80d_smartatpg_artifact_is_rejected(self):
        import cpp_podem
        legacy = Path(self.temp.name) / "legacy.emb"
        legacy.write_text(
            "SMARTATPG_EMBEDDINGS_V2\n"
            "backend smartatpg\n"
            "feature_schema SMARTATPG_FEATURES_V1\n"
            f"snapshot {'0' * 64}\n"
            f"circuit_hash {self.graph.circuit_hash}\n"
            "dimension 80\n"
            "count 0\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Legacy 80-dimensional"):
            _load_cpp_embedding_artifact(legacy, expected_backend="smartatpg")
        actor = Path(self.temp.name) / "actor.txt"
        export_actor(self.agent().policy_old.state_dict(), actor)
        with self.assertRaisesRegex(RuntimeError, "Legacy 80-dimensional"):
            cpp_podem.validate_actor_artifacts(
                str(legacy), str(actor), self.graph.circuit_hash,
                list(self.graph.names), "smartatpg",
            )


if __name__ == "__main__":
    unittest.main()

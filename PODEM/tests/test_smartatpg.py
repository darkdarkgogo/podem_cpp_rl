import dataclasses
import math
from collections import Counter
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rl_podem.smartatpg_features import (
    FEATURE_DIM, GATE_TYPES, GRAPH_CONFIG, load_circuit_graph,
)
from rl_podem.smartatpg import (
    ACTION_MASK_DIM, ACTOR_INPUT_DIM, DECISION_STATE_DIM, GATE_EMBEDDING_DIM,
    POLICY_STATE_DIM, GraphGate, SmartATPGPPOAgent, SmartATPGPolicy,
)
from rl_podem.gat_gru import (
    GATGRUSmartATPGPPOAgent, GATGRUSmartATPGPolicy,
)
from rl_podem.curriculum import CppPodemCurriculumEvaluator
from rl_podem.cpp_bridge import (
    _load_cpp_embedding_artifact, catalog_cpp_podem, export_actor_v2_state_dict,
)
from rl_podem.smartatpg_artifacts import export_actor, export_descriptors, snapshot_id, policy_from_state
from rl_podem.artifact_paths import training_output_paths
from smartatpg_portable import (
    compute_embeddings as compute_portable_embeddings,
    export_embeddings as export_portable_embeddings,
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
        self.assertEqual(FEATURE_DIM, 12)
        self.assertEqual(g.x.shape, (6, FEATURE_DIM))
        self.assertEqual((g.cc0[i["y"]], g.cc1[i["y"]]), (2, 4))
        self.assertEqual((g.cc0[i["q"]], g.cc1[i["q"]]), (2, 4))
        self.assertEqual(g.co[i["y"]], 0)
        self.assertEqual(g.co[i["q"]], 0)
        self.assertEqual(g.co[i["n"]], 2)
        self.assertEqual(g.co[i["a"]], 3)
        self.assertEqual(g.co[i["b"]], 3)
        self.assertTrue(math.isinf(g.co[i["unused"]]))
        self.assertEqual(float(g.x[i["unused"], 11]), 1.0)
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

    def test_mask_is_not_an_actor_input(self):
        policy = SmartATPGPolicy()
        embeddings = policy.graph_embeddings(self.graph)
        descriptors = policy.descriptors(
            self.graph, [self.graph.name_to_index["y"]], embeddings
        )
        self.assertEqual(embeddings.shape[1], GATE_EMBEDDING_DIM)
        self.assertEqual(GATE_EMBEDDING_DIM, 12)
        self.assertEqual(descriptors.shape, (1, ACTOR_INPUT_DIM))
        self.assertEqual(ACTOR_INPUT_DIM, 12)
        self.assertEqual(policy.backtrace_actor[0].in_features, ACTOR_INPUT_DIM)
        self.assertFalse(hasattr(policy, "gate_encoder"))
        self.assertFalse(hasattr(policy, "objective_value_embedding"))
        self.assertEqual(ACTION_MASK_DIM, 2)
        self.assertEqual(DECISION_STATE_DIM, 14)
        self.assertEqual(POLICY_STATE_DIM, 14)

        gates = self.gates()
        left = self.agent().select_backtrace_action_deterministic(
            gates["y"], 1, [gates["n"], gates["b"]], [True, False]
        )
        right = self.agent().select_backtrace_action_deterministic(
            gates["y"], 1, [gates["n"], gates["b"]], [False, True]
        )
        self.assertIs(left, gates["n"])
        self.assertIs(right, gates["b"])

    def test_smartatpg_default_learning_rates(self):
        agent = SmartATPGPPOAgent({"test": self.graph}, rnd_beta=0)
        self.assertEqual(agent.lr_actor, 0.001)
        self.assertEqual(agent.lr_critic, 0.01)

    def test_direct_actor_inputs_and_objective_concatenation(self):
        from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer
        for policy_class, agent_class, width in (
            (SmartATPGPolicy, SmartATPGPPOAgent, 12),
            (GATGRUSmartATPGPolicy, GATGRUSmartATPGPPOAgent, 13),
        ):
            policy = policy_class()
            self.assertFalse(hasattr(policy, "gate_encoder"))
            self.assertFalse(hasattr(policy, "objective_value_embedding"))
            self.assertEqual(policy.backtrace_actor[0].in_features, width)
            self.assertEqual(policy.critic[0].in_features, width)
            seen = []
            critic_seen = []
            actor_hook = policy.backtrace_actor.register_forward_pre_hook(
                lambda module, args: seen.append(args[0].detach().clone())
            )
            critic_hook = policy.critic.register_forward_pre_hook(
                lambda module, args: critic_seen.append(args[0].detach().clone())
            )
            descriptor = torch.linspace(0.1, 1.2, 12)
            try:
                results = [policy.backtrace_logits(descriptor, value) for value in (0, 1)]
            finally:
                actor_hook.remove()
                critic_hook.remove()
            for value in (0, 1):
                expected = descriptor if width == 12 else torch.cat((descriptor, torch.tensor([float(value)])))
                torch.testing.assert_close(seen[value], expected.unsqueeze(0), atol=0, rtol=0)
                torch.testing.assert_close(critic_seen[value], seen[value], atol=0, rtol=0)
            if width == 12:
                torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
            else:
                with self.assertRaisesRegex(ValueError, "binary objective"):
                    policy.backtrace_logits(descriptor, 2)
            agent = agent_class({"test": self.graph}, rnd_beta=0, k_epochs=1)
            self.assertEqual(agent.training_state_dict()["actor_input_dim"], width)
            restored = agent_class({"test": self.graph}, rnd_beta=0, k_epochs=1)
            restored.load_training_state_dict(agent.training_state_dict())
            CppPodemBacktraceV2Trainer(self.graph, agent=restored)
            old_state = dict(agent.training_state_dict(), feature_schema="SMARTATPG_FEATURES_V2_11D")
            with self.assertRaisesRegex(ValueError, "Incompatible SmartATPG checkpoint"):
                restored.load_training_state_dict(old_state)

    def test_co_models_export_through_benchmark_bundle(self):
        import cpp_podem
        import prepare_smartatpg_benchmark as prepare_bundle
        import benchmark_smartatpg as benchmark
        root = Path(self.temp.name)
        samples = root / "sample_circuits"
        samples.mkdir()
        (samples / "c432.bench").write_text(BENCH.replace("unused = BUF(a)\n", ""), encoding="utf-8")
        mean, gat = root / "mean.txt", root / "gat.txt"
        for policy, path in ((SmartATPGPolicy(), mean), (GATGRUSmartATPGPolicy(), gat)):
            export_actor(policy.state_dict(), path, best_round=1, best_score=(-1, 2, 3, -4, 1))
        bundle = root / "bundle"
        with patch.object(prepare_bundle, "ROOT", root), patch.object(prepare_bundle, "CIRCUITS", ("c432",)):
            manifest = prepare_bundle.prepare(bundle, mean, gat)
            self.assertEqual(prepare_bundle.prepare(bundle, mean, gat, resume=True), manifest)
        with patch.object(benchmark, "CIRCUITS", ("c432",)):
            benchmark._validate_manifest(manifest, bundle)
        self.assertEqual(manifest["gate_embedding_dim"], 12)
        self.assertEqual(manifest["models"]["smartatpg_mean"]["actor_input_dim"], 12)
        self.assertEqual(manifest["models"]["smartatpg_gat_gru"]["actor_input_dim"], 13)
        model_paths = {name: bundle / record["path"] for name, record in manifest["models"].items()}
        for item in manifest["circuits"]:
            item["circuit"] = str(bundle / item["circuit"])
            item["fault_map"] = str(bundle / item["fault_map"])
        models, _, _ = benchmark._prepare_models(model_paths, manifest, root / "comparison")
        graph = load_circuit_graph(manifest["circuits"][0]["circuit"])
        for name in model_paths:
            emb = models[name]["embeddings"]["c432"]
            cpp_podem.validate_actor_artifacts(str(emb), str(model_paths[name]), graph.circuit_hash, list(graph.names), "smartatpg")

    def test_co_export_rejects_11d_encoder_tensors(self):
        for policy in (SmartATPGPolicy(), GATGRUSmartATPGPolicy()):
            state = policy.state_dict()
            name = next(key for key in state if key.startswith("graph_encoder.") and key.endswith("weight"))
            state[name] = state[name][:-1]
            with self.assertRaisesRegex(ValueError, "graph tensor shape"):
                export_actor(state, Path(self.temp.name) / "bad_graph.txt")

    def test_direct_actor_artifacts_reject_invalid_output_shapes(self):
        for policy_class in (SmartATPGPolicy, GATGRUSmartATPGPolicy):
            state = policy_class().state_dict()
            model_path = Path(self.temp.name) / "invalid_actor.txt"
            export_actor(state, model_path)
            valid_text = model_path.read_text(encoding="utf-8")
            for name in ("backtrace_actor.0.bias", "backtrace_actor.2.weight", "backtrace_actor.2.bias"):
                with self.subTest(policy=policy_class.__name__, tensor=name):
                    invalid = dict(state)
                    shape = list(state[name].shape)
                    shape[0] += 1
                    invalid[name] = torch.zeros(shape)
                    with self.assertRaisesRegex(ValueError, "Actor tensor shape"):
                        export_actor(invalid, model_path)
                    lines = valid_text.splitlines()
                    index = next(i for i, line in enumerate(lines) if line.startswith(f"tensor {name} "))
                    rows, cols = ((1, shape[0]) if len(shape) == 1 else shape)
                    lines[index] = f"tensor {name} {rows} {cols}"
                    lines[index + 1] = " ".join(["0"] * (rows * cols))
                    model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Actor tensor shape"):
                        load_portable_model(model_path)

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
        self.assertEqual(metadata["gate_embedding_dim"], "12")
        self.assertEqual(metadata["actor_input_dim"], "12")
        self.assertEqual(metadata["action_mask_dim"], "2")
        self.assertEqual(metadata["decision_state_dim"], "14")
        self.assertTrue(all(vector.numel() == 12 for vector in table.values()))
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
                with torch.no_grad():
                    expected = policy.batch_logits(vector.unsqueeze(0), [value])[0][0]
                    single, _ = policy.backtrace_logits(vector, value)
                    torch.testing.assert_close(single, expected)
                actual = torch.tensor(cpp_podem.score_actor_v2(str(actor), vector.tolist(), value))
                torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)
                self.assertEqual(int(expected.argmax()), int(actual.argmax()))
        _load_cpp_embedding_artifact(embeddings)
        with self.assertRaises(ValueError):
            export_actor_v2_state_dict(state, actor)
        with self.assertRaises(RuntimeError):
            cpp_podem.validate_actor_artifacts(str(embeddings), str(actor), self.graph.circuit_hash,
                                              list(self.graph.names), "unsupported")
        different = {key: value.clone() for key, value in state.items()}
        different["graph_encoder.layer.bias"].add_(1)
        export_actor(different, actor)
        with self.assertRaisesRegex(RuntimeError, "snapshot mismatch"):
            cpp_podem.validate_actor_artifacts(str(embeddings), str(actor), self.graph.circuit_hash,
                                              list(self.graph.names), "smartatpg")

    def test_v8_contains_fanin_mean_encoder_and_portable_inference_matches_torch(self):
        state = self.agent().policy_old.state_dict()
        model_path = Path(self.temp.name) / "model_v8.txt"
        export_actor(state, model_path, best_round=4, best_score=(-200, 3, 40, -5, 4))
        model = load_portable_model(model_path)
        self.assertEqual(model.model_format, "SMARTATPG_MODEL_V8")
        self.assertEqual(model.actor_input_dim, 12)
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
        changed_path = Path(self.temp.name) / "changed_v8.txt"
        export_actor(changed, changed_path)
        changed_model = load_portable_model(changed_path)
        changed_embedding = compute_portable_embeddings(changed_model, portable_graph)
        self.assertNotEqual(changed_embedding, tuple(map(tuple, portable.tolist())))

    def test_gat_gru_dimensions_gradients_and_portable_parity(self):
        import cpp_podem
        policy = GATGRUSmartATPGPolicy()
        encoder = policy.graph_encoder
        self.assertFalse(hasattr(encoder, "input_projection"))
        self.assertEqual(tuple(encoder.forward_pass.projection.weight.shape), (12, 12))
        self.assertEqual(tuple(encoder.reverse_pass.projection.weight.shape), (12, 12))
        self.assertEqual(tuple(encoder.forward_pass.attention.shape), (24,))
        self.assertEqual(tuple(encoder.forward_pass.gru.weight_ih.shape), (36, 12))
        self.assertIsNot(
            encoder.forward_pass.projection.weight,
            encoder.reverse_pass.projection.weight,
        )
        embeddings = policy.graph_embeddings(self.graph)
        self.assertEqual(tuple(embeddings.shape), (len(self.graph.names), 12))
        embeddings.sum().backward()
        for direction in (encoder.forward_pass, encoder.reverse_pass):
            for parameter in direction.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

        model_path = Path(self.temp.name) / "gat_gru_v8.txt"
        export_actor(policy.state_dict(), model_path)
        model = load_portable_model(model_path)
        self.assertEqual(model.encoder_variant, "level_gat_gru")
        self.assertEqual(model.model_format, "SMARTATPG_MODEL_V8")
        self.assertEqual(model.actor_input_dim, 13)
        self.assertFalse(any(
            name.startswith(("gate_encoder.", "objective_value_embedding."))
            for name in model.tensors
        ))
        portable = torch.tensor(compute_portable_embeddings(
            model, load_portable_graph(self.path)
        ))
        torch.testing.assert_close(portable, embeddings.detach(), atol=2e-6, rtol=2e-5)
        embedding_path = Path(self.temp.name) / "gat_gru.emb"
        export_descriptors(policy.state_dict(), self.graph, embedding_path)
        cpp_podem.validate_actor_artifacts(
            str(embedding_path), str(model_path), self.graph.circuit_hash,
            list(self.graph.names), "smartatpg",
        )
        vector = embeddings[0].detach().cpu()
        for value in (0, 1):
            expected, _ = policy.backtrace_logits(vector, value)
            actual = torch.tensor(
                cpp_podem.score_actor_v2(str(model_path), vector.tolist(), value)
            )
            torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)

        baseline_path = Path(self.temp.name) / "baseline_v8.txt"
        export_actor(SmartATPGPolicy().state_dict(), baseline_path)
        with self.assertRaisesRegex(RuntimeError, "graph configuration|encoder variant|snapshot"):
            cpp_podem.validate_actor_artifacts(
                str(embedding_path), str(baseline_path), self.graph.circuit_hash,
                list(self.graph.names), "smartatpg",
            )

    def test_gat_gru_agent_updates_both_directions(self):
        path = Path(self.temp.name) / "intermediate.bench"
        path.write_text(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\n"
            "mid=AND(a,b)\nout=OR(mid,c)\nOUTPUT(out)\n",
            encoding="utf-8",
        )
        graph = load_circuit_graph(path)
        agent = GATGRUSmartATPGPPOAgent(
            {"test": graph}, advantage_method="gae",
            normalize_returns=False, normalize_advantages=True,
            return_scale=100, rnd_beta=0, k_epochs=1,
        )
        gates = {
            name: GraphGate(name, graph.circuit_hash, index)
            for index, name in enumerate(graph.names)
        }
        agent.select_backtrace_action(gates["mid"], 1, [gates["a"], gates["b"]])
        agent.finish_episode(10)
        before = {key: value.clone() for key, value in agent.policy.state_dict().items()}
        agent.update()
        for prefix in (
            "graph_encoder.forward_pass", "graph_encoder.reverse_pass",
        ):
            self.assertTrue(any(
                key.startswith(prefix) and not torch.equal(before[key], value)
                for key, value in agent.policy.state_dict().items()
            ))

    def test_legacy_v5_model_and_v3_embeddings_remain_compatible(self):
        import cpp_podem
        from rl_podem.ppo import BacktraceActorCriticV2
        from rl_podem.smartatpg import MeanGraphEncoder
        legacy_policy = BacktraceActorCriticV2(11, 32)
        legacy_policy.graph_encoder = MeanGraphEncoder()
        legacy_policy.graph_encoder.layer = torch.nn.Linear(22, 11)
        state = {
            key: value.detach().cpu().clone()
            for key, value in legacy_policy.state_dict().items()
            if not key.startswith("critic.")
        }
        state["gate_encoder.0.weight"] = torch.cat((
            state["gate_encoder.0.weight"],
            torch.zeros(state["gate_encoder.0.weight"].shape[0], 2),
        ), dim=1)
        snapshot = "1" * 64
        model_path = Path(self.temp.name) / "legacy_v5.txt"
        lines = [
            "SMARTATPG_MODEL_V5", "backend smartatpg",
            "feature_schema SMARTATPG_FEATURES_V2_11D",
            "graph_config fanin_mean_1x22x11", "gate_embedding_dim 11",
            "policy_state_dim 13", f"snapshot {snapshot}",
            "best_round 1", "best_score -1,2,3,-4,1", "hidden_dim 32",
        ]
        tensor_names = (
            "graph_encoder.layer.weight", "graph_encoder.layer.bias",
            "gate_encoder.0.weight", "gate_encoder.0.bias",
            "objective_value_embedding.weight", "backtrace_actor.0.weight",
            "backtrace_actor.0.bias", "backtrace_actor.2.weight",
            "backtrace_actor.2.bias",
        )
        for name in tensor_names:
            tensor = state[name]
            rows, cols = ((1, tensor.numel()) if tensor.ndim == 1 else tensor.shape)
            values = " ".join(format(float(value), ".9g") for value in tensor.flatten())
            lines.append(f"tensor {name} {rows} {cols} {values}")
        lines.append("end")
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        model = load_portable_model(model_path)
        self.assertEqual(model.model_format, "SMARTATPG_MODEL_V5")
        self.assertEqual(model.actor_input_dim, 13)
        embedding_path = Path(self.temp.name) / "legacy_v3.emb"
        graph = load_portable_graph(self.path)
        export_portable_embeddings(model, graph, embedding_path)
        self.assertEqual(
            embedding_path.read_text(encoding="utf-8").splitlines()[0],
            "SMARTATPG_EMBEDDINGS_V3",
        )
        cpp_podem.validate_actor_artifacts(
            str(embedding_path), str(model_path), self.graph.circuit_hash,
            list(self.graph.names), "smartatpg",
        )
        embedding = compute_portable_embeddings(model, graph)[0]
        logits = cpp_podem.score_actor_v2(
            str(model_path), [*embedding, 1.0, 0.0], 1
        )
        self.assertEqual(len(logits), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(logits))))

    def test_native_backtrace_lock_reuses_an_unfinished_rl_choice(self):
        import cpp_podem
        fixture = Path(__file__).resolve().parents[1] / "sample_circuits/c432_binary.bench"
        circuit = Path(self.temp.name) / fixture.name
        fault_map = circuit.with_suffix(".faultmap")
        shutil.copy2(fixture, circuit)
        shutil.copy2(fixture.with_suffix(".faultmap"), fault_map)
        fault_id = catalog_cpp_podem(circuit, fault_map)["faults"][0]["fault_id"]
        decisions = []
        backtrace_steps = []

        def choose(request):
            action = 0 if request["action_mask"][0] else 1
            decisions.append({
                "sequence": int(request["sequence"]),
                "objective": request["objective_name"],
                "mask": tuple(request["action_mask"]),
                "action": action,
            })
            return action

        def event(value):
            if value["event"] == "backtrace_step":
                backtrace_steps.append(int(value["decision_sequence"]))

        cpp_podem.run_stuck_at(
            str(circuit), choose, event, 20, 14, [fault_id], True,
            "backtrace_rl", str(fault_map),
        )
        sequences = [request["sequence"] for request in decisions]
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertTrue(any(request["mask"] == (True, True) for request in decisions))
        self.assertTrue(any(sum(request["mask"]) == 1 for request in decisions))
        self.assertTrue(all(
            request["mask"][request["action"]] for request in decisions
        ))
        self.assertGreater(max(Counter(backtrace_steps).values()), 1)

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

    def test_embedding_v1_artifact_is_rejected(self):
        legacy = Path(self.temp.name) / "v1.emb"
        legacy.write_text("SMARTATPG_EMBEDDINGS_V1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported embedding format"):
            _load_cpp_embedding_artifact(legacy)


if __name__ == "__main__":
    unittest.main()

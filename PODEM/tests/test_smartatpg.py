import dataclasses
from collections import Counter
import json
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
    CppPodemBacktraceV2Trainer, _load_cpp_embedding_artifact,
    catalog_cpp_podem, export_actor_v2_state_dict,
)
from rl_podem.smartatpg_rewards import (
    GAT_REWARD_SCHEME,
    MEAN_REWARD_SCHEME,
    reward_scheme_for_encoder,
    smartatpg_backtrack_reward,
    smartatpg_pi_reward,
)
from rl_podem.smartatpg_artifacts import (
    export_actor as _export_actor,
    export_descriptors,
    encoder_variant,
    snapshot_id,
    policy_from_state,
)
from rl_podem.artifact_paths import training_output_paths
from rl_podem.smartatpg_portable import (
    CIRCUITS,
    compute_embeddings as compute_portable_embeddings,
    load_graph as load_portable_graph,
    load_model as load_portable_model,
)


BENCH = """INPUT(a)
INPUT(b)
n = NOT(a)
y = AND(n, b)
q = NOR(n, b)
OUTPUT(y)
OUTPUT(q)
"""

TRAINING_PROTOCOL = {
    "manifest_hash": "a" * 64,
    "backtrack_limit": 100,
    "normal_rounds": 5,
    "training_circuit_count": 1024,
    "validation_circuit_count": 6,
    "reward_scheme": "legacy_pi_exponential",
}
MEAN_BATCHED_TRAINING_PROTOCOL = {
    **TRAINING_PROTOCOL,
    "normal_rounds": 2,
    "faults_per_update": 8,
    "k_epochs": 1,
}
GAT_BATCHED_TRAINING_PROTOCOL = {
    **MEAN_BATCHED_TRAINING_PROTOCOL,
    "reward_scheme": GAT_REWARD_SCHEME,
    "k_epochs": 4,
}


def export_actor(state, path, **kwargs):
    variant = encoder_variant(state)
    default_protocol = (
        GAT_BATCHED_TRAINING_PROTOCOL
        if variant == "level_gat_gru"
        else TRAINING_PROTOCOL
    )
    kwargs["training_protocol"] = {
        **kwargs.get("training_protocol", default_protocol),
        "reward_scheme": reward_scheme_for_encoder(variant),
    }
    return _export_actor(state, path, **kwargs)


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

    def test_backtrack_reward_protocol(self):
        self.assertEqual(
            reward_scheme_for_encoder("level_gat_gru"), GAT_REWARD_SCHEME,
        )
        self.assertEqual(
            reward_scheme_for_encoder("fanin_mean"), MEAN_REWARD_SCHEME,
        )
        self.assertAlmostEqual(
            smartatpg_backtrack_reward(1), -0.500009802960494,
        )
        self.assertAlmostEqual(
            sum(smartatpg_backtrack_reward(i) for i in range(1, 101)),
            -300.0,
            places=6,
        )
        for invalid in (0, 101):
            with self.assertRaises(ValueError):
                smartatpg_backtrack_reward(invalid)

    def test_native_reward_diagnostic_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "src/python_bindings.cpp"
        ).read_text(encoding="utf-8")
        required_fragments = (
            "SMARTATPG_REWARD_WARN",
            "SMARTATPG_NONFINITE",
            "exponent >= 600.0",
            "!std::isfinite(step_reward)",
            "!std::isfinite(reward_after)",
            "fault=%s",
            "seq=%lu",
            "B=%d",
            "P=%lu",
            "BplusP=%.0f",
            "exponent=%.6f",
            "reward_before=%.17g",
            "step_reward=%.17g",
            "reward_after=%.17g",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_initial_mandatory_implication_emits_no_pi_reward_event(self):
        import cpp_podem

        source = (
            Path(__file__).resolve().parents[1]
            / "data/train_mean/c6288.bench"
        )
        circuit = Path(self.temp.name) / "c6288.bench"
        shutil.copyfile(source, circuit)
        events = []
        cpp_podem.run_stuck_at(
            str(circuit),
            lambda request: int(request["heuristic_action"]),
            events.append,
            100,
            14,
            ["dummy_gate1:GO:sa0", "G548:GI0:sa1"],
            True,
            "backtrace_rl",
            "",
            True,
        )
        invalid = [
            event
            for event in events
            if event["event"] == "pi_not_done"
            and (
                int(event["decision_sequence"]) == 0
                or int(event["pi_visits"]) == 0
            )
        ]
        self.assertEqual(invalid, [])
        valid = [
            event
            for event in events
            if event["event"] == "pi_not_done"
            and int(event["decision_sequence"]) > 0
            and int(event["pi_visits"]) > 0
        ]
        self.assertTrue(valid)

    def test_mean_ignores_pi_event_without_policy_step(self):
        trainer = CppPodemBacktraceV2Trainer(
            self.graph,
            agent=self.agent(),
            auto_update=False,
            reward_scheme=MEAN_REWARD_SCHEME,
        )
        trainer.event_callback({"event": "episode_start"})
        trainer.event_callback({
            "event": "pi_not_done",
            "decision_sequence": 0,
            "backtracks": 0,
            "pi_visits": 0,
        })
        self.assertEqual(trainer._episode_extrinsic_reward, 0.0)
        self.assertEqual(trainer._legacy_pi_reward_sum, 0.0)

    def test_encoder_specific_reward_events(self):
        cases = (
            (
                SmartATPGPPOAgent,
                MEAN_REWARD_SCHEME,
                100.0 - 0.1 + smartatpg_pi_reward(2, 1),
            ),
            (
                GATGRUSmartATPGPPOAgent,
                GAT_REWARD_SCHEME,
                100.0 - 0.1
                + smartatpg_backtrack_reward(1)
                + smartatpg_backtrack_reward(2),
            ),
        )
        for agent_class, scheme, expected in cases:
            agent = agent_class(
                {"test": self.graph}, rnd_beta=0, k_epochs=1,
            )
            trainer = CppPodemBacktraceV2Trainer(
                self.graph, agent=agent, auto_update=False,
                reward_scheme=scheme,
            )
            trainer.event_callback({"event": "episode_start"})
            trainer.decision_callback({
                "mode": "backtrace", "objective_name": "y",
                "objective_value": 1, "candidate_names": ["n", "b"],
                "action_mask": [True, True], "sequence": 1,
            })
            trainer.event_callback({
                "event": "backtrace_step", "decision_sequence": 1,
            })
            trainer.event_callback({
                "event": "backtrack", "decision_sequence": 1,
            })
            trainer.event_callback({
                "event": "backtrack", "decision_sequence": 1,
            })
            trainer.event_callback({
                "event": "pi_not_done", "decision_sequence": 1,
                "backtracks": 2, "pi_visits": 1,
            })
            trainer.event_callback({
                "event": "episode_end", "fault_id": "f0", "outcome": 1,
                "backtracks": 2, "backtrace_steps": 1, "pi_visits": 1,
            })
            self.assertAlmostEqual(
                trainer.episode_metrics[-1]["extrinsic_reward_sum"], expected,
            )

    def test_features_and_controllability(self):
        g = self.graph
        i = g.name_to_index
        self.assertEqual(GATE_TYPES, ("PI", "AND", "NAND", "OR", "NOR", "NOT"))
        self.assertEqual(FEATURE_DIM, 11)
        self.assertEqual(g.x.shape, (5, FEATURE_DIM))
        self.assertEqual((g.cc0[i["y"]], g.cc1[i["y"]]), (2, 4))
        self.assertEqual((g.cc0[i["q"]], g.cc1[i["q"]]), (2, 4))
        self.assertEqual(g.co[i["y"]], 0)
        self.assertEqual(g.co[i["q"]], 0)
        self.assertEqual(g.co[i["n"]], 2)
        self.assertEqual(g.co[i["a"]], 3)
        self.assertEqual(g.co[i["b"]], 3)
        self.assertTrue(torch.isfinite(g.x).all())
        self.assertEqual(g.fanouts[i["n"]], 2)
        self.assertEqual(GRAPH_CONFIG["layers"], 1)

    def test_invalid_inputs(self):
        for text in ("INPUT(a)\nOUTPUT(missing)",
                     "a=NOT(b)\nb=NOT(a)\nOUTPUT(a)",
                     "INPUT(a)\ny=AND(a,a,a)\nOUTPUT(y)",
                     "INPUT(a)\na=NOT(a)\nOUTPUT(a)",
                      "INPUT(a)\nINPUT(b)\ny=XOR(a,b)\nOUTPUT(y)",
                      "INPUT(a)\nINPUT(b)\ny=XNOR(a,b)\nOUTPUT(y)",
                      "INPUT(a)\ny=BUF(a)\nOUTPUT(y)",
                      "INPUT(a)\ny=BUFF(a)\nOUTPUT(y)"):
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
            layer.weight[0, 6] = 1
            layer.weight[1, FEATURE_DIM + 6] = 1
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
        self.assertEqual(GATE_EMBEDDING_DIM, 11)
        self.assertEqual(descriptors.shape, (1, ACTOR_INPUT_DIM))
        self.assertEqual(ACTOR_INPUT_DIM, 11)
        self.assertEqual(policy.backtrace_actor[0].in_features, ACTOR_INPUT_DIM)
        self.assertFalse(hasattr(policy, "gate_encoder"))
        self.assertFalse(hasattr(policy, "objective_value_embedding"))
        self.assertEqual(ACTION_MASK_DIM, 2)
        self.assertEqual(DECISION_STATE_DIM, 13)
        self.assertEqual(POLICY_STATE_DIM, 13)

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

        gat_agent = GATGRUSmartATPGPPOAgent({"test": self.graph}, rnd_beta=0)
        self.assertEqual(gat_agent.lr_actor, 0.0003)
        self.assertEqual(gat_agent.lr_critic, 0.001)
        self.assertEqual(gat_agent.k_epochs, 4)

    def test_deferred_trainer_collects_eight_faults_for_one_update(self):
        from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer

        agent = SmartATPGPPOAgent(
            {"test": self.graph}, rnd_beta=0, k_epochs=1,
        )
        trainer = CppPodemBacktraceV2Trainer(
            self.graph, agent=agent, auto_update=False,
        )
        for index in range(8):
            trainer.event_callback({"event": "episode_start"})
            trainer.decision_callback({
                "mode": "backtrace",
                "objective_name": "y",
                "objective_value": 1,
                "candidate_names": ["n", "b"],
                "action_mask": [True, True],
                "sequence": index + 1,
            })
            trainer.event_callback({
                "event": "episode_end", "fault_id": f"f{index}",
                "outcome": 1, "backtracks": 0,
                "backtrace_steps": 1, "pi_visits": 1,
            })
            self.assertEqual(agent.update_count, 0)
        self.assertEqual(len(agent.buffer.steps), 8)
        metrics = agent.update()
        self.assertEqual(agent.update_count, 1)
        self.assertEqual(metrics["epochs"], 1)
        self.assertEqual(metrics["steps"], 8)
        self.assertEqual(len(agent.buffer.steps), 0)

    def test_deferred_empty_fault_does_not_reward_previous_trajectory(self):
        from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer

        agent = SmartATPGPPOAgent(
            {"test": self.graph}, rnd_beta=0, k_epochs=1,
        )
        trainer = CppPodemBacktraceV2Trainer(
            self.graph, agent=agent, auto_update=False,
        )
        trainer.event_callback({"event": "episode_start"})
        trainer.decision_callback({
            "mode": "backtrace", "objective_name": "y",
            "objective_value": 1, "candidate_names": ["n", "b"],
            "action_mask": [True, True], "sequence": 1,
        })
        trainer.event_callback({
            "event": "episode_end", "fault_id": "with-step", "outcome": 1,
            "backtracks": 0, "backtrace_steps": 1, "pi_visits": 1,
        })
        previous_reward = agent.buffer.steps[-1].reward

        trainer.event_callback({"event": "episode_start"})
        trainer.event_callback({
            "event": "episode_end", "fault_id": "without-step", "outcome": 0,
            "backtracks": 0, "backtrace_steps": 0, "pi_visits": 0,
        })

        self.assertEqual(len(agent.buffer.steps), 1)
        self.assertEqual(agent.buffer.steps[-1].reward, previous_reward)
        self.assertEqual(trainer.episode_metrics[-1]["steps"], 0)

    def test_direct_actor_inputs_and_objective_concatenation(self):
        from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer
        for policy_class, agent_class, width in (
            (SmartATPGPolicy, SmartATPGPPOAgent, 11),
            (GATGRUSmartATPGPolicy, GATGRUSmartATPGPPOAgent, 12),
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
            descriptor = torch.linspace(0.1, 1.1, 11)
            try:
                results = [policy.backtrace_logits(descriptor, value) for value in (0, 1)]
            finally:
                actor_hook.remove()
                critic_hook.remove()
            for value in (0, 1):
                expected = descriptor if width == 11 else torch.cat((descriptor, torch.tensor([float(value)])))
                torch.testing.assert_close(seen[value], expected.unsqueeze(0), atol=0, rtol=0)
                torch.testing.assert_close(critic_seen[value], seen[value], atol=0, rtol=0)
            if width == 11:
                torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
            else:
                with self.assertRaisesRegex(ValueError, "binary objective"):
                    policy.backtrace_logits(descriptor, 2)
            agent = agent_class({"test": self.graph}, rnd_beta=0, k_epochs=1)
            self.assertEqual(agent.training_state_dict()["actor_input_dim"], width)
            restored = agent_class({"test": self.graph}, rnd_beta=0, k_epochs=1)
            restored.load_training_state_dict(agent.training_state_dict())
            CppPodemBacktraceV2Trainer(self.graph, agent=restored)
            old_state = dict(agent.training_state_dict(), feature_schema="SMARTATPG_FEATURES_V3_12D_CO")
            with self.assertRaisesRegex(ValueError, "Incompatible SmartATPG checkpoint"):
                restored.load_training_state_dict(old_state)

    def test_co_export_rejects_11d_encoder_tensors(self):
        for policy in (SmartATPGPolicy(), GATGRUSmartATPGPolicy()):
            state = policy.state_dict()
            name = next(key for key in state if key.startswith("graph_encoder.") and key.endswith("weight"))
            state[name] = state[name][:-1]
            with self.assertRaisesRegex(ValueError, "graph tensor shape"):
                export_actor(state, Path(self.temp.name) / "bad_graph.txt")

    def test_v12_export_requires_data_split_training_protocol(self):
        with self.assertRaisesRegex(ValueError, "requires data-split"):
            _export_actor(
                SmartATPGPolicy().state_dict(),
                Path(self.temp.name) / "missing_protocol.txt",
            )

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

    def test_data_split_validation_evaluator_does_not_update_ppo_or_rnd(self):
        from rl_podem.cpp_bridge import CppPodemBacktraceV2Evaluator

        agent = GATGRUSmartATPGPPOAgent(
            {"test": self.graph}, rnd_beta=0.05, k_epochs=1
        )
        evaluator = CppPodemBacktraceV2Evaluator(self.graph, agent=agent)
        before = agent.training_state_dict()
        request = {
            "mode": "backtrace",
            "objective_name": "y",
            "objective_value": 1,
            "candidate_names": ["n", "b"],
            "action_mask": [True, True],
            "sequence": 1,
        }
        self.assertIn(evaluator.decision_callback(request), (0, 1))
        self.assertFalse(agent.buffer.steps)
        after = agent.training_state_dict()
        self.assertEqual(after["update_count"], before["update_count"])
        self.assertEqual(after["optimizer"], before["optimizer"])
        self.assertEqual(after["rnd_optimizer"], before["rnd_optimizer"])
        self.assertEqual(after["rnd_error_stats"], before["rnd_error_stats"])
        for section in ("policy", "policy_old", "rnd"):
            for key, value in before[section].items():
                torch.testing.assert_close(value, after[section][key], rtol=0, atol=0)

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
        self.assertEqual(metadata["gate_embedding_dim"], "11")
        self.assertEqual(metadata["actor_input_dim"], "11")
        self.assertEqual(metadata["action_mask_dim"], "2")
        self.assertEqual(metadata["decision_state_dim"], "13")
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

    def test_native_validation_matches_python_policy_without_callbacks(self):
        import cpp_podem
        from rl_podem.cpp_bridge import CppPodemBacktraceV2Evaluator
        from rl_podem.training import _evaluate_fault

        fault_ids = [
            item["fault_id"]
            for item in catalog_cpp_podem(self.path)["faults"]
        ]
        cases = (
            (MEAN_REWARD_SCHEME, self.agent()),
            (
                GAT_REWARD_SCHEME,
                GATGRUSmartATPGPPOAgent(
                    {"test": self.graph}, rnd_beta=0, k_epochs=4,
                ),
            ),
        )
        for scheme, agent in cases:
            with self.subTest(reward_scheme=scheme):
                state = agent.policy_old.state_dict()
                actor = Path(self.temp.name) / f"{scheme}.actor.txt"
                embeddings = Path(self.temp.name) / f"{scheme}.emb"
                journal = Path(self.temp.name) / f"{scheme}.jsonl"
                protocol = (
                    GAT_BATCHED_TRAINING_PROTOCOL
                    if scheme == GAT_REWARD_SCHEME
                    else MEAN_BATCHED_TRAINING_PROTOCOL
                )
                _export_actor(state, actor, training_protocol=protocol)
                export_descriptors(state, self.graph, embeddings)
                evaluator = CppPodemBacktraceV2Evaluator(
                    self.graph, agent=agent,
                )
                expected = [
                    _evaluate_fault(
                        evaluator,
                        {"name": "test", "circuit": str(self.path)},
                        fault_id,
                        100,
                        14,
                        scheme,
                    )
                    for fault_id in fault_ids
                ]
                native = cpp_podem.run_native_validation(
                    str(self.path), str(embeddings), str(actor), 100, 14,
                    fault_ids, scheme, str(journal), "test",
                )

                self.assertEqual(
                    [item["fault_id"] for item in native], fault_ids
                )
                journal_records = [
                    json.loads(line)
                    for line in journal.read_text("utf-8").splitlines()
                ]
                self.assertEqual(
                    [item["fault_id"] for item in journal_records], fault_ids
                )
                for actual, reference in zip(native, expected):
                    self.assertEqual(actual["outcome"], reference["outcome"])
                    self.assertEqual(
                        actual["backtracks"], reference["backtracks"]
                    )
                    self.assertEqual(
                        actual["backtrace_steps"],
                        reference["backtrace_steps"],
                    )
                    self.assertAlmostEqual(
                        actual["return"], reference["return"], places=9,
                    )
                    self.assertGreaterEqual(actual["atpg_seconds"], 0.0)

                with self.assertRaisesRegex(ValueError, "reward scheme"):
                    cpp_podem.run_native_validation(
                        str(self.path), str(embeddings), str(actor), 100, 14,
                        fault_ids, "unknown", "", "",
                    )
                mismatched_scheme = (
                    GAT_REWARD_SCHEME
                    if scheme == MEAN_REWARD_SCHEME
                    else MEAN_REWARD_SCHEME
                )
                with self.assertRaisesRegex(ValueError, "actor encoder"):
                    cpp_podem.run_native_validation(
                        str(self.path), str(embeddings), str(actor), 100, 14,
                        fault_ids, mismatched_scheme, "", "",
                    )
                with self.assertRaisesRegex(ValueError, "backtrack_limit=100"):
                    cpp_podem.run_native_validation(
                        str(self.path), str(embeddings), str(actor), 200, 14,
                        fault_ids, scheme, "", "",
                    )

    def test_native_scoap_validation_matches_python_policy(self):
        import cpp_podem
        from rl_podem.validation import ScoapValidationEvaluator
        from rl_podem.training import _evaluate_fault

        fault_ids = [
            item["fault_id"]
            for item in catalog_cpp_podem(self.path)["faults"]
        ]
        for scheme in (MEAN_REWARD_SCHEME, GAT_REWARD_SCHEME):
            with self.subTest(reward_scheme=scheme):
                evaluator = ScoapValidationEvaluator()
                expected = [
                    _evaluate_fault(
                        evaluator,
                        {"name": "test", "circuit": str(self.path)},
                        fault_id,
                        100,
                        14,
                        scheme,
                    )
                    for fault_id in fault_ids
                ]
                native = cpp_podem.run_native_scoap_validation(
                    str(self.path), 100, 14, fault_ids, scheme, "test",
                )

                self.assertEqual(
                    [item["fault_id"] for item in native], fault_ids
                )
                for actual, reference in zip(native, expected):
                    self.assertEqual(actual["fault_id"], reference["fault_id"])
                    self.assertEqual(actual["outcome"], reference["outcome"])
                    self.assertEqual(
                        actual["backtracks"], reference["backtracks"]
                    )
                    self.assertEqual(
                        actual["backtrace_steps"], reference["backtrace_steps"]
                    )
                    self.assertAlmostEqual(
                        actual["return"], reference["return"], places=9,
                    )
                    self.assertGreaterEqual(actual["atpg_seconds"], 0.0)

                with self.assertRaisesRegex(ValueError, "fault IDs"):
                    cpp_podem.run_native_scoap_validation(
                        str(self.path), 100, 14, [], scheme, "test",
                    )
                with self.assertRaisesRegex(ValueError, "reward scheme"):
                    cpp_podem.run_native_scoap_validation(
                        str(self.path), 100, 14, fault_ids, "unknown", "test",
                    )
                with self.assertRaisesRegex(ValueError, "backtrack_limit=100"):
                    cpp_podem.run_native_scoap_validation(
                        str(self.path), 200, 14, fault_ids, scheme, "test",
                    )

    def test_native_scoap_validation_counts_sequence_zero_reward_events(self):
        import cpp_podem
        from rl_podem.validation import ScoapValidationEvaluator
        from rl_podem.training import _evaluate_fault

        path = (
            Path(__file__).resolve().parent / "fixtures"
            / "native_scoap_sequence_zero.bench"
        )
        fault_id = "y:GO:sa0"
        for scheme in (MEAN_REWARD_SCHEME, GAT_REWARD_SCHEME):
            with self.subTest(reward_scheme=scheme):
                reference = _evaluate_fault(
                    ScoapValidationEvaluator(),
                    {"name": "sequence-zero", "circuit": str(path)},
                    fault_id,
                    100,
                    14,
                    scheme,
                )
                native = cpp_podem.run_native_scoap_validation(
                    str(path), 100, 14, [fault_id], scheme, "sequence-zero",
                )

                self.assertEqual(native[0]["fault_id"], fault_id)
                self.assertEqual(native[0]["outcome"], reference["outcome"])
                self.assertEqual(
                    native[0]["backtracks"], reference["backtracks"]
                )
                self.assertEqual(
                    native[0]["backtrace_steps"],
                    reference["backtrace_steps"],
                )
                self.assertAlmostEqual(
                    native[0]["return"], reference["return"], places=9,
                )
                self.assertAlmostEqual(native[0]["return"], 99.8, places=9)

    def test_v12_contains_fanin_mean_encoder_and_portable_inference_matches_torch(self):
        state = self.agent().policy_old.state_dict()
        model_path = Path(self.temp.name) / "model_v12.txt"
        export_actor(state, model_path, best_round=4, best_score=(-200, 3, 40, -5, 4))
        model = load_portable_model(model_path)
        self.assertEqual(model.model_format, "SMARTATPG_MODEL_V12")
        self.assertEqual(model.manifest_hash, "a" * 64)
        self.assertEqual(model.backtrack_limit, 100)
        self.assertEqual(model.reward_scheme, MEAN_REWARD_SCHEME)
        self.assertEqual(model.normal_rounds, 5)
        self.assertEqual(model.actor_input_dim, 11)
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
        changed_path = Path(self.temp.name) / "changed_v12.txt"
        export_actor(changed, changed_path)
        changed_model = load_portable_model(changed_path)
        changed_embedding = compute_portable_embeddings(changed_model, portable_graph)
        self.assertNotEqual(changed_embedding, tuple(map(tuple, portable.tolist())))

    def test_encoder_specific_actor_formats_and_training_protocols(self):
        import cpp_podem

        state = self.agent().policy_old.state_dict()
        training_protocol = MEAN_BATCHED_TRAINING_PROTOCOL
        model_path = Path(self.temp.name) / "model_v13.txt"
        embedding_path = Path(self.temp.name) / "model_v13.emb"
        _export_actor(
            state, model_path, training_protocol=training_protocol,
        )
        export_descriptors(state, self.graph, embedding_path)
        model = load_portable_model(model_path)
        self.assertEqual(
            model.model_format, "SMARTATPG_MODEL_V13_BATCH8_EPOCH1"
        )
        self.assertEqual(model.normal_rounds, 2)
        self.assertEqual(model.faults_per_update, 8)
        self.assertEqual(model.k_epochs, 1)
        cpp_podem.validate_actor_artifacts(
            str(embedding_path), str(model_path), self.graph.circuit_hash,
            list(self.graph.names), "smartatpg",
        )
        gat_state = GATGRUSmartATPGPPOAgent(
            {"test": self.graph}, rnd_beta=0, k_epochs=4,
        ).policy_old.state_dict()
        gat_path = Path(self.temp.name) / "model_v14.txt"
        gat_embeddings = Path(self.temp.name) / "model_v14.emb"
        _export_actor(
            gat_state,
            gat_path,
            training_protocol=GAT_BATCHED_TRAINING_PROTOCOL,
        )
        export_descriptors(gat_state, self.graph, gat_embeddings)
        gat_model = load_portable_model(gat_path)
        self.assertEqual(
            gat_model.model_format,
            "SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4",
        )
        self.assertEqual(gat_model.k_epochs, 4)
        cpp_podem.validate_actor_artifacts(
            str(gat_embeddings), str(gat_path), self.graph.circuit_hash,
            list(self.graph.names), "smartatpg",
        )
        old_gat_path = Path(self.temp.name) / "old_gat_v13.txt"
        old_gat_path.write_text(
            gat_path.read_text(encoding="utf-8").replace(
                "SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4",
                "SMARTATPG_MODEL_V13_BATCH8_EPOCH1",
                1,
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_portable_model(old_gat_path)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            cpp_podem.validate_actor_artifacts(
                str(gat_embeddings), str(old_gat_path),
                self.graph.circuit_hash, list(self.graph.names), "smartatpg",
            )
        with self.assertRaisesRegex(ValueError, "protocol metadata is invalid"):
            _export_actor(
                gat_state,
                Path(self.temp.name) / "old_gat_v13.txt",
                training_protocol={
                    **GAT_BATCHED_TRAINING_PROTOCOL,
                    "k_epochs": 1,
                },
            )

        old_checkpoint = GATGRUSmartATPGPPOAgent(
            {"test": self.graph}, rnd_beta=0, k_epochs=4,
        ).training_state_dict()
        old_checkpoint["format"] = (
            "RL_PODEM_SMARTATPG_GAT_GRU_PPO_V5_11D_CO_NO_BUF"
        )
        with self.assertRaisesRegex(ValueError, "Incompatible SmartATPG checkpoint"):
            GATGRUSmartATPGPPOAgent(
                {"test": self.graph}, rnd_beta=0, k_epochs=4,
            ).load_training_state_dict(old_checkpoint)

    def test_gat_gru_dimensions_gradients_and_portable_parity(self):
        import cpp_podem
        policy = GATGRUSmartATPGPolicy()
        encoder = policy.graph_encoder
        self.assertFalse(hasattr(encoder, "input_projection"))
        self.assertEqual(tuple(encoder.forward_pass.projection.weight.shape), (11, 11))
        self.assertEqual(tuple(encoder.reverse_pass.projection.weight.shape), (11, 11))
        self.assertEqual(tuple(encoder.forward_pass.attention.shape), (22,))
        self.assertEqual(tuple(encoder.forward_pass.gru.weight_ih.shape), (33, 11))
        self.assertIsNot(
            encoder.forward_pass.projection.weight,
            encoder.reverse_pass.projection.weight,
        )
        embeddings = policy.graph_embeddings(self.graph)
        self.assertEqual(tuple(embeddings.shape), (len(self.graph.names), 11))
        embeddings.sum().backward()
        for direction in (encoder.forward_pass, encoder.reverse_pass):
            for parameter in direction.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

        model_path = Path(self.temp.name) / "gat_gru_v14.txt"
        export_actor(
            policy.state_dict(), model_path,
            training_protocol=GAT_BATCHED_TRAINING_PROTOCOL,
        )
        model = load_portable_model(model_path)
        self.assertEqual(model.encoder_variant, "level_gat_gru")
        self.assertEqual(
            model.model_format, "SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4"
        )
        self.assertEqual(model.actor_input_dim, 12)
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

        baseline_path = Path(self.temp.name) / "baseline_v12.txt"
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
            return_scale=100, rnd_beta=0, k_epochs=4,
        )
        gates = {
            name: GraphGate(name, graph.circuit_hash, index)
            for index, name in enumerate(graph.names)
        }
        agent.select_backtrace_action(gates["mid"], 1, [gates["a"], gates["b"]])
        agent.finish_episode(10)
        before = {key: value.clone() for key, value in agent.policy.state_dict().items()}
        with patch.object(
            agent.optimizer, "step", wraps=agent.optimizer.step
        ) as optimizer_step:
            metrics = agent.update()
        self.assertEqual(optimizer_step.call_count, 4)
        self.assertEqual(metrics["epochs"], 4)
        for prefix in (
            "graph_encoder.forward_pass", "graph_encoder.reverse_pass",
        ):
            self.assertTrue(any(
                key.startswith(prefix) and not torch.equal(before[key], value)
                for key, value in agent.policy.state_dict().items()
            ))

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

    def test_native_summary_reports_nonnegative_atpg_seconds(self):
        import cpp_podem
        fault_id = catalog_cpp_podem(self.path)["faults"][0]["fault_id"]

        summary = cpp_podem.run_stuck_at(
            str(self.path),
            lambda request: 0 if request["action_mask"][0] else 1,
            None, 20, 14, [fault_id], True,
        )

        self.assertGreaterEqual(summary["atpg_seconds"], 0.0)

    def test_old_smartatpg_artifacts_are_rejected(self):
        import cpp_podem
        actor = Path(self.temp.name) / "actor.txt"
        export_actor(self.agent().policy_old.state_dict(), actor)
        for version in range(2, 7):
            legacy = Path(self.temp.name) / f"legacy_v{version}.emb"
            legacy.write_text(f"SMARTATPG_EMBEDDINGS_V{version}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported embedding format"):
                _load_cpp_embedding_artifact(legacy, expected_backend="smartatpg")
            with self.assertRaisesRegex(RuntimeError, "Unsupported embedding format"):
                cpp_podem.validate_actor_artifacts(
                    str(legacy), str(actor), self.graph.circuit_hash,
                    list(self.graph.names), "smartatpg",
                )
        for version in range(5, 12):
            legacy_model = Path(self.temp.name) / f"legacy_model_v{version}.txt"
            legacy_model.write_text(f"SMARTATPG_MODEL_V{version}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Unsupported SmartATPG model format"
            ):
                load_portable_model(legacy_model)

    def test_embedding_v1_artifact_is_rejected(self):
        legacy = Path(self.temp.name) / "v1.emb"
        legacy.write_text("SMARTATPG_EMBEDDINGS_V1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported embedding format"):
            _load_cpp_embedding_artifact(legacy)


if __name__ == "__main__":
    unittest.main()

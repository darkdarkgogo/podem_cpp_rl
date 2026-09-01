import copy
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

from rl_podem.advantages import full_fault_targets
from rl_podem.cpp_bridge import CppPodemBacktraceV2Trainer, EmbeddingGate
from rl_podem.curriculum import (
    REWARD_CONFIG, CppPodemCurriculumTrainer, pretrain_actor, rollout_length_stats,
)
from rl_podem.ppo import BacktracePPOAgentV2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_curriculum import (
    CHECKPOINT_FORMAT,
    ROUND_CHECKPOINT_FORMAT,
    _parse_args,
    _run_reward_split,
    _round_stage_plan,
    _validate_checkpoint_metadata,
)


def make_agent(**overrides):
    options = dict(
        gate_embedding_dim=2, hidden_dim=8, k_epochs=1, rnd_beta=0.0,
        normalize_returns=False, normalize_advantages=True,
        advantage_method="gae", gae_lambda=0.97, return_scale=100.0,
    )
    options.update(overrides)
    return BacktracePPOAgentV2(**options)


def collect(agent, rewards=(10.0, 100.0)):
    candidates = [EmbeddingGate("a", torch.tensor([1.0, 0.0])),
                  EmbeddingGate("b", torch.tensor([0.0, 1.0]))]
    for index, reward in enumerate(rewards):
        agent.select_backtrace_action(candidates[index % 2], index % 2, candidates)
        agent.add_reward(reward)
    agent.finish_episode(0.0)


class TargetTests(unittest.TestCase):
    def targets(self, **kwargs):
        options = dict(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)
        options.update(kwargs)
        return full_fault_targets([10.0, 100.0], torch.tensor([0.4, 0.5]),
                                  [False, True], **options)

    def test_hand_computed_mc(self):
        result = self.targets(advantage_method="mc")
        torch.testing.assert_close(result.value_targets, torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(result.raw_advantages, torch.tensor([0.6, 0.5]))

    def test_hand_computed_gae(self):
        result = self.targets()
        torch.testing.assert_close(result.raw_advantages, torch.tensor([0.51, 0.5]))
        torch.testing.assert_close(result.value_targets, torch.tensor([0.91, 1.0]))

    def test_lambda_one_equals_mc(self):
        for gamma in (0.0, 0.99, 1.0):
            with self.subTest(gamma=gamma):
                mc = self.targets(gamma=gamma, advantage_method="mc")
                gae = self.targets(gamma=gamma, gae_lambda=1.0)
                torch.testing.assert_close(gae.raw_advantages, mc.raw_advantages)
                torch.testing.assert_close(gae.value_targets, mc.value_targets)

    def test_lambda_zero_is_one_step_td(self):
        result = self.targets(gae_lambda=0.0)
        torch.testing.assert_close(result.raw_advantages, torch.tensor([0.15, 0.5]))
        torch.testing.assert_close(result.value_targets, torch.tensor([0.55, 1.0]))

    def test_terminal_boundary_does_not_leak(self):
        for method in ("mc", "gae"):
            result = full_fault_targets([100.0, -100.0], torch.tensor([0.7, 20.0]),
                                        [True, True], advantage_method=method,
                                        normalize_advantages=False)
            torch.testing.assert_close(result.value_targets, torch.tensor([1.0, -1.0]))

    def test_scaled_rewards_preserve_success_and_failure(self):
        for method in ("mc", "gae"):
            for reward in (100.0, -100.0):
                result = full_fault_targets([reward], torch.tensor(0.7), [True],
                                            advantage_method=method)
                self.assertEqual(result.value_targets.shape, (1,))
                self.assertAlmostEqual(result.value_targets.item(), reward / 100.0)
                torch.testing.assert_close(result.raw_advantages, result.actor_advantages)

    def test_actor_normalization_does_not_change_targets(self):
        raw = self.targets()
        normalized = self.targets(normalize_advantages=True)
        torch.testing.assert_close(raw.raw_advantages, normalized.raw_advantages)
        torch.testing.assert_close(raw.value_targets, normalized.value_targets)
        self.assertAlmostEqual(normalized.actor_advantages.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(normalized.actor_advantages.std(unbiased=False).item(), 1.0, places=5)
        self.assertNotEqual(normalized.actor_advantages.data_ptr(), normalized.raw_advantages.data_ptr())

    def test_constant_advantages_are_finite(self):
        result = full_fault_targets([100.0, 100.0], torch.zeros(2), [True, True])
        torch.testing.assert_close(result.actor_advantages, torch.zeros(2))
        torch.testing.assert_close(result.value_targets, torch.ones(2))

    def test_targets_are_detached(self):
        result = full_fault_targets(torch.tensor([100.0], requires_grad=True),
                                    torch.tensor([0.7], requires_grad=True), [True])
        for tensor in vars(result).values():
            self.assertFalse(tensor.requires_grad)

    def test_reject_unfinished_rollout(self):
        with self.assertRaisesRegex(ValueError, "terminal final step"):
            full_fault_targets([1.0], torch.zeros(1), [False])

    def test_reject_invalid_options_and_data(self):
        for options in ({"gamma": -0.1}, {"gamma": float("nan")},
                        {"gae_lambda": 1.1}, {"gae_lambda": float("inf")},
                        {"return_scale": 0}, {"return_scale": float("nan")},
                        {"advantage_method": "td"}, {"normalize_returns": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.targets(**options)
        for rewards, values, dones in (([], [], []), ([1], [0, 0], [True]),
                                       ([float("nan")], [0], [True]),
                                       ([1], [float("inf")], [True])):
            with self.assertRaises(ValueError):
                full_fault_targets(rewards, torch.tensor(values), dones)


class AgentTests(unittest.TestCase):
    def test_post_bc_critic_can_learn_state_dependent_values(self):
        torch.manual_seed(42)
        agent = make_agent()
        tables = {"test": {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}}
        samples = [dict(circuit="test", objective_name=name, objective_value=index,
                        action_counts=[1, 0], difficulty_counts={"easy": 1})
                   for index, name in enumerate(("a", "b"))]
        pretrain_actor(agent, samples, samples, tables, epochs=1, batch_size=2)
        for index, embedding in enumerate(tables["test"].values()):
            _, value = agent.policy_old.backtrace_logits(embedding, index)
            self.assertAlmostEqual(value.item(), 1.0)
        hidden_before = agent.policy.critic[0].weight.detach().clone()
        for _ in range(2):
            collect(agent)
            agent.update()
        self.assertFalse(torch.equal(hidden_before, agent.policy.critic[0].weight))

    def test_update_both_methods_uses_unnormalized_critic_targets(self):
        for method in ("mc", "gae"):
            with self.subTest(method=method):
                torch.manual_seed(42)
                agent = make_agent(advantage_method=method)
                collect(agent)
                before = copy.deepcopy(agent.policy.state_dict())
                expected = full_fault_targets(
                    [step.reward for step in agent.buffer.steps],
                    torch.stack([step.state_value for step in agent.buffer.steps]),
                    [False, True], advantage_method=method,
                )
                observed = []

                def value_loss(value, target):
                    self.assertFalse(target.requires_grad)
                    observed.append(target.detach().cpu())
                    return (value - target).square()

                with patch.object(agent.mse_loss, "forward", side_effect=value_loss):
                    metrics = agent.update()
                torch.testing.assert_close(torch.stack(observed), expected.value_targets.cpu())
                self.assertEqual(metrics["steps"], 2)
                self.assertEqual(metrics["advantage_method"], method)
                self.assertEqual(agent.update_count, 1)
                self.assertFalse(agent.buffer.steps)
                self.assertIsNone(agent.update())
                for prefix in ("backtrace_actor.", "critic.", "gate_encoder."):
                    self.assertTrue(any(not torch.equal(before[k], v)
                                        for k, v in agent.policy.state_dict().items()
                                        if k.startswith(prefix)))
                for key, value in agent.policy.state_dict().items():
                    torch.testing.assert_close(value, agent.policy_old.state_dict()[key])
                self.assertTrue(all(math.isfinite(v) for v in metrics.values()
                                    if isinstance(v, (int, float))))

    def test_one_step_update_and_roundtrip(self):
        agent = make_agent()
        collect(agent, (-100.0,))
        metrics = agent.update()
        self.assertAlmostEqual(metrics["value_target_mean"], -1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pth"
            torch.save(agent.training_state_dict(), path)
            restored = make_agent()
            restored.load_training_state_dict(torch.load(path, map_location="cpu"))
        self.assertEqual(restored.hyperparameters(), agent.hyperparameters())
        self.assertEqual(restored.update_count, 1)
        collect(agent)
        collect(restored)
        # Compare exact resume using the same rollout rather than new sampled actions.
        restored.buffer = copy.deepcopy(agent.buffer)
        agent.update()
        restored.update()
        for key, value in agent.policy.state_dict().items():
            torch.testing.assert_close(value, restored.policy.state_dict()[key])

    def test_checkpoint_options_cannot_silently_change(self):
        state = make_agent().training_state_dict()
        for changes in ({"advantage_method": "mc"}, {"gae_lambda": 0.5},
                        {"gamma": 1.0}, {"return_scale": 10.0},
                        {"normalize_advantages": False}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "hyperparameters changed"):
                make_agent(**changes).load_training_state_dict(state)

    def test_legacy_normalized_checkpoint_rejected_for_new_training(self):
        legacy = BacktracePPOAgentV2(2, hidden_dim=8).training_state_dict()
        for key in ("advantage_method", "gae_lambda", "normalize_advantages"):
            legacy["hyperparameters"].pop(key)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            make_agent().load_training_state_dict(legacy)
        # Old paper entry points can still explicitly resume their original semantics.
        restored = BacktracePPOAgentV2(2, hidden_dim=8)
        restored.load_training_state_dict(legacy)
        self.assertTrue(restored.normalize_returns)

    def test_actor_only_warm_start_preserves_fresh_critic(self):
        legacy = BacktracePPOAgentV2(2, hidden_dim=8)
        agent = make_agent()
        critic = copy.deepcopy(agent.policy.critic.state_dict())
        actor_state = {k: v for k, v in legacy.policy_old.state_dict().items()
                       if not k.startswith("critic.")}
        agent.load_actor_state_dict(actor_state)
        for key, value in actor_state.items():
            torch.testing.assert_close(value, agent.policy.state_dict()[key])
            torch.testing.assert_close(value, agent.policy_old.state_dict()[key])
        for key, value in critic.items():
            torch.testing.assert_close(value, agent.policy.critic.state_dict()[key])
        collect(agent)
        agent.update()
        with self.assertRaisesRegex(ValueError, "fresh agent"):
            agent.load_actor_state_dict(actor_state)

    def test_legacy_extreme_reward_remains_supported(self):
        for rewards in ((1e300,), (1e300, 0.0)):
            agent = BacktracePPOAgentV2(2, hidden_dim=8, k_epochs=1, rnd_beta=0.0)
            collect(agent, rewards)
            metrics = agent.update()
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()
                                if isinstance(value, (int, float))))
            if len(rewards) == 2:
                self.assertAlmostEqual(metrics["scaled_reward_std"] / 1e299, 5.0)


class ConfigurationTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self):
        args = _parse_args(["manifest", "state", "actor"])
        self.assertEqual((args.advantage_method, args.gae_lambda, args.gamma,
                          args.return_scale, args.normalize_advantages),
                         ("gae", 0.97, 0.99, 100.0, True))
        args = _parse_args(["manifest", "state", "actor", "--advantage-method", "mc",
                           "--gamma", "1", "--return-scale", "50",
                           "--no-normalize-advantages", "--log-rollouts"])
        self.assertEqual((args.advantage_method, args.gamma, args.return_scale), ("mc", 1.0, 50.0))
        self.assertFalse(args.normalize_advantages)
        self.assertTrue(args.log_rollouts)
        rounds = _parse_args([
            "manifest", "state", "actor", "--curriculum-rounds", "20"
        ])
        self.assertEqual(rounds.curriculum_rounds, 20)

    def test_cli_rejects_invalid_numbers(self):
        for option, value in (("--gamma", "nan"), ("--gae-lambda", "1.1"),
                              ("--return-scale", "0"), ("--return-scale", "inf")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                _parse_args(["manifest", "state", "actor", option, value])
        with self.assertRaises(ValueError):
            _parse_args([
                "manifest", "state", "actor", "--curriculum-rounds", "0"
            ])

    def test_round_stage_plan_is_easy_medium_hard_for_every_round(self):
        plan = _round_stage_plan(20)
        self.assertEqual(len(plan), 60)
        self.assertEqual(plan[:3], [
            (1, 0, "easy"), (1, 1, "medium"), (1, 2, "hard")
        ])
        self.assertEqual(plan[-3:], [
            (20, 0, "easy"), (20, 1, "medium"), (20, 2, "hard")
        ])
        with self.assertRaises(ValueError):
            _round_stage_plan(0)

    def test_reward_split_uses_fault_micro_average(self):
        circuits = [
            {
                "name": "a",
                "circuit": "a.bench",
                "fault_map": "a.map",
                "training_faults": [
                    {"fault_id": "a1", "backtracks": 1, "backtrace_steps": 1},
                    {"fault_id": "a2", "backtracks": 1, "backtrace_steps": 1},
                ],
            },
            {
                "name": "b",
                "circuit": "b.bench",
                "fault_map": "b.map",
                "training_faults": [
                    {"fault_id": "b1", "backtracks": 1, "backtrace_steps": 1},
                ],
            },
        ]

        class Evaluator:
            def __init__(self, count, reward_sum):
                self.run_metrics = {
                    "fault_count": count,
                    "reward_sum": reward_sum,
                    "mean_reward": reward_sum / count,
                }

            def run(self, *args, **kwargs):
                count = self.run_metrics["fault_count"]
                return {
                    "episodes": count,
                    "detected": count,
                    "aborted": 0,
                    "redundant": 0,
                    "decisions": count,
                    "backtracks": 0,
                    "backtrace_steps": 0,
                }

        result, _ = _run_reward_split(
            circuits,
            {"a": Evaluator(2, 30.0), "b": Evaluator(1, 60.0)},
            "training",
            2026,
        )
        self.assertEqual(result["aggregate"]["fault_count"], 3)
        self.assertEqual(result["aggregate"]["reward_sum"], 90.0)
        self.assertEqual(result["aggregate"]["mean_reward"], 30.0)

    def test_zero_baseline_preserves_infinite_selection_penalty(self):
        circuits = [{
            "name": "a",
            "circuit": "a.bench",
            "fault_map": "a.map",
            "training_faults": [{
                "fault_id": "a1", "backtracks": 0, "backtrace_steps": 1,
            }],
        }]

        class Evaluator:
            run_metrics = {"fault_count": 1, "reward_sum": 99.0}

            def run(self, *args, **kwargs):
                return {"detected": 1, "aborted": 0, "decisions": 1,
                        "backtracks": 1, "backtrace_steps": 1}

        result, score = _run_reward_split(
            circuits,
            {"a": Evaluator()},
            "training",
            2026,
        )
        self.assertTrue(math.isinf(score[2]))
        self.assertIsNone(result["aggregate"]["mean_backtrack_ratio"])
        self.assertIsNone(result["aggregate"]["score"][2])
        self.assertNotIn("Infinity", json.dumps(result))

    def test_config_mismatch_rejected(self):
        state = dict(format=CHECKPOINT_FORMAT, manifest_hash="test", config={"gamma": 1})
        with self.assertRaisesRegex(ValueError, "new checkpoint path"):
            _validate_checkpoint_metadata(state, "test", {"gamma": 0.99})
        round_state = dict(
            format=ROUND_CHECKPOINT_FORMAT,
            manifest_hash="test",
            config={"curriculum_rounds": 20},
        )
        _validate_checkpoint_metadata(
            round_state, "test", {"curriculum_rounds": 20}
        )

    def test_rollout_lengths_include_zero_decision_faults(self):
        self.assertEqual(rollout_length_stats([0, 1, 3, 128, 350]), dict(
            count=5, min=0, mean=96.4, median=3, p90=350, max=350, zero_decision_faults=1))
        self.assertEqual(rollout_length_stats([])["count"], 0)
        self.assertEqual(rollout_length_stats([2, 4])["median"], 3)
        with self.assertRaises(ValueError):
            rollout_length_stats([-1])

    def test_trainer_aggregation_with_mixed_and_all_zero_rollouts(self):
        for lengths in ((0, 1, 2), (0, 0)):
            trainer = object.__new__(CppPodemCurriculumTrainer)
            trainer.agent = make_agent()
            trainer.baselines = {"fault": {"backtracks": 10, "backtrace_steps": 100}}
            trainer.reward_config = dict(REWARD_CONFIG)
            trainer.sequence_to_step = {}
            trainer.episode_metrics = []
            for length in lengths:
                trainer.event_callback({"event": "episode_start", "fault_id": "fault"})
                if length:
                    collect(trainer.agent, (0.0,) * length)
                trainer.event_callback(dict(event="episode_end", fault_id="fault",
                                            outcome=1, backtracks=0, backtrace_steps=0))
            trainer.run_metrics = dict(steps=sum(lengths), episodes=len(lengths))
            with patch.object(CppPodemBacktraceV2Trainer, "run", return_value={}):
                trainer.run()
            self.assertEqual(trainer.run_metrics["rollout_steps"], rollout_length_stats(lengths))
            self.assertEqual(trainer.run_metrics["episodes_with_updates"], sum(n > 0 for n in lengths))
            self.assertTrue(math.isfinite(trainer.run_metrics["value_loss_mean"]))
            if not any(lengths):
                self.assertEqual(trainer.run_metrics["value_target_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()

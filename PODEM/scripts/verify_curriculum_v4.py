import math
import re
import tempfile
from pathlib import Path

import torch

import cpp_podem
from rl_podem.cpp_bridge import profile_cpp_podem
from rl_podem.cpp_bridge import EmbeddingGate, export_actor_v2_state_dict
from rl_podem.curriculum import (
    DIFFICULTIES,
    REWARD_CONFIG,
    REWARD_DISTRIBUTION,
    CppPodemCurriculumTrainer,
    baseline_relative_potential,
    baseline_relative_reward,
    filter_unseen_teacher_samples,
    pretrain_actor,
    stratified_split,
    teacher_accuracy,
)
from rl_podem.ppo import BacktracePPOAgentV2
from train_curriculum import (
    CHECKPOINT_FORMAT,
    _stage_fault_ids,
    _validate_checkpoint_metadata,
    _validation_score,
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _verify_teacher_actions():
    expected = {
        "AND": {0: 0, 1: 1},
        "OR": {0: 1, 1: 0},
        "NAND": {0: 1, 1: 0},
        "NOR": {0: 0, 1: 1},
    }
    root = Path(__file__).resolve().parents[1]
    circuit = root / "sample_circuits" / "c432_binary.bench"
    fault_map = circuit.with_suffix(".faultmap")
    gate_types = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"^\s*([^#\s][^=\s]*)\s*=\s*(AND|OR|NAND|NOR)\(",
            circuit.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    }
    fault_ids = [
        item["fault_id"]
        for item in profile_cpp_podem(circuit, 500, 14, fault_map)[:100]
    ]
    observed = set()

    def teacher(request):
        gate_type = gate_types[request["objective_name"]]
        objective_value = int(request["objective_value"])
        action = int(request["heuristic_action"])
        _check(
            action == expected[gate_type][objective_value],
            f"Incorrect {gate_type}/{objective_value} teacher label {action}.",
        )
        observed.add((gate_type, objective_value, action))
        return action

    cpp_podem.run_stuck_at(
        str(circuit.relative_to(root)),
        teacher,
        None,
        500,
        14,
        fault_ids,
        True,
        "backtrace_rl",
        str(fault_map.relative_to(root)),
    )
    _check(len(observed) >= 5, "Teacher verification did not cover enough gate cases.")
    _check({item[2] for item in observed} == {0, 1}, "Teacher did not emit both actions.")


def _verify_stratification():
    profiles = [
        {
            "fault_id": f"f{index:03d}",
            "outcome": 1,
            "backtracks": index,
            "backtrace_steps": index * 10,
        }
        for index in range(200)
    ]
    train_counts = {name: 2 for name in DIFFICULTIES}
    validation_counts = {name: 1 for name in DIFFICULTIES}
    first = stratified_split(profiles, 2026, train_counts, validation_counts)
    second = stratified_split(profiles, 2026, train_counts, validation_counts)
    _check(first == second, "Curriculum split is not deterministic.")
    training, validation, available = first
    _check(available == {"easy": 80, "medium": 80, "hard": 40}, "Bad strata.")
    _check(len(training) == 6 and len(validation) == 3, "Bad split lengths.")
    _check(
        not ({item["fault_id"] for item in training} & {item["fault_id"] for item in validation}),
        "Curriculum fault splits overlap.",
    )


def _verify_reward():
    baseline = {"backtracks": 10, "backtrace_steps": 100}
    reward, parts = baseline_relative_reward(1, 5, 50, baseline)
    _check(math.isclose(reward, 115.0), "Improvement reward is incorrect.")
    _check(
        math.isclose(parts["backtrack_gain"], 0.5)
        and math.isclose(parts["backtrace_gain"], 0.5),
        "Normalized gains are incorrect.",
    )
    reward, _ = baseline_relative_reward(1, 100, 1000, baseline)
    _check(math.isclose(reward, 40.0), "Negative gain clipping is incorrect.")
    reward, _ = baseline_relative_reward(0, 0, 0, baseline)
    _check(math.isclose(reward, -70.0), "Detection priority bound is incorrect.")

    initial, _ = baseline_relative_potential(0, 0, baseline)
    after_trace, _ = baseline_relative_potential(0, 1, baseline)
    after_backtrack, _ = baseline_relative_potential(1, 1, baseline)
    _check(math.isclose(initial, 30.0), "Initial search potential is incorrect.")
    _check(
        math.isclose(after_trace - initial, -0.1),
        "Backtrace potential delta is incorrect.",
    )
    _check(
        math.isclose(after_backtrack - after_trace, -2.0),
        "Backtrack potential delta is incorrect.",
    )

    zero_backtrack = {"backtracks": 0, "backtrace_steps": 100}
    zero_initial, _ = baseline_relative_potential(0, 0, zero_backtrack)
    one_backtrack, _ = baseline_relative_potential(1, 0, zero_backtrack)
    two_backtracks, _ = baseline_relative_potential(2, 0, zero_backtrack)
    three_backtracks, _ = baseline_relative_potential(3, 0, zero_backtrack)
    _check(
        math.isclose(one_backtrack - zero_initial, -20.0)
        and math.isclose(two_backtracks - one_backtrack, -20.0)
        and math.isclose(three_backtracks - two_backtracks, 0.0),
        "Zero-baseline potential clipping is incorrect.",
    )
    _check(
        REWARD_DISTRIBUTION == "incremental_potential_v1",
        "Reward-distribution checkpoint version is incorrect.",
    )


def _verify_incremental_reward_attribution():
    torch.manual_seed(11)
    agent = BacktracePPOAgentV2(
        2,
        hidden_dim=8,
        gamma=1.0,
        k_epochs=1,
        rnd_beta=0.0,
        normalize_returns=False,
        return_scale=100.0,
    )
    trainer = object.__new__(CppPodemCurriculumTrainer)
    trainer.agent = agent
    trainer.baselines = {"fault": {"backtracks": 10, "backtrace_steps": 100}}
    trainer.reward_config = dict(REWARD_CONFIG)
    trainer.sequence_to_step = {}
    trainer.episode_metrics = []
    trainer.last_metrics = None
    trainer.event_callback({"event": "episode_start", "fault_id": "fault"})

    objective = EmbeddingGate("g", torch.tensor([1.0, 0.0]))
    candidates = [
        EmbeddingGate("a", torch.tensor([1.0, 0.0])),
        EmbeddingGate("b", torch.tensor([0.0, 1.0])),
    ]
    agent.select_backtrace_action(objective, 0, candidates)
    trainer.sequence_to_step[11] = agent.last_selected_step_idx
    agent.select_backtrace_action(objective, 1, candidates)
    trainer.sequence_to_step[22] = agent.last_selected_step_idx

    trainer.event_callback({"event": "backtrace_step", "decision_sequence": 11})
    trainer.event_callback({"event": "backtrack", "decision_sequence": 22})
    trainer.event_callback({"event": "backtrace_step", "decision_sequence": 999})
    rewards_before_terminal = [step.reward for step in agent.buffer.steps]
    _check(
        math.isclose(rewards_before_terminal[0], -0.1)
        and math.isclose(rewards_before_terminal[1], -2.0),
        "Potential deltas were not attributed to their PPO decisions.",
    )

    final_reward, _ = baseline_relative_reward(
        1, 2, 3, trainer.baselines["fault"], trainer.reward_config
    )
    trainer.event_callback(
        {
            "event": "episode_end",
            "fault_id": "fault",
            "outcome": 1,
            "backtracks": 2,
            "backtrace_steps": 3,
        }
    )
    metrics = trainer.episode_metrics[-1]
    _check(
        math.isclose(metrics["extrinsic_reward_sum"], final_reward, abs_tol=1e-9),
        "Incremental and terminal rewards changed the episode objective.",
    )
    _check(
        math.isclose(
            metrics["distributed_potential_reward"]
            + metrics["terminal_reward_residual"],
            final_reward,
            abs_tol=1e-9,
        ),
        "Terminal residual did not recover unattributed or missing events.",
    )
    _check(
        metrics["attributed_backtracks"] == 1
        and metrics["attributed_backtrace_steps"] == 1,
        "Attributed event counts are incorrect.",
    )
    _check(
        metrics["backtrack_event_mismatch"] == 1
        and metrics["backtrace_event_mismatch"] == 1,
        "Authoritative counter mismatches were not reported.",
    )
    _check(
        metrics["backtrack_event_mismatch_abs"] == 1
        and metrics["backtrace_event_mismatch_abs"] == 1
        and metrics["counter_mismatch_episode"] == 1,
        "Absolute mismatch auditing is incorrect.",
    )

    no_decision = object.__new__(CppPodemCurriculumTrainer)
    no_decision.agent = BacktracePPOAgentV2(2, hidden_dim=8, rnd_beta=0.0)
    no_decision.baselines = {"fault": {"backtracks": 0, "backtrace_steps": 10}}
    no_decision.reward_config = dict(REWARD_CONFIG)
    no_decision.sequence_to_step = {}
    no_decision.episode_metrics = []
    no_decision.last_metrics = None
    no_decision.event_callback({"event": "episode_start", "fault_id": "fault"})
    expected, _ = baseline_relative_reward(
        1, 0, 2, no_decision.baselines["fault"], no_decision.reward_config
    )
    no_decision.event_callback(
        {
            "event": "episode_end",
            "fault_id": "fault",
            "outcome": 1,
            "backtracks": 0,
            "backtrace_steps": 2,
        }
    )
    audit = no_decision.episode_metrics[-1]
    _check(
        not audit["updated"]
        and audit["steps"] == 0
        and math.isclose(audit["extrinsic_reward_sum"], expected),
        "No-decision episode was not audited without a PPO update.",
    )


def _verify_native_incremental_reward():
    root = Path(__file__).resolve().parents[1]
    circuit = root / "sample_circuits" / "c432_binary.bench"
    fault_map = circuit.with_suffix(".faultmap")
    embedding = root / "artifacts" / "v2_smoke" / "c432_binary.emb"
    profiles = profile_cpp_podem(circuit, 500, 14, fault_map)
    baseline = next(
        item
        for item in profiles
        if int(item["outcome"]) == 1 and int(item["backtracks"]) > 0
    )
    torch.manual_seed(29)
    trainer = CppPodemCurriculumTrainer(
        embedding, {str(baseline["fault_id"]): baseline}
    )
    trainer.agent.gamma = 1.0
    trainer.agent.k_epochs = 1
    trainer.agent.rnd_beta = 0.0
    trainer.agent.normalize_returns = False
    trainer.agent.return_scale = 100.0
    summary = trainer.run(
        circuit,
        backtrack_limit=500,
        seed=14,
        fault_ids=[str(baseline["fault_id"])],
        quiet=True,
        fault_map_path=fault_map,
    )
    expected, _ = baseline_relative_reward(
        1,
        int(summary["backtracks"]),
        int(summary["backtrace_steps"]),
        baseline,
    )
    metrics = trainer.episode_metrics[-1]
    _check(metrics["steps"] > 1, "Native reward test did not exercise PPO decisions.")
    _check(
        metrics["attributed_backtrace_steps"] > 0,
        "Native C++ backtrace events were not attributed.",
    )
    _check(
        metrics["backtrack_event_mismatch_abs"] == 0
        and metrics["backtrace_event_mismatch_abs"] == 0,
        "Native C++ event counters do not match authoritative totals.",
    )
    _check(
        math.isclose(
            metrics["reward_sum"] - metrics["scaled_intrinsic_reward_sum"],
            expected,
            abs_tol=1e-8,
        ),
        "Native PPO reward sum differs from baseline-relative reward.",
    )


def _verify_reward_checkpoint_versioning():
    config = {"reward_distribution": REWARD_DISTRIBUTION}
    state = {
        "format": CHECKPOINT_FORMAT,
        "manifest_hash": "manifest",
        "config": dict(config),
    }
    _validate_checkpoint_metadata(state, "manifest", config)
    old_state = dict(state)
    old_state["config"] = {}
    try:
        _validate_checkpoint_metadata(old_state, "manifest", config)
    except ValueError:
        pass
    else:
        raise AssertionError("Pre-potential checkpoint was accepted.")


def _teacher_sample(circuit, gate, value, action, difficulty):
    counts = [0, 0]
    counts[action] = 1
    difficulty_counts = {name: 0 for name in DIFFICULTIES}
    difficulty_counts[difficulty] = 1
    return {
        "circuit": circuit,
        "objective_name": gate,
        "objective_value": value,
        "action_counts": counts,
        "difficulty_counts": difficulty_counts,
    }


def _verify_pretraining():
    torch.manual_seed(7)
    tables = {
        "a": {"left": torch.tensor([1.0, 0.0]), "right": torch.tensor([0.0, 1.0])},
        "b": {"left": torch.tensor([1.0, 0.0]), "right": torch.tensor([0.0, 1.0])},
    }
    samples = []
    for circuit in tables:
        samples.extend(
            [
                _teacher_sample(circuit, "left", 0, 0, "easy"),
                _teacher_sample(circuit, "left", 1, 1, "medium"),
                _teacher_sample(circuit, "right", 0, 1, "medium"),
                _teacher_sample(circuit, "right", 1, 0, "hard"),
            ]
        )
    agent = BacktracePPOAgentV2(
        2,
        hidden_dim=8,
        gamma=1.0,
        normalize_returns=False,
        entropy_coef=0.001,
    )
    before = teacher_accuracy(agent.policy, samples, tables)["mean_per_circuit"]
    result = pretrain_actor(
        agent,
        samples,
        samples,
        tables,
        epochs=80,
        batch_size=8,
        learning_rate=0.02,
        seed=7,
    )
    after = teacher_accuracy(agent.policy_old, samples, tables)["mean_per_circuit"]
    _check(after > before and after == 1.0, "Behavior cloning did not learn teacher actions.")
    _check(
        result["history"][-1]["loss"] < result["history"][0]["loss"],
        "Behavior-cloning loss did not decrease.",
    )
    _check(agent.gamma == 1.0 and not agent.normalize_returns, "Bad V4 PPO returns.")
    with torch.no_grad():
        _, critic_value = agent.policy_old.backtrace_logits(
            tables["a"]["left"], 0
        )
    _check(
        math.isclose(float(critic_value.item()), 1.0, abs_tol=1e-6),
        "Behavior cloning did not initialize the detection-value baseline.",
    )


def _verify_unseen_validation_and_resume():
    training = [_teacher_sample("a", "g0", 0, 0, "easy")]
    validation = [
        _teacher_sample("a", "g0", 0, 0, "easy"),
        _teacher_sample("a", "g1", 1, 1, "medium"),
    ]
    unseen = filter_unseen_teacher_samples(training, validation)
    _check(len(unseen) == 1 and unseen[0]["objective_name"] == "g1", "BC leakage filter failed.")

    tables = {
        "a": {
            "g0": torch.tensor([1.0, 0.0]),
            "g1": torch.tensor([0.0, 1.0]),
        }
    }
    samples = [
        _teacher_sample("a", "g0", 0, 0, "easy"),
        _teacher_sample("a", "g0", 1, 1, "medium"),
        _teacher_sample("a", "g1", 0, 1, "medium"),
        _teacher_sample("a", "g1", 1, 0, "hard"),
    ]
    torch.manual_seed(19)
    full_agent = BacktracePPOAgentV2(2, hidden_dim=8)
    pretrain_actor(
        full_agent, samples, samples, tables, epochs=8, batch_size=2,
        learning_rate=0.01, seed=19,
    )

    captured = {}
    torch.manual_seed(19)
    partial_agent = BacktracePPOAgentV2(2, hidden_dim=8)
    pretrain_actor(
        partial_agent, samples, samples, tables, epochs=4, batch_size=2,
        learning_rate=0.01, seed=19,
        epoch_callback=lambda state: captured.update({"state": state}),
    )
    resumed_agent = BacktracePPOAgentV2(2, hidden_dim=8)
    pretrain_actor(
        resumed_agent, samples, samples, tables, epochs=8, batch_size=2,
        learning_rate=0.01, seed=19, resume_state=captured["state"],
    )
    for name, expected in full_agent.policy_old.state_dict().items():
        _check(
            torch.equal(expected.cpu(), resumed_agent.policy_old.state_dict()[name].cpu()),
            f"Behavior-cloning resume changed tensor {name}.",
        )


def _verify_scaled_ppo_and_parity():
    torch.manual_seed(23)
    agent = BacktracePPOAgentV2(
        4,
        hidden_dim=8,
        gamma=1.0,
        k_epochs=1,
        rnd_beta=0.0,
        normalize_returns=False,
        entropy_coef=0.001,
        return_scale=100.0,
        max_grad_norm=1.0,
    )
    objective = EmbeddingGate("g", torch.tensor([0.1, 0.2, 0.3, 0.4]))
    candidates = [EmbeddingGate("a", torch.zeros(4)), EmbeddingGate("b", torch.ones(4))]
    agent.select_backtrace_action(objective, 1, candidates)
    agent.finish_episode(100.0)
    metrics = agent.update()
    _check(math.isclose(metrics["return_mean"], 1.0), "V4 return scaling failed.")
    _check(all(math.isfinite(float(value)) for value in metrics.values()), "Non-finite PPO metric.")
    _check(metrics["rnd_loss"] == 0.0, "Final-stage RND should be disabled.")

    with tempfile.TemporaryDirectory(prefix="podem_v4_actor_") as directory:
        actor_path = Path(directory) / "actor.txt"
        export_actor_v2_state_dict(agent.policy_old.state_dict(), actor_path)
        python_logits, _ = agent.policy_old.backtrace_logits(
            objective.deepgate_embedding, 1
        )
        cpp_logits = cpp_podem.score_actor_v2(
            str(actor_path), objective.deepgate_embedding.tolist(), 1
        )
        _check(
            torch.allclose(python_logits.detach().cpu(), torch.tensor(cpp_logits), atol=1e-5),
            "Python/C++ V4 actor logits differ.",
        )


def _verify_curriculum_and_ranking():
    item = {
        "name": "c",
        "training_faults": [
            {"fault_id": f"{difficulty}{index}", "difficulty": difficulty}
            for difficulty in DIFFICULTIES
            for index in range(4)
        ],
        "validation_faults": [
            {"fault_id": "v0", "backtracks": 4, "backtrace_steps": 40},
            {"fault_id": "v1", "backtracks": 6, "backtrace_steps": 60},
        ],
    }
    selected = _stage_fault_ids(item, 2, 1, 2026)
    _check(
        set(selected) == set(DIFFICULTIES)
        and len({len(value) for value in selected.values()}) == 1,
        "Curriculum strata are not balanced.",
    )
    score, aggregate = _validation_score(
        [item],
        {"c": {"detected": 2, "aborted": 0, "decisions": 3, "backtracks": 5, "backtrace_steps": 50}},
    )
    _check(score[2:4] == (0.5, 0.5), "Validation ratios are not exact RL/baseline values.")
    _check(aggregate["detected"] == 2, "Validation aggregation failed.")


def main():
    _verify_teacher_actions()
    _verify_stratification()
    _verify_reward()
    _verify_incremental_reward_attribution()
    _verify_native_incremental_reward()
    _verify_reward_checkpoint_versioning()
    _verify_pretraining()
    _verify_unseen_validation_and_resume()
    _verify_scaled_ppo_and_parity()
    _verify_curriculum_and_ranking()
    print("PASS curriculum potential reward, resume, PPO stability, and parity")


if __name__ == "__main__":
    main()

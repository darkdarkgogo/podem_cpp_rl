import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from rl_podem.cpp_bridge import (
    CppPodemBacktraceV2Evaluator,
    CppPodemBacktraceV2Trainer,
    EmbeddingGate,
    export_actor_v2_state_dict,
    load_cpp_embedding_table,
    profile_cpp_podem,
    smartatpg_pi_reward,
)
from rl_podem.ppo import BacktracePPOAgentV2
from prepare_paper_training import _split_hard_faults
from train_paper_rnd import _validate_manifest, _validation_score


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expect_error(callback, text):
    try:
        callback()
    except (ValueError, FileNotFoundError) as error:
        _check(text in str(error), f"Unexpected error: {error}")
        return
    raise AssertionError(f"Expected an error containing: {text}")


def _states_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_states_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_states_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _run_command(command, root, expected_success=True):
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    if expected_success:
        _check(result.returncode == 0, result.stdout)
    else:
        _check(result.returncode != 0, "An incompatible checkpoint was accepted.")
    return result.stdout


class _RewardRecorder:
    def __init__(self):
        self.buffer = type("Buffer", (), {"steps": [None]})()
        self.rewards = []
        self.terminal_rewards = []
        self.rnd_beta = 0.05

    def add_reward_to_step(self, step_idx, reward):
        self.rewards.append((step_idx, reward))

    def finish_episode(self, reward):
        self.terminal_rewards.append(reward)

    def update(self):
        return None


def _reward_trainer():
    trainer = object.__new__(CppPodemBacktraceV2Trainer)
    trainer.agent = _RewardRecorder()
    trainer.sequence_to_step = {7: 0}
    trainer.reward_alpha = 7.5
    trainer.reward_beta = 0.07
    trainer.non_pi_reward = -0.1
    trainer.detected_reward = 100.0
    trainer.undetected_reward = -100.0
    trainer.episode_metrics = []
    trainer._episode_extrinsic_reward = 0.0
    trainer.last_metrics = None
    return trainer


def _verify_reward_and_agent_state():
    expected = 10.0 - 7.5 * math.exp(0.07 * 4)
    _check(
        math.isclose(smartatpg_pi_reward(2, 2), expected, rel_tol=1e-12),
        "Paper PI reward differs from the SmartATPG formula.",
    )
    _expect_error(lambda: smartatpg_pi_reward(-1, 1), "out of range")

    trainer = _reward_trainer()
    trainer.event_callback({"event": "backtrace_step", "decision_sequence": 7})
    _check(trainer.agent.rewards == [(0, -0.1)], "Non-PI reward is not -0.1.")
    trainer.event_callback(
        {
            "event": "pi_not_done",
            "decision_sequence": 7,
            "backtracks": 2,
            "pi_visits": 2,
        }
    )
    _check(
        math.isclose(trainer.agent.rewards[-1][1], expected, rel_tol=1e-12),
        "Non-terminal PI reward was not attached to the selected action.",
    )
    for outcome, expected_terminal in ((1, 100.0), (0, -100.0), (2, -100.0)):
        terminal_trainer = _reward_trainer()
        terminal_trainer.event_callback({"event": "episode_end", "outcome": outcome})
        _check(
            terminal_trainer.agent.terminal_rewards == [expected_terminal],
            f"Incorrect terminal reward for outcome {outcome}.",
        )

    torch.manual_seed(7)
    agent = BacktracePPOAgentV2(gate_embedding_dim=4, hidden_dim=8)
    state = agent.training_state_dict()
    restored = BacktracePPOAgentV2(gate_embedding_dim=4, hidden_dim=8)
    restored.load_training_state_dict(state)
    changed = BacktracePPOAgentV2(
        gate_embedding_dim=4, hidden_dim=8, rnd_bonus_clip=4.0
    )
    _expect_error(
        lambda: changed.load_training_state_dict(state), "hyperparameters changed"
    )

    objective = EmbeddingGate("objective", torch.zeros(4))
    candidates = [
        EmbeddingGate("left", torch.zeros(4)),
        EmbeddingGate("right", torch.ones(4)),
    ]
    agent.select_backtrace_action(objective, 0, candidates)
    agent.add_reward(1e300)
    agent.finish_episode(0.0)
    metrics = agent.update()
    _check(metrics is not None, "The finite-reward update did not run.")
    for key in ("total_loss", "return_mean", "adv_mean", "rnd_loss"):
        _check(math.isfinite(metrics[key]), f"Non-finite PPO metric: {key}")

    profiles = [
        {
            "fault_id": f"fault_{index:03d}",
            "outcome": 1,
            "backtracks": index,
            "backtrace_steps": 300 - index,
        }
        for index in range(200)
    ]
    first = _split_hard_faults(profiles, 100, 50, 2026, "synthetic")
    second = _split_hard_faults(profiles, 100, 50, 2026, "synthetic")
    first_ids = [[item["fault_id"] for item in part] for part in first[:2]]
    second_ids = [[item["fault_id"] for item in part] for part in second[:2]]
    _check(first_ids == second_ids, "The 100/50 split is not deterministic.")
    _check(
        not set(first_ids[0]) & set(first_ids[1]),
        "The 100/50 training and validation splits overlap.",
    )


def _verify_manifest(root, circuit, fault_map, embeddings, fault_id):
    source = root / "sample_circuits" / "c432.bench"
    hashes = {
        "source_circuit": _sha256(source),
        "circuit": _sha256(circuit),
        "fault_map": _sha256(fault_map),
        "embeddings": _sha256(embeddings),
    }
    manifest = {
        "format": "RL_PODEM_PAPER_TRAINING_V3",
        "training_fault_count_per_circuit": 1,
        "validation_fault_count_per_circuit": 1,
        "backtrack_limits": {"profile": 500, "training": 500, "validation": 500},
        "circuits": [
            {
                "name": "c432",
                "source_circuit": str(source),
                "circuit": str(circuit),
                "fault_map": str(fault_map),
                "embeddings": str(embeddings),
                "artifact_sha256": hashes,
                "training_fault_ids": [fault_id],
                "validation_fault_ids": [fault_id + "_validation"],
            }
        ],
    }
    _check(_validate_manifest(manifest, 500), "A valid V3 manifest was rejected.")
    changed_manifest = dict(manifest)
    changed_circuit = dict(manifest["circuits"][0])
    changed_hashes = dict(hashes)
    changed_hashes["embeddings"] = "0" * 64
    changed_circuit["artifact_sha256"] = changed_hashes
    changed_manifest["circuits"] = [changed_circuit]
    _expect_error(
        lambda: _validate_manifest(changed_manifest, 500),
        "content changed since profiling",
    )
    _expect_error(lambda: _validate_manifest(manifest, 97), "requires")


def _verify_pi_count_and_terminal_boundary(root, cpp_podem):
    verify_dir = root / "artifacts" / "v3_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    synthetic = verify_dir / "two_input_pi_count.bench"
    synthetic.write_text(
        "INPUT(a)\nINPUT(b)\nOUTPUT(z)\nz = AND(a, b)\n", encoding="ascii"
    )
    summary = dict(
        cpp_podem.run_stuck_at(
            str(synthetic.relative_to(root)),
            lambda request: 0,
            lambda event: None,
            97,
            14,
            ["z:GO:sa0"],
            True,
            "backtrace_rl",
            "",
        )
    )
    _check(summary["detected"] == 1, "Synthetic two-input fault was not detected.")
    _check(
        summary["pi_visits"] == 2,
        "A two-input unique implication did not count both simulated PI assignments.",
    )

    events = []
    aborted = dict(
        cpp_podem.run_stuck_at(
            "sample_circuits/c432_binary.bench",
            lambda request: 0,
            events.append,
            1,
            14,
            ["dummy_gate24:GO:sa0"],
            True,
            "backtrace_rl",
            "sample_circuits/c432_binary.faultmap",
        )
    )
    _check(aborted["aborted"] == 1, "Boundary fault did not hit the limit.")
    pi_events = [event for event in events if event["event"] == "pi_not_done"]
    _check(pi_events, "Boundary fault produced no intermediate PI events.")
    _check(
        all(int(event["backtracks"]) < 1 for event in pi_events),
        "The limit-reaching PI assignment received a non-terminal reward.",
    )
    _check(
        int(pi_events[-1]["pi_visits"]) < int(aborted["pi_visits"]),
        "The terminal PI assignment was not counted in the episode summary.",
    )


def _verify_cpp_bridge(root):
    import cpp_podem

    _verify_pi_count_and_terminal_boundary(root, cpp_podem)

    circuit = root / "sample_circuits" / "c432_binary.bench"
    fault_map = root / "sample_circuits" / "c432_binary.faultmap"
    embeddings = root / "artifacts" / "v2_smoke" / "c432_binary.emb"
    for path in (circuit, fault_map, embeddings):
        _check(path.is_file(), f"Missing verification artifact: {path}")

    detected_profiles = [
        item
        for item in profile_cpp_podem(
            circuit, backtrack_limit=500, seed=14, fault_map_path=fault_map
        )
        if int(item["outcome"]) == 1
    ]
    _check(detected_profiles, "c432 has no detected fault for event verification.")
    fault_id = detected_profiles[0]["fault_id"]
    events = []

    def decision(request):
        return 0

    summary = dict(
        cpp_podem.run_stuck_at(
            str(circuit.relative_to(root)),
            decision,
            events.append,
            500,
            14,
            [fault_id],
            True,
            "backtrace_rl",
            str(fault_map.relative_to(root)),
        )
    )
    _check(summary["detected"] == 1, "Selected c432 fault was not detected.")
    step_events = sum(event["event"] == "backtrace_step" for event in events)
    pi_events = sum(event["event"] == "pi_not_done" for event in events)
    _check(
        step_events == summary["backtrace_steps"],
        "C++ backtrace-step events disagree with the authoritative summary.",
    )
    _check(
        pi_events + 1 == summary["pi_visits"],
        "A detected terminal PI visit received duplicate non-terminal reward.",
    )

    table = load_cpp_embedding_table(embeddings)
    embedding_dim = next(iter(table.values())).numel()
    torch.manual_seed(11)
    agent = BacktracePPOAgentV2(gate_embedding_dim=embedding_dim, hidden_dim=32)
    evaluator = CppPodemBacktraceV2Evaluator(embeddings, agent=agent)
    policy_before = {
        key: value.detach().cpu().clone()
        for key, value in agent.policy_old.state_dict().items()
    }
    rnd_before = agent.rnd_error_stats.state_dict()
    agent_before = copy.deepcopy(agent.training_state_dict())
    torch_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    first_validation = evaluator.run(
        circuit,
        backtrack_limit=500,
        seed=14,
        fault_ids=[fault_id],
        quiet=True,
        fault_map_path=fault_map,
    )
    second_validation = evaluator.run(
        circuit,
        backtrack_limit=500,
        seed=14,
        fault_ids=[fault_id],
        quiet=True,
        fault_map_path=fault_map,
    )
    _check(first_validation == second_validation, "Validation is not deterministic.")
    _check(not agent.buffer.steps, "Validation populated the PPO rollout buffer.")
    _check(
        rnd_before == agent.rnd_error_stats.state_dict(),
        "Validation mutated RND running statistics.",
    )
    for key, value in agent.policy_old.state_dict().items():
        _check(torch.equal(policy_before[key], value.detach().cpu()), "Policy mutated")
    _check(
        _states_equal(agent_before, agent.training_state_dict()),
        "Validation mutated PPO, RND, optimizer, or running-stat state.",
    )
    _check(torch.equal(torch_rng_before, torch.get_rng_state()), "CPU RNG mutated.")
    if cuda_rng_before is not None:
        _check(
            all(
                torch.equal(before, after)
                for before, after in zip(cuda_rng_before, torch.cuda.get_rng_state_all())
            ),
            "CUDA RNG mutated during validation.",
        )

    trainer_agent = BacktracePPOAgentV2(
        gate_embedding_dim=embedding_dim, hidden_dim=32
    )
    trainer = CppPodemBacktraceV2Trainer(embeddings, agent=trainer_agent)
    trainer.run(
        circuit,
        backtrack_limit=500,
        seed=14,
        fault_ids=[fault_id],
        quiet=True,
        fault_map_path=fault_map,
    )
    metrics = trainer.run_metrics
    expected_total = (
        metrics["extrinsic_reward_sum"] + metrics["scaled_intrinsic_reward_sum"]
    )
    _check(
        math.isclose(metrics["combined_reward_sum"], expected_total, abs_tol=1e-5),
        "Applied external and RND rewards do not add up to rollout rewards.",
    )
    _check("ratio_mean" in metrics, "Aggregated PPO ratio metric is missing.")

    actor_path = root / "artifacts" / "v3_verify" / "actor_v2.txt"
    export_actor_v2_state_dict(agent.policy_old.state_dict(), actor_path)
    parity_cases = []
    for objective in list(table.values())[:8]:
        for objective_value in (0, 1):
            parity_cases.append((objective, objective_value))
    for objective, objective_value in parity_cases:
        with torch.no_grad():
            python_logits, _ = agent.policy_old.backtrace_logits(
                objective, objective_value
            )
        cpp_logits = cpp_podem.score_actor_v2(
            str(actor_path.relative_to(root)), objective.tolist(), objective_value
        )
        error = max(
            abs(float(left) - float(right))
            for left, right in zip(python_logits.detach().cpu(), cpp_logits)
        )
        _check(error < 1e-5, f"Python/C++ V2 actor mismatch: {error}")

    saved_actor = root / "artifacts" / "v3_verify" / "trainer_save_v2.txt"
    trainer.save(saved_actor)
    _check(saved_actor.read_text(encoding="utf-8").startswith("SMARTATPG_ACTOR_V2"),
           "V2Trainer.save() did not export a V2 actor.")

    _verify_manifest(root, circuit, fault_map, embeddings, fault_id)
    score = _validation_score(
        {
            "detected": 2,
            "aborted": 1,
            "backtracks": 3,
            "backtrace_steps": 4,
            "decisions": 5,
        }
    )
    _check(score == (-2, 1, 3, 4, 5), "Best-model ordering changed.")


def _write_training_manifest(path, root, circuits):
    source = root / "sample_circuits" / "c432.bench"
    circuit = root / "sample_circuits" / "c432_binary.bench"
    fault_map = root / "sample_circuits" / "c432_binary.faultmap"
    embeddings = root / "artifacts" / "v2_smoke" / "c432_binary.emb"
    artifact_hashes = {
        "source_circuit": _sha256(source),
        "circuit": _sha256(circuit),
        "fault_map": _sha256(fault_map),
        "embeddings": _sha256(embeddings),
    }
    entries = []
    for name, training_faults, validation_faults in circuits:
        entries.append(
            {
                "name": name,
                "source_circuit": str(source),
                "circuit": str(circuit),
                "fault_map": str(fault_map),
                "embeddings": str(embeddings),
                "artifact_sha256": artifact_hashes,
                "training_fault_ids": training_faults,
                "validation_fault_ids": validation_faults,
            }
        )
    manifest = {
        "format": "RL_PODEM_PAPER_TRAINING_V3",
        "training_fault_count_per_circuit": len(circuits[0][1]),
        "validation_fault_count_per_circuit": len(circuits[0][2]),
        "backtrack_limits": {"profile": 500, "training": 500, "validation": 500},
        "profile_seed": 14,
        "split_seed": 2026,
        "circuits": entries,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _training_command(root, manifest, state, best, latest, sweeps):
    return [
        sys.executable,
        "-u",
        str(root / "scripts" / "train_paper_rnd.py"),
        str(manifest),
        str(state),
        str(best),
        "--latest-actor-output",
        str(latest),
        "--sweeps",
        str(sweeps),
        "--seed",
        "2026",
        "--rnd-beta",
        "0.05",
        "--backtrack-limit",
        "500",
    ]


def _unlink_outputs(*paths):
    for path in paths:
        if path.exists():
            path.unlink()


def _verify_training_workflow(root):
    verify_dir = root / "artifacts" / "v3_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify_dir / "workflow_manifest.json"
    state_path = verify_dir / "workflow_state.pth"
    best_actor = verify_dir / "workflow_best.txt"
    latest_actor = verify_dir / "workflow_latest.txt"
    _unlink_outputs(state_path, best_actor, latest_actor)
    _write_training_manifest(
        manifest,
        root,
        [
            (
                "c432",
                ["G199:GO:sa1", "G426:GO:sa0"],
                ["G430:GI1:sa1", "G418:GO:sa0"],
            )
        ],
    )
    command = _training_command(
        root, manifest, state_path, best_actor, latest_actor, 2
    )
    _run_command(command, root)
    state = torch.load(state_path, map_location="cpu")
    _check(len(state["progress"]) == 2, "Two-sweep progress was not persisted.")
    _check(
        len(state["validation_history"]) == 3,
        "Initial and per-sweep validation were not persisted.",
    )
    scores = [tuple(item["score"]) for item in state["validation_history"]]
    _check(tuple(state["best_score"]) == min(scores), "Best score is incorrect.")
    _check(
        scores[-1] > tuple(state["best_score"]),
        "Smoke seed no longer demonstrates a later degraded policy.",
    )
    _check(
        _sha256(best_actor) != _sha256(latest_actor),
        "A degraded latest policy overwrote the best actor.",
    )

    original_best = _sha256(best_actor)
    original_latest = _sha256(latest_actor)
    best_actor.unlink()
    latest_actor.unlink()
    _run_command(command, root)
    resumed = torch.load(state_path, map_location="cpu")
    _check(len(resumed["progress"]) == 2, "Resume repeated completed training.")
    _check(_sha256(best_actor) == original_best, "Best actor regeneration changed.")
    _check(
        _sha256(latest_actor) == original_latest,
        "Latest actor regeneration changed.",
    )

    changed_plan = _training_command(
        root, manifest, state_path, best_actor, latest_actor, 3
    )
    changed_output = _run_command(changed_plan, root, expected_success=False)
    _check(
        "training configuration changed" in changed_output,
        "Changing planned sweeps was not diagnosed.",
    )

    v2_state = verify_dir / "incompatible_v2.pth"
    v2_best = verify_dir / "incompatible_best.txt"
    v2_latest = verify_dir / "incompatible_latest.txt"
    _unlink_outputs(v2_state, v2_best, v2_latest)
    torch.save({"format": "RL_PODEM_PAPER_RND_TRAINING_V2"}, v2_state)
    v2_output = _run_command(
        _training_command(root, manifest, v2_state, v2_best, v2_latest, 2),
        root,
        expected_success=False,
    )
    _check("incompatible V1/V2" in v2_output, "V2 rejection was not diagnosed.")

    half_manifest = verify_dir / "half_sweep_manifest.json"
    half_state = verify_dir / "half_sweep_state.pth"
    half_best = verify_dir / "half_sweep_best.txt"
    half_latest = verify_dir / "half_sweep_latest.txt"
    _unlink_outputs(half_state, half_best, half_latest)
    _write_training_manifest(
        half_manifest,
        root,
        [
            ("c432_a", ["G199:GO:sa1"], ["G430:GI1:sa1"]),
            ("c432_b", ["G426:GO:sa0"], ["G418:GO:sa0"]),
        ],
    )
    half_command = _training_command(
        root, half_manifest, half_state, half_best, half_latest, 1
    )
    process = subprocess.Popen(
        half_command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    saw_first_result = False
    try:
        for line in process.stdout:
            if line.startswith("TRAIN_RESULT"):
                saw_first_result = True
                process.terminate()
                break
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
    _check(saw_first_result, "Could not interrupt after the first circuit run.")
    interrupted = torch.load(half_state, map_location="cpu")
    _check(
        len(interrupted["progress"]) == 1,
        "Interrupted checkpoint did not contain exactly one circuit run.",
    )
    _check(
        [item["sweep"] for item in interrupted["validation_history"]] == [0],
        "A half-completed sweep was incorrectly validated.",
    )
    _run_command(half_command, root)
    completed = torch.load(half_state, map_location="cpu")
    progress_keys = [
        (item["sweep"], item["circuit"]) for item in completed["progress"]
    ]
    _check(
        len(progress_keys) == 2 and len(set(progress_keys)) == 2,
        "Half-sweep resume duplicated or skipped a circuit run.",
    )
    _check(
        [item["sweep"] for item in completed["validation_history"]] == [0, 1],
        "Resumed sweep was not validated exactly once after completion.",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    _verify_reward_and_agent_state()
    _verify_cpp_bridge(root)
    _verify_training_workflow(root)
    print("PASS paper V3 reward, checkpoint, manifest, events, validation, and parity")


if __name__ == "__main__":
    main()

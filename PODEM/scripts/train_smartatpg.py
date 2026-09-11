"""Train a selected 12D SCOAP SmartATPG graph policy without BC or curriculum."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from prepare_smartatpg_training import (
    BACKTRACK_LIMIT, FAULT_FILTER, HEURISTIC, MANIFEST_FORMAT,
    MAX_REINFORCEMENT_ROUNDS, NORMAL_TRAINING_ROUNDS, SELECTION,
    select_hard_faults,
    sha256_file,
)
from rl_podem.backends import smartatpg_metadata
from rl_podem.cpp_bridge import (
    CppPodemBacktraceV2Evaluator, CppPodemBacktraceV2Trainer,
    smartatpg_pi_reward,
)
from rl_podem.ppo import device
from rl_podem.gat_gru import GATGRUSmartATPGPPOAgent
from rl_podem.smartatpg_artifacts import export_actor
from rl_podem.smartatpg_features import load_circuit_graph
from smartatpg_portable import CIRCUITS


CHECKPOINT_FORMAT = "SMARTATPG_12D_CO_TRAINING_V4"
BEST_CHECKPOINT_FORMAT = "SMARTATPG_12D_CO_BEST_V4"
REINFORCEMENT_CHECKPOINT_FORMAT = "SMARTATPG_GAT_REINFORCEMENT_V2"
REINFORCEMENT_BEST_FORMAT = "SMARTATPG_GAT_REINFORCEMENT_BEST_V2"
FAULTS_PER_CIRCUIT = 50
AGENT_TYPES = {
    "level_gat_gru": GATGRUSmartATPGPPOAgent,
}
PAPER_REWARD = {
    "non_pi": -0.1,
    "alpha": 7.5,
    "beta": 0.07,
    "detected": 100.0,
    "undetected": -100.0,
}
TRAINING_PROTOCOL_KEYS = (
    "heuristic", "circuit_order", "faults_per_circuit", "normal_rounds",
    "reinforcement_rounds",
)


def _training_protocol(config):
    return {key: config[key] for key in TRAINING_PROTOCOL_KEYS}


def _manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validation_score(summary, round_number):
    return (
        -int(summary["detected_faults"]),
        int(summary["backtracks_total"]),
        int(summary["backtrace_steps_total"]),
        -float(summary["return_total"]),
        int(round_number),
    )


def _validate_manifest(manifest):
    expected = {
        "format": MANIFEST_FORMAT, "fault_filter": FAULT_FILTER,
        **smartatpg_metadata(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest is not compatible with detected-only SmartATPG training; prepare a new fault set")
    if int(manifest.get("backtrack_limit", -1)) != BACKTRACK_LIMIT:
        raise ValueError(
            f"SmartATPG training requires a {BACKTRACK_LIMIT}-backtrack manifest"
        )
    circuits = list(manifest.get("circuits", []))
    if [item.get("name") for item in circuits] != list(CIRCUITS):
        raise ValueError("Training requires all 16 benchmark circuits")
    count = int(manifest.get("fault_count_per_circuit", -1))
    if count != FAULTS_PER_CIRCUIT:
        raise ValueError(
            f"Training requires exactly {FAULTS_PER_CIRCUIT} faults per circuit"
        )
    if manifest.get("heuristic") != HEURISTIC:
        raise ValueError("Training requires a SCOAP heuristic manifest")
    if manifest.get("circuit_order") != list(CIRCUITS):
        raise ValueError("Training manifest circuit order metadata is invalid")
    if manifest.get("selection") != SELECTION:
        raise ValueError("Training manifest fault ranking metadata is invalid")
    if int(manifest.get("normal_rounds", -1)) != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"Training manifest must specify {NORMAL_TRAINING_ROUNDS} normal rounds"
        )
    reinforcement_rounds = int(manifest.get("reinforcement_rounds", -1))
    if not 0 <= reinforcement_rounds <= MAX_REINFORCEMENT_ROUNDS:
        raise ValueError("Training manifest reinforcement rounds are invalid")
    for item in circuits:
        if len(item.get("training_fault_ids", [])) != count:
            raise ValueError(
                f"Circuit {item['name']} must contain {FAULTS_PER_CIRCUIT} fault IDs"
            )
        required_artifacts = {"source_circuit", "circuit", "fault_map", "profile"}
        if item["name"].startswith("s"):
            required_artifacts.add("scan_circuit")
        if set(item.get("artifact_sha256", {})) != required_artifacts:
            raise ValueError(f"Circuit {item['name']} artifact list is incomplete")
        for key, expected_hash in item["artifact_sha256"].items():
            path = Path(item[key])
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Manifest artifact changed: {path}")
        profiles = json.loads(Path(item["profile"]).read_text(encoding="utf-8"))
        selected = select_hard_faults(profiles, count)
        expected_ids = [row["fault_id"] for row in selected]
        if item["training_fault_ids"] != expected_ids:
            raise ValueError(
                f"Circuit {item['name']} faults are not the baseline detected "
                f"top {FAULTS_PER_CIRCUIT}"
            )
        if item.get("training_faults") != selected:
            raise ValueError(f"Circuit {item['name']} ranking metadata changed")
    fault_keys = [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["training_fault_ids"]
    ]
    expected_faults = len(CIRCUITS) * FAULTS_PER_CIRCUIT
    if len(fault_keys) != expected_faults or len(set(fault_keys)) != expected_faults:
        raise ValueError(
            f"Training manifest must contain exactly {expected_faults} unique faults"
        )
    return circuits


def _episode_order(circuits, seed, round_number):
    episodes = [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["training_fault_ids"]
    ]
    random.Random(seed + round_number).shuffle(episodes)
    return episodes


def unresolved_faults(evaluation):
    return [
        (circuit["circuit"], episode["fault_id"])
        for circuit in evaluation["circuits"]
        for episode in circuit["episodes"]
        if not int(episode["detected"])
    ]


def _training_fault_set(circuits):
    return {
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["training_fault_ids"]
    }


def _validate_reinforcement_evaluation(evaluation, circuits):
    expected = _training_fault_set(circuits)
    evaluated = [
        (circuit["circuit"], episode["fault_id"])
        for circuit in evaluation["circuits"]
        for episode in circuit["episodes"]
    ]
    if len(evaluated) != len(expected) or set(evaluated) != expected:
        raise ValueError(
            "GAT reinforcement evaluation must cover every training fault exactly once"
        )
    return unresolved_faults(evaluation)


def _validate_reinforcement_faults(faults, circuits):
    expected = _training_fault_set(circuits)
    normalized = []
    for fault in faults:
        if not isinstance(fault, (list, tuple)) or len(fault) != 2:
            raise ValueError("Invalid fault entry in GAT reinforcement checkpoint")
        normalized.append(tuple(fault))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate fault in GAT reinforcement checkpoint")
    if not set(normalized).issubset(expected):
        raise ValueError(
            "GAT reinforcement checkpoint contains a non-training fault"
        )
    return normalized


def _reinforcement_order(faults, seed, round_number):
    episodes = [tuple(fault) for fault in faults]
    random.Random(seed + round_number).shuffle(episodes)
    return episodes


def _evaluate_fault(evaluator, item, fault_id, backtrack_limit, seed):
    extrinsic_return = 0.0

    def event_callback(event):
        nonlocal extrinsic_return
        if event["event"] == "backtrace_step":
            extrinsic_return += PAPER_REWARD["non_pi"]
        elif event["event"] == "pi_not_done":
            extrinsic_return += smartatpg_pi_reward(
                int(event["backtracks"]),
                int(event["pi_visits"]),
                PAPER_REWARD["alpha"],
                PAPER_REWARD["beta"],
            )
        elif event["event"] == "episode_end":
            extrinsic_return += (
                PAPER_REWARD["detected"]
                if int(event["outcome"]) == 1
                else PAPER_REWARD["undetected"]
            )

    summary = evaluator.run(
        item["circuit"],
        backtrack_limit=backtrack_limit,
        seed=seed,
        fault_ids=[fault_id],
        fault_map_path=item["fault_map"],
        use_scoap=True,
        event_callback=event_callback,
    )
    return {
        "fault_id": fault_id,
        "detected": int(summary["detected"]),
        "backtracks": int(summary["backtracks"]),
        "backtrace_steps": int(summary["backtrace_steps"]),
        "return": float(extrinsic_return),
    }


def evaluate_round(circuits, evaluators, round_number, backtrack_limit, seed):
    circuit_records = []
    totals = {
        "episodes": 0,
        "detected_faults": 0,
        "backtracks_total": 0,
        "backtrace_steps_total": 0,
        "return_total": 0.0,
    }
    for item in circuits:
        records = [
            _evaluate_fault(
                evaluators[item["name"]], item, fault_id, backtrack_limit, seed
            )
            for fault_id in item["training_fault_ids"]
        ]
        circuit_records.append({"circuit": item["name"], "episodes": records})
        totals["episodes"] += len(records)
        totals["detected_faults"] += sum(row["detected"] for row in records)
        totals["backtracks_total"] += sum(row["backtracks"] for row in records)
        totals["backtrace_steps_total"] += sum(
            row["backtrace_steps"] for row in records
        )
        totals["return_total"] += sum(row["return"] for row in records)
    count = max(1, totals["episodes"])
    totals.update({
        "fault_coverage": totals["detected_faults"] / count,
        "backtracks_mean": totals["backtracks_total"] / count,
        "backtrace_steps_mean": totals["backtrace_steps_total"] / count,
        "return_mean": totals["return_total"] / count,
    })
    return {
        "round": round_number,
        **totals,
        "circuits": circuit_records,
    }


def _save_state(path, agent, state):
    payload = dict(state)
    payload["agent"] = agent.training_state_dict()
    payload["torch_random_state"] = torch.get_rng_state()
    payload["torch_cuda_random_state"] = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    _atomic_torch_save(path, payload)


def _validate_resume_config(saved, current):
    if dict(saved or {}) != dict(current):
        raise ValueError("Training configuration changed since checkpoint")
    return int(current["rounds"])


def _validate_round_target(current_round, episode_index, target_rounds):
    if current_round <= 0 or episode_index < 0:
        raise ValueError("Checkpoint training position is invalid")
    last_started_round = current_round if episode_index else current_round - 1
    if last_started_round > target_rounds:
        raise ValueError(
            f"Checkpoint has already started round {last_started_round}; "
            f"cannot reduce the target to {target_rounds} rounds"
        )


def _validate_reinforcement_state(state, normal_rounds):
    required = {
        "current_round", "episode_index", "completed_episodes",
        "current_unresolved", "round_metrics", "best_score", "best_round",
        "best_validation_round", "best_agent", "best_evaluation", "agent",
        "torch_random_state", "torch_cuda_random_state",
    }
    if not required.issubset(state):
        raise ValueError("GAT reinforcement checkpoint is incomplete")
    current_round = int(state["current_round"])
    best_round = int(state["best_round"])
    round_metrics = state["round_metrics"]
    if not isinstance(round_metrics, list) or len(round_metrics) != current_round:
        raise ValueError("GAT reinforcement checkpoint metrics are inconsistent")
    if not 0 <= best_round < current_round:
        raise ValueError("GAT reinforcement checkpoint best round is invalid")
    if round_metrics[best_round] != state["best_evaluation"]:
        raise ValueError("GAT reinforcement checkpoint best evaluation is inconsistent")
    validation_round = int(state["best_validation_round"])
    if best_round and validation_round != normal_rounds + best_round:
        raise ValueError("GAT reinforcement checkpoint validation round is inconsistent")
    expected_score = validation_score(state["best_evaluation"], validation_round)
    if tuple(state["best_score"]) != expected_score:
        raise ValueError("GAT reinforcement checkpoint best score is inconsistent")
    if not isinstance(state["best_agent"], dict) or not isinstance(state["agent"], dict):
        raise ValueError("GAT reinforcement checkpoint agent state is invalid")


def _save_reinforcement_best(
    path, model_path, state, manifest_digest, config, evaluation,
):
    validation_round = int(state["best_validation_round"])
    payload = {
        "format": REINFORCEMENT_BEST_FORMAT,
        "manifest_hash": manifest_digest,
        "source_best_checkpoint_hash": state["source_best_checkpoint_hash"],
        "config": config,
        "round": validation_round,
        "reinforcement_round": int(state["best_round"]),
        "score": list(state["best_score"]),
        "evaluation": evaluation,
        "agent": state["best_agent"],
    }
    _atomic_torch_save(path, payload)
    export_actor(
        state["best_agent"]["policy_old"], model_path,
        best_round=validation_round, best_score=state["best_score"],
        training_protocol=_training_protocol(config),
    )


def _run_gat_reinforcement(
    *, agent, circuits, trainers, evaluators, output_dir, best_checkpoint_path,
    manifest_digest, normal_rounds, reinforcement_rounds, backtrack_limit, seed,
    writer,
):
    if normal_rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"GAT reinforcement requires exactly {NORMAL_TRAINING_ROUNDS} "
            "normal training rounds"
        )
    if not 1 <= reinforcement_rounds <= MAX_REINFORCEMENT_ROUNDS:
        raise ValueError(
            f"GAT reinforcement rounds must be between 1 and "
            f"{MAX_REINFORCEMENT_ROUNDS}"
        )
    checkpoint_path = output_dir / "reinforcement_state.pth"
    best_path = output_dir / "best_reinforced_training_state.pth"
    model_path = output_dir / "model_best_reinforced.txt"
    metrics_path = output_dir / "reinforcement_metrics.json"
    unresolved_path = output_dir / "reinforcement_unresolved_faults.json"
    source_hash = _manifest_hash(best_checkpoint_path)
    config = {
        "rounds": reinforcement_rounds,
        "seed": seed,
        "normal_rounds": normal_rounds,
        "backtrack_limit": backtrack_limit,
        "encoder_variant": "level_gat_gru",
        "device": str(device),
        "faults_per_episode": 1,
        "training_scope": "current_unresolved_training_faults",
        "heuristic": HEURISTIC,
        "circuit_order": [item["name"] for item in circuits],
        "faults_per_circuit": FAULTS_PER_CIRCUIT,
        "reinforcement_rounds": reinforcement_rounds,
    }

    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state.get("format") != REINFORCEMENT_CHECKPOINT_FORMAT:
            raise ValueError("Unsupported GAT reinforcement checkpoint format")
        if state.get("manifest_hash") != manifest_digest:
            raise ValueError("Training manifest changed since GAT reinforcement")
        if state.get("source_best_checkpoint_hash") != source_hash:
            raise ValueError("Best GAT checkpoint changed since reinforcement started")
        if state.get("config") != config:
            raise ValueError("GAT reinforcement configuration changed since checkpoint")
        _validate_reinforcement_state(state, normal_rounds)
        current_round = int(state.get("current_round", 0))
        episode_index = int(state.get("episode_index", -1))
        if not 1 <= current_round <= reinforcement_rounds + 1:
            raise ValueError("GAT reinforcement checkpoint round is invalid")
        if episode_index < 0 or int(state.get("completed_episodes", -1)) < 0:
            raise ValueError("GAT reinforcement checkpoint episode is invalid")
        current_unresolved = _validate_reinforcement_faults(
            state.get("current_unresolved", []), circuits
        )
        state["current_unresolved"] = [list(fault) for fault in current_unresolved]
        saved_cuda_state = state["torch_cuda_random_state"]
        if str(device).startswith("cuda") != (saved_cuda_state is not None):
            raise ValueError(
                "GAT reinforcement checkpoint RNG state does not match the device"
            )
        agent.load_training_state_dict(state["agent"])
        torch.set_rng_state(state["torch_random_state"])
        if saved_cuda_state is not None:
            torch.cuda.set_rng_state_all(saved_cuda_state)
        print(
            f"REINFORCEMENT_RESUME round={state['current_round']} "
            f"episode={state['episode_index']} "
            f"unresolved={len(state['current_unresolved'])}",
            flush=True,
        )
    else:
        source = torch.load(best_checkpoint_path, map_location="cpu")
        if source.get("format") != BEST_CHECKPOINT_FORMAT:
            raise ValueError("GAT reinforcement requires the current best checkpoint")
        if source.get("manifest_hash") != manifest_digest:
            raise ValueError("Best GAT checkpoint uses a different training manifest")
        source_config = source.get("config", {})
        if source_config.get("encoder_variant") != "level_gat_gru":
            raise ValueError("Only the level_gat_gru model can be reinforced")
        expected_source_protocol = {
            "heuristic": HEURISTIC,
            "circuit_order": [item["name"] for item in circuits],
            "faults_per_circuit": FAULTS_PER_CIRCUIT,
            "normal_rounds": normal_rounds,
            "reinforcement_rounds": reinforcement_rounds,
        }
        if any(
            source_config.get(key) != value
            for key, value in expected_source_protocol.items()
        ):
            raise ValueError("Best GAT checkpoint training protocol is incompatible")
        if source_config.get("device") != str(device):
            raise ValueError(
                "Best GAT checkpoint device does not match reinforcement device"
            )
        agent.load_training_state_dict(source["agent"])
        initial_evaluation = evaluate_round(
            circuits, evaluators, 0, backtrack_limit, seed
        )
        initial_unresolved = _validate_reinforcement_evaluation(
            initial_evaluation, circuits
        )
        initial_evaluation.update({
            "reinforcement_round": 0,
            "trained_faults": [],
            "unresolved_faults": [list(fault) for fault in initial_unresolved],
            "is_best": True,
        })
        source_round = int(source["round"])
        initial_score = validation_score(initial_evaluation, source_round)
        state = {
            "format": REINFORCEMENT_CHECKPOINT_FORMAT,
            "manifest_hash": manifest_digest,
            "source_best_checkpoint_hash": source_hash,
            "config": config,
            "current_round": 1,
            "episode_index": 0,
            "completed_episodes": 0,
            "current_unresolved": [list(fault) for fault in initial_unresolved],
            "round_metrics": [initial_evaluation],
            "best_score": list(initial_score),
            "best_round": 0,
            "best_validation_round": source_round,
            "best_agent": _clone(agent.training_state_dict()),
            "best_evaluation": initial_evaluation,
        }
        _save_state(checkpoint_path, agent, state)

    circuit_by_name = {item["name"]: item for item in circuits}
    _save_reinforcement_best(
        best_path, model_path, state, manifest_digest, config,
        state["best_evaluation"],
    )
    _atomic_json(metrics_path, state["round_metrics"])
    _atomic_json(unresolved_path, {
        "reinforcement_round": int(state["current_round"]) - 1,
        "faults": state["current_unresolved"],
    })
    while state["current_round"] <= reinforcement_rounds:
        round_number = int(state["current_round"])
        if not state["current_unresolved"]:
            print(
                f"REINFORCEMENT_COMPLETE round={round_number - 1} "
                "unresolved=0 reason=all_detected",
                flush=True,
            )
            break
        order = _reinforcement_order(
            state["current_unresolved"], seed, round_number
        )
        episode_index = int(state["episode_index"])
        if episode_index > len(order):
            raise ValueError("GAT reinforcement checkpoint episode is out of range")
        for index in range(episode_index, len(order)):
            circuit_name, fault_id = order[index]
            item = circuit_by_name[circuit_name]
            trainer = trainers[circuit_name]
            trainer.run(
                item["circuit"],
                backtrack_limit=backtrack_limit,
                seed=seed + normal_rounds + round_number,
                fault_ids=[fault_id],
                fault_map_path=item["fault_map"],
                use_scoap=True,
            )
            metrics = trainer.episode_metrics[0]
            state["episode_index"] = index + 1
            state["completed_episodes"] += 1
            step = int(state["completed_episodes"])
            writer.add_scalar(
                "reinforcement/episode_backtracks", metrics["backtracks"], step
            )
            writer.add_scalar(
                "reinforcement/episode_backtrace_steps",
                metrics["backtrace_steps"], step,
            )
            writer.add_scalar(
                "reinforcement/episode_return", metrics["combined_reward_sum"], step
            )
            writer.add_scalar(
                "reinforcement/episode_detected", metrics["detected"], step
            )
            writer.flush()
            _save_state(checkpoint_path, agent, state)
            print(
                f"REINFORCEMENT_EPISODE round={round_number}/{reinforcement_rounds} "
                f"index={index + 1}/{len(order)} circuit={circuit_name} "
                f"fault={fault_id} detected={metrics['detected']} "
                f"backtracks={metrics['backtracks']}",
                flush=True,
            )

        evaluation = evaluate_round(
            circuits, evaluators, round_number, backtrack_limit, seed
        )
        next_unresolved = _validate_reinforcement_evaluation(
            evaluation, circuits
        )
        evaluation.update({
            "reinforcement_round": round_number,
            "trained_faults": [list(fault) for fault in order],
            "unresolved_faults": [list(fault) for fault in next_unresolved],
        })
        validation_round = normal_rounds + round_number
        score = validation_score(evaluation, validation_round)
        is_best = score < tuple(state["best_score"])
        evaluation["is_best"] = bool(is_best)
        state["round_metrics"].append(evaluation)
        if is_best:
            state["best_score"] = list(score)
            state["best_round"] = round_number
            state["best_validation_round"] = validation_round
            state["best_agent"] = _clone(agent.training_state_dict())
            state["best_evaluation"] = evaluation
            _save_reinforcement_best(
                best_path, model_path, state, manifest_digest, config, evaluation
            )
        state["current_unresolved"] = [list(fault) for fault in next_unresolved]
        state["current_round"] = round_number + 1
        state["episode_index"] = 0
        writer.add_scalar(
            "reinforcement/round_detected_faults",
            evaluation["detected_faults"], round_number,
        )
        writer.add_scalar(
            "reinforcement/round_unresolved_faults",
            len(next_unresolved), round_number,
        )
        writer.add_scalar(
            "reinforcement/round_backtracks_total",
            evaluation["backtracks_total"], round_number,
        )
        writer.add_scalar(
            "reinforcement/round_is_best", int(is_best), round_number
        )
        writer.flush()
        _atomic_json(metrics_path, state["round_metrics"])
        _atomic_json(unresolved_path, {
            "reinforcement_round": round_number,
            "faults": state["current_unresolved"],
        })
        _save_state(checkpoint_path, agent, state)
        print(
            f"REINFORCEMENT_ROUND round={round_number}/{reinforcement_rounds} "
            f"trained={len(order)} "
            f"detected={evaluation['detected_faults']}/{evaluation['episodes']} "
            f"unresolved={len(next_unresolved)} best={int(is_best)}",
            flush=True,
        )

    print(
        f"REINFORCEMENT_FINISHED rounds={state['current_round'] - 1} "
        f"episodes={state['completed_episodes']} "
        f"best_round={state['best_round']} "
        f"unresolved={len(state['current_unresolved'])}",
        flush=True,
    )
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rnd-beta", type=float, default=0.05)
    parser.add_argument("--k-epochs", type=int, default=8)
    parser.add_argument(
        "--reinforcement-rounds", type=int, default=0,
        help="Extra unresolved-fault rounds; supported only by level_gat_gru.",
    )
    parser.add_argument(
        "--encoder", choices=tuple(AGENT_TYPES), default="level_gat_gru",
        help="Graph encoder variant.",
    )
    args = parser.parse_args(argv)
    if args.rounds <= 0 or args.k_epochs <= 0:
        raise ValueError("Rounds and PPO epochs must be positive")
    if args.reinforcement_rounds < 0:
        raise ValueError("Reinforcement rounds cannot be negative")
    if args.reinforcement_rounds > MAX_REINFORCEMENT_ROUNDS:
        raise ValueError(
            f"Reinforcement rounds cannot exceed {MAX_REINFORCEMENT_ROUNDS}"
        )
    if args.encoder != "level_gat_gru" and args.reinforcement_rounds:
        raise ValueError("Only level_gat_gru supports reinforcement training")
    if args.rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"SmartATPG training requires exactly {NORMAL_TRAINING_ROUNDS} "
            "normal training rounds"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    circuits = _validate_manifest(manifest)
    if int(manifest["normal_rounds"]) != args.rounds:
        raise ValueError("Requested normal rounds do not match the training manifest")
    if int(manifest["reinforcement_rounds"]) != args.reinforcement_rounds:
        raise ValueError(
            "Requested reinforcement rounds do not match the training manifest"
        )
    backtrack_limit = int(manifest["backtrack_limit"])
    graphs = {item["name"]: load_circuit_graph(item["circuit"]) for item in circuits}
    agent = AGENT_TYPES[args.encoder](
        graphs,
        hidden_dim=32,
        lr_actor=0.001,
        lr_critic=0.01,
        rnd_beta=args.rnd_beta,
        k_epochs=args.k_epochs,
    )
    trainers = {
        item["name"]: CppPodemBacktraceV2Trainer(graphs[item["name"]], agent=agent)
        for item in circuits
    }
    evaluators = {
        item["name"]: CppPodemBacktraceV2Evaluator(graphs[item["name"]], agent=agent)
        for item in circuits
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_state.pth"
    best_checkpoint_path = output_dir / "best_training_state.pth"
    model_best_path = output_dir / "model_best.txt"
    model_latest_path = output_dir / "model_latest.txt"
    metrics_path = output_dir / "round_metrics.json"
    manifest_digest = _manifest_hash(args.manifest)
    config = {
        "rounds": args.rounds,
        "seed": args.seed,
        "rnd_beta": args.rnd_beta,
        "k_epochs": args.k_epochs,
        "backtrack_limit": backtrack_limit,
        "actor_lr": 0.001,
        "critic_lr": 0.01,
        "faults_per_round": sum(
            len(item["training_fault_ids"]) for item in circuits
        ),
        "bc_epochs": 0,
        "curriculum_stages": 0,
        "device": str(device),
        "paper_reward": PAPER_REWARD,
        "encoder_variant": args.encoder,
        "heuristic": HEURISTIC,
        "circuit_order": list(CIRCUITS),
        "faults_per_circuit": FAULTS_PER_CIRCUIT,
        "normal_rounds": args.rounds,
        "reinforcement_rounds": args.reinforcement_rounds,
    }
    state = {
        "format": CHECKPOINT_FORMAT,
        "manifest_hash": manifest_digest,
        "config": config,
        "current_round": 1,
        "episode_index": 0,
        "completed_episodes": 0,
        "round_metrics": [],
        "best_score": None,
        "best_round": None,
        "best_agent": None,
    }
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu")
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Legacy SmartATPG checkpoint is incompatible with 12D CO training")
        if saved.get("manifest_hash") != manifest_digest:
            raise ValueError("Training manifest changed since checkpoint")
        target_rounds = _validate_resume_config(saved.get("config"), config)
        current_round = int(saved.get("current_round", 0))
        episode_index = int(saved.get("episode_index", 0))
        _validate_round_target(current_round, episode_index, target_rounds)
        agent.load_training_state_dict(saved["agent"])
        state.update({key: saved[key] for key in state})
        state["config"] = config
        torch.set_rng_state(saved["torch_random_state"])
        if saved.get("torch_cuda_random_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["torch_cuda_random_state"])
        print(
            f"RESUME round={state['current_round']} "
            f"episode={state['episode_index']} total={state['completed_episodes']}",
            flush=True,
        )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("Install tensorboard before SmartATPG training") from error
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    circuit_by_name = {item["name"]: item for item in circuits}
    training_protocol = _training_protocol(config)
    export_actor(
        agent.policy_old.state_dict(), model_latest_path,
        training_protocol=training_protocol,
    )
    if state["best_agent"] is not None:
        export_actor(
            state["best_agent"]["policy_old"],
            model_best_path,
            best_round=state["best_round"],
            best_score=state["best_score"],
            training_protocol=training_protocol,
        )

    try:
        while state["current_round"] <= args.rounds:
            round_number = int(state["current_round"])
            order = _episode_order(circuits, args.seed, round_number)
            for index in range(int(state["episode_index"]), len(order)):
                circuit_name, fault_id = order[index]
                item = circuit_by_name[circuit_name]
                trainer = trainers[circuit_name]
                trainer.run(
                    item["circuit"],
                    backtrack_limit=backtrack_limit,
                    seed=args.seed + round_number,
                    fault_ids=[fault_id],
                    fault_map_path=item["fault_map"],
                    use_scoap=True,
                )
                metrics = trainer.episode_metrics[0]
                state["episode_index"] = index + 1
                state["completed_episodes"] += 1
                step = int(state["completed_episodes"])
                writer.add_scalar("episode/backtracks", metrics["backtracks"], step)
                writer.add_scalar("episode/backtrace_steps", metrics["backtrace_steps"], step)
                writer.add_scalar("episode/return", metrics["combined_reward_sum"], step)
                writer.add_scalar("episode/extrinsic_return", metrics["extrinsic_reward_sum"], step)
                writer.add_scalar("episode/intrinsic_return", metrics["scaled_intrinsic_reward_sum"], step)
                writer.add_scalar("episode/detected", metrics["detected"], step)
                writer.add_scalar("episode/ppo_loss", metrics["total_loss"], step)
                writer.add_scalar("episode/rnd_loss", metrics["rnd_loss"], step)
                writer.flush()
                export_actor(
                    agent.policy_old.state_dict(), model_latest_path,
                    training_protocol=training_protocol,
                )
                _save_state(checkpoint_path, agent, state)
                print(
                    f"EPISODE round={round_number}/{args.rounds} "
                    f"index={index + 1}/{len(order)} circuit={circuit_name} "
                    f"fault={fault_id} backtracks={metrics['backtracks']} "
                    f"backtrace_steps={metrics['backtrace_steps']}",
                    flush=True,
                )

            evaluation = evaluate_round(
                circuits, evaluators, round_number, backtrack_limit, args.seed
            )
            score = validation_score(evaluation, round_number)
            is_best = state["best_score"] is None or score < tuple(state["best_score"])
            evaluation["is_best"] = bool(is_best)
            state["round_metrics"].append(evaluation)
            if is_best:
                state["best_score"] = list(score)
                state["best_round"] = round_number
                state["best_agent"] = _clone(agent.training_state_dict())
                best_payload = {
                    "format": BEST_CHECKPOINT_FORMAT,
                    "manifest_hash": manifest_digest,
                    "config": config,
                    "round": round_number,
                    "score": list(score),
                    "evaluation": evaluation,
                    "agent": state["best_agent"],
                }
                _atomic_torch_save(best_checkpoint_path, best_payload)
                export_actor(
                    state["best_agent"]["policy_old"], model_best_path,
                    best_round=round_number, best_score=score,
                    training_protocol=training_protocol,
                )
            writer.add_scalar("round/backtracks_total", evaluation["backtracks_total"], round_number)
            writer.add_scalar("round/backtracks_mean", evaluation["backtracks_mean"], round_number)
            writer.add_scalar("round/backtrace_steps_total", evaluation["backtrace_steps_total"], round_number)
            writer.add_scalar("round/backtrace_steps_mean", evaluation["backtrace_steps_mean"], round_number)
            writer.add_scalar("round/return_total", evaluation["return_total"], round_number)
            writer.add_scalar("round/return_mean", evaluation["return_mean"], round_number)
            writer.add_scalar("round/detected_faults", evaluation["detected_faults"], round_number)
            writer.add_scalar("round/fault_coverage", evaluation["fault_coverage"], round_number)
            writer.add_scalar("round/is_best", int(is_best), round_number)
            writer.flush()
            _atomic_json(metrics_path, state["round_metrics"])
            state["current_round"] = round_number + 1
            state["episode_index"] = 0
            _save_state(checkpoint_path, agent, state)
            print(
                f"ROUND round={round_number}/{args.rounds} "
                f"detected={evaluation['detected_faults']}/{evaluation['episodes']} "
                f"backtracks={evaluation['backtracks_total']} "
                f"backtrace_steps={evaluation['backtrace_steps_total']} "
                f"best={int(is_best)}",
                flush=True,
            )
        if args.reinforcement_rounds:
            if not best_checkpoint_path.is_file():
                raise RuntimeError("GAT training completed without a best checkpoint")
            _run_gat_reinforcement(
                agent=agent,
                circuits=circuits,
                trainers=trainers,
                evaluators=evaluators,
                output_dir=output_dir,
                best_checkpoint_path=best_checkpoint_path,
                manifest_digest=manifest_digest,
                normal_rounds=args.rounds,
                reinforcement_rounds=args.reinforcement_rounds,
                backtrack_limit=backtrack_limit,
                seed=args.seed,
                writer=writer,
            )
    finally:
        writer.close()

    if not args.reinforcement_rounds:
        export_actor(
            agent.policy_old.state_dict(), model_latest_path,
            training_protocol=training_protocol,
        )
    print(
        f"TRAINING_COMPLETE rounds={args.rounds} "
        f"reinforcement_rounds={args.reinforcement_rounds} "
        f"episodes={state['completed_episodes']} best_round={state['best_round']} "
        f"device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Train the 11D SmartATPG GraphSAGE policy without BC or curriculum."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from prepare_smartatpg_training import (
    MANIFEST_FORMAT, select_hard_faults, sha256_file,
)
from rl_podem.backends import smartatpg_metadata
from rl_podem.cpp_bridge import (
    CppPodemBacktraceV2Evaluator, CppPodemBacktraceV2Trainer,
    smartatpg_pi_reward,
)
from rl_podem.ppo import device
from rl_podem.smartatpg import SmartATPGPPOAgent
from rl_podem.smartatpg_artifacts import export_actor
from rl_podem.smartatpg_features import load_circuit_graph


CHECKPOINT_FORMAT = "SMARTATPG_11D_TRAINING_V1"
BEST_CHECKPOINT_FORMAT = "SMARTATPG_11D_BEST_V1"
PAPER_REWARD = {
    "non_pi": -0.1,
    "alpha": 7.5,
    "beta": 0.07,
    "detected": 100.0,
    "undetected": -100.0,
}


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
    expected = {"format": MANIFEST_FORMAT, **smartatpg_metadata()}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest is not compatible with SmartATPG 11D training")
    circuits = list(manifest.get("circuits", []))
    if [item.get("name") for item in circuits] != ["c6288", "s38417"]:
        raise ValueError("Training requires exactly c6288 and s38417")
    count = int(manifest.get("fault_count_per_circuit", -1))
    if count != 100:
        raise ValueError("Paper training requires exactly 100 faults per circuit")
    for item in circuits:
        if len(item.get("training_fault_ids", [])) != count:
            raise ValueError(f"Circuit {item['name']} must contain 100 fault IDs")
        required_artifacts = {"source_circuit", "circuit", "fault_map", "profile"}
        if item["name"] == "s38417":
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
            raise ValueError(f"Circuit {item['name']} faults are not the baseline top 100")
        if item.get("training_faults") != selected:
            raise ValueError(f"Circuit {item['name']} ranking metadata changed")
    return circuits


def _episode_order(circuits, seed, round_number):
    episodes = [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["training_fault_ids"]
    ]
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rnd-beta", type=float, default=0.05)
    parser.add_argument("--k-epochs", type=int, default=8)
    args = parser.parse_args(argv)
    if args.rounds <= 0 or args.k_epochs <= 0:
        raise ValueError("Rounds and PPO epochs must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    circuits = _validate_manifest(manifest)
    backtrack_limit = int(manifest["backtrack_limit"])
    graphs = {item["name"]: load_circuit_graph(item["circuit"]) for item in circuits}
    agent = SmartATPGPPOAgent(
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
        "faults_per_round": 200,
        "bc_epochs": 0,
        "curriculum_stages": 0,
        "device": str(device),
        "paper_reward": PAPER_REWARD,
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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu")
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Legacy SmartATPG checkpoint is incompatible with 11D training")
        if saved.get("manifest_hash") != manifest_digest or saved.get("config") != config:
            raise ValueError("Training manifest or configuration changed since checkpoint")
        agent.load_training_state_dict(saved["agent"])
        state.update({key: saved[key] for key in state})
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
    export_actor(agent.policy_old.state_dict(), model_latest_path)
    if state["best_agent"] is not None:
        export_actor(
            state["best_agent"]["policy_old"],
            model_best_path,
            best_round=state["best_round"],
            best_score=state["best_score"],
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
                export_actor(agent.policy_old.state_dict(), model_latest_path)
                _save_state(checkpoint_path, agent, state)
                print(
                    f"EPISODE round={round_number}/{args.rounds} "
                    f"index={index + 1}/200 circuit={circuit_name} "
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
                f"detected={evaluation['detected_faults']}/200 "
                f"backtracks={evaluation['backtracks_total']} "
                f"backtrace_steps={evaluation['backtrace_steps_total']} "
                f"best={int(is_best)}",
                flush=True,
            )
    finally:
        writer.close()

    export_actor(agent.policy_old.state_dict(), model_latest_path)
    print(
        f"TRAINING_COMPLETE rounds={args.rounds} "
        f"episodes={state['completed_episodes']} best_round={state['best_round']} "
        f"device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()

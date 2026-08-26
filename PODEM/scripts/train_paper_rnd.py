import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from rl_podem.cpp_bridge import (
    CppPodemBacktraceV2Evaluator,
    CppPodemBacktraceV2Trainer,
    export_actor_v2_state_dict,
    load_cpp_embedding_table,
)
from rl_podem.ppo import BacktracePPOAgentV2


CHECKPOINT_FORMAT = "RL_PODEM_PAPER_RND_TRAINING_V3"
MANIFEST_FORMAT = "RL_PODEM_PAPER_TRAINING_V3"
PAPER_REWARD = {
    "non_pi": -0.1,
    "alpha": 7.5,
    "beta": 0.07,
    "detected": 100.0,
    "undetected": -100.0,
}


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_policy_state(state_dict):
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def _default_latest_actor(best_actor: Path) -> Path:
    stem = best_actor.stem
    if stem.endswith("_best"):
        stem = stem[: -len("_best")] + "_latest"
    else:
        stem += "_latest"
    return best_actor.with_name(stem + best_actor.suffix)


def _save_checkpoint(
    path,
    agent,
    progress,
    validation_history,
    best_policy_state,
    best_score,
    best_sweep,
    rng,
    manifest_hash,
    seed,
    backtrack_limit,
    rnd_beta,
    training_config,
):
    state = {
        "format": CHECKPOINT_FORMAT,
        "agent": agent.training_state_dict(),
        "progress": progress,
        "validation_history": validation_history,
        "best_policy_state": best_policy_state,
        "best_score": list(best_score),
        "best_sweep": best_sweep,
        "python_random_state": rng.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "manifest_hash": manifest_hash,
        "seed": seed,
        "rl_mode": "backtrace_rl",
        "backtrack_limit": backtrack_limit,
        "paper_reward": PAPER_REWARD,
        "rnd_beta": rnd_beta,
        "training_config": training_config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _validation_score(summary):
    return (
        -int(summary["detected"]),
        int(summary["aborted"]),
        int(summary["backtracks"]),
        int(summary["backtrace_steps"]),
        int(summary["decisions"]),
    )


def _run_validation(circuits, evaluators, sweep, backtrack_limit, seed):
    aggregate = {
        "episodes": 0,
        "detected": 0,
        "redundant": 0,
        "aborted": 0,
        "decisions": 0,
        "backtracks": 0,
        "backtrace_steps": 0,
        "pi_visits": 0,
    }
    circuit_results = []
    for item in circuits:
        summary = evaluators[item["name"]].run(
            item["circuit"],
            backtrack_limit=backtrack_limit,
            seed=seed,
            fault_ids=list(item["validation_fault_ids"]),
            quiet=True,
            rl_mode="backtrace_rl",
            fault_map_path=item["fault_map"],
        )
        summary = {key: int(value) for key, value in summary.items()}
        for key in aggregate:
            aggregate[key] += summary[key]
        circuit_results.append({"circuit": item["name"], "summary": summary})

    score = _validation_score(aggregate)
    return {
        "sweep": sweep,
        "summary": aggregate,
        "score": list(score),
        "circuits": circuit_results,
    }, score


def _validate_manifest(manifest, backtrack_limit):
    if backtrack_limit != 500:
        raise ValueError("Paper-reward V3 training requires --backtrack-limit=500.")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Paper reward training requires a V3 manifest.")
    limits = manifest.get("backtrack_limits", {})
    expected = {"profile", "training", "validation"}
    if set(limits) != expected or any(
        int(limits[name]) != backtrack_limit for name in expected
    ):
        raise ValueError(
            "Manifest profile/training/validation backtrack limits must all match "
            f"--backtrack-limit={backtrack_limit}."
        )
    circuits = manifest.get("circuits", [])
    if not circuits:
        raise ValueError("Training manifest contains no circuits.")
    training_count = int(manifest.get("training_fault_count_per_circuit", -1))
    validation_count = int(manifest.get("validation_fault_count_per_circuit", -1))
    names = [item.get("name") for item in circuits]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Manifest circuit names must be non-empty and unique.")
    for item in circuits:
        training = list(item.get("training_fault_ids", []))
        validation = list(item.get("validation_fault_ids", []))
        if not training or not validation:
            raise ValueError(
                f"Circuit {item.get('name', '<unnamed>')} has an empty fault split."
            )
        if len(training) != training_count or len(validation) != validation_count:
            raise ValueError("Fault split lengths do not match the manifest counts.")
        if len(training) != len(set(training)) or len(validation) != len(
            set(validation)
        ):
            raise ValueError("Fault splits must not contain duplicate IDs.")
        if set(training) & set(validation):
            raise ValueError("Training and validation fault IDs must be disjoint.")
        artifact_keys = {"source_circuit", "circuit", "fault_map", "embeddings"}
        artifact_hashes = item.get("artifact_sha256", {})
        if set(artifact_hashes) != artifact_keys:
            raise ValueError(
                f"Circuit {item['name']} must provide SHA256 for every input artifact."
            )
        for key in sorted(artifact_keys):
            path = Path(item[key])
            if not path.is_file():
                raise FileNotFoundError(f"Missing {key} for {item['name']}: {path}")
            actual_hash = _sha256_file(path)
            if actual_hash != artifact_hashes[key]:
                raise ValueError(
                    f"Circuit {item['name']} {key} content changed since profiling."
                )
    return circuits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO+RND with SmartATPG rewards and best-model validation."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("best_actor_output", type=Path)
    parser.add_argument("--latest-actor-output", type=Path)
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rnd-beta", type=float, default=0.05)
    parser.add_argument("--backtrack-limit", type=int, default=500)
    args = parser.parse_args()
    if args.sweeps <= 0:
        raise ValueError("--sweeps must be positive.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    circuits = _validate_manifest(manifest, args.backtrack_limit)
    latest_actor_output = (
        args.latest_actor_output
        if args.latest_actor_output is not None
        else _default_latest_actor(args.best_actor_output)
    )
    output_paths = {
        args.checkpoint.resolve(),
        args.best_actor_output.resolve(),
        latest_actor_output.resolve(),
    }
    if len(output_paths) != 3:
        raise ValueError("Checkpoint, best actor, and latest actor paths must differ.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    first_embeddings = load_cpp_embedding_table(circuits[0]["embeddings"])
    embedding_dim = next(iter(first_embeddings.values())).numel()
    agent = BacktracePPOAgentV2(
        gate_embedding_dim=embedding_dim,
        hidden_dim=32,
        rnd_beta=args.rnd_beta,
    )
    training_config = {
        "rl_mode": "backtrace_rl",
        "planned_sweeps": args.sweeps,
        "backtrack_limit": args.backtrack_limit,
        "paper_reward": PAPER_REWARD,
        "agent": agent.hyperparameters(),
        "gate_embedding_dim": agent.gate_embedding_dim,
        "hidden_dim": agent.hidden_dim,
    }
    trainers = {
        item["name"]: CppPodemBacktraceV2Trainer(item["embeddings"], agent=agent)
        for item in circuits
    }
    evaluators = {
        item["name"]: CppPodemBacktraceV2Evaluator(item["embeddings"], agent=agent)
        for item in circuits
    }
    progress = []
    validation_history = []
    best_policy_state = None
    best_score = None
    best_sweep = None
    manifest_digest = _manifest_hash(args.manifest)

    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Cannot resume an incompatible V1/V2 checkpoint.")
        if state["manifest_hash"] != manifest_digest:
            raise ValueError("Training manifest changed since the checkpoint was saved.")
        if state.get("rl_mode") != "backtrace_rl":
            raise ValueError("V3 checkpoints must use backtrace_rl.")
        if int(state.get("seed", -1)) != args.seed:
            raise ValueError("Checkpoint seed does not match the command.")
        if int(state.get("backtrack_limit", -1)) != args.backtrack_limit:
            raise ValueError("Checkpoint backtrack limit does not match the command.")
        if state.get("paper_reward") != PAPER_REWARD:
            raise ValueError("Checkpoint does not use the configured paper reward.")
        if float(state.get("rnd_beta", -1.0)) != args.rnd_beta:
            raise ValueError("Checkpoint RND beta does not match the command.")
        if state.get("training_config") != training_config:
            raise ValueError("Checkpoint PPO/RND training configuration changed.")
        agent.load_training_state_dict(state["agent"])
        progress = list(state["progress"])
        validation_history = list(state["validation_history"])
        best_policy_state = _clone_policy_state(state["best_policy_state"])
        best_score = tuple(int(value) for value in state["best_score"])
        best_sweep = int(state["best_sweep"])
        circuit_names = {item["name"] for item in circuits}
        progress_keys = [
            (int(item["sweep"]), item["circuit"]) for item in progress
        ]
        if any(sweep <= 0 or name not in circuit_names for sweep, name in progress_keys):
            raise ValueError("Checkpoint training progress is invalid.")
        if len(progress_keys) != len(set(progress_keys)):
            raise ValueError("Checkpoint training progress contains duplicates.")
        validation_sweeps = [int(item["sweep"]) for item in validation_history]
        if not validation_sweeps or validation_sweeps[0] != 0:
            raise ValueError("Checkpoint is missing the initial validation.")
        if len(validation_sweeps) != len(set(validation_sweeps)):
            raise ValueError("Checkpoint validation history contains duplicates.")
        completed_keys = set(progress_keys)
        for sweep in validation_sweeps:
            if sweep > 0 and any(
                (sweep, circuit_name) not in completed_keys
                for circuit_name in circuit_names
            ):
                raise ValueError("Checkpoint validates an incomplete sweep.")
        score_by_sweep = {
            int(item["sweep"]): tuple(int(value) for value in item["score"])
            for item in validation_history
        }
        if score_by_sweep.get(best_sweep) != best_score:
            raise ValueError("Checkpoint best score is inconsistent with validation.")
        rng.setstate(state["python_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        cuda_random_state = state.get("torch_cuda_random_state")
        if cuda_random_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_random_state)
        print(
            f"RESUME runs={len(progress)} validations={len(validation_history)} "
            f"best_sweep={best_sweep}",
            flush=True,
        )
    else:
        validation, best_score = _run_validation(
            circuits, evaluators, 0, args.backtrack_limit, args.seed
        )
        validation_history.append(validation)
        best_policy_state = _clone_policy_state(agent.policy_old.state_dict())
        best_sweep = 0
        print(f"VALIDATION {json.dumps(validation, sort_keys=True)}", flush=True)

    export_actor_v2_state_dict(agent.policy_old.state_dict(), latest_actor_output)
    export_actor_v2_state_dict(best_policy_state, args.best_actor_output)
    _save_checkpoint(
        args.checkpoint,
        agent,
        progress,
        validation_history,
        best_policy_state,
        best_score,
        best_sweep,
        rng,
        manifest_digest,
        args.seed,
        args.backtrack_limit,
        args.rnd_beta,
        training_config,
    )

    completed = {(int(item["sweep"]), item["circuit"]) for item in progress}
    validated_sweeps = {int(item["sweep"]) for item in validation_history}
    for sweep in range(1, args.sweeps + 1):
        circuit_order = list(circuits)
        random.Random(args.seed + sweep).shuffle(circuit_order)
        for item in circuit_order:
            key = (sweep, item["name"])
            if key in completed:
                continue
            fault_ids = list(item["training_fault_ids"])
            random.Random(f"{args.seed}:{sweep}:{item['name']}").shuffle(fault_ids)
            print(
                f"TRAIN_START sweep={sweep}/{args.sweeps} circuit={item['name']} "
                f"faults={len(fault_ids)}",
                flush=True,
            )
            summary = trainers[item["name"]].run(
                item["circuit"],
                backtrack_limit=args.backtrack_limit,
                seed=args.seed + sweep,
                fault_ids=fault_ids,
                quiet=True,
                rl_mode="backtrace_rl",
                fault_map_path=item["fault_map"],
            )
            record = {
                "sweep": sweep,
                "circuit": item["name"],
                "summary": {key: int(value) for key, value in summary.items()},
                "learning": trainers[item["name"]].run_metrics,
            }
            progress.append(record)
            completed.add(key)
            export_actor_v2_state_dict(
                agent.policy_old.state_dict(), latest_actor_output
            )
            _save_checkpoint(
                args.checkpoint,
                agent,
                progress,
                validation_history,
                best_policy_state,
                best_score,
                best_sweep,
                rng,
                manifest_digest,
                args.seed,
                args.backtrack_limit,
                args.rnd_beta,
                training_config,
            )
            print(f"TRAIN_RESULT {json.dumps(record, sort_keys=True)}", flush=True)

        if sweep not in validated_sweeps:
            validation, score = _run_validation(
                circuits, evaluators, sweep, args.backtrack_limit, args.seed
            )
            validation_history.append(validation)
            validated_sweeps.add(sweep)
            if score < best_score:
                best_score = score
                best_sweep = sweep
                best_policy_state = _clone_policy_state(
                    agent.policy_old.state_dict()
                )
                export_actor_v2_state_dict(
                    best_policy_state, args.best_actor_output
                )
                selection = "best"
            else:
                selection = "kept"
            _save_checkpoint(
                args.checkpoint,
                agent,
                progress,
                validation_history,
                best_policy_state,
                best_score,
                best_sweep,
                rng,
                manifest_digest,
                args.seed,
                args.backtrack_limit,
                args.rnd_beta,
                training_config,
            )
            print(
                f"VALIDATION selection={selection} "
                f"{json.dumps(validation, sort_keys=True)}",
                flush=True,
            )

    print(
        f"TRAINING_COMPLETE runs={len(progress)} updates={agent.update_count} "
        f"best_sweep={best_sweep} best_score={list(best_score)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

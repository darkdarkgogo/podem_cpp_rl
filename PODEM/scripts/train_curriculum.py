import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from rl_podem.cpp_bridge import (
    CppPodemBacktraceV2Evaluator,
    export_actor_v2_state_dict,
)
from rl_podem.curriculum import (
    DIFFICULTIES,
    REWARD_CONFIG,
    REWARD_DISTRIBUTION,
    CppPodemCurriculumTrainer,
    filter_unseen_teacher_samples,
    load_embedding_tables,
    pretrain_actor,
)
from rl_podem.ppo import BacktracePPOAgentV2


CHECKPOINT_FORMAT = "RL_PODEM_CURRICULUM_TRAINING_V4"
MANIFEST_FORMAT = "RL_PODEM_CURRICULUM_V4"
RND_SCHEDULE = (0.05, 0.02, 0.0)
ENTROPY_SCHEDULE = (0.01, 0.005, 0.001)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_policy_state(state_dict):
    return {
        name: value.detach().cpu().clone() for name, value in state_dict.items()
    }


def _atomic_torch_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _manifest_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_checkpoint_metadata(state, manifest_digest, config):
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported curriculum checkpoint format.")
    if state.get("manifest_hash") != manifest_digest or state.get("config") != config:
        raise ValueError("Curriculum manifest or training configuration changed.")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_manifest(manifest):
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Curriculum training requires a V4 manifest.")
    if int(manifest.get("backtrack_limit", -1)) != 500:
        raise ValueError("V4 manifest backtrack limit must be 500.")
    circuits = list(manifest.get("circuits", []))
    if not circuits:
        raise ValueError("V4 manifest has no circuits.")
    names = [str(item.get("name", "")) for item in circuits]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("V4 circuit names must be non-empty and unique.")
    for item in circuits:
        hashes = item.get("artifact_sha256", {})
        if set(hashes) != {"circuit", "fault_map", "embeddings", "profile"}:
            raise ValueError(f"Incomplete artifact hashes for {item['name']}.")
        for key, expected in hashes.items():
            path = Path(item[key])
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"V4 artifact changed or is missing: {path}")
        training = list(item.get("training_faults", []))
        validation = list(item.get("validation_faults", []))
        if not training or not validation:
            raise ValueError(f"Empty fault split for {item['name']}.")
        training_ids = {str(fault["fault_id"]) for fault in training}
        validation_ids = {str(fault["fault_id"]) for fault in validation}
        if len(training_ids) != len(training) or len(validation_ids) != len(validation):
            raise ValueError(f"Duplicate fault ID in {item['name']} split.")
        if training_ids & validation_ids:
            raise ValueError(f"Overlapping fault split for {item['name']}.")
        for fault in training + validation:
            if fault.get("difficulty") not in DIFFICULTIES:
                raise ValueError(f"Invalid fault difficulty in {item['name']}.")
            for key in ("outcome", "backtracks", "backtrace_steps"):
                int(fault[key])
    for kind in ("training", "validation"):
        path = Path(manifest[f"teacher_{kind}"])
        expected = manifest["teacher_sha256"][kind]
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Teacher {kind} artifact changed or is missing.")
    return circuits


def _baseline_map(faults):
    return {str(fault["fault_id"]): dict(fault) for fault in faults}


def _validation_score(circuits, results):
    detected = 0
    aborted = 0
    decisions = 0
    backtrack_ratios = []
    backtrace_ratios = []
    for item in circuits:
        summary = results[item["name"]]
        baselines = list(item["validation_faults"])
        baseline_backtracks = sum(int(fault["backtracks"]) for fault in baselines)
        baseline_backtrace = sum(int(fault["backtrace_steps"]) for fault in baselines)
        detected += int(summary["detected"])
        aborted += int(summary["aborted"])
        decisions += int(summary["decisions"])
        if baseline_backtrace <= 0:
            raise ValueError(
                f"Validation backtrace baseline must be positive for {item['name']}."
            )
        rl_backtracks = int(summary["backtracks"])
        backtrack_ratios.append(
            rl_backtracks / baseline_backtracks
            if baseline_backtracks > 0
            else (1.0 if rl_backtracks == 0 else float("inf"))
        )
        backtrace_ratios.append(int(summary["backtrace_steps"]) / baseline_backtrace)
    mean_backtrack_ratio = sum(backtrack_ratios) / len(backtrack_ratios)
    mean_backtrace_ratio = sum(backtrace_ratios) / len(backtrace_ratios)
    score = (
        -detected,
        aborted,
        mean_backtrack_ratio,
        mean_backtrace_ratio,
        decisions,
    )
    return score, {
        "detected": detected,
        "aborted": aborted,
        "decisions": decisions,
        "mean_backtrack_ratio": mean_backtrack_ratio,
        "mean_backtrace_ratio": mean_backtrace_ratio,
        "score": list(score),
    }


def _run_validation(circuits, evaluators, label, seed):
    results = {}
    for item in circuits:
        summary = evaluators[item["name"]].run(
            item["circuit"],
            backtrack_limit=500,
            seed=seed,
            fault_ids=[str(fault["fault_id"]) for fault in item["validation_faults"]],
            quiet=True,
            fault_map_path=item["fault_map"],
        )
        results[item["name"]] = {key: int(value) for key, value in summary.items()}
    score, aggregate = _validation_score(circuits, results)
    return {
        "label": label,
        "aggregate": aggregate,
        "circuits": results,
    }, score


def _stage_fault_ids(item, stage_index, sweep, seed):
    allowed = DIFFICULTIES[: stage_index + 1]
    groups = {
        difficulty: [
            str(fault["fault_id"])
            for fault in item["training_faults"]
            if fault["difficulty"] == difficulty
        ]
        for difficulty in allowed
    }
    selected_count = min(len(group) for group in groups.values())
    selected = {}
    for difficulty, group in groups.items():
        ordered = list(group)
        random.Random(f"{seed}:{item['name']}:{difficulty}").shuffle(ordered)
        offset = ((sweep - 1) * selected_count) % len(ordered)
        selected[difficulty] = [
            ordered[(offset + index) % len(ordered)] for index in range(selected_count)
        ]
    return selected


def _checkpoint_state(
    agent,
    manifest_digest,
    config,
    pretraining,
    progress,
    validation_history,
    best_policy_state,
    best_score,
    best_label,
):
    return {
        "format": CHECKPOINT_FORMAT,
        "manifest_hash": manifest_digest,
        "config": config,
        "agent": agent.training_state_dict(),
        "pretraining": pretraining,
        "progress": progress,
        "validation_history": validation_history,
        "best_policy_state": best_policy_state,
        "best_score": list(best_score),
        "best_label": best_label,
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _agent_from_hyperparameters(embedding_dim, hyperparameters):
    return BacktracePPOAgentV2(
        gate_embedding_dim=embedding_dim,
        hidden_dim=32,
        **dict(hyperparameters),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Behavior-clone and curriculum-train the V4 backtrace actor."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("best_actor_output", type=Path)
    parser.add_argument("--latest-actor-output", type=Path)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--stage-sweeps", type=int, nargs=3, default=(2, 2, 3))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bc_epochs <= 0 or args.bc_batch_size <= 0:
        raise ValueError("Behavior-cloning settings must be positive.")
    if any(value <= 0 for value in args.stage_sweeps):
        raise ValueError("Every curriculum stage must have at least one sweep.")

    manifest = _load_json(args.manifest)
    circuits = _validate_manifest(manifest)
    embedding_tables = load_embedding_tables(circuits)
    embedding_dim = next(iter(next(iter(embedding_tables.values())).values())).numel()
    training_samples = _load_json(manifest["teacher_training"])
    validation_samples = filter_unseen_teacher_samples(
        training_samples, _load_json(manifest["teacher_validation"])
    )
    latest_actor_output = args.latest_actor_output or args.best_actor_output.with_name(
        args.best_actor_output.stem.replace("_best", "_latest")
        + args.best_actor_output.suffix
    )
    output_paths = {
        args.checkpoint.resolve(),
        args.best_actor_output.resolve(),
        latest_actor_output.resolve(),
    }
    if len(output_paths) != 3:
        raise ValueError("Checkpoint, best actor, and latest actor paths must differ.")
    bc_checkpoint = args.checkpoint.with_suffix(args.checkpoint.suffix + ".bc")

    config = {
        "seed": args.seed,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "stage_sweeps": list(args.stage_sweeps),
        "rnd_schedule": list(RND_SCHEDULE),
        "entropy_schedule": list(ENTROPY_SCHEDULE),
        "reward": dict(REWARD_CONFIG),
        "reward_distribution": REWARD_DISTRIBUTION,
        "gamma": 1.0,
        "normalize_returns": False,
        "return_scale": 100.0,
        "max_grad_norm": 1.0,
        "critic_initial_value": 1.0,
    }
    manifest_digest = _manifest_hash(args.manifest)
    torch.manual_seed(args.seed)

    state = None
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        _validate_checkpoint_metadata(state, manifest_digest, config)
        saved_hyperparameters = state["agent"]["hyperparameters"]
        agent = _agent_from_hyperparameters(embedding_dim, saved_hyperparameters)
        agent.load_training_state_dict(state["agent"])
        torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and state["torch_cuda_random_state"] is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])
        pretraining = state["pretraining"]
        progress = list(state["progress"])
        validation_history = list(state["validation_history"])
        best_policy_state = state["best_policy_state"]
        best_score = tuple(state["best_score"])
        best_label = str(state["best_label"])
        print(
            f"RESUME units={len(progress)} validations={len(validation_history)} "
            f"best={best_label}",
            flush=True,
        )
    else:
        agent = BacktracePPOAgentV2(
            gate_embedding_dim=embedding_dim,
            hidden_dim=32,
            gamma=1.0,
            rnd_beta=RND_SCHEDULE[0],
            normalize_returns=False,
            entropy_coef=ENTROPY_SCHEDULE[0],
            return_scale=100.0,
            max_grad_norm=1.0,
        )
        bc_resume_state = None
        if bc_checkpoint.exists():
            bc_state = torch.load(bc_checkpoint, map_location="cpu")
            if (
                bc_state.get("format") != CHECKPOINT_FORMAT + "_BC"
                or bc_state.get("manifest_hash") != manifest_digest
                or bc_state.get("config") != config
            ):
                raise ValueError("Behavior-cloning checkpoint configuration changed.")
            bc_resume_state = bc_state["pretraining_state"]
            print(
                f"BC_RESUME epoch={bc_resume_state['epoch']}/{args.bc_epochs}",
                flush=True,
            )

        def save_bc_epoch(pretraining_state):
            _atomic_torch_save(
                bc_checkpoint,
                {
                    "format": CHECKPOINT_FORMAT + "_BC",
                    "manifest_hash": manifest_digest,
                    "config": config,
                    "pretraining_state": pretraining_state,
                },
            )

        pretraining = pretrain_actor(
            agent,
            training_samples,
            validation_samples,
            embedding_tables,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            seed=args.seed,
            resume_state=bc_resume_state,
            epoch_callback=save_bc_epoch,
        )
        progress = []
        validation_history = []
        best_policy_state = _clone_policy_state(agent.policy_old.state_dict())
        best_score = (float("inf"),) * 5
        best_label = "none"
        print(
            f"BC_RESULT best_accuracy={pretraining['best_validation_accuracy']:.6f}",
            flush=True,
        )

    trainers = {
        item["name"]: CppPodemCurriculumTrainer(
            item["embeddings"],
            _baseline_map(item["training_faults"]),
            agent=agent,
        )
        for item in circuits
    }
    evaluators = {
        item["name"]: CppPodemBacktraceV2Evaluator(item["embeddings"], agent=agent)
        for item in circuits
    }

    validated_labels = {record["label"] for record in validation_history}
    if "behavior_cloning" not in validated_labels:
        validation, score = _run_validation(
            circuits, evaluators, "behavior_cloning", args.seed
        )
        validation_history.append(validation)
        if score < best_score:
            best_score = score
            best_label = "behavior_cloning"
            best_policy_state = _clone_policy_state(agent.policy_old.state_dict())
            export_actor_v2_state_dict(best_policy_state, args.best_actor_output)
        _atomic_torch_save(
            args.checkpoint,
            _checkpoint_state(
                agent,
                manifest_digest,
                config,
                pretraining,
                progress,
                validation_history,
                best_policy_state,
                best_score,
                best_label,
            ),
        )
        if bc_checkpoint.exists():
            bc_checkpoint.unlink()
        print(f"VALIDATION {json.dumps(validation, sort_keys=True)}", flush=True)

    completed = {
        (int(item["stage"]), int(item["sweep"]), item["circuit"], item["difficulty"])
        for item in progress
    }
    validated_labels = {record["label"] for record in validation_history}
    for stage_index, stage_name in enumerate(DIFFICULTIES):
        agent.rnd_beta = RND_SCHEDULE[stage_index]
        agent.entropy_coef = ENTROPY_SCHEDULE[stage_index]
        for sweep in range(1, args.stage_sweeps[stage_index] + 1):
            units = []
            for item in circuits:
                selected = _stage_fault_ids(item, stage_index, sweep, args.seed)
                for difficulty, fault_ids in selected.items():
                    units.append((item, difficulty, fault_ids))
            random.Random(f"{args.seed}:{stage_index}:{sweep}").shuffle(units)
            for item, difficulty, fault_ids in units:
                key = (stage_index, sweep, item["name"], difficulty)
                if key in completed:
                    continue
                trainer = trainers[item["name"]]
                trainer.set_exploration(
                    RND_SCHEDULE[stage_index], ENTROPY_SCHEDULE[stage_index]
                )
                print(
                    f"TRAIN_START stage={stage_name} sweep={sweep}/"
                    f"{args.stage_sweeps[stage_index]} circuit={item['name']} "
                    f"difficulty={difficulty} faults={len(fault_ids)}",
                    flush=True,
                )
                summary = trainer.run(
                    item["circuit"],
                    backtrack_limit=500,
                    seed=args.seed + stage_index * 100 + sweep,
                    fault_ids=fault_ids,
                    quiet=True,
                    fault_map_path=item["fault_map"],
                )
                record = {
                    "stage": stage_index,
                    "stage_name": stage_name,
                    "sweep": sweep,
                    "circuit": item["name"],
                    "difficulty": difficulty,
                    "summary": {key: int(value) for key, value in summary.items()},
                    "learning": trainer.run_metrics,
                }
                progress.append(record)
                completed.add(key)
                export_actor_v2_state_dict(
                    agent.policy_old.state_dict(), latest_actor_output
                )
                _atomic_torch_save(
                    args.checkpoint,
                    _checkpoint_state(
                        agent,
                        manifest_digest,
                        config,
                        pretraining,
                        progress,
                        validation_history,
                        best_policy_state,
                        best_score,
                        best_label,
                    ),
                )
                print(f"TRAIN_RESULT {json.dumps(record, sort_keys=True)}", flush=True)

            label = f"{stage_name}_sweep_{sweep}"
            if label not in validated_labels:
                validation, score = _run_validation(
                    circuits, evaluators, label, args.seed
                )
                validation_history.append(validation)
                validated_labels.add(label)
                if score < best_score:
                    best_score = score
                    best_label = label
                    best_policy_state = _clone_policy_state(
                        agent.policy_old.state_dict()
                    )
                    export_actor_v2_state_dict(
                        best_policy_state, args.best_actor_output
                    )
                    selection = "best"
                else:
                    selection = "kept"
                _atomic_torch_save(
                    args.checkpoint,
                    _checkpoint_state(
                        agent,
                        manifest_digest,
                        config,
                        pretraining,
                        progress,
                        validation_history,
                        best_policy_state,
                        best_score,
                        best_label,
                    ),
                )
                print(
                    f"VALIDATION selection={selection} "
                    f"{json.dumps(validation, sort_keys=True)}",
                    flush=True,
                )

    export_actor_v2_state_dict(best_policy_state, args.best_actor_output)
    export_actor_v2_state_dict(agent.policy_old.state_dict(), latest_actor_output)
    print(
        f"TRAINING_COMPLETE best={best_label} score={list(best_score)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

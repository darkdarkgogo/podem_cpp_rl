"""Train versioned 11D SmartATPG and select the best model on validation."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch

from prepare_smartatpg_training import (
    BACKTRACK_LIMIT,
    FAULTS_PER_UPDATE,
    HEURISTIC,
    LEGACY_MANIFEST_FORMAT,
    LEGACY_TRAINING_ROUNDS,
    LAZY_VALIDATION_MANIFEST_FORMAT,
    MANIFEST_FORMAT,
    NORMAL_TRAINING_ROUNDS,
    PPO_EPOCHS_PER_UPDATE,
    _validate_manifest as _validate_prepared_manifest,
    resolve_manifest_path,
    sha256_file,
    validation_fault_ids,
)
from rl_podem.cpp_bridge import (
    CppPodemBacktraceV2Evaluator,
    CppPodemBacktraceV2Trainer,
    catalog_cpp_podem,
    smartatpg_pi_reward,
)
from rl_podem.gat_gru import GATGRUSmartATPGPPOAgent
from rl_podem.ppo import device
from rl_podem.smartatpg import SmartATPGPPOAgent
from rl_podem.smartatpg_artifacts import export_actor
from rl_podem.smartatpg_features import load_circuit_graph


CHECKPOINT_FORMAT = "SMARTATPG_DATA_SPLIT_TRAINING_V6_11D_CO_NO_BUF"
BEST_CHECKPOINT_FORMAT = "SMARTATPG_DATA_SPLIT_BEST_V6_11D_CO_NO_BUF"
VALIDATION_STATE_FORMAT = "SMARTATPG_DATA_SPLIT_VALIDATION_STATE_V2_JSONL"
AGENT_TYPES = {
    "fanin_mean": SmartATPGPPOAgent,
    "level_gat_gru": GATGRUSmartATPGPPOAgent,
}
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


def _atomic_json_lines(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def _append_json_line(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validation_score(summary, round_number):
    return (
        -int(summary["detected_faults"]),
        int(summary["backtracks_total"]),
        int(summary["backtrace_steps_total"]),
        -float(summary["return_total"]),
        int(round_number),
    )


def _resolve_circuit_records(manifest, manifest_path):
    _validate_prepared_manifest(manifest, manifest_path)
    result = {}
    for split, key in (
        ("train", "train_circuits"),
        ("validation", "validation_circuits"),
    ):
        records = []
        for raw in manifest[key]:
            item = dict(raw)
            item["circuit"] = str(
                resolve_manifest_path(manifest_path, raw["circuit"])
            )
            if "profile" in raw:
                item["profile"] = str(
                    resolve_manifest_path(manifest_path, raw["profile"])
                )
            records.append(item)
        result[split] = records
    names = [item["name"] for split in result.values() for item in split]
    if len(names) != len(set(names)):
        raise ValueError("Training and validation circuit names must be disjoint")
    return result["train"], result["validation"]


def _catalog_fault_ids(circuit_path):
    return validation_fault_ids(catalog_cpp_podem(circuit_path), circuit_path)


def _load_validation_catalogs(manifest, circuits):
    if manifest.get("format") == LEGACY_MANIFEST_FORMAT:
        return circuits
    for index, item in enumerate(circuits, 1):
        item["episode_fault_ids"] = _catalog_fault_ids(item["circuit"])
        print(
            f"CATALOG split=validation index={index}/{len(circuits)} "
            f"circuit={item['name']} faults={len(item['episode_fault_ids'])}",
            flush=True,
        )
    return circuits


def _validation_catalog_hash(circuits):
    payload = [
        {"name": item["name"], "fault_ids": item["episode_fault_ids"]}
        for item in circuits
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fault_update_boundary(next_index, total_faults, batch_size):
    if batch_size <= 0 or not 1 <= next_index <= total_faults:
        raise ValueError("Fault-update boundary arguments are invalid")
    return next_index == total_faults or next_index % batch_size == 0


def _episode_order(circuits, seed, round_number):
    episodes = [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["episode_fault_ids"]
    ]
    random.Random(seed + round_number).shuffle(episodes)
    return episodes


def _validation_order(circuits):
    return [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["episode_fault_ids"]
    ]


def _evaluate_fault(evaluator, item, fault_id, backtrack_limit, seed):
    extrinsic_return = 0.0
    terminal = None

    def event_callback(event):
        nonlocal extrinsic_return, terminal
        if event["event"] == "backtrace_step":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if decision_sequences is None or int(event["decision_sequence"]) in decision_sequences:
                extrinsic_return += PAPER_REWARD["non_pi"]
        elif event["event"] == "pi_not_done":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if decision_sequences is None or int(event["decision_sequence"]) in decision_sequences:
                extrinsic_return += smartatpg_pi_reward(
                    int(event["backtracks"]),
                    int(event["pi_visits"]),
                    PAPER_REWARD["alpha"],
                    PAPER_REWARD["beta"],
                )
        elif event["event"] == "episode_end":
            if terminal is not None:
                raise RuntimeError("Validation fault produced multiple terminal events")
            terminal = dict(event)
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
        use_scoap=True,
        event_callback=event_callback,
    )
    if int(summary["episodes"]) != 1 or terminal is None:
        raise RuntimeError("Validation fault did not produce exactly one episode")
    if terminal.get("fault_id") != fault_id:
        raise RuntimeError("Validation terminal fault does not match the request")
    return {
        "circuit": item["name"],
        "fault_id": fault_id,
        "outcome": int(terminal["outcome"]),
        "detected": int(int(terminal["outcome"]) == 1),
        "backtracks": int(terminal["backtracks"]),
        "backtrace_steps": int(terminal["backtrace_steps"]),
        "return": float(extrinsic_return),
    }


def _summarize_validation(records, circuits, round_number):
    expected = _validation_order(circuits)
    actual = [(item.get("circuit"), item.get("fault_id")) for item in records]
    if actual != expected:
        raise ValueError("Validation must cover the full fault catalog exactly once")
    totals = {
        "episodes": len(records),
        "detected_faults": sum(int(item["detected"]) for item in records),
        "backtracks_total": sum(int(item["backtracks"]) for item in records),
        "backtrace_steps_total": sum(
            int(item["backtrace_steps"]) for item in records
        ),
        "return_total": sum(float(item["return"]) for item in records),
    }
    count = max(1, totals["episodes"])
    totals.update(
        fault_coverage=totals["detected_faults"] / count,
        backtracks_mean=totals["backtracks_total"] / count,
        backtrace_steps_mean=totals["backtrace_steps_total"] / count,
        return_mean=totals["return_total"] / count,
    )
    return {"round": round_number, **totals}


def _save_state(path, agent, state):
    payload = dict(state)
    payload["agent"] = agent.training_state_dict()
    payload["torch_random_state"] = torch.get_rng_state()
    payload["torch_cuda_random_state"] = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    _atomic_torch_save(path, payload)


def _restore_torch_rng(saved):
    if "torch_random_state" not in saved:
        raise ValueError("Checkpoint is missing the PyTorch random state")
    torch.set_rng_state(saved["torch_random_state"])
    if saved.get("torch_cuda_random_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["torch_cuda_random_state"])


def _training_protocol(config):
    protocol = {
        "manifest_hash": config["manifest_hash"],
        "backtrack_limit": config["backtrack_limit"],
        "normal_rounds": config["rounds"],
        "training_circuit_count": config["training_circuit_count"],
        "validation_circuit_count": config["validation_circuit_count"],
    }
    if "faults_per_update" in config:
        protocol["faults_per_update"] = config["faults_per_update"]
        protocol["k_epochs"] = config["k_epochs"]
    return protocol


def _initial_state(manifest_digest, config, continuation=None):
    return {
        "format": CHECKPOINT_FORMAT,
        "manifest_hash": manifest_digest,
        "config": config,
        "phase": "training",
        "current_round": 1,
        "episode_index": 0,
        "completed_episodes": 0,
        "validation_metrics": [],
        "best_score": None,
        "best_round": None,
        "best_agent": None,
        "continuation": continuation,
    }


def _validate_resume(saved, manifest_digest, config):
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is incompatible with data-split SmartATPG training")
    if saved.get("manifest_hash") != manifest_digest:
        raise ValueError("Training manifest changed since checkpoint")
    if saved.get("config") != config:
        raise ValueError("Training configuration changed since checkpoint")
    if saved.get("phase") not in ("training", "validation"):
        raise ValueError("Checkpoint phase is invalid")
    current_round = int(saved.get("current_round", 0))
    episode_index = int(saved.get("episode_index", -1))
    rounds = int(config["rounds"])
    if not 1 <= current_round <= rounds + 1 or episode_index < 0:
        raise ValueError("Checkpoint training position is invalid")
    phase = saved["phase"]
    episodes_per_round = int(config["training_episode_count"])
    if episode_index > episodes_per_round:
        raise ValueError("Checkpoint episode position exceeds the training fault set")
    if phase == "validation" and episode_index != 0:
        raise ValueError("Validation checkpoint must not contain a training position")
    if current_round == rounds + 1 and (
        phase != "training" or episode_index != 0
    ):
        raise ValueError("Completed checkpoint has an invalid phase")
    expected_completed = (current_round - 1) * episodes_per_round
    if phase == "training":
        expected_completed += episode_index
    else:
        expected_completed += episodes_per_round
    if int(saved.get("completed_episodes", -1)) != expected_completed:
        raise ValueError("Checkpoint completed-episode count is inconsistent")
    batch_size = int(config.get("faults_per_update", 1))
    if (
        phase == "training"
        and episode_index not in (0, episodes_per_round)
        and episode_index % batch_size != 0
    ):
        raise ValueError("Checkpoint is not at a fault-update boundary")
    return saved


def _load_continuation(path, agent):
    path = Path(path).resolve()
    saved = torch.load(path, map_location="cpu")
    if saved.get("format") not in (CHECKPOINT_FORMAT, BEST_CHECKPOINT_FORMAT):
        raise ValueError("Continuation requires a current data-split 11D checkpoint")
    if not isinstance(saved.get("agent"), dict):
        raise ValueError("Continuation checkpoint has no complete agent state")
    agent.load_training_state_dict(saved["agent"])
    _restore_torch_rng(saved)
    return {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": sha256_file(path),
        "source_manifest_hash": saved.get("manifest_hash"),
        "source_format": saved["format"],
    }


def _load_validation_state(path, records_path, manifest_digest, round_number):
    path = Path(path)
    records_path = Path(records_path)
    fresh = {
        "format": VALIDATION_STATE_FORMAT,
        "manifest_hash": manifest_digest,
        "round": round_number,
        "next_index": 0,
        "complete": False,
    }
    if not path.is_file():
        _atomic_json_lines(records_path, [])
        _atomic_json(path, fresh)
        return fresh, []
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("round") != round_number and state.get("complete") is True:
        _atomic_json_lines(records_path, [])
        _atomic_json(path, fresh)
        return fresh, []
    if state.get("round") != round_number:
        raise ValueError("Incomplete validation state belongs to another round")
    expected = {
        "format": VALIDATION_STATE_FORMAT,
        "manifest_hash": manifest_digest,
        "round": round_number,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Validation resume state is incompatible")
    if not records_path.is_file():
        raise ValueError("Validation resume records are missing")
    lines = records_path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError("Validation resume records are corrupted") from error
            _atomic_json_lines(records_path, records)
    next_index = int(state.get("next_index", -1))
    if next_index < 0 or len(records) < next_index:
        raise ValueError("Validation resume position is invalid")
    if len(records) > next_index:
        state["next_index"] = len(records)
        _atomic_json(path, state)
    return state, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rnd-beta", type=float, default=0.05)
    parser.add_argument("--k-epochs", type=int)
    parser.add_argument(
        "--encoder", choices=tuple(AGENT_TYPES), default="level_gat_gru"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--continue-from", type=Path)
    args = parser.parse_args(argv)
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    supported_formats = (
        LEGACY_MANIFEST_FORMAT,
        LAZY_VALIDATION_MANIFEST_FORMAT,
        MANIFEST_FORMAT,
    )
    if manifest.get("format") not in supported_formats:
        raise ValueError("Training requires a supported data-split manifest")
    batched_training = manifest.get("format") == MANIFEST_FORMAT
    expected_rounds = (
        NORMAL_TRAINING_ROUNDS if batched_training else LEGACY_TRAINING_ROUNDS
    )
    if args.rounds is None:
        args.rounds = expected_rounds
    if args.k_epochs is None:
        args.k_epochs = PPO_EPOCHS_PER_UPDATE if batched_training else 8
    if args.rounds != expected_rounds:
        raise ValueError(
            f"This SmartATPG manifest requires exactly {expected_rounds} rounds"
        )
    if args.k_epochs <= 0:
        raise ValueError("PPO epochs must be positive")
    if batched_training and args.k_epochs != PPO_EPOCHS_PER_UPDATE:
        raise ValueError(
            f"V8 SmartATPG training requires k_epochs={PPO_EPOCHS_PER_UPDATE}"
        )
    train_circuits, validation_circuits = _resolve_circuit_records(
        manifest, args.manifest
    )
    validation_circuits = _load_validation_catalogs(
        manifest, validation_circuits
    )
    if int(manifest["normal_rounds"]) != args.rounds:
        raise ValueError("Requested rounds do not match the training manifest")
    if int(manifest["backtrack_limit"]) != BACKTRACK_LIMIT:
        raise ValueError(f"Training requires backtrack limit {BACKTRACK_LIMIT}")

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "training_state.pth"
    best_checkpoint_path = output_dir / "best_training_state.pth"
    model_best_path = output_dir / "model_best.txt"
    model_latest_path = output_dir / "model_latest.txt"
    metrics_path = output_dir / "validation_metrics.json"
    validation_state_path = output_dir / "validation_state.json"
    validation_records_path = output_dir / "validation_records.jsonl"
    if args.continue_from and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("--continue-from requires an empty output directory")
    if (
        args.resume
        and output_dir.exists()
        and any(output_dir.iterdir())
        and not checkpoint_path.is_file()
    ):
        raise FileExistsError("--resume requires training_state.pth in a non-empty directory")
    if (
        not args.resume
        and not args.continue_from
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError("Fresh training requires an empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    all_circuits = [*train_circuits, *validation_circuits]
    graphs = {
        item["name"]: load_circuit_graph(item["circuit"]) for item in all_circuits
    }
    agent = AGENT_TYPES[args.encoder](
        graphs,
        hidden_dim=32,
        lr_actor=0.001,
        lr_critic=0.01,
        rnd_beta=args.rnd_beta,
        k_epochs=args.k_epochs,
    )
    trainers = {
        item["name"]: CppPodemBacktraceV2Trainer(
            graphs[item["name"]], agent=agent,
            auto_update=not batched_training,
        )
        for item in train_circuits
    }
    evaluators = {
        item["name"]: CppPodemBacktraceV2Evaluator(graphs[item["name"]], agent=agent)
        for item in validation_circuits
    }

    manifest_digest = _manifest_hash(args.manifest)
    config = {
        "rounds": args.rounds,
        "seed": args.seed,
        "rnd_beta": args.rnd_beta,
        "k_epochs": args.k_epochs,
        "backtrack_limit": BACKTRACK_LIMIT,
        "actor_lr": 0.001,
        "critic_lr": 0.01,
        "training_episode_count": sum(
            len(item["episode_fault_ids"]) for item in train_circuits
        ),
        "validation_episode_count": sum(
            len(item["episode_fault_ids"]) for item in validation_circuits
        ),
        "training_circuit_count": len(train_circuits),
        "validation_circuit_count": len(validation_circuits),
        "device": str(device),
        "paper_reward": PAPER_REWARD,
        "encoder_variant": args.encoder,
        "heuristic": HEURISTIC,
        "manifest_hash": manifest_digest,
    }
    if manifest.get("format") != LEGACY_MANIFEST_FORMAT:
        config["validation_catalog_hash"] = _validation_catalog_hash(
            validation_circuits
        )
    if batched_training:
        config["faults_per_update"] = FAULTS_PER_UPDATE
    state = _initial_state(manifest_digest, config)
    if args.resume:
        if not checkpoint_path.is_file():
            print("RESUME requested without checkpoint; starting a fresh run", flush=True)
        else:
            saved = _validate_resume(
                torch.load(checkpoint_path, map_location="cpu"), manifest_digest, config
            )
            agent.load_training_state_dict(saved["agent"])
            state.update({key: saved[key] for key in state})
            _restore_torch_rng(saved)
            print(
                f"RESUME phase={state['phase']} round={state['current_round']} "
                f"episode={state['episode_index']} total={state['completed_episodes']}",
                flush=True,
            )
    elif args.continue_from:
        continuation = _load_continuation(args.continue_from, agent)
        state = _initial_state(manifest_digest, config, continuation=continuation)
        print(
            f"CONTINUE_FROM checkpoint={continuation['source_checkpoint']} ",
            flush=True,
        )

    if not checkpoint_path.is_file():
        _save_state(checkpoint_path, agent, state)
    if state["continuation"] is not None:
        _atomic_json(output_dir / "continuation.json", state["continuation"])

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("Install tensorboard before SmartATPG training") from error
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    train_by_name = {item["name"]: item for item in train_circuits}
    validation_by_name = {item["name"]: item for item in validation_circuits}
    training_protocol = _training_protocol(config)
    export_actor(
        agent.policy_old.state_dict(),
        model_latest_path,
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
            if state["phase"] == "training":
                order = _episode_order(train_circuits, args.seed, round_number)
                for index in range(int(state["episode_index"]), len(order)):
                    circuit_name, fault_id = order[index]
                    item = train_by_name[circuit_name]
                    trainer = trainers[circuit_name]
                    trainer.run(
                        item["circuit"],
                        backtrack_limit=BACKTRACK_LIMIT,
                        seed=args.seed + round_number,
                        fault_ids=[fault_id],
                        use_scoap=True,
                    )
                    if len(trainer.episode_metrics) != 1:
                        raise RuntimeError("Training episode did not produce one metric")
                    metrics = trainer.episode_metrics[0]
                    state["episode_index"] = index + 1
                    state["completed_episodes"] += 1
                    step = int(state["completed_episodes"])
                    episode_keys = [
                        "backtracks", "backtrace_steps", "detected",
                        "extrinsic_reward_sum", "scaled_intrinsic_reward_sum",
                        "combined_reward_sum",
                    ]
                    if not batched_training:
                        episode_keys.extend(("total_loss", "rnd_loss"))
                    for key in episode_keys:
                        writer.add_scalar(f"episode/{key}", metrics[key], step)
                    writer.flush()
                    print(
                        f"EPISODE round={round_number}/{args.rounds} "
                        f"index={index + 1}/{len(order)} circuit={circuit_name} "
                        f"fault={fault_id} backtracks={metrics['backtracks']} "
                        f"backtrace_steps={metrics['backtrace_steps']}",
                        flush=True,
                    )
                    update_boundary = _fault_update_boundary(
                        index + 1, len(order),
                        FAULTS_PER_UPDATE if batched_training else 1,
                    )
                    if batched_training and update_boundary:
                        update_metrics = agent.update()
                        batch_faults = (index + 1) % FAULTS_PER_UPDATE
                        if batch_faults == 0:
                            batch_faults = FAULTS_PER_UPDATE
                        update_step = (
                            int(agent.update_count)
                            if update_metrics is not None
                            else (index + 1 + FAULTS_PER_UPDATE - 1)
                            // FAULTS_PER_UPDATE
                        )
                        if update_metrics is not None:
                            for key in (
                                "total_loss", "policy_loss", "value_loss",
                                "entropy", "rnd_loss", "steps",
                            ):
                                writer.add_scalar(
                                    f"update/{key}", update_metrics[key],
                                    update_step,
                                )
                            writer.flush()
                        print(
                            f"UPDATE round={round_number}/{args.rounds} "
                            f"faults={batch_faults} completed={index + 1}/"
                            f"{len(order)} optimizer_step="
                            f"{int(update_metrics is not None)}",
                            flush=True,
                        )
                    if not batched_training or update_boundary:
                        export_actor(
                            agent.policy_old.state_dict(),
                            model_latest_path,
                            training_protocol=training_protocol,
                        )
                        _save_state(checkpoint_path, agent, state)
                state["phase"] = "validation"
                state["episode_index"] = 0
                _save_state(checkpoint_path, agent, state)

            validation_order = _validation_order(validation_circuits)
            validation_state, validation_records = _load_validation_state(
                validation_state_path,
                validation_records_path,
                manifest_digest,
                round_number,
            )
            completed_validation = [
                (record.get("circuit"), record.get("fault_id"))
                for record in validation_records
            ]
            if completed_validation != validation_order[:len(completed_validation)]:
                raise ValueError("Validation resume records are not the expected prefix")
            if int(validation_state["next_index"]) > len(validation_order):
                raise ValueError("Validation resume position exceeds the fault catalog")
            for index in range(int(validation_state["next_index"]), len(validation_order)):
                circuit_name, fault_id = validation_order[index]
                record = _evaluate_fault(
                    evaluators[circuit_name],
                    validation_by_name[circuit_name],
                    fault_id,
                    BACKTRACK_LIMIT,
                    args.seed,
                )
                _append_json_line(validation_records_path, record)
                validation_records.append(record)
                validation_state["next_index"] = index + 1
                _atomic_json(validation_state_path, validation_state)
                print(
                    f"VALIDATE round={round_number}/{args.rounds} "
                    f"index={index + 1}/{len(validation_order)} "
                    f"circuit={circuit_name} fault={fault_id} "
                    f"outcome={record['outcome']}",
                    flush=True,
                )

            evaluation = _summarize_validation(
                validation_records, validation_circuits, round_number
            )
            score = validation_score(evaluation, round_number)
            is_best = state["best_score"] is None or score < tuple(state["best_score"])
            evaluation["is_best"] = bool(is_best)
            state["validation_metrics"].append(evaluation)
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
                    "validation": evaluation,
                    "agent": state["best_agent"],
                    "torch_random_state": torch.get_rng_state(),
                    "torch_cuda_random_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None
                    ),
                    "continuation": state["continuation"],
                }
                _atomic_torch_save(best_checkpoint_path, best_payload)
                export_actor(
                    state["best_agent"]["policy_old"],
                    model_best_path,
                    best_round=round_number,
                    best_score=score,
                    training_protocol=training_protocol,
                )
            for key in (
                "backtracks_total", "backtracks_mean", "backtrace_steps_total",
                "backtrace_steps_mean", "return_total", "return_mean",
                "detected_faults", "fault_coverage",
            ):
                writer.add_scalar(f"validation/{key}", evaluation[key], round_number)
            writer.add_scalar("validation/is_best", int(is_best), round_number)
            writer.flush()
            _atomic_json(metrics_path, state["validation_metrics"])
            validation_state["complete"] = True
            _atomic_json(validation_state_path, validation_state)
            state["current_round"] = round_number + 1
            state["episode_index"] = 0
            state["phase"] = "training"
            _save_state(checkpoint_path, agent, state)
            print(
                f"ROUND round={round_number}/{args.rounds} "
                f"validation_detected={evaluation['detected_faults']}/"
                f"{evaluation['episodes']} backtracks={evaluation['backtracks_total']} "
                f"backtrace_steps={evaluation['backtrace_steps_total']} "
                f"best={int(is_best)}",
                flush=True,
            )
    finally:
        writer.close()

    export_actor(
        agent.policy_old.state_dict(),
        model_latest_path,
        training_protocol=training_protocol,
    )
    print(
        f"TRAINING_COMPLETE rounds={args.rounds} "
        f"episodes={state['completed_episodes']} best_round={state['best_round']} "
        f"device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()

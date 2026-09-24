"""Train one SmartATPG encoder and persist validation-independent rounds."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from .artifact_io import (
    INFERENCE_CHECKPOINT_FORMAT,
    atomic_json as _atomic_json,
    manifest_hash as _manifest_hash,
)
from .data_split import (
    BACKTRACK_LIMIT,
    FAULTS_PER_UPDATE,
    HEURISTIC,
    LEGACY_MANIFEST_FORMAT,
    LEGACY_TRAINING_ROUNDS,
    LAZY_VALIDATION_MANIFEST_FORMAT,
    MANIFEST_FORMAT,
    NORMAL_TRAINING_ROUNDS,
    _validate_manifest as _validate_prepared_manifest,
    resolve_manifest_path,
    sha256_file,
    training_hyperparameters,
    validation_fault_ids,
)
from .cpp_bridge import (
    _native_circuit_path,
    CppPodemBacktraceV2Trainer,
    catalog_cpp_podem,
)
from .smartatpg_rewards import (
    BACKTRACK_MAX,
    GAT_REWARD_SCHEME,
    MEAN_REWARD_SCHEME,
    reward_scheme_for_encoder,
    smartatpg_backtrack_reward,
    smartatpg_pi_reward,
)
from .gat_gru import GATGRUSmartATPGPPOAgent
from .ppo import device
from .smartatpg import SmartATPGPPOAgent
from .smartatpg_artifacts import export_actor
from .smartatpg_features import load_circuit_graph


CHECKPOINT_FORMAT = "SMARTATPG_DATA_SPLIT_TRAINING_V6_11D_CO_NO_BUF"
BEST_CHECKPOINT_FORMAT = "SMARTATPG_DATA_SPLIT_BEST_V6_11D_CO_NO_BUF"
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


def _native_validation_batch(
    item, fault_ids, embedding_path, actor_path, records_path, seed,
    reward_scheme,
):
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Native validation requires the rebuilt cpp_podem extension"
        ) from error
    if not hasattr(cpp_podem, "run_native_validation"):
        raise RuntimeError(
            "cpp_podem is stale; rebuild it with: python -m pip install -e ."
        )
    native_records = cpp_podem.run_native_validation(
        _native_circuit_path(item["circuit"]),
        _native_circuit_path(embedding_path),
        _native_circuit_path(actor_path),
        BACKTRACK_LIMIT, seed, fault_ids, reward_scheme,
        _native_circuit_path(records_path), item["name"],
    )
    if len(native_records) != len(fault_ids):
        raise RuntimeError("Native validation returned the wrong fault count")
    records = []
    for fault_id, raw in zip(fault_ids, native_records):
        if raw["fault_id"] != fault_id:
            raise RuntimeError("Native validation returned faults out of order")
        outcome = int(raw["outcome"])
        if not math.isfinite(float(raw["return"])):
            raise ValueError(
                f"Non-finite validation return for {item['name']} {fault_id}"
            )
        if not math.isfinite(float(raw["atpg_seconds"])):
            raise ValueError(
                f"Non-finite validation time for {item['name']} {fault_id}"
            )
        records.append({
            "circuit": item["name"],
            "fault_id": fault_id,
            "outcome": outcome,
            "detected": int(outcome == 1),
            "redundant": int(outcome == 0),
            "aborted": int(outcome not in (0, 1)),
            "backtracks": int(raw["backtracks"]),
            "backtrace_steps": int(raw["backtrace_steps"]),
            "return": float(raw["return"]),
            "test_vectors": int(outcome == 1),
            "atpg_seconds": float(raw["atpg_seconds"]),
        })
    return records


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
    records = []
    for raw in manifest["train_circuits"]:
        item = dict(raw)
        item["circuit"] = str(
            resolve_manifest_path(manifest_path, raw["circuit"])
        )
        item["profile"] = str(
            resolve_manifest_path(manifest_path, raw["profile"])
        )
        if "fault_map" in raw:
            item["fault_map"] = str(
                resolve_manifest_path(manifest_path, raw["fault_map"])
            )
        records.append(item)
    return records


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


def _evaluate_fault(
    evaluator, item, fault_id, backtrack_limit, seed, reward_scheme,
):
    if backtrack_limit != BACKTRACK_MAX:
        raise ValueError(
            f"SmartATPG validation requires backtrack_limit={BACKTRACK_MAX}"
        )
    if reward_scheme not in (GAT_REWARD_SCHEME, MEAN_REWARD_SCHEME):
        raise ValueError(f"Unknown SmartATPG reward scheme: {reward_scheme}")
    agent = getattr(evaluator, "agent", None)
    if agent is not None:
        expected_scheme = reward_scheme_for_encoder(agent.encoder_variant)
        if reward_scheme != expected_scheme:
            raise ValueError(
                "SmartATPG reward scheme does not match evaluator encoder"
            )
    extrinsic_return = 0.0
    terminal = None
    backtrack_count = 0

    def event_callback(event):
        nonlocal extrinsic_return, terminal, backtrack_count
        if event["event"] == "backtrace_step":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if decision_sequences is None or int(event["decision_sequence"]) in decision_sequences:
                extrinsic_return += PAPER_REWARD["non_pi"]
        elif event["event"] == "backtrack":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if (
                reward_scheme == GAT_REWARD_SCHEME
                and (
                    decision_sequences is None
                    or int(event["decision_sequence"]) in decision_sequences
                )
            ):
                backtrack_count += 1
                extrinsic_return += smartatpg_backtrack_reward(backtrack_count)
        elif event["event"] == "pi_not_done":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if (
                reward_scheme == MEAN_REWARD_SCHEME
                and (
                    decision_sequences is None
                    or int(event["decision_sequence"]) in decision_sequences
                )
            ):
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
    outcome = int(terminal["outcome"])
    if not math.isfinite(extrinsic_return):
        raise ValueError(
            f"Non-finite validation return for {item['name']} {fault_id}"
        )
    atpg_seconds = float(summary["atpg_seconds"])
    if not math.isfinite(atpg_seconds):
        raise ValueError(
            f"Non-finite validation time for {item['name']} {fault_id}"
        )
    return {
        "circuit": item["name"],
        "fault_id": fault_id,
        "outcome": outcome,
        "detected": int(outcome == 1),
        "redundant": int(outcome == 0),
        "aborted": int(outcome not in (0, 1)),
        "backtracks": int(terminal["backtracks"]),
        "backtrace_steps": int(terminal["backtrace_steps"]),
        "return": float(extrinsic_return),
        "test_vectors": int(outcome == 1),
        "atpg_seconds": atpg_seconds,
    }


def _summarize_fault_records(records):
    for item in records:
        if not math.isfinite(float(item["return"])):
            raise ValueError(
                "Non-finite validation return for "
                f"{item.get('circuit', '<unknown>')} "
                f"{item.get('fault_id', '<unknown>')}"
            )
        if not math.isfinite(float(item["atpg_seconds"])):
            raise ValueError(
                "Non-finite validation time for "
                f"{item.get('circuit', '<unknown>')} "
                f"{item.get('fault_id', '<unknown>')}"
            )
    count = len(records)
    totals = {
        "episodes": count,
        "detected_faults": sum(int(item["detected"]) for item in records),
        "redundant_faults": sum(int(item["redundant"]) for item in records),
        "aborted_faults": sum(int(item["aborted"]) for item in records),
        "backtracks_total": sum(int(item["backtracks"]) for item in records),
        "backtrace_steps_total": sum(
            int(item["backtrace_steps"]) for item in records
        ),
        "return_total": sum(float(item["return"]) for item in records),
        "test_vectors": sum(int(item["test_vectors"]) for item in records),
        "atpg_seconds": sum(float(item["atpg_seconds"]) for item in records),
    }
    divisor = max(1, count)
    totals.update(
        fault_coverage=totals["detected_faults"] / divisor,
        backtracks_mean=totals["backtracks_total"] / divisor,
        backtrace_steps_mean=totals["backtrace_steps_total"] / divisor,
        return_mean=totals["return_total"] / divisor,
    )
    if not all(math.isfinite(float(totals[key])) for key in (
        "return_total", "return_mean",
    )):
        raise ValueError("Non-finite validation return summary")
    return totals


def _summarize_validation(records, circuits, round_number):
    expected = _validation_order(circuits)
    actual = [(item.get("circuit"), item.get("fault_id")) for item in records]
    if actual != expected:
        raise ValueError("Validation must cover the full fault catalog exactly once")
    totals = _summarize_fault_records(records)
    per_circuit = [
        {
            "circuit": item["name"],
            **_summarize_fault_records([
                record for record in records if record["circuit"] == item["name"]
            ]),
        }
        for item in circuits
    ]
    return {"round": round_number, **totals, "circuits": per_circuit}


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
        "reward_scheme": config["reward_scheme"],
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
        "continuation": continuation,
    }


def _validate_resume(saved, manifest_digest, config):
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is incompatible with data-split SmartATPG training")
    if saved.get("manifest_hash") != manifest_digest:
        raise ValueError("Training manifest changed since checkpoint")
    if saved.get("config") != config:
        raise ValueError("Training configuration changed since checkpoint")
    if saved.get("phase") != "training":
        raise ValueError("Checkpoint phase is invalid")
    current_round = int(saved.get("current_round", 0))
    episode_index = int(saved.get("episode_index", -1))
    rounds = int(config["rounds"])
    if not 1 <= current_round <= rounds + 1 or episode_index < 0:
        raise ValueError("Checkpoint training position is invalid")
    episodes_per_round = int(config["training_episode_count"])
    if episode_index > episodes_per_round:
        raise ValueError("Checkpoint episode position exceeds the training fault set")
    if current_round == rounds + 1 and (
        episode_index != 0
    ):
        raise ValueError("Completed checkpoint has an invalid phase")
    expected_completed = (current_round - 1) * episodes_per_round
    expected_completed += episode_index
    if int(saved.get("completed_episodes", -1)) != expected_completed:
        raise ValueError("Checkpoint completed-episode count is inconsistent")
    batch_size = int(config.get("faults_per_update", 1))
    if (
        episode_index not in (0, episodes_per_round)
        and episode_index % batch_size != 0
    ):
        raise ValueError("Checkpoint is not at a fault-update boundary")
    return saved


def _inference_payload(agent, config, round_number):
    """Return the complete CPU inference state needed for later validation."""
    return {
        "format": INFERENCE_CHECKPOINT_FORMAT,
        "manifest_hash": config["manifest_hash"],
        "encoder_variant": config["encoder_variant"],
        "reward_scheme": config["reward_scheme"],
        "backtrack_limit": config["backtrack_limit"],
        "seed": config["seed"],
        "round": int(round_number),
        "policy_old": _clone(agent.policy_old.state_dict()),
    }


def _save_inference_artifacts(output_dir, agent, config, round_number, *, final=False):
    suffix = "final" if final else f"round_{int(round_number):02d}"
    state = agent.policy_old.state_dict()
    _atomic_torch_save(
        Path(output_dir) / f"inference_{suffix}.pth",
        _inference_payload(agent, config, round_number),
    )
    export_actor(
        state,
        Path(output_dir) / f"model_{suffix}.txt",
        training_protocol=_training_protocol(config),
    )


def _load_continuation(path, agent, expected_config):
    path = Path(path).resolve()
    saved = torch.load(path, map_location="cpu")
    if saved.get("format") not in (CHECKPOINT_FORMAT, BEST_CHECKPOINT_FORMAT):
        raise ValueError("Continuation requires a current data-split 11D checkpoint")
    if not isinstance(saved.get("agent"), dict):
        raise ValueError("Continuation checkpoint has no complete agent state")
    saved_config = saved.get("config")
    protocol_keys = ("encoder_variant", "reward_scheme", "backtrack_limit")
    if not isinstance(saved_config, dict) or any(
        saved_config.get(key) != expected_config.get(key)
        for key in protocol_keys
    ):
        raise ValueError(
            "Continuation checkpoint uses an incompatible SmartATPG protocol"
        )
    agent.load_training_state_dict(saved["agent"])
    _restore_torch_rng(saved)
    return {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": sha256_file(path),
        "source_manifest_hash": saved.get("manifest_hash"),
        "source_format": saved["format"],
    }


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
    supported_formats = (MANIFEST_FORMAT,)
    if manifest.get("format") not in supported_formats:
        raise ValueError("Training requires a supported data-split manifest")
    manifest_encoder = manifest.get("encoder_variant")
    if (
        manifest.get("format") == MANIFEST_FORMAT
        and manifest_encoder is not None
        and manifest_encoder != args.encoder
    ):
        raise ValueError(
            f"Training encoder {args.encoder} does not match manifest encoder "
            f"{manifest_encoder}"
        )
    batched_training = manifest.get("format") == MANIFEST_FORMAT
    hyperparameters = training_hyperparameters(args.encoder)
    expected_rounds = (
        NORMAL_TRAINING_ROUNDS if batched_training else LEGACY_TRAINING_ROUNDS
    )
    if args.rounds is None:
        args.rounds = expected_rounds
    if args.k_epochs is None:
        args.k_epochs = hyperparameters["k_epochs"] if batched_training else 8
    if args.rounds != expected_rounds:
        raise ValueError(
            f"This SmartATPG manifest requires exactly {expected_rounds} rounds"
        )
    if args.k_epochs <= 0:
        raise ValueError("PPO epochs must be positive")
    if batched_training and args.k_epochs != hyperparameters["k_epochs"]:
        raise ValueError(
            f"SmartATPG {args.encoder} training requires "
            f"k_epochs={hyperparameters['k_epochs']}"
        )
    train_circuits = _resolve_circuit_records(manifest, args.manifest)
    if int(manifest["normal_rounds"]) != args.rounds:
        raise ValueError("Requested rounds do not match the training manifest")
    if int(manifest["backtrack_limit"]) != BACKTRACK_LIMIT:
        raise ValueError(f"Training requires backtrack limit {BACKTRACK_LIMIT}")

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "training_state.pth"
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
    graphs = {
        item["name"]: load_circuit_graph(item["circuit"]) for item in train_circuits
    }
    agent = AGENT_TYPES[args.encoder](
        graphs,
        hidden_dim=32,
        lr_actor=hyperparameters["actor_lr"],
        lr_critic=hyperparameters["critic_lr"],
        rnd_beta=args.rnd_beta,
        k_epochs=args.k_epochs,
    )
    reward_scheme = reward_scheme_for_encoder(args.encoder)
    trainers = {
        item["name"]: CppPodemBacktraceV2Trainer(
            graphs[item["name"]], agent=agent,
            auto_update=not batched_training,
            reward_scheme=reward_scheme,
        )
        for item in train_circuits
    }
    manifest_digest = _manifest_hash(args.manifest)
    config = {
        "rounds": args.rounds,
        "seed": args.seed,
        "rnd_beta": args.rnd_beta,
        "k_epochs": args.k_epochs,
        "backtrack_limit": BACKTRACK_LIMIT,
        "actor_lr": hyperparameters["actor_lr"],
        "critic_lr": hyperparameters["critic_lr"],
        "training_episode_count": sum(
            len(item["episode_fault_ids"]) for item in train_circuits
        ),
        "training_circuit_count": len(train_circuits),
        "validation_circuit_count": int(manifest["validation_circuit_count"]),
        "device": str(device),
        "paper_reward": PAPER_REWARD,
        "reward_scheme": reward_scheme,
        "encoder_variant": args.encoder,
        "heuristic": HEURISTIC,
        "manifest_hash": manifest_digest,
    }
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
        continuation = _load_continuation(args.continue_from, agent, config)
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

    try:
        while state["current_round"] <= args.rounds:
            round_number = int(state["current_round"])
            if state["phase"] == "training":
                order = _episode_order(train_circuits, args.seed, round_number)
                for index in range(int(state["episode_index"]), len(order)):
                    circuit_name, fault_id = order[index]
                    item = train_by_name[circuit_name]
                    trainer = trainers[circuit_name]
                    run_kwargs = {
                        "backtrack_limit": BACKTRACK_LIMIT,
                        "seed": args.seed + round_number,
                        "fault_ids": [fault_id],
                        "use_scoap": True,
                    }
                    if item.get("fault_map"):
                        run_kwargs["fault_map_path"] = item["fault_map"]
                    trainer.run(item["circuit"], **run_kwargs)
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
                        _save_state(checkpoint_path, agent, state)
                _save_inference_artifacts(
                    output_dir, agent, config, round_number,
                )
                state["current_round"] = round_number + 1
                state["episode_index"] = 0
                _save_state(checkpoint_path, agent, state)
            print(
                f"ROUND round={round_number}/{args.rounds} "
                f"training_episodes={len(order)} saved=1",
                flush=True,
            )
    finally:
        writer.close()

    _save_inference_artifacts(
        output_dir, agent, config, args.rounds, final=True,
    )
    print(
        f"TRAINING_COMPLETE rounds={args.rounds} "
        f"episodes={state['completed_episodes']} "
        f"device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()

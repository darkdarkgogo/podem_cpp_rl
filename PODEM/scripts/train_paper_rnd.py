import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from rl_podem.cpp_bridge import (
    CppPodemPPOTrainer,
    export_actor_state_dict,
    load_cpp_embedding_table,
)
from rl_podem.ppo import RLGuidedPPOAgent


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_checkpoint(path, agent, progress, rng, manifest_hash, seed, rl_mode):
    state = {
        "format": "RL_PODEM_PAPER_RND_TRAINING_V1",
        "agent": agent.training_state_dict(),
        "progress": progress,
        "python_random_state": rng.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "manifest_hash": manifest_hash,
        "seed": seed,
        "rl_mode": rl_mode,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a fresh PPO+RND actor on paper-style hard-fault sets."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("actor_output", type=Path)
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rnd-beta", type=float, default=0.05)
    parser.add_argument("--backtrack-limit", type=int, default=97)
    parser.add_argument(
        "--rl-mode",
        choices=("backtrace_rl", "propagate_rl", "both_rl"),
        default="backtrace_rl",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "RL_PODEM_PAPER_TRAINING_V1":
        raise ValueError("Unsupported paper training manifest format.")
    circuits = manifest["circuits"]
    if not circuits:
        raise ValueError("Training manifest contains no circuits.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    first_embeddings = load_cpp_embedding_table(circuits[0]["embeddings"])
    embedding_dim = next(iter(first_embeddings.values())).numel()
    agent = RLGuidedPPOAgent(
        gate_embedding_dim=embedding_dim,
        rnd_beta=args.rnd_beta,
    )
    trainers = {
        item["name"]: CppPodemPPOTrainer(item["embeddings"], agent=agent)
        for item in circuits
    }
    progress = []
    manifest_digest = _manifest_hash(args.manifest)

    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        if state.get("format") != "RL_PODEM_PAPER_RND_TRAINING_V1":
            raise ValueError("Unsupported paper PPO/RND checkpoint format.")
        if state["manifest_hash"] != manifest_digest:
            raise ValueError("Training manifest changed since the checkpoint was saved.")
        checkpoint_mode = state.get("rl_mode", "both_rl")
        if checkpoint_mode != args.rl_mode:
            raise ValueError(
                f"Checkpoint uses {checkpoint_mode}, but --rl-mode is {args.rl_mode}."
            )
        agent.load_training_state_dict(state["agent"])
        progress = list(state["progress"])
        rng.setstate(state["python_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        cuda_random_state = state.get("torch_cuda_random_state")
        if cuda_random_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_random_state)
        print(f"Resumed checkpoint with {len(progress)} completed circuit runs.")

    completed = {(int(item["sweep"]), item["circuit"]) for item in progress}
    for sweep in range(1, args.sweeps + 1):
        if all((sweep, item["name"]) in completed for item in circuits):
            continue
        circuit_order = list(circuits)
        random.Random(args.seed + sweep).shuffle(circuit_order)
        for item in circuit_order:
            key = (sweep, item["name"])
            if key in completed:
                continue
            fault_ids = list(item["fault_ids"])
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
                rl_mode=args.rl_mode,
            )
            record = {"sweep": sweep, "circuit": item["name"], "summary": dict(summary)}
            progress.append(record)
            completed.add(key)
            _save_checkpoint(
                args.checkpoint,
                agent,
                progress,
                rng,
                manifest_digest,
                args.seed,
                args.rl_mode,
            )
            args.actor_output.parent.mkdir(parents=True, exist_ok=True)
            export_actor_state_dict(agent.policy_old.state_dict(), args.actor_output)
            print(f"TRAIN_RESULT {json.dumps(record, sort_keys=True)}", flush=True)

    print(
        f"TRAINING_COMPLETE runs={len(progress)} updates={agent.update_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()

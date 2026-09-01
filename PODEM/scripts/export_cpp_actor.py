import argparse
import torch
from pathlib import Path

from rl_podem.cpp_bridge import export_actor_checkpoint
from rl_podem.backends import resolve_backend
from rl_podem.artifact_paths import validate_output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a PPO actor for native C++ inference")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--embedding-backend", choices=("smartatpg", "deepgate"))
    parser.add_argument("--snapshot", choices=("best", "latest"), default="best")
    args = parser.parse_args()
    output = Path(args.output)
    validate_output_paths([output, output.with_suffix(output.suffix + ".tmp")], [args.checkpoint])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backend = resolve_backend(checkpoint.get("agent", checkpoint), args.embedding_backend)
    if backend == "smartatpg":
        from rl_podem.smartatpg_artifacts import checkpoint_policy, export_actor
        export_actor(checkpoint_policy(checkpoint, args.snapshot, backend), args.output)
    elif "agent" in checkpoint:
        from rl_podem.cpp_bridge import export_actor_v2_state_dict
        state = checkpoint["best_policy_state"] if args.snapshot == "best" else checkpoint["agent"]["policy_old"]
        export_actor_v2_state_dict(state, args.output)
    else:
        export_actor_checkpoint(args.checkpoint, args.output)
    print(f"Exported C++ actor to {args.output}")


if __name__ == "__main__":
    main()

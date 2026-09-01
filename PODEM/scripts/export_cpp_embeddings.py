import argparse

import torch
from pathlib import Path
from rl_podem.backends import resolve_backend
from rl_podem.artifact_paths import validate_output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export matching DeepGate or SmartATPG inputs for C++ PODEM")
    parser.add_argument("bench")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-backend", choices=("smartatpg", "deepgate"))
    parser.add_argument("--snapshot", choices=("best", "latest"), default="best")
    args = parser.parse_args()
    output = Path(args.output)
    validate_output_paths([output, output.with_suffix(output.suffix + ".tmp")], [args.bench, args.checkpoint])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backend = resolve_backend(checkpoint.get("agent", checkpoint), args.embedding_backend)
    if backend == "smartatpg":
        if args.device not in (None, "cpu"):
            parser.error("SmartATPG static export currently uses CPU; omit --device or use cpu")
        from rl_podem.smartatpg_artifacts import checkpoint_policy, export_descriptors
        from rl_podem.smartatpg_features import load_circuit_graph
        state = checkpoint_policy(checkpoint, args.snapshot, backend)
        count, dimension = export_descriptors(state, load_circuit_graph(args.bench), args.output)
    else:
        from rl_podem.deepgate_bridge import export_cpp_embeddings
        count, dimension = export_cpp_embeddings(args.bench, args.checkpoint, args.output, device=args.device)
    print(f"Exported {count} embeddings with dimension {dimension} to {args.output}")


if __name__ == "__main__":
    main()

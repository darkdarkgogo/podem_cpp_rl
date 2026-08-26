import argparse

from rl_podem.deepgate_bridge import export_cpp_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DeepGate embeddings for C++ PODEM")
    parser.add_argument("bench")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    count, dimension = export_cpp_embeddings(
        args.bench, args.checkpoint, args.output, device=args.device
    )
    print(f"Exported {count} embeddings with dimension {dimension} to {args.output}")


if __name__ == "__main__":
    main()

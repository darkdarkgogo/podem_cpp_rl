import argparse

from rl_podem.cpp_bridge import export_actor_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a PPO actor for native C++ inference")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    export_actor_checkpoint(args.checkpoint, args.output)
    print(f"Exported C++ actor to {args.output}")


if __name__ == "__main__":
    main()


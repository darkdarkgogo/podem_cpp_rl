import argparse

from rl_podem.cpp_bridge import CppPodemPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO against the C++ PODEM engine")
    parser.add_argument("circuit")
    parser.add_argument("embeddings")
    parser.add_argument("checkpoint")
    parser.add_argument("actor_output")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--backtrack-limit", type=int, default=97)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument(
        "--rl-mode",
        choices=("backtrace_rl", "propagate_rl", "both_rl"),
        default="backtrace_rl",
    )
    args = parser.parse_args()

    trainer = CppPodemPPOTrainer(args.embeddings, args.checkpoint)
    for pass_index in range(args.passes):
        summary = trainer.run(
            args.circuit,
            args.backtrack_limit,
            args.seed + pass_index,
            rl_mode=args.rl_mode,
        )
        print(f"pass={pass_index + 1}/{args.passes} summary={summary}")
        if trainer.last_metrics:
            print(f"last_update={trainer.last_metrics}")
        trainer.save(args.actor_output)


if __name__ == "__main__":
    main()

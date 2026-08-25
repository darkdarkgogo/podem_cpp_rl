import argparse
import json
from pathlib import Path

from rl_podem.cpp_bridge import profile_cpp_podem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile C++ PODEM and select deterministic hard faults."
    )
    parser.add_argument("circuit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--backtrack-limit", type=int, default=97)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    profiles = profile_cpp_podem(
        args.circuit, backtrack_limit=args.backtrack_limit, seed=args.seed
    )
    eligible = [profile for profile in profiles if int(profile["outcome"]) != 0]
    eligible.sort(key=lambda item: (-int(item["backtracks"]), item["fault_id"]))
    if len(eligible) < args.count:
        raise RuntimeError(
            f"Only {len(eligible)} non-redundant faults are available; "
            f"cannot select {args.count}."
        )

    selected = eligible[: args.count]
    payload = {
        "format": "RL_PODEM_HARD_FAULTS_V1",
        "circuit": str(args.circuit.resolve()),
        "backtrack_limit": args.backtrack_limit,
        "seed": args.seed,
        "profiled_faults": len(profiles),
        "eligible_faults": len(eligible),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Selected {len(selected)} hard faults from {len(profiles)} profiled faults; "
        f"top backtracks={selected[0]['backtracks']} cutoff={selected[-1]['backtracks']}"
    )
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from rl_podem.cpp_bridge import profile_cpp_podem


def _select_hard_faults(profiles, count):
    eligible = [profile for profile in profiles if int(profile["outcome"]) != 0]
    eligible.sort(key=lambda item: (-int(item["backtracks"]), item["fault_id"]))
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} non-redundant faults are available; "
            f"cannot select {count}."
        )
    return eligible[:count], len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile c6288 and full-scan s38417 for paper-style training."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--backtrack-limit", type=int, default=97)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    configurations = [
        ("c6288", root / "sample_circuits" / "c6288.bench"),
        ("s38417_scan", root / "sample_circuits" / "s38417_scan.bench"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_circuits = []

    for name, circuit in configurations:
        embeddings = args.output_dir / f"{name}.emb"
        if not embeddings.exists():
            raise FileNotFoundError(
                f"Missing embeddings for {name}: {embeddings}. Export them first."
            )
        print(f"PROFILE_START circuit={name}", flush=True)
        profiles = profile_cpp_podem(
            circuit, backtrack_limit=args.backtrack_limit, seed=args.seed
        )
        selected, eligible_count = _select_hard_faults(profiles, args.count)
        profile_path = args.output_dir / f"{name}_profile.json"
        profile_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        manifest_circuits.append(
            {
                "name": name,
                "circuit": str(circuit.resolve()),
                "embeddings": str(embeddings.resolve()),
                "fault_ids": [item["fault_id"] for item in selected],
                "profiled_faults": len(profiles),
                "eligible_faults": eligible_count,
                "top_backtracks": int(selected[0]["backtracks"]),
                "cutoff_backtracks": int(selected[-1]["backtracks"]),
            }
        )
        print(
            f"PROFILE_RESULT circuit={name} profiled={len(profiles)} "
            f"eligible={eligible_count} selected={len(selected)} "
            f"top={selected[0]['backtracks']} cutoff={selected[-1]['backtracks']}",
            flush=True,
        )

    manifest = {
        "format": "RL_PODEM_PAPER_TRAINING_V1",
        "fault_count_per_circuit": args.count,
        "backtrack_limit": args.backtrack_limit,
        "profile_seed": args.seed,
        "circuits": manifest_circuits,
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

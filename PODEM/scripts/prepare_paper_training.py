import argparse
import hashlib
import json
import random
from pathlib import Path

from convert_binary_bench import convert_binary_bench
from rl_podem.cpp_bridge import profile_cpp_podem
from rl_podem.deepgate_bridge import export_cpp_embeddings


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_hard_faults(profiles, count):
    eligible = [profile for profile in profiles if int(profile["outcome"]) != 0]
    eligible.sort(
        key=lambda item: (
            -int(item["backtracks"]),
            -int(item.get("backtrace_steps", 0)),
            item["fault_id"],
        )
    )
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} non-redundant faults are available; "
            f"cannot select {count}."
        )
    return eligible[:count], len(eligible)


def _split_hard_faults(profiles, training_count, validation_count, seed, name):
    selected, eligible_count = _select_hard_faults(
        profiles, training_count + validation_count
    )
    shuffled = list(selected)
    random.Random(f"{seed}:{name}").shuffle(shuffled)
    return (
        shuffled[:training_count],
        shuffled[training_count:],
        selected,
        eligible_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile c6288 and full-scan s38417 for paper-style training."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=50)
    parser.add_argument("--backtrack-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--deepgate-checkpoint", type=Path)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.count <= 0 or args.validation_count <= 0:
        raise ValueError("Training and validation fault counts must be positive.")
    if args.backtrack_limit != 500:
        raise ValueError("Paper-reward V3 preparation requires --backtrack-limit=500.")

    root = Path(__file__).resolve().parents[1]
    source_configurations = [
        ("c6288", root / "sample_circuits" / "c6288.bench"),
        ("s38417_scan", root / "sample_circuits" / "s38417_scan.bench"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_circuits = []

    for name, source_circuit in source_configurations:
        circuit = source_circuit.with_name(
            f"{source_circuit.stem}_binary{source_circuit.suffix}"
        )
        fault_map = circuit.with_suffix(".faultmap")
        print(f"CONVERT_START circuit={name}", flush=True)
        convert_binary_bench(source_circuit, circuit, fault_map)
        embeddings = args.output_dir / f"{name}_binary.emb"
        if args.deepgate_checkpoint and (
            args.force_embeddings or not embeddings.exists()
        ):
            export_cpp_embeddings(
                circuit,
                args.deepgate_checkpoint,
                embeddings,
                device=args.device,
            )
        if not embeddings.exists():
            raise FileNotFoundError(
                f"Missing binary embeddings for {name}: {embeddings}. Pass "
                "--deepgate-checkpoint to generate them."
            )
        print(f"PROFILE_START circuit={name}", flush=True)
        profiles = profile_cpp_podem(
            circuit,
            backtrack_limit=args.backtrack_limit,
            seed=args.seed,
            fault_map_path=fault_map,
        )
        training_faults, validation_faults, selected, eligible_count = (
            _split_hard_faults(
                profiles,
                args.count,
                args.validation_count,
                args.split_seed,
                name,
            )
        )
        if set(item["fault_id"] for item in training_faults) & set(
            item["fault_id"] for item in validation_faults
        ):
            raise RuntimeError(f"Training and validation faults overlap for {name}.")
        profile_path = args.output_dir / f"{name}_profile.json"
        profile_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        manifest_circuits.append(
            {
                "name": name,
                "source_circuit": str(source_circuit.resolve()),
                "circuit": str(circuit.resolve()),
                "fault_map": str(fault_map.resolve()),
                "embeddings": str(embeddings.resolve()),
                "artifact_sha256": {
                    "source_circuit": _sha256_file(source_circuit),
                    "circuit": _sha256_file(circuit),
                    "fault_map": _sha256_file(fault_map),
                    "embeddings": _sha256_file(embeddings),
                },
                "training_fault_ids": [
                    item["fault_id"] for item in training_faults
                ],
                "validation_fault_ids": [
                    item["fault_id"] for item in validation_faults
                ],
                "profiled_faults": len(profiles),
                "eligible_faults": eligible_count,
                "top_backtracks": int(selected[0]["backtracks"]),
                "cutoff_backtracks": int(selected[-1]["backtracks"]),
                "top_backtrace_steps": int(selected[0].get("backtrace_steps", 0)),
                "cutoff_backtrace_steps": int(
                    selected[-1].get("backtrace_steps", 0)
                ),
            }
        )
        print(
            f"PROFILE_RESULT circuit={name} profiled={len(profiles)} "
            f"eligible={eligible_count} train={len(training_faults)} "
            f"validation={len(validation_faults)} "
            f"top={selected[0]['backtracks']} cutoff={selected[-1]['backtracks']}",
            flush=True,
        )

    manifest = {
        "format": "RL_PODEM_PAPER_TRAINING_V3",
        "training_fault_count_per_circuit": args.count,
        "validation_fault_count_per_circuit": args.validation_count,
        "backtrack_limits": {
            "profile": args.backtrack_limit,
            "training": args.backtrack_limit,
            "validation": args.backtrack_limit,
        },
        "profile_seed": args.seed,
        "split_seed": args.split_seed,
        "circuits": manifest_circuits,
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

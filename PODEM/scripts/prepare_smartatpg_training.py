"""Prepare the 16-circuit, top-50 SCOAP-detected SmartATPG experiment."""

import argparse
import hashlib
import json
from pathlib import Path

from convert_binary_bench import convert_binary_bench
from convert_full_scan_bench import convert_full_scan
from rl_podem.backends import smartatpg_metadata
from rl_podem.cpp_bridge import profile_cpp_podem
from smartatpg_portable import CIRCUITS


MANIFEST_FORMAT = "SMARTATPG_ALL_CIRCUITS_TRAINING_V3"
FAULT_FILTER = "baseline_detected_only"
BACKTRACK_LIMIT = 2000
FAULTS_PER_CIRCUIT = 50
NORMAL_TRAINING_ROUNDS = 8
MAX_REINFORCEMENT_ROUNDS = 5
HEURISTIC = "scoap_heuristic"
SELECTION = ["backtracks_desc", "backtrace_steps_desc", "fault_id_asc"]
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_hard_faults(profiles, count):
    if count <= 0:
        raise ValueError("Detected fault count must be positive")
    ranked = sorted(
        (dict(item) for item in profiles if int(item["outcome"]) == 1),
        key=lambda item: (
            -int(item["backtracks"]),
            -int(item.get("backtrace_steps", 0)),
            str(item["fault_id"]),
        ),
    )
    if len(ranked) < count:
        raise RuntimeError(
            f"Only {len(ranked)} baseline-detected faults are available; "
            f"cannot select {count}. Undetected faults will not be used to fill the set."
        )
    return ranked[:count]


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _validate_fault_selection(circuits, count):
    fault_keys = [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item.get("training_fault_ids", [])
    ]
    expected = len(CIRCUITS) * count
    if len(fault_keys) != expected or len(set(fault_keys)) != expected:
        raise ValueError(
            f"SmartATPG training manifest must contain exactly {expected} "
            "unique circuit/fault pairs"
        )


def _validate_resume(
    manifest_path, count, backtrack_limit, seed, normal_rounds,
    reinforcement_rounds,
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format": MANIFEST_FORMAT,
        "fault_filter": FAULT_FILTER,
        "fault_count_per_circuit": count,
        "backtrack_limit": backtrack_limit,
        "profile_seed": seed,
        "heuristic": HEURISTIC,
        "circuit_order": list(CIRCUITS),
        "normal_rounds": normal_rounds,
        "reinforcement_rounds": reinforcement_rounds,
        "selection": SELECTION,
        **smartatpg_metadata(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Existing SmartATPG manifest configuration changed; "
            "use a new output directory to prepare the detected-only fault set"
        )
    if [item.get("name") for item in manifest.get("circuits", [])] != list(CIRCUITS):
        raise ValueError("SmartATPG training manifest must contain all 16 circuits")
    for item in manifest["circuits"]:
        for key, expected_hash in item["artifact_sha256"].items():
            path = Path(item[key])
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Manifest artifact changed: {path}")
        if len(item.get("training_faults", [])) != count:
            raise ValueError(f"Circuit {item['name']} does not contain {count} faults")
        profiles = json.loads(Path(item["profile"]).read_text(encoding="utf-8"))
        try:
            selected = select_hard_faults(profiles, count)
        except RuntimeError as error:
            raise RuntimeError(f"Circuit {item['name']}: {error}") from error
        if (
            item["training_faults"] != selected
            or item.get("training_fault_ids") != [row["fault_id"] for row in selected]
        ):
            raise ValueError(f"Circuit {item['name']} faults are not the baseline detected top {count}")
    _validate_fault_selection(manifest["circuits"], count)
    return manifest


def prepare(
    output_dir, count=FAULTS_PER_CIRCUIT, backtrack_limit=BACKTRACK_LIMIT,
    seed=14, normal_rounds=NORMAL_TRAINING_ROUNDS,
    reinforcement_rounds=MAX_REINFORCEMENT_ROUNDS, resume=False,
):
    if count != FAULTS_PER_CIRCUIT:
        raise ValueError(
            f"SmartATPG training preparation requires exactly "
            f"{FAULTS_PER_CIRCUIT} faults per circuit"
        )
    if backtrack_limit != BACKTRACK_LIMIT:
        raise ValueError(
            f"SmartATPG training preparation requires backtrack limit "
            f"{BACKTRACK_LIMIT}"
        )
    if normal_rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"SmartATPG training preparation requires exactly "
            f"{NORMAL_TRAINING_ROUNDS} normal rounds"
        )
    if not 0 <= reinforcement_rounds <= MAX_REINFORCEMENT_ROUNDS:
        raise ValueError(
            f"Reinforcement rounds must be between 0 and "
            f"{MAX_REINFORCEMENT_ROUNDS}"
        )
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "training_manifest.json"
    if resume and manifest_path.is_file():
        manifest = _validate_resume(
            manifest_path, count, backtrack_limit, seed, normal_rounds,
            reinforcement_rounds,
        )
        print(f"MANIFEST_REUSED {manifest_path}", flush=True)
        return manifest

    inputs_dir = output_dir / "inputs"
    profiles_dir = output_dir / "profiles"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    configurations = tuple(
        (name, ROOT / "sample_circuits" / f"{name}.bench", name.startswith("s"))
        for name in CIRCUITS
    )
    circuits = []
    for name, source, sequential in configurations:
        if not source.is_file():
            raise FileNotFoundError(f"Missing SmartATPG source circuit: {source}")
        conversion_source = source
        scan_path = None
        if sequential:
            scan_path = inputs_dir / f"{name}_scan.bench"
            convert_full_scan(source, scan_path)
            conversion_source = scan_path
        binary_path = inputs_dir / (
            f"{name}_scan_binary.bench" if sequential else f"{name}_binary.bench"
        )
        fault_map = binary_path.with_suffix(".faultmap")
        print(f"CONVERT circuit={name}", flush=True)
        convert_binary_bench(conversion_source, binary_path, fault_map)
        print(f"PROFILE circuit={name}", flush=True)
        profiles = profile_cpp_podem(
            binary_path,
            backtrack_limit=backtrack_limit,
            seed=seed,
            fault_map_path=fault_map,
            use_scoap=True,
        )
        try:
            selected = select_hard_faults(profiles, count)
        except RuntimeError as error:
            raise RuntimeError(f"Circuit {name}: {error}") from error
        profile_path = profiles_dir / f"{name}_baseline_profile.json"
        _atomic_json(profile_path, profiles)
        artifact_paths = {
            "source_circuit": source.resolve(),
            "circuit": binary_path.resolve(),
            "fault_map": fault_map.resolve(),
            "profile": profile_path.resolve(),
        }
        if scan_path is not None:
            artifact_paths["scan_circuit"] = scan_path.resolve()
        circuits.append({
            "name": name,
            **{key: str(path) for key, path in artifact_paths.items()},
            "artifact_sha256": {
                key: sha256_file(path) for key, path in artifact_paths.items()
            },
            "profiled_faults": len(profiles),
            "training_faults": selected,
            "training_fault_ids": [item["fault_id"] for item in selected],
        })
        print(
            f"SELECT circuit={name} faults={len(selected)} "
            f"top_backtracks={selected[0]['backtracks']} "
            f"cutoff_backtracks={selected[-1]['backtracks']}",
            flush=True,
        )

    _validate_fault_selection(circuits, count)
    manifest = {
        "format": MANIFEST_FORMAT,
        "fault_filter": FAULT_FILTER,
        **smartatpg_metadata(),
        "selection": SELECTION,
        "fault_count_per_circuit": count,
        "backtrack_limit": backtrack_limit,
        "profile_seed": seed,
        "heuristic": HEURISTIC,
        "circuit_order": list(CIRCUITS),
        "normal_rounds": normal_rounds,
        "reinforcement_rounds": reinforcement_rounds,
        "circuits": circuits,
    }
    _atomic_json(manifest_path, manifest)
    print(f"MANIFEST {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=FAULTS_PER_CIRCUIT)
    parser.add_argument("--backtrack-limit", type=int, default=BACKTRACK_LIMIT)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--normal-rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument(
        "--reinforcement-rounds", type=int,
        default=MAX_REINFORCEMENT_ROUNDS,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.count <= 0 or args.backtrack_limit <= 0:
        raise ValueError("Fault count and backtrack limit must be positive")
    prepare(
        args.output_dir,
        count=args.count,
        backtrack_limit=args.backtrack_limit,
        seed=args.seed,
        normal_rounds=args.normal_rounds,
        reinforcement_rounds=args.reinforcement_rounds,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

"""Prepare the two-circuit, top-100-fault SmartATPG paper experiment."""

import argparse
import hashlib
import json
from pathlib import Path

from convert_binary_bench import convert_binary_bench
from convert_full_scan_bench import convert_full_scan
from rl_podem.backends import smartatpg_metadata
from rl_podem.cpp_bridge import profile_cpp_podem


MANIFEST_FORMAT = "SMARTATPG_PAPER_TRAINING_V1"
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_hard_faults(profiles, count):
    ranked = sorted(
        (dict(item) for item in profiles),
        key=lambda item: (
            -int(item["backtracks"]),
            -int(item.get("backtrace_steps", 0)),
            str(item["fault_id"]),
        ),
    )
    if len(ranked) < count:
        raise RuntimeError(
            f"Only {len(ranked)} faults are available; cannot select {count}."
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


def _validate_resume(manifest_path, count, backtrack_limit, seed):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format": MANIFEST_FORMAT,
        "fault_count_per_circuit": count,
        "backtrack_limit": backtrack_limit,
        "profile_seed": seed,
        **smartatpg_metadata(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing SmartATPG manifest configuration changed")
    if [item.get("name") for item in manifest.get("circuits", [])] != [
        "c6288", "s38417"
    ]:
        raise ValueError("SmartATPG training manifest must contain c6288 and s38417")
    for item in manifest["circuits"]:
        for key, expected_hash in item["artifact_sha256"].items():
            path = Path(item[key])
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Manifest artifact changed: {path}")
        if len(item.get("training_faults", [])) != count:
            raise ValueError(f"Circuit {item['name']} does not contain {count} faults")
    return manifest


def prepare(output_dir, count=100, backtrack_limit=500, seed=14, resume=False):
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "training_manifest.json"
    if resume and manifest_path.is_file():
        manifest = _validate_resume(
            manifest_path, count, backtrack_limit, seed
        )
        print(f"MANIFEST_REUSED {manifest_path}", flush=True)
        return manifest

    inputs_dir = output_dir / "inputs"
    profiles_dir = output_dir / "profiles"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    configurations = (
        ("c6288", ROOT / "sample_circuits/c6288.bench", False),
        ("s38417", ROOT / "sample_circuits/s38417.bench", True),
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
        )
        selected = select_hard_faults(profiles, count)
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

    manifest = {
        "format": MANIFEST_FORMAT,
        **smartatpg_metadata(),
        "selection": ["backtracks_desc", "backtrace_steps_desc", "fault_id_asc"],
        "fault_count_per_circuit": count,
        "backtrack_limit": backtrack_limit,
        "profile_seed": seed,
        "circuits": circuits,
    }
    _atomic_json(manifest_path, manifest)
    print(f"MANIFEST {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--backtrack-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.count <= 0 or args.backtrack_limit <= 0:
        raise ValueError("Fault count and backtrack limit must be positive")
    prepare(
        args.output_dir,
        count=args.count,
        backtrack_limit=args.backtrack_limit,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

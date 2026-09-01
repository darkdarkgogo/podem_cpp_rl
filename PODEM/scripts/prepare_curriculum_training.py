import argparse
import copy
import hashlib
import json
from pathlib import Path
from rl_podem.backends import MANIFEST_V5, resolve_backend, smartatpg_metadata

from rl_podem.cpp_bridge import profile_cpp_podem
from rl_podem.curriculum import (
    DEFAULT_TRAIN_COUNTS,
    DEFAULT_VALIDATION_COUNTS,
    collect_teacher_samples,
    stratified_split,
)


MANIFEST_FORMAT = "RL_PODEM_CURRICULUM_V4"
PROFILE_FORMAT = "RL_PODEM_CURRICULUM_PROFILE_V4"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_circuits(root):
    benchmark_embeddings = root / "artifacts" / "paper_v3" / "benchmark_embeddings"
    return [
        ("c432", root / "artifacts" / "v2_smoke" / "c432_binary.emb"),
        ("c499", benchmark_embeddings / "c499_binary.emb"),
        ("c1355", benchmark_embeddings / "c1355_binary.emb"),
        ("c1908", benchmark_embeddings / "c1908_binary.emb"),
        ("c2670", benchmark_embeddings / "c2670_binary.emb"),
        ("c3540", benchmark_embeddings / "c3540_binary.emb"),
        ("c5315", benchmark_embeddings / "c5315_binary.emb"),
        ("c7552", benchmark_embeddings / "c7552_binary.emb"),
        ("c6288", root / "artifacts" / "paper_v3" / "c6288_binary.emb"),
        (
            "s38417_scan",
            root / "artifacts" / "paper_v3" / "s38417_scan_binary.emb",
        ),
    ]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _combine_teacher_samples(per_circuit_samples):
    combined = []
    for samples in per_circuit_samples:
        combined.extend(samples)
    return sorted(
        combined,
        key=lambda item: (
            item["circuit"],
            item["objective_name"],
            int(item["objective_value"]),
        ),
    )


def _from_source_manifest(source, output_dir, backend):
    from train_curriculum import _validate_manifest
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if source.parent == output_dir:
        raise ValueError("Use a new output directory; never replace the source manifest")
    manifest = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    if manifest.get("format") not in (MANIFEST_FORMAT, MANIFEST_V5):
        raise ValueError("Unsupported source manifest format")
    if backend == "smartatpg":
        manifest.update(format=MANIFEST_V5, **smartatpg_metadata())
        for item in manifest["circuits"]:
            item.pop("embeddings", None)
            item["artifact_sha256"].pop("embeddings", None)
    elif manifest.get("embedding_backend", "deepgate") != "deepgate":
        raise ValueError("Switching to DeepGate requires a manifest with DeepGate artifacts")
    _validate_manifest(manifest)
    manifest["source_manifest_sha256"] = _sha256_file(source)
    path = output_dir / "training_manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {path}")
    _write_json(path, manifest)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-circuit V4 curriculum and teacher data."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--embedding-backend", choices=("smartatpg", "deepgate"),
                        help="New datasets default to smartatpg; resume uses the existing manifest backend.")
    parser.add_argument("--source-manifest", type=Path,
                        help="Reuse exact fault splits and teachers in a new directory, without loading DeepGate embeddings.")
    parser.add_argument("--backtrack-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Select one training and one validation fault from each stratum.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete per-circuit profile JSON files after an interrupted run.",
    )
    args = parser.parse_args()
    if args.source_manifest:
        if args.smoke or args.resume:
            parser.error("--source-manifest reuses exact splits; do not combine it with --smoke/--resume")
        path = _from_source_manifest(args.source_manifest, args.output_dir, args.embedding_backend or "smartatpg")
        print(f"MANIFEST {path}", flush=True)
        return
    existing_path = args.output_dir / "training_manifest.json"
    if existing_path.exists():
        if not args.resume:
            raise FileExistsError(f"Existing manifest is protected; use --resume or a new directory: {existing_path}")
        from train_curriculum import _validate_manifest
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        resolve_backend(existing, args.embedding_backend)
        expected = {"backtrack_limit": args.backtrack_limit, "profile_seed": args.seed,
                    "split_seed": args.split_seed, "smoke": args.smoke}
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("Preparation resume configuration differs from existing manifest")
        _validate_manifest(existing)
        print(f"MANIFEST_REUSED {existing_path.resolve()}", flush=True)
        return
    args.embedding_backend = args.embedding_backend or "smartatpg"
    if args.backtrack_limit != 500:
        raise ValueError("V4 uses a fixed backtrack limit of 500.")

    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_counts = (
        {name: 1 for name in DEFAULT_TRAIN_COUNTS}
        if args.smoke
        else dict(DEFAULT_TRAIN_COUNTS)
    )
    validation_counts = (
        {name: 1 for name in DEFAULT_VALIDATION_COUNTS}
        if args.smoke
        else dict(DEFAULT_VALIDATION_COUNTS)
    )

    manifest_circuits = []
    training_teacher = []
    validation_teacher = []
    for name, embeddings in _default_circuits(root):
        circuit = root / "sample_circuits" / f"{name}_binary.bench"
        fault_map = circuit.with_suffix(".faultmap")
        for artifact in ((circuit, fault_map, embeddings) if args.embedding_backend == "deepgate" else (circuit, fault_map)):
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing V4 input artifact: {artifact}")

        print(f"PROFILE_START circuit={name}", flush=True)
        profile_path = args.output_dir / f"{name}_profile.json"
        profile_config = {
            "format": PROFILE_FORMAT,
            "circuit_sha256": _sha256_file(circuit),
            "fault_map_sha256": _sha256_file(fault_map),
            "backtrack_limit": args.backtrack_limit,
            "seed": args.seed,
        }
        if args.resume and profile_path.is_file():
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            cached_config = {
                key: cached.get(key) for key in profile_config
            } if isinstance(cached, dict) else {}
            if cached_config != profile_config:
                raise ValueError(
                    f"Cached profile configuration changed: {profile_path}"
                )
            profiles = cached.get("profiles")
            if not isinstance(profiles, list) or not profiles:
                raise ValueError(f"Invalid cached profile: {profile_path}")
            print(
                f"PROFILE_RESUME circuit={name} faults={len(profiles)}",
                flush=True,
            )
        else:
            profiles = profile_cpp_podem(
                circuit,
                backtrack_limit=args.backtrack_limit,
                seed=args.seed,
                fault_map_path=fault_map,
            )
            _write_json(profile_path, {**profile_config, "profiles": profiles})
        training, validation, available = stratified_split(
            profiles,
            seed=f"{args.split_seed}:{name}",
            train_counts=train_counts,
            validation_counts=validation_counts,
        )
        if {item["fault_id"] for item in training} & {
            item["fault_id"] for item in validation
        }:
            raise RuntimeError(f"Training and validation faults overlap for {name}.")

        print(
            f"TEACHER_START circuit={name} train={len(training)} "
            f"validation={len(validation)}",
            flush=True,
        )
        train_samples, train_summary = collect_teacher_samples(
            name,
            circuit,
            fault_map,
            training,
            backtrack_limit=args.backtrack_limit,
            seed=args.seed,
        )
        validation_samples, validation_summary = collect_teacher_samples(
            name,
            circuit,
            fault_map,
            validation,
            backtrack_limit=args.backtrack_limit,
            seed=args.seed,
        )
        training_teacher.append(train_samples)
        validation_teacher.append(validation_samples)

        manifest_circuits.append(
            {
                "name": name,
                "circuit": str(circuit.resolve()),
                "fault_map": str(fault_map.resolve()),
                **({"embeddings": str(embeddings.resolve())} if args.embedding_backend == "deepgate" else {}),
                "profile": str(profile_path.resolve()),
                "artifact_sha256": {
                    "circuit": _sha256_file(circuit),
                    "fault_map": _sha256_file(fault_map),
                    **({"embeddings": _sha256_file(embeddings)} if args.embedding_backend == "deepgate" else {}),
                    "profile": _sha256_file(profile_path),
                },
                "available_by_difficulty": available,
                "training_faults": training,
                "validation_faults": validation,
                "teacher_summary": {
                    "training": train_summary,
                    "validation": validation_summary,
                },
            }
        )
        print(
            f"PREPARED circuit={name} eligible={sum(available.values())} "
            f"train_states={len(train_samples)} validation_states={len(validation_samples)}",
            flush=True,
        )

    training_teacher_path = args.output_dir / "teacher_training.json"
    validation_teacher_path = args.output_dir / "teacher_validation.json"
    _write_json(training_teacher_path, _combine_teacher_samples(training_teacher))
    _write_json(validation_teacher_path, _combine_teacher_samples(validation_teacher))
    manifest = {
        "format": MANIFEST_FORMAT if args.embedding_backend == "deepgate" else MANIFEST_V5,
        **(smartatpg_metadata() if args.embedding_backend == "smartatpg" else {}),
        "backtrack_limit": args.backtrack_limit,
        "profile_seed": args.seed,
        "split_seed": args.split_seed,
        "smoke": args.smoke,
        "train_counts_by_difficulty": train_counts,
        "validation_counts_by_difficulty": validation_counts,
        "teacher_training": str(training_teacher_path.resolve()),
        "teacher_validation": str(validation_teacher_path.resolve()),
        "teacher_sha256": {
            "training": _sha256_file(training_teacher_path),
            "validation": _sha256_file(validation_teacher_path),
        },
        "circuits": manifest_circuits,
    }
    manifest_path = args.output_dir / "training_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"MANIFEST {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

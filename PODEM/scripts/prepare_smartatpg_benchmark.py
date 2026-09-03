"""Create a relocatable SmartATPG benchmark bundle in the training environment."""

import argparse
import json
from pathlib import Path
import shutil

from convert_binary_bench import convert_binary_bench
from convert_full_scan_bench import convert_full_scan
from smartatpg_portable import (
    CIRCUITS,
    FEATURE_SCHEMA,
    GATE_EMBEDDING_DIM,
    GRAPH_CONFIG,
    POLICY_STATE_DIM,
    load_model,
    sha256_file,
)


MANIFEST_FORMAT = "SMARTATPG_BENCHMARK_BUNDLE_V2"
ROOT = Path(__file__).resolve().parents[1]
def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _bundle_path(bundle_root, relative):
    root = Path(bundle_root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Bundle path escapes its root: {relative}") from error
    return path


def _validate_resume(path):
    bundle_root = path.parent.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": MANIFEST_FORMAT,
        "backend": "smartatpg",
        "feature_schema": FEATURE_SCHEMA,
        "graph_config": GRAPH_CONFIG,
        "gate_embedding_dim": GATE_EMBEDDING_DIM,
        "policy_state_dim": POLICY_STATE_DIM,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing benchmark bundle has an incompatible format")
    if [item.get("name") for item in manifest.get("circuits", [])] != list(CIRCUITS):
        raise ValueError("Existing benchmark bundle does not contain all 16 circuits")
    records = [manifest["model"], *manifest["circuits"]]
    for record in records:
        for key, expected_hash in record["artifact_sha256"].items():
            artifact = _bundle_path(bundle_root, record[key])
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"Benchmark bundle artifact changed: {artifact}")
    model = load_model(_bundle_path(bundle_root, manifest["model"]["path"]))
    if model.snapshot != manifest["snapshot"]:
        raise ValueError("Benchmark model snapshot changed")
    return manifest


def prepare(output_dir, model_path, resume=False):
    output_dir = Path(output_dir).resolve()
    source_model = Path(model_path).resolve()
    source_model_hash = sha256_file(source_model)
    manifest_path = output_dir / "bundle_manifest.json"
    if resume and manifest_path.is_file():
        manifest = _validate_resume(manifest_path)
        if manifest["model"]["artifact_sha256"]["path"] == source_model_hash:
            print(f"BENCHMARK_BUNDLE_REUSED {manifest_path}", flush=True)
            return manifest
        print("BENCHMARK_BUNDLE_MODEL_CHANGED rebuilding", flush=True)

    model = load_model(source_model)
    if model.best_round <= 0 or model.best_score is None:
        raise ValueError("Benchmark bundle requires model_best.txt, not the latest model")
    bundled_model = output_dir / "model" / "model_best.txt"
    bundled_model.parent.mkdir(parents=True, exist_ok=True)
    if bundled_model.exists() and sha256_file(bundled_model) != source_model_hash and not resume:
        raise ValueError("Existing bundled model differs; use a new output directory")
    if not bundled_model.exists() or sha256_file(bundled_model) != source_model_hash:
        shutil.copy2(source_model, bundled_model)

    inputs = output_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    records = []
    for name in CIRCUITS:
        source = ROOT / "sample_circuits" / f"{name}.bench"
        if not source.is_file():
            raise FileNotFoundError(f"Missing benchmark source circuit: {source}")
        conversion_source = source
        if name.startswith("s"):
            scan = inputs / f"{name}_scan.bench"
            convert_full_scan(source, scan)
            conversion_source = scan
        binary = inputs / (
            f"{name}_scan_binary.bench" if name.startswith("s")
            else f"{name}_binary.bench"
        )
        fault_map = binary.with_suffix(".faultmap")
        convert_binary_bench(conversion_source, binary, fault_map)
        records.append({
            "name": name,
            "circuit": binary.relative_to(output_dir).as_posix(),
            "fault_map": fault_map.relative_to(output_dir).as_posix(),
            "artifact_sha256": {
                "circuit": sha256_file(binary),
                "fault_map": sha256_file(fault_map),
            },
        })
        print(f"BENCHMARK_BUNDLE_INPUT circuit={name}", flush=True)

    model_relative = bundled_model.relative_to(output_dir).as_posix()
    manifest = {
        "format": MANIFEST_FORMAT,
        "backend": "smartatpg",
        "feature_schema": FEATURE_SCHEMA,
        "graph_config": GRAPH_CONFIG,
        "gate_embedding_dim": GATE_EMBEDDING_DIM,
        "policy_state_dim": POLICY_STATE_DIM,
        "snapshot": model.snapshot,
        "best_round": model.best_round,
        "best_score": list(model.best_score),
        "model": {
            "path": model_relative,
            "artifact_sha256": {"path": sha256_file(bundled_model)},
        },
        "circuits": records,
    }
    _atomic_json(manifest_path, manifest)
    print(f"BENCHMARK_BUNDLE {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    prepare(args.output_dir, args.model, resume=args.resume)


if __name__ == "__main__":
    main()

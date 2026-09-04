"""Create a relocatable SmartATPG benchmark bundle in the training environment."""

import argparse
import json
from pathlib import Path
import shutil

from convert_binary_bench import convert_binary_bench
from convert_full_scan_bench import convert_full_scan
from smartatpg_portable import (
    ACTION_MASK_DIM,
    ACTOR_INPUT_DIM,
    CIRCUITS,
    GAT_GRU_GRAPH_CONFIG,
    FEATURE_SCHEMA,
    GATE_EMBEDDING_DIM,
    GRAPH_CONFIG,
    POLICY_STATE_DIM,
    load_model,
    sha256_file,
)


MANIFEST_FORMAT = "SMARTATPG_BENCHMARK_BUNDLE_V5"
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
        "gate_embedding_dim": GATE_EMBEDDING_DIM,
        "action_mask_dim": ACTION_MASK_DIM,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing benchmark bundle has an incompatible format")
    if [item.get("name") for item in manifest.get("circuits", [])] != list(CIRCUITS):
        raise ValueError("Existing benchmark bundle does not contain all 16 circuits")
    if set(manifest.get("models", {})) != {"smartatpg_mean", "smartatpg_gat_gru"}:
        raise ValueError("Benchmark bundle must contain both SmartATPG models")
    records = [*manifest["models"].values(), *manifest["circuits"]]
    for record in records:
        for key, expected_hash in record["artifact_sha256"].items():
            artifact = _bundle_path(bundle_root, record[key])
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"Benchmark bundle artifact changed: {artifact}")
    for name, record in manifest["models"].items():
        model = load_model(_bundle_path(bundle_root, record["path"]))
        if model.snapshot != record["snapshot"]:
            raise ValueError(f"Benchmark model snapshot changed: {name}")
    return manifest


def prepare(output_dir, baseline_model_path, gat_gru_model_path, resume=False):
    output_dir = Path(output_dir).resolve()
    source_models = {
        "smartatpg_mean": Path(baseline_model_path).resolve(),
        "smartatpg_gat_gru": Path(gat_gru_model_path).resolve(),
    }
    source_hashes = {name: sha256_file(path) for name, path in source_models.items()}
    manifest_path = output_dir / "bundle_manifest.json"
    if resume and manifest_path.is_file():
        manifest = _validate_resume(manifest_path)
        if all(
            manifest["models"][name]["artifact_sha256"]["path"] == digest
            for name, digest in source_hashes.items()
        ):
            print(f"BENCHMARK_BUNDLE_REUSED {manifest_path}", flush=True)
            return manifest
        print("BENCHMARK_BUNDLE_MODEL_CHANGED rebuilding", flush=True)

    models = {name: load_model(path) for name, path in source_models.items()}
    expected_variants = {
        "smartatpg_mean": ("fanin_mean", GRAPH_CONFIG),
        "smartatpg_gat_gru": ("level_gat_gru", GAT_GRU_GRAPH_CONFIG),
    }
    model_records = {}
    for name, model in models.items():
        variant, graph_config = expected_variants[name]
        if model.encoder_variant != variant or model.graph_config != graph_config:
            raise ValueError(f"Wrong encoder variant for benchmark model {name}")
        expected_dim = ACTOR_INPUT_DIM + int(variant == "level_gat_gru")
        if model.model_format != "SMARTATPG_MODEL_V8" or model.actor_input_dim != expected_dim:
            raise ValueError(f"Benchmark requires a direct Actor model for {name}")
        if model.best_round <= 0 or model.best_score is None:
            raise ValueError(f"Benchmark requires a best checkpoint for {name}")
        bundled_model = output_dir / "models" / f"{name}.txt"
        bundled_model.parent.mkdir(parents=True, exist_ok=True)
        if bundled_model.exists() and sha256_file(bundled_model) != source_hashes[name] and not resume:
            raise ValueError("Existing bundled model differs; use a new output directory")
        if not bundled_model.exists() or sha256_file(bundled_model) != source_hashes[name]:
            shutil.copy2(source_models[name], bundled_model)
        model_records[name] = {
            "path": bundled_model.relative_to(output_dir).as_posix(),
            "encoder_variant": variant,
            "actor_input_dim": model.actor_input_dim,
            "decision_state_dim": model.decision_state_dim,
            "graph_config": graph_config,
            "snapshot": model.snapshot,
            "best_round": model.best_round,
            "best_score": list(model.best_score),
            "parameter_count": sum(
                tensor.rows * tensor.cols for tensor in model.tensors.values()
            ),
            "artifact_sha256": {"path": sha256_file(bundled_model)},
        }

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

    manifest = {
        "format": MANIFEST_FORMAT,
        "backend": "smartatpg",
        "feature_schema": FEATURE_SCHEMA,
        "gate_embedding_dim": GATE_EMBEDDING_DIM,
        "action_mask_dim": ACTION_MASK_DIM,
        "models": model_records,
        "circuits": records,
    }
    _atomic_json(manifest_path, manifest)
    print(f"BENCHMARK_BUNDLE {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("baseline_model", type=Path)
    parser.add_argument("gat_gru_model", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    prepare(
        args.output_dir, args.baseline_model, args.gat_gru_model,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

"""Benchmark heuristic PODEM against SmartATPG best and final policies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import torch

from rl_podem.cpp_bridge import _native_circuit_path
from rl_podem.smartatpg_artifacts import (
    checkpoint_policy,
    export_actor,
    export_descriptors,
    policy_from_state,
    snapshot_id,
)
from rl_podem.smartatpg_features import load_circuit_graph


PATTERNS = {
    "detected": r"#total number of detected faults = (\d+)",
    "total_faults": r"#total number of gate faults \(uncollapsed\) = (\d+)",
    "equivalent_detected": r"#number of equivalent detected faults = (\d+)",
    "equivalent_faults": r"#number of equivalent gate faults \(collapsed\) = (\d+)",
    "aborted": r"#number of aborted faults = (\d+)",
    "redundant": r"#number of redundant faults = (\d+)",
    "backtracks": r"#total number of backtracks = (\d+)",
    "backtrace_steps": r"#total number of backtrace steps = (\d+)",
    "test_vectors": r"#number of test vectors = (\d+)",
}
TIMING_PATTERN = re.compile(
    r"cputime for test pattern generation .*: ([0-9.]+)s ([0-9.]+)s"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def percentage_change(reference, candidate):
    reference = float(reference)
    candidate = float(candidate)
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - reference) / reference * 100.0


def _parse_native_output(output, log_path):
    record = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, output)
        if not match:
            raise RuntimeError(f"Missing {key} in native benchmark log: {log_path}")
        record[key] = int(match.group(1))
    timing = TIMING_PATTERN.search(output)
    if not timing:
        raise RuntimeError(f"Missing native timing in benchmark log: {log_path}")
    record["atpg_seconds"] = float(timing.group(1))
    record["native_total_seconds"] = float(timing.group(2))
    return record


def _stage_circuit_copy(item, output_dir):
    source = Path(item["circuit"]).resolve()
    destination = output_dir / "inputs" / item["name"] / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item["artifact_sha256"]["circuit"]
    if _sha256(source) != expected:
        raise ValueError(f"Circuit artifact hash changed: {source}")
    if destination.exists():
        if _sha256(destination) != expected:
            raise ValueError(
                f"Staged benchmark circuit differs; use a new output directory: "
                f"{destination}"
            )
    else:
        shutil.copy2(source, destination)
    return destination


def _prepare_models(checkpoint, manifest, output_dir):
    states = {
        "rl_best": checkpoint_policy(checkpoint, "best"),
        "rl_final": checkpoint_policy(checkpoint, "latest"),
    }
    graphs = {}
    preprocessing = []
    models = {"heuristic": {}}
    for model, state in states.items():
        model_dir = output_dir / "models" / model
        actor = model_dir / "actor.txt"
        export_actor(state, actor)
        policy = policy_from_state(state)
        models[model] = {
            "actor": actor,
            "snapshot": snapshot_id(state),
            "embeddings": {},
        }
        for item in manifest["circuits"]:
            name = item["name"]
            if name not in graphs:
                started = time.perf_counter()
                graphs[name] = load_circuit_graph(item["circuit"])
                preprocessing.append({
                    "circuit": name,
                    "operation": "graph_parse",
                    "seconds": time.perf_counter() - started,
                })
            embedding = model_dir / f"{name}.emb"
            started = time.perf_counter()
            export_descriptors(state, graphs[name], embedding, policy)
            preprocessing.append({
                "circuit": name,
                "model": model,
                "operation": "descriptor_export",
                "seconds": time.perf_counter() - started,
            })
            models[model]["embeddings"][name] = embedding
    return models, preprocessing


def _protocol(native_executable, manifest, models, repeats, seed, backtrack_limit):
    return {
        "format": "SMARTATPG_FINAL_BENCHMARK_V1",
        "native_executable": str(native_executable),
        "native_sha256": _sha256(native_executable),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "models": {
            name: (
                {"snapshot": value["snapshot"], "actor_sha256": _sha256(value["actor"])}
                if name != "heuristic"
                else None
            )
            for name, value in models.items()
        },
        "warmups_per_model_circuit": 1,
        "measured_repeats": repeats,
        "seed": seed,
        "backtrack_limit": backtrack_limit,
        "model_order": "rotated by repeat and circuit",
        "scope": "all faults on the same manifest circuits",
        "timing": "native ATPG interval and whole-process wall time; preprocessing excluded",
    }


def _run_records(
    manifest,
    models,
    native_executable,
    output_dir,
    repeats,
    seed,
    backtrack_limit,
):
    raw_path = output_dir / "raw_results.json"
    records = _load_json(raw_path, [])
    completed = {
        (int(item["repeat"]), item["circuit"], item["model"]) for item in records
    }
    staged_circuits = {
        item["name"]: _stage_circuit_copy(item, output_dir)
        for item in manifest["circuits"]
    }
    model_names = list(models)
    for repeat in range(repeats + 1):
        for circuit_index, item in enumerate(manifest["circuits"]):
            rotation = (repeat + circuit_index) % len(model_names)
            ordered_models = model_names[rotation:] + model_names[:rotation]
            for model in ordered_models:
                key = (repeat, item["name"], model)
                if key in completed:
                    continue
                command = [
                    str(native_executable),
                    "-bt", str(backtrack_limit),
                    "-seed", str(seed),
                    "-fault-map", _native_circuit_path(item["fault_map"]),
                ]
                if model != "heuristic":
                    command.extend([
                        "-rl-actor", _native_circuit_path(models[model]["actor"]),
                        "-rl-emb", _native_circuit_path(
                            models[model]["embeddings"][item["name"]]
                        ),
                        "-rl-embedding-backend", "smartatpg",
                        "-rl-mode", "backtrace_rl",
                    ])
                command.append(_native_circuit_path(staged_circuits[item["name"]]))
                started = time.perf_counter()
                result = subprocess.run(
                    command,
                    cwd=native_executable.parents[1],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
                wall_seconds = time.perf_counter() - started
                log_path = output_dir / "logs" / (
                    f"repeat{repeat}_{item['name']}_{model}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_bytes(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"Native benchmark failed: {log_path}")
                parsed = _parse_native_output(
                    result.stdout.decode("utf-8", errors="replace"), log_path
                )
                records.append({
                    "repeat": repeat,
                    "circuit": item["name"],
                    "model": model,
                    "wall_seconds": wall_seconds,
                    "command": command,
                    **parsed,
                })
                completed.add(key)
                _atomic_json_save(raw_path, records)
            print(
                f"BENCHMARK repeat={repeat}/{repeats} circuit={item['name']}",
                flush=True,
            )
    return records


def _summarize(records, manifest, model_names, repeats):
    rows = []
    for item in manifest["circuits"]:
        for model in model_names:
            samples = [
                record for record in records
                if int(record["repeat"]) > 0
                and record["circuit"] == item["name"]
                and record["model"] == model
            ]
            if len(samples) != repeats:
                raise RuntimeError(
                    f"Incomplete benchmark samples for {item['name']}/{model}."
                )
            row = {"circuit": item["name"], "model": model}
            for key in PATTERNS:
                values = {sample[key] for sample in samples}
                if len(values) != 1:
                    raise RuntimeError(
                        f"Nondeterministic {key} for {item['name']}/{model}."
                    )
                row[key] = samples[0][key]
            for key in ("atpg_seconds", "native_total_seconds", "wall_seconds"):
                row[key] = statistics.median(sample[key] for sample in samples)
            rows.append(row)

    numeric_keys = list(PATTERNS) + [
        "atpg_seconds", "native_total_seconds", "wall_seconds"
    ]
    totals = {
        model: {
            key: sum(row[key] for row in rows if row["model"] == model)
            for key in numeric_keys
        }
        for model in model_names
    }
    comparisons = {}
    heuristic = totals["heuristic"]
    for model in ("rl_best", "rl_final"):
        comparisons[model] = {}
        for key in ("backtracks", "backtrace_steps", "atpg_seconds", "wall_seconds"):
            change = percentage_change(heuristic[key], totals[model][key])
            comparisons[model][key] = {
                "change_percent": change,
                "reduction_percent": -change if change is not None else None,
            }
    return rows, totals, comparisons


def _write_reports(output_dir, rows, totals, comparisons, checkpoint):
    result = {
        "totals": totals,
        "relative_to_heuristic": comparisons,
        "best_label": checkpoint["best_label"],
        "best_score": checkpoint["best_score"],
    }
    _atomic_json_save(output_dir / "final_comparison.json", result)
    with (output_dir / "final_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Final SmartATPG Comparison",
        "",
        "All models use the same native executable, circuits, faults, seed, and backtrack limit.",
        "Positive reduction percentages indicate fewer steps or less time than heuristic.",
        "",
        "| Model | Detected / total | Aborted | Redundant | Backtracks | Backtrace steps | Test vectors | ATPG s | Wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("heuristic", "rl_best", "rl_final"):
        item = totals[model]
        lines.append(
            f"| {model} | {item['detected']}/{item['total_faults']} | "
            f"{item['aborted']} | {item['redundant']} | {item['backtracks']} | "
            f"{item['backtrace_steps']} | {item['test_vectors']} | "
            f"{item['atpg_seconds']:.6f} | {item['wall_seconds']:.6f} |"
        )
    lines.extend([
        "",
        "| Model | Backtrack reduction | Backtrace reduction | ATPG-time reduction | Wall-time reduction |",
        "|---|---:|---:|---:|---:|",
    ])
    for model in ("rl_best", "rl_final"):
        values = comparisons[model]
        formatted = []
        for key in ("backtracks", "backtrace_steps", "atpg_seconds", "wall_seconds"):
            value = values[key]["reduction_percent"]
            formatted.append("n/a" if value is None else f"{value:.3f}%")
        lines.append(f"| {model} | " + " | ".join(formatted) + " |")
    lines.extend([
        "",
        f"Best checkpoint label: `{checkpoint['best_label']}`.",
        "SmartATPG graph and descriptor preprocessing is reported separately and excluded from ATPG time.",
        "",
    ])
    (output_dir / "FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("COMPARISON " + json.dumps(result, sort_keys=True), flush=True)
    return result


def run_benchmark(
    manifest_path,
    checkpoint_path,
    native_executable,
    output_dir,
    repeats=5,
    seed=14,
    backtrack_limit=500,
):
    if repeats <= 0:
        raise ValueError("Benchmark repeats must be positive.")
    manifest = _load_json(manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    native_executable = Path(native_executable).resolve()
    if not native_executable.is_file():
        raise FileNotFoundError(f"Missing native executable: {native_executable}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models, preprocessing = _prepare_models(checkpoint, manifest, output_dir)
    protocol = _protocol(
        native_executable, manifest, models, repeats, seed, backtrack_limit
    )
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and _load_json(protocol_path) != protocol:
        raise ValueError("Benchmark protocol changed; use a new output directory.")
    _atomic_json_save(protocol_path, protocol)
    _atomic_json_save(output_dir / "preprocessing.json", preprocessing)
    records = _run_records(
        manifest,
        models,
        native_executable,
        output_dir,
        repeats,
        seed,
        backtrack_limit,
    )
    rows, totals, comparisons = _summarize(
        records, manifest, list(models), repeats
    )
    return _write_reports(output_dir, rows, totals, comparisons, checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("native_executable", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=500)
    args = parser.parse_args(argv)
    run_benchmark(
        args.manifest,
        args.checkpoint,
        args.native_executable,
        args.output_dir,
        args.repeats,
        args.seed,
        args.backtrack_limit,
    )


if __name__ == "__main__":
    main()

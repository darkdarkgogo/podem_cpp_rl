"""Reproducible training and paired native benchmark against the saved V6 actor."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE = ROOT / "artifacts/paper_v6_xor_filtered/training_manifest.json"
OLD_CHECKPOINT = ROOT / "artifacts/paper_v6_xor_filtered/training_state.pth"
OLD_ACTOR = ROOT / "artifacts/paper_v6_xor_filtered/actor_v2_best.txt"
NEW_CHECKPOINT = RUN / "training_state.pth"
NEW_ACTOR = RUN / "actor_v2_best.txt"
MANIFEST = RUN / "training_manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def prepare():
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    inputs = RUN / "inputs"
    inputs.mkdir(exist_ok=True)
    for item in manifest["circuits"]:
        for key in ("circuit", "fault_map"):
            source = Path(item[key])
            destination = inputs / source.name
            if destination.exists():
                if sha(source) != sha(destination):
                    raise RuntimeError(f"Existing experimental input changed: {destination}")
            else:
                shutil.copyfile(source, destination)
            item[key] = str(destination)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing experimental manifest changed.")
    else:
        save_json(MANIFEST, manifest)
    return manifest


def train():
    command = [sys.executable, "-u", str(ROOT / "scripts/train_curriculum.py"),
               str(MANIFEST), str(NEW_CHECKPOINT), str(NEW_ACTOR),
               "--advantage-method", "gae", "--gamma", "0.99", "--gae-lambda", "0.97",
               "--return-scale", "100", "--bc-epochs", "20", "--bc-batch-size", "256",
               "--stage-sweeps", "2", "2", "3", "--seed", "2026", "--log-rollouts"]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    metadata = {
        "source_manifest_sha256": sha(SOURCE), "manifest_sha256": sha(MANIFEST),
        "old_actor_sha256": sha(OLD_ACTOR), "old_checkpoint_sha256": sha(OLD_CHECKPOINT),
        "native_executable_sha256": sha(ROOT / "src/atpg.exe"),
        "training_command": command, "training_device": "cpu", "torch_threads": 1,
        "comparison": "new GAE pipeline vs existing V6, not GAE-only ablation",
        "code_sha256": {name: sha(ROOT / name) for name in (
            "python/rl_podem/advantages.py", "python/rl_podem/ppo.py",
            "python/rl_podem/curriculum.py", "scripts/train_curriculum.py")},
    }
    save_json(RUN / "experiment.json", metadata)
    print("TRAINING_START " + time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    started = time.perf_counter()
    with (RUN / "train.log").open("ab") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT)
    metadata["last_training_invocation_seconds"] = time.perf_counter() - started
    metadata["training_exit_code"] = result.returncode
    save_json(RUN / "experiment.json", metadata)
    if result.returncode:
        raise RuntimeError(f"Training failed with {result.returncode}; inspect train.log")
    print(f"TRAINING_FINISHED elapsed={metadata['last_training_invocation_seconds']:.1f}s", flush=True)


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


def benchmark(manifest, repeats):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rl_podem.cpp_bridge import _native_circuit_path

    directory = RUN / "benchmark"
    directory.mkdir(exist_ok=True)
    models = {"heuristic": None, "old_v6": OLD_ACTOR, "gae": NEW_ACTOR}
    records = []
    for repeat in range(repeats + 1):
        for circuit_index, item in enumerate(manifest["circuits"]):
            names = list(models)
            rotation = (repeat + circuit_index) % len(names)
            names = names[rotation:] + names[:rotation]
            for model in names:
                command = [str(ROOT / "src/atpg.exe"), "-bt", "500", "-seed", "14",
                           "-fault-map", _native_circuit_path(item["fault_map"])]
                if models[model] is not None:
                    command += ["-rl-emb", _native_circuit_path(item["embeddings"]),
                                "-rl-actor", _native_circuit_path(models[model]),
                                "-rl-mode", "backtrace_rl"]
                command.append(_native_circuit_path(item["circuit"]))
                started = time.perf_counter()
                result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=180)
                wall = time.perf_counter() - started
                log_path = directory / f"repeat{repeat}_{item['name']}_{model}.log"
                log_path.write_bytes(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"Native benchmark failed: {log_path}")
                output = result.stdout.decode("utf-8", errors="replace")
                record = dict(circuit=item["name"], model=model, repeat=repeat,
                              wall_seconds=wall, command=command)
                for key, pattern in PATTERNS.items():
                    match = re.search(pattern, output)
                    if not match:
                        raise RuntimeError(f"Missing {key} in {log_path}")
                    record[key] = int(match.group(1))
                timing = re.search(r"cputime for test pattern generation .*: ([0-9.]+)s ([0-9.]+)s", output)
                if not timing:
                    raise RuntimeError(f"Missing CPU time in {log_path}")
                record["cpu_seconds"] = float(timing.group(1))
                record["total_cpu_seconds"] = float(timing.group(2))
                records.append(record)
                save_json(directory / "raw_results.json", records)
            print(f"BENCHMARK repeat={repeat}/{repeats} circuit={item['name']}", flush=True)

    rows = []
    for item in manifest["circuits"]:
        for model in models:
            samples = [record for record in records if record["repeat"] > 0
                       and record["circuit"] == item["name"] and record["model"] == model]
            for key in PATTERNS:
                if len({sample[key] for sample in samples}) != 1:
                    raise RuntimeError(f"Nondeterministic counts for {item['name']}/{model}/{key}")
            row = dict(circuit=item["name"], model=model)
            row.update({key: samples[0][key] for key in PATTERNS})
            for key in ("cpu_seconds", "total_cpu_seconds", "wall_seconds"):
                row[key] = statistics.median(sample[key] for sample in samples)
                row[key + "_samples"] = [sample[key] for sample in samples]
            rows.append(row)
    save_json(directory / "summary.json", rows)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(directory / "protocol.json", {
        "warmups_per_circuit_model": 1, "measured_repeats": repeats,
        "seed": 14, "backtrack_limit": 500, "model_order": "rotated within each repeat/circuit",
        "cpu_time": "native reported test-pattern-generation interval",
        "wall_time": "end-to-end native process including loading and output",
        "evaluation_scope": "all faults on the same ten circuits (includes training/validation faults)",
        "model_sha256": {name: sha(path) for name, path in models.items() if path},
    })
    return rows


def report(rows):
    import torch

    old = torch.load(OLD_CHECKPOINT, map_location="cpu")
    new = torch.load(NEW_CHECKPOINT, map_location="cpu")
    totals = {}
    for model in ("heuristic", "old_v6", "gae"):
        selected = [row for row in rows if row["model"] == model]
        totals[model] = {key: sum(row[key] for row in selected) for key in (
            "detected", "total_faults", "equivalent_detected", "equivalent_faults",
            "aborted", "backtracks", "backtrace_steps", "cpu_seconds", "wall_seconds")}
    selection = {"old_best": old["best_label"], "old_validation_score": old["best_score"],
                 "new_best": new["best_label"], "new_validation_score": new["best_score"],
                 "new_config": new["config"], "training_units": len(new["progress"]),
                 "ppo_updates": new["agent"]["update_count"]}
    save_json(RUN / "comparison.json", {"selection": selection, "totals": totals})
    lines = ["# GAE vs V6: One-Seed Experiment", "",
             "Same ten-circuit fault split, seed 2026, BC 20 epochs, curriculum sweeps 2/2/3.",
             "CPU single-thread training; native evaluation uses seed 14 and backtrack limit 500.",
             "Each model/circuit gets one warmup and five interleaved measured runs; times are medians.",
             "This compares the new pipeline, including gamma/Advantage normalization/Critic initialization changes, not GAE alone.",
             "Full-circuit evaluation includes training/validation faults; it is not an unseen-circuit generalization test.", "",
             f"Old selected checkpoint: {old['best_label']}, validation score {old['best_score']}.",
             f"New selected checkpoint: {new['best_label']}, validation score {new['best_score']}.", "",
             "| Circuit | V6 detected | GAE detected | V6 abort | GAE abort | V6 CPU s | GAE CPU s | CPU change |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in dict.fromkeys(row["circuit"] for row in rows):
        before = next(row for row in rows if row["circuit"] == name and row["model"] == "old_v6")
        after = next(row for row in rows if row["circuit"] == name and row["model"] == "gae")
        delta = (after["cpu_seconds"] / before["cpu_seconds"] - 1) * 100 if before["cpu_seconds"] else 0
        lines.append(f"| {name} | {before['detected']} | {after['detected']} | {before['aborted']} | {after['aborted']} | {before['cpu_seconds']:.4f} | {after['cpu_seconds']:.4f} | {delta:+.1f}% |")
    lines += ["", "Aggregate (sum of per-circuit medians for time):", "", "```json", json.dumps(totals, indent=2), "```", "",
              "Detected counts are uncollapsed weighted fault counts; aborted counts are collapsed fault attempts.",
              "Keep coverage and runtime separate when deciding whether a model is better."]
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("EXPERIMENT_COMPLETE " + json.dumps({"selection": selection, "totals": totals}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    manifest = prepare()
    if not args.benchmark_only:
        train()
    rows = benchmark(manifest, args.repeats)
    report(rows)


if __name__ == "__main__":
    main()

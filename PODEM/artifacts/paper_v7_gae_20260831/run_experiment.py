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
NATIVE_EXE = ROOT / "build/atpg_rl_v4.exe"


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
        "native_executable": str(NATIVE_EXE),
        "native_executable_sha256": sha(NATIVE_EXE),
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


def benchmark(manifest, repeats, models=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rl_podem.cpp_bridge import _native_circuit_path

    os.chdir(ROOT)
    directory = RUN / "benchmark"
    if directory.exists():
        directory = RUN / ("benchmark_" + time.strftime("%Y%m%d_%H%M%S"))
    directory.mkdir()
    if models is None:
        models = {"heuristic": None, "old_v6": OLD_ACTOR, "gae": NEW_ACTOR,
                  "gae_final": RUN / "actor_v2_latest.txt"}
    records = []
    for repeat in range(repeats + 1):
        for circuit_index, item in enumerate(manifest["circuits"]):
            names = list(models)
            rotation = (repeat + circuit_index) % len(names)
            names = names[rotation:] + names[:rotation]
            for model in names:
                command = [str(NATIVE_EXE), "-bt", "500", "-seed", "14",
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
                    raise RuntimeError(f"Missing native elapsed time in {log_path}")
                record["atpg_seconds"] = float(timing.group(1))
                record["native_total_seconds"] = float(timing.group(2))
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
            for key in ("atpg_seconds", "native_total_seconds", "wall_seconds"):
                row[key] = statistics.median(sample[key] for sample in samples)
                row[key + "_samples"] = [sample[key] for sample in samples]
            rows.append(row)
    write_summary(directory, rows)
    save_json(directory / "protocol.json", {
        "warmups_per_circuit_model": 1, "measured_repeats": repeats,
        "seed": 14, "backtrack_limit": 500, "model_order": "rotated within each repeat/circuit",
        "native_executable": str(NATIVE_EXE), "native_executable_sha256": sha(NATIVE_EXE),
        "atpg_time": "native reported test-pattern-generation interval; Windows CRT elapsed time, not process CPU time",
        "wall_time": "end-to-end native process including loading and output",
        "evaluation_scope": "all faults on the same ten circuits (includes training/validation faults)",
        "model_sha256": {name: sha(path) for name, path in models.items() if path},
    })
    return rows, directory


def write_summary(directory, rows):
    save_json(directory / "summary.json", rows)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(rows, directory):
    import torch

    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    old = torch.load(OLD_CHECKPOINT, map_location="cpu")
    new = torch.load(NEW_CHECKPOINT, map_location="cpu")
    totals = {}
    for model in ("heuristic", "old_v6", "gae"):
        selected = [row for row in rows if row["model"] == model]
        totals[model] = {key: sum(row[key] for row in selected) for key in (
            "detected", "total_faults", "equivalent_detected", "equivalent_faults",
            "aborted", "backtracks", "backtrace_steps", "atpg_seconds", "wall_seconds")}
    selection = {"old_best": old["best_label"], "old_validation_score": old["best_score"],
                 "new_best": new["best_label"], "new_validation_score": new["best_score"],
                 "new_selected_is_ppo_trained": new["best_label"] != "behavior_cloning",
                 "new_validation_history": [
                     {"label": record["label"], "score": record["aggregate"]["score"]}
                     for record in new.get("validation_history", [])
                 ],
                 "new_config": new["config"], "training_units": len(new["progress"]),
                 "ppo_updates": new["agent"]["update_count"]}
    save_json(RUN / "comparison.json", {"selection": selection, "totals": totals,
                                        "benchmark_directory": str(directory)})
    lines = ["# GAE vs V6: One-Seed Experiment", "",
             "Same ten-circuit fault split, seed 2026, BC 20 epochs, curriculum sweeps 2/2/3.",
             "CPU single-thread training; native evaluation uses seed 14 and backtrack limit 500.",
             f"Each model/circuit gets {protocol['warmups_per_circuit_model']} warmup and {protocol['measured_repeats']} interleaved measured runs; times are medians.",
             "ATPG time is the native stage interval. On this Windows build, CRT clock() measures elapsed time, not process CPU time.",
             "Timing reference: [Microsoft CRT documentation](https://learn.microsoft.com/en-us/cpp/c-runtime-library/time-management?view=msvc-170).",
             "This compares the new pipeline, including gamma/Advantage normalization/Critic initialization changes, not GAE alone.",
             "Full-circuit evaluation includes training/validation faults; it is not an unseen-circuit generalization test.", "",
             f"Old selected checkpoint: {old['best_label']}, validation score {old['best_score']}.",
             f"New selected checkpoint: {new['best_label']}, validation score {new['best_score']}.", "",
             ("The new pipeline selected a post-PPO policy."
              if selection["new_selected_is_ppo_trained"] else
              "IMPORTANT: The new pipeline fell back to behavior cloning; no post-GAE/PPO checkpoint beat it on the selection score. The new selected benchmark columns describe this fallback, not the final PPO policy."), "",
             "## Detection and Aborts", "",
             "| Circuit | Heuristic detected | V6 detected | New selected detected | Heuristic abort | V6 abort | New selected abort |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in dict.fromkeys(row["circuit"] for row in rows):
        heuristic = next(row for row in rows if row["circuit"] == name and row["model"] == "heuristic")
        before = next(row for row in rows if row["circuit"] == name and row["model"] == "old_v6")
        after = next(row for row in rows if row["circuit"] == name and row["model"] == "gae")
        lines.append(f"| {name} | {heuristic['detected']} | {before['detected']} | {after['detected']} | {heuristic['aborted']} | {before['aborted']} | {after['aborted']} |")
    lines += ["", "## ATPG Elapsed Time", "",
              "| Circuit | Heuristic s | V6 s | New selected s | New vs V6 | New vs heuristic |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in dict.fromkeys(row["circuit"] for row in rows):
        heuristic = next(row for row in rows if row["circuit"] == name and row["model"] == "heuristic")
        before = next(row for row in rows if row["circuit"] == name and row["model"] == "old_v6")
        after = next(row for row in rows if row["circuit"] == name and row["model"] == "gae")
        delta = (after["atpg_seconds"] / before["atpg_seconds"] - 1) * 100 if before["atpg_seconds"] else 0
        vs_heuristic = (after["atpg_seconds"] / heuristic["atpg_seconds"] - 1) * 100 if heuristic["atpg_seconds"] else 0
        lines.append(f"| {name} | {heuristic['atpg_seconds']:.4f} | {before['atpg_seconds']:.4f} | {after['atpg_seconds']:.4f} | {delta:+.1f}% | {vs_heuristic:+.1f}% |")
    lines += ["", "Aggregate (sum of per-circuit medians for time):", "", "```json", json.dumps(totals, indent=2), "```", "",
              "Detected counts are uncollapsed weighted fault counts; aborted counts are collapsed fault attempts.",
              "Keep coverage and runtime separate when deciding whether a model is better."]
    lines += ["", "## New Pipeline Validation History", "",
              "| Checkpoint | Detected / 500 | Aborted | Mean backtrack ratio | Mean backtrace ratio |",
              "|---|---:|---:|---:|---:|"]
    for record in new.get("validation_history", []):
        aggregate = record["aggregate"]
        lines.append(f"| {record['label']} | {aggregate['detected']} | {aggregate['aborted']} | {aggregate['mean_backtrack_ratio']:.4f} | {aggregate['mean_backtrace_ratio']:.4f} |")
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(RUN / "comparison.md", directory / "comparison.md")
    shutil.copyfile(RUN / "comparison.json", directory / "comparison.json")
    print("EXPERIMENT_COMPLETE " + json.dumps({"selection": selection, "totals": totals}), flush=True)


def refresh_report(directory):
    """Correct timing labels from the already-launched runner without rerunning measurements."""
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if "cpu_time" in protocol:
        replacements = {"total_cpu_seconds": "native_total_seconds", "cpu_seconds": "atpg_seconds"}
        for name in ("raw_results.json", "summary.json"):
            path = directory / name
            shutil.copyfile(path, directory / (name + ".original"))
            records = json.loads(path.read_text(encoding="utf-8"))
            renamed = []
            for record in records:
                updated = {}
                for key, value in record.items():
                    for before, after in replacements.items():
                        key = key.replace(before, after)
                    updated[key] = value
                renamed.append(updated)
            save_json(path, renamed)
        shutil.copyfile(protocol_path, directory / "protocol.json.original")
        protocol.pop("cpu_time")
        protocol["atpg_time"] = "native reported test-pattern-generation interval; Windows CRT elapsed time, not process CPU time"
        save_json(protocol_path, protocol)
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    write_summary(directory, rows)
    report(rows, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--report-only", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.report_only is not None:
        if args.benchmark_only:
            parser.error("--report-only cannot be combined with --benchmark-only")
        refresh_report(args.report_only.resolve())
        return
    manifest = prepare()
    if not args.benchmark_only:
        train()
    rows, directory = benchmark(manifest, args.repeats)
    report(rows, directory)


if __name__ == "__main__":
    main()

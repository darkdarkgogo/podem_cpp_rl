"""Paired five-model native evaluation of the completed SmartATPG experiment."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OLD = ROOT / "artifacts/paper_v7_gae_20260831"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
MODELS = ("heuristic", "deepgate_gae_best", "deepgate_gae_final", "smartatpg_best", "smartatpg_final")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_training():
    import torch
    from train_curriculum import _stage_fault_ids
    state = torch.load(RUN / "training_state.pth", map_location="cpu")
    old = torch.load(OLD / "training_state.pth", map_location="cpu")
    manifest = load(RUN / "experiment_manifest.json")
    assert state["manifest_hash"] == sha(RUN / "experiment_manifest.json")
    items = {item["name"]: item for item in manifest["circuits"]}
    assert len(state["progress"]) == 150
    assert len(state["validation_history"]) == 8
    assert len(state["pretraining"]["history"]) == 20
    assert state["config"]["advantage_method"] == "gae"
    assert state["agent"]["hyperparameters"] == old["agent"]["hyperparameters"]
    for key, value in old["config"].items():
        assert state["config"][key] == value, (key, state["config"][key], value)
    units = state["progress"]
    episodes = sum(u["summary"]["episodes"] for u in units)
    steps = sum(u["summary"]["decisions"] for u in units)
    assert episodes == 4200
    for unit in units:
        assert unit["learning"]["steps"] == unit["summary"]["decisions"]
        for key, value in unit["learning"].items():
            if isinstance(value, (int, float)):
                assert math.isfinite(value), (key, value)
    for section in ("policy", "policy_old", "rnd"):
        assert all(bool(torch.isfinite(tensor).all()) for tensor in state["agent"][section].values())
    expected_keys = set()
    for unit in units:
        faults = _stage_fault_ids(items[unit["circuit"]], unit["stage"], unit["sweep"], state["config"]["seed"])[unit["difficulty"]]
        assert len(faults) == unit["summary"]["episodes"]
        expected_keys.update((unit["stage_name"], unit["sweep"], unit["circuit"], fault) for fault in faults)
    assert len(expected_keys) == episodes
    # A resumed unit can have already emitted rollouts before its atomic save.
    # Retain the last attempt, but only for units present in the final checkpoint.
    records = {}
    raw_count = 0
    malformed_count = 0
    for line in (RUN / "train.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("ROLLOUT_RESULT "):
            raw_count += 1
            try:
                record = json.loads(line.split(" ", 1)[1])
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            key = (record["stage"], record["sweep"], record["circuit"], record["fault_id"])
            if key in expected_keys:
                records[key] = record
    assert set(records) == expected_keys
    rollouts = list(records.values())
    assert all(math.isfinite(value) for record in rollouts for value in record.values()
               if isinstance(value, (int, float)))
    assert len(rollouts) == episodes
    assert sum(r["steps"] for r in rollouts) == steps
    assert sum(bool(r["updated"]) for r in rollouts) == state["agent"]["update_count"]
    mismatches = sum(r["counter_mismatch_episode"] for r in rollouts)
    assert mismatches == 0
    audit = {"episodes": episodes, "decision_steps": steps,
             "ppo_updates": state["agent"]["update_count"], "work_units": len(units),
             "best_label": state["best_label"], "best_score": state["best_score"],
             "bc_best_accuracy": state["pretraining"]["best_validation_accuracy"],
             "old_best_label": old["best_label"], "old_best_score": old["best_score"],
             "old_bc_best_accuracy": old["pretraining"]["best_validation_accuracy"],
             "shared_agent_hyperparameters": state["agent"]["hyperparameters"],
             "validation_history": state["validation_history"],
             "old_validation_history": old["validation_history"],
             "raw_rollout_log_records": raw_count,
             "malformed_rollout_log_records": malformed_count,
             "duplicate_or_uncommitted_log_records": raw_count - len(rollouts),
             "counter_mismatch_episodes": mismatches, "numeric_metrics_finite": True}
    save(RUN / "training_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    assert args.repeats > 0
    os.chdir(ROOT)
    experiment = load(RUN / "experiment.json")
    assert experiment.get("exit_code") == 0 and experiment["code_unchanged"]
    assert sha(RUN / "experiment_manifest.json") == experiment["manifest_sha256"]
    assert all(sha(ROOT / path) == digest for path, digest in experiment["code_sha256"].items())
    audit = audit_training()
    spec = importlib.util.spec_from_file_location("old_experiment", OLD / "run_experiment.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from rl_podem.cpp_bridge import _native_circuit_path
    from rl_podem.smartatpg_features import load_circuit_graph
    from rl_podem.smartatpg_artifacts import checkpoint_policy, policy_from_state, export_descriptors
    import cpp_podem
    import torch
    torch.set_num_threads(1)
    manifest = load(RUN / "experiment_manifest.json")
    old_manifest = load(OLD / "training_manifest.json")
    old_items = {i["name"]: i for i in old_manifest["circuits"]}
    from train_curriculum import _validate_manifest
    _validate_manifest(manifest)
    assert len(manifest["circuits"]) == len(old_items) == 10
    assert {i["name"] for i in manifest["circuits"]} == set(old_items)
    snapshots = {"smartatpg_best": load(RUN / "actor_best.txt.json"),
                 "smartatpg_final": load(RUN / "actor_latest.txt.json")}
    executable = (ROOT / "build/atpg_rl_smartatpg.exe").resolve()
    # The existing distutils build appended .exe to an already-suffixed name.
    # Hash and execute the actual file, not Windows' implicit launch fallback.
    if not executable.is_file():
        executable = executable.with_suffix(".exe.exe")
    assert executable.is_file(), executable
    print(f"NATIVE_EXECUTABLE {executable}", flush=True)
    directory = RUN / ("benchmark_" + time.strftime("%Y%m%d_%H%M%S"))
    directory.mkdir()
    checkpoint = torch.load(RUN / "training_state.pth", map_location="cpu")
    states = {"smartatpg_best": checkpoint_policy(checkpoint, "best"),
              "smartatpg_final": checkpoint_policy(checkpoint, "latest")}
    policies = {name: policy_from_state(state) for name, state in states.items()}
    preprocessing = []
    pairs = {}
    artifact_hashes = {}
    for item in manifest["circuits"]:
        name = item["name"]
        started = time.perf_counter()
        graph = load_circuit_graph(item["circuit"])
        graph_seconds = time.perf_counter() - started
        pairs[name] = {"heuristic": None,
                       "deepgate_gae_best": (OLD / "actor_v2_best.txt", old_items[name]["embeddings"], "deepgate"),
                       "deepgate_gae_final": (OLD / "actor_v2_latest.txt", old_items[name]["embeddings"], "deepgate")}
        for model, snapshot in snapshots.items():
            entry = snapshot["circuits"][name]
            assert entry["circuit_hash"] == graph.circuit_hash
            pairs[name][model] = (snapshot["actor"], entry["embeddings"], "smartatpg")
            cpp_podem.validate_actor_artifacts(_native_circuit_path(entry["embeddings"]),
                _native_circuit_path(snapshot["actor"]), graph.circuit_hash, list(graph.names), "smartatpg")
            probe = directory / "offline_export_probe" / f"{name}_{model}.emb"
            started = time.perf_counter()
            export_descriptors(states[model], graph, probe, policies[model])
            export_seconds = time.perf_counter() - started
            assert sha(probe) == sha(entry["embeddings"]), (name, model, "export mismatch")
            preprocessing.append({"circuit": name, "model": model,
                                  "graph_parse_seconds": graph_seconds,
                                  "graph_encode_and_descriptor_export_seconds": export_seconds})
        for pair in pairs[name].values():
            if pair:
                for path in pair[:2]:
                    artifact_hashes[str(path)] = sha(path)
    protocol = {"warmups_per_model_circuit": 1, "measured_repeats": args.repeats,
                "seed": 14, "backtrack_limit": 500, "models": list(MODELS),
                "native_executable": str(executable), "native_sha256": sha(executable),
                "model_order": "rotated within each circuit and repeat",
                "scope": "all faults on the same ten circuits; includes training/validation faults; not unseen-circuit generalization",
                "timing": "native ATPG interval and whole-process wall time; excludes offline graph encoding/export",
                "artifact_sha256": artifact_hashes}
    save(directory / "preprocessing.json", preprocessing)
    save(directory / "protocol.json", protocol)
    records = []
    for repeat in range(args.repeats + 1):
        for index, item in enumerate(manifest["circuits"]):
            shift = (repeat + index) % len(MODELS)
            for model in MODELS[shift:] + MODELS[:shift]:
                command = [str(executable), "-bt", "500", "-seed", "14", "-fault-map",
                           _native_circuit_path(item["fault_map"])]
                pair = pairs[item["name"]][model]
                if pair:
                    actor, embeddings, backend = pair
                    command += ["-rl-actor", _native_circuit_path(actor),
                                "-rl-emb", _native_circuit_path(embeddings),
                                "-rl-embedding-backend", backend, "-rl-mode", "backtrace_rl"]
                command.append(_native_circuit_path(item["circuit"]))
                start = time.perf_counter()
                result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=300)
                wall = time.perf_counter() - start
                log = directory / f"repeat{repeat}_{item['name']}_{model}.log"
                log.write_bytes(result.stdout)
                assert result.returncode == 0, log
                output = result.stdout.decode("utf-8", errors="replace")
                record = {"model": model, "circuit": item["name"], "repeat": repeat,
                          "wall_seconds": wall, "command": command}
                for key, pattern in helper.PATTERNS.items():
                    match = re.search(pattern, output)
                    assert match, (log, key)
                    record[key] = int(match.group(1))
                timing = re.search(r"cputime for test pattern generation .*: ([0-9.]+)s ([0-9.]+)s", output)
                assert timing, log
                record["atpg_seconds"] = float(timing.group(1))
                record["native_total_seconds"] = float(timing.group(2))
                records.append(record)
                save(directory / "raw_results.json", records)
            print(f"BENCHMARK repeat={repeat}/{args.repeats} circuit={item['name']}", flush=True)
    rows = []
    for item in manifest["circuits"]:
        for model in MODELS:
            samples = [r for r in records if r["repeat"] > 0 and r["circuit"] == item["name"] and r["model"] == model]
            assert len(samples) == args.repeats
            row = {"circuit": item["name"], "model": model}
            for key in helper.PATTERNS:
                assert len({s[key] for s in samples}) == 1, (item["name"], model, key)
                row[key] = samples[0][key]
            for key in ("atpg_seconds", "native_total_seconds", "wall_seconds"):
                row[key] = statistics.median(s[key] for s in samples)
                row[key + "_samples"] = [s[key] for s in samples]
            rows.append(row)
    helper.write_summary(directory, rows)
    assert {(r["circuit"], r["model"]) for r in rows} == {
        (name, model) for name in old_items for model in MODELS}
    old_rows = load(OLD / "benchmark_20260831_111818/summary.json")
    aliases = {"heuristic": "heuristic", "deepgate_gae_best": "gae", "deepgate_gae_final": "gae_final"}
    matched_rows = 0
    for row in rows:
        if row["model"] in aliases:
            old = next(r for r in old_rows if r["circuit"] == row["circuit"] and r["model"] == aliases[row["model"]])
            assert all(row[k] == old[k] for k in helper.PATTERNS), (row["circuit"], row["model"], "baseline changed")
            matched_rows += 1
    assert matched_rows == len(old_items) * len(aliases)
    assert all(sha(path) == digest for path, digest in artifact_hashes.items())
    assert sha(executable) == protocol["native_sha256"]
    keys = list(helper.PATTERNS) + ["atpg_seconds", "wall_seconds"]
    totals = {model: {key: sum(r[key] for r in rows if r["model"] == model) for key in keys} for model in MODELS}
    save(RUN / "comparison.json", {"totals": totals, "benchmark_directory": str(directory),
                                  "training": audit, "historical_baseline_count_rows_matched": matched_rows,
                                  "artifact_hashes_unchanged": True})
    lines = ["# SmartATPG-Style Backend: One-Seed Experiment", "",
             "Same ten circuits, fault splits, teachers, seed 2026, BC 20 epochs and GAE sweeps 2/2/3.",
             "CPU single-thread training; this is a backend pipeline comparison, not exact paper reproduction.",
             f"SmartATPG best selection: {audit['best_label']}; previous DeepGate best: {audit['old_best_label']}.",
             "Best may be a BC fallback; final always denotes the last PPO policy.", "",
             "## Native Full-Circuit Evaluation", "",
             f"All five models use the same executable, seed 14, bt=500, one warmup and {args.repeats} rotated measured runs per circuit.",
             "Times sum per-circuit medians. ATPG is the native elapsed stage interval; wall time includes loading/inference/output, not offline encoding/export.",
             "This scope includes training/validation faults, so it does not establish unseen-circuit generalization.", "",
             "| Model | Detected / total | Aborted | Backtracks | Backtrace steps | ATPG s | Wall s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = totals[model]
        lines.append(f"| {model} | {row['detected']}/{row['total_faults']} | {row['aborted']} | {row['backtracks']} | {row['backtrace_steps']} | {row['atpg_seconds']:.3f} | {row['wall_seconds']:.3f} |")
    lines += ["", "## Per-Circuit Final Policies", "",
              "| Circuit | Heuristic detected | DeepGate final detected | Smart final detected | Heuristic BT | DeepGate final BT | Smart final BT |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for item in manifest["circuits"]:
        selected = {r["model"]: r for r in rows if r["circuit"] == item["name"]}
        models = ("heuristic", "deepgate_gae_final", "smartatpg_final")
        values = [selected[m][key] for key in ("detected", "backtracks") for m in models]
        lines.append("| " + item["name"] + " | " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Fixed 500-Fault Validation", "", "| Stage | DeepGate detected | Smart detected |", "|---|---:|---:|"]
    for validation in audit["validation_history"]:
        old = next(r for r in audit["old_validation_history"] if r["label"] == validation["label"])
        lines.append(f"| {validation['label']} | {old['aggregate']['detected']} | {validation['aggregate']['detected']} |")
    lines += ["", "## Audit", "",
              f"4200 episodes, 150 work units, {audit['ppo_updates']} PPO updates and {audit['decision_steps']} decisions; all checked metrics/weights finite and reward counter mismatches zero.",
              f"All {matched_rows} historical heuristic/DeepGate rows match structural counters on the rebuilt executable; inference artifact hashes unchanged.",
              "One training seed only; repeated benchmark timings are not independent training seeds.", ""]
    (RUN / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("COMPARISON " + json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()

"""Bounded two-circuit BC/MC/GAE, resume, export and native parity verification."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import cpp_podem

from rl_podem.cpp_bridge import CppPodemBacktraceV2Evaluator, _load_cpp_embedding_artifact, _native_circuit_path
from rl_podem.smartatpg import SmartATPGPPOAgent
from rl_podem.smartatpg_features import load_circuit_graph
from rl_podem.smartatpg_artifacts import checkpoint_policy, policy_from_state
from prepare_curriculum_training import _from_source_manifest
from train_curriculum import _sha256_file, _validate_manifest, _atomic_torch_save, main as train_main


def fixture(source, directory):
    converted = _from_source_manifest(source, directory / "converted", "smartatpg")
    manifest = json.loads(converted.read_text(encoding="utf-8"))
    circuits = [copy.deepcopy(item) for item in manifest["circuits"] if item["name"] in ("c432", "c499")]
    assert len(circuits) == 2
    for item in circuits:
        for kind in ("training", "validation"):
            item[f"{kind}_faults"] = [min(
                (f for f in item[f"{kind}_faults"] if f["difficulty"] == difficulty),
                key=lambda f: (f["backtracks"], f["backtrace_steps"]))
                for difficulty in ("easy", "medium", "hard")]
        for key in ("circuit", "fault_map", "profile"):
            destination = directory / Path(item[key]).name
            shutil.copyfile(item[key], destination)
            item[key] = str(destination.resolve())
            item["artifact_sha256"][key] = _sha256_file(destination)
    manifest["circuits"] = circuits
    manifest["smoke"] = True
    for kind in ("training", "validation"):
        samples = json.loads(Path(manifest[f"teacher_{kind}"]).read_text(encoding="utf-8"))
        samples = [s for s in samples if s["circuit"] in ("c432", "c499")]
        path = directory / f"teacher_{kind}.json"
        path.write_text(json.dumps(samples), encoding="utf-8")
        manifest[f"teacher_{kind}"] = str(path.resolve())
        manifest["teacher_sha256"][kind] = _sha256_file(path)
    path = directory / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _validate_manifest(manifest)
    return path, manifest


def verify_method(method, manifest_path, manifest, directory, executable):
    checkpoint = directory / f"{method}.pth"
    actor = directory / f"{method}_best.txt"
    latest = directory / f"{method}_latest.txt"
    args = [str(manifest_path), str(checkpoint), str(actor), "--embedding-backend", "smartatpg",
            "--advantage-method", method, "--bc-epochs", "2", "--stage-sweeps", "1", "1", "1", "--log-rollouts"]
    with (directory / f"{method}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        train_main(args)
        train_main(args)
    state = torch.load(checkpoint, map_location="cpu")
    assert len(state["progress"]) == 12 and len(state["pretraining"]["history"]) == 2
    updates = sum(unit["learning"]["episodes_with_updates"] for unit in state["progress"])
    assert state["agent"]["update_count"] == updates and updates > 0
    for unit in state["progress"]:
        assert unit["learning"]["steps"] == unit["summary"]["decisions"]
        for value in unit["learning"].values():
            if isinstance(value, (int, float)):
                assert math.isfinite(value)
    graphs = {item["name"]: load_circuit_graph(item["circuit"]) for item in manifest["circuits"]}
    agent = SmartATPGPPOAgent(graphs, **state["agent"]["hyperparameters"])
    agent.load_training_state_dict(state["agent"])
    snapshot = json.loads(latest.with_suffix(".txt.json").read_text(encoding="utf-8"))
    parity_count, native_runs = 0, 0
    for item in manifest["circuits"]:
        graph = graphs[item["name"]]
        embedding_path = snapshot["circuits"][item["name"]]["embeddings"]
        actor_path = _native_circuit_path(snapshot["actor"])
        _, table, metadata = _load_cpp_embedding_artifact(embedding_path, expected_backend="smartatpg", include_metadata=True)
        assert metadata["snapshot"] == snapshot["snapshot"]
        cpp_podem.validate_actor_artifacts(_native_circuit_path(embedding_path), actor_path,
                                          graph.circuit_hash, list(graph.names), "smartatpg")
        policy = policy_from_state(checkpoint_policy(state, "latest"))
        with torch.no_grad():
            live_descriptors = policy.descriptors(graph, list(range(len(graph.names))))
            for index, name in enumerate(graph.names):
                torch.testing.assert_close(live_descriptors[index], table[name], atol=1e-5, rtol=1e-4)
                if index % max(1, len(graph.names) // 24) != 0:
                    continue
                for value in (0, 1):
                    expected = policy.batch_logits(live_descriptors[index:index + 1], [value])[0][0]
                    actual = torch.tensor(cpp_podem.score_actor_v2(actor_path, table[name].tolist(), value))
                    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)
                    assert int(expected.argmax()) == int(actual.argmax())
                    parity_count += 1
        evaluator = CppPodemBacktraceV2Evaluator(graph, agent)
        ids = [f["fault_id"] for f in item["validation_faults"]]
        live = evaluator.run(item["circuit"], fault_ids=ids, seed=2026, fault_map_path=item["fault_map"])

        def native_callback(request):
            logits = cpp_podem.score_actor_v2(actor_path, table[request["objective_name"]].tolist(), request["objective_value"])
            return int(logits[1] > logits[0])

        frozen = dict(cpp_podem.run_stuck_at(_native_circuit_path(item["circuit"]), native_callback,
            None, 500, 2026, ids, True, "backtrace_rl", _native_circuit_path(item["fault_map"])))
        assert live == frozen, (live, frozen)
        assert live == state["validation_history"][-1]["circuits"][item["name"]]
        command = [str(executable.resolve()), "-bt", "500", "-seed", "14", "-fault-map",
                   _native_circuit_path(item["fault_map"]), "-rl-emb", _native_circuit_path(embedding_path),
                   "-rl-actor", actor_path, "-rl-embedding-backend", "smartatpg", _native_circuit_path(item["circuit"])]
        run = subprocess.run(command, capture_output=True, timeout=90)
        (directory / f"{method}_{item['name']}_native.log").write_bytes(run.stdout + run.stderr)
        assert run.returncode == 0, run.stderr.decode(errors="replace")
        bad = list(command)
        bad[bad.index("smartatpg")] = "deepgate"
        rejection = subprocess.run(bad, capture_output=True, timeout=30)
        assert rejection.returncode != 0 and b"backend" in rejection.stderr
        native_runs += 1

    interrupted = directory / f"interrupted_{method}.pth"
    interrupted_args = [str(manifest_path), str(interrupted), str(directory / f"interrupted_{method}_best.txt"), *args[3:]]

    class Interruption(RuntimeError):
        pass

    def save_interrupt(path, value):
        _atomic_torch_save(path, value)
        if path == interrupted and len(value.get("progress", [])) == 3:
            raise Interruption()

    with (directory / f"{method}_interrupted.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        try:
            with patch("train_curriculum._atomic_torch_save", side_effect=save_interrupt):
                train_main(interrupted_args)
        except Interruption:
            pass
        else:
            raise AssertionError("Failed to interrupt persisted work unit")
        train_main(interrupted_args)
    resumed = torch.load(interrupted, map_location="cpu")
    assert resumed["progress"] == state["progress"]
    for section in ("policy", "policy_old", "rnd"):
        for name, tensor in state["agent"][section].items():
            torch.testing.assert_close(tensor, resumed["agent"][section][name], atol=0, rtol=0)
    assert not any(name.startswith("deepgate_recgnn") for name in sys.modules)
    return {"method": method, "updates": updates, "work_units": 12, "logit_pairs": parity_count,
            "native_runs": native_runs, "interrupted_resume": "exact", "deepgate_imported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("artifacts/paper_v6_xor_filtered/training_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smartatpg_backend_smoke_20260831"))
    parser.add_argument("--native-exe", type=Path, default=Path("build/atpg_rl_smartatpg.exe"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, manifest = fixture(args.source_manifest, args.output_dir)
    results = []
    for method in ("mc", "gae"):
        print(f"SMARTATPG_SMOKE_START method={method}", flush=True)
        results.append(verify_method(method, manifest_path, manifest, args.output_dir, args.native_exe))
        print("SMARTATPG_SMOKE " + json.dumps(results[-1], sort_keys=True), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

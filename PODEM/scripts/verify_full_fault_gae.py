"""Isolated native curriculum smoke: BC, MC/GAE, checkpoint resume and evaluation."""

import argparse
import copy
import io
import json
import math
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import cpp_podem

from rl_podem.cpp_bridge import CppPodemBacktraceV2Evaluator
from rl_podem.curriculum import load_embedding_tables
from train_curriculum import (
    _atomic_torch_save,
    _agent_from_hyperparameters,
    _sha256_file,
    _validate_manifest,
    main as train_main,
)


def prepare_fixture(manifest_path, directory):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = next(item for item in manifest["circuits"] if item["name"] == "c432")
    item = copy.deepcopy(source)
    for kind in ("training", "validation"):
        item[f"{kind}_faults"] = [
            min((fault for fault in source[f"{kind}_faults"]
                 if fault["difficulty"] == difficulty),
                key=lambda fault: (fault["backtracks"], fault["backtrace_steps"]))
            for difficulty in ("easy", "medium", "hard")
        ]
    # Keep all native outputs, including .bench.uf, outside tracked artifacts.
    for key in ("circuit", "fault_map", "embeddings", "profile"):
        destination = directory / Path(item[key]).name
        shutil.copyfile(item[key], destination)
        item[key] = str(destination)
        item["artifact_sha256"][key] = _sha256_file(destination)
    manifest["circuits"] = [item]
    for kind in ("training", "validation"):
        samples = json.loads(Path(manifest[f"teacher_{kind}"]).read_text(encoding="utf-8"))
        samples = [sample for sample in samples if sample["circuit"] == "c432"]
        destination = directory / f"teacher_{kind}.json"
        destination.write_text(json.dumps(samples), encoding="utf-8")
        manifest[f"teacher_{kind}"] = str(destination)
        manifest["teacher_sha256"][kind] = _sha256_file(destination)
    destination = directory / "manifest.json"
    destination.write_text(json.dumps(manifest), encoding="utf-8")
    _validate_manifest(manifest)
    return destination, manifest


def verify_method(method, manifest_path, manifest, directory):
    checkpoint = directory / f"{method}.pth"
    actor = directory / f"{method}_best.txt"
    args = [str(manifest_path), str(checkpoint), str(actor),
            "--advantage-method", method, "--bc-epochs", "2",
            "--stage-sweeps", "1", "1", "1", "--log-rollouts"]
    output = io.StringIO()
    with redirect_stdout(output):
        train_main(args)
    state = torch.load(checkpoint, map_location="cpu")
    assert state["config"]["advantage_method"] == method
    assert state["config"]["normalize_returns"] is False
    assert len(state["pretraining"]["history"]) == 2
    assert len(state["progress"]) == 6
    rollouts = [json.loads(line.removeprefix("ROLLOUT_RESULT "))
                for line in output.getvalue().splitlines()
                if line.startswith("ROLLOUT_RESULT ")]
    assert len(rollouts) == 6
    assert max(item["steps"] for item in rollouts) > 1
    updates = 0
    for record in state["progress"]:
        metrics = record["learning"]
        assert metrics["advantage_method"] == method
        assert metrics["steps"] == record["summary"]["decisions"]
        assert metrics["rollout_steps"]["count"] == record["summary"]["episodes"]
        assert math.isclose(metrics["rollout_steps"]["mean"] * metrics["rollout_steps"]["count"],
                            metrics["steps"])
        updates += metrics["episodes_with_updates"]
        for value in metrics.values():
            if isinstance(value, (float, int)):
                assert math.isfinite(value)
    assert state["agent"]["update_count"] == updates
    assert actor.is_file() and actor.with_name(f"{method}_latest.txt").is_file()

    resume_output = io.StringIO()
    with redirect_stdout(resume_output):
        train_main(args)
    assert "RESUME units=6" in resume_output.getvalue()
    assert "TRAIN_START" not in resume_output.getvalue()
    resumed = torch.load(checkpoint, map_location="cpu")
    assert resumed["agent"]["update_count"] == updates
    for key, tensor in state["agent"]["policy"].items():
        torch.testing.assert_close(tensor, resumed["agent"]["policy"][key])

    circuits = manifest["circuits"]
    tables = load_embedding_tables(circuits)
    dimension = next(iter(tables["c432"].values())).numel()
    agent = _agent_from_hyperparameters(dimension, state["agent"]["hyperparameters"])
    agent.load_training_state_dict(state["agent"])
    for embedding in list(tables["c432"].values())[:3]:
        for objective_value in (0, 1):
            logits, _ = agent.policy_old.backtrace_logits(embedding, objective_value)
            native_logits = cpp_podem.score_actor_v2(
                str(actor.with_name(f"{method}_latest.txt")),
                embedding.tolist(), objective_value,
            )
            torch.testing.assert_close(logits.detach().cpu(), torch.tensor(native_logits),
                                       atol=1e-5, rtol=1e-5)
    item = circuits[0]
    evaluator = CppPodemBacktraceV2Evaluator(item["embeddings"], agent=agent)
    summary = evaluator.run(
        item["circuit"], backtrack_limit=500, seed=2026,
        fault_ids=[fault["fault_id"] for fault in item["validation_faults"]],
        quiet=True, fault_map_path=item["fault_map"],
    )
    assert summary == state["validation_history"][-1]["circuits"]["c432"]
    assert agent.update_count == updates and not agent.buffer.steps

    # Interrupt after a persisted work unit, before that stage's validation.
    interrupted_checkpoint = directory / f"interrupted_{method}.pth"
    interrupted_actor = directory / f"interrupted_{method}_best.txt"
    interrupted_args = [str(manifest_path), str(interrupted_checkpoint),
                        str(interrupted_actor), *args[3:]]

    class SimulatedInterruption(RuntimeError):
        pass

    def save_then_interrupt(path, value):
        _atomic_torch_save(path, value)
        if path == interrupted_checkpoint and len(value.get("progress", [])) == 3:
            raise SimulatedInterruption("Checkpoint saved; simulating interrupted training.")

    try:
        with redirect_stdout(io.StringIO()), patch(
            "train_curriculum._atomic_torch_save", side_effect=save_then_interrupt
        ):
            train_main(interrupted_args)
    except SimulatedInterruption:
        pass
    else:
        raise AssertionError("Smoke did not interrupt the intended curriculum work unit.")
    with redirect_stdout(io.StringIO()):
        train_main(interrupted_args)
    interrupted = torch.load(interrupted_checkpoint, map_location="cpu")
    assert interrupted["agent"]["update_count"] == updates
    assert interrupted["progress"] == state["progress"]
    for key in ("policy", "policy_old", "rnd"):
        for name, tensor in state["agent"][key].items():
            torch.testing.assert_close(tensor, interrupted["agent"][key][name], atol=0, rtol=0)
    return {
        "method": method, "updates": updates,
        "rollout_steps": [item["steps"] for item in rollouts],
        "validation_detected": summary["detected"],
        "checkpoint_resume": "passed",
        "native_actor_parity": "passed",
        "interrupted_resume": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "artifacts/paper_v6_xor_filtered/training_manifest.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="podem_full_fault_gae_") as temporary:
        directory = Path(temporary)
        manifest_path, manifest = prepare_fixture(args.manifest, directory)
        for method in ("mc", "gae"):
            print("GAE_SMOKE " + json.dumps(
                verify_method(method, manifest_path, manifest, directory), sort_keys=True
            ), flush=True)


if __name__ == "__main__":
    main()

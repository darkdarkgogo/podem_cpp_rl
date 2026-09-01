"""Run the approved full SmartATPG experiment without altering prior artifacts."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    source = RUN / "training_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "artifacts/paper_v7_gae_20260831/training_manifest.json").read_text(encoding="utf-8"))
    old_circuits = {item["name"]: item for item in previous["circuits"]}
    assert manifest["teacher_sha256"] == previous["teacher_sha256"]
    assert len(manifest["circuits"]) == len(old_circuits) == 10
    inputs = RUN / "inputs"
    inputs.mkdir(exist_ok=True)
    for item in manifest["circuits"]:
        old = old_circuits[item["name"]]
        for key in ("training_faults", "validation_faults"):
            assert item[key] == old[key], (item["name"], key)
        for key in ("circuit", "fault_map"):
            original = Path(item[key])
            assert sha(original) == sha(old[key]) == item["artifact_sha256"][key]
            target = inputs / original.name
            if target.exists():
                assert sha(target) == sha(original), target
            else:
                shutil.copyfile(original, target)
            item[key] = str(target)
    manifest_path = RUN / "experiment_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    else:
        save(manifest_path, manifest)
    command = [sys.executable, "-u", str(ROOT / "scripts/train_curriculum.py"),
               str(manifest_path), str(RUN / "training_state.pth"), str(RUN / "actor_best.txt"),
               "--embedding-backend", "smartatpg", "--advantage-method", "gae",
               "--gamma", "0.99", "--gae-lambda", "0.97", "--return-scale", "100",
               "--bc-epochs", "20", "--bc-batch-size", "256",
               "--stage-sweeps", "2", "2", "3", "--seed", "2026", "--log-rollouts"]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       PYTHONPATH=str(ROOT / "python") + os.pathsep + str(ROOT / "scripts"))
    files = list((ROOT / "python/rl_podem").glob("*.py")) + [
        ROOT / "scripts/train_curriculum.py", ROOT / "python/cpp_podem.cp39-win_amd64.pyd"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    metadata = {
        "command": command, "device": "cpu", "torch_threads": 1,
        "seed": 2026, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_manifest_sha256": sha(source), "manifest_sha256": sha(manifest_path),
        "same_fault_splits_and_teacher_as_previous_gae": True,
        "expected_training_units": 150, "expected_episodes": 4200,
        "code_sha256": hashes,
        "comparison_scope": "SmartATPG-style backend vs previous DeepGate GAE pipeline, not a paper reproduction or unseen-circuit test",
    }
    started = time.perf_counter()
    with (RUN / "train.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        metadata["training_pid"] = process.pid
        save(RUN / "experiment.json", metadata)
        print(f"TRAINING_START pid={process.pid} circuits=10 episodes=4200 device=cpu threads=1", flush=True)
        while True:
            try:
                code = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                lines = (RUN / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()
                status = next((line for line in reversed(lines) if line.startswith(("BC epoch=", "TRAIN_START", "TRAINING_COMPLETE", "BC_RESULT"))), "initializing/BC")
                units = sum(line.startswith("TRAIN_RESULT ") for line in lines)
                episodes = sum(line.startswith("ROLLOUT_RESULT ") for line in lines)
                print(f"PROGRESS seconds={time.perf_counter()-started:.0f} units={units}/150 episodes={episodes}/4200 {status}", flush=True)
    metadata.update(elapsed_seconds=time.perf_counter() - started, exit_code=code,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    code_unchanged=all(sha(ROOT / path) == digest for path, digest in hashes.items()))
    save(RUN / "experiment.json", metadata)
    print(f"TRAINING_EXIT code={code} seconds={metadata['elapsed_seconds']:.1f}", flush=True)
    if code:
        raise SystemExit(code)
    assert metadata["code_unchanged"]


if __name__ == "__main__":
    main()

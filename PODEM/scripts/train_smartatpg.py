"""Prepare training data and train GAT-GRU and Mean SmartATPG models."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BACKTRACK_LIMIT = 100
NORMAL_TRAINING_ROUNDS = 2


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _environment(gpu=None):
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join([
            str(ROOT / "python"), environment.get("PYTHONPATH", ""),
        ]),
    })
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _run_logged(command, log_path, environment, prefix=""):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(prefix + line, end="", flush=True)
        return process.wait()


def _run_parallel(jobs):
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _run_logged, job["command"], job["log"], job["environment"],
                job["prefix"],
            ): job["name"]
            for job in jobs
        }
        failures = []
        for future in as_completed(futures):
            code = future.result()
            if code:
                failures.append((futures[future], code))
        if failures:
            detail = ", ".join(f"{name}={code}" for name, code in failures)
            raise RuntimeError(f"SmartATPG training failed: {detail}")


def _required_artifacts(model_dir, rounds):
    paths = [
        model_dir / "training_state.pth",
        model_dir / "inference_final.pth",
        model_dir / "model_final.txt",
    ]
    for round_number in range(1, rounds + 1):
        paths.extend((
            model_dir / f"inference_round_{round_number:02d}.pth",
            model_dir / f"model_round_{round_number:02d}.txt",
        ))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/smartatpg_dual",
    )
    parser.add_argument("--gat-gpu", type=int, default=0)
    parser.add_argument("--mean-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument("--backtrack-limit", type=int, default=BACKTRACK_LIMIT)
    parser.add_argument("--gat-continue-from", type=Path)
    parser.add_argument("--mean-continue-from", type=Path)
    args = parser.parse_args(argv)
    if args.rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(f"SmartATPG training requires {NORMAL_TRAINING_ROUNDS} rounds")
    if args.backtrack_limit != BACKTRACK_LIMIT:
        raise ValueError(f"SmartATPG training requires backtrack limit {BACKTRACK_LIMIT}")
    if min(args.gat_gpu, args.mean_gpu) < 0 or args.gat_gpu == args.mean_gpu:
        raise ValueError("GAT and Mean require distinct non-negative GPU IDs")

    output_dir = args.output_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    preparation = output_dir / "preparation"
    training = output_dir / "training"
    logs = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_environment = _environment()
    manifests = {}
    for name, encoder in (("gat", "level_gat_gru"), ("mean", "fanin_mean")):
        preparation_dir = preparation / name
        command = [
            sys.executable, "-m", "rl_podem.data_split",
            str(dataset_root), str(preparation_dir),
            "--encoder", encoder,
            "--seed", str(args.profile_seed),
            "--normal-rounds", str(args.rounds),
            "--backtrack-limit", str(args.backtrack_limit),
            "--resume",
        ]
        code = _run_logged(
            command, logs / f"prepare_{name}.log", base_environment,
            prefix=f"[{name.upper()}-PREP] ",
        )
        if code:
            raise RuntimeError(f"{name} data preparation failed with exit code {code}")
        manifests[name] = preparation_dir / "training_manifest.json"

    jobs = []
    for name, encoder, gpu, continuation in (
        ("gat", "level_gat_gru", args.gat_gpu, args.gat_continue_from),
        ("mean", "fanin_mean", args.mean_gpu, args.mean_continue_from),
    ):
        model_dir = training / name
        command = [
            sys.executable, "-m", "rl_podem.training",
            str(manifests[name]), str(model_dir),
            "--encoder", encoder, "--rounds", str(args.rounds),
            "--seed", str(args.seed),
        ]
        if continuation is None:
            command.append("--resume")
        else:
            command.extend(("--continue-from", str(continuation.resolve())))
        jobs.append({
            "name": name,
            "command": command,
            "log": logs / f"train_{name}.log",
            "environment": _environment(gpu),
            "prefix": f"[{name.upper()}] ",
        })

    started = time.perf_counter()
    _run_parallel(jobs)
    missing = [
        str(path)
        for name in ("gat", "mean")
        for path in _required_artifacts(training / name, args.rounds)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("Training completed without required artifacts: " + ", ".join(missing))
    _atomic_json(output_dir / "train_summary.json", {
        "format": "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION",
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "profile_seed": args.profile_seed,
        "rounds": args.rounds,
        "backtrack_limit": args.backtrack_limit,
        "manifests": {key: str(value) for key, value in manifests.items()},
        "training_dirs": {key: str(training / key) for key in ("gat", "mean")},
        "elapsed_s": time.perf_counter() - started,
    })
    print(f"TRAINING_COMPLETE output={output_dir}", flush=True)


if __name__ == "__main__":
    main()

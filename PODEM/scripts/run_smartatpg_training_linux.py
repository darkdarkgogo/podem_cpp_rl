"""Run resumable SmartATPG training and export a portable benchmark bundle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _tee_command(command, log_path, environment):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _check_cpp_extension():
    try:
        import cpp_podem  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Training requires the cpp_podem Python extension. Install it in "
            "the PyTorch environment with: python -m pip install -e ."
        ) from error


def _run(command, log_path, environment):
    code = _tee_command(command, log_path, environment)
    if code:
        raise SystemExit(code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/smartatpg_paper_11d",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=500)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This training launcher is intended for Linux")
    if args.rounds <= 0 or args.backtrack_limit <= 0:
        raise ValueError("Rounds and backtrack limit must be positive")
    _check_cpp_extension()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join([
            str(ROOT / "python"),
            str(ROOT / "scripts"),
            environment.get("PYTHONPATH", ""),
        ]),
    })
    prepare_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_smartatpg_training.py"),
        str(output_dir),
        "--count", "100",
        "--backtrack-limit", str(args.backtrack_limit),
        "--seed", str(args.profile_seed),
        "--resume",
    ]
    train_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_smartatpg.py"),
        str(output_dir / "training_manifest.json"),
        str(output_dir),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
    ]
    bundle_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_smartatpg_benchmark.py"),
        str(output_dir / "benchmark_bundle"),
        str(output_dir / "model_best.txt"),
        "--resume",
    ]
    metadata = {
        "format": "SMARTATPG_TRAINING_RUN_V2",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "rounds": args.rounds,
        "seed": args.seed,
        "profile_seed": args.profile_seed,
        "backtrack_limit": args.backtrack_limit,
        "commands": [prepare_command, train_command, bundle_command],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = output_dir / "training_run_metadata.json"
    _atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    _run(prepare_command, output_dir / "prepare_training.log", environment)
    _run(train_command, output_dir / "train.log", environment)
    if not (output_dir / "model_best.txt").is_file():
        raise RuntimeError("Training completed without model_best.txt")
    _run(bundle_command, output_dir / "prepare_bundle.log", environment)
    metadata.update({
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bundle": str(output_dir / "benchmark_bundle"),
    })
    _atomic_json(metadata_path, metadata)
    print(
        f"TRAINING_COMPLETE bundle={output_dir / 'benchmark_bundle'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

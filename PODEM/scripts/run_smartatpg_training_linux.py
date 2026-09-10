"""Train both SmartATPG encoders and export one comparison bundle."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
BACKTRACK_LIMIT = 2000
NORMAL_TRAINING_ROUNDS = 20
GAT_REINFORCEMENT_ROUNDS = 5


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _tee_command(command, log_path, environment, prefix="", on_start=None):
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
        if on_start is not None:
            on_start(process)
        for line in process.stdout:
            print(prefix + line, end="", flush=True)
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
    started = time.perf_counter()
    code = _tee_command(command, log_path, environment)
    if code:
        raise SystemExit(code)
    return time.perf_counter() - started


def _run_parallel(jobs):
    active = {}
    lock = threading.Lock()

    def run_one(name, command, log_path, environment, prefix):
        def register(process):
            with lock:
                active[name] = process

        started = time.perf_counter()
        code = _tee_command(command, log_path, environment, prefix, register)
        if code:
            raise RuntimeError(f"{name} training failed with exit code {code}")
        return time.perf_counter() - started

    executor = ThreadPoolExecutor(max_workers=len(jobs))
    futures = {
        executor.submit(run_one, name, *settings): name
        for name, settings in jobs.items()
    }
    timings = {}
    try:
        for future in as_completed(futures):
            name = futures[future]
            timings[name] = future.result()
    except BaseException:
        with lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return timings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/smartatpg_12d_co_bt2000",
    )
    parser.add_argument("--rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=BACKTRACK_LIMIT)
    parser.add_argument(
        "--gat-reinforcement-rounds",
        type=int,
        default=GAT_REINFORCEMENT_ROUNDS,
    )
    parser.add_argument("--mean-gpu", type=int, default=0)
    parser.add_argument("--gat-gpu", type=int, default=1)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This training launcher is intended for Linux")
    if (
        args.rounds <= 0
        or args.backtrack_limit <= 0
        or args.gat_reinforcement_rounds <= 0
        or args.gat_reinforcement_rounds > GAT_REINFORCEMENT_ROUNDS
    ):
        raise ValueError(
            "Rounds and backtrack limit must be positive; GAT reinforcement "
            f"rounds must be between 1 and {GAT_REINFORCEMENT_ROUNDS}"
        )
    if args.rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"GAT reinforcement requires exactly {NORMAL_TRAINING_ROUNDS} "
            "normal training rounds"
        )
    if args.mean_gpu < 0 or args.gat_gpu < 0 or args.mean_gpu == args.gat_gpu:
        raise ValueError("Mean and GAT-GRU training require two distinct non-negative GPU IDs")
    if args.backtrack_limit != BACKTRACK_LIMIT:
        raise ValueError(
            f"SmartATPG training requires backtrack limit {BACKTRACK_LIMIT}"
        )
    _check_cpp_extension()
    gpu_count = torch.cuda.device_count()
    if max(args.mean_gpu, args.gat_gpu) >= gpu_count:
        raise RuntimeError(
            f"Requested GPUs {args.mean_gpu} and {args.gat_gpu}, but PyTorch sees "
            f"only {gpu_count} CUDA device(s)"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preparation_dir = output_dir / "preparation"
    baseline_dir = output_dir / "smartatpg_mean"
    gat_gru_dir = output_dir / "smartatpg_gat_gru"
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
    mean_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(args.mean_gpu))
    gat_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(args.gat_gpu))
    prepare_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_smartatpg_training.py"),
        str(preparation_dir),
        "--count", "100",
        "--backtrack-limit", str(args.backtrack_limit),
        "--seed", str(args.profile_seed),
        "--resume",
    ]
    baseline_train_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_smartatpg.py"),
        str(preparation_dir / "training_manifest.json"),
        str(baseline_dir),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
        "--encoder", "fanin_mean",
    ]
    gat_gru_train_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_smartatpg.py"),
        str(preparation_dir / "training_manifest.json"),
        str(gat_gru_dir),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
        "--encoder", "level_gat_gru",
        "--reinforcement-rounds", str(args.gat_reinforcement_rounds),
    ]
    bundle_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_smartatpg_benchmark.py"),
        str(output_dir / "benchmark_bundle"),
        str(baseline_dir / "model_best.txt"),
        str(gat_gru_dir / "model_best_reinforced.txt"),
        "--resume",
    ]
    metadata = {
        "format": "SMARTATPG_TRAINING_RUN_V2",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "physical_gpu_assignment": {
            "smartatpg_mean": args.mean_gpu,
            "smartatpg_gat_gru": args.gat_gpu,
        },
        "rounds": args.rounds,
        "gat_reinforcement_rounds": args.gat_reinforcement_rounds,
        "seed": args.seed,
        "profile_seed": args.profile_seed,
        "backtrack_limit": args.backtrack_limit,
        "commands": [
            prepare_command, baseline_train_command,
            gat_gru_train_command, bundle_command,
        ],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = output_dir / "training_run_metadata.json"
    _atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    timings = {
        "preparation_seconds": _run(
            prepare_command, output_dir / "prepare_training.log", environment
        )
    }
    parallel_timings = _run_parallel({
        "smartatpg_mean": (
            baseline_train_command, baseline_dir / "train.log",
            mean_environment, "[mean] ",
        ),
        "smartatpg_gat_gru": (
            gat_gru_train_command, gat_gru_dir / "train.log",
            gat_environment, "[gat-gru] ",
        ),
    })
    timings.update({
        f"{name}_training_seconds": seconds
        for name, seconds in parallel_timings.items()
    })
    required_models = (
        baseline_dir / "model_best.txt",
        gat_gru_dir / "model_best.txt",
        gat_gru_dir / "model_best_reinforced.txt",
    )
    for model_path in required_models:
        if not model_path.is_file():
            raise RuntimeError(f"Training completed without {model_path}")
    timings["bundle_seconds"] = _run(
        bundle_command, output_dir / "prepare_bundle.log", environment
    )
    metadata.update({
        "elapsed_seconds": time.perf_counter() - started,
        "timings": timings,
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

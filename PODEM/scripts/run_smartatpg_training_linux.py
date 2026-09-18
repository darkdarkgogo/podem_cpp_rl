"""Train data-split GAT-GRU SmartATPG on Linux and export its bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BACKTRACK_LIMIT = 100
NORMAL_TRAINING_ROUNDS = 2
FAULTS_PER_UPDATE = 8
PPO_EPOCHS_PER_UPDATE = 1


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        import cpp_podem
    except ImportError as error:
        raise RuntimeError(
            "Training requires the cpp_podem Python extension. Install it in "
            "the PyTorch environment with: python -m pip install -e ."
        ) from error
    if not hasattr(cpp_podem, "run_native_validation"):
        raise RuntimeError(
            "The cpp_podem extension is stale. Rebuild it in the PyTorch "
            "environment with: python -m pip install -e ."
        )


def _run(command, log_path, environment):
    started = time.perf_counter()
    code = _tee_command(command, log_path, environment)
    if code:
        raise SystemExit(code)
    return time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/smartatpg_top30_hard_2rounds_batch8_bt100",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument("--rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=BACKTRACK_LIMIT)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This training launcher is intended for Linux")
    if args.rounds <= 0 or args.backtrack_limit <= 0:
        raise ValueError("Rounds and backtrack limit must be positive")
    if args.rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(f"SmartATPG training requires exactly {NORMAL_TRAINING_ROUNDS} rounds")
    if args.gpu < 0:
        raise ValueError("GPU ID must be non-negative")
    if args.backtrack_limit != BACKTRACK_LIMIT:
        raise ValueError(
            f"SmartATPG training requires backtrack limit {BACKTRACK_LIMIT}"
        )
    if torch is None:
        raise RuntimeError("Training requires PyTorch in the active environment")
    _check_cpp_extension()
    gpu_count = torch.cuda.device_count()
    if args.gpu >= gpu_count:
        raise RuntimeError(
            f"Requested GPU {args.gpu}, but PyTorch sees "
            f"only {gpu_count} CUDA device(s)"
        )

    output_dir = args.output_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preparation_dir = output_dir / "preparation"
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
    gat_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(args.gpu))
    prepare_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_smartatpg_training.py"),
        str(dataset_root),
        str(preparation_dir),
        "--seed", str(args.profile_seed),
        "--resume",
    ]
    gat_gru_train_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_smartatpg.py"),
        str(preparation_dir / "training_manifest.json"),
        str(gat_gru_dir),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
        "--k-epochs", str(PPO_EPOCHS_PER_UPDATE),
        "--encoder", "level_gat_gru",
    ]
    if args.continue_from:
        gat_gru_train_command.extend([
            "--continue-from", str(args.continue_from.resolve())
        ])
    else:
        gat_gru_train_command.append("--resume")
    bundle_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_smartatpg_benchmark.py"),
        str(output_dir / "benchmark_bundle"),
        str(gat_gru_dir / "model_best.txt"),
        "--resume",
    ]
    metadata = {
        "format": "SMARTATPG_TRAINING_RUN_V4_DATA_SPLIT",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "physical_gpu_assignment": {
            "smartatpg_gat_gru": args.gpu,
        },
        "rounds": args.rounds,
        "seed": args.seed,
        "profile_seed": args.profile_seed,
        "backtrack_limit": args.backtrack_limit,
        "dataset_root": str(dataset_root),
        "continue_from": str(args.continue_from.resolve()) if args.continue_from else None,
        "commands": [
            prepare_command, gat_gru_train_command, bundle_command,
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
    manifest_path = preparation_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["training_protocol"] = {
        "manifest_hash": _sha256(manifest_path),
        "backtrack_limit": manifest["backtrack_limit"],
        "reward_scheme": "cubic_backtrack_v1",
        "normal_rounds": manifest["normal_rounds"],
        "training_circuit_count": len(manifest["train_circuits"]),
        "validation_circuit_count": len(manifest["validation_circuits"]),
        "faults_per_update": manifest["faults_per_update"],
        "k_epochs": manifest["k_epochs"],
    }
    _atomic_json(metadata_path, metadata)
    timings["smartatpg_gat_gru_training_seconds"] = _run(
        gat_gru_train_command, output_dir / "train_gat_gru.log", gat_environment
    )
    required_models = (
        gat_gru_dir / "model_best.txt",
        gat_gru_dir / "model_latest.txt",
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

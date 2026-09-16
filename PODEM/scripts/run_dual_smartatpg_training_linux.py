"""Prepare once, train GAT and mean SmartATPG models concurrently, then compare."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

try:
    import torch
except ModuleNotFoundError:
    torch = None

from run_smartatpg_training_linux import _check_cpp_extension, _tee_command


ROOT = Path(__file__).resolve().parents[1]
BACKTRACK_LIMIT = 200
NORMAL_TRAINING_ROUNDS = 2
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


def _run(command, log_path, environment, prefix=""):
    started = time.perf_counter()
    code = _tee_command(command, log_path, environment, prefix=prefix)
    if code:
        raise SystemExit(code)
    return time.perf_counter() - started


def _run_parallel_training(jobs: list[dict]) -> dict[str, float]:
    """Run the two training commands and stop the peer on the first failure."""
    processes = {}
    process_lock = threading.Lock()
    launch_gate = threading.Event()
    cancelled = threading.Event()

    def run_job(job):
        started = time.perf_counter()

        def on_start(process):
            with process_lock:
                processes[job["name"]] = process
                cancel_this_process = cancelled.is_set()
                if len(processes) == len(jobs):
                    launch_gate.set()
            if cancel_this_process:
                # A sibling can fail before this Popen reaches on_start.  The
                # controller's cancellation remains sticky so this late child
                # is never allowed to continue after registration.
                process.terminate()
            # A process can exit immediately.  Hold its stream loop until both
            # children are registered so the coordinator always has a peer it
            # can terminate on the first non-zero exit.
            launch_gate.wait()

        code = _tee_command(
            job["command"],
            job["log_path"],
            job["environment"],
            prefix=job.get("prefix", ""),
            on_start=on_start,
        )
        return job["name"], code, time.perf_counter() - started

    def cancel_training(except_name=None):
        cancelled.set()
        launch_gate.set()
        with process_lock:
            active = [
                process for name, process in processes.items()
                if name != except_name and process.poll() is None
            ]
        for process in active:
            process.terminate()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_job, job): job["name"]
            for job in jobs
        }
        timings = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                job_name, code, elapsed = future.result()
            except BaseException:
                # An exception can occur before on_start (for example Popen or
                # stream setup).  Release a registered child from the startup
                # gate, terminate it, and then propagate the original error.
                cancel_training()
                for peer in futures:
                    if peer is not future:
                        try:
                            peer.result()
                        except BaseException:
                            pass
                raise
            if code:
                cancel_training(name)
                for peer in futures:
                    if peer is not future:
                        try:
                            peer.result()
                        except BaseException:
                            pass
                raise SystemExit(code)
            timings[job_name] = elapsed
    return timings


def _allowed_physical_gpu_ids(gpu_count):
    """Return the physical GPU IDs assigned to this controller process."""
    inherited_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited_mask is None:
        return set(range(gpu_count))
    tokens = [token.strip() for token in inherited_mask.split(",")]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain numeric physical GPU IDs; "
            "UUID and malformed masks are unsupported"
        )
    physical_ids = [int(token) for token in tokens]
    if len(physical_ids) != len(set(physical_ids)):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain distinct numeric physical GPU IDs"
        )
    return set(physical_ids)


def _validate_args(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This training launcher is intended for Linux")
    if args.rounds <= 0 or args.backtrack_limit <= 0:
        raise ValueError("Rounds and backtrack limit must be positive")
    if args.rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"SmartATPG training requires exactly {NORMAL_TRAINING_ROUNDS} rounds"
        )
    if args.backtrack_limit != BACKTRACK_LIMIT:
        raise ValueError(
            f"SmartATPG training requires backtrack limit {BACKTRACK_LIMIT}"
        )
    if args.gat_gpu < 0 or args.mean_gpu < 0:
        raise ValueError("GPU IDs must be non-negative")
    if args.gat_gpu == args.mean_gpu:
        raise ValueError("GAT and mean GPU IDs must be distinct")
    if torch is None:
        raise RuntimeError("Training requires PyTorch in the active environment")
    _check_cpp_extension()
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(
            f"Dual training requires two CUDA devices, but PyTorch sees only "
            f"{gpu_count} CUDA device(s)"
        )
    allowed_gpu_ids = _allowed_physical_gpu_ids(gpu_count)
    if len(allowed_gpu_ids) < 2:
        raise RuntimeError(
            "Dual training requires two physical GPU IDs in "
            "CUDA_VISIBLE_DEVICES"
        )
    for gpu in (args.gat_gpu, args.mean_gpu):
        if gpu not in allowed_gpu_ids:
            inherited_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
            if inherited_mask is not None:
                raise RuntimeError(
                    f"Requested GPU {gpu} is outside inherited "
                    f"CUDA_VISIBLE_DEVICES={inherited_mask}"
                )
            raise RuntimeError(
                f"Requested GPU {gpu}, but PyTorch sees only {gpu_count} "
                "CUDA device(s)"
            )


def _training_command(manifest, output_dir, encoder, args, continuation):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_smartatpg.py"),
        str(manifest),
        str(output_dir),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
        "--k-epochs", str(PPO_EPOCHS_PER_UPDATE),
        "--encoder", encoder,
    ]
    if continuation:
        command.extend(("--continue-from", str(continuation.resolve())))
    else:
        command.append("--resume")
    return command


def _required_run_artifacts(directory):
    return (
        Path(directory) / "model_best.txt",
        Path(directory) / "model_latest.txt",
        Path(directory) / "validation_identity.json",
        Path(directory) / "validation_metrics.json",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/smartatpg_dual_top30_hard_2rounds_batch8_bt200",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument("--rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=BACKTRACK_LIMIT)
    parser.add_argument("--gat-gpu", type=int, default=0)
    parser.add_argument("--mean-gpu", type=int, default=1)
    parser.add_argument("--gat-continue-from", type=Path)
    parser.add_argument("--mean-continue-from", type=Path)
    args = parser.parse_args(argv)
    _validate_args(args)

    output_dir = args.output_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preparation_dir = output_dir / "preparation"
    manifest_path = preparation_dir / "training_manifest.json"
    gat_dir = output_dir / "smartatpg_gat_gru"
    mean_dir = output_dir / "smartatpg_mean"
    comparison_dir = output_dir / "validation_comparison"
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join([
            str(ROOT / "python"), str(ROOT / "scripts"),
            environment.get("PYTHONPATH", ""),
        ]),
    })
    preparation_command = [
        sys.executable, "-u", str(ROOT / "scripts/prepare_smartatpg_training.py"),
        str(dataset_root), str(preparation_dir), "--seed", str(args.profile_seed),
        "--normal-rounds", str(args.rounds), "--backtrack-limit",
        str(args.backtrack_limit), "--resume",
    ]

    started = time.perf_counter()
    preparation_seconds = _run(
        preparation_command, output_dir / "prepare_training.log", environment
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gat_command = _training_command(
        manifest_path, gat_dir, "level_gat_gru", args, args.gat_continue_from
    )
    mean_command = _training_command(
        manifest_path, mean_dir, "fanin_mean", args, args.mean_continue_from
    )
    gat_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(args.gat_gpu))
    mean_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(args.mean_gpu))
    comparison_command = [
        sys.executable, "-u", str(ROOT / "scripts/compare_smartatpg_validation.py"),
        str(manifest_path), str(gat_dir), str(mean_dir), str(comparison_dir),
        "--seed", str(args.seed),
    ]
    metadata = {
        "format": "SMARTATPG_DUAL_TRAINING_RUN_V1",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "physical_gpu_assignment": {
            "smartatpg_gat_gru": args.gat_gpu,
            "smartatpg_mean": args.mean_gpu,
        },
        "dataset_root": str(dataset_root),
        "rounds": args.rounds,
        "seed": args.seed,
        "profile_seed": args.profile_seed,
        "backtrack_limit": args.backtrack_limit,
        "training_protocol": {
            "manifest_hash": _sha256(manifest_path),
            "backtrack_limit": manifest["backtrack_limit"],
            "normal_rounds": manifest["normal_rounds"],
            "faults_per_update": manifest["faults_per_update"],
            "k_epochs": manifest["k_epochs"],
            "training_circuit_count": len(manifest["train_circuits"]),
            "validation_circuit_count": len(manifest["validation_circuits"]),
        },
        "commands": [preparation_command, gat_command, mean_command, comparison_command],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = output_dir / "training_run_metadata.json"
    _atomic_json(metadata_path, metadata)

    train_timings = _run_parallel_training([
        {
            "name": "smartatpg_gat_gru", "command": gat_command,
            "log_path": output_dir / "train_gat_gru.log",
            "environment": gat_environment, "output_prefix": gat_dir,
            "prefix": "[GAT] ",
        },
        {
            "name": "smartatpg_mean", "command": mean_command,
            "log_path": output_dir / "train_mean.log",
            "environment": mean_environment, "output_prefix": mean_dir,
            "prefix": "[MEAN] ",
        },
    ])
    for directory in (gat_dir, mean_dir):
        for artifact in _required_run_artifacts(directory):
            if not artifact.is_file():
                raise RuntimeError(f"Training completed without {artifact}")
    comparison_seconds = _run(
        comparison_command, output_dir / "compare_validation.log", environment
    )
    metadata.update({
        "elapsed_seconds": time.perf_counter() - started,
        "timings": {
            "preparation_seconds": preparation_seconds,
            **{f"{name}_training_seconds": seconds for name, seconds in train_timings.items()},
            "validation_comparison_seconds": comparison_seconds,
        },
        "validation_comparison": str(comparison_dir),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _atomic_json(metadata_path, metadata)
    print(f"DUAL_TRAINING_COMPLETE comparison={comparison_dir}", flush=True)


if __name__ == "__main__":
    main()

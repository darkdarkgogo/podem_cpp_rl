"""Run resumable SmartATPG curriculum training and final benchmarking on Linux."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/paper_v8_smartatpg/training_manifest.json"


def _matches_artifact_hash(path, expected):
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    if path.suffix.lower() != ".json":
        return False

    lf_data = data.replace(b"\r\n", b"\n")
    crlf_data = lf_data.replace(b"\n", b"\r\n")
    return expected in {
        hashlib.sha256(lf_data).hexdigest(),
        hashlib.sha256(crlf_data).hexdigest(),
    }


def _atomic_json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_relocated_path(raw_path, repository_root, manifest_directory):
    raw_path = str(raw_path)
    repository_root = Path(repository_root).resolve()
    normalized = raw_path.replace("\\", "/")
    candidates = []
    direct = Path(raw_path)
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.extend([
            Path(manifest_directory) / direct,
            repository_root / direct,
        ])
    marker = "/PODEM/"
    marker_index = normalized.lower().rfind(marker.lower())
    if marker_index >= 0:
        candidates.append(repository_root / normalized[marker_index + len(marker):])
    elif normalized.lower().startswith("podem/"):
        candidates.append(repository_root / normalized[len("PODEM/"):])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            try:
                resolved.relative_to(repository_root)
            except ValueError as error:
                raise ValueError(
                    f"Manifest artifact is outside the repository: {resolved}"
                ) from error
            return resolved
    raise FileNotFoundError(f"Cannot relocate manifest artifact: {raw_path}")


def relocate_manifest(source_path, output_path, repository_root=ROOT):
    source_path = Path(source_path).resolve()
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    relocated = copy.deepcopy(manifest)
    for kind in ("training", "validation"):
        key = f"teacher_{kind}"
        path = _resolve_relocated_path(
            relocated[key], repository_root, source_path.parent
        )
        expected = relocated["teacher_sha256"][kind]
        if not _matches_artifact_hash(path, expected):
            raise ValueError(f"Teacher artifact hash changed: {path}")
        relocated[key] = str(path)
    for item in relocated["circuits"]:
        for key, expected in item["artifact_sha256"].items():
            path = _resolve_relocated_path(
                item[key], repository_root, source_path.parent
            )
            if not _matches_artifact_hash(path, expected):
                raise ValueError(f"Circuit artifact hash changed: {path}")
            item[key] = str(path)

    output_path = Path(output_path).resolve()
    if output_path == source_path:
        raise ValueError("Portable manifest must not overwrite the source manifest.")
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != relocated:
            raise ValueError(
                "Existing portable manifest differs; use a new output directory."
            )
    else:
        _atomic_json_save(output_path, relocated)
    return relocated


def _tee_command(command, cwd, log_path, environment=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
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
            "Cannot import the Linux cpp_podem extension. From the PODEM root, "
            "run: python -m pip install -r python-requirements.txt && "
            "python -m pip install -e ."
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/smartatpg_linux_20rounds",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--log-rollouts", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This launcher is intended for Linux.")
    if args.rounds <= 0 or args.benchmark_repeats <= 0:
        raise ValueError("Rounds and benchmark repeats must be positive.")

    _check_cpp_extension()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    portable_manifest = output_dir / "training_manifest.json"
    relocate_manifest(args.source_manifest, portable_manifest)

    checkpoint = output_dir / "training_state.pth"
    actor_best = output_dir / "actor_best.txt"
    actor_latest = output_dir / "actor_latest.txt"
    round_metrics = output_dir / "round_metrics.json"
    train_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_curriculum.py"),
        str(portable_manifest),
        str(checkpoint),
        str(actor_best),
        "--latest-actor-output", str(actor_latest),
        "--embedding-backend", "smartatpg",
        "--advantage-method", "gae",
        "--gamma", "0.99",
        "--gae-lambda", "0.97",
        "--return-scale", "100",
        "--bc-epochs", "20",
        "--bc-batch-size", "256",
        "--curriculum-rounds", str(args.rounds),
        "--round-metrics-output", str(round_metrics),
        "--seed", str(args.seed),
    ]
    if args.log_rollouts:
        train_command.append("--log-rollouts")

    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join([
            str(ROOT / "python"), str(ROOT / "scripts"),
            environment.get("PYTHONPATH", ""),
        ]),
    })
    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "format": "SMARTATPG_LINUX_20_ROUND_RUN_V1",
        "source_manifest": str(args.source_manifest.resolve()),
        "portable_manifest": str(portable_manifest),
        "rounds": args.rounds,
        "seed": args.seed,
        "benchmark_repeats": args.benchmark_repeats,
        "python": sys.executable,
        "training_command": train_command,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_json_save(metadata_path, metadata)
    training_started = time.perf_counter()
    training_code = _tee_command(
        train_command, ROOT, output_dir / "train.log", environment
    )
    metadata.update({
        "training_exit_code": training_code,
        "training_seconds": time.perf_counter() - training_started,
    })
    _atomic_json_save(metadata_path, metadata)
    if training_code:
        raise SystemExit(training_code)
    if args.skip_benchmark:
        print("TRAINING_AND_EVALUATION_COMPLETE benchmark=skipped", flush=True)
        return

    native_executable = output_dir / "native" / "atpg_rl_smartatpg"
    build_command = [
        sys.executable,
        str(ROOT / "scripts/build_native.py"),
        "--output", str(native_executable),
    ]
    build_code = _tee_command(
        build_command, ROOT, output_dir / "build_native.log", environment
    )
    if build_code:
        raise SystemExit(build_code)

    benchmark_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/benchmark_smartatpg.py"),
        str(portable_manifest),
        str(checkpoint),
        str(native_executable),
        str(output_dir / "final_benchmark"),
        "--repeats", str(args.benchmark_repeats),
        "--seed", str(args.seed),
        "--backtrack-limit", "500",
    ]
    benchmark_started = time.perf_counter()
    benchmark_code = _tee_command(
        benchmark_command, ROOT, output_dir / "benchmark.log", environment
    )
    metadata.update({
        "native_executable": str(native_executable),
        "benchmark_command": benchmark_command,
        "benchmark_exit_code": benchmark_code,
        "benchmark_seconds": time.perf_counter() - benchmark_started,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _atomic_json_save(metadata_path, metadata)
    if benchmark_code:
        raise SystemExit(benchmark_code)
    print("TRAINING_AND_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

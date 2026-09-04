"""Rebuild native PODEM and benchmark a portable SmartATPG bundle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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


def _run(command, log_path, environment):
    code = _tee_command(command, log_path, environment)
    if code:
        raise SystemExit(code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=500)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This benchmark launcher is intended for Linux")
    if args.repeats <= 0 or args.backtrack_limit <= 0:
        raise ValueError("Repeats and backtrack limit must be positive")

    bundle = args.bundle.resolve()
    manifest = bundle / "bundle_manifest.json" if bundle.is_dir() else bundle
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing benchmark bundle manifest: {manifest}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join([
            str(ROOT / "scripts"), environment.get("PYTHONPATH", "")
        ]),
    })
    executable = output_dir / "native" / "atpg_rl_smartatpg"
    build_command = [
        sys.executable,
        str(ROOT / "scripts/build_native.py"),
        "--output", str(executable),
    ]
    benchmark_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/benchmark_smartatpg.py"),
        str(manifest),
        str(executable),
        str(output_dir / "comparison"),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--backtrack-limit", str(args.backtrack_limit),
    ]
    metadata = {
        "format": "SMARTATPG_BENCHMARK_RUN_V3",
        "python": sys.executable,
        "bundle_manifest": str(manifest),
        "repeats": args.repeats,
        "seed": args.seed,
        "backtrack_limit": args.backtrack_limit,
        "timing_scope": "C++ ATPG interval only; embedding and compilation excluded",
        "commands": [build_command, benchmark_command],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = output_dir / "benchmark_run_metadata.json"
    _atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    _run(build_command, output_dir / "build_native.log", environment)
    _run(benchmark_command, output_dir / "benchmark.log", environment)
    metadata.update({
        "elapsed_seconds_debug_only": time.perf_counter() - started,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _atomic_json(metadata_path, metadata)
    print(
        f"BENCHMARK_COMPLETE results={output_dir / 'comparison'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

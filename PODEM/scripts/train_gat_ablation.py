"""Train three GAT ablations concurrently using an existing full GAT run."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("no_cache", "no_lock", "no_object_value")
VARIANTS = {
    "no_cache": "level_gat_gru",
    "no_lock": "level_gat_gru",
    "no_object_value": "level_gat_gru_no_object_value",
}


def _run(command, environment, log_path, prefix):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(prefix + line, end="", flush=True)
        return process.wait()


def _read_full_run(path):
    summary_path = path / "train_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Existing full GAT run is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION":
        raise ValueError("Existing full GAT training summary has an incompatible format")
    if summary.get("rounds") != 1 or summary.get("backtrack_limit") != 100:
        raise ValueError("Existing full GAT run must use one round and backtrack limit 100")
    if summary.get("backtrace_lock") not in ("auto", "on"):
        raise ValueError("Existing full GAT run must have the backtrace lock enabled")
    manifest = Path(summary["manifests"]["gat"]).resolve()
    model_dir = Path(summary["training_dirs"]["gat"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Existing full GAT manifest is missing: {manifest}")
    for name in ("inference_round_01.pth", "model_round_01.txt"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Existing full GAT artifact is missing: {model_dir / name}")
    return summary, manifest, model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-dir", type=Path, default=ROOT / "artifacts/smartatpg_dual")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/gat_ablation")
    parser.add_argument("--no-cache-gpu", type=int, default=0)
    parser.add_argument("--no-lock-gpu", type=int, default=1)
    parser.add_argument("--no-object-value-gpu", type=int, default=2)
    args = parser.parse_args(argv)
    gpu_ids = {
        "no_cache": args.no_cache_gpu,
        "no_lock": args.no_lock_gpu,
        "no_object_value": args.no_object_value_gpu,
    }
    if min(gpu_ids.values()) < 0 or len(set(gpu_ids.values())) != 3:
        raise ValueError("The three ablations require distinct non-negative GPU IDs")
    full_run_dir = args.full_run_dir.resolve()
    full_summary, manifest, full_model_dir = _read_full_run(full_run_dir)
    output_dir = args.output_dir.resolve()
    if (output_dir == full_run_dir or output_dir in full_run_dir.parents or
            full_run_dir in output_dir.parents):
        raise ValueError("Ablation output must be separate from the existing full run")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = {}
    jobs = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for group in GROUPS:
            model_dir = output_dir / "training" / group
            lock = "off" if group == "no_lock" else "on"
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": os.pathsep.join((str(ROOT / "python"), environment.get("PYTHONPATH", ""))),
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "CUDA_VISIBLE_DEVICES": str(gpu_ids[group]),
            })
            command = [
                sys.executable, "-m", "rl_podem.training", str(manifest),
                str(model_dir), "--encoder", "level_gat_gru", "--ablation", group,
                "--rounds", "1", "--seed", str(full_summary["seed"]),
                "--backtrace-lock", lock, "--require-cuda", "--resume",
            ]
            future = executor.submit(
                _run, command, environment,
                output_dir / "logs" / f"train_{group}.log", f"[{group}] ",
            )
            jobs[future] = group
            training[group] = {
                "encoder_variant": VARIANTS[group],
                "backtrace_lock": lock == "on",
                "model_dir": str(model_dir),
                "gpu": gpu_ids[group],
            }
        failures = []
        for future in as_completed(jobs):
            group = jobs[future]
            try:
                code = future.result()
            except Exception as error:
                failures.append(f"{group}: {error}")
            else:
                if code:
                    failures.append(f"{group}: exit {code}")
        if failures:
            raise RuntimeError("GAT ablation training failed: " + ", ".join(failures))
    for group in GROUPS:
        model_dir = Path(training[group]["model_dir"])
        for filename in ("training_state.pth", "inference_round_01.pth", "model_round_01.txt"):
            if not (model_dir / filename).is_file():
                raise RuntimeError(f"Missing {group} artifact: {filename}")
    summary = {
        "format": "SMARTATPG_GAT_ABLATION_TRAINING_V2",
        "full_run_dir": str(full_run_dir),
        "full_model_dir": str(full_model_dir),
        "dataset_root": str(Path(full_summary["dataset_root"]).resolve()),
        "manifest": str(manifest),
        "seed": full_summary["seed"],
        "profile_seed": full_summary["profile_seed"],
        "rounds": 1,
        "backtrack_limit": 100,
        "groups": training,
    }
    target = output_dir / "train_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"GAT_ABLATION_TRAINING_COMPLETE output={output_dir}", flush=True)


if __name__ == "__main__":
    main()

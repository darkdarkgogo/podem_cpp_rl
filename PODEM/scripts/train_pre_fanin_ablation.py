"""Train two pre-fanin GAT ablations using an existing full GAT run."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAINED_GROUPS = ("no_lock", "no_object_value")
VARIANTS = {
    "no_lock": "level_gat_gru",
    "no_object_value": "level_gat_gru_no_object_value",
}


def _run(command, environment, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Command failed; see {log_path}")


def _read_full_run(path):
    summary_path = path / "train_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Existing full GAT run is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION":
        raise ValueError("Existing full GAT training summary has an incompatible format")
    if summary.get("rounds") != 2 or summary.get("backtrack_limit") != 100:
        raise ValueError("Existing full GAT must use the two-round, 100-backtrack protocol")
    if summary.get("backtrace_lock") not in ("auto", "on"):
        raise ValueError("Existing full GAT must have backtrace lock enabled")
    manifest = Path(summary["manifests"]["gat"]).resolve()
    model_dir = Path(summary["training_dirs"]["gat"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Existing full GAT manifest is missing: {manifest}")
    for number in (1, 2):
        for filename in (f"inference_round_{number:02d}.pth", f"model_round_{number:02d}.txt"):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Existing full GAT artifact is missing: {model_dir / filename}")
    records = path / "validation" / "detailed" / "gat_records.jsonl"
    if not records.is_file():
        raise FileNotFoundError(f"Existing full GAT validation records are missing: {records}")
    return summary, manifest, model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-dir", type=Path, default=ROOT / "artifacts/smartatpg_dual")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/pre_fanin_ablation")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    if args.gpu < 0:
        raise ValueError("GPU ID must be non-negative")
    full_run_dir = args.full_run_dir.resolve()
    full_summary, manifest, full_model_dir = _read_full_run(full_run_dir)
    dataset_root = Path(full_summary["dataset_root"]).resolve()
    output_dir = args.output_dir.resolve()
    if (output_dir == full_run_dir or output_dir in full_run_dir.parents or
            full_run_dir in output_dir.parents):
        raise ValueError("Ablation output must be separate from the full run")
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT / "python"), environment.get("PYTHONPATH", ""))),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
    })
    _run([
        sys.executable, "-c",
        "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "
        "'One visible CUDA GPU is required'",
    ], environment, output_dir / "logs" / "gpu_preflight.log")
    groups = {
        "full": {
            "encoder_variant": "level_gat_gru", "backtrace_lock": True,
            "model_dir": str(full_model_dir), "source_group": "existing_full",
        },
        "no_cache": {
            "encoder_variant": "level_gat_gru", "backtrace_lock": True,
            "model_dir": str(full_model_dir), "source_group": "full",
            "logits_cache": False,
        },
    }
    for group in TRAINED_GROUPS:
        model_dir = output_dir / "training" / group
        lock = "off" if group == "no_lock" else "on"
        _run([
            sys.executable, "-m", "rl_podem.training", str(manifest),
            str(model_dir), "--encoder", "level_gat_gru", "--ablation", group,
            "--rounds", "2", "--seed", str(full_summary["seed"]),
            "--backtrace-lock", lock, "--resume",
        ], environment, output_dir / "logs" / f"train_{group}.log")
        for filename in ("training_state.pth", "inference_round_02.pth", "model_round_02.txt"):
            if not (model_dir / filename).is_file():
                raise RuntimeError(f"Missing {group} artifact: {filename}")
        groups[group] = {
            "encoder_variant": VARIANTS[group],
            "backtrace_lock": lock == "on",
            "model_dir": str(model_dir),
        }
    summary = {
        "format": "SMARTATPG_PRE_FANIN_ABLATION_TRAINING_V1",
        "base_commit": "7d92ab8",
        "full_run_dir": str(full_run_dir),
        "dataset_root": str(dataset_root),
        "manifest": str(manifest),
        "seed": full_summary["seed"],
        "profile_seed": full_summary["profile_seed"],
        "rounds": 2,
        "backtrack_limit": 100,
        "groups": groups,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "train_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"PRE_FANIN_ABLATION_TRAINING_COMPLETE output={output_dir}", flush=True)


if __name__ == "__main__":
    main()

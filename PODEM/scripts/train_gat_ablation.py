"""Train four independent GAT ablation groups from one prepared manifest."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("full", "no_gru", "no_lock", "no_object_value")
VARIANTS = {
    "full": "level_gat_gru",
    "no_gru": "level_gat_no_gru",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/gat_ablation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    if args.gpu < 0:
        raise ValueError("GPU ID must be non-negative")
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "python"), environment.get("PYTHONPATH", ""))
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    manifest_dir = output_dir / "preparation" / "gat"
    _run([
        sys.executable, "-m", "rl_podem.data_split",
        str(dataset_root), str(manifest_dir), "--encoder", "level_gat_gru",
        "--seed", str(args.profile_seed), "--normal-rounds", "1",
        "--backtrack-limit", "100", "--resume",
    ], environment, output_dir / "logs" / "prepare.log")
    manifest = manifest_dir / "training_manifest.json"
    training = {}
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for group in GROUPS:
        model_dir = output_dir / "training" / group
        lock = "off" if group == "no_lock" else "on"
        _run([
            sys.executable, "-m", "rl_podem.training", str(manifest),
            str(model_dir), "--encoder", "level_gat_gru", "--ablation", group,
            "--rounds", "1", "--seed", str(args.seed),
            "--backtrace-lock", lock, "--resume",
        ], environment, output_dir / "logs" / f"train_{group}.log")
        for filename in ("training_state.pth", "inference_round_01.pth", "model_round_01.txt"):
            if not (model_dir / filename).is_file():
                raise RuntimeError(f"Missing {group} artifact: {filename}")
        training[group] = {
            "encoder_variant": VARIANTS[group],
            "backtrace_lock": lock == "on",
            "model_dir": str(model_dir),
        }
    summary = {
        "format": "SMARTATPG_GAT_ABLATION_TRAINING_V1",
        "dataset_root": str(dataset_root),
        "manifest": str(manifest),
        "seed": args.seed,
        "profile_seed": args.profile_seed,
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

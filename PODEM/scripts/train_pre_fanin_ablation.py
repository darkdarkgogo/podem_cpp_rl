"""Train three pre-fanin GAT ablations concurrently on three GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ROUNDS = 1
TRAINED_GROUPS = ("no_cache", "no_lock", "no_object_value")
VARIANTS = {
    "no_cache": "level_gat_gru",
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


def _environment(gpu):
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT / "python"), environment.get("PYTHONPATH", ""))),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _run_parallel(jobs):
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(_run, command, environment, log_path): group
            for group, command, environment, log_path in jobs
        }
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        raise RuntimeError("Ablation training failed: " + "; ".join(failures))


def _manifest_dataset_root(manifest):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (data.get("encoder_variant") != "level_gat_gru" or
            data.get("normal_rounds") != ROUNDS or
            data.get("backtrack_limit") != 100):
        raise ValueError("The training manifest must use the one-round pre-fanin GAT protocol")
    root = Path(data["dataset_root"])
    return (root if root.is_absolute() else manifest.parent / root).resolve(), data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path,
                        help="Use an existing GAT training manifest instead of preparing one")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/pre_fanin_ablation_1round")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-seed", type=int, default=14)
    parser.add_argument("--no-cache-gpu", type=int, default=0)
    parser.add_argument("--no-lock-gpu", type=int, default=1)
    parser.add_argument("--no-object-value-gpu", type=int, default=2)
    args = parser.parse_args(argv)
    gpus = {
        "no_cache": args.no_cache_gpu,
        "no_lock": args.no_lock_gpu,
        "no_object_value": args.no_object_value_gpu,
    }
    if min(gpus.values()) < 0 or len(set(gpus.values())) != len(TRAINED_GROUPS):
        raise ValueError("The three ablations require distinct non-negative GPU IDs")
    output_dir = args.output_dir.resolve()
    manifest = (
        args.manifest.resolve() if args.manifest is not None
        else output_dir / "preparation" / "gat" / "training_manifest.json"
    )
    for group, gpu in gpus.items():
        _run([
            sys.executable, "-c",
            "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "
            "'One visible CUDA GPU is required'",
        ], _environment(gpu), output_dir / "logs" / f"gpu_preflight_{group}.log")
    if args.manifest is None:
        dataset_root = args.dataset_root.resolve()
        if output_dir == dataset_root or dataset_root in output_dir.parents:
            raise ValueError("Ablation output must be separate from the dataset")
        _run([
            sys.executable, "-m", "rl_podem.data_split", str(dataset_root),
            str(manifest.parent), "--encoder", "level_gat_gru",
            "--seed", str(args.profile_seed), "--normal-rounds", str(ROUNDS),
            "--backtrack-limit", "100", "--resume",
        ], _environment(None), output_dir / "logs" / "prepare_gat.log")
    if not manifest.is_file():
        raise FileNotFoundError(f"GAT training manifest is missing: {manifest}")
    dataset_root, manifest_data = _manifest_dataset_root(manifest)
    if args.manifest is not None and args.dataset_root != ROOT / "data":
        if args.dataset_root.resolve() != dataset_root:
            raise ValueError("--dataset-root differs from the supplied manifest")
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise ValueError("Ablation output must be separate from the dataset")
    groups = {}
    jobs = []
    for group in TRAINED_GROUPS:
        model_dir = output_dir / "training" / group
        lock = "off" if group == "no_lock" else "on"
        command = [
            sys.executable, "-m", "rl_podem.training", str(manifest),
            str(model_dir), "--encoder", "level_gat_gru", "--ablation", group,
            "--rounds", str(ROUNDS), "--seed", str(args.seed),
            "--backtrace-lock", lock, "--resume",
        ]
        jobs.append((
            group, command, _environment(gpus[group]),
            output_dir / "logs" / f"train_{group}.log",
        ))
        groups[group] = {
            "encoder_variant": VARIANTS[group],
            "backtrace_lock": lock == "on",
            "model_dir": str(model_dir),
            "gpu": gpus[group],
            "logits_cache": group != "no_cache",
        }
    _run_parallel(jobs)
    for group in TRAINED_GROUPS:
        model_dir = output_dir / "training" / group
        for filename in ("training_state.pth", "inference_round_01.pth", "model_round_01.txt"):
            if not (model_dir / filename).is_file():
                raise RuntimeError(f"Missing {group} artifact: {filename}")
    summary = {
        "format": "SMARTATPG_PRE_FANIN_ABLATION_TRAINING_V3_THREE_GROUPS",
        "base_commit": "7d92ab8",
        "dataset_root": str(dataset_root),
        "manifest": str(manifest),
        "seed": args.seed,
        "profile_seed": manifest_data["profile_seed"],
        "rounds": ROUNDS,
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

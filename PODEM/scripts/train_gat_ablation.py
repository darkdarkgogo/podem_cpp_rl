"""Train three GAT ablations concurrently without a full GAT run."""

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


def _read_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f"GAT training manifest is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("encoder_variant") != "level_gat_gru" or
            data.get("normal_rounds") != 1 or
            data.get("backtrack_limit") != 100):
        raise ValueError("A one-round GAT training manifest is required")
    root = Path(data["dataset_root"])
    return (root if root.is_absolute() else path.parent / root).resolve(), data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/gat_ablation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile-seed", type=int, default=14)
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
    output_dir = args.output_dir.resolve()
    old_summary_path = output_dir / "train_summary.json"
    old_summary = (
        json.loads(old_summary_path.read_text(encoding="utf-8"))
        if old_summary_path.is_file() else None
    )
    if old_summary is not None and old_summary.get("format") not in (
        "SMARTATPG_GAT_ABLATION_TRAINING_V2", "SMARTATPG_GAT_ABLATION_TRAINING_V3_THREE_GROUPS",
    ):
        raise ValueError("Existing GAT ablation training summary is incompatible")
    manifest = (
        args.manifest.resolve() if args.manifest is not None else
        Path(old_summary["manifest"]).resolve() if old_summary is not None else
        output_dir / "preparation" / "gat" / "training_manifest.json"
    )
    if old_summary is None and args.manifest is None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((
            str(ROOT / "python"), environment.get("PYTHONPATH", ""),
        ))
        preparation = [
            sys.executable, "-m", "rl_podem.data_split", str(args.dataset_root.resolve()),
            str(manifest.parent), "--encoder", "level_gat_gru",
            "--seed", str(args.profile_seed), "--normal-rounds", "1",
            "--backtrack-limit", "100", "--resume",
        ]
        if _run(preparation, environment, output_dir / "logs" / "prepare_gat.log", "[prepare] "):
            raise RuntimeError("GAT training data preparation failed")
    dataset_root, manifest_data = _read_manifest(manifest)
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise ValueError("Ablation output must be separate from the dataset")
    if old_summary is not None:
        if (Path(old_summary["manifest"]).resolve() != manifest or
                Path(old_summary["dataset_root"]).resolve() != dataset_root):
            raise ValueError("Existing GAT ablation manifest or dataset differs")
    seed = old_summary["seed"] if old_summary is not None else (
        2026 if args.seed is None else args.seed
    )
    if args.seed is not None and args.seed != seed:
        raise ValueError("Training seed differs from existing GAT ablations")
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
                "--rounds", "1", "--seed", str(seed),
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
        "format": "SMARTATPG_GAT_ABLATION_TRAINING_V3_THREE_GROUPS",
        "dataset_root": str(dataset_root),
        "manifest": str(manifest),
        "seed": seed,
        "profile_seed": manifest_data["profile_seed"],
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

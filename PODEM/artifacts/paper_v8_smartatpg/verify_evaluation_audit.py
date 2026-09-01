"""Exercise the audit on completed historical data without modifying it."""

from pathlib import Path
from unittest.mock import patch

import evaluate_experiment as experiment


def main():
    old = experiment.OLD
    original_load, original_sha = experiment.load, experiment.sha
    original_read = Path.read_text
    historical_log = original_read(old / "train.log", encoding="utf-8")
    duplicate = next(line for line in historical_log.splitlines() if line.startswith("ROLLOUT_RESULT "))

    def mapped(path):
        path = Path(path)
        return old / "training_manifest.json" if path.name == "experiment_manifest.json" else path

    def read_with_duplicate(path, *args, **kwargs):
        if path == old / "train.log":
            return historical_log + '\nROLLOUT_RESULT {"truncated":\n' + duplicate + "\n"
        return original_read(path, *args, **kwargs)

    with patch.object(experiment, "RUN", old), \
            patch.object(experiment, "load", lambda path: original_load(mapped(path))), \
            patch.object(experiment, "sha", lambda path: original_sha(mapped(path))), \
            patch.object(experiment, "save") as save, \
            patch.object(Path, "read_text", read_with_duplicate):
        audit = experiment.audit_training()
        assert audit["episodes"] == 4200
        assert audit["ppo_updates"] == 4001
        assert audit["decision_steps"] == 358655
        assert audit["raw_rollout_log_records"] == 4202
        assert audit["duplicate_or_uncommitted_log_records"] == 2
        assert audit["malformed_rollout_log_records"] == 1
        assert save.call_count == 1
    print("AUDIT_VERIFIED historical episodes=4200 updates=4001 steps=358655 duplicate=1 truncated=1; no historical artifacts written")


if __name__ == "__main__":
    main()

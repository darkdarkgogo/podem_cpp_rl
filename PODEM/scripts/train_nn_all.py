"""Train the NN-All PODEM baseline for 30,000 epochs on paper circuits."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from rl_podem.nn_all_training import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "nn_all_podem")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    run_training(
        args.dataset_root, args.output_dir, seed=args.seed,
    )


if __name__ == "__main__":
    main()

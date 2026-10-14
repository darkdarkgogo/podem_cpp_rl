"""Validate the pre-fanin GAT ablation groups."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rl_podem.pre_fanin_ablation import main


if __name__ == "__main__":
    main()

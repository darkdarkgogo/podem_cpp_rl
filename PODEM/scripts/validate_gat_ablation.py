"""Validate the four GAT ablation groups on one ordered fault catalog."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from rl_podem.gat_ablation import main


if __name__ == "__main__":
    main()

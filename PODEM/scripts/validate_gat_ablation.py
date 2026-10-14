"""Validate three GAT ablations against an existing full GAT result."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from rl_podem.gat_ablation import main


if __name__ == "__main__":
    main()

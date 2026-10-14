"""Validate the old GAT Actor with its native logits cache disabled."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rl_podem.pre_fanin_ablation import no_cache_main


if __name__ == "__main__":
    no_cache_main()

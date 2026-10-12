# NN-All paper training circuits

The NN-All paper uses the complete ISCAS85 circuits `c1908` and `c2670` for training. This directory keeps the downloaded BENCH source under `source/` and the equivalent, binary, BUF-free BENCH files used by this project's PODEM and feature extractor at the top level.

| Circuit | Source SHA-256 | Normalized SHA-256 | URL |
| --- | --- | --- | --- |
| `c1908.bench` | `948e4c168d412bb5f1507a8c926125d851db5e877132c869ee95fbd4adbd2b81` | `571a2217fea0f599c5d0fd661ab4892ea17690027785b6ba665fa26cf101e902` | https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench/c1908.bench |
| `c2670.bench` | `f9cea6f6734725b8ce22c938e5c71ba527a998880b10ed7253b948974c23d928` | `e805c8b1a0e20d970bcd2f24cded9ad885f856edd2f9008485d05af1256eb1d8` | https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench/c2670.bench |

The sources were retrieved on 2026-10-12. To regenerate the normalized files from `PODEM/`:

```python
from pathlib import Path
from tools.generate_normal_bench_dataset import parse_bench, normalize_bench, write_bench

root = Path("data/nn_all_paper_train")
for name in ("c1908.bench", "c2670.bench"):
    write_bench(
        normalize_bench(parse_bench(root / "source" / name)), root / name,
        header=(
            "Normalized from ISCAS85 " + name,
            "Source: https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench/" + name,
        ),
    )
```

Normalization removes buffers, lowers complex gates to the supported binary gates, and preserves the circuit logic. It changes gate count and internal line names, so fault counts here need not match the original FAN/ATALANTA counts.

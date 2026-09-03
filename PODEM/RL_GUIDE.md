# SmartATPG RL 使用入口

当前 SmartATPG 使用11维 gate 特征、一次 fanin-mean GraphSAGE 和13维 policy state。训练环境与 C++ 编译评测环境已经分离。

完整中文说明见 [`docs/SMARTATPG_11D_使用说明.md`](docs/SMARTATPG_11D_使用说明.md)。

训练环境：

```bash
python scripts/run_smartatpg_training_linux.py \
  --output-dir artifacts/smartatpg_paper_11d
```

编译评测环境：

```bash
python3 scripts/run_smartatpg_benchmark_linux.py \
  /path/to/benchmark_bundle \
  --output-dir benchmark_results
```

正式时间比较只使用 C++ PODEM 报告的 ATPG 区间时间，不包含 GraphSAGE embedding、编译和 Python 编排时间。

DeepGate 源码当前保持不变，后续将单独替换为 DeepGate2。

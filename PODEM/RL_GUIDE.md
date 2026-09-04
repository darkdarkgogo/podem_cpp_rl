# SmartATPG RL 使用入口

当前对比包含11维 fanin-mean SmartATPG 和11维逐 level 双向 GAT-GRU。Actor 只接收11维 gate embedding；两位 action mask 是逻辑决策状态的一部分，仅在 logits 之后用于屏蔽，不进入 Actor。训练环境与 C++ 编译评测环境已经分离。

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

正式时间比较只使用 C++ PODEM 报告的 ATPG 区间时间，不包含图 embedding、编译和 Python 编排时间。

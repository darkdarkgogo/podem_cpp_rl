# SmartATPG RL 使用入口

当前正式训练与评测使用逐 level 双向 GAT-GRU（agentATPG）。节点初始特征和图 embedding 均为11维：6维 gate 类型 one-hot（不含 BUF）加 level、fanout、静态 SCOAP CC0、CC1、CO。Actor/Critic 在11维 embedding 后拼接1维目标值 `object_val`，因此输入为12维；mask 仅在 logits 后使用。训练环境与 C++ 编译评测环境已经分离。

完整中文说明见 [`docs/SMARTATPG_11D_使用说明.md`](docs/SMARTATPG_11D_使用说明.md)。

训练环境：

```bash
chmod +x train_smartatpg_linux.sh benchmark_smartatpg_linux.sh tensorboard_smartatpg_linux.sh
./train_smartatpg_linux.sh
```

训练脚本读取 `data/train` 的全部1024个电路和 `data/validation` 的6个固定验证电路。SCOAP PODEM 在 backtrack 上限200下建立完整 fault catalog；训练只使用 `outcome == 1` 的全部可检测 fault，验证不筛选、测试全部 fault。GAT-GRU 正常训练5轮，不再有额外强化阶段；最佳模型只由每轮验证结果决定。

同一任务中断后直接重跑命令即可从 `training_state.pth` 恢复。要把当前完整 checkpoint 迁移到另一批电路或 fault 清单继续训练，指定新的空输出目录并传入：

```bash
./train_smartatpg_linux.sh --output-dir artifacts/next_run \
  --dataset-root /path/to/next/data \
  --continue-from /path/to/current/best_training_state.pth
```

该方式完整继承图编码器、Actor、Critic、`policy_old`、PPO/RND 优化器、RND 统计和 PyTorch 随机数状态，但重新开始新任务的5轮进度与最佳验证记录。

TensorBoard：

```bash
./tensorboard_smartatpg_linux.sh
```

编译评测环境：

```bash
./benchmark_smartatpg_linux.sh
```

正式时间比较只使用 C++ PODEM 报告的 ATPG 区间时间，不包含图 embedding、编译和 Python 编排时间。

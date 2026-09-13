# SmartATPG RL 使用入口

当前正式训练与评测使用逐 level 双向 GAT-GRU（agentATPG）。节点初始特征和图 embedding 均为11维：6维 gate 类型 one-hot（不含 BUF）加 level、fanout、静态 SCOAP CC0、CC1、CO。Actor/Critic 在11维 embedding 后拼接1维目标值 `object_val`，因此输入为12维；mask 仅在 logits 后使用。训练环境与 C++ 编译评测环境已经分离。

完整中文说明见 [`docs/SMARTATPG_11D_使用说明.md`](docs/SMARTATPG_11D_使用说明.md)。

训练环境：

```bash
chmod +x train_smartatpg_linux.sh benchmark_smartatpg_linux.sh tensorboard_smartatpg_linux.sh
./train_smartatpg_linux.sh
```

训练脚本先为16个电路生成每个50条 hard-detected fault 的固定清单，再在指定 GPU 上训练 GAT-GRU 8轮，并执行最多5轮失败 fault 强化；完成后生成只含 GAT-GRU 模型的评测包。

TensorBoard：

```bash
./tensorboard_smartatpg_linux.sh
```

编译评测环境：

```bash
./benchmark_smartatpg_linux.sh
```

正式时间比较只使用 C++ PODEM 报告的 ATPG 区间时间，不包含图 embedding、编译和 Python 编排时间。

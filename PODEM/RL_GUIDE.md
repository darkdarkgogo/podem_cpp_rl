# SmartATPG RL 使用入口

当前对比包含12维 fanin-mean SmartATPG 基线和12维逐 level 双向 GAT-GRU（agentATPG）。节点特征包含静态 SCOAP CC0、CC1、CO。SmartATPG 的12维 embedding 直接进入 Actor/Critic；agentATPG 拼接1维目标值 object_val 后，以13维直接进入 Actor/Critic。两者都没有前置 gate_encoder 或目标值查表相加；mask 仅在 logits 后使用。训练环境与 C++ 编译评测环境已经分离。

完整中文说明见 [`docs/SMARTATPG_11D_使用说明.md`](docs/SMARTATPG_11D_使用说明.md)。

训练环境：

```bash
chmod +x train_smartatpg_linux.sh benchmark_smartatpg_linux.sh tensorboard_smartatpg_linux.sh
./train_smartatpg_linux.sh
```

TensorBoard：

```bash
./tensorboard_smartatpg_linux.sh
```

编译评测环境：

```bash
./benchmark_smartatpg_linux.sh
```

正式时间比较只使用 C++ PODEM 报告的 ATPG 区间时间，不包含图 embedding、编译和 Python 编排时间。

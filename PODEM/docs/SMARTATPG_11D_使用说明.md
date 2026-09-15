# SmartATPG 11D BUF-free 使用说明

## 模型结构

每个 gate 的初始特征和图 embedding 都固定为11维：

```text
[PI, AND, NAND, OR, NOR, NOT, level, fanout, CC0, CC1, CO]
```

前6维是 gate 类型 one-hot，不含 BUF。显式 `BUF`/`BUFF` 输入会被拒绝，应在生成数据时先规范化掉。fanout stem 是“一条 wire 被多个 gate 使用”的拓扑关系，不是 BUF gate，因此仍然保留。

当前正式模型为逐 level 双向 GAT-GRU。它先按 level 从 PI 到 PO 聚合 fanin，再用独立参数从 PO 到 PI 聚合 fanout，始终输出11维 embedding。Actor/Critic 再拼接当前1维目标值 `object_val`，所以输入为12维。两位 action mask 不进入 embedding 或 Actor/Critic，只在 logits 后屏蔽不可选动作。

CC0、CC1 从 PI 向 PO 计算，CO 从 PO 向 PI 反向计算。它们是静态结构代价，不随当前 PI 赋值变化，也不能单独证明故障可检测。

## 数据目录

默认训练数据位于：

```text
data/
├── train/       1024 个 .bench 电路
└── validation/  b12_C、b15_C、b17_C、b20_C、b21_C、b22_C
```

准备脚本严格检查训练电路数量、验证电路名称、重复名称和非 `.bench` 文件。manifest 中只写相对路径，因此把整个工程复制到 Linux 后可以直接重新准备或训练，不依赖 Windows 绝对路径。

如果 `data/` 在本地仍未纳入 Git，迁移到 Linux 时必须把它与源码一起单独复制；仅克隆代码仓库不会自动得到训练数据。

## fault 选择规则

准备阶段对每个训练和验证电路运行传统 SCOAP 启发式 PODEM，backtrack 上限统一为200，并遍历完整 collapsed fault catalog。

- 训练集：只从启发式结果 `outcome == 1` 的 fault 中选择。按 backtracks 降序、backtrace steps 降序、fault ID 升序稳定排序，每个电路最多保留最难检测的30个；可检测 fault 不足30个时全部保留，不使用 aborted 或 redundant fault 补足。
- 验证集：保留完整 fault catalog，`outcome == 0/1/2` 都进入每轮验证，不根据启发式结果筛选。

生成的 fault profile 会记录 `outcome`、backtracks 和 backtrace steps。准备过程可以按电路断点恢复；源电路、配置或已生成 profile 的哈希发生变化时会拒绝混用。

## 五轮训练流程

训练固定执行5轮，每轮分为训练和验证两个阶段：

1. 将1024个训练电路各自选出的最多30个难检测 fault 合并成当轮 episode 清单。
2. 使用 `seed + round` 对完整清单做可复现打乱；每个 fault 在这一轮恰好训练一次。
3. 每个 episode 都用 GAT-GRU 策略运行 PODEM，backtrack 上限为200，并立即进行一次 PPO/RND 更新。
4. 训练清单全部完成后，切换为确定性策略，在6个验证电路的完整 fault catalog 上逐项测试。验证阶段不写入 rollout、不更新模型。
5. 用验证结果选择最佳 checkpoint，比较顺序为：检出 fault 更多、总 backtracks 更少、总 backtrace steps 更少、总外在回报更高；完全相同时保留更早轮次。
6. 保存本轮验证结果和断点，然后进入下一轮。第5轮验证结束后停止，不再执行额外的失败 fault 强化训练。

因此，“五轮”不是把同一个电路连续跑5次，而是把完整训练 fault 集训练一遍并完整验证一次，重复5次。`model_best.txt` 是五次验证中最优轮次，`model_latest.txt` 是第5轮结束时的最新策略，两者可能不同。

## Linux 训练

在 `PODEM` 目录执行：

```bash
conda activate d2l
python -m pip install -r python-requirements.txt
python -m pip install -e .

chmod +x train_smartatpg_linux.sh benchmark_smartatpg_linux.sh \
  tensorboard_smartatpg_linux.sh
./train_smartatpg_linux.sh
```

默认输出目录是 `artifacts/smartatpg_top30_hard_5rounds_bt200`。`--gpu` 指定物理 GPU；子进程通过 `CUDA_VISIBLE_DEVICES` 只看到该卡，所以 PyTorch 内显示为 `cuda:0`。

主要输出：

- `preparation/training_manifest.json`：训练/验证电路及各自 episode fault 清单；
- `preparation/profiles/`：传统 SCOAP PODEM 的逐电路完整 fault profile；
- `smartatpg_gat_gru/training_state.pth`：同一任务断点恢复；
- `smartatpg_gat_gru/best_training_state.pth`：验证最优轮次的完整 checkpoint；
- `smartatpg_gat_gru/model_best.txt`：验证最优的 C++ 推理模型；
- `smartatpg_gat_gru/model_latest.txt`：最后一轮的 C++ 推理模型；
- `smartatpg_gat_gru/validation_metrics.json`：五轮完整验证汇总；
- `smartatpg_gat_gru/validation_records.jsonl`：当前或最后一轮的逐 fault 验证明细，用于断点恢复；
- `benchmark_bundle/`：可复制到无 PyTorch 环境进行原生评测的自包含目录。

同一 manifest 和配置中断后，重新运行 `./train_smartatpg_linux.sh` 会自动传入 `--resume`，从当前训练 fault 或验证 fault 边界继续。

## 迁移到新电路继续训练

要从当前 checkpoint 在另一批电路或另一份 fault manifest 上继续训练，使用新的空输出目录：

```bash
./train_smartatpg_linux.sh \
  --dataset-root /path/to/new/data \
  --output-dir artifacts/new_circuits \
  --continue-from /path/to/old/best_training_state.pth
```

`--continue-from` 执行完整继承：

- GAT-GRU 图编码器、Actor、Critic 和 `policy_old`；
- PPO optimizer 状态；
- RND predictor、固定 target、optimizer 和运行统计；
- PyTorch CPU/CUDA 随机数状态。

新任务的轮次、episode 位置、验证历史和 best 记录会清零，从新 manifest 的第1轮开始。`--resume` 只用于原任务，要求 manifest 哈希和全部训练配置完全一致；两者不能同时使用。旧格式 checkpoint 不兼容。

## 编译与评测

训练完成后运行：

```bash
./benchmark_smartatpg_linux.sh
```

评测包仍包含项目的16个标准 benchmark 电路，用相同 backtrack 上限200比较 SCOAP heuristic 和验证最优 GAT-GRU。正式 runtime 只采用 C++ 输出的 ATPG 区间时间，不包含图特征/embedding 计算、C++ 编译、Python 编排或进程启动时间。

新电路必须由同一份模型重新计算11维 embedding，不能复用其他电路的 embedding。原生日志还会记录 RL 策略选择次数、Actor 前向次数及相应耗时。

## 工件与兼容性

当前唯一支持的推理模型格式是 `SMARTATPG_MODEL_V12`，embedding 格式是 `SMARTATPG_EMBEDDINGS_V7`，benchmark bundle 格式是 `SMARTATPG_BENCHMARK_BUNDLE_V9_DATA_SPLIT_11D_CO_NO_BUF`。V12 模型记录训练 manifest 哈希、backtrack=200、normal rounds=5，以及训练/验证电路数量。

训练 manifest 使用 `SMARTATPG_DATA_SPLIT_MANIFEST_V6_TOP30_11D_CO_NO_BUF`，并记录 `train_faults_per_circuit=30`。旧 manifest 不能用于恢复 Top-30 任务；兼容的11维 V6 checkpoint 仍可通过 `--continue-from` 初始化新的 Top-30 训练，但不能通过 `--resume` 混入新 manifest。更早格式的 checkpoint、V10/V11 model 或其他 embedding 格式会在加载阶段被拒绝。新任务应使用空输出目录重新准备、训练和导出。

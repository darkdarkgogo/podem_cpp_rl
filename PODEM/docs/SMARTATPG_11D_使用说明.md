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
├── train/       GAT 使用的1024个 .bench 电路
├── train_mean/  mean 使用的 c6288 与 s38417 转换资产及 fault map
└── validation/  b12_C、b15_C、b17_C、b20_C、b21_C、b22_C
```

准备脚本按 encoder 检查数据：GAT 使用 `train`；mean 直接使用 `c6288.bench`，并在 `s38417_scan_binary.bench` 上加载 `s38417_scan_binary.faultmap`，使 ATPG 使用 binary 电路但 fault 数量和 ID 仍来自 scan 版本。`.uf` 文件不参与训练协议。manifest 中只写相对路径，因此复制到 Linux 后不依赖 Windows 绝对路径。

如果 `data/` 在本地仍未纳入 Git，迁移到 Linux 时必须把它与源码一起单独复制；仅克隆代码仓库不会自动得到训练数据。

## fault 选择规则

准备阶段对训练电路运行传统 SCOAP 启发式 PODEM，正式 backtrack 上限统一为100，并遍历完整 collapsed fault catalog。验证电路在训练时加载完整 catalog。

- GAT 训练集：只从 `outcome == 1` 中按 backtracks 降序、backtrace steps 降序、fault ID 升序选择，每个电路最多30个。
- mean 训练集：采用相同排序，每个电路最多100个；不足100个时使用全部已有的可检测 fault，不使用 aborted 或 redundant fault 补足。
- 验证集：保留完整 fault catalog，`outcome == 0/1/2` 都进入每轮验证，不根据启发式结果筛选。

生成的 fault profile 会记录 `outcome`、backtracks 和 backtrace steps。准备过程可以按电路断点恢复；源电路、配置或已生成 profile 的哈希发生变化时会拒绝混用。

## 两轮训练流程

训练固定执行2轮，每轮分为训练和验证两个阶段：

1. 按当前 encoder 的 manifest 合并训练 fault：GAT 为1024个电路各最多30个，mean 为两个电路各最多100个。
2. 使用 `seed + round` 对完整清单做可复现打乱；每个 fault 在这一轮恰好训练一次。
3. 每个 episode 使用对应 encoder 策略运行 PODEM，backtrack 上限为100；每累计8个 fault（末尾不足8个也刷新）进行一次 PPO/RND 更新。GAT 对同一批 rollout 执行4个 PPO epoch，Actor/Critic 学习率分别为 `0.0003`/`0.001`；mean 保持1个 PPO epoch，学习率分别为 `0.001`/`0.01`。RND predictor 每批仍只更新一次。
4. 训练清单全部完成后，切换为确定性策略，在6个验证电路的完整 fault catalog 上逐项测试。验证阶段不写入 rollout、不更新模型。
5. 用验证结果选择最佳 checkpoint，比较顺序为：检出 fault 更多、总 backtracks 更少、总 backtrace steps 更少、总外在回报更高；完全相同时保留更早轮次。
6. 保存本轮验证结果和断点，然后进入下一轮。第2轮验证结束后停止。

`model_best.txt` 是两次验证中的最优轮次，`model_latest.txt` 是第2轮结束时的最新策略，两者可能不同。

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

单 GAT 默认输出目录是 `artifacts/smartatpg_top30_hard_2rounds_batch8_bt100`；双模型默认输出到 `artifacts/smartatpg_dual_top30_hard_2rounds_batch8_bt100`，并为 GAT 和 mean 分别生成 manifest。`--gpu` 指定物理 GPU；子进程通过 `CUDA_VISIBLE_DEVICES` 只看到该卡，所以 PyTorch 内显示为 `cuda:0`。

主要输出：

- `preparation/training_manifest.json`：训练/验证电路及各自 episode fault 清单；
- `preparation/profiles/`：传统 SCOAP PODEM 的逐电路完整 fault profile；
- `smartatpg_gat_gru/training_state.pth`：同一任务断点恢复；
- `smartatpg_gat_gru/best_training_state.pth`：验证最优轮次的完整 checkpoint；
- `smartatpg_gat_gru/model_best.txt`：验证最优的 C++ 推理模型；
- `smartatpg_gat_gru/model_latest.txt`：最后一轮的 C++ 推理模型；
- `smartatpg_gat_gru/validation_metrics.json`：两轮完整验证汇总；
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

新任务的轮次、episode 位置、验证历史和 best 记录会清零，从新 manifest 的第1轮开始。`--resume` 只用于原任务，要求 manifest 哈希和全部训练配置完全一致；两者不能同时使用。旧 GAT epoch-1 checkpoint 不兼容；mean 的 epoch-1 checkpoint 语义不变。

## 编译与评测

训练完成后运行：

```bash
./benchmark_smartatpg_linux.sh
```

评测包仍包含项目的16个标准 benchmark 电路，用相同 backtrack 上限100比较 SCOAP heuristic 和验证最优模型。正式 runtime 只采用 C++ 输出的 ATPG 区间时间，不包含图特征/embedding 计算、C++ 编译、Python 编排或进程启动时间。

新电路必须由同一份模型重新计算11维 embedding，不能复用其他电路的 embedding。原生日志还会记录 RL 策略选择次数、Actor 前向次数及相应耗时。

## 工件与兼容性

正式 GAT 推理模型格式是 `SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4`，只接受 `level_gat_gru`、batch8/epoch4；旧 GAT V12/V13 模型会被拒绝。mean 保持 `SMARTATPG_MODEL_V13_BATCH8_EPOCH1`，仍接受 `fanin_mean`、batch8/epoch1，并继续兼容其旧 V12 工件。两种模型都记录训练 manifest 哈希、backtrack=100、normal rounds=2 和训练/验证电路数量。embedding 格式仍是 `SMARTATPG_EMBEDDINGS_V7`，benchmark bundle 格式仍是 `SMARTATPG_BENCHMARK_BUNDLE_V9_DATA_SPLIT_11D_CO_NO_BUF`。

当前训练 manifest 使用 `SMARTATPG_DATA_SPLIT_MANIFEST_V9_ENCODER_TRAIN_SPLIT_11D_CO_NO_BUF`，记录 encoder、训练 split 和每电路 fault 上限。GAT 与 mean 的 manifest 不可互换；旧 manifest 不能通过 `--resume` 混入新任务。新任务应使用空输出目录重新准备、训练和导出。

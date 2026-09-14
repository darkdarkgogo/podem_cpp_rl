# 基于训练集与验证集划分的 SmartATPG 训练设计

## 目标

将当前固定16个电路的训练流程替换为数据集驱动流程：

- 使用 `PODEM/data/train` 下的全部 BENCH 电路训练；
- 使用 `PODEM/data/validation` 下的全部 BENCH 电路选择最佳模型；
- 启发式筛选、PPO 训练、验证以及后续基准评测统一使用200次 backtrack 上限；
- 只执行5轮普通 PPO 训练，删除额外的失败 fault 强化阶段；
- 当前11维、无 BUF 的 checkpoint 可以在保留完整学习状态的前提下，继续训练其他电路和 fault。

图特征及工件协议继续使用 `SMARTATPG_FEATURES_V4_11D_CO_NO_BUF`。本次改动不改变 C++ PODEM 内部的 fanout stem 表示。

## 数据集约束

默认数据集根目录为 `PODEM/data`，其中包含：

- `train/`：恰好1024个归一化后的组合逻辑 BENCH 子电路；
- `validation/`：6个完整验证电路：`b12_C`、`b15_C`、`b17_C`、`b20_C`、`b21_C` 和 `b22_C`。

两个目录都按照文件名稳定排序。每个输入必须是普通 `.bench` 文件，文件名主干必须唯一，并且必须通过11维、无 BUF 的 Python 图加载器校验。存在显式 `BUF`/`BUFF`、不支持的门、错误驱动关系或组合环时，必须在开始 fault profiling 前拒绝该数据集。

preparation 输出与源数据集分开保存，记录源文件路径、SHA-256、电路哈希、图统计信息、profiling 工件以及 fault 清单。数据集文件被修改、增加、删除或重命名后，已有 manifest 必须判定为不兼容。

## Fault profiling 与数据划分语义

preparation 使用传统启发式 SCOAP PODEM，在 `backtrack_limit=200` 下枚举并测试每个电路的完整折叠 fault catalog。

对于训练电路，manifest 保留启发式 profile 中所有 `outcome == 1` 的 fault。不进行难度排名、不截取 hard fault，也不要求每个电路具有固定 fault 数量。如果某个训练电路没有任何 `outcome == 1` fault，则直接报错，因为该电路无法产生有效训练 episode。

对于验证电路，manifest 保留枚举出的完整 fault catalog，不根据启发式结果过滤。可检测 fault、不可测 fault，以及启发式在200次 backtrack 内未解决的 fault 都必须成为验证 episode。启发式 outcome 只作为 profiling 元数据，不能用于过滤验证集。

训练集和验证集必须通过目录及 manifest 分区保持隔离。验证 episode 不得调用 optimizer 更新，也不得修改 PPO 或 RND 的训练及统计状态。

## Preparation 与断点恢复

preparation 采用固定 manifest 方案。数据集只需完整扫描一次；每完成一个电路，就立即写入对应的独立 profile 文件。原子写入的 preparation 状态记录数据集清单和已完成的 profile，使包含1030个电路的 profiling 过程在中断后可以继续，而不必重复处理已经完成且哈希一致的电路。

所有 profile 完成并通过校验后，preparation 才原子发布正式训练 manifest。manifest 使用独立的 `train_circuits` 和 `validation_circuits` 字段，不能将两种电路混在同一个列表中。每条电路记录包含 BENCH 路径与哈希、完整 profile 路径与哈希，以及对应的 episode fault ID：

- 训练记录只包含 `outcome == 1` 的 fault ID；
- 验证记录包含完整 catalog 的全部 fault ID。

恢复 preparation 时，必须校验复用的每个源文件及 profile 文件哈希。如果任一文件发生变化，应明确报错；用户需要选择新的 preparation 目录或显式重新生成 manifest。局部状态和最终文件都通过临时文件替换的方式原子写入。

## 五轮训练流程

Linux 启动器先准备数据集 manifest，再为全部训练电路创建图和原生 trainer，并为全部验证电路创建只读 evaluator。训练固定执行5轮，backtrack 上限固定为200。

每轮中，每个被保留的训练 fault 恰好出现一次。给定相同 seed 和轮次时，episode 顺序必须确定；所有训练电路的 episode 会放在一起打乱，避免文件名顺序形成隐式课程学习。保留现有的逐 episode checkpoint，使训练在一轮中断后能够从下一个 episode 继续，不重复已经完成的参数更新。

每个完整训练轮次结束后，当前 policy 必须在验证集的每一个 fault 上恰好评估一次，而且不得更新任何参数或 RND 统计。最佳模型的比较顺序如下：

1. 验证集检出 fault 数量最大，即固定 catalog 下的覆盖率最高；
2. 验证集总 backtracks 最少；
3. 验证集总 backtrace steps 最少；
4. 验证集总 extrinsic return 最大；
5. 以上完全相同时，优先保留更早的轮次。

latest checkpoint 与 best checkpoint 分开保存。如果程序在验证过程中中断，必须记录验证进度；恢复后从未完成的验证 fault 继续，不能进入下一轮训练，也不能使用不完整的验证结果更新最佳模型。

从本流程中删除失败 fault 强化阶段及其 checkpoint、metrics、未解决 fault 清单、CLI 参数、启动器步骤和 reinforced model 输出。最终 benchmark bundle 使用验证集选出的 `model_best.txt`。

## Checkpoint 使用模式

系统支持两种含义不同的 checkpoint 模式。

### 恢复同一个训练任务

`--resume` 要求 manifest 哈希、输出目录、5轮配置、backtrack 上限、编码器类型和 seed 完全一致。恢复内容包括全部模型和学习状态，以及精确的训练轮次、episode、验证进度和随机数生成器状态。

### 在不同数据集上继续训练

`--continue-from <checkpoint>` 用于启动一个新的训练任务，并且不能与恢复已有进度同时使用。它只接受当前无 BUF、11维的 training checkpoint 或 best checkpoint，并恢复完整 agent 学习状态：

- 图编码器、Actor、Critic 和 old-policy 权重；
- PPO optimizer 状态；
- RND target、predictor、optimizer 及归一化统计；
- 连续训练所需的模型侧随机状态。

以下内容不从旧任务继承：旧 manifest 绑定、旧电路 trainer、轮次和 episode 计数、验证进度、最佳分数及最佳 checkpoint。新任务根据新的 manifest 重新初始化这些状态，并完全使用新的验证集重新选择最佳模型。

Tensor 形状、编码器类型、特征 schema、图配置及 checkpoint 格式必须严格匹配。旧12维 checkpoint 和其他历史工件格式必须拒绝。

新输出目录记录来源 checkpoint 的路径及 SHA-256，保证继续训练的来源可追踪。`--continue-from` 不能写入已有内容的输出目录；非空输出目录只能通过同任务 `--resume` 使用，禁止静默覆盖。

## Linux 入口与输出工件

`train_smartatpg_linux.sh` 默认使用：

- 数据集根目录 `data`；
- 5轮训练；
- 200次 backtrack 上限；
- 物理 GPU 0 上的 GAT-GRU 编码器；
- 名称中明确标记11维、无 BUF、数据集划分和5轮协议的新输出目录。

启动器生成：

- `preparation/training_manifest.json` 及可断点恢复的逐电路 profile；
- `smartatpg_gat_gru/training_state.pth`，用于恢复同一训练任务；
- `smartatpg_gat_gru/best_training_state.pth`，只由验证集结果决定；
- `smartatpg_gat_gru/model_latest.txt` 和 `model_best.txt`；
- 每轮训练与验证 metrics，以及 TensorBoard events；
- 只包含验证集最佳 GAT-GRU 模型的 `benchmark_bundle/`。

导出的 Actor 升级为 `SMARTATPG_MODEL_V12`。其头部记录 preparation manifest 哈希、5轮训练协议及 backtrack 上限，不把1024个电路名称全部写入模型头。portable 和 C++ 原生加载器只接受该新模型格式。

11维 descriptor 文件结构不变，继续使用 `SMARTATPG_EMBEDDINGS_V7`，并通过 snapshot 哈希与模型配对。manifest、checkpoint、training state 和 benchmark bundle 的格式标识全部升级，防止旧的16电路、每电路50 fault、8轮协议被误认为当前流程。

绝对路径不作为工件身份。manifest 保存相对于 preparation 或数据集根目录的可迁移路径，并同时保存内容哈希，使仓库和数据集从 Windows 移至 Linux 后仍能继续使用。

## 错误处理

出现以下情况时必须明确失败：

- 任一数据划分不存在、为空、存在重复文件名主干，或者选中的处理项不是 BENCH 文件；
- 任一图违反11维无 BUF 输入协议；
- profiling 失败、返回重复 fault ID，或者训练电路不存在 `outcome == 1` fault；
- 恢复过程中发现源文件或 profile 哈希变化；
- 验证没有对 manifest 中完整 fault catalog 恰好执行一次；
- 同任务恢复配置不一致；
- 继续训练使用的 checkpoint 架构或格式不兼容；
- 同时请求 `--resume` 和 `--continue-from`；
- 继续训练的目标输出目录不是空目录。

错误信息必须包含数据划分、被处理电路、源路径和操作名称，确保大规模数据集中的失败可以直接定位。

## 验证要求

测试必须覆盖：

1. 按稳定顺序发现全部1024个训练 BENCH 和6个验证 BENCH；
2. 训练 fault 清单包含所有且仅包含 `outcome == 1` 的 profile；
3. 验证 fault 清单包含完整 catalog，包括 outcome 不为1的 fault；
4. profiling、训练、验证、Linux 启动器和 benchmark 元数据的 backtrack 上限均为200；
5. 确定性 episode 顺序中，每个保留的训练 fault 恰好出现一次；
6. 验证过程不更新 optimizer 或 RND，并且完全控制最佳模型选择；
7. 只允许5轮普通训练，且不运行强化阶段；
8. preparation 可以按电路恢复 profiling，并拒绝哈希发生变化的文件；
9. 同任务 resume 能恢复精确进度；
10. 跨 manifest 继续训练能恢复完整学习状态，同时重置进度和最佳验证状态；
11. 不兼容的历史或12维 checkpoint 会被拒绝；
12. manifest 和 checkpoint 在仓库迁移到 Linux 路径后仍能使用；
13. 小型端到端 fixture 能完成训练、验证、resume、切换 manifest 继续训练、导出最佳 V12 模型并准备 benchmark bundle。

生产数据集的完整 profiling 和5轮 GPU 训练可能包含大量 fault episode，属于正式运行验收，不作为普通单元测试执行。

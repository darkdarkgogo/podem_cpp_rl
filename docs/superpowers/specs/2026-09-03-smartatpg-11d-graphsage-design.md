# SmartATPG 11维 GraphSAGE 状态设计

## 范围

本次只修改 SmartATPG 论文复现流程。DeepGate 以及计划中的 DeepGate2 迁移均不在本次修改范围内。

SmartATPG 策略必须使用每个 gate 自己的 GraphSAGE embedding，不使用整张电路的全局 pooling embedding，不进行行为克隆（BC），也不使用课程训练。

## Gate 特征

支持的 gate 类型及其 one-hot 固定顺序为：

`PI, AND, NAND, OR, NOR, NOT, BUF`

不支持 `XOR` 和 `XNOR`。SmartATPG 的 BENCH 解析器遇到这两种 gate 时必须给出明确错误。进入 SmartATPG 预处理前，应由现有二值化转换流程将不支持的逻辑门转换为受支持的 gate。

每个 gate 的初始特征向量为11维：

| 字段 | 维度 | 定义 |
| --- | ---: | --- |
| Gate 类型 | 7 | 按上述固定顺序编码的 one-hot |
| Level | 1 | 拓扑层级，在当前电路内归一化 |
| Fanout | 1 | 该 gate 驱动的输入引脚数量，使用对数归一化 |
| CC0 | 1 | 结构化 SCOAP 0-可控性，使用对数归一化 |
| CC1 | 1 | 结构化 SCOAP 1-可控性，使用对数归一化 |

不再计算或加入 SCOAP CO。必须更新特征 schema 标识，旧的14维特征、80维 descriptor 和相关 checkpoint 不能被当作新模型加载。

## GraphSAGE 编码器

编码器沿 fanin 方向只执行一轮均值聚合，输出仍保持为11维：

```text
h_v^0 = x_v
m_v^k = mean(h_u^(k-1) for u in fanins(v))
h_v^1 = ReLU(W [h_v^0 || m_v^1] + b)
```

拼接后的输入为22维，输出为11维，因此 GraphSAGE 层是一个可训练的 `Linear(22, 11)`。没有 fanin 的节点使用全零的11维邻居均值。

最终 gate embedding 是 `h_v^1`，维度为11。设计中不存在第二轮或第三轮聚合、整图 mean pooling、全局 graph context、64维隐藏表示或全局 context cache。

`W` 和 `b` 属于策略模型参数。它们随新的 SmartATPG 模型一起初始化，并通过 PPO 梯度与 Actor、Critic 一起训练。GraphSAGE 参数必须保存在 checkpoint 中，并参与 snapshot 身份计算。

## 策略状态与 Mask

对 gate `v` 进行 backtrace 决策时，策略状态定义为：

```text
state_v = [h_v^1 || action_mask]
```

Gate embedding 为11维。二输入 action mask 是独立的2维状态字段，因此 Actor/Critic 的 state 总维度为13。mask 同时用于屏蔽非法 action 的 logit，确保策略不能选择不可用的输入。

Mask 不属于 gate embedding，不能写入静态 gate embedding 表。Python 训练在每次决策时单独提供 mask；C++ 原生推理在选中当前 gate 的11维静态 embedding 后，再动态附加当前的两个 mask 值，然后计算策略输出。

保留现有 objective value embedding 和输出两个 logit 的 backtrace action head。强制移动继续由 PODEM 求解器控制。

## 训练

SmartATPG 使用独立的论文复现准备和训练入口，不再作为 `train_curriculum.py` 的训练分支。每次新的 SmartATPG 训练都从随机初始化的 GraphSAGE、Actor 和 Critic 开始执行 PPO。本次改动不改变 DeepGate 的训练行为。

训练只使用两个电路：

- `c6288`；
- 扫描化组合版本的 `s38417`。

每个电路选择100个 hard-to-detect fault，共计200个训练 fault。选择流程固定如下：

1. 使用不带神经网络策略的基础 PODEM；
2. 使用固定 seed 和统一的 backtrack limit，对该电路完整 fault catalog 中的每个 fault 分别运行；
3. 首先按 `backtracks` 从高到低排序；
4. `backtracks` 相同时，按 `backtrace_steps` 从高到低排序；
5. 仍然相同时，按稳定的 `fault_id` 字典序从小到大排序；
6. 直接取排序后的前100个，不随机打乱，也不划分 curriculum 难度阶段。

选定的 fault ID、排序统计、基础 PODEM seed、backtrack limit、电路与 fault map 哈希必须写入 manifest，以便重复实验时得到同一训练集合。

论文复现默认超参数固定为：

| 参数 | 数值 |
| --- | ---: |
| Actor 学习率 | `0.001` |
| Critic 学习率 | `0.01` |
| 奖励参数 alpha | `7.5` |
| 奖励参数 beta | `0.07` |
| 训练轮数 | `20` |

SmartATPG 无条件跳过行为克隆，也不执行 easy/medium/hard 分级训练、stage sweep 或 curriculum round。训练流程直接在上述200个固定 fault 上执行 PPO+RND。

一轮训练定义为：`c6288` 的100个训练 fault 和 `s38417` 的100个训练 fault 各执行一次，共执行200个 episode。默认训练20轮，即总计4000个训练 episode。每轮内的 fault 顺序使用 `训练 seed + round` 生成确定性随机排列，以降低固定顺序偏差，并确保断点恢复后顺序完全一致。

一个 fault 对应一个 episode。一个 episode 内先完整收集该 fault 的全部 backtrace 决策 step，episode 结束后立即执行一次 PPO 更新。该 episode 的全部 step 构成当前更新 batch，不与其他 fault 的 episode 合并，也不再切分 minibatch。因为每个 fault 的搜索路径长度不同，所以每次更新的 batch step 数可以不同。

Rollout buffer 只保存重算策略所需的电路身份、当前 gate 索引、mask、objective value、action、旧 log probability、旧 value 和 reward，不把 detached gate embedding 当作训练输入永久保存。执行 PPO 更新时，使用当前 GraphSAGE 参数重新计算该 episode 涉及 gate 的 `h_v^1`，使 PPO loss 的梯度能够更新 `W` 和 `b`。

一次 PPO update（内部可以按 PPO epoch 执行多次 optimizer step）完成后，把新策略同步为下一 episode 的采样策略，并使之前计算的全部 gate embedding cache 失效。下一个 episode 第一次使用某个电路时，必须用更新后的 `W` 和 `b` 重新计算该电路所有 gate 的 embedding。同一次 episode 内参数不变，可以复用本 episode 的 embedding 计算结果。

## 每轮评估与最佳参数

每轮训练完成后，冻结参数并使用确定性 `argmax` 策略重新运行固定的200个训练 fault。评估过程不采样 action、不计算 PPO 更新，也不更新 RND、GraphSAGE、Actor 或 Critic。

每轮生成一个完整 checkpoint。最佳参数采用以下字典序判定：

1. 检测到的 fault 数量最多；
2. 检测数量相同时，总 backtracks 最少；
3. 仍相同时，总 backtrace steps 最少；
4. 仍相同时，总 return 最高；
5. 全部相同时，保留更早的 round。

训练 episode 的 return 表示送入 PPO 的奖励总和，可包含按配置缩放后的 RND 内在奖励。每轮确定性评估的 return 只统计论文定义的外在 reward，不加入 RND bonus；总 return 是200个评估 episode 外在 return 的总和。使用 fault detection 作为第一条件，避免模型通过少检测 fault 获得虚假的低 backtracks，也避免随训练变化的 RND 预测误差干扰 best 参数选择。

训练目录至少保存：

- `training_state.pth`：最近一轮的完整状态，用于断点恢复；
- `best_training_state.pth`：最佳轮的 GraphSAGE、Actor、Critic 和相关训练元数据；
- `actor_best.txt`：最佳轮用于 C++ 推理的 Actor；
- `actor_latest.txt`：最近一轮的 Actor；
- `round_metrics.json`：20轮确定性评估记录；
- `tensorboard/`：TensorBoard event 文件。

保留论文定义的 reward、PPO、RND、checkpoint 保存、best model 选择和断点恢复能力。训练顺序必须由 manifest 和固定 seed 决定并可复现。

断点恢复只接受满足以下条件的新 checkpoint：使用新的特征 schema、单层 GraphSAGE 配置、11维 gate embedding 和13维 policy state。旧 SmartATPG checkpoint 必须被拒绝。

## Linux 训练入口

提供一个 Linux 一键入口，从 PODEM 仓库根目录完成数据准备、基础 PODEM fault 排序、20轮训练、最佳参数保存、C++ 原生程序构建和最终 benchmark。入口必须支持中断后从 `training_state.pth` 精确恢复；已经完成的 fault、round 评估和 benchmark 组合不得重复覆盖。

默认使用 CUDA GPU；CUDA 不可用时允许回退到 CPU，并在日志和运行元数据中明确记录实际 device。运行目录保存完整命令、Python/PyTorch/CUDA 版本、随机 seed、训练耗时和最终退出状态。

训练时可以从另一个终端启动：

```bash
tensorboard --logdir artifacts/<run-name>/tensorboard --host 0.0.0.0 --port 6006
```

本机浏览器访问 `http://localhost:6006`；从另一台电脑查看时，使用 Linux 主机地址或 SSH 端口转发。

## 训练后全电路比较

20轮训练完成后，冻结 `best_training_state.pth`，为每个目标电路重新计算该 best GraphSAGE 对应的11维 gate embedding，并只比较以下两种策略：

- `heuristic`：不加载神经网络的基础 PODEM；
- `rl_best`：加载 best GraphSAGE embedding 和 best Actor 的 SmartATPG。

默认 benchmark 覆盖论文中的全部 ISCAS'85 和 ISCAS'89 电路：

```text
c432, c499, c1355, c1908, c2670, c3540, c5315, c6288, c7552,
s5378, s9234, s13207, s15850, s35932, s38417, s38584
```

其中时序电路先采用全扫描方式转换成组合电路，再执行相同的二输入 gate 规范化和 fault map 生成。当前 workspace 中16个原始 BENCH 均可找到；准备流程必须把最终使用的规范化电路和哈希固定到 benchmark manifest。任何清单文件缺失或哈希变化都必须终止并明确报告，不能静默跳过电路。

每个电路都运行完整 fault catalog。两种策略必须使用完全相同的规范化电路、fault map、fault 顺序、seed 和 backtrack limit。每个“策略×电路”先执行1次不计入统计的预热，再正式执行5次；运行顺序按 repeat 和 circuit 轮换，降低先后顺序造成的系统偏差。

每个电路和全部电路汇总以下指标：

- detected、aborted、redundant、fault coverage 和 test vectors；
- total backtracks；
- total backtrace steps；
- ATPG runtime；
- 整个原生进程 wall time；
- RL 相对 heuristic 的 backtracks、backtrace steps 和 runtime 改善百分比。

改善百分比统一定义为：

```text
improvement = (heuristic - rl_best) / heuristic * 100%
```

正数表示 RL 更好，负数表示 RL 更差。Backtracks 和 backtrace steps 应在相同 seed 下保持确定；runtime 使用5次正式运行的中位数，同时保留每次原始结果。GraphSAGE 图解析和 gate embedding 导出时间单独报告，不计入 ATPG runtime；整个进程 wall time另行报告，不得与 ATPG 搜索时间混为一项。

最终生成机器可读的 JSON、CSV，面向人工检查的 Markdown 汇总，以及每次原生运行的完整日志。主比较表只展示 `heuristic` 和 `rl_best`。

## TensorBoard 监控

训练期间继续写入 TensorBoard。至少提供以下逐 episode 指标：

- `episode/backtracks`；
- `episode/backtrace_steps`；
- `episode/return`；
- `episode/extrinsic_return` 和 `episode/intrinsic_return`；
- `episode/detected`；
- `episode/ppo_loss`；
- `episode/rnd_loss`。

每轮确定性评估至少写入：

- `round/backtracks_total` 和 `round/backtracks_mean`；
- `round/backtrace_steps_total` 和 `round/backtrace_steps_mean`；
- `round/return_total` 和 `round/return_mean`；
- `round/detected_faults` 和 `round/fault_coverage`；
- `round/is_best`。

TensorBoard 的 global step 对逐 episode 指标使用已完成的训练 episode 数，对逐 round 指标使用 round 编号。断点恢复时必须继续原有 step，不能覆盖或重置历史曲线。

## Artifact 与 C++ 原生推理

训练期间不把 gate embedding 当作固定输入文件。只有导出 C++ 原生推理模型时，才冻结某个 checkpoint 的 GraphSAGE，并为目标电路中的每个 gate 导出一行11维 `h_v^1`。Actor 元数据记录 backend、特征 schema、GraphSAGE 配置、gate embedding 维度、policy state 维度和 snapshot 身份。Actor 与 gate embedding 文件必须来自同一个 snapshot。

C++ artifact reader 必须区分11维 gate embedding 和13维 policy state。旧 SmartATPG 的80维 descriptor 及其 checkpoint 与新模型不兼容，加载时必须给出明确错误，不能静默地重新解释为新格式。

## 验证要求

测试必须覆盖：

- 11列 gate 特征的固定顺序和归一化结果；
- 输入电路含 XOR 或 XNOR 时明确拒绝；
- graph 数据和导出 artifact 中均不存在 CO；
- GraphSAGE 只包含一个 `Linear(22, 11)` 聚合层；
- 不存在整图 pooling 或共享的全电路 context；
- 一跳 fanin 特征发生变化时，对应 gate embedding 会改变；
- mask 不写入 embedding 文件，但会进入13维 policy state；
- PPO 更新重新计算 gate embedding，且梯度能够更新 GraphSAGE 的 `W` 和 `b`；
- 每次 PPO update 完成后旧 embedding cache 失效，下一个 episode 使用新参数重新计算；
- 基础 PODEM 排序结果可复现，并且 `c6288`、`s38417` 各选择前100个 fault；
- SmartATPG 实际执行的 BC epoch、curriculum stage 和 curriculum round 都是0；
- Actor/Critic 默认学习率分别严格为 `0.001` 和 `0.01`；
- 默认执行20轮，每轮恰好包含固定200个训练 fault；
- 每轮确定性评估和最佳 checkpoint 选择符合检测数、backtracks、backtrace steps、return 的字典序规则；
- TensorBoard 同时记录逐 episode 和逐 round 的 backtracks、backtrace steps 与 return；
- 断点恢复后 TensorBoard step、训练顺序和最佳轮选择保持连续；
- Linux 一键入口能够完成准备、20轮训练、best 导出和最终 benchmark；
- 全部16个 ISCAS 电路均完成 heuristic 与 `rl_best` 的完整 fault 比较；
- 每种策略每个电路执行1次预热和5次正式测量，runtime 汇总使用中位数；
- 最终 JSON、CSV 和 Markdown 正确报告 backtracks、backtrace steps、runtime 及改善百分比；
- GraphSAGE 预处理时间与 ATPG runtime 分开报告；
- 旧 SmartATPG schema 的 checkpoint 和 artifact 会被拒绝；
- 相同 snapshot 下，Python 与 C++ 的 logits 和最终选择保持一致。

实现验证不需要启动完整的大规模训练。使用一个小型确定性电路和一次短 PPO smoke test 即可。

# SmartATPG 11维 GraphSAGE 状态设计

## 范围

本次只修改 SmartATPG 后端。DeepGate 以及计划中的 DeepGate2 迁移均不在本次修改范围内。

SmartATPG 策略必须使用每个 gate 自己的 GraphSAGE embedding，不使用整张电路的全局 pooling embedding，也不进行行为克隆（BC）。

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

编码器沿 fanin 方向执行三轮均值聚合。每一轮都保持节点表示为11维，并分别拥有独立、可训练的权重矩阵和偏置：

```text
h_v^0 = x_v
m_v^k = mean(h_u^(k-1) for u in fanins(v))
h_v^k = ReLU(W_k [h_v^(k-1) || m_v^k] + b_k), k = 1, 2, 3
```

每一轮拼接后的输入为22维，输出为11维。没有 fanin 的节点使用全零的11维邻居均值。

最终 gate embedding 是 `h_v^3`，维度为11。设计中不存在整图 mean pooling、全局 graph context、64维隐藏表示或全局 context cache。

`W_1`、`W_2` 和 `W_3` 属于策略模型参数。它们随新的 SmartATPG 模型一起初始化，并通过 PPO 梯度与 Actor、Critic 一起训练。三个权重矩阵必须保存在 checkpoint 中，并参与 snapshot 身份计算。

## 策略状态与 Mask

对 gate `v` 进行 backtrace 决策时，策略状态定义为：

```text
state_v = [h_v^3 || action_mask]
```

Gate embedding 为11维。二输入 action mask 是独立的2维状态字段，因此 Actor/Critic 的 state 总维度为13。mask 同时用于屏蔽非法 action 的 logit，确保策略不能选择不可用的输入。

Mask 不属于 gate embedding，不能写入静态 gate embedding 表。Python 训练在每次决策时单独提供 mask；C++ 原生推理在选中当前 gate 的11维静态 embedding 后，再动态附加当前的两个 mask 值，然后计算策略输出。

保留现有 objective value embedding 和输出两个 logit 的 backtrace action head。强制移动继续由 PODEM 求解器控制。

## 训练

SmartATPG 无条件跳过行为克隆。每次新的 SmartATPG 训练都从随机初始化的 GraphSAGE、Actor 和 Critic 开始执行 PPO。本次改动不改变 DeepGate 的训练行为。

现有 PPO reward、GAE/MC 选择、RND 行为、curriculum fault 顺序、checkpoint 保存频率、best model 选择和断点恢复语义均保持不变。

断点恢复只接受满足以下条件的新 checkpoint：使用新的特征 schema、三层 GraphSAGE 配置、11维 gate embedding 和13维 policy state。旧 SmartATPG checkpoint 必须被拒绝。

## Artifact 与 C++ 原生推理

SmartATPG 为电路中的每个 gate 导出一行11维 `h_v^3`。Actor 元数据记录 backend、特征 schema、GraphSAGE 配置、gate embedding 维度、policy state 维度和 snapshot 身份。Actor 与 gate embedding 文件必须来自同一个 snapshot。

C++ artifact reader 必须区分11维 gate embedding 和13维 policy state。旧 SmartATPG 的80维 descriptor 及其 checkpoint 与新模型不兼容，加载时必须给出明确错误，不能静默地重新解释为新格式。

## 验证要求

测试必须覆盖：

- 11列 gate 特征的固定顺序和归一化结果；
- 输入电路含 XOR 或 XNOR 时明确拒绝；
- graph 数据和导出 artifact 中均不存在 CO；
- GraphSAGE 包含三个相互独立的 `Linear(22, 11)` 聚合层；
- 不存在整图 pooling 或共享的全电路 context；
- 三跳以内的 fanin 特征发生变化时，对应 gate embedding 会改变；
- mask 不写入 embedding 文件，但会进入13维 policy state；
- PPO 梯度能够更新全部三层 GraphSAGE 参数；
- 无论通用 BC 参数如何设置，SmartATPG 实际执行的 BC epoch 都是0；
- 旧 SmartATPG schema 的 checkpoint 和 artifact 会被拒绝；
- 相同 snapshot 下，Python 与 C++ 的 logits 和最终选择保持一致。

实现验证不需要启动完整的大规模训练。使用一个小型确定性电路和一次短 PPO smoke test 即可。

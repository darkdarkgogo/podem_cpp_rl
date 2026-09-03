# 逐 Level 双向 GAT-GRU 全面对比设计

## 目标

新增 `SmartATPG-GAT-GRU-11`，作为第二种可训练的图策略模型，与现有
11 维 SmartATPG 模型进行全面对比。现有模型继续作为基线，不会被替换。

两种模型必须使用相同的训练电路、故障样本、课程顺序、奖励函数、
PPO/RND 参数、随机种子、验证规则以及最终 16 个电路的原生基准测试协议。
编码器架构及其直接导致的维度差异是本次对比中有意保留的变量。

## 模型架构

每个 gate 继续使用现有的 11 维输入特征，并直接将其作为初始 hidden state，
不添加升维层：

```text
h0 = x
```

图编码器随后严格执行两次完整的逐 level 扫描：

1. 正向扫描从 level 1 到最大 level。对当前 level 的每个 gate，使用单头
   GAT 聚合其 fanin hidden state，再由正向 GRU cell 根据聚合消息和当前
   hidden state 更新该 gate。
2. 反向扫描从最大 level 到 level 0。对当前 level 的每个 gate，使用另一套
   单头 GAT 聚合其 fanout hidden state，再由独立的反向 GRU cell 在正向
   结果的基础上继续更新。

某个方向上没有邻居的节点保留当前 hidden state，不执行空消息更新。同一
level 内所有节点同时更新；只有当前 level 全部完成后，下一个 level 才能读取
这些新状态。因此“一次正向传播”或“一次反向传播”均表示遍历完整电路深度的
一次逐 level sweep，而不是对全部边做一次同步更新。

正向与反向使用互相独立的参数，但各自的参数在不同 level 之间共享。每个方向
包含：

- GAT 投影矩阵：数学形状为 `11 x 11`，PyTorch linear weight 形状为
  `[11, 11]`。
- Attention 向量：形状为 `[22]`，作用于变换后 target 与 source state
  的拼接结果。
- GRU input size：11。
- GRU hidden size：11。

最终 gate embedding 严格为 11 维。两位 action mask 不属于 embedding，
只在构造决策描述符时追加，因此 Actor 接收的 policy state 仍为 13 维。

## Attention 与更新语义

对于当前传播方向中的一条邻接边 `j -> i`，单头 Attention 计算如下：

```text
z_i = W h_i
z_j = W h_j
e_ij = LeakyReLU(a^T [z_i || z_j])
alpha_ij = softmax_j(e_ij)
m_i = sum_j(alpha_ij * z_j)
h_i' = GRUCell(m_i, h_i)
```

softmax 针对每个 target gate 的当前有效邻居独立归一化。正向扫描使用原始
fanin 边；反向扫描将相同的边反转，使 fanout 信息向 primary input 传播。
不额外添加 self-loop，因为 GRU 的 hidden 参数已经保留 gate 自身的旧状态。

## 软件边界

新实现与基线编码器隔离：

- `smartatpg_features.py` 继续负责确定性的电路解析、特征、拓扑和 level，
  并提供 Torch 与 portable inference 共用的反向邻接和 level 分组。
- 新建独立的 GAT-GRU 模块，负责 11 维编码器、Policy 和 PPO Agent 变体。
- 训练命令必须显式选择编码器变体，并写入独立输出目录；现有基线 checkpoint
  兼容性保持不变。
- Snapshot 导出记录编码器变体、图配置、embedding 维度、policy state 维度
  以及全部编码器张量。
- Portable Python inference 在不依赖 Torch runtime 的环境中实现相同的逐
  level 计算，并为原生 C++ 基准测试导出固定的逐 gate embedding。
- 原生 C++ Actor loader 继续使用现有的 11 维 embedding 和 13 维 policy
  state，并通过编码器变体及图配置标识防止工件混用。

新模型仍属于 SmartATPG 系列，不引入新的外部后端或 vendor 依赖。

## 工件兼容性

现有基线训练 checkpoint、V5 model 和 V3 embedding table 保持可读。新编码器
使用新的版本化模型格式和 graph configuration ID。虽然两种模型的 embedding
维度相同，也必须防止基线模型与 GAT-GRU embedding 被错误配对。

新模型与 embedding 工件必须验证：

- encoder variant 与 graph configuration；
- gate embedding 维度 11 和 policy state 维度 13；
- circuit hash、tensor 名称、tensor 形状、有限数值和 snapshot ID；
- Actor 参数与导出 embedding 的精确配对关系。

不支持的格式或混合工件必须在 ATPG 启动前失败，不能猜测默认值。写入继续采用
临时文件完成后替换的原子方式。

## 训练与对比协议

两种模型分别从随机初始化开始训练，共用同一份已准备 manifest 和所有共享
超参数。每一轮训练使用相同且确定的 episode 顺序。验证过程和最佳 checkpoint
选择顺序沿用当前规则，不作修改。

最终报告比较三种模式：

- heuristic PODEM；
- 基线 SmartATPG 11D fanin-mean GraphSAGE；
- SmartATPG-GAT-GRU-11。

每个电路及汇总结果都必须报告 detected、aborted、redundant faults、
backtracks、backtrace steps、生成向量数量以及 C++ ATPG 区间耗时。同时报告
模型参数量、训练 wall time、可获得时的训练峰值内存、图预处理时间和 embedding
导出时间。与现有基准定义一致，embedding 生成时间不计入 C++ ATPG 区间耗时。

## 错误处理

图加载器继续拒绝环、缺失 driver、不支持的 gate type 和非法 level schedule。
GAT 实现拒绝应当更新却没有有效邻居的 attention group、非有限 attention
score 或 hidden state、错误维度及非法 action mask。Portable 与 Torch
实现遇到 metadata 或 tensor shape 不一致时必须直接失败，不允许使用猜测默认值。

## 测试与验收标准

单元测试必须验证不存在输入升维、正反向参数互相独立、attention 归一化、
同 level 同步更新、正向和反向信息流、11 维 embedding 以及 13 维决策描述符。
梯度测试必须证明梯度能够到达两个方向的 GAT、两个 GRU cell 和 Actor。

一致性测试需要在明确的浮点误差范围内对齐 Torch 与 portable embedding，
并对齐 portable 与原生 Actor logits。原生工件测试同时覆盖基线格式和新格式，
包括拒绝交叉配对的 model 与 embedding。

端到端验收要求两种模型能够基于同一 manifest 分别训练和导出，三方基准测试
能够在配置的完整电路集合上结束，所有现有基线测试继续通过，并且不重新引入
已删除的外部图编码器代码或依赖。

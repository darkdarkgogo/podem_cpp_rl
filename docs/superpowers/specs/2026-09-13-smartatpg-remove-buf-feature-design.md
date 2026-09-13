# SmartATPG 删除 BUF 特征设计

## 背景

当前 SmartATPG 节点特征包含七类门的 one-hot 编码：`PI`、`AND`、
`NAND`、`OR`、`NOR`、`NOT` 和 `BUF`，再拼接 level、fanout、CC0、
CC1、CO 五项结构特征，总维度为 12。

目标数据与训练流程不包含显式 `BUF` 逻辑门。C++ PODEM 中的 fanout stem
由一条 wire 的多个消费节点表示，不依赖 `BUF` 类型。因此应从 SmartATPG
特征契约中删除 `BUF`，同时保留 C++ 对 fanout stem 的现有处理。

## 目标

- SmartATPG gate-type one-hot 只保留 `PI`、`AND`、`NAND`、`OR`、
  `NOR`、`NOT` 六类。
- 保留 level、fanout、CC0、CC1、CO 五项结构特征。
- Mean 和 GAT-GRU gate embedding 均改为 11 维。
- 不保留任何旧 SmartATPG 模型格式兼容性。
- 显式包含 `BUF(...)` 的 BENCH 输入必须立即失败，并给出明确错误。
- 不改变 C++ PODEM 的 fanout stem、故障建模或搜索行为。

## 非目标

- 不删除 C++ PODEM 内部的 `BUF` 枚举或相关仿真逻辑。
- 不改变 fanout stem 的表示方式。
- 不自动折叠或旁路 BENCH 中的显式 `BUF` 节点。
- 不转换或迁移旧 checkpoint、导出模型或 embedding artifact。
- 不改变 PPO、RND、奖励或训练协议。

## 新特征契约

每个 gate 的原始特征顺序固定为：

```text
[PI, AND, NAND, OR, NOR, NOT, level, fanout, CC0, CC1, CO]
```

维度计算如下：

```text
6 类 gate-type one-hot + 5 项结构特征 = 11 维
```

level、fanout 与 SCOAP 特征的计算和归一化保持不变。由于 one-hot 列数减少，
后续结构特征的列索引整体前移一位。

特征 schema、graph config ID、checkpoint format 和导出模型 format 必须升级为
新的唯一标识。新标识需要明确区分于此前的两种模型：

- 旧 11 维：包含 `BUF`，但没有 CO；
- 当前 12 维：包含 `BUF` 和 CO。

加载器只能接受新的 11 维含 CO、无 `BUF` 格式。上述旧格式必须被拒绝，不能
通过维度相同而误判为兼容。

## 编码器与策略维度

### Fan-in mean

Mean encoder 将当前 gate 的 11 维特征与 fan-in 平均 11 维特征拼接为 22 维，
再通过 `Linear(22, 11)` 和 ReLU，输出 11 维 gate embedding。

Mean Actor 直接接收 11 维 gate embedding。逻辑 decision-state 元数据仍额外记录
2 维 action mask，因此为 13 维；action mask 的现有调度语义不变。

### Level-wise GAT-GRU

GAT-GRU 的初始 hidden state、attention projection、message 和 GRU hidden state
均为 11 维。前向 level sweep 聚合 fan-in，反向 level sweep 聚合 fan-out，最终
输出 11 维 gate embedding。

GAT Actor 输入由 11 维 gate embedding 与 1 维 objective value 拼接而成，共
12 维。逻辑 decision-state 元数据再加 2 维 action mask，因此为 14 维。

Actor/Critic 的 32 单元隐藏层保持不变；它不是 gate embedding 维度。

## 输入验证

Python graph loader 与 portable loader 的支持门类型必须一致。读取显式
`BUF(...)` 或 `BUFF(...)` 时，加载器应报出包含 gate 名称或行号的错误，说明新
SmartATPG schema 不支持 BUF。不得将 BUF 编成全零 one-hot，也不得将其误当作
NOT 或 PI。

数据生成与转换工具如果位于 SmartATPG 训练数据路径中，也不得生成显式 BUF。
需要通过保持导线别名、直接连接或在上游规范化阶段消除 BUF；不得为了适配特征
schema 而改变电路逻辑。

## 模型格式与兼容策略

这是一次有意的不兼容变更：

- 删除旧 11 维无 CO 模型的读取分支、常量和测试；
- 删除当前 12 维含 BUF 模型的读取分支、常量和测试；
- Python checkpoint、portable 模型和 C++ native loader 只接受新格式；
- 已训练的 Mean 与 GAT-GRU 权重全部失效，必须重新训练并重新导出；
- 格式错误必须在加载阶段失败，不能等到矩阵运算时才暴露维度不匹配。

训练清单和 benchmark 清单也必须携带新的 schema、graph config 和维度元数据，
从而阻止旧 artifact 混入新实验。

## 受影响组件

- `python/rl_podem/smartatpg_features.py`：门类型、特征维度、列索引和 schema。
- `python/rl_podem/smartatpg.py`：Mean encoder、RND observation 与维度校验。
- `python/rl_podem/gat_gru.py`：GAT-GRU 和 Actor 输入维度。
- `python/rl_podem/backends.py`、`smartatpg_artifacts.py`、`cpp_bridge.py`：
  元数据、格式和导出校验。
- `scripts/smartatpg_portable.py`：特征生成、模型读取和推理维度。
- 训练、准备和 benchmark 脚本：格式名称、manifest 校验和输出目录默认值。
- `src/rl_policy.cpp`：native 模型格式、schema、graph config 和 tensor shape 校验。
- SmartATPG、legacy、SCOAP、portable、native parity 相关测试与 fixtures。

C++ PODEM 的门级仿真、fault list 和 fanout stem 代码不在修改范围内。

## 测试与验收

实现完成后必须满足以下条件：

1. 特征顺序严格等于设计中的 11 项，所有输出为有限 `float32`。
2. Mean encoder 输出形状为 `[node_count, 11]`，权重形状为 `[11, 22]`。
3. GAT-GRU 输出形状为 `[node_count, 11]`，双向 level sweep 行为保持不变。
4. Mean Actor 输入为 11 维；GAT Actor 输入为 12 维。
5. Python、portable 和 native 三条路径对同一新模型产生一致动作结果。
6. 显式 `BUF`/`BUFF` BENCH 在 Python 与 portable 路径均被明确拒绝。
7. 所有旧 11 维无 CO 和旧 12 维模型均在加载阶段被拒绝。
8. fanout 大于 1 且不含显式 BUF 的电路仍能正常提取特征、训练和推理。
9. 训练准备、checkpoint resume、导出和 benchmark 的新格式链路通过端到端测试。

## 迁移结果

合入后，仓库只存在一种受支持的 SmartATPG 模型契约：11 维 gate embedding、
包含 CO、不包含 BUF one-hot。所有实验必须从新 schema 重新生成 manifest、训练
checkpoint 与部署模型。

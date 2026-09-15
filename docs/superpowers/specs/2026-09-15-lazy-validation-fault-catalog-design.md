# Validation 故障目录延迟生成设计

## 目标

数据准备阶段不再对 validation 电路运行启发式 PODEM profile。
Validation 只需要稳定的故障目录，不会使用 profile 中的检测结果、回溯次数或
反向追踪步数来筛选故障。

训练电路的数据准备逻辑保持不变：对每个训练电路执行 profile，然后选取最难检测的
30 个可检测故障用于训练。

## Manifest 兼容性

为新准备的数据集引入新的 manifest 格式版本。新格式保持训练记录不变，validation
记录只保留电路标识和完整性元数据，不再包含 validation profile 产物或预计算的
episode 故障列表。

训练入口同时接受新旧两种格式：

- 现有 V6 manifest 继续使用其保存的 validation 故障顺序，以保证现有
  checkpoint 和部分完成的 validation 记录可以继续恢复。
- 新格式 manifest 在训练进程加载数据集时，为每个 validation 电路现场生成
  故障目录。

不删除已有的 validation profile 文件。新的数据准备过程不再创建这些文件，
新格式的运行时也不读取它们。

## 运行时数据流

对于新格式 manifest，训练进程加载每个 validation 电路，并调用
`catalog_cpp_podem()`。该接口只执行电路初始化和 `generate_fault_list()`，
不调用 `atpg.test()`。

返回的故障 ID 构成 validation episode 顺序。Validation 仍然逐个故障运行，从而保留
逐故障指标和现有的安全断点续跑能力。

程序检查现场生成的目录是否为空、是否存在缺失的故障 ID，以及是否包含重复 ID。
故障目录通过 manifest 中的电路产物哈希绑定到特定电路。与当前行为一样，恢复
validation 时，已保存的记录必须是现场生成顺序的严格前缀。

## 数据准备和校验规则

- 只对 train 分区的电路执行 profile。
- 保持原有训练 top-30 选择逻辑不变。
- 新格式 validation 记录不要求 `profile`、`profiled_faults`、
  `episode_faults` 或 `episode_fault_ids` 字段。
- 新格式的 validation episode 数量由运行时生成的目录推导，不持久化 profile
  结果。
- 加载 V6 manifest 时继续校验所有原有字段和产物。

## 错误处理

如果 validation 电路无法生成故障目录、生成了无效或重复的故障 ID，或电路与
manifest 中的产物哈希不一致，则立即停止加载并输出明确错误。如果已保存的
validation 记录与现场目录的前缀不匹配，则停止恢复。

## 测试

测试将验证：

1. 新数据准备过程不会对 validation 电路调用 PODEM profile。
2. 新 validation 记录不包含 profile 专用字段。
3. 运行时生成的故障目录会将每个 validation 故障交给现有的逐故障评估器。
4. 现有 V6 manifest 和 validation 恢复状态仍可加载。
5. 空目录、重复故障 ID 和电路变更会产生明确的校验错误。
6. 训练 profile 和 top-30 选择逻辑保持不变。

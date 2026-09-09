# 展开 XOR 的输入端故障保留设计

## 目标

让展开 XOR 的 fault map 遵循原来的折叠线故障规则。当 XOR 输入信号的
上级 wire 还有其他逻辑 fanout 时，保留该 XOR 输入分支的 `GI-SA0` 和
`GI-SA1` 故障。当 XOR 是这条 wire 唯一的逻辑负载时，输入端故障继续与
上级 gate output fault 折叠，不单独保留。

本次修改不在 BENCH 电路中增加 gate 或 wire。现有的五门 NAND/NOT XOR
展开结构保持不变。

## 逻辑 fanout 的判定

fanout 按逻辑单元的边界计算。同一个 XOR 输入在 NAND/NOT 展开结构中形成的
两条物理连接，合并视为一个 XOR 逻辑负载。这个展开 XOR 以外的所有消费者，
分别视为其他逻辑负载。

对于 XOR 的一个逻辑输入 `a`：

- 如果 XOR 是 `a` 唯一的逻辑消费者，不新增 XOR 输入端故障；
- 如果 `a` 还连接到其他 gate 或 output 分支，保留 `a` 到该 XOR 分支上的
  `GI-SA0` 和 `GI-SA1` 故障。

因此，上级 wire 的 stem fault 和 XOR 输入 branch fault 在有 fanout 时不会
被错误地视为等价故障。

## fault map 的处理

转换器继续从转换前的 BENCH 生成 collapsed fault catalog。对于每个被识别的
展开 XOR，转换器检查两个逻辑输入各自的上级 wire 是否还有其他逻辑 fanout。
如果存在 fanout，就保留原有 fault catalog 中该 XOR 输入分支的两个 stuck-at
故障；否则保持原来的等价折叠结果。

这些故障的外部 ID 使用 XOR 的逻辑输出、输入位置和 stuck-at 类型，确保它们
与上级 wire 的 GO fault 可以明确区分。fault map 会把它们映射回用于 PODEM
执行的展开网表位置。

只属于 `W`、`Z`、`X`、`Y` 私有展开节点的内部实现故障仍然删除。每个 XOR
输出仍然只保留一个 `GO-SA0` 和一个 `GO-SA1`，两者的等价故障权重均为 1。
最终 fault map 的 collapsed fault 数和 uncollapsed fault 总数根据保留后的
记录重新计算，并由 C++ fault-map loader 校验。

## 验证方式

使用小型展开 XOR 网表验证以下情况：

1. 输入 wire 没有其他逻辑消费者时，不生成单独的 XOR `GI` 故障；
2. 输入 wire 还有其他逻辑消费者时，同时保留该 XOR 分支的 `GI-SA0` 和
   `GI-SA1`，并继续保留上级 wire 的 `GO` 故障；
3. XOR 的两个输入分别独立判断 fanout；
4. 与 XOR 输入分支无关的私有展开故障不会残留；
5. XOR 输出的 `GO-SA0` 和 `GO-SA1` 处理保持不变；
6. 生成的 fault map 能由 C++ 正常加载，collapsed 与 uncollapsed fault 数量
   和文件头记录一致。


# 每电路启发式 Top-30 难检测 Fault 选择设计

## 目标与范围

将 SmartATPG 训练集的 fault 选择规则由“保留每个训练电路中所有启发式可检测 fault”改为“每个训练电路最多保留 30 个最难检测的启发式可检测 fault”。难度依据现有 SCOAP 启发式 PODEM profiling 的实际搜索代价确定。

本次改动只影响训练集 episode fault 清单。验证集仍保留每个电路的完整折叠 fault catalog，以保证验证覆盖率和最佳模型选择不因预先筛选而产生偏差。C++ PODEM、图特征、GAT-GRU、PPO 更新、五轮训练协议、200 次 backtrack 上限和 benchmark 流程均不改变。

## Fault 资格与难度排序

每个训练电路继续使用 SCOAP 启发式 PODEM，在 `backtrack_limit=200` 下 profile 完整折叠 fault catalog。

只有 profile 中 `outcome == 1` 的 fault 有资格进入训练集。`outcome == 0` 或 `outcome == 2` 的 fault 不用于补足数量，因为它们没有可比较的成功检测搜索代价，也可能是不可测 fault 或在限制内未解决的 fault。

对符合资格的 fault 使用以下稳定排序键：

1. `backtracks` 降序；
2. `backtrace_steps` 降序；
3. `fault_id` 的字符串值升序。

排序后的前 30 个 fault 构成该电路的训练 episode fault 清单。此规则直接使用启发式算法的实际搜索工作量：更多 backtrack 表示更难完成决策搜索，backtrack 相同时用更多 backtrace step 区分剩余难度，最后用 fault ID 消除并列项的不确定性。

选择过程不得依赖 profile 原始返回顺序。相同 profile、seed 和程序版本必须生成完全相同的 fault ID 顺序。

## 数量不足与错误处理

如果某个训练电路只有 1 到 29 个 `outcome == 1` fault，则全部保留并继续 preparation，不重复 fault，也不使用未检出 fault 补足到 30 个。

如果一个训练电路没有任何 `outcome == 1` fault，则维持当前行为并明确报错。错误信息应包含数据划分和电路名称，使 1024 个训练电路的 preparation 问题可以直接定位。

验证电路无论启发式 outcome 如何，仍保留完整 profile catalog，且原始 catalog 顺序不变。

## 实现边界与工件协议

在 preparation 脚本中定义命名常量 `TRAIN_FAULTS_PER_CIRCUIT = 30`，训练 fault 选择函数负责过滤、稳定排序和截取。manifest 中的每条训练电路记录继续同时保存完整 profile 的路径和哈希，以及筛选后的 `episode_faults` 与 `episode_fault_ids`。`profiled_faults` 仍表示完整 profile 数量，而不是选中数量。

更新 manifest 格式标识和 fault filter 标识，使新协议明确包含“启发式已检测、按实际搜索代价排序、每电路最多 30 个”的语义。旧 manifest 必须作为配置不兼容被拒绝，避免恢复旧的全量 fault preparation。preparation resume 和训练入口沿用现有校验路径，从原始 profile 重新计算 Top-30 清单，并与 manifest 中的 fault 记录和 ID 严格比较。

训练 episode 总数继续由所有训练电路实际选中的 fault 数量求和，因此其上限为 `训练电路数 × 30`，但允许因小电路的可检测 fault 不足而低于该上限。轮内确定性打乱、逐 episode checkpoint 与精确恢复语义不改变。

## 数据流

1. 对电路的完整折叠 fault catalog 执行启发式 profiling。
2. 校验 fault ID、outcome、backtracks 和 backtrace steps。
3. 训练集过滤出 `outcome == 1` 的记录。
4. 按固定难度键排序并截取最多 30 条。
5. 将完整 profile 工件及筛选后的训练清单写入 manifest。
6. 训练器只遍历筛选后的 fault；验证器仍遍历完整 catalog。

## 测试与验收

单元测试必须覆盖：

- 超过 30 个可检测 fault 时只选择前 30 个；
- `backtracks` 为首要降序键；
- `backtracks` 相同时由 `backtrace_steps` 降序决定；
- 两项计数都相同时由 `fault_id` 升序稳定决定；
- `outcome != 1` 的 fault 即使计数很高也被排除；
- 只有 1 到 29 个可检测 fault 时全部保留；
- 没有可检测 fault 时明确失败；
- 验证 fault 选择仍返回完整 catalog 且不改变顺序；
- manifest 校验接受由 profile 重算得到的 Top-30 清单，拒绝旧格式、错误顺序或被篡改的选择；
- training episode count 使用每个电路实际选择数求和，而不是假设固定为 30。

普通测试不要求重新 profile 完整的 1024 个训练电路。生产验收是在新的 preparation 输出目录运行 profiling，确认每个训练电路的 `episode_fault_ids` 不超过 30、全部对应 `outcome == 1`，并符合固定排序规则；6 个验证电路的 fault 数量和顺序应保持完整 catalog 语义。

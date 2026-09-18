# GAT Backtrack 奖励设计

## 范围

将 `level_gat_gru` SmartATPG 的奖励改为 `SmartATPG_Backtrack_Reward_Design_CN.docx` 定义的有界非线性 backtrack 奖励。`fanin_mean` 保留原奖励行为。两个 encoder 的正式 SmartATPG backtrack 上限均从 200 改为 100。

不增加奖励变体、实验开关、消融运行、训练运行或 benchmark 运行。验证仅包括必要的定向单元测试、静态检查以及用于确认实现正确性的原生扩展构建和测试。

## 奖励协议

实现提供两种具名奖励协议：

- `level_gat_gru` 使用 `cubic_backtrack_v1`。
- `fanin_mean` 使用 `legacy_pi_exponential`。

奖励协议由 encoder 决定。调用方不得选择与 encoder 元数据冲突的奖励协议。

两种协议均保留现有的 RL 相关 backtrace-step 奖励 `-0.1`，以及 detected fault 的 terminal reward `+100` 和其他结果的 terminal reward `-100`。

### GAT 奖励

GAT 的每个 episode 均从 backtrack 计数 0 开始。第 `B` 次与策略相关的 backtrack 满足 `1 <= B <= 100`，其奖励为：

```text
r_bt(B) = -(0.5 + 9.802960494 * (B / 100.0)^3)
```

该奖励分配给 backtrack 事件所指向的 decision step。`pi_not_done` 仍是有效事件，但不再贡献奖励。backtrack 计数超过 100 视为协议错误。

前 100 次 backtrack penalty 的总和必须在 `1e-6` 误差内等于 `-300`。

### Mean 奖励

mean 的 backtrack 事件继续不贡献奖励。`pi_not_done` 继续使用现有奖励：

```text
10.0 - 7.5 * exp(0.07 * (backtracks + pi_visits))
```

不对 mean 奖励进行截断、替换或其他数值修改。mean 唯一的行为协议变化是 backtrack 上限从 200 改为 100。现有有限数检查继续保持严格，遇到非有限的 mean validation return 时仍应明确失败。

## 共享实现边界

Python trainer 根据 agent 的 encoder variant 获取奖励协议。事件处理器只应用 GAT backtrack 奖励或 mean 旧 PI 奖励中的一种，不能同时应用两种奖励。

C++ 原生 validation 显式接收与 encoder 兼容的奖励协议，并复现相同的 Python 事件计分逻辑。原生日志和返回的 per-fault record 在持久化前必须拒绝非有限 return。

Python 模型 validation 和 SCOAP validation 使用与被评估模型相同的奖励实现和事件语义。共享 helper 集中保存常数和奖励计算，避免 Python training 与 Python validation 发生偏差。

## 协议身份与兼容性

正式 SmartATPG 的 `backtrack_limit` 在训练 manifest 准备、Linux 启动器、checkpoint 配置、validation identity、模型导出元数据、portable/native 模型加载、benchmark 准备和 comparison validation 中统一为 100。

Checkpoint、validation 和导出模型元数据记录奖励协议。恢复运行和加载工件时，奖励协议必须与 encoder 匹配。backtrack 上限为 200 的现有工件与新协议不兼容，必须报告明确的 identity 或 metadata 错误，不能静默复用。

## 指标与比较

Per-fault return、`return_total` 和 `return_mean` 在写入前必须是有限数。保留 comparison 现有的有限数检查。

Return 继续用于各模型自身的 validation metrics 和最佳轮次评分。由于 GAT 与 mean 使用不同的奖励定义，跨模型及模型与 SCOAP 的报告行不得对 return 做相减或其他直接比较。故障覆盖率、backtracks、backtrace steps 和 ATPG runtime 仍可直接比较。

奖励组成日志分别记录 backtrace-step penalty、GAT backtrack penalty、mean 旧 PI reward 和 terminal reward，但不改变 PPO combined reward 的计算方式，也不改变现有 RND 系数。

## 验证

定向测试覆盖：

- GAT 公式在代表性计数上的精确结果，以及 100 次累计 penalty 为 `-300 +/- 1e-6`。
- GAT 严格拒绝 1 至 100 范围外的 backtrack 计数。
- GAT 事件处理对 backtrack 计分，且 `pi_not_done` 不贡献奖励。
- mean 事件处理忽略 backtrack，并保留旧 `pi_not_done` 奖励。
- 对等事件轨迹下，Python training、Python validation 和 C++ 原生 validation 在相同协议中产生一致的 extrinsic return。
- 所有正式协议校验均要求 backtrack 上限为 100，并要求 encoder 与奖励协议元数据正确匹配。
- Validation 持久化拒绝非有限的 per-fault 和汇总 return。
- Comparison 输出不计算不同奖励协议之间的 return 差值。

验证范围不包括任何训练、benchmark、奖励对比实验或消融实验。

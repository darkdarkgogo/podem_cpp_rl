# SmartATPG PI Visit 语义修复与 Non-finite 诊断设计

## 范围与原则

本次修改只覆盖 MEAN/SmartATPG 兼容路径的 PI visit 语义和 native validation 数值诊断：

- PI visit 只统计 Agent 正常 backtrace 到达 PI/PPI，以及 backtrack 时翻转 PI/PPI。
- initial mandatory backward implication 到达 PI 时不计 visit。
- MEAN 继续使用 `10 - 7.5 * exp(0.07 * (B + P))`。
- GAT 的 cubic reward、backtrack 定义、backtrack 最大值 100、PODEM 搜索控制流和 fault selection 全部保持不变。
- 不做 reward clip、normalize 或 exponent clamp。

附件 PDF 是设计参考；实际修改范围以用户要求和当前仓库代码为准。

## 方案比较

### 方案 A：只修 PI visit 计数

删除 mandatory implication 的计数，保留其余逻辑。改动最小，但如果 validation 仍产生 non-finite，只能继续依靠通用异常猜测数值来源。

### 方案 B：修 PI visit，并在溢出点增加诊断（采用）

在方案 A 基础上，于 native validation 的指数 reward 计算点记录 fault、sequence、B、P、B+P、exponent、step reward 和累计 reward。它不改变 reward，只让下一次异常可直接定位。

### 方案 C：修计数并截断指数

可以规避部分溢出，但会改变原 SmartATPG reward 语义，掩盖真实 B/P，因此本次不采用。

## PI Visit 修复

每个 fault 开始时，现有 `rl_pi_visits` 和 `rl_pending_pi_assignments` 清零逻辑保持不变。

在 `PODEM/src/podem.cpp` 中：

- 删除 `ATPG::backward_imply()` 到达输入线时对 `rl_pending_pi_assignments` 的增加。该路径服务于 initial mandatory implication，不代表 Agent visit。
- 保留 `ATPG::find_pi_assignment()` 真正到达 PI 时的增加。
- 保留普通 backtrack 和 multiple-pattern 分支两处 PI flip 的增加。

`ATPG::notify_pi_result()` 保持不变：仍把 pending 数累计到 fault 内的 `rl_pi_visits`，清空 pending，并将累计 P 传给训练/验证策略。

## Native Validation 诊断

在 `PODEM/src/python_bindings.cpp` 的 `NativeValidationPolicy::on_pi_not_done()` 中，只对 `legacy_pi_exponential` 且 sequence 属于当前策略决策的事件执行：

1. 分解计算 `b`、`p`、`b_plus_p`、`exponent`、`reward_before`、`step_reward` 和 `reward_after`。
2. 当 `exponent >= 600` 时向 `stderr` 输出 `SMARTATPG_REWARD_WARN`，随后仍按原公式继续计算。
3. 当 `step_reward` 或 `reward_after` 非有限时，输出 `SMARTATPG_NONFINITE`，刷新 `stderr`，并立即抛出带有日志提示的异常。
4. 数值正常时把 `reward_after` 写回累计 reward。

日志包含当前 fault ID、sequence、B、P、B+P、exponent、reward_before、step_reward；non-finite 日志还包含 reward_after。原有 `on_episode_end()` 的 finite 检查继续保留为第二道保护。

## 测试与验收

采用定向验证，不运行完整训练或 benchmark：

- 静态/单元测试确认 mandatory `backward_imply()` 不再增加 visit，而正常 PI 到达与两处 PI flip 仍增加。
- native validation 正常小 B/P 时 reward 与修改前公式完全一致，且不打印 warning/error。
- 构造 exponent 大于等于 600 时打印 `SMARTATPG_REWARD_WARN`，不改变计算路径。
- 构造指数溢出时打印字段完整的 `SMARTATPG_NONFINITE` 并抛出异常。
- 现有 episode 末尾 non-finite 检查、MEAN reward scheme、GAT reward scheme和 backtrack limit 100 均保留。
- 重编译 `cpp_podem`，随后在 `d2l` 环境运行一个小 circuit/fault smoke test。

## 提交拆分

按设计参考拆成两个实现提交：

1. `fix: align SmartATPG PI visit semantics`
2. `chore: add SmartATPG nonfinite reward diagnostics`

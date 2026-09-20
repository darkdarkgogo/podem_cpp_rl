# SmartATPG PI Reward 事件边界最终修复设计

## 问题与根因

上一次修复已经使 `ATPG::backward_imply()` 的 initial mandatory implication 不再增加 PI visit。初始化路径仍会调用 `notify_pi_result(find_test)`，因此在尚未发生任何 RL decision 时发送 `pi_not_done`：

- `last_policy_decision_sequence == 0`
- `rl_pi_visits == 0`
- Python trainer 中不存在可归因的 PPO step

当前 Python handler 在检查 `step_idx` 前先调用 `smartatpg_pi_reward()`，`pi_visits == 0` 因而触发 `ValueError: SmartATPG reward counters are out of range.`。

初始化逻辑属于 PODEM，不属于 RL transition。没有 RL action 时既不应计 SmartATPG PI visit，也不应产生 PI-not-done reward。

## 采用方案

采用事件源修复与 Python 防御相结合的最小方案：

1. 删除 initial mandatory implication 完成后的 `notify_pi_result(find_test)`。
2. 保留正常 PODEM objective/backtrace 到 PI 后的 `notify_pi_result(detected_this_assignment)`。
3. Python trainer 只有找到有效 `step_idx` 后才计算 `smartatpg_pi_reward()`。

不采用以下方案：

- 不把 `smartatpg_pi_reward()` 改成允许 `pi_visits == 0`，因为此时没有可归因的 RL PI event。
- 不只在 `notify_pi_result()` 内用 `sequence == 0` 过滤，因为初始化路径本身就不应调用 RL reward 通知。

## C++ 事件边界

在 `PODEM/src/podem.cpp` 中：

- 保持每个 fault 开始时 `rl_pi_visits = 0` 和 `rl_pending_pi_assignments = 0`。
- 保持 `backward_imply()` 到 PI 时不增加 pending visit。
- 从 `set_uniquely_implied_value(fault)` 的初始化 `case TRUE` 中删除 `notify_pi_result(find_test)`。
- 保留主循环正常 `wpi` 路径中的 `notify_pi_result(detected_this_assignment)`。
- 保留 `find_pi_assignment()` 到达 PI，以及普通和 multiple-pattern 两处 PI flip 的 pending visit 增量。

若 mandatory implication 已直接检测 fault，episode 仍通过现有 episode-end 事件结算 terminal reward，但不会产生无对应 RL step 的 PI reward。

## Python 防御

在 `PODEM/python/rl_podem/cpp_bridge.py` 的 `pi_not_done` handler 中：

1. 非 `legacy_pi_exponential` 直接返回。
2. 通过 `decision_sequence` 查找 `step_idx`。
3. `step_idx is None` 或不在当前 buffer 范围内时直接返回。
4. 只有有效 step 存在时才调用 `smartatpg_pi_reward()`，再更新该 step、episode reward 和 legacy PI reward 汇总。

`PODEM/scripts/train_smartatpg.py` 的 validation 路径已经按 decision sequence 过滤事件；C++ native validation 也只处理 `decision_sequences_` 中的 sequence，因此本次不做重复修改。

## 保持不变

- MEAN reward：`10 - 7.5 * exp(0.07 * (B + P))`。
- `smartatpg_pi_reward()` 继续要求 `pi_visits > 0`。
- GAT cubic reward、PODEM backtrack 定义和 fault selection 不变。
- 正式 SmartATPG `backtrack_limit` 继续为 100。
- 现有 `SMARTATPG_REWARD_WARN`、`SMARTATPG_NONFINITE` 和 episode-end finite 检查全部保留，不替换为附件中较弱的只打印实现。
- 不增加可选 episode 统计日志。

## 必要验证

只执行与本次问题直接相关的检查：

- C++ 定向测试确认初始化路径不会发送 PI reward 通知，正常 `wpi` 路径仍保留通知。
- Python 单测确认未知 sequence 或越界 step 的 `pi_not_done` 不调用 reward，也不改变 episode reward。
- 现有有效 sequence 的 MEAN reward 测试继续通过，确认公式与 credit assignment 不变。
- 在 `d2l` 环境重编译 `cpp_podem`。
- 使用少量 fault 做一次 Mean smoke，`backtrack_limit=100`，确认不再出现 `pi_visits=0` 的参数错误。

不运行完整双模型训练、完整 validation 或大范围回归。

## 验收标准

- 初始化 mandatory implication 不计 PI visit，也不触发 PI-not-done reward。
- 正常 RL-controlled backtrace 到 PI 仍产生 PI reward。
- 无 PPO step 的 PI event 被安全忽略，且不会先计算 reward。
- reward 公式、alpha、beta、GAT 路径和 backtrack 100 均未改变。
- 当前 non-finite 诊断继续存在。

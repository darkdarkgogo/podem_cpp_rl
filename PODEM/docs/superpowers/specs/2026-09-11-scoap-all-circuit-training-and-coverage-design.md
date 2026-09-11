# SCOAP 启发式、全电路训练与覆盖率口径设计

## 目标

将当前基于拓扑层级的 SAF PODEM 启发式改为真正使用 SCOAP 可控性与可观测性；将训练集扩大到全部 16 个正式评测电路，每个电路选择 50 个基础启发式能够检出的最困难故障；只训练 `level_gat_gru` 8 轮，并对普通训练后仍未成功的故障强化最多 5 轮。最终评测只比较 SCOAP heuristic 与 `level_gat_gru`，覆盖率把冗余故障按其非折叠等价权重计为成功，覆盖率图显示 98%–100%。

## 范围

正式电路集合与当前 benchmark 清单一致：

`c432`、`c499`、`c1355`、`c1908`、`c2670`、`c3540`、`c5315`、`c6288`、`c7552`、`s5378`、`s9234`、`s13207`、`s15850`、`s35932`、`s38417`、`s38584`。

本次不训练 `fanin_mean`，也不在最终报告和图中展示它。本次不重写 X-path 搜索，不引入动态 SCOAP，也不改变 PPO、RND、奖励函数和网络结构。

## SCOAP heuristic

### 计算

SAF PODEM 在开始故障搜索前为当前二值化或扫描转换后的组合网表计算静态 SCOAP：

- `CC0(w)`：把线 `w` 控制为 0 的代价；
- `CC1(w)`：把线 `w` 控制为 1 的代价；
- `CO(w)`：从线 `w` 向主输出传播并观察故障效应的代价。

复用项目现有 SCOAP 存储和计算逻辑，保证训练准备、原生 heuristic benchmark 与 Python 图特征使用同一套定义。无法到达输出的值使用现有上限或哨兵规则，不参与优先选择。

### 回溯选择

保留现有 PODEM 对“容易控制”和“困难控制”的调用语义，但不再按正向 level 或输入顺序判断：

- 递归目标输入值为 0 时使用该输入线的 `CC0`；
- 递归目标输入值为 1 时使用该输入线的 `CC1`；
- `find_easiest_control` 选择代价最小的未知输入；
- `find_hardest_control` 选择代价最大的未知输入；
- 代价相同时按原输入顺序选择，保证结果确定。

### 传播选择

D-frontier 出现多个可用传播门时，SCOAP heuristic 选择输出线 `CO` 最小的候选门。代价相同则按已有稳定网表顺序选择，保证相同 seed 和输入生成确定结果。现有 X-path 存在性检查继续作为候选过滤条件。

命令行 `-scoap` 必须实际启用上述 SAF 选择规则。正式训练准备和 benchmark 都显式传入该选项，日志与模型名统一使用 `scoap_heuristic`，避免把旧 level heuristic 误标为 SCOAP。

## 训练数据

对 16 个正式电路逐一运行基础 SCOAP PODEM，统一使用转换后的正式网表、原始 fault map、seed 14 和 backtrack limit 2000。只保留成功检出的代表故障，不使用冗余或 aborted 故障补足数量。

每个电路按以下稳定键排序并选择前 50 个：

1. `backtracks` 降序；
2. `backtrace_steps` 降序；
3. `fault_id` 升序。

最终 manifest 必须精确包含 16×50＝800 个唯一故障。任何电路可检出的故障不足 50 个时立即失败并报告电路名和实际数量。

## 训练流程

只创建一个 `level_gat_gru` 训练进程，默认使用物理 GPU 0：

1. 普通训练 8 轮，每轮遍历共享 manifest 中全部 800 个 episode；
2. 每轮使用确定性的 episode 洗牌顺序并评估全部 800 个故障；
3. 按现有全量验证评分保存最佳 checkpoint；
4. 从最佳普通训练 checkpoint 找出仍未成功的故障；
5. 对当前未成功集合强化最多 5 轮，每轮每个故障训练一次；
6. 每个强化轮结束后重新评估全部 800 个故障，只有全量结果更优时才更新最佳模型；
7. 若全部故障成功则提前结束强化。

训练仍使用现有 PPO、RND、奖励、12 维 SCOAP 特征、目标值拼接、Actor/Critic 和 checkpoint 恢复机制。新的 manifest、训练 checkpoint 和最佳模型使用新格式版本，并验证电路顺序、每电路故障数、普通轮数、强化轮数与 SCOAP heuristic 标识，防止误恢复旧的两电路、每电路 100 故障、20 轮状态。

## 覆盖率定义

正式覆盖率统一使用非折叠故障单位。每个折叠代表故障按 `eqv_fault_num` 展开：

`successful_uncollapsed = detected_uncollapsed + redundant_uncollapsed`

`fault_coverage = successful_uncollapsed / total_uncollapsed`

其中：

- `detected_uncollapsed` 为已检测代表故障的等价权重之和；
- `redundant_uncollapsed` 为标记 `REDUNDANT` 的代表故障等价权重之和；
- `aborted` 不计成功；
- 覆盖率必须位于 `[0, 1]`，否则 benchmark 立即报错。

C++ 结果新增非折叠冗余故障数，并保留折叠检测数、折叠冗余数和 aborted 数用于诊断。CSV 明确记录 `successful_faults` 与 `redundant_uncollapsed`；Markdown 汇总表显示“Successful / total”，避免把成功数误写成单纯 detected。

## 最终比较和绘图

最终 benchmark 只运行：

- `scoap_heuristic`；
- `smartatpg_gat_gru`。

回溯次数、回溯步数和原生 ATPG 时间以 SCOAP heuristic 为基准，图中绘制 `smartatpg_gat_gru / scoap_heuristic`，并以横线表示基准值 1。覆盖率图绘制两种方法的绝对成功覆盖率，纵轴固定为 98%–100%。低于 98% 的值允许存在并在日志中明确告警；图仍按用户指定范围裁剪，不静默修改数据。

## 错误处理与兼容性

- 拒绝缺失任一正式电路、每电路不是 50 个故障或包含非成功基线故障的 manifest；
- 拒绝从旧训练格式恢复，以免训练集合或轮数不一致；
- benchmark 缺少非折叠冗余计数时明确失败，不把折叠冗余数直接加入未折叠 detected；
- 最终绘图拒绝缺少两种必需模型、重复电路/模型行、非有限值或超出 `[0, 1]` 的覆盖率；
- 保留旧模型读取兼容性，但新的训练启动器不再生成 `fanin_mean` 工件。

## 验证

新增或更新测试以覆盖：

1. `CC0/CC1` 随目标值改变 fanin 选择，容易/困难分支分别选择最小/最大代价；
2. 多个 D-frontier 候选按最小 `CO` 选择，平局确定；
3. 未启用 `-scoap` 时不意外改变兼容路径，正式流程则始终显式启用；
4. manifest 精确包含 16 个电路、每个 50 个检测成功故障；
5. 训练仅启动 `level_gat_gru`，普通 8 轮、强化最多 5 轮；
6. 冗余故障按 `eqv_fault_num` 加权，aborted 不进入成功覆盖率；
7. benchmark 与总计均使用 `(detected_uncollapsed + redundant_uncollapsed) / total_uncollapsed`；
8. 最终比较不含 `fanin_mean`，相对指标以 SCOAP heuristic 为基准；
9. 覆盖率图纵轴严格为 98%–100%；
10. 原生 C++ 测试、Python 单元测试及一组小规模端到端 smoke run 通过。

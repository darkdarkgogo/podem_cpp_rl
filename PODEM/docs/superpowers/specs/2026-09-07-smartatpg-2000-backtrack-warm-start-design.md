# SmartATPG 2000 回溯上限续训设计

## 目标

将当前 SmartATPG 实验的故障筛选、mean 与 GAT-GRU 训练验证、portable/native
推理和最终 benchmark 的 `backtrack_limit` 从 500 统一提高到 2000。新的实验重新
profile 两个训练电路，并在传统启发式于 2000 次回溯内检测到的故障中，每个电路
选择回溯代价最高的 100 个。

新实验从现有 500-limit 实验的两个最佳模型继续学习，同时保留旧实验目录和工件，
避免覆盖已有 checkpoint、TensorBoard 数据和 benchmark 结果。

## 方案选择

采用 weights-only warm start：为 mean 和 GAT-GRU 分别创建全新的训练状态，只从
各自旧 `best_training_state.pth` 的 `policy_old` 加载 Actor 与图编码器参数。新的
Critic、RND 网络、优化器、轮次、最佳分数和 TensorBoard 日志从零开始。

不直接恢复旧 checkpoint 的完整训练状态，因为新 manifest 的故障集合和回溯上限
已经改变，旧 Critic、优化器动量、最佳分数和 episode 位置不再对应新的训练问题。
也不从随机权重重新训练，因为用户要求在已有模型基础上继续训练。

## 回溯上限与数据流

Linux 训练入口默认使用 2000，并把同一个值传给准备脚本。准备脚本对 `c6288` 和
full-scan `s38417` 的全部折叠故障运行传统启发式 PODEM，禁用 fault dropping，
记录每个故障的 outcome、backtracks 和 backtrace steps。筛选仍只接受 detected
故障，并按 backtracks、backtrace steps 降序选择每个电路的前 100 个。

生成的 `training_manifest.json` 保存 `backtrack_limit: 2000`。训练脚本继续只从
manifest 读取上限，使训练 episode 和每轮固定 200-fault 验证使用相同的 2000。

训练完成后，mean 与 GAT-GRU 的新最佳模型组成新的 benchmark bundle。Linux
benchmark 入口和底层 benchmark 脚本默认使用 2000，并将该值传给 heuristic、
mean 和 GAT-GRU 三种模式，确保最终推理比较采用同一搜索预算。

## Warm start 接口

`train_smartatpg.py` 新增可选的 `--warm-start-checkpoint` 参数。该参数只允许在目标
输出目录没有 `training_state.pth` 时使用。脚本读取 `best_training_state.pth`，验证
外层 checkpoint 格式、内部 agent 状态、encoder variant 和参数形状，然后调用
agent 已有的 `load_actor_state_dict()` 完成 weights-only 初始化。

如果目标目录已有当前训练 checkpoint，脚本按现有逻辑恢复完整状态，并拒绝同时
指定 warm start，以免不明确地覆盖恢复后的参数。mean checkpoint 不能用于
GAT-GRU，反向也一样；形状或 encoder metadata 不匹配时在第一个 episode 前失败。

双模型 Linux 启动器新增 mean 和 GAT-GRU 两个 warm-start checkpoint 参数，并把
它们分别传给对应训练进程。顶层 shell 入口使用新的 2000-limit 输出目录，并默认
指向现有 500-limit 实验中两个模型的最佳训练 checkpoint。源 checkpoint 不存在时
应给出包含完整路径的错误信息。

## 工件隔离与可复现性

2000-limit 实验使用新输出目录，准备 manifest、训练 checkpoint、TensorBoard、
导出模型、bundle 和 benchmark 结果都不会与旧 500-limit 实验混写。训练 metadata
记录回溯上限和两个 warm-start 来源；训练 checkpoint 的 config 继续包含 2000，
后续只能在相同 manifest 和配置下恢复。

随机种子保持当前分工：profile seed 控制 C++ 随机填值，training seed 控制 Torch
随机状态和每轮 episode 顺序，benchmark seed 控制三种模式的最终向量补全。默认
seed 不因回溯上限变化而改变。

## 错误处理

- 重新使用包含 500-limit manifest 的准备目录时，现有 resume 校验应拒绝它；新
  实验必须使用新目录重新 profiling。
- warm-start checkpoint 缺失、格式错误、encoder 不匹配或参数形状不匹配时，
  在训练开始前失败。
- 已存在训练 checkpoint 与 warm-start 参数同时出现时失败，并说明应选择恢复现有
  训练或使用空的新输出目录。
- benchmark 的回溯上限必须为正；生成的 run metadata 必须记录实际值 2000。

## 验证

自动测试覆盖：默认回溯上限为 2000；准备命令、两个训练命令和 benchmark 命令都
收到 2000；fresh agent 能从同架构 checkpoint warm start；跨 encoder checkpoint、
非法 checkpoint 以及非空目标 checkpoint 加 warm start 被拒绝；warm start 后
Critic 和优化器没有继承旧状态。

端到端检查使用小型测试电路验证 profiling、训练调用和推理调用均收到同一个
2000。Linux 实际续训启动前检查两个旧最佳 checkpoint 存在，并确认新 manifest
重新生成且包含 `backtrack_limit: 2000`。训练完成后，最终 benchmark metadata 和
原生命令必须显示 `-bt 2000`。

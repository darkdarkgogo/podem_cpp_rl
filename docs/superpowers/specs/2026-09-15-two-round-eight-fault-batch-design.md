# 两轮八故障批次训练设计

## 目标

新训练协议执行两轮训练。训练期间连续收集 8 个 training fault 的完整轨迹，
合并后执行一次 PPO 更新。每次更新只遍历该批数据一次，即 `k_epochs=1`。
Validation fault 只用于评估，不进入训练批次，也不会更新模型。

## 协议版本与兼容性

引入新的 V8 manifest 和训练协议。新准备的数据固定使用以下参数：

- `normal_rounds=2`
- `faults_per_update=8`
- `k_epochs=1`

现有 V6/V7 manifest 和 checkpoint 保持原有语义：五轮训练、每个 fault 更新一次，
`k_epochs` 继续使用旧任务保存的值（默认为 8）。加载旧 checkpoint 时必须使用
旧协议参数，避免在续跑过程中改变优化器和采样语义。

模型导出、便携模型加载和 benchmark 校验同时接受旧的五轮协议与新的两轮协议，
但会拒绝轮数和协议版本不匹配的组合。

## 训练数据流

每个 fault 仍独立运行 PODEM，并在 episode 结束时标记轨迹终点，但不立即调用
`agent.update()`。共享 rollout buffer 跨 fault 保留轨迹，直到累计完成 8 个 fault。

达到批次边界后：

1. 使用 buffer 中 8 个 fault 的全部决策步骤计算 return 和 advantage。
2. 执行一次 PPO 前向、反向传播和 `optimizer.step()`。
3. 执行一次 RND predictor 更新。
4. 将新策略同步到 `policy_old`。
5. 清空 rollout buffer，并保存模型和 checkpoint。

如果每轮末尾剩余不足 8 个 fault，则使用剩余轨迹执行一次更新，然后才能进入
validation。没有产生策略决策步骤的 fault 仍计入批次；如果整个批次没有任何决策
步骤，则安全跳过优化器更新，但仍推进批次和训练进度。

每轮训练完成后执行一次完整 validation，然后进入下一轮。第二轮 validation 完成后
训练结束，并从两轮结果中保留 validation 得分最好的模型。

## 训练器接口

`CppPodemBacktraceV2Trainer` 增加延迟更新能力。默认行为继续保持每个 episode 自动
更新，以免影响其他调用方。新 V8 训练入口显式关闭自动更新，由训练循环在批次边界
调用共享 agent 的 `update()`。

旧 V6/V7 训练继续启用原有自动更新路径。新旧行为由 manifest 协议版本决定，不能
在一次训练中动态切换。

## 指标与日志

每个 fault 继续记录 outcome、backtracks、backtrace steps 和奖励。PPO loss、RND loss
和 batch 中的决策步数改为在每次 8-fault 更新时记录。控制台新增批次更新进度，明确
显示本批 fault 数量和累计完成的 fault 数量。

## Checkpoint 与断点续跑

V8 checkpoint 只在完成一次批次更新后保存。checkpoint 中的 `episode_index` 和
`completed_episodes` 因此始终位于批次边界。

如果进程在尚未凑满 8 个 fault 时中断，尚未更新的 rollout buffer 不持久化；恢复时
从最近的批次边界重新执行，最多重跑 7 个 fault。这样不需要序列化带梯度相关状态的
临时 rollout，并能保证恢复后的优化结果仍来自完整、确定的批次。

每轮最后的不足 8 个 fault 在更新和保存 checkpoint 后，才切换到 validation 状态。

## 校验与错误处理

V8 manifest、CLI 参数、checkpoint config 和模型元数据必须一致声明两轮、8-fault
batch 和单 epoch。发现不一致时立即停止，不能静默采用默认值。

恢复 V8 checkpoint 时还要校验 manifest 哈希、validation catalog 哈希和完整训练
配置。旧 checkpoint 继续按原字段和原协议校验。

## 测试

测试将验证：

1. 新协议固定为两轮、每批 8 个 fault、`k_epochs=1`。
2. 前 7 个 fault 不执行优化，第 8 个 fault 后只执行一次 optimizer step。
3. 每轮末尾不足 8 个 fault 时执行一次最终更新。
4. Validation 不向 rollout buffer 添加训练数据，也不更新模型。
5. Checkpoint 只在更新边界推进，恢复时最多重跑 7 个 fault。
6. 每轮执行一次 validation，并从两轮结果中选择最佳模型。
7. V6/V7 checkpoint 保持原有五轮、单 fault 更新行为。
8. 新旧模型均可通过便携加载和 benchmark 协议校验。

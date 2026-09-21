# SmartATPG PPO Epoch-4 与学习率协议设计

## 目标

仅将 `level_gat_gru` SmartATPG 正式训练协议从每批一次 PPO 优化改为每批
四次优化，同时降低它的 Actor 和 Critic 学习率：

- `normal_rounds=2`
- `faults_per_update=8`
- `k_epochs=4`
- `actor_lr=0.0003`
- `critic_lr=0.001`

`fanin_mean` 保持原协议：`normal_rounds=2`、`faults_per_update=8`、
`k_epochs=1`、Actor `0.001`、Critic `0.01`。本次不增加 KL、clip fraction
或额外实验指标。

## 协议兼容性

这是一次仅针对 GAT 的有意不向后兼容切换。新的 GAT 训练、恢复、导出、便携加载
和 benchmark 流程只接受 epoch-4 协议。旧的 GAT epoch-1 manifest、checkpoint
和 V13 模型不得被新版流程加载或续训，必须以清晰的格式或协议错误拒绝。

GAT 正式模型格式从带有 `BATCH8_EPOCH1` 语义的 V13 升级为新的
`BATCH8_EPOCH4` 格式；GAT checkpoint 的训练格式标识也同步升级。GAT 的生产
入口、导出元数据、便携加载器、benchmark 校验和测试夹具必须使用同一新格式与
参数，不能保留静默接受旧 GAT 格式的分支。

Mean 继续使用现有 V13 epoch-1 模型格式、训练格式和工件。加载器必须按
`encoder_variant` 区分协议：V13 只允许 `fanin_mean` epoch-1，新的 epoch-4
格式只允许 `level_gat_gru`。本次不得改变 Mean 的加载、恢复或推理行为。

## 训练数据流

对于 GAT，每个 fault 仍是一个独立完整 episode。训练循环继续累计八个 fault 的
轨迹，在批次边界调用一次 `agent.update()`。一次 `agent.update()` 内对同一
rollout buffer 执行四个完整 PPO epoch，因此产生四次 PPO optimizer step；RND
predictor 仍按现有语义在整个 PPO epoch 循环结束后只更新一次。

每次 GAT PPO 更新完成后继续同步 `policy_old`、清空 rollout buffer、导出最新
Actor 并保存 checkpoint。每轮末尾不足八个 fault 的批次仍按相同的四 epoch
规则更新。两轮训练、每轮后的完整 validation 和最佳模型选择逻辑保持不变。

Mean 的数据流不变：仍以八个 fault 为一批，每批只执行一个 PPO epoch。

## 学习率

训练入口创建 GAT agent 时显式使用 Actor `0.0003` 和 Critic `0.001`；创建 Mean
agent 时继续使用 Actor `0.001` 和 Critic `0.01`。checkpoint 配置和相关断言必须
按 encoder 记录并校验精确值。

`GATGRUSmartATPGPPOAgent` 的默认值同步为新的 GAT 学习率，Mean agent 和基础通用
PPO 类的默认学习率不修改。

## 错误处理

- GAT epoch-1 manifest 在准备复用或训练加载阶段必须因协议字段不匹配而失败。
- GAT epoch-1 checkpoint 不得恢复或作为 continuation 使用。
- GAT V13 epoch-1 Actor 模型不得通过新的 Python 或便携加载路径。
- GAT CLI 传入非 `4` 的 `--k-epochs` 时必须立即失败。
- Mean CLI 继续要求 `--k-epochs=1`。
- 导出或 benchmark 元数据与对应 encoder 协议不一致时必须立即失败。

错误不得通过自动改写旧工件或忽略旧字段来规避。用户需要重新准备数据并重新训练。

## 文档和格式标识

当前使用说明、训练协议说明、GAT 模型格式常量、GAT 测试名和 GAT 输出目录中带有
epoch-1 语义的活动内容都更新为 epoch-4。Mean 的 epoch-1 内容保留。历史设计文档
作为历史记录保留，不回写其原始决策；新的设计与实现文件定义当前分流协议。

## 验证

必要检查包括：

1. 单元测试证明 GAT 为 batch 8、epoch 4、两轮及新学习率，Mean 仍为 batch 8、
   epoch 1、两轮及原学习率。
2. GAT 训练批次测试证明前七个 fault 不更新，第八个 fault 触发一次
   `agent.update()`，且其中执行四次 PPO optimizer step；Mean 仍只执行一次。
3. GAT manifest、checkpoint、Actor 导出和便携模型测试证明新协议可往返加载。
4. 负向测试证明旧 GAT epoch-1 manifest、checkpoint 和 V13 模型被拒绝。
5. 回归测试证明现有 Mean epoch-1 manifest、checkpoint 和 V13 模型仍可加载。
6. SmartATPG 相关训练、Linux 启动、导出/加载和 benchmark 测试通过。

不运行完整两轮大数据训练；测试使用现有小型 fixture 验证协议和优化步骤语义。

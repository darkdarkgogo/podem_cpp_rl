# SmartATPG PPO Epoch-4 与学习率协议设计

## 目标

将当前 SmartATPG 正式训练协议从每批一次 PPO 优化改为每批四次优化，
同时降低 Actor 和 Critic 学习率：

- `normal_rounds=2`
- `faults_per_update=8`
- `k_epochs=4`
- `actor_lr=0.0003`
- `critic_lr=0.001`

本次只调整 PPO epoch 数和两组学习率，不增加 KL、clip fraction 或额外实验指标。

## 协议兼容性

这是一次有意的不向后兼容切换。新的训练、恢复、导出、便携加载和 benchmark
流程只接受 epoch-4 协议。旧的 epoch-1 manifest、checkpoint 和 V13 模型不得被
新版流程加载或续训，必须以清晰的格式或协议错误拒绝。

正式模型格式从带有 `BATCH8_EPOCH1` 语义的 V13 升级为新的
`BATCH8_EPOCH4` 格式。所有生产入口、导出元数据、便携加载器、benchmark
校验和测试夹具必须使用同一新格式与参数，不能保留静默接受旧格式的分支。

## 训练数据流

每个 fault 仍是一个独立完整 episode。训练循环继续累计八个 fault 的轨迹，
在批次边界调用一次 `agent.update()`。一次 `agent.update()` 内对同一 rollout
buffer 执行四个完整 PPO epoch，因此产生四次 PPO optimizer step；RND predictor
仍按现有语义在整个 PPO epoch 循环结束后只更新一次。

每次 PPO 更新完成后继续同步 `policy_old`、清空 rollout buffer、导出最新 Actor
并保存 checkpoint。每轮末尾不足八个 fault 的批次仍按相同的四 epoch 规则更新。
两轮训练、每轮后的完整 validation 和最佳模型选择逻辑保持不变。

## 学习率

训练入口创建 SmartATPG agent 时显式使用 Actor `0.0003` 和 Critic `0.001`。
SmartATPG agent 自身的默认值也同步为相同数值，避免直接实例化与正式训练入口产生
不同语义。checkpoint 配置和相关断言必须记录并校验这两个精确值。

基础通用 PPO 类的默认学习率不属于 SmartATPG 正式协议，本次不修改。

## 错误处理

- epoch-1 manifest 在准备复用或训练加载阶段必须因协议字段不匹配而失败。
- epoch-1 checkpoint 不得恢复或作为 continuation 使用。
- V13 epoch-1 Actor 模型不得通过新的 Python 或便携加载路径。
- CLI 传入非 `4` 的 `--k-epochs` 时必须立即失败。
- 导出或 benchmark 元数据与 epoch-4 协议不一致时必须立即失败。

错误不得通过自动改写旧工件或忽略旧字段来规避。用户需要重新准备数据并重新训练。

## 文档和格式标识

当前使用说明、训练协议说明、模型格式常量、测试名和输出目录中带有 epoch-1
语义的活动内容都更新为 epoch-4。历史设计文档作为历史记录保留，不回写其原始决策；
新的设计与实现文件定义当前协议。

## 验证

必要检查包括：

1. 单元测试证明正式常量为 batch 8、epoch 4、两轮及新学习率。
2. 训练批次测试证明前七个 fault 不更新，第八个 fault 触发一次
   `agent.update()`，且其中执行四次 PPO optimizer step。
3. manifest、checkpoint、Actor 导出和便携模型测试证明新协议可往返加载。
4. 负向测试证明旧 epoch-1 manifest、checkpoint 和 V13 模型被拒绝。
5. SmartATPG 相关训练、Linux 启动、导出/加载和 benchmark 测试通过。

不运行完整两轮大数据训练；测试使用现有小型 fixture 验证协议和优化步骤语义。

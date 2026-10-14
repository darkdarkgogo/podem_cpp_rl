# GAT 单因素消融实验设计

## 目标

使用现有完整 GAT-GRU 训练与验证结果作基线。从头训练三个消融组：去缓存、关 backtrace 锁、去目标值动态输入。训练与验证由不同脚本启动，不在开发机执行实验。

## 实验组

| 组名 | 图编码器 | 缓存 | 路径锁 | Actor/Critic 目标值输入 | 训练 GPU |
|---|---|---|---|---|---|
| `full` | GAT-GRU | 开 | 开 | 开 | 复用现有结果 |
| `no_cache` | GAT-GRU | 关 | 开 | 开 | 0 |
| `no_lock` | GAT-GRU | 开 | 关 | 开 | 1 |
| `no_object_value` | GAT-GRU | 开 | 开 | 关 | 2 |

`no_cache` 只在原生验证时绕过 C++ Actor logits 缓存，Python 图 embedding 缓存保持原样。GRU 结构与参数保留。该缓存只用于推理重复计算优化，因此该组主要反映运行开销，预测指标预期接近完整模型。`no_object_value` 仅删除网络输入中的目标值；PODEM 逻辑仍使用目标值。

三组直接使用完整模型的 GAT manifest、训练 seed、奖励和 backtrack limit。启动器为三组建立独立进程，分别设置 `CUDA_VISIBLE_DEVICES=0,1,2`，并发训练；每个进程要求恰好一张可见 CUDA GPU，不能静默退回 CPU。只在三个进程全部成功并生成工件后写入实验摘要。

## 验证和结果

验证脚本从完整模型原有验证目录读取 `detailed/gat_records.jsonl`、选择的轮次及训练 checkpoint，核对 seed、路径锁、manifest、有序故障目录以及已有报告的汇总指标。无缓存组的 Actor 快照必须与完整模型完全相同，否则拒绝缓存效果对比。随后对三个消融组逐故障运行原生验证，检查 checkpoint 的组名和路径锁状态。输出每组逐故障 JSONL、逐电路及 TOTAL CSV/JSON，列出 detected、redundant、aborted、coverage、backtracks、backtrace steps、ATPG 时间及相对完整模型的差值。

原双模型训练与验证入口保留。完整模型不重新训练。本机仅编辑代码，不执行训练或验证。

# GAT 单因素消融实验设计

## 目标

以现有完整 `level_gat_gru` 模型为基线，分别移除 GRU 记忆、backtrace 路径锁和目标值动态输入。新增独立训练与验证脚本，不改变既有双模型实验入口。不在开发机启动训练或验证。

## 四组

| 组名 | 图编码器 | 路径锁 | Actor/Critic 目标值输入 |
|---|---|---|---|
| `full` | 双向逐 level GAT-GRU | 开 | 开 |
| `no_gru` | 双向逐 level GAT，注意力消息加残差后 ELU | 开 | 开 |
| `no_lock` | 双向逐 level GAT-GRU | 关 | 开 |
| `no_object_value` | 双向逐 level GAT-GRU | 开 | 关 |

`no_object_value` 仅移除神经网络输入中的目标值；PODEM 计算逻辑目标及 fanin 所需值仍需目标值。四组均从随机初始化训练。所有组使用同一训练 manifest、故障目录、种子、超参数、奖励、验证电路和 backtrack limit。路径锁训练与验证状态一致。

## 工件与验证

新结构变体使用独立的 encoder variant、graph config 与维度元数据。checkpoint 额外记录消融组名和路径锁状态。验证脚本先核对训练摘要、checkpoint 和 Actor/embedding 工件，再执行原生逐故障验证。输出每组逐故障 JSONL、逐电路与 TOTAL CSV/JSON；包括 detected、redundant、aborted、coverage、backtracks、backtrace steps、ATPG 时间及相对完整模型的差值。

## 边界与验收

原双模型训练和四方验证行为保持不变。独立脚本只使用 GAT 训练 manifest。验证拒绝组名、encoder variant、锁状态、维度或快照不匹配的工件。代码交付后由用户在 GPU 环境运行；本机不执行实验。

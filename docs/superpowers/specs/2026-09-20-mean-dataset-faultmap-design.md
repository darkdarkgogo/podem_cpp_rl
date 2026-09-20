# Mean 独立训练集与 Fault Map 设计

## 范围

SmartATPG 的两个 encoder 使用不同训练集，但共享同一验证集：

- `level_gat_gru` 继续使用 `PODEM/data/train`。
- `fanin_mean` 改用 `PODEM/data/train_mean`。
- 两者都使用 `PODEM/data/validation` 的六个固定验证电路，并在每轮验证时测试各验证电路的完整 collapsed fault catalog。

本次修改只改变训练数据路由、mean 训练 fault 选择和相关协议身份，不改变 PPO/RND、奖励函数、网络结构、训练轮数或 validation 最佳模型评分规则。

## 数据契约

### GAT 训练集

GAT 保持现有契约：`data/train` 必须包含 1024 个 `.bench` 电路。每个电路使用 SCOAP heuristic PODEM profiling，仅从 `outcome == 1` 的可检测 fault 中选择最多 30 个困难 fault。

困难度排序按以下确定性顺序执行：

1. `backtracks` 降序；
2. `backtrace_steps` 降序；
3. `fault_id` 字典序升序。

### Mean 训练集

`data/train_mean` 必须包含下列资产：

- `c6288.bench`
- `s38417.bench`
- `s38417_scan.bench`
- `s38417_scan_binary.bench`
- `s38417_scan_binary.bench.uf`
- `s38417_scan_binary.faultmap`

其中只有两个 ATPG 训练对象：

- `c6288`：使用 `c6288.bench` 建图、profiling 和训练，不使用 fault map。
- `s38417`：使用 `s38417_scan_binary.bench` 建图并运行 profiling/训练，同时加载 `s38417_scan_binary.faultmap`。fault map 保存 `s38417_scan.bench` 的 collapsed fault 身份并映射到拆分为两输入门的 binary netlist，因此训练 fault 的数量与 ID 均以 scan 版本为准。

每个 mean 训练对象仅从 `outcome == 1` 的可检测 fault 中选择恰好 100 个困难 fault。排序规则与 GAT 相同：`backtracks` 降序、`backtrace_steps` 降序、`fault_id` 升序。若任一电路不足 100 个可检测 fault，准备过程必须明确失败，不得静默缩减。mean 每轮训练总计 200 个 fault episode。

目录中的原始时序版、scan 版和 `.uf` 文件是可追溯转换资产，不得被误当成额外训练电路。准备状态必须记录所有 mean 数据资产的相对路径与 SHA-256，使恢复运行能拒绝任一源文件、转换结果或映射文件发生变化。

## Manifest 与准备流程

现有准备脚本扩展为显式接收训练配置，不通过扫描目录猜测 encoder：

- GAT 配置选择 `train`、1024 个训练电路、每电路最多 30 个 fault。
- mean 配置选择 `train_mean`、两个训练对象、每电路恰好 100 个 fault，并为 `s38417` 记录 fault map。

每种配置生成独立的 preparation 目录和 `training_manifest.json`。manifest 必须记录：

- encoder variant 或等价的训练集身份；
- 训练 split 名称；
- 每电路 fault 选择数；
- 训练与 validation 电路记录；
- 每个执行电路的路径和哈希；
- 可选 fault-map 路径和哈希；
- mean 转换源资产清单和哈希；
- profiling seed、backtrack limit 和训练协议字段。

恢复准备时必须重新验证数据 inventory、manifest、profile、fault map 和协议字段。GAT manifest 不能用于 mean，mean manifest 也不能用于 GAT。

## Profiling 与训练数据流

正式 SmartATPG profiling、训练和 validation 的 `backtrack_limit` 均严格为 100。

profiling 对普通训练电路调用现有 C++ profile 接口；对 `s38417` 同时传入 binary BENCH 和 fault-map 路径。返回的 profile 中 `fault_id` 是 scan fault 身份，随后按 mean 的 top-100 规则保存到 manifest。

训练脚本解析 manifest 中的可选 `fault_map`：

- 图特征始终从实际执行电路生成；`s38417` 因而从 `s38417_scan_binary.bench` 建图。
- 每个训练 episode 调用 C++ trainer 时传入该电路的 fault-map 路径。
- `c6288` 和 GAT 训练电路不传 fault map。

validation 记录来自同一个 `data/validation`，不继承训练电路的 fault map。GAT 与 mean 的 validation catalog hash、validation 电路名和每个电路的 fault ID 列表必须完全一致，保证比较基于同一测试集。

## 双模型启动与输出隔离

双 GPU 启动器先顺序生成两份 preparation：

- GAT preparation 使用 `data/train` 与 `data/validation`。
- mean preparation 使用 `data/train_mean` 与同一个 `data/validation`。

随后并行启动两个训练进程，各自传入匹配的 manifest。comparison 不再接收单一共享 manifest，而是分别接收 GAT 和 mean manifest；它先验证两份 manifest 的正式协议兼容，再严格验证 validation 身份一致。

运行 metadata 必须分别记录两份 manifest 的路径、哈希、训练电路数、训练 episode 数和 validation 身份，避免将 mean 的 200-episode 训练误报成 GAT 的训练规模。

单 GAT 启动器继续只准备并训练 GAT，不因 `train_mean` 的存在改变行为。

## Backtrack 上限与残留清理

当前正式 SmartATPG 工作流中的下列位置必须统一并校验 `backtrack_limit == 100`：

- preparation 和 profile metadata；
- GAT/mean 训练启动器与 shell wrapper；
- training config、checkpoint 恢复和 validation identity；
- 模型导出、portable/native 加载与 benchmark；
- comparison validation。

清理现行脚本、测试、shell wrapper、`RL_GUIDE.md` 和 `docs/SMARTATPG_11D_使用说明.md` 中仍描述正式流程为 backtrack 200、旧训练轮数或 GAT/mean 共用 `data/train` 的残留。

历史规格/计划文档记录的是过去实验，不改写其中的旧参数。通用 ATPG API 的独立默认值（例如 97）、明确测试其他限制的 fixture，以及不属于正式 SmartATPG 工作流的独立工具参数，不机械改为 100；必须通过定向审计确认每个保留值都有明确用途。

## 错误处理

以下情况必须在训练开始前失败，并给出包含资产或电路名的错误：

- mean 所需的六个资产缺失或出现未允许的额外资产；
- fault map 格式、源 hash 或 binary circuit hash 不匹配；
- `s38417` 未通过 fault map 获得 scan 版本 fault catalog；
- 任一 mean 电路不足 100 个 `outcome == 1` fault；
- manifest 的 encoder、训练 split、fault 数或 backtrack limit 与训练命令不匹配；
- GAT 与 mean 的 validation 电路或 fault catalog 不一致；
- resume 时任一数据、profile、fault map 或 manifest identity 改变。

## 验证

定向测试覆盖：

- GAT discovery 仍要求 `data/train` 的 1024 个 BENCH 和固定 validation。
- mean discovery 只产生 `c6288` 与映射后的 `s38417` 两个训练记录，并验证六个资产。
- mean 每电路按约定排序选择恰好 100 个可检测 fault，总计 200 个。
- `s38417` profiling 与训练都向 C++ 传入 `s38417_scan_binary.faultmap`，图来自 binary BENCH，manifest fault ID 来自映射后的 scan catalog。
- GAT 与 mean 使用独立 manifest，但 validation catalog 完全相同。
- 双训练启动器向两个 encoder 传入各自 manifest，metadata 不混淆训练规模。
- 正式入口拒绝非 100 的 backtrack limit。
- 静态残留扫描不再发现现行工作流中的 `bt200`、正式 `backtrack_limit=200` 或 mean 使用 `data/train` 的旧假设。

验证只运行单元测试、静态检查和必要的原生接口定向测试，不启动正式训练或 benchmark。

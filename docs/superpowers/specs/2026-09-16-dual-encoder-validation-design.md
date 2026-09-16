# GAT-GRU 与 MEAN 双模型训练验证设计

## 目标

在同一份 SmartATPG V8 数据划分和训练协议上并行训练两种图编码器：

- GAT-GRU 使用物理 GPU 0；
- fanin-MEAN 使用物理 GPU 1；
- 两者都训练两轮，每 8 个 fault 更新一次，每批轨迹只使用一次；
- 每轮结束后都在同一组 6 个 validation 大电路上验证；
- SCOAP heuristic 在这 6 个 validation 电路上运行一次，作为固定基线；
- 输出两轮的完整对比结果，不再运行或生成 ISCAS benchmark。

训练集固定为 `data/train` 中的 1024 个电路。Validation 集固定为：

- `b12_C`
- `b15_C`
- `b17_C`
- `b20_C`
- `b21_C`
- `b22_C`

## 启动方式

新增一个 Linux 双模型总入口。它只执行一次数据 preparation，随后并行启动两个独立训练进程：

```text
shared preparation manifest
├── GAT-GRU training + round validation  -> physical GPU 0
└── MEAN training + round validation     -> physical GPU 1
```

GPU 编号可以通过命令行覆盖，但默认分别为 0 和 1。两个进程使用相同的 manifest、训练 fault 顺序、validation fault catalog、训练种子和验证种子。这样编码器类型是两次实验之间唯一的模型结构差异。

总入口负责进程生命周期和日志归属。任一训练进程失败时，它会终止仍在运行的另一个进程，保留双方已有 checkpoint，并返回非零状态。失败后再次启动默认使用 `--resume` 从各自最近的批次边界继续。

## 模型与训练协议

`train_smartatpg.py` 的编码器选择扩展为：

- `level_gat_gru`：现有 `GATGRUSmartATPGPPOAgent`；
- `fanin_mean`：现有 `SmartATPGPPOAgent`。

两种模型都遵循 V8 协议：

- `normal_rounds=2`
- `faults_per_update=8`
- `k_epochs=1`
- `backtrack_limit=200`

每个 fault 单独形成 episode，8 个 fault 的 rollout 累积后调用一次 `agent.update()`；每轮末尾不足 8 个 fault 的剩余轨迹也更新一次。两种模型使用不同输出目录、checkpoint、TensorBoard 事件和模型文件，不能互相恢复 checkpoint。

模型导出继续使用 V13 训练协议元数据。模型文件通过已有 `encoder_variant` 和 `graph_config` 区分 GAT-GRU 与 MEAN，Python 与 C++ 加载器都必须接受两种 V13 模型。

## Validation 与 SCOAP 基线

GAT-GRU 和 MEAN 在每轮训练结束后分别验证一次。每次验证覆盖 6 个 validation 电路的完整运行时 fault catalog，不向 rollout buffer 写数据，也不更新模型。

SCOAP heuristic 与 preparation 解耦，不生成提前持久化的 validation profile。双模型总入口在 validation 电路上运行一次固定基线，并缓存结果供两个轮次共同引用。SCOAP 不按训练轮次重复运行，因为它的行为不随模型训练变化。

每种模型仍根据现有 validation score 独立选择自己的 `model_best.txt`。统一报告必须同时保留 round 1 和 round 2，不能只展示最佳轮次。

## 输出结构

默认输出目录使用能够表明双模型实验的名称，内部结构为：

```text
artifacts/smartatpg_dual_top30_hard_2rounds_batch8_bt200/
├── preparation/
├── smartatpg_gat_gru/
├── smartatpg_mean/
├── train_gat_gru.log
├── train_mean.log
├── scoap_validation.json
├── validation_comparison.json
├── validation_comparison.csv
└── training_run_metadata.json
```

不再创建 `benchmark_bundle`，也不调用 ISCAS benchmark 脚本。

`validation_comparison.json` 保存机器可读的完整数据和实验身份；`validation_comparison.csv` 提供便于查看和作图的扁平表。报告包含每个 validation 电路以及 6 个电路总体汇总，至少记录：

- fault coverage；
- detected、redundant、aborted；
- backtracks；
- backtrace steps；
- ATPG 时间；
- test vectors；
- GAT-GRU 相对 SCOAP 的变化；
- MEAN 相对 SCOAP 的变化；
- 同一轮 GAT-GRU 与 MEAN 的直接差异；
- 两种模型各自的最佳轮次和 validation score。

## 一致性与错误处理

生成统一报告前必须校验：

- 两个训练目录引用相同的 manifest 哈希；
- 两个模型都声明 V8 的两轮、batch 8、epoch 1 协议；
- 两边 validation catalog 哈希一致；
- 两边都完整包含 round 1 和 round 2 的 validation 记录；
- SCOAP 结果覆盖相同的 6 个电路和相同 fault ID 集合。

任何身份、协议或记录不一致都应停止并给出明确错误，不能生成部分对比报告。报告文件使用临时文件加原子替换，避免中断后留下半写文件。

## 兼容性

现有单 GAT-GRU 训练入口继续可用，避免破坏已有自动化。V6/V7 manifest 与 V12 模型仍保持原有五轮、逐 fault 更新的兼容行为。新双模型总入口只接受 V8 manifest。

现有独立 benchmark 脚本可以保留供旧实验使用，但新双模型入口不调用它，也不把 ISCAS 数据纳入本实验结果。

## 测试

测试覆盖：

1. `fanin_mean` 可以通过统一训练入口创建正确的 agent，并遵守 batch 8、epoch 1 协议。
2. GAT-GRU 和 MEAN 使用相同的 manifest、fault 顺序与种子，但输出目录彼此隔离。
3. 双进程默认映射到物理 GPU 0 和 1，并允许参数覆盖。
4. 任一子进程失败时，另一个子进程被终止且总入口返回失败。
5. 两种模型每轮各验证一次，报告包含四组模型结果和一组固定 SCOAP 结果。
6. SCOAP 只执行一次，并使用与模型 validation 相同的电路和 fault 集合。
7. JSON 和 CSV 的逐电路值、总体汇总和差值一致。
8. 缺失轮次、manifest 不同、catalog 不同或协议不同时拒绝生成报告。
9. Python 与 C++ 加载器均能读取 GAT-GRU 和 MEAN 的 V13 模型。
10. 旧单模型训练、V6/V7 manifest 和 V12 模型兼容测试继续通过。

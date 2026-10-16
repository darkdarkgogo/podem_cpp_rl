# SmartATPG RL 使用入口

## GAT 单因素消融（独立入口）

拼接版 GAT 消融只训练和验证三组：`no_cache`、`no_lock`、`no_object_value`。每组训练一轮，分别使用 GPU 0、1、2；控制台训练输出带组名前缀。验证只读取这三组的训练摘要和模型文件，不读取旧 GAT full 的 checkpoint、验证记录或报告。

```bash
cd PODEM
python3 -m pip install -e .
python3 scripts/train_gat_ablation.py
python3 scripts/validate_gat_ablation.py
```

如果三组已经训练完成，只需更新代码并直接运行最后一条验证命令；旧版 `artifacts/gat_ablation/train_summary.json` 仍可使用，无需重新训练。新训练会在 `artifacts/gat_ablation/preparation/gat` 准备一份共享 GAT manifest；也可用 `--manifest PATH` 指向现有的一轮 GAT manifest。已有三组训练摘要时，重复训练命令会复用其中的 manifest 和 seed 以恢复训练，不读取 full。可用 `--no-cache-gpu`、`--no-lock-gpu`、`--no-object-value-gpu` 指定三张 GPU。

验证结果保存在 `artifacts/gat_ablation/validation/validation_gat_ablation.csv` 和同名 JSON，并分别保存三组逐故障 JSONL。表格只含三组逐电路和 TOTAL 的 detected、redundant、aborted、coverage、backtracks、backtrace steps 和 ATPG runtime。此表没有 full 基线及相对 full 的差值。

`no_cache` 在原生验证时关闭 C++ Actor logits 缓存；训练使用 Python Actor 回调，不受该开关影响。三个组各自使用自己的 checkpoint。路径锁只在 `no_lock` 组关闭；`no_object_value` 的 Actor 输入为 22 维，Critic 输入为 11 维。验证仍核对各组 checkpoint 的组名、锁状态、seed 和 manifest。

训练不会读取 validation fault catalog 或自动启动验证；训练结束后需显式运行验证命令。

## 训练

```bash
cd PODEM
python scripts/train_smartatpg.py
```

GAT-GRU 与 Mean 分别在两张 GPU 上训练。训练 manifest 仅包含 train circuits，验证目录缺失也不影响训练准备。每轮都会同时保存完整的 `inference_round_XX.pth` 与 native `model_round_XX.txt`；前者包含稍后生成 validation embedding 所需的图编码器权重。`training_state.pth` 只用于训练恢复。

训练和验证命令都支持 `--backtrace-lock {auto,on,off}`。默认 `auto` 会为 GAT-GRU 开启 backtrace 路径锁、为 Mean 关闭路径锁；`on` 或 `off` 可在该次命令中强制两个模型使用相同行为。

正式 PPO 协议对两个 encoder 一致：每个 rollout 收集 4 个完整 fault，先在保持时序和 terminal 边界的条件下计算一次 GAE，再于每个 PPO epoch 独立打乱所有 transitions，并按 512 个 transition 切分 minibatch。GAT-GRU 与 Mean 都执行 4 个 epoch；最后不足 512 个 transition 的 minibatch 和 round 末不足 4 个 fault 的 rollout 都不会丢弃。

日志中的 PPO loss、entropy、ratio、KL 与 clip fraction 是全部 4 个 epoch、所有 minibatch 按 transition 数加权的均值，同时保留 `*_last_epoch` 指标。模型协议包含 `faults_per_update=4`、`minibatch_size=512` 与 `k_epochs=4`；旧 MINIBATCH128、Batch8 或 Mean-K1 checkpoint 不能恢复到当前训练。

## 验证

```bash
python scripts/validate_smartatpg.py
```

默认读取项目内的 `artifacts/smartatpg_dual` 与 `artifacts/nn_all_podem`，运行 SCOAP、GAT-GRU、Mean、NN-All 四方验证，并将结果写入前者的 `validation` 子目录。因此须先完成 GAT/Mean 和新版 NN-All 训练。需要验证其他训练运行时，可用 `--run-dir PATH` 与 `--nn-all-run-dir PATH` 覆盖默认目录；`--dataset-root`、`--output-dir` 和 `--seed` 仍可按需指定。若仅需原有三方验证，使用 `--three-way-only`。

验证运行时才独立发现 `data/validation` 中的全部 `.bench` 电路，并按文件名顺序评估 GAT-GRU 各轮、Mean 各轮、fresh SCOAP baseline 和 NN-All，分别选择前两个模型的 best round，并生成：

目录中的电路必须是 SmartATPG 支持的 normal BENCH：`AND`、`NAND`、`OR`、`NOR` 均为二输入，`NOT` 为单输入，并且不包含 `BUF`。多输入 BENCH 可先用 `tools/convert_binary_bench.py` 转换。

- `validation_three_way_comparison.csv`
- `validation_three_way_comparison.json`
- `validation_four_way_comparison.csv`
- `validation_four_way_comparison.json`
- `model_selection.json`
- `detailed/{scoap,gat,mean,nn_all_podem}_records.jsonl`

四方主表对每个验证电路并排给出四种方法的 backtracks、backtrace steps、ATPG runtime 和 fault coverage，最后一行为 TOTAL。三方表同时保留。验证不提供 resume，也不会读取旧的 SCOAP cache。

## NN-All PODEM baseline

NN-All 单独训练，只读取 `data/nn_all_paper_train/c1908.bench` 和 `c2670.bench` 两份完整 ISCAS85 电路，不复用 GAT-GRU 或 Mean 的权重。默认 seed 为 2026，对两份电路遍历完整 stuck-at 故障目录。原始 BENCH 及规范化后的输入都保存在 `data/nn_all_paper_train`；来源和哈希见该目录的 README。按论文 §3.2.2，候选输入为 12 维：LDPI、SDPI、SCOAP CC0、SCOAP CC1、COP 概率和当前回溯门类型的 7 维 one-hot；门类型口径与作者公开推理源码一致。LDPI 和两个 SCOAP 值在每个电路的相同 SDPI 线路内分别作 min-max 归一化；COP 保持原概率，SDPI 保持原层级，不再施加 `log1p` 或训练集全局标准化。论文只指定按 SDPI 分组归一化至 [0,1]，没有写出具体公式；这里以 min-max 落实。网络为 12→60→1 sigmoid，在 PODEM 回溯的每一层为合法候选打分。旧的 11 维模型与特征文件格式均不兼容，验证会拒绝加载。

所有收集到的正负路径样本都参与训练，不划分留出电路，也不截断每类样本。训练使用论文的普通 BCE 和 30,000 个 epoch；相同特征与标签的重复样本按出现次数加权，计算出的全样本 BCE 与逐条计算相同。完成最后一轮后直接保存 `model_final.txt`，没有挑选 checkpoint 的步骤。样本暂存到磁盘以控制内存；`train_summary.json` 记录路径数量、层级分布、训练损失和模型 SHA-256。

采样以一次 PI 回溯链为单位：PI 被翻转或回溯到死路时，该链上的可选择决策标为负；成功检测路径上的决策标为正。只有一个未赋值输入的强制选择不进入训练。`train_summary.json` 保存模型 SHA-256，验证时会同时检查模型、manifest 与特征文件的身份。

```bash
cd PODEM
python scripts/train_nn_all.py
python scripts/validate_smartatpg.py
python tools/plot_final_comparison.py artifacts/smartatpg_dual/validation/validation_four_way_comparison.csv --output-dir artifacts/smartatpg_dual/validation/plots
```

验证使用和三方比较完全相同的 validation 电路、逐故障顺序、backtrack limit 100 和 `atpg_seconds` 定义；保留原有三方文件，另写 `validation_four_way_comparison.csv/.json` 与 `detailed/nn_all_podem_records.jsonl`。四方表包含每种方法的 detected/redundant/aborted 数、backtracks、backtrace steps、ATPG runtime、fault coverage，以及相对 SCOAP 的变化。使用 `--three-way-only` 时仅生成原有三方结果。

这是一份基于本项目 PODEM 引擎的 NN-All 移植，不是 Atalanta/FAN 原实现的逐位复现。训练电路与 epoch 数依照论文；求解器、规范化后的电路和特征实现仍有差异，因此不能称为论文数值实验的严格复现。所给仓库的 [NN_All/fan.cpp](https://github.com/lzzh97/NN-for-ATPG/blob/main/NN_All/fan.cpp) 实际以 `cont0/cont1` 决定回溯输入；神经网络的 11 维输入与推理代码位于 [NN_Hyb/fan.cpp](https://github.com/lzzh97/NN-for-ATPG/blob/main/NN_Hyb/fan.cpp)，并限制在特定层级。本实现依据论文对全层 NN 的定义重实现，因此实验结果应标作 `nn_all_podem`，并在报告中注明此源码差异。

正式 runtime 只使用逐 fault `atpg_seconds` 的和：模型/embedding 加载、图构建、circuit input、levelize、fault-list generation 与写盘不计时；`ATPG::test()` 内的 Actor forward 计时。

## 内部模块边界

- `rl_podem.training` 只负责训练、恢复和每轮模型导出；
- `rl_podem.validation` 只负责 fresh validation 编排与结果写出；
- `rl_podem.validation_core` 负责 fault catalog、native batch、记录汇总和 best-round score，且不依赖 PyTorch；
- `rl_podem.artifact_io` 提供训练和验证共用的 checkpoint 格式标识、manifest hash 与原子 JSON 写入。

`validation` 不导入 `training`，`training` 也不导入 `validation_core`。

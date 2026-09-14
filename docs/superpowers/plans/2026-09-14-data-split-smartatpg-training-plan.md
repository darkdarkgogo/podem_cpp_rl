# 基于数据划分的 SmartATPG 训练实施计划

用户已于2026-09-14确认中文设计：
`docs/superpowers/specs/2026-09-14-data-split-smartatpg-training-design.md`。
当前环境未安装 writing-plans 技能，因此本文件作为等价的可执行实施计划。

## 任务一：重构数据集 preparation

修改 `PODEM/scripts/prepare_smartatpg_training.py`：

1. 用 `--dataset-root` 读取 `data/train/*.bench` 和
   `data/validation/*.bench`，稳定排序并校验1024/6文件契约。
2. 对每个文件运行11维无 BUF 图校验和启发式 fault profiling，统一使用
   backtrack 200。
3. 训练记录保留所有且仅保留 `outcome == 1` 的 fault；验证记录保留 profile
   中的完整 fault catalog。
4. 将 profile 逐电路原子写入 preparation 目录，并通过 preparation state
   支持中断恢复和哈希校验。
5. 发布使用相对路径和内容哈希的新版 manifest，包含独立的
   `train_circuits` 与 `validation_circuits`。
6. 在 `PODEM/tests/test_smartatpg_training.py` 中覆盖数据发现、fault 选择、恢复、
   哈希变化和相对路径迁移。

## 任务二：将训练与验证彻底分离

修改 `PODEM/scripts/train_smartatpg.py`：

1. 加载并严格校验新版 manifest 和两个数据划分。
2. 每轮确定性打乱全部训练 fault，每个 fault 恰好更新一次 PPO/RND。
3. 为验证集建立只读 evaluator，验证完整 catalog 且不更新 agent、optimizer、
   RND 或归一化统计。
4. 将最佳模型评分改为验证集检出数、总 backtracks、总 backtrace steps、总
   extrinsic return 和较早轮次的确定性字典序。
5. 增加可恢复的逐 fault 验证状态，禁止部分验证结果成为 best。
6. 固定普通训练为5轮、backtrack 为200，删除失败 fault 强化训练及相关参数和工件。
7. 补充单元测试，证明训练/验证隔离、best 选择和轮次约束。

## 任务三：支持跨 manifest 完整继续训练

扩展训练 CLI 和 checkpoint 状态：

1. 新增 `--continue-from <checkpoint>`，与同任务 `--resume` 互斥。
2. 从当前新版 training/best checkpoint 恢复完整 agent state、PPO optimizer、
   RND 网络、optimizer、统计和模型侧随机状态。
3. 对新 manifest 重置轮次、episode、验证进度和 best 状态，并记录来源路径及
   SHA-256。
4. 拒绝非空目标目录、架构不匹配、旧12维及其他历史 checkpoint。
5. 测试同任务精确 resume 和跨 manifest continuation 的差异。

## 任务四：升级模型、工件和原生读取协议

修改 `smartatpg_artifacts.py`、`cpp_bridge.py`、`smartatpg_portable.py`、
`backends.py`、`rl_policy.cpp/.h` 以及 benchmark preparation：

1. 导出并只接受 `SMARTATPG_MODEL_V12`，记录新版 manifest 哈希、5轮协议和
   backtrack 200。
2. 保持11维 `SMARTATPG_EMBEDDINGS_V7` 和 snapshot 配对规则不变。
3. 升级 manifest、checkpoint、training state 和 benchmark bundle 格式标识。
4. 移除原来固定16电路、50 faults、8轮和 reinforcement 的协议校验。
5. 保持 Python portable inference 与 C++ 原生 Actor 的数值与元数据一致。
6. 增加 V12 正向与旧格式拒绝测试，并重新编译原生 C++。

## 任务五：更新 Linux 入口与文档

修改 `run_smartatpg_training_linux.py`、三个 Linux shell 入口、`RL_GUIDE.md` 和
中文使用说明：

1. 默认读取 `PODEM/data`，训练5轮并统一使用 backtrack 200。
2. 打包验证集选出的 `model_best.txt`，不再引用 reinforced model。
3. 暴露同任务 resume 和 `--continue-from` 用法。
4. 文档说明 `PODEM/data` 当前若未加入 Git，迁移 Linux 时必须单独复制或提交。

## 任务六：验证与提交

1. 运行 Python 语法检查、纯 Python preparation/manifest 测试、可用的 SmartATPG
   单元测试和 shell 语法检查。
2. 若环境仍没有 PyTorch，明确记录无法运行的 Torch 测试，但使用小型 fixture
   检查不依赖 Torch 的数据路径。
3. 用 MinGW 编译当前 Windows 原生程序；代码同时保持 Linux g++/C++11 兼容。
4. 扫描旧协议常量、2000 backtrack、8轮及 reinforcement 活跃引用。
5. 运行 `git diff --check`，请求独立代码审查，修复全部 Critical/Important 问题。
6. 只提交本任务文件；不混入原有 `atpg.exe`、`input.cpp`、数据目录或旧数据集设计
   文档的未提交改动。

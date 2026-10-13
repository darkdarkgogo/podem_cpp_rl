# GAT 消融实验实施计划

**Goal:** 新增完整 GAT-GRU、去 GRU、去路径锁、去目标值四组单因素训练与验证入口。

**Architecture:** 复用现有训练 manifest、PPO 与 native batch 验证；模型结构变体使用独立工件标识。独立脚本组织四组训练、工件检查和同目录结果汇总。

**Tech Stack:** Python, PyTorch, C++ native Actor, CSV/JSON。

**Spec:** `docs/superpowers/specs/2026-10-13-gat-ablation-design.md`

## Global Constraints

- 训练与验证分别由新脚本启动。
- 四组共享训练与验证电路、seed、奖励和 backtrack limit 100。
- 本机不运行训练或验证。

---

### Task 1: 模型与工件变体

**Files:** `python/rl_podem/gat_gru.py`, `python/rl_podem/training.py`, `python/rl_podem/smartatpg_artifacts.py`, `python/rl_podem/backends.py`, `python/rl_podem/cpp_bridge.py`, `python/rl_podem/smartatpg_portable.py`, `python/rl_podem/smartatpg_rewards.py`, `src/rl_policy.cpp`

- [x] 添加无 GRU 和无目标值 Policy/Agent，保持完整模型现有格式。
- [x] 在 checkpoint、Actor 和 embedding 中记录并校验新变体。
- [x] 让 native Actor 按变体接受 22D 或 23D 候选输入，并校验相应编码器张量。

### Task 2: 独立训练入口

**Files:** `scripts/train_gat_ablation.py`, `python/rl_podem/training.py`

- [x] 准备一次 GAT manifest，逐组从头训练，分别指定锁状态。
- [x] 保存包含组别、路径、manifest、seed 和训练协议的摘要。

### Task 3: 独立验证入口与表格

**Files:** `scripts/validate_gat_ablation.py`, `python/rl_podem/gat_ablation.py`

- [x] 校验四组工件身份、锁状态和相同 fault catalog。
- [x] 顺序执行原生逐故障验证，生成逐组记录和逐电路汇总。
- [x] 输出 CSV/JSON，并计算相对完整模型的差值。

### Task 4: 文档与静态审阅

**Files:** `RL_GUIDE.md`, `docs/superpowers/specs/2026-10-13-gat-ablation-design.md`

- [x] 记录 GPU 机器上的训练、验证命令和输出位置。
- [x] 检查 diff、工件标识、所有变体分支；不启动本机训练或验证。

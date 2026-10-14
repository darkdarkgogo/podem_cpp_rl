# GAT 消融实验实施计划

**Goal:** 复用已有完整 GAT-GRU 结果，并行训练无缓存、无 backtrace 锁、无 object value 三组，再分别验证和对比。

**Spec:** `docs/superpowers/specs/2026-10-13-gat-ablation-design.md`

## 工作项

- [x] 无缓存组保留 GRU 和 Python 图 embedding 缓存，仅在原生验证时绕过 C++ Actor logits 缓存。
- [x] 训练入口读取现有完整 GAT 训练摘要及 manifest，不训练完整组。
- [x] 三个独立进程并发训练，默认使用 GPU 0、1、2；各自产生 checkpoint 和日志。
- [x] 验证入口读取已有完整组逐故障结果，核对协议和有序 fault catalog，再验证三个消融组。
- [x] CSV/JSON 报告包含四组指标和相对完整组差值。
- [x] 更新运行说明。本机不执行训练或验证。

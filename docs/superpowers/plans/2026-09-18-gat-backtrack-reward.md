# GAT Backtrack 奖励实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** GAT 使用有界三次 backtrack 奖励，mean 保留旧 PI 指数奖励，同时把两者的正式 backtrack 上限统一为 100。

**架构：** 新增一个 Python 奖励协议模块，集中定义 encoder 到 reward scheme 的映射、常数和纯函数。Python 训练、Python/SCOAP validation 与 C++ 原生 validation 显式携带 reward scheme，并按 scheme 处理事件。协议元数据记录 scheme，比较报告不再跨不同 scheme 比较 return。

**技术栈：** Python 3、PyTorch、pybind11/C++、`unittest`、CMake/pip editable build。

**规格：** `docs/superpowers/specs/2026-09-18-gat-backtrack-reward-design.md`

## 全局约束

- `level_gat_gru` 使用 `cubic_backtrack_v1`。
- `fanin_mean` 使用 `legacy_pi_exponential`，旧奖励公式不得截断或改写。
- 两个 encoder 的正式 `backtrack_limit` 都是 100。
- GAT 的第 B 次 backtrack 奖励为 `-(0.5 + 9.802960494 * (B / 100.0)^3)`。
- 不运行训练、benchmark、奖励对比实验或消融实验。
- 只运行定向单元测试、静态检查和必要的原生扩展构建验证。

---

### 任务 1：建立奖励协议纯函数

**文件：**
- 新建：`PODEM/python/rl_podem/smartatpg_rewards.py`
- 修改：`PODEM/python/rl_podem/__init__.py`
- 测试：`PODEM/tests/test_smartatpg.py`

**接口：**
- 产出：`reward_scheme_for_encoder(encoder_variant: str) -> str`
- 产出：`smartatpg_backtrack_reward(backtrack_count: int) -> float`
- 保留：`smartatpg_pi_reward(backtracks, pi_visits, alpha=7.5, beta=0.07) -> float`

- [ ] **步骤 1：编写失败测试**

增加测试，断言：

```python
self.assertEqual(reward_scheme_for_encoder("level_gat_gru"), "cubic_backtrack_v1")
self.assertEqual(reward_scheme_for_encoder("fanin_mean"), "legacy_pi_exponential")
self.assertAlmostEqual(smartatpg_backtrack_reward(1), -0.500009802960494)
self.assertAlmostEqual(
    sum(smartatpg_backtrack_reward(i) for i in range(1, 101)),
    -300.0,
    places=6,
)
for invalid in (0, 101):
    with self.assertRaises(ValueError):
        smartatpg_backtrack_reward(invalid)
```

- [ ] **步骤 2：运行测试确认失败**

运行：`python -m unittest PODEM.tests.test_smartatpg.SmartATPGTests.test_backtrack_reward_protocol`

预期：因新模块或函数尚不存在而失败。

- [ ] **步骤 3：实现纯函数**

在新模块中定义：

```python
BACKTRACK_MAX = 100
BACKTRACK_BASE = 0.5
BACKTRACK_POWER = 3
BACKTRACK_SCALE = 9.802960494
GAT_REWARD_SCHEME = "cubic_backtrack_v1"
MEAN_REWARD_SCHEME = "legacy_pi_exponential"

def reward_scheme_for_encoder(encoder_variant): ...
def smartatpg_backtrack_reward(backtrack_count): ...
def smartatpg_pi_reward(backtracks, pi_visits, alpha=7.5, beta=0.07): ...
```

从 `cpp_bridge.py` 移出旧 `smartatpg_pi_reward`，并在包入口重新导出，保持现有 import 兼容。

- [ ] **步骤 4：运行定向测试**

运行：`python -m unittest PODEM.tests.test_smartatpg.SmartATPGTests.test_backtrack_reward_protocol`

预期：通过。

### 任务 2：按 encoder 分流 Python 训练奖励

**文件：**
- 修改：`PODEM/python/rl_podem/cpp_bridge.py`
- 修改：`PODEM/scripts/train_smartatpg.py`
- 测试：`PODEM/tests/test_smartatpg.py`
- 测试：`PODEM/tests/test_smartatpg_training.py`

**接口：**
- `CppPodemBacktraceV2Trainer(..., reward_scheme: str | None = None)`；省略时从 `agent.encoder_variant` 推导。
- `_evaluate_fault(..., reward_scheme: str)` 按指定协议计算 return。

- [ ] **步骤 1：编写 GAT 与 mean 事件测试**

构造相同事件轨迹：一个 `backtrace_step`、两个 `backtrack`、一个 `pi_not_done` 和 detected terminal。断言：

```python
gat_return = 100.0 - 0.1 + sum(
    smartatpg_backtrack_reward(i) for i in (1, 2)
)
mean_return = 100.0 - 0.1 + smartatpg_pi_reward(2, 1)
```

同时断言 GAT 的 `pi_not_done` 不改变 return，mean 的 `backtrack` 不改变 return，并检查奖励加入对应 decision step。

- [ ] **步骤 2：运行测试确认失败**

运行：`python -m unittest PODEM.tests.test_smartatpg PODEM.tests.test_smartatpg_training`

预期：新 scheme 参数和 GAT backtrack 行为相关测试失败。

- [ ] **步骤 3：实现训练事件分流**

在 trainer 的 `episode_start` 清零 `_episode_backtrack_count` 和奖励组成。事件规则为：

```python
if event_type == "backtrack" and scheme == GAT_REWARD_SCHEME:
    self._episode_backtrack_count += 1
    reward = smartatpg_backtrack_reward(self._episode_backtrack_count)
elif event_type == "pi_not_done" and scheme == MEAN_REWARD_SCHEME:
    reward = smartatpg_pi_reward(...)
```

保留 backtrace-step、terminal 和 RND 现有语义，并把奖励组成加入 episode metrics。

- [ ] **步骤 4：实现 Python validation/SCOAP 分流**

让 `_evaluate_fault` 维护同样的 episode backtrack 计数，并显式接收 scheme。训练主流程根据 `args.encoder` 得到 scheme，传给 trainer、原生 validation 和 Python/SCOAP validation。

- [ ] **步骤 5：在汇总写入前检查有限数**

对 per-fault `return`、`atpg_seconds` 以及汇总 `return_total`、`return_mean` 使用 `math.isfinite`；错误信息包含 round、circuit 和 fault ID（适用时）。

- [ ] **步骤 6：运行 Python 定向测试**

运行：`python -m unittest PODEM.tests.test_smartatpg PODEM.tests.test_smartatpg_training`

预期：通过。

### 任务 3：让 C++ 原生 validation 与 Python 奖励一致

**文件：**
- 修改：`PODEM/src/python_bindings.cpp`
- 测试：`PODEM/tests/test_smartatpg.py`

**接口：**
- `run_native_validation(..., reward_scheme: str)` 新增必需参数。
- `NativeValidationPolicy` 构造函数验证 scheme，只接受 `cubic_backtrack_v1` 或 `legacy_pi_exponential`。

- [ ] **步骤 1：扩充原生一致性测试**

现有 native-vs-Python 测试分别以 GAT 和 mean scheme 运行，比较 fault ID、outcome、backtracks、backtrace steps 与 return。增加未知 scheme 拒绝测试。

- [ ] **步骤 2：运行测试确认失败**

运行：`python -m unittest PODEM.tests.test_smartatpg.SmartATPGTests.test_native_validation_matches_python_policy_without_callbacks`

预期：原生函数尚不接受 scheme 或 GAT return 不一致。

- [ ] **步骤 3：实现 C++ scheme 分流**

`NativeValidationPolicy` 为 GAT 在 episode 开始清零 `backtrack_count_`，在 `on_backtrack` 中计算三次 penalty；为 mean 保留 `on_pi_not_done` 的旧指数公式。两种 scheme 均保留 `on_backtrace_step` 和 terminal reward。

在写 journal、保存 record 和返回 Python 前使用 `std::isfinite` 检查 reward/seconds；GAT backtrack 超过 100 时抛出协议错误。

- [ ] **步骤 4：更新 pybind 接口和 Python 调用点**

给 `run_native_validation` 增加 `py::arg("reward_scheme")`，并从训练脚本传入 encoder 对应 scheme。

- [ ] **步骤 5：重新构建原生扩展**

运行：`python -m pip install -e PODEM`

预期：构建成功。

- [ ] **步骤 6：运行原生定向测试**

运行：`python -m unittest PODEM.tests.test_smartatpg`

预期：通过。

### 任务 4：统一 backtrack limit 和协议元数据

**文件：**
- 修改：`PODEM/scripts/prepare_smartatpg_training.py`
- 修改：`PODEM/scripts/run_smartatpg_training_linux.py`
- 修改：`PODEM/scripts/run_dual_smartatpg_training_linux.py`
- 修改：`PODEM/scripts/benchmark_smartatpg.py`
- 修改：`PODEM/scripts/run_smartatpg_benchmark_linux.py`
- 修改：`PODEM/scripts/prepare_smartatpg_benchmark.py`
- 修改：`PODEM/scripts/smartatpg_portable.py`
- 修改：`PODEM/python/rl_podem/cpp_bridge.py`
- 修改：`PODEM/src/rl_policy.cpp`
- 修改：`PODEM/scripts/train_smartatpg.py`
- 测试：`PODEM/tests/test_smartatpg.py`
- 测试：`PODEM/tests/test_smartatpg_training.py`
- 测试：`PODEM/tests/test_linux_smartatpg.py`

**接口：**
- 所有正式 SmartATPG 协议使用 `backtrack_limit=100`。
- checkpoint、validation identity 和模型导出元数据包含 `reward_scheme`。

- [ ] **步骤 1：把协议测试期望改为 100 并加入 scheme 断言**

更新正式训练/benchmark fixture 中的 200 为 100；不修改与 SmartATPG 正式协议无关的通用 ATPG 测试值。断言 GAT/mean metadata 分别携带正确 scheme。

- [ ] **步骤 2：运行协议测试确认失败**

运行：`python -m unittest PODEM.tests.test_smartatpg_training PODEM.tests.test_linux_smartatpg`

预期：生产常数和元数据仍为旧值，测试失败。

- [ ] **步骤 3：更新生产协议常数和校验**

把所有正式 SmartATPG 路径的 200 改为 100，包括 Python 常数、portable/native loader 的严格校验和 C++ 模型元数据校验。通用 ATPG 默认值 97、独立 SCOAP 测试值 2000 等不属于本协议，不修改。

- [ ] **步骤 4：传播 reward scheme 元数据**

训练 config、checkpoint protocol、validation identity、模型文本导出和加载校验都写入并验证 `reward_scheme`。旧 backtrack=200 或缺少 scheme 的新格式工件明确失败。

- [ ] **步骤 5：运行协议测试**

运行：`python -m unittest PODEM.tests.test_smartatpg PODEM.tests.test_smartatpg_training PODEM.tests.test_linux_smartatpg`

预期：通过。

### 任务 5：修正不同奖励协议的 validation 比较

**文件：**
- 修改：`PODEM/scripts/compare_smartatpg_validation.py`
- 测试：`PODEM/tests/test_linux_smartatpg.py`

**接口：**
- GAT 和 mean 的 identity 各自校验 encoder/scheme 映射，两者仍共享 backtrack limit 100。
- `COMPARABLE_METRICS` 排除 `return_total` 和 `return_mean`。

- [ ] **步骤 1：编写比较语义测试**

Fixture 为 GAT/mean 写入不同 scheme，断言比较成功；原始模型 validation metrics 仍含 return，但 `gat_minus_mean` 和 model-vs-SCOAP 比较行不含 return 差值或 SCOAP return 对比列。错误 encoder/scheme 组合必须失败。

- [ ] **步骤 2：运行测试确认失败**

运行：`python -m unittest PODEM.tests.test_linux_smartatpg.ValidationComparisonTests`

预期：现有 identity 字段和直接比较仍假定相同奖励语义，测试失败。

- [ ] **步骤 3：实现比较字段分离**

保留 `RAW_METRICS` 用于逐模型完整性校验；新增不含 return 的可比较字段集合供 `_comparison_row` 和 `_direct_row` 使用。输出 identity 分别保存两种 model scheme，不把 scheme 当作两模型必须相等的共享字段。

- [ ] **步骤 4：运行比较测试**

运行：`python -m unittest PODEM.tests.test_linux_smartatpg.ValidationComparisonTests`

预期：通过。

### 任务 6：最终定向验证

**文件：**
- 检查：本计划涉及的全部修改文件

- [ ] **步骤 1：运行代码差异检查**

运行：`git diff --check`

预期：无 whitespace 错误。

- [ ] **步骤 2：确认未遗留正式协议 200**

运行：`rg -n 'BACKTRACK_LIMIT = 200|backtrack_limit == 200|backtrack_limit != 200|\\(2, 8, 1, 200\\)' PODEM/scripts PODEM/python PODEM/src`

预期：无正式 SmartATPG 协议命中；通用或历史文档不在检查范围。

- [ ] **步骤 3：运行三个定向测试模块**

运行：`python -m unittest PODEM.tests.test_smartatpg PODEM.tests.test_smartatpg_training PODEM.tests.test_linux_smartatpg`

预期：全部通过。

- [ ] **步骤 4：确认没有执行实验**

检查命令历史和输出目录；不得出现训练、benchmark、消融或新模型工件。


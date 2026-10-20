# SWE 数据工厂工程总结

本文件把 **AweAgent（本仓库）作为 SWE 数据合成工厂**的工程要点整理成一份可直接执行的说明：
组件边界、Scale-SWE 生产配置、轨迹为什么"可训练"、以及换用本工厂时需要注意的地方。

结论先说：**本工厂把"可训练性"写进了脚手架与运行时**——学生原生工具格式（CodeAct-XML）、
环内 token 级 loss mask、隔离 evaluator 与判分前测试恢复、两套上下文压缩器。
此前本地那套采集器缺的正是这一层，导致"判分通过"与"样本可用"脱节；那套代码已从宿主项目移除，
只保留产物作为后续提效对比的基线（见第 5 节）。

---

## 1. 组件与职责

| 层 | 位置 | 职责 |
|---|---|---|
| 执行核心 | `aweagent/core/agent/` | `policy.py` 定义 `step(ctx) -> action`；`loop.py` 跑多步循环；`context.py` 是状态总线；`protocol.py` 定义接口 |
| 脚手架 | `aweagent/scaffold/` | `search_swe`（OpenHands 风格 CodeAct-XML，`enable_search: false` 即非搜索模式）、`deepsearch`、`iter_research`、`terminus_2` |
| 运行时 | `aweagent/core/runtime/` | Docker 后端；`configs/tasks/*.yaml` 里配置 `timeout` 与 `resource_limits` |
| 评测 | `aweagent/core/eval/` | `base.py` 的 `PatchTestEvaluator`（应用补丁→跑测试）；`isolation.py` 的 `IsolatedEvaluator`（**在全新容器里评测**）；`utils.py` 的 `restore_test_files`（判分前把测试文件恢复到 HEAD） |
| 上下文治理 | `aweagent/core/condenser/` | `tool_result_omission.py`（省略旧工具结果）、`truncation.py`（截断）；不需要自己写观察裁剪 |
| LLM 接入 | `aweagent/core/llm/` | `backends/`（openai / azure / ark / anthropic / sglang / openai_response）、`format/`（openai / xml / terminus_json）、`middleware/retry.py` |
| 训练态 | `aweagent/core/agent/training.py` | `TrainingState`：模型 token 记 `loss_mask=1`、环境观测记 `0`，并累积 logprob 与 `weight_version`；`to_rl_data()` 供训练框架消费 |
| 安全 | `configs/tasks/*.yaml` 的 `security.bash_blocklist` + 代码内默认 | 命令级黑名单；任务可追加按实例的 blocked patterns |
| 任务族 | `aweagent/tasks/` | `scale_swe`、`beyond_swe`、`denovoswe`、`nl2repo`、`swe_bench_pro`、`swebench_verified`、`terminal_bench_v2` 等 |

## 2. Scale-SWE 数据生产配置（以仓库内实际文件为准）

`configs/tasks/scale_swe.yaml`：

```yaml
runtime:
  backend: docker
  timeout: 14400          # 单实例 4 小时
  resource_limits: {cpu: "4", memory: "8Gi"}
agent:
  type: search_swe
  max_steps: 200          # 步数上限
  enable_search: false    # 非搜索模式 = OpenHands 风格
  bash_timeout: 1200      # 单条 bash 20 分钟
  tool_call_format: codeact_xml   # 直接产出学生格式
task: {type: scale_swe, dataset_id: scale_swe, data_file: ${DATA_FILE}}
eval: {enabled: true, timeout: 3600}
execution: {max_concurrent: 50, max_retries: 3, output_path: ./results/scale_swe}
```

读这张表的三个要点：

1. **`tool_call_format: codeact_xml`** —— 教师直接按学生的工具格式产出，不需要"function calling → XML"的离线转换。
2. **预算宽**：200 步、单条 bash 1200 s、单实例 4 h、并发 50。长任务不会被自己的上限判成失败。
3. **评测独立**：`eval.timeout: 3600`，且 `IsolatedEvaluator` 在全新容器里跑，agent 容器里的改动不会污染判分。

## 3. 运行方式

数据文件需自行下载（Hugging Face `AweAI-Team/Scale-SWE`），LLM 在 `configs/llm/` 配好。
`recipes/scale_swe/README.md` 给出的顺序是"先不碰 Docker，再单题，再批量"：

```bash
# 0) 预检：列出实例（不起容器）
python recipes/scale_swe/run.py --data-file /path/to/scaleswe.jsonl --mode dry-run

# 1) 看某题真正送给模型的 prompt 与工具格式
python recipes/scale_swe/run.py --data-file /path/to/scaleswe.jsonl \
    --instance-id <ID> --mode prompt

# 2) 单题调试（起容器）
python recipes/scale_swe/run.py --data-file /path/to/scaleswe.jsonl \
    --instance-id <ID> --mode debug --verbose

# 3) 批量
python recipes/scale_swe/run.py --data-file /path/to/scaleswe.jsonl --mode batch
# 或指定若干实例
python recipes/scale_swe/run.py --data-file /path/to/scaleswe.jsonl --mode batch \
    --instance-ids inst_001 inst_002
```

前置条件：Docker 可用、LLM API 可调、数据文件就位。`--mode prompt` 是零成本检查工具格式与提示词的最快方式，**批量前务必先跑一次**。

## 4. 为什么这套轨迹天然"可训练"

| 机制 | 位置 | 效果 |
|---|---|---|
| 学生原生格式 | `tool_call_format: codeact_xml` + `core/llm/format/xml.py` | 轨迹即训练序列，无转换层 |
| 环内 loss mask | `core/agent/training.py` | 模型 token=1、观测 token=0，逐 token 累积 |
| logprob 与权重版本 | 同上 | 可直接供 RL（on-policy 校正） |
| 隔离 evaluator | `core/eval/isolation.py` | 判分环境与 agent 环境分离 |
| 判分前恢复测试 | `core/eval/utils.py: restore_test_files` | 改测试无法换通过 |
| 上下文治理 | `core/condenser/` | 长轨迹不爆窗口，也不需要手工裁剪观察 |
| 命令黑名单 | `security.bash_blocklist` | 阻断历史/网络取答案路径 |

## 5. 与我们此前本地工厂的对照基线（保留产物，供提效对比）

宿主项目 `local_train` 里原来的本地采集器（独立 judge、资格验证、多次尝试、Qwen 模板导出）**代码与文档已移除**，
只保留运行产物与汇总，位置：`local_train/storage/sft_runs/`（含 `baseline-2026-10-20/` 汇总副本）。

保留这些数字的用途只有一个：**换成本工厂后对比产量与干净率**。它们不代表本工厂的表现。

| 指标（本地旧采集器，16 道 Scale-SWE 抽样题） | 实测 |
|---|---|
| 单次尝试：干净可训练样本 | **3 / 16** |
| 3 次尝试（取首个干净样本） | 7 / 16，41 次尝试，354,068 token |
| 加上"禁止改测试/夹具/快照"一句约束 | 5 道原污染题 **5/5 首次干净**，4,025 token/条（原 50,581） |
| 主要缺陷 | 11 条 accepted 里 7 条轨迹写过测试文件；1 条 63,697 token 超训练窗口被拒 |
| 预算 | 60 步 / bash 120 s / 并发 10 |

对照结论：本地旧采集器的"质量低"主因是**缺数据契约**（未禁止改测试、未按可训练性准入），
而本工厂把这些约束放在脚手架与运行时里，这是换用本工厂的主要理由。

## 6. 在本工厂上产出 SFT 干净数据的建议

1. **先 `--mode prompt` 核对工具格式与提示词**，确认与目标学生模型的 tokenizer/模板一致，再 `--mode batch`。
2. **明确禁止修改测试文件**。本地实测这条约束能把"污染样本"变成"首次即干净"，成本同时降一个量级；
   若任务确实需要更新 fixture，应在资格验证阶段标记为不可解，而不是让模型改测试。
3. **限制同题可接受样本数**（例如每题最多 1–3 条），避免简单题重复采样拉偏分布。
4. **按学生窗口反推长度**：单条序列若接近学生训练窗口上限，应在导出前显式过滤，而不是训练时报错。
5. **确认数据来源与评测集不重叠**：Scale-SWE 是训练集，SWE-bench Verified 等是测试集，不要混用。

## 7. 未核实的部分（避免误用）

- 本文件依据仓库源码与配置整理，**未实际运行本工厂的批量管线**；吞吐、成本、成品率均未在此测过。
- 多可接受样本上限、去重策略、长轨迹过滤在本仓库中的实现细节未逐一核对，第 6 节第 2–4 条是按工程需要提出的建议，不代表仓库现有行为。
- `restore_test_files` 只保证判分不受测试改动影响，**不自动剔除"轨迹里改过测试"的样本**；若要 SFT 干净数据，仍需在导出侧按轨迹过滤或在前置提示里禁止。

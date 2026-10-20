# DSH 工具面对齐工程报告

日期：2026-10-20（Asia/Shanghai）。目标：把本工厂的 agent 脚手架改成**与 DSH 评测端逐字对齐**，
使产出的轨迹能被同一个学生模型直接消费。所有改动都是新增适配层，未改 core。

## 0. 结论

**三层全部对齐并通过验证：5 道题 4/5 accepted（原脚手架 2/5），每条 SFT 样本 18,293 token（原来 34,082，超窗）。**

| 指标 | 原脚手架（finish + str_replace_editor） | 对齐后（DSH 三工具） |
|---|---:|---:|
| 5 题通过 | 2/5 | **4/5** |
| 每样本 token | 34,082（> 32,768 窗口） | **18,293（窗内）** |
| 工具面 | bash / str_replace_editor / think / finish | **bash / read / edit** |
| 终止 | 显式 finish 工具 | **自然停止** |
| 工具调用形态 | 文本内嵌 DSML 标记 | **原生 `tool_calls` + Qwen 模板渲染** |

## 1. 对齐的三层与实现

| 层 | DSH 侧（来源） | 本工厂实现 |
|---|---|---|
| 工具 schema | `packages/shell/tool-bash/src/index.ts`（`command` + `description` 均必填）；`packages/fs/tool-fs/src/read.ts`（`file_path`、`offset` 默认 1、`limit` 默认 400 / 上限 2000）；`edit.ts`（`file_path`、`old_string`、`new_string`、`replace_all` 默认 false，多匹配拒绝） | `aweagent/scaffold/dsh_compat/tools.py` 三个 Tool 子类，参数名、必填项、默认值逐字照抄 |
| 调用格式 | vLLM `tool parser = qwen3_xml`，模型输出 `<tool_call><function=...>` | `tool_call_format: openai_function`，原生 `tool_calls`；导出时用学生 Qwen 模板渲染成同样的 `<tool_call>` 块 |
| 终止语义 | 无 `finish`，`stopReason=stop` 且无工具调用即结束 | 移除 `finish` 与 `think`；`get_no_tool_call_prompt()` 返回 `None`（与 v3s 关闭 noEditStop 一致） |

配套改动：

- `aweagent/scaffold/registry.py`：注册 `dsh_compat`；
- `recipes/scale_swe/run.py`：按 `agent.type` 分派（debug 与 batch 都走注册表）；
- `configs/tasks/dsh_scaleswe.yaml`：`type: dsh_compat`、`bash_timeout: 120`、`bash_max_timeout: 600`、`tool_call_format: openai_function`；
- `configs/llm/local_deepseek.yaml`：本机 DeepSeek 配置（非思考）。

提示词由两段组成：**评测 persona 原文**（从 `eval/profiles/swe-agent-v3s/cordis.patch.yml` 逐字复制）+ **数据契约**（禁止修改测试/夹具/快照；改完要跑目标测试）。两者分开标注，避免把数据侧约束混进评测口径。

## 2. 验证证据（先验证，再花钱）

### 2.1 schema 对齐（不调用教师）

```
tools = [bash, read, edit]
bash: required=[command, description]
read: required=[file_path], optional=[offset, limit]（描述注明默认 400、上限 2000）
edit: required=[file_path, old_string, new_string], optional=[replace_all]（默认 false）
禁止项检查：不存在 finish / think / str_replace_editor / execute_bash
```

### 2.2 格式往返（学生模板）

用生产环境的 tokenizer 与 `chat_template.jinja` 渲染一条含两次工具调用的对话：

```
rendered 563 chars / 140 tokens / 2 个 <tool_call> 块 / DSML 标记 0 个
```

断言：渲染结果包含 `<function=bash>`、`<parameter=command>`、`<function=edit>`、`<parameter=file_path>`，
且分词 → 解码往返后这些结构仍在。**这条是"数据不会废掉"的关键检查**：工厂过去输出的是自己的 DSML 标记，
vLLM 的 `qwen3_xml` 解析不到。

### 2.3 5 题真实运行

| instance | 通过 | 步数 | 补丁字符 | 工具调用 |
|---|---|---:|---:|---|
| adamtheturtle_sybil-extras_pr296 | ✅ | 10 | 1,108 | bash 8 / edit 1 |
| asottile_blacken-docs_pr2 | ✅ | 14 | 2,954 | bash 13 |
| asyncgui_asynckivy_pr121 | ✅ | 19 | 4,800 | bash 18 |
| assertpy_assertpy_pr117 | ✅ | 45 | 11,099 | bash 44 |
| atlassian_asap-authentication-python_pr2 | ❌ | 12 | 5,886 | bash 11 |

- **聚合工具分布：bash 94 / edit 1**，无 `finish` 调用（`tool_usage` 里没有该键）；
- 轨迹是**结构化**的：assistant 带 `tool_calls`（`function.name` + JSON `arguments` 字符串），tool 消息带 `content`/`name`/`tool_call_id`；
- 5 题 0 error。

### 2.4 SFT 转换与长度

| 指标 | 值 |
|---|---:|
| 可用样本 | 4（1 条失败被丢弃） |
| 序列 token 合计 | 73,171 |
| 监督 token | **16,698（22.8%）** |
| 每样本 | **18,293 token / 4,175 监督** |
| 块数 | 180（连续 tool 合并为一个块） |

**每样本 18,293 已落在 32,768 窗口内**，这是相对旧脚手架（34,082）最实际的改善之一。

## 3. 一处必须记录的实现取舍

`read` / `edit` 没有走 shell 内嵌 Python，而是用 `session.download_file` / `upload_file` 读写：
避免 shell 转义问题，并精确保留字节（`edit` 的 `old_string` 唯一匹配、`replace_all` 语义与 DSH 一致）。

代价：批量运行时 `agent_factory` 在建 agent 时拿不到 per-instance 的 workdir，所以路径限制放宽为
"绝对路径且位于 `/workspace` 下，拒绝 `..`"。单实例运行（debug）仍用精确 workdir 前缀检查。

## 4. 已知限制

1. **执行环境仍不同**：本工厂的 bash 跑在 Docker 容器，DSH 的 `bash-local` 跑在宿主工作区。工具名与参数一致，但路径、可用命令、超时来源不同。
2. **`edit` 几乎没被用**（94 次 bash vs 1 次 edit）：模型偏好用 shell 改文件。这与评测侧"bash 占 91%"的观察一致，但如果要教 `edit`，需要单独配方，不能靠当前数据。
3. **只跑了 5 题、单模型**（`deepseek-chat` 非思考）、单次运行；4/5 不能当成品率。
4. `bash` 的 `timeout` 参数**没有暴露给模型**（与 DSH 一致：模型只给 `command` + `description`），工厂内部按 120 s / 上限 600 s 截断。
5. 未验证 `atlassian` 那条失败是早退还是真改错；也未做池 B（纠错对）导出。

## 5. 复现

```bash
# 1) 工具面对齐 + 格式往返（零成本，不调教师）
python results/verify_dsh_alignment.py

# 2) 5 题真跑（新脚手架）
python results/run_factory.py --data-file <clean.jsonl> \
    --config configs/tasks/dsh_scaleswe.yaml --llm-config configs/llm/local_deepseek.yaml \
    --mode batch --instance-ids <ID...> --max-steps 60 --max-concurrent 5 --output ./results/dsh_pilot

# 3) 转 SFT（Qwen 模板 + assistant-only mask）
python results/to_sft_qwen.py --run results/dsh_pilot/<run_dir> --out results/dsh_pilot/<run_dir>/sft_qwen.jsonl
```

产物：`results/dsh_pilot/deepseek-chat_20261020_230801/`（`results.jsonl`、`trajectories.jsonl`、`sft_qwen.jsonl`）。

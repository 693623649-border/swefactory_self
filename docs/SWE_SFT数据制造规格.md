# SWE SFT 数据制造规格

本文件把"从能力缺口反推 SFT 数据规格"的思路落到本工厂上：**要什么数据、怎么用本工厂产、产出来怎么验收**。
配套阅读：[SWE数据工厂工程总结](SWE数据工厂工程总结.md)（组件与配置）。

证据分两段，不要混用：

- **缺口证据**来自 `local_train` 的 SWE 诊断十题（Astropy，v3 续写协议）：3/10 resolved、8/10 有内容补丁、4/10 自然结束；以及 LCB 分层 30 题的双预算臂。属于单仓、已调参的诊断集，**方向可信、量级别外推**。
- **产出证据**来自本仓库 2026-10-20 的 pilot：5 道 Scale-SWE 题、`max_steps=60`、并发 5，**2/5 accepted、0 error**。

---

## 1. 缺口 → 数据要求

| 缺口 | 证据 | 对数据的要求 |
|---|---|---|
| 语义判断错（改动方向错） | 8 条有补丁的轨迹只对 3 条；一题用 `FutureWarning` 保留旧行为、一题 +222 行但目标测试全败 | 池 A 必须是"改对且最小 diff"的完整轨迹；池 B 必须示范"错了之后按测试反馈改方向" |
| 不交付（只写不给） | 2/10 补丁零代码，3,130–3,239 token/步纯独白；LCB 交付 fence 仅 30–33% | 池 C 的落地样本：分析 ≤3 步必须出现文件修改 |
| 无验证闭环 | 一题唯一一次 edit 在最后一步、改完没跑测试；两题合计弄红 50 项 P2P | 池 A/C 要求"edit 后必须跑目标测试，通过后再跑一次相关套件" |
| 停止判断差 | 5/10 截断终止全部失败且用满 12 次续写；一题 327 步/321 次 bash 反复重试 | 池 C 的及时停样本 + 诚实失败样本 |
| 工具经济性差 | bash 占 91%；改仓库测试 4 题（判分时被覆盖，纯浪费） | 准入硬门：不碰测试文件；统计工具调用分布 |
| 预算不是主因 | LCB 32K 臂：输出上限翻倍，交付 +1、通过 +1、token +77%，hard 仍 0/10 | 先把数据做对，再谈调预算 |

## 2. 三类池（按项目现有池名）

### 池 A · verified_sft（目标占比 50–60%）

单条 = issue 文本 + 仓库快照 → 轨迹 → 补丁 → **容器内真实执行收据**（目标测试过 + 相关套件无回归）。

轨迹要示范的形状（成功轨迹的共性）：

1. 少量定位读取（`ls`/`grep`/`read`），不漫游；
2. 最小 diff 的编辑（±几行到几十行，不是 +222 行大改）；
3. 编辑后**立即跑目标测试**；
4. 通过后跑一次相关回归；
5. 简短总结并**停止**。

### 池 B · correction_sft（目标占比 25–30%）

成对样本：第一轮犯**真实类型**的错（语义反了 / 破坏既有测试 / 只修入口），工具观察暴露失败，下一轮修正到全绿。

- 监督只算修正轮的 assistant 输出；第一轮的错误输出作为上下文，`loss=0`——否则在教模型犯错。
- 三类必配：① 跑套件发现回归 → 收窄改动；② 端到端验证失败 → 补全功能；③ issue 语义歧义 → 对照测试反馈修正方向。
- **本工厂怎么产**：同一 `instance_id` 用 `max_retries` 跑多次，把失败轮（`success=false` 且有补丁）与成功轮配对。工厂默认每轮独立保存轨迹，配对在导出侧完成。

### 池 C · 行为纪律（目标占比 15–20%）

- **落地样本**：分析 ≤3 步就出现文件修改（反独白）；
- **及时停样本**：测试通过且 diff 非空 → 一句话总结立即结束（反循环）；
- **诚实失败样本**：不可解时以空补丁或说明收尾，不为交差乱改。

这三类都可以从轨迹里**按规则筛**（首个 edit 的位置、测试与结束的先后、是否空补丁），不需要单独采集。

## 3. 横切验收门（缺一即弃）

| 门 | 标准 | 对应缺口 |
|---|---|---|
| 真实执行 | 每条有容器内测试收据（`eval_result.accepted` / `score`），禁止伪造 | 全部 |
| 内容 diff | 补丁 ≥1 个代码 hunk，0 条 mode-only | 不交付 |
| 验证闭环 | edit 之后必有测试执行，且通过后才结束 | 验证闭环 |
| 无回归 | P2P 全绿（工厂 `eval_result` 已含 f2p/p2p 计数） | 验证闭环 |
| 不碰测试文件 | 轨迹与补丁都不含测试文件写入 | 工具经济性 |
| 轨迹长度 | 步数中位 30–60、上限 ~100；token/步 100–500 | 停止判断 |
| 无重复 | 同一命令/同一 diff 不连续出现 ≥3 次 | 停止判断 |
| 隔离 | 不含 SWE-bench Verified / LCB 的 ID 与近重复；按家族分 split | — |

## 4. 本工厂的产出契约（实测）

一次批量跑（`--mode batch`）落三个文件到 `execution.output_path/<model>_<时间>/`：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 每题一行：`instance_id / success / score / finish_reason / error` |
| `trajectories.jsonl` | 每题一行：`messages`（扁平 `role`+`content`）、`patch`、`stats`（步数、工具分布、耗时）、`eval_result` |
| `run_config.json` | 本次配置快照 |

关键点：**`messages` 里的工具调用已经内嵌在 assistant 文本中**（CodeAct-XML），观测在 user 消息里。
所以 SFT 不需要再套模板转换——`assistant` 即监督段、其余为上下文，与"教师-学生同构"的要求一致。

转换脚本见 `results/to_sft.py`，输出的每条样本形如：

```json
{"instance_id": "...", "messages": [{"role": "...", "content": "..."}],
 "labels": [0, 1, ...], "score": 1.0, "steps": 47,
 "stats": {"tokens": ..., "supervised_tokens": ...}}
```

## 5. Pilot 实测（2026-10-20）

| 项 | 值 |
|---|---|
| 任务 | 5 道 Scale-SWE（本地镜像已就绪） |
| 配置 | `deepseek-chat`、`max_steps=60`、并发 5、`codeact_xml` |
| 结果 | **2/5 accepted**（`atlassian_asap-authentication-python_pr2`、`assertpy_assertpy_pr117`），0 error |
| 单条样例 | 47 步、96 消息、补丁 14,402 字符、f2p 16 / p2p 2、65.8 秒 |
| SFT 转换 | 2 条样本、210 消息、68,164 token、**监督 19,113（28.0%）** |
| 每样本 | 34,082 token / 51.5 个 assistant 回合 |

两个必须记录的发现：

1. **步数不是越多越好**。`asottile_blacken-docs_pr2` 在 `max_steps=30` 时 resolved（12/12 通过），改用 60 步反而失败——多出来的步数带来多余改动与回归。与"预算不是主因"的结论一致。
2. **长度门会咬人**。每样本 34,082 token，已超过 32,768 窗口。要么把学生窗口提到 40k，要么在导出时按窗口截断并保证 mask 与截断对齐。这是必须在采大批之前定的前置。

## 6. 运行配方（Windows 上实测可用）

工厂默认跑 Linux；在本机 Windows 需要四处环境修正，**不改工厂代码**：

| 问题 | 现象 | 处理 |
|---|---|---|
| 默认编码 | 读数据文件报 `'gbk' codec can't decode` | 启动前设 `PYTHONUTF8=1`（进程内设无效，必须解释器启动前） |
| 代理模式 | `httpx.InvalidURL: Invalid port: ':1]'` | 宿主 `NO_PROXY` 含 IPv6 项；在导入 httpx 前把 `NO_PROXY/no_proxy` 改为 `localhost,127.0.0.1`（见 `results/run_factory.py` 包装层） |
| 数据尾部 | 样本最后一行 JSON 被截断，加载即崩 | 先剔除非法行生成干净 JSONL（158 行） |
| LLM 配置 | 默认走 `gpt-5.4` | 用 `--llm-config configs/llm/local_deepseek.yaml`（本仓库新增，`deepseek-chat`、非思考） |

```bash
# 0) 依赖（只用本仓库 venv）
uv sync --frozen

# 1) 零成本预检
python results/run_factory.py --data-file <clean.jsonl> --llm-config configs/llm/local_deepseek.yaml --mode dry-run
python results/run_factory.py --data-file <clean.jsonl> --llm-config configs/llm/local_deepseek.yaml --mode prompt --instance-id <ID>

# 2) 单题真跑
python results/run_factory.py --data-file <clean.jsonl> --llm-config configs/llm/local_deepseek.yaml \
    --mode debug --instance-id <ID> --max-steps 30

# 3) 批量（才会落 trajectories.jsonl）
python results/run_factory.py --data-file <clean.jsonl> --llm-config configs/llm/local_deepseek.yaml \
    --mode batch --instance-ids <ID...> --max-steps 60 --max-concurrent 5 --output ./results/pilot

# 4) 转 SFT（assistant-only mask + token 统计）
python results/to_sft.py --run results/pilot/<run_dir> --out results/pilot/sft.jsonl
```

## 7. 规模与顺序

原分析的建议是"每池都过全部验收门"再凑 1,000–2,000 条，pilot 用 A 500 / B 400 / C 300。落到本工厂：

1. **先定长度门与窗口**（第 5 节第 2 点），否则采完才发现要重切；
2. **先冻结工具面**：本工厂是 `codeact_xml` + `execute_bash`/`str_replace_editor`/`finish`，学生服务端必须用同一套 schema 与模板；
3. **池 A 先行**：`--mode batch` 直接产；验收用 `results.jsonl` 的 `success/score` + 第 3 节的门；
4. **池 B 靠同题重试配对**：失败轮与成功轮成对导出，失败轮 `labels=0`；
5. **池 C 靠轨迹筛选**：不新增采集，只按规则选。

按 pilot 的 2/5 = 40% 通过率粗算，凑 500 条池 A 约需 1,250 次尝试；这是**单批粗估**，未计入难度分层与去重。

## 8. 未验证与限制

- pilot 只有 5 题、单模型（`deepseek-chat` 非思考）、单次运行；40% 不能当题库成品率。
- 池 B 的配对导出尚未实现（本文件给的是可行做法，不是已跑通的结果）。
- 工厂的 `training.py`（token 级 loss mask + logprob）只在 SGLang 训练模式下生效；本次批量产出的轨迹**不含** loss mask，mask 由 `to_sft.py` 在导出侧重建。
- `str_replace_editor` 与 `finish` 是本工厂脚手架的工具名；若换成我们评测端的 `bash/read/edit`，必须重新采集——**不要混用两套工具面的轨迹训练**。

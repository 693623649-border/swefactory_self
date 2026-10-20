# 多轮 token 构成与低思考蒸馏

日期：2026-10-20（Asia/Shanghai）。对象：`dsh_compat` 脚手架跑出的两批轨迹——非思考（`deepseek-chat`）与低思考（`deepseek-flash`，effort=low）。

## 0. 结论

1. **多轮对话里 72% 的 token 是上下文，不是监督信号**；工具观察单独占 69.5%。
2. **平均 token 会骗人**：非思考批 4 条样本平均 18,293，但最长一条 **47,154**，超 32,768 窗口。窗口问题只能按样本看尾部。
3. **低思考可用且更好**：3/3 通过，每样本 15,201 token，最长 26,418（窗内）；思考占 assistant 输出的 **33.8%**。
4. 过程中发现并修掉一个**会静默废掉数据**的 bug：工厂把思考存成 `reasoning_raw`，而 Qwen 模板只认 `reasoning_content`，不映射就会把思考整段丢掉（监督 token 10,996 → 6,867，**丢 37.6%**）。

## 1. 多轮 token 构成（非思考批，5 题 100 轮）

| 部分 | token | 占比 | 是否参与训练 |
|---|---:|---:|---|
| assistant 输出（含 tool_calls 参数） | 24,267 | 27.6% | ✅ loss=1 |
| 工具观察 | 61,188 | 69.5% | ❌ loss=0 |
| system | 885 | 1.0% | ❌ |
| user（issue 文本） | 1,712 | 1.9% | ❌ |
| 合计 | 88,052 | 100% | 监督仅 27.6% |

上下文增速（每题结束时累计）：

| 题 | 轮数 | assistant | tool | 末轮上下文 |
|---|---:|---:|---:|---:|
| sybil-extras_pr296 | 10 | 1,582 | 3,245 | 5,239 |
| blacken-docs_pr2 | 14 | 2,601 | 5,205 | 8,201 |
| asyncgui_asynckivy_pr121 | 19 | 4,375 | 9,708 | 14,561 |
| atlassian_asap_pr2 | 12 | 4,894 | 7,477 | 12,900 |
| assertpy_assertpy_pr117 | 45 | 10,815 | 35,553 | **47,151** |

**结论**：轮数一多，工具观察线性堆积，45 轮的题直接顶穿 32,768。长度门必须按样本尾部卡，不能看均值。

## 2. 低思考（effort=low）实测

配置：`deepseek-flash` + `thinking: {type: enabled}` + `reasoning_effort: low`，`reasoning.preserve: true`（带工具时必须回放历史思考，否则工具循环退化），其余与 DSH 对齐批一致（`max_steps=60`、并发 3）。

| 题 | 非思考 | 低思考 |
|---|---|---|
| sybil-extras_pr296 | 5,242 tok / 10 步 / 通过 | 12,719 tok / 14 步 / 通过 |
| blacken-docs_pr2 | 8,204 tok / 14 步 / 通过 | 7,239 tok / 10 步 / 通过 |
| asyncgui_asynckivy_pr121 | 14,562 tok / 19 步 / 通过 | 26,418 tok / 17 步 / 通过 |

- 低思考批：**3/3 通过**，合计 45,603 token，监督 10,996（24.1%），每样本 15,201，**最长 26,418，无超窗**。
- 思考 token 4,094，占 assistant 输出（思考+正文+工具参数）的 **33.8%**——不是"想很久"，是"想一点就动手"。
- 单题有增有减：`blacken-docs` 反而更省（10 步 vs 14 步），另两题更长。样本太小，不下"普遍更省"的结论。

## 3. 发现并修复的 bug：思考被静默丢弃

工厂把模型思考写进 `reasoning_raw`，而学生模板（`chat_template.jinja`）只在 `reasoning_content` 存在时才渲染 `<think>`。导出器最初只读 `content`，于是：

| 版本 | 监督 token | 后果 |
|---|---:|---|
| 未映射 | 6,867（16.6%） | 训练的是"不思考直接动手"的模型，与低思考目标相反 |
| 映射后 | 10,996（24.1%） | `<think>` 块进入序列，思考成为监督信号 |

修复：导出器在渲染前把 `reasoning_raw` 映射为 `reasoning_content`。**这条必须固化成校验**（见第 5 节）。

## 4. 为什么低思考更适合本批数据

- 学生是 thinking 模型，评测端模板会渲染 `<think>`；若教师轨迹没有思考，学生学到的是"空 think + 直接动作"，推理时行为不一致。
- 思考占比 33.8%（而非 60%+ 的长思考），与"先想一点、尽快落地"的验收门（分析 ≤3 步要出现文件修改）方向一致。
- 低思考批无超窗样本，长度压力更小。

仍需注意：低思考也会让**思考成为监督信号的一部分**，因此导出时必须确认 `<think>` 非空且被 mask 覆盖，否则等于既没教思考、又占了上下文。

## 5. 建议（按优先级）

| 优先级 | 改动 | 理由 |
|---|---|---|
| P0 | 导出前自动断言：每条样本至少一个非空 `<think>` 块，且落在 loss=1 | 本次 bug 就是漏了这条 |
| P0 | 长度门按样本尾部：>32,768 直接丢弃并记录 | 非思考批已有 47,154 的样本 |
| P1 | 默认教师改为低思考配置 | 3/3、无超窗、思考占比 33.8% |
| P1 | 训练侧按 mask 归一化损失 | 上下文占 72%，不归一化会稀释监督 |
| P2 | 统计每轮上下文增速，对 >60 轮的任务预筛 | 45 轮即顶窗 |

## 6. 复现

```bash
# 低思考批量
python results/run_factory.py --data-file <clean.jsonl> --config configs/tasks/dsh_scaleswe.yaml \
  --llm-config configs/llm/local_deepthink.yaml --mode batch --instance-ids <ID...> \
  --max-steps 60 --max-concurrent 3 --output ./results/dsh_think

# token 构成
python analysis/../results/analyze_turns.py dsh_pilot

# 转 SFT（已含 reasoning_raw -> reasoning_content 映射）
python results/to_sft_qwen.py --run results/dsh_think/<run_dir> --out results/dsh_think/<run_dir>/sft_qwen.jsonl
```

## 7. 限制

- 低思考只有 3 题、单次运行，`blacken-docs` 上更省不能外推。
- 非思考批 5 题、低思考批 3 题，两批题目不完全重合，同题对照只有 3 条。
- 未测 effort=high 或 thinking_budget 的对照，也未验证低思考对学生最终能力的影响（需要训练后才能回答）。

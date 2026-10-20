# 池 A 规模化与非思考/低思考的长度账

日期：2026-10-20（Asia/Shanghai）。本轮：用低思考配置把池 A 规模做起来，同时启用 `<think>` 断言与长度门。

## 0. 结论

**22 题跑出 14 条通过（63.6%），但只有 7 条能进 32,768 窗口——拦住数据的是长度，不是通过率。**

| 阶段 | 数量 | 说明 |
|---|---:|---|
| 计划（已剔除 2 道已知不可解） | 22 | doccmd、slotscheck 事先排除 |
| 教师通过（`success=true`） | **14** | 8 条失败，0 error |
| 通过但超 32,768 | **7** | 中位 39,506，最长 66,619 |
| **可导出 SFT** | **7** | 84,011 token，监督 23,595（28.1%） |

## 1. 长度分布（14 条通过样本）

| 题 | 步数 | token | 是否超窗 |
|---|---:|---:|---|
| betodealmeida_python-rison_pr7 | 13 | 9,148 | 否 |
| app-sre_qontract-reconcile_pr3 | 12 | 4,892 | 否 |
| astropy_astroplan_pr535 | 9 | 7,178 | 否 |
| app-sre_qontract-reconcile_pr300 | 13 | 14,953 | 否 |
| beetbox_beets_pr5356 | 9 | 8,810 | 否 |
| appium_python-client_pr446 | 10 | 13,769 | 否 |
| arviz-devs_preliz_pr670 | 32 | 26,766 | 否 |
| beetbox_beets_pr3661 | 22 | 39,506 | **是** |
| arviz-devs_preliz_pr182 | 24 | 41,104 | **是** |
| andialbrecht_sqlparse_pr330 | 30 | 43,676 | **是** |
| beetbox_beets_pr4386 | 41 | 45,902 | **是** |
| asottile_add-trailing-comma_pr166 | 26 | 47,683 | **是** |
| asottile_add-trailing-comma_pr13 | 38 | 47,742 | **是** |
| app-sre_qontract-reconcile_pr2162 | 43 | 66,619 | **是** |

**分界在 ~30 步**：≤32 步的样本基本落在窗内，≥38 步的一律超窗。根因是工具观察堆积（占总 token 70% 左右），不是思考变长。

## 2. 导出器两道门的实际表现

| 门 | 拦截数 | 说明 |
|---|---:|---|
| 长度门（>32,768 丢弃并计数） | 7 | 全部来自 ≥22 步的样本 |
| `<think>` 断言 | 0（修正后） | 见第 3 节 |
| 失败样本 | 8 | `success=false`，未进入转换 |

导出结果：7 条、84,011 token、监督 23,595（28.1%）、每样本 12,002 token / 3,371 监督。

## 3. 又发现一个"过严即全废"的 bug

第一版 `<think>` 断言写成"渲染文本里**不能出现**空 think 块"，结果 14 条全被拒（`no_think=7` + `over_length=7`）。

根因：模板对**没有 reasoning 的那几轮**也会渲染 `<think>\n\n</think>`；而一条轨迹往往有 1–3 轮没有思考（例如 6/9、30/32）。要求"全轮都有思考"等于要求整条轨迹零空块——过严。

修正为"**至少一个非空 think 块**"后，7 条正常导出，`no_think` 归零。这条要记住：数据质量门写成"全称命题"时，先数一下分母。

## 4. 决策点：窗口还是步数

现状是"通过率高、但一半样本超窗"。两条路：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 提窗口到 65,536** | 学生模型 `max_position_embeddings=262144`，技术上可行；训练显存与吞吐按序列平方增长 | 需要重跑训练 smoke 确认显存 |
| **B. 压步数到 ~30** | `max_steps=30`，按上面分布可把绝大多数样本压进 32K | 会长任务在解出前被截断，`preliz_pr249` 这类需要 60 步的会掉 |

建议先做 B 的小批对照（同 22 题、`max_steps=30`），同时用 A 的 65K 配置各跑一次，比较"可导出条数 / 每条监督 token"，再决定默认口径。**不要现在就按 32K 硬砍**——那等于用长度门替模型决定哪些任务能学。

## 5. 已知限制

1. 22 题里 8 条失败未逐题归因；其中 `preliz_pr249` 跑满 60 步仍只产出 335 字符补丁，疑似早退。
2. 只跑了低思考一档，没有非思考的同题对照（上一轮只有 3 题对照）。
3. 长度用 Qwen tokenizer 估算整段渲染，不含训练时的 padding 与模板差异。
4. `<think>` 断言目前只验证"至少一处非空"，未验证思考与动作的对应关系。

## 6. 复现

```bash
python results/run_factory.py --data-file <clean.jsonl> --config configs/tasks/dsh_scaleswe.yaml \
  --llm-config configs/llm/local_deepthink.yaml --mode batch --instance-ids <22 ids> \
  --max-steps 60 --max-concurrent 5 --output ./results/pool_a
python results/to_sft_qwen.py --run results/pool_a/<run_dir> --out results/pool_a/<run_dir>/sft_qwen.jsonl
```

产物：`results/pool_a/deepseek-flash_20261020_232511/`（`results.jsonl`、`trajectories.jsonl`、`sft_qwen.jsonl`）。

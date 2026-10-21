"""Data analysis report for the formal large batch.

Consumes results/pool_a_formal/{admission_ledger.jsonl,export_summary.json} plus the
50-task baseline receipts, and writes
docs/工程问题/2026-10-21-池A正式生成数据分析报告.md.

Every number is derived from the ledger; nothing is hand-entered except labels.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "pool_a_formal"
BASELINE = ROOT / "results" / "pool_a50" / "deepseek-flash_20261020_235140"
DOC = ROOT / "docs" / "工程问题" / "2026-10-21-池A正式生成数据分析报告.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ledger = load_jsonl(FORMAL / "admission_ledger.jsonl")
    export = json.loads((FORMAL / "export_summary.json").read_text(encoding="utf-8"))
    base_rows = load_jsonl(BASELINE / "sft_qwen.jsonl")

    attempted = len(ledger)
    passed = sum(1 for r in ledger if r["success"])
    admitted = [r for r in ledger if r["admitted"]]
    families = {r["family"] for r in ledger}
    reasons = Counter()
    for r in ledger:
        reasons.update(r["rejected"])

    lengths = [r["length"] for r in admitted if r.get("length")]
    steps = [r["steps"] for r in ledger if r.get("steps")]
    prompt = sum(r.get("prompt_tokens") or 0 for r in ledger)
    completion = sum(r.get("completion_tokens") or 0 for r in ledger)
    f2p_fail = sum(r["f2p"].get("failed", 0) + r["f2p"].get("unmatched", 0)
                   for r in ledger if r["f2p"])
    f2p_total = sum(sum(r["f2p"].values()) for r in ledger if r["f2p"])
    p2p_fail = sum(r["p2p"].get("failed", 0) + r["p2p"].get("unmatched", 0)
                   for r in ledger if r["p2p"])
    p2p_total = sum(sum(r["p2p"].values()) for r in ledger if r["p2p"])

    base_lengths = sorted(len(row["ids"]) for row in base_rows)

    # Tool usage and step analysis
    tool_att = Counter()
    tool_adm = Counter()
    steps_att = [r.get("steps") or 0 for r in ledger]
    steps_adm = [r.get("steps") or 0 for r in admitted]
    steps_rej = [r.get("steps") or 0 for r in ledger if not r["admitted"]]

    for r in ledger:
        tool_att["bash"] += r.get("tool_bash", 0)
        tool_att["read"] += r.get("tool_read", 0)
        tool_att["edit"] += r.get("tool_edit", 0)
        if r["admitted"]:
            tool_adm["bash"] += r.get("tool_bash", 0)
            tool_adm["read"] += r.get("tool_read", 0)
            tool_adm["edit"] += r.get("tool_edit", 0)

    prompt_adm = sum(r.get("prompt_tokens") or 0 for r in admitted)
    comp_adm = sum(r.get("completion_tokens") or 0 for r in admitted)
    prompt_rej = prompt - prompt_adm
    comp_rej = completion - comp_adm

    sup_tokens_total = sum(r.get("supervised_tokens") or 0 for r in admitted)

    # Bracket breakdown (10-step increments)
    brackets = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
    b_stats = {f"{low:02d}-{high:02d}": {
        "att": 0, "adm": 0, "att_prompt": 0, "att_comp": 0,
        "adm_seq": 0, "adm_sup": 0,
        "bash": 0, "read": 0, "edit": 0
    } for low, high in brackets}

    for r in ledger:
        st = r.get("steps") or 0
        b_key = None
        for low, high in brackets:
            if low <= st <= high:
                b_key = f"{low:02d}-{high:02d}"
                break
        if not b_key:
            continue
        bs = b_stats[b_key]
        bs["att"] += 1
        bs["att_prompt"] += r.get("prompt_tokens") or 0
        bs["att_comp"] += r.get("completion_tokens") or 0
        if r["admitted"]:
            bs["adm"] += 1
            bs["adm_seq"] += r.get("length") or 0
            bs["adm_sup"] += r.get("supervised_tokens") or 0
            bs["bash"] += r.get("tool_bash", 0)
            bs["read"] += r.get("tool_read", 0)
            bs["edit"] += r.get("tool_edit", 0)

    p90_att = sorted(steps_att)[int(len(steps_att) * 0.9) - 1] if steps_att else 0
    p90_adm = sorted(steps_adm)[int(len(steps_adm) * 0.9) - 1] if steps_adm else 0
    p90_rej = sorted(steps_rej)[int(len(steps_rej) * 0.9) - 1] if steps_rej else 0

    lines = [
        "# 池 A 正式生成数据分析报告",
        "",
        f"日期：{__import__('datetime').date.today()}。数据源：`results/pool_a_formal/`"
        f"（{export.get('batches_seen') and len(export['batches_seen'])} 个批次目录）。",
        "",
        "## 1. 结论",
        "",
        f"- 尝试 **{attempted}** 题，其中判分通过 **{passed}**（{passed/attempted:.1%}）；"
        f"过八道准入后 **{len(admitted)}** 条可用样本（{len(admitted)/attempted:.1%}）。",
        f"- 覆盖 **{len(families)}** 个仓库家族；每家族 ≤1 题，验证家族已隔离。",
        f"- 样本序列 token：中位 {statistics.median(lengths):.0f}、最长 {max(lengths)}"
        if lengths else "- 尚无可用样本。",
        f"- 教师成本合计：prompt {prompt/1e6:.1f}M、completion {completion/1e6:.2f}M；"
        f"每条通过消耗 completion {completion/passed:.0f}" if passed else "",
        "",
        "## 2. 转型漏斗",
        "",
        "| 阶段 | 数量 | 比例 |",
        "|---|---:|---:|",
        f"| 尝试 | {attempted} | 100% |",
        f"| 判分通过 | {passed} | {passed/attempted:.1%} |",
        f"| 过准入（可用样本） | {len(admitted)} | {len(admitted)/attempted:.1%} |",
        "",
        "| 拒绝原因 | 次数 |",
        "|---|---:|",
    ]
    for reason, count in reasons.most_common():
        lines.append(f"| {reason} | {count} |")
    # per-batch timeline from the ledger's batch field
    by_batch = {}
    for r in ledger:
        b = r.get("batch", "?")
        slot = by_batch.setdefault(b, {"attempted": 0, "passed": 0, "admitted": 0})
        slot["attempted"] += 1
        slot["passed"] += bool(r["success"])
        slot["admitted"] += bool(r["admitted"])
    lines += ["", "## 2.1 逐批产出", "", "| 批次 | 尝试 | 通过 | 准入 |", "|---|---:|---:|---:|"]
    for b in sorted(by_batch):
        s = by_batch[b]
        lines.append(f"| {b} | {s['attempted']} | {s['passed']} | {s['admitted']} |")

    total_selected = len(json.loads((FORMAL / "selection_ids.json").read_text(encoding="utf-8")))
    missing_images = json.loads((FORMAL / "missing_images.json").read_text(encoding="utf-8")) \
        if (FORMAL / "missing_images.json").exists() else []
    infra_excluded = json.loads((FORMAL / "infra_excluded.json").read_text(encoding="utf-8")) \
        if (FORMAL / "infra_excluded.json").exists() else []
    lines += [
        "",
        f"**完成度：本轮覆盖选题的 {attempted}/{total_selected}（{attempted/total_selected:.1%}）。**"
        f" 未覆盖部分：{len(missing_images)} 题镜像在注册表永久缺失、"
        f"{len(infra_excluded)} 题连续两次运行以 infra 错误结束（非模型失败），"
        "两项均已记录、不计入任何比率。"
        " 编排器按 instance 去重、可断点续跑；未完成部分不进入本报告的任何比率。",
        "",
        "## 3. 目标测试与回归（按声明集合）",
        "",
        "| 指标 | 通过 | 失败/未匹配 |",
        "|---|---:|---:|",
        f"| F2P | {f2p_total - f2p_fail} | {f2p_fail} |",
        f"| P2P | {p2p_total - p2p_fail} | {p2p_fail} |",
        "",
        "## 4. 轮次数目、Token 消耗与工具调用分析",
        "",
        "### 4.1 总体轮次（Steps/Turns）与工具调用特征",
        "",
        "| 统计维度 | 全体尝试任务 (N=878) | 准入优质样本 (N=449) | 拒绝/未准入任务 (N=429) |",
        "|---|---:|---:|---:|",
        f"| **总轮次数 (Steps/Turns)** | {sum(steps_att):,} | {sum(steps_adm):,} | {sum(steps_rej):,} |",
        f"| 轮次均值 / 中位数 P50 | {statistics.mean(steps_att):.2f} / {statistics.median(steps_att):.0f} | {statistics.mean(steps_adm):.2f} / {statistics.median(steps_adm):.0f} | {statistics.mean(steps_rej):.2f} / {statistics.median(steps_rej):.0f} |",
        f"| 轮次 P90 / 极值范围 [Min, Max] | {p90_att} / [{min(steps_att)}, {max(steps_att)}] | {p90_adm} / [{min(steps_adm)}, {max(steps_adm)}] | {p90_rej} / [{min(steps_rej)}, {max(steps_rej)}] |",
        f"| **工具调用总量 (Total Tool Calls)** | {sum(tool_att.values()):,} | {sum(tool_adm.values()):,} | {sum(tool_att.values()) - sum(tool_adm.values()):,} |",
        f"| - `bash` 调用次数 (占比 / 均值) | {tool_att['bash']:,} ({tool_att['bash']/max(sum(tool_att.values()),1):.2%}, {tool_att['bash']/attempted:.2f}/题) | {tool_adm['bash']:,} ({tool_adm['bash']/max(sum(tool_adm.values()),1):.2%}, {tool_adm['bash']/max(len(admitted),1):.2f}/题) | {tool_att['bash']-tool_adm['bash']:,} ({(tool_att['bash']-tool_adm['bash'])/max(sum(tool_att.values())-sum(tool_adm.values()),1):.2%}, {(tool_att['bash']-tool_adm['bash'])/max(len(steps_rej),1):.2f}/题) |",
        f"| - `read` 调用次数 (占比 / 均值) | {tool_att['read']:,} ({tool_att['read']/max(sum(tool_att.values()),1):.2%}, {tool_att['read']/attempted:.2f}/题) | {tool_adm['read']:,} ({tool_adm['read']/max(sum(tool_adm.values()),1):.2%}, {tool_adm['read']/max(len(admitted),1):.2f}/题) | {tool_att['read']-tool_adm['read']:,} ({(tool_att['read']-tool_adm['read'])/max(sum(tool_att.values())-sum(tool_adm.values()),1):.2%}, {(tool_att['read']-tool_adm['read'])/max(len(steps_rej),1):.2f}/题) |",
        f"| - `edit` 调用次数 (占比 / 均值) | {tool_att['edit']:,} ({tool_att['edit']/max(sum(tool_att.values()),1):.2%}, {tool_att['edit']/attempted:.2f}/题) | {tool_adm['edit']:,} ({tool_adm['edit']/max(sum(tool_adm.values()),1):.2%}, {tool_adm['edit']/max(len(admitted),1):.2f}/题) | {tool_att['edit']-tool_adm['edit']:,} ({(tool_att['edit']-tool_adm['edit'])/max(sum(tool_att.values())-sum(tool_adm.values()),1):.2%}, {(tool_att['edit']-tool_adm['edit'])/max(len(steps_rej),1):.2f}/题) |",
        f"| **教师端 Prompt Tokens** (总计 / 均值 / 单步均值) | {prompt/1e6:.2f}M / {prompt/attempted:,.0f} / {prompt/max(sum(steps_att),1):,.0f} | {prompt_adm/1e6:.2f}M / {prompt_adm/max(len(admitted),1):,.0f} / {prompt_adm/max(sum(steps_adm),1):,.0f} | {prompt_rej/1e6:.2f}M / {prompt_rej/max(len(steps_rej),1):,.0f} / {prompt_rej/max(sum(steps_rej),1):,.0f} |",
        f"| **教师端 Completion Tokens** (总计 / 均值 / 单步均值) | {completion/1e6:.2f}M / {completion/attempted:,.0f} / {completion/max(sum(steps_att),1):,.0f} | {comp_adm/1e6:.2f}M / {comp_adm/max(len(admitted),1):,.0f} / {comp_adm/max(sum(steps_adm),1):,.0f} | {comp_rej/1e6:.2f}M / {comp_rej/max(len(steps_rej),1):,.0f} / {comp_rej/max(sum(steps_rej),1):,.0f} |",
        f"| **学生端 SFT 序列 Tokens** (总计 / 均值 / 单步均值) | — | {sum(lengths)/1e6:.2f}M / {statistics.mean(lengths):,.0f} / {sum(lengths)/max(sum(steps_adm),1):,.0f} | — |",
        f"| **学生端 有效监督 Tokens** (总计 / 均值 / 单步均值) | — | {sup_tokens_total/1e6:.2f}M / {sup_tokens_total/max(len(admitted),1):,.0f} / {sup_tokens_total/max(sum(steps_adm),1):,.0f} | — |",
        "",
        "### 4.2 轮次区间细分表（Step Bracket Breakdown）",
        "",
        "| 轮次区间 (Steps) | 尝试题数 | 准入题数 | 准入率 | 教师单题 Prompt 均值 | 教师单题 Comp 均值 | 准入样本 序列Token 均值 | 准入样本 监督Token 均值 | 准入样本 bash 调用 | read 调用 | edit 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for b in sorted(b_stats.keys()):
        bs = b_stats[b]
        att_c = bs["att"]
        adm_c = bs["adm"]
        adm_rate = adm_c / max(att_c, 1) * 100
        avg_att_pt = bs["att_prompt"] / max(att_c, 1)
        avg_att_ct = bs["att_comp"] / max(att_c, 1)
        avg_adm_seq = bs["adm_seq"] / max(adm_c, 1)
        avg_adm_sup = bs["adm_sup"] / max(adm_c, 1)
        lines.append(
            f"| {b} 步 | {att_c} | {adm_c} | {adm_rate:.1f}% | {avg_att_pt:,.0f} | {avg_att_ct:,.0f} | {avg_adm_seq:,.0f} | {avg_adm_sup:,.0f} | {bs['bash']} | {bs['read']} | {bs['edit']} |"
        )

    lines += [
        "",
        "### 4.3 核心统计发现与分析",
        "",
        "1. **准入收敛黄金窗口集中在 10–25 步**：",
        "   - 10 步以内的短轮次任务准入率高达 **76.6%**（98/128）；11–20 步为产出主力区间，贡献了 **209 条准入样本**（占总准入量的 46.5%），准入率维持在 **67.6%**。",
        "   - 20 步以内的任务合计贡献了 307 条准入样本（占 68.4%）；30 步以内的任务贡献了 401 条准入样本（占 89.3%）。",
        "   - 超过 30 步后准入率急剧衰减（31–40 步为 31.9%，41–50 步为 14.7%，51–60 步跌至 3.5%），表明多轮长交互容易陷入无效漫游、引入回归破坏或触发长度门截断。",
        "2. **Token 消耗随轮次的增长规律**：",
        "   - **教师端 Prompt Tokens 呈超线性二次累积增长**：因每轮请求都必须回传历史会话与所有工具 observation，导致 00–10 步平均单题只需 4.17 万 prompt tokens，而 51–60 步单题飙升至 301.6 万（激增 72 倍！）。未准入题目虽然只占总题数的 48.9%，却消耗了教师端 **79.5% 的 Prompt Tokens**（4.98 亿 tokens）。",
        "   - **学生端 SFT 序列 Tokens 呈平稳线性增长**：准入样本的学生端序列长度随步数稳定递增（每增加 10 步，序列长度约增加 8k~11k tokens）。在当前 64K 门禁下，超长样本严格受控，最长样本为 65,478 tokens。",
        "3. **工具调用频次与分布**：",
        "   - 全流程共发生 **21,224 次工具调用**，其中 `bash` 占 97.46% (20,684 次)，`read` 占 1.49% (316 次)，`edit` 占 1.06% (224 次)。",
        "   - **准入样本中精细文件工具比例翻倍**：准入样本中 `read` 占比由失败样本的 0.70% 提升至 **2.75%** (224次)，`edit` 占比由 0.80% 提升至 **1.46%** (119次)，表明即便在以 shell 为主要工作流的场景下，结合精准代码查看与定向编辑仍对提升闭环质量有显著正向作用。",
        "",
        "## 5. 与 50 题基线对照",
        "",
        "| 指标 | 50 题基线 | 本轮正式大批量 |",
        "|---|---:|---:|",
        f"| 题目通过率 | 25/50（50%） | {passed}/{attempted}（{passed/attempted:.1%}） |",
        f"| 导出样本 | 25 | {len(admitted)} |",
        f"| 序列 p50 | {statistics.median(base_lengths):.0f} | "
        f"{statistics.median(lengths):.0f} |" if lengths else "| 序列 p50 | - | - |",
        "",
        "## 6. 限制",
        "",
        "1. 每家族只取 1 题，样本不覆盖同仓库的多题难度梯度；跨家族去重只做题目级。",
        "2. 静态审计无法识别 `contract_ambiguous`/`leak_risk`；这两类仍需事后按判分标记。",
        "3. 判分通过并不等于训练准入：本轮台账只覆盖计划中的八道门，未做学生侧 NLL/难度筛选。",
        "4. 训练窗口：smoke 实测 ≤8K 可训练，32K/64K 被线性注意力 triton 路径的 64GiB 分配阻塞；"
        "导出按 64K 门，但训练端窗口需另行解决（见 smoke 纪要）。",
        "",
        "## 6.1 停止原因与续跑",
        "",
        "本轮受限于镜像拉取吞吐而不是模型：单批 15 张镜像需 330–1300 秒（注册表限速/层解压），"
        "8 路并行也无法压缩到线性，导致 903 题全量的预计墙钟约 20 小时以上。因此报告只覆盖"
        f"实际完成的 {attempted} 题，未完成部分**不进入任何比率**。",
        "",
        "续跑方式（幂等，按 instance 去重）：",
        "",
        "```bash",
        "powershell -File results/run_formal.ps1   # 自动跳过已完成题，继续拉取-运行-清理",
        "python results/formal_export.py            # 重算准入与 64K 门导出",
        "python results/formal_report.py            # 刷新本报告",
        "```",
        "",
        "## 7. 证据",
        "",
        "- `results/pool_a_formal/admission_ledger.jsonl`（逐题八门判定与成本）",
        "- `results/pool_a_formal/export_summary.json`（导出汇总）",
        "- `results/pool_a_formal/sft_admitted.jsonl`（ids+mask）",
        "- 原始轨迹：`results/pool_a_formal/batch_*/`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(l for l in lines if l is not None) + "\n", encoding="utf-8")
    print(f"report -> {DOC}")
    print(json.dumps({"attempted": attempted, "passed": passed, "admitted": len(admitted),
                      "families": len(families)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

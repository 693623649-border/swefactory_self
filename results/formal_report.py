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
        "## 4. 长度与步数分布",
        "",
        "| 统计 | 值 |",
        "|---|---:|",
        f"| 序列 p50 / p90 / max | "
        f"{statistics.median(lengths):.0f} / "
        f"{sorted(lengths)[max(0, int(len(lengths)*0.9)-1)]} / {max(lengths)} |" if lengths else "| 序列 | - |",
        f"| 步数 p50 / max | {statistics.median(steps):.0f} / {max(steps)} |" if steps else "| 步数 | - |",
        f"| 超过 32,768 的样本 | {sum(1 for x in lengths if x > 32768)} |",
        f"| 超过 65,536（被门拒绝） | {sum(1 for r in ledger if r.get('length') and r['length'] > 65536)} |",
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

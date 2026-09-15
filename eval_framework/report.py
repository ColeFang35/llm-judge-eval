# -*- coding: utf-8 -*-
"""把评测结果渲染成可读的 Markdown 报告（含失败归因定位）。"""
from __future__ import annotations

from .metrics import (PASS_THRESHOLD, dimension_means, failure_breakdown,
                      judge_vs_human, self_consistency)
from .rubric import RUBRIC
from .schema import Report


def render_markdown(report: Report, extra: dict | None = None) -> str:
    extra = extra or {}
    L: list[str] = []
    L.append("# Agent 评测报告")
    L.append("")
    L.append(f"- 评分器：**{report.meta.get('judge', '-')}**")
    L.append(f"- 用例数：**{report.total}**    通过：**{report.passed}**    通过率：**{report.pass_rate:.0%}**"
             f"（通过阈值 = 各维度均分 ≥ {PASS_THRESHOLD}）")
    L.append("")

    # ---- 各维度均分 ----
    L.append("## 一、各维度均分")
    L.append("")
    L.append("| 维度 | 说明 | 均分 |")
    L.append("|---|---|---|")
    for dim, val in dimension_means(report).items():
        label = "人工标注" if dim == "human" else f"{RUBRIC[dim]['name']}（{RUBRIC[dim]['judged_by']}）"
        desc = "-" if dim == "human" else RUBRIC[dim]["desc"]
        L.append(f"| {label} | {desc} | **{val:.2f}** |")
    L.append("")

    # ---- 失败归因 ----
    L.append("## 二、失败归因（问题定位）")
    L.append("")
    fb = failure_breakdown(report)
    if fb:
        L.append("| 失败类型 | 次数 |")
        L.append("|---|---|")
        for k, v in fb.items():
            L.append(f"| {k} | {v} |")
    else:
        L.append("无失败用例。")
    L.append("")

    # ---- Judge 可靠性 ----
    jh = judge_vs_human(report)
    L.append("## 三、评分器可靠性（Judge vs 人工标注）")
    L.append("")
    if jh:
        L.append(f"- 样本数：**{jh['n']}**")
        L.append(f"- 一致率（按通过/不通过二分类）：**{jh['agreement']:.0%}**")
        L.append(f"- **Cohen's kappa：{jh['kappa']}**（>0.6 一般认为一致性良好）")
        L.append(f"- 平均绝对误差 MAE：**{jh['mae']}**")
    else:
        L.append("（无人工标注数据，跳过）")
    L.append("")

    sc = extra.get("self_consistency")
    if sc:
        L.append("## 四、自一致性（重复评测稳定性）")
        L.append("")
        L.append(f"- 重复次数：{sc['repeats']}    分数标准差均值：**{sc['mean_std']}**    最大：{sc['max_std']}")
        L.append(f"- **结论翻转率：{sc['flip_rate']:.0%}**（同一用例重复评测时通过/不通过翻转的比例）")
        L.append("")

    bias = extra.get("position_bias")
    if bias:
        L.append("## 五、位置偏见检测")
        L.append("")
        L.append(f"- 顺序①偏好：{bias['order1_prefers']}    顺序②偏好：{bias['order2_prefers']}")
        L.append(f"- 结论：**{bias['verdict']}**")
        L.append("")

    # ---- 逐用例明细 ----
    L.append("## 六、逐用例明细")
    L.append("")
    L.append("| 用例 | 类别 | 任务 | 总分 | 通过 | 失败归因 |")
    L.append("|---|---|---|---|---|---|")
    for r in report.results:
        task = r.case.task[:24] + ("…" if len(r.case.task) > 24 else "")
        fail = "、".join(r.failures) if r.failures else "-"
        L.append(f"| {r.case.case_id} | {r.case.category} | {task} | {r.overall:.2f} "
                 f"| {'✅' if r.passed else '❌'} | {fail} |")
    L.append("")

    L.append("### 评分理由（Judge 给出的依据）")
    L.append("")
    for r in report.results:
        L.append(f"**{r.case.case_id}**　{r.case.task}")
        for j in r.judgments:
            if j.source == "human":
                continue
            L.append(f"- `{j.dimension}` = {j.score:.2f}（{j.source}）：{j.rationale}")
        L.append("")

    return "\n".join(L)

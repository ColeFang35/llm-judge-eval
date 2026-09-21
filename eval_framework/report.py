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

    # 章节编号按实际出现的顺序走：自一致性与位置偏见都是可选的，
    # 写死"一二三四五六"会在没跑某个选项时跳号（曾经就跳过号）。
    _CN = "一二三四五六七八九十"
    _sec = [0]

    def head(title: str) -> str:
        _sec[0] += 1
        return f"## {_CN[_sec[0] - 1]}、{title}"

    L.append("# Agent 评测报告")
    L.append("")
    L.append(f"- 评分器：**{report.meta.get('judge', '-')}**")
    L.append(f"- 用例数：**{report.total}**    通过：**{report.passed}**    通过率：**{report.pass_rate:.0%}**"
             f"（通过阈值 = 各维度均分 ≥ {PASS_THRESHOLD}）")

    # 回显真实用量：否则"配了 key 却回退成离线 Judge"的报告看着和真结果一样
    st = extra.get("judge_stats")
    if st:
        cost = "单价未配置" if st["cost_usd"] is None else f"${st['cost_usd']:.4f}"
        L.append(f"- 评分模型：**{st['model']}**（thinking={st['thinking']}）")
        L.append(f"- 实际调用 **{st['calls']}** 次 · 回退离线 Judge **{st['fallbacks']}** 次 · "
                 f"token {st['in_tokens']}+{st['out_tokens']} · 花费 {cost}")
        if st["fallbacks"]:
            L.append("")
            L.append(f"> ⚠️ **有 {st['fallbacks']} 次回退到了离线规则 Judge**，下面的分数里混了规则打分，"
                     f"不是纯 LLM 结果。最近一次错误：`{st['last_error']}`")
    L.append("")

    # ---- 各维度均分 ----
    L.append(head("各维度均分"))
    L.append("")
    L.append("| 维度 | 说明 | 均分 |")
    L.append("|---|---|---|")
    for dim, val in dimension_means(report).items():
        label = "人工标注" if dim == "human" else f"{RUBRIC[dim]['name']}（{RUBRIC[dim]['judged_by']}）"
        desc = "-" if dim == "human" else RUBRIC[dim]["desc"]
        L.append(f"| {label} | {desc} | **{val:.2f}** |")
    L.append("")

    # ---- 失败归因 ----
    L.append(head("失败归因（问题定位）"))
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
    L.append(head("评分器可靠性（Judge vs 人工标注）"))
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
        L.append(head("自一致性（重复评测稳定性）"))
        L.append("")
        L.append(f"- 重复次数：{sc['repeats']}    分数标准差均值：**{sc['mean_std']}**    最大：{sc['max_std']}")
        L.append(f"- **结论翻转率：{sc['flip_rate']:.0%}**（同一用例重复评测时通过/不通过翻转的比例）")
        L.append("")
        L.append("> ⚠️ 稳定 ≠ 正确。判分稳定只说明它可复现；如果它稳定地判错，"
                 "这一栏照样好看。要和上面的「Judge vs 人工标注」一起看。")
        L.append("")

    bias = extra.get("position_bias")
    if bias:
        L.append(head("位置偏见检测（成对比较，交换顺序各问一次）"))
        L.append("")
        L.append(f"- 对数：**{bias['n_pairs']}**（每对 = 真实轨迹 vs 同任务的降级轨迹）")
        L.append(f"- 两种顺序下**结论一致**：**{bias['consistent']}/{bias['n_pairs']}**（含两次都判平局）")
        L.append(f"- 两次顺序都选了**排在前面的那个**：**{bias['first_position_wins']}/{bias['n_pairs']}**"
                 "（这一项 > 0 就是位置偏见）")
        L.append(f"- 两次都选中了更好的那条：**{bias['picked_better']}/{bias['n_pairs']}**")
        L.append(f"- 结论：**{bias['verdict']}**")
        L.append("")
        L.append("| 用例 | 好答案在前时选了 | 好答案在后时选了 |")
        L.append("|---|---|---|")
        for r in bias["rows"]:
            L.append(f"| {r['case_id']} | {r['pick_when_good_first'] or '平'} "
                     f"| {r['pick_when_good_second'] or '平'} |")
        L.append("")

    # ---- 逐用例明细 ----
    L.append(head("逐用例明细"))
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

# -*- coding: utf-8 -*-
"""Judge 偏见检测：位置偏见（position bias）。

做法：把同一对执行（好 / 差）放进**同一个 prompt** 让裁判二选一，
再交换两者的位置问一次。

- 两次都选中**同一个内容** → 结论稳定，无位置偏见迹象
- 两次都选中**排在前面的那个** → 位置偏见：它看的是位置，不是内容

⚠️ 这个检测**必须用成对比较**。本文件早先的写法是拿两次独立的绝对评分相减
（`score(A) - score(B)` 与 `score(B) - score(A)`）—— 那两个数恒为相反数，
于是 `consistent` 恒为 True、`first_wins` 恒为 False，报告永远输出
「一致（无位置偏见迹象）」。**它消耗了 12 次模型调用，却没有测任何东西。**
这和「看起来有、其实测不出东西的指标」是同一类错，跟 decision-layer-lab 里
那个「ECE 0.009 看着最好看、其实毫无区分度」是同一个病。
"""
from __future__ import annotations

from .judges.base import BaseJudge
from .schema import AgentTrace, EvalCase


def _degrade(trace: AgentTrace) -> AgentTrace:
    """造一个同任务、但明显更差的对照。

    位置偏见要测的是「同样的内容换个位置，裁判还认不认」，
    所以两条轨迹必须是同一个任务 —— 否则偏好的差异来自任务本身，不是位置。
    """
    return AgentTrace(
        case_id=trace.case_id,
        steps=trace.steps[:1],
        tool_calls=[],
        final_answer=(trace.final_answer or "")[:12] + "……",
    )


def _pick(winner: str, first_name: str, second_name: str) -> str | None:
    """把「甲/乙/平」翻译成内容层面的选择（好/差）。平局返回 None。"""
    if winner == "甲":
        return first_name
    if winner == "乙":
        return second_name
    return None


def position_bias(judge: BaseJudge, cases: list[EvalCase],
                  traces: dict[str, AgentTrace]) -> dict:
    rows: list[dict] = []
    for case in cases:
        trace = traces.get(case.case_id)
        if trace is None:
            continue
        good, bad = trace, _degrade(trace)
        r1 = judge.compare(case, good, bad)     # 甲=好、乙=差
        r2 = judge.compare(case, bad, good)     # 甲=差、乙=好
        pick1 = _pick(r1["winner"], "好", "差")
        pick2 = _pick(r2["winner"], "差", "好")
        rows.append({
            "case_id": case.case_id,
            "round1": r1["winner"], "round2": r2["winner"],
            "pick_when_good_first": pick1,
            "pick_when_good_second": pick2,
            # 两次顺序下结论一致（含两次都判平局）→ 稳定
            "consistent": pick1 == pick2,
            # 两次都选了排在前面的那个 → 位置偏见
            "prefers_first": r1["winner"] == "甲" and r2["winner"] == "甲",
            # 两次都选中了好答案 → 裁判本身判得对
            "picked_better": pick1 == "好" and pick2 == "好",
        })

    n = len(rows)
    if not n:
        return {}
    first_wins = sum(1 for r in rows if r["prefers_first"])
    consistent = sum(1 for r in rows if r["consistent"])
    picked_better = sum(1 for r in rows if r["picked_better"])
    if first_wins:
        verdict = f"存在位置偏见：{first_wins}/{n} 对在两种顺序下都选了排在前面的那个"
    elif consistent == n:
        verdict = f"未发现位置偏见（{n}/{n} 对在两种顺序下结论一致）"
    else:
        verdict = (f"未发现位置偏见（0/{n} 对两次都选前面），"
                   f"但有 {n - consistent}/{n} 对的结论随顺序改变 —— 裁判在这几对上本身判不稳")
    return {
        "n_pairs": n,
        "consistent": consistent,
        "first_position_wins": first_wins,
        "picked_better": picked_better,
        "verdict": verdict,
        "rows": rows,
    }

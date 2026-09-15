# -*- coding: utf-8 -*-
"""Judge 偏见检测：位置偏见（position bias）。

做法：把同一对答案 (A, B) 以两种顺序各评一次，看结论是否一致。
若两种顺序下都偏好同一个位置（而不是同一个答案），说明存在位置偏见。
"""
from __future__ import annotations

from statistics import mean

from .judges.base import BaseJudge
from .schema import AgentTrace, EvalCase


def position_bias(judge: BaseJudge, case: EvalCase, a: AgentTrace, b: AgentTrace) -> dict:
    """返回 {first_pref_rate, consistent, verdict}。"""
    # 顺序 1：A 在前；顺序 2：B 在前
    score_a1 = _ask_pair(judge, case, a, b)     # >0 表示更偏好 A
    score_a2 = _ask_pair(judge, case, b, a)     # >0 表示更偏好 B（因为 B 在前）
    consistent = (score_a1 > 0 and score_a2 < 0) or (score_a1 < 0 and score_a2 > 0)
    first_wins = (score_a1 > 0) and (score_a2 > 0)   # 两种顺序都偏好"排在前面的那个"
    return {
        "order1_prefers": "A" if score_a1 > 0 else "B",
        "order2_prefers": "B" if score_a2 > 0 else "A",
        "consistent": consistent,
        "first_position_bias": first_wins,
        "verdict": "位置偏见：两次都选了排在前面的答案" if first_wins
                   else ("一致（无位置偏见迹象）" if consistent else "结论不一致，需人工复核"),
    }


def _ask_pair(judge: BaseJudge, case: EvalCase, first: AgentTrace, second: AgentTrace) -> float:
    """用离线可用的方式做相对比较：比较两次评分的总分，返回 first - second。"""
    s1 = mean(j.score for j in judge.judge(case, first))
    s2 = mean(j.score for j in judge.judge(case, second))
    return round(s1 - s2, 3)

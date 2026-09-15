# -*- coding: utf-8 -*-
"""评测执行器：把可编程校验与 Judge 打分合成一个用例的结果。"""
from __future__ import annotations

from statistics import mean

from . import programmatic
from .judges.base import BaseJudge
from .judges.mock_judge import MockJudge
from .schema import AgentTrace, CaseResult, EvalCase, Report


def evaluate(case: EvalCase, trace: AgentTrace, judge: BaseJudge | None = None) -> CaseResult:
    judge = judge or MockJudge()

    # 1) 可编程硬指标（能用代码判的，先判）
    judgments, failures = programmatic.check(case, trace)
    # 2) 开放式维度交给 Judge
    judgments += judge.judge(case, trace)
    # 3) 人工标注也放进 judgments，方便统一对比
    if case.human_label is not None:
        from .schema import Judgment
        judgments.append(Judgment(case.case_id, "human", case.human_label, "人工标注", "human"))

    return CaseResult(case=case, trace=trace, judgments=judgments, failures=failures)


def evaluate_all(cases: list[EvalCase], traces: dict[str, AgentTrace],
                 judge: BaseJudge | None = None) -> Report:
    judge = judge or MockJudge()
    results = [evaluate(c, traces[c.case_id], judge) for c in cases if c.case_id in traces]
    return Report(results=results, meta={"judge": judge.name, "n_cases": len(results)})


def evaluate_repeats(case: EvalCase, trace: AgentTrace, judge: BaseJudge,
                     repeats: int = 3) -> list[float]:
    """同一用例重复评测 N 次，返回每次的总分 —— 用于自一致性分析。"""
    return [round(mean(j.score for j in evaluate(case, trace, judge).judgments
                       if j.source != "human"), 3) for _ in range(repeats)]

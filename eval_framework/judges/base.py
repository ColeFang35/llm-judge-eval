# -*- coding: utf-8 -*-
"""Judge 接口：所有评分器实现同一契约，便于替换与做一致性对比。"""
from __future__ import annotations

from ..schema import AgentTrace, EvalCase, Judgment


class BaseJudge:
    name = "base"

    def judge(self, case: EvalCase, trace: AgentTrace) -> list[Judgment]:
        """对一次 Agent 执行给出若干维度的评分（0~1 + 理由）。"""
        raise NotImplementedError

    def compare(self, case: EvalCase, first: AgentTrace, second: AgentTrace) -> dict:
        """成对比较两次执行，返回 {"winner": "甲"|"乙"|"平", "rationale": str}。

        位置偏见检测必须走这条路 —— 两个候选要在**同一个 prompt 里**呈现，
        再交换顺序问一次。拿两次独立的绝对评分相减是测不出位置偏见的：
        那两次的结果恒为相反数，结论永远是「一致」，等于什么都没测。
        """
        raise NotImplementedError

    # 只对"需要 LLM 判"的维度打分；可编程维度由 programmatic 负责
    LLM_DIMENSIONS = ("task_completion", "faithfulness", "step_quality")

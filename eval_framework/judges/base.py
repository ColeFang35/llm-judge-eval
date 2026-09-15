# -*- coding: utf-8 -*-
"""Judge 接口：所有评分器实现同一契约，便于替换与做一致性对比。"""
from __future__ import annotations

from ..schema import AgentTrace, EvalCase, Judgment


class BaseJudge:
    name = "base"

    def judge(self, case: EvalCase, trace: AgentTrace) -> list[Judgment]:
        """对一次 Agent 执行给出若干维度的评分（0~1 + 理由）。"""
        raise NotImplementedError

    # 只对"需要 LLM 判"的维度打分；可编程维度由 programmatic 负责
    LLM_DIMENSIONS = ("task_completion", "faithfulness", "step_quality")

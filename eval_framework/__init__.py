# -*- coding: utf-8 -*-
"""LLM-as-a-Judge Agent 评测框架。

设计要点（面试可讲）：
- 双层评测：**可编程硬指标**（能用代码判的就不问 LLM）+ **LLM-as-a-Judge**（开放式质量）
- Rubric 驱动：把"好不好"拆成可打分的维度 + 分档描述，降低 Judge 主观性
- 可靠性分析：Judge 与人工标注的一致率 / Cohen's kappa；重复评测的稳定性（方差）
- 偏见检测：成对比较的位置偏见（交换顺序看结论是否翻转）
- 失败归因：把失败分类定位，输出可读报告
"""
from .schema import EvalCase, AgentTrace, ToolCall, Judgment, CaseResult, Report
from .rubric import RUBRIC, DIMENSIONS
from .runner import evaluate, evaluate_all
from .report import render_markdown

__all__ = [
    "EvalCase", "AgentTrace", "ToolCall", "Judgment", "CaseResult", "Report",
    "RUBRIC", "DIMENSIONS", "evaluate", "evaluate_all", "render_markdown",
]

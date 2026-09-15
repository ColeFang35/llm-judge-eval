# -*- coding: utf-8 -*-
"""评测框架的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Agent 的一次工具调用。"""
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class AgentTrace:
    """被评测对象：一次 Agent 执行的完整轨迹。"""
    case_id: str
    steps: list[str] = field(default_factory=list)        # 推理/思考步骤
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    latency_ms: float | None = None
    token_usage: int | None = None


@dataclass
class EvalCase:
    """评测用例：任务 + 期望（用于判定）。"""
    case_id: str
    task: str
    expected_tools: list[str] = field(default_factory=list)        # 可编程校验：期望调用的工具
    expected_args: dict[str, dict] = field(default_factory=dict)   # 可编程校验：工具={参数:期望值}
    answer_points: list[str] = field(default_factory=list)         # Judge 用：答案应覆盖的要点
    forbidden: list[str] = field(default_factory=list)             # 禁止出现的内容（如编造、越权）
    human_label: float | None = None      # 人工标注总分(0~1)，用于一致性分析
    category: str = "general"
    kind: str = "single"                  # single / multi_step


@dataclass
class Judgment:
    """一条评分（某个维度、某个用例、某个评分器）。"""
    case_id: str
    dimension: str
    score: float            # 0~1
    rationale: str = ""
    source: str = "mock"    # programmatic / llm / mock / human


@dataclass
class CaseResult:
    """单个用例的评测结果。"""
    case: EvalCase
    trace: AgentTrace
    judgments: list[Judgment] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)     # 失败归因标签

    @property
    def passed(self) -> bool:
        """通过 = 无硬性失败归因 且 总分达到阈值（两个口径一致）。"""
        from .rubric import PASS_THRESHOLD
        return (not self.failures) and self.overall >= PASS_THRESHOLD

    def score(self, dimension: str) -> float | None:
        vals = [j.score for j in self.judgments if j.dimension == dimension]
        return sum(vals) / len(vals) if vals else None

    @property
    def overall(self) -> float:
        vals = [j.score for j in self.judgments if j.source != "human"]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class Report:
    """整体评测报告。"""
    results: list[CaseResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

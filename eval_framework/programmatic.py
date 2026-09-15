# -*- coding: utf-8 -*-
"""可编程硬指标：能用代码判定的，绝不交给 LLM。

判定项：
1. 期望工具是否都被调用（missing）
2. 是否调用了非期望工具（extra）
3. 工具参数是否与期望一致（arg mismatch）
4. 工具是否报错（tool error）
5. 是否出现禁止内容（forbidden）
"""
from __future__ import annotations

from .schema import AgentTrace, EvalCase, Judgment

DIM = "tool_use"


def check(case: EvalCase, trace: AgentTrace) -> tuple[list[Judgment], list[str]]:
    called = [tc.name for tc in trace.tool_calls]
    failures: list[str] = []

    missing = [t for t in case.expected_tools if t not in called]
    extra = [t for t in called if t not in case.expected_tools]
    arg_bad: list[str] = []
    for tc in trace.tool_calls:
        exp = case.expected_args.get(tc.name)
        if not exp:
            continue
        for k, v in exp.items():
            if str(tc.args.get(k)) != str(v):
                arg_bad.append(f"{tc.name}.{k} 期望 {v}，实际 {tc.args.get(k)}")

    errored = [tc.name for tc in trace.tool_calls if tc.error]
    forbidden_hit = [w for w in case.forbidden if w and w in trace.final_answer]

    # ---- 打分：按严重程度加权 ----
    score = 1.0
    if missing:
        score = 0.0
        failures.append("缺失关键工具调用")
    if arg_bad:
        score = min(score, 0.5)
        failures.append("工具参数错误")
    if extra:
        score = min(score, 0.75)
        failures.append("多余工具调用")
    if errored:
        score = min(score, 0.5)
        failures.append("工具执行报错")
    if forbidden_hit:
        score = 0.0
        failures.append("出现禁止内容")

    detail = []
    if missing:      detail.append(f"缺失工具={missing}")
    if extra:        detail.append(f"多余工具={extra}")
    if arg_bad:      detail.append(f"参数错误={arg_bad}")
    if errored:      detail.append(f"报错={errored}")
    if forbidden_hit: detail.append(f"禁止内容={forbidden_hit}")

    j = Judgment(
        case_id=case.case_id, dimension=DIM, score=score,
        rationale="；".join(detail) if detail else "工具调用全部符合期望",
        source="programmatic",
    )
    return [j], failures

# -*- coding: utf-8 -*-
"""评分 Rubric：把"好不好"拆成可打分的维度 + 分档描述。

设计要点：
- **维度正交**：每个维度只判一件事，避免互相污染
- **分档有描述**：不给"1~10 分"这种自由发挥的空间，而是给出每一档的判定标准
- **可编程优先**：能用代码判定的维度（如工具是否调对）不交给 LLM
"""
from __future__ import annotations

DIMENSIONS = ["task_completion", "tool_use", "faithfulness", "step_quality"]

# 通过阈值：各维度均分达到该值、且没有硬性失败归因，才算通过
PASS_THRESHOLD = 0.75

RUBRIC: dict[str, dict] = {
    "task_completion": {
        "name": "任务完成度",
        "desc": "最终答案是否真正解决了用户的请求",
        "judged_by": "llm",          # 开放式质量，交给 LLM 判
        "levels": {
            1.00: "完整解决用户请求，无遗漏",
            0.75: "基本解决，但有次要信息遗漏",
            0.50: "只解决了部分，关键信息缺失",
            0.25: "方向错误，仅勉强相关",
            0.00: "未回应请求或答非所问",
        },
    },
    "tool_use": {
        "name": "工具调用",
        "desc": "该调的工具是否调了、参数是否正确、有没有多余或越权调用",
        "judged_by": "programmatic",  # 可以精确判定 → 用代码判
        "levels": {
            1.00: "期望工具全部按正确参数调用，无多余调用",
            0.75: "工具调用正确，但存在不影响结果的多余调用",
            0.50: "关键工具缺失或参数有误，但未导致错误结论",
            0.00: "该调的没调，或调用了错误/越权工具",
        },
    },
    "faithfulness": {
        "name": "事实忠实度",
        "desc": "回答是否忠于工具返回的事实，没有编造数据",
        "judged_by": "llm",
        "levels": {
            1.00: "所有事实性陈述都能在工具结果中找到依据",
            0.75: "主体有依据，个别表述略有外推",
            0.50: "存在部分无依据的陈述",
            0.25: "多处编造",
            0.00: "核心数据为编造",
        },
    },
    "step_quality": {
        "name": "推理步骤质量",
        "desc": "推理是否有条理、无冗余绕路、无跳步",
        "judged_by": "llm",
        "levels": {
            1.00: "步骤清晰必要，直指目标",
            0.75: "基本合理，有少量冗余",
            0.50: "存在明显绕路或跳步",
            0.25: "推理混乱",
            0.00: "无有效推理",
        },
    },
}


def rubric_prompt() -> str:
    """把 Rubric 渲染成给 Judge 的评分说明（结构化输出的评分标准）。"""
    lines = ["请严格按照下面的评分维度与分档说明给这次 Agent 执行打分（每项取最贴近的一档数值）。", ""]
    for key in DIMENSIONS:
        r = RUBRIC[key]
        if r["judged_by"] != "llm":
            continue
        lines.append(f"【{r['name']}】{r['desc']}")
        for score, desc in sorted(r["levels"].items(), reverse=True):
            lines.append(f"  {score:.2f} = {desc}")
        lines.append("")
    return "\n".join(lines)

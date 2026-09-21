# -*- coding: utf-8 -*-
"""离线规则 Judge：不调任何模型，确定性输出。

用途：
- 无网络 / 无 API Key 时让整条评测链路可跑通、可回归
- 作为 LLM Judge 的**基线对照**（两者差异本身就是分析素材）

它不是"假装在打分"：每个维度都用可解释的启发式规则算出来，理由里写清依据。
"""
from __future__ import annotations

import re
from statistics import mean

from ..schema import AgentTrace, EvalCase, Judgment
from .base import BaseJudge

_NUM = re.compile(r"\d+(?:\.\d+)?")


class MockJudge(BaseJudge):
    name = "mock"

    def compare(self, case: EvalCase, first: AgentTrace, second: AgentTrace) -> dict:
        """离线基线：按自己的评分比大小。确定性、无位置效应，用作对照。"""
        s1 = mean(j.score for j in self.judge(case, first))
        s2 = mean(j.score for j in self.judge(case, second))
        if abs(s1 - s2) < 1e-9:
            return {"winner": "平", "rationale": f"规则分相同（{s1:.2f}）"}
        return {"winner": "甲" if s1 > s2 else "乙",
                "rationale": f"规则分 甲={s1:.2f} 乙={s2:.2f}"}

    def judge(self, case: EvalCase, trace: AgentTrace) -> list[Judgment]:
        return [
            self._task_completion(case, trace),
            self._faithfulness(case, trace),
            self._step_quality(case, trace),
        ]

    # ---- 任务完成度：答案覆盖了多少"应覆盖要点" ----
    def _task_completion(self, case: EvalCase, trace: AgentTrace) -> Judgment:
        ans = trace.final_answer or ""
        if not ans.strip():
            return Judgment(case.case_id, "task_completion", 0.0, "最终答案为空", self.name)
        pts = case.answer_points or []
        if not pts:
            return Judgment(case.case_id, "task_completion", 1.0, "无要点要求，答案非空", self.name)
        hit = [p for p in pts if p in ans]
        score = len(hit) / len(pts)
        miss = [p for p in pts if p not in ans]
        return Judgment(
            case.case_id, "task_completion", round(score, 2),
            f"覆盖 {len(hit)}/{len(pts)} 个要点" + (f"，缺：{miss[:3]}" if miss else ""),
            self.name,
        )

    # ---- 事实忠实度：答案里的数字是否都能在工具结果里找到出处 ----
    def _faithfulness(self, case: EvalCase, trace: AgentTrace) -> Judgment:
        ans = trace.final_answer or ""
        # 证据集 = 工具结果 + 推理步骤 + 工具调用参数 + 用户原始任务
        # （订单号/手机号等数字往往来自提问或参数，不能算"无依据"）
        evidence = " ".join([
            " ".join(str(tc.result) for tc in trace.tool_calls if tc.result is not None),
            " ".join(str(tc.args) for tc in trace.tool_calls),
            " ".join(trace.steps),
            case.task,
        ])
        nums = _NUM.findall(ans)
        if not nums:
            return Judgment(case.case_id, "faithfulness", 1.0, "答案无数字断言", self.name)
        unsupported = [n for n in nums if n not in evidence]
        score = round(1.0 - len(unsupported) / len(nums), 2)
        reason = (f"{len(nums)} 个数字断言中 {len(unsupported)} 个无工具依据"
                  + (f"：{unsupported[:3]}" if unsupported else ""))
        return Judgment(case.case_id, "faithfulness", max(0.0, score), reason, self.name)

    # ---- 步骤质量：步数是否在合理区间 + 有无工具报错 ----
    def _step_quality(self, case: EvalCase, trace: AgentTrace) -> Judgment:
        n = len(trace.steps)
        errs = sum(1 for tc in trace.tool_calls if tc.error)
        if n == 0:
            return Judgment(case.case_id, "step_quality", 0.25, "无推理步骤", self.name)
        if n <= 2:
            score, why = 0.75, f"步骤偏少（{n} 步），可能跳步"
        elif n <= 8:
            score, why = 1.0, f"步骤数合理（{n} 步）"
        else:
            score, why = 0.75, f"步骤偏多（{n} 步），可能绕路"
        if errs:
            score = min(score, 0.5)
            why += f"；有 {errs} 次工具报错"
        return Judgment(case.case_id, "step_quality", score, why, self.name)

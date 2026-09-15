# -*- coding: utf-8 -*-
"""LLM-as-a-Judge：用大模型给开放式维度打分。

工程要点（面试可讲）：
- **结构化输出**：要求返回 JSON，字段固定 → 便于程序消费，也降低解析失败率
- **temperature=0**：尽量让评分可复现；配合 repeats 做**自一致性**分析
- **重试 + 回退**：解析失败重试；无 Key / 连续失败回退到离线 Judge，保证评测不中断
- **只判开放的**：工具调用这类可精确判定的维度交给 programmatic，不浪费 token 也不引入噪声
"""
from __future__ import annotations

import json
import os
import urllib.request

from ..rubric import rubric_prompt
from ..schema import AgentTrace, EvalCase, Judgment
from .base import BaseJudge
from .mock_judge import MockJudge

_SYS = (
    "你是一个严格的 AI Agent 评测员。你的任务是按给定 Rubric 给一次 Agent 执行打分，"
    "只输出 JSON，不要任何解释性文字。评分必须严格依据 Rubric 的分档说明，"
    "并且忠实度维度要特别警惕「答案里出现工具结果中不存在的数据」。"
)

_KEYS = ("task_completion", "faithfulness", "step_quality")


class LLMJudge(BaseJudge):
    name = "llm"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 60.0,
                 retries: int = 2, fallback: BaseJudge | None = None) -> None:
        self.model = model or os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.timeout = timeout
        self.retries = retries
        self.fallback = fallback or MockJudge()

    # ---------- 组装给 Judge 的输入 ----------
    @staticmethod
    def _render_trace(trace: AgentTrace) -> str:
        lines = [f"【用户任务】{trace.case_id}", ""]
        lines.append("【推理步骤】")
        lines += [f"  {i+1}. {s}" for i, s in enumerate(trace.steps)] or ["  （无）"]
        lines.append("")
        lines.append("【工具调用】")
        if trace.tool_calls:
            for tc in trace.tool_calls:
                res = json.dumps(tc.result, ensure_ascii=False)[:300] if tc.result is not None else "（无结果）"
                err = f"  [报错: {tc.error}]" if tc.error else ""
                lines.append(f"  - {tc.name}({json.dumps(tc.args, ensure_ascii=False)}) → {res}{err}")
        else:
            lines.append("  （无）")
        lines.append("")
        lines.append(f"【最终答案】\n{trace.final_answer or '（空）'}")
        return "\n".join(lines)

    def _user_prompt(self, case: EvalCase, trace: AgentTrace) -> str:
        pts = "\n".join(f"  - {p}" for p in case.answer_points) or "  （无）"
        schema = ", ".join(f'"{k}": {{"score": <0~1>, "rationale": "<一句话依据>"}}' for k in _KEYS)
        return (
            f"{rubric_prompt()}\n"
            f"【该用例应覆盖的要点】\n{pts}\n\n"
            f"{self._render_trace(trace)}\n\n"
            f"请只输出如下 JSON（不要多余文字）：\n{{{schema}}}"
        )

    # ---------- 调用 ----------
    def _call(self, user_prompt: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return _extract_json(body["choices"][0]["message"]["content"])

    def judge(self, case: EvalCase, trace: AgentTrace) -> list[Judgment]:
        if not self.api_key or not self.base_url:
            return self.fallback.judge(case, trace)

        prompt = self._user_prompt(case, trace)
        for attempt in range(self.retries + 1):
            try:
                data = self._call(prompt)
                out: list[Judgment] = []
                for key in _KEYS:
                    item = data.get(key)
                    if not isinstance(item, dict):
                        continue
                    score = float(item.get("score", 0.0))
                    out.append(Judgment(
                        case_id=case.case_id, dimension=key,
                        score=max(0.0, min(1.0, score)),
                        rationale=str(item.get("rationale", ""))[:200],
                        source=self.name,
                    ))
                if len(out) == len(_KEYS):
                    return out
            except Exception:
                continue
        # 连续失败 → 回退，保证评测不中断
        return self.fallback.judge(case, trace)


def _extract_json(text: str) -> dict:
    """从模型输出里抠 JSON（容忍 ```json 围栏与前后废话）。"""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except ValueError:
            pass
    return {}

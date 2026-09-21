# -*- coding: utf-8 -*-
"""LLM-as-a-Judge：用大模型给开放式维度打分。

工程要点（面试可讲）：
- **结构化输出**：要求返回 JSON，字段固定 → 便于程序消费，也降低解析失败率
- **temperature=0**：尽量让评分可复现；配合 repeats 做**自一致性**分析
- **重试 + 回退**：解析失败重试；无 Key / 连续失败回退到离线 Judge，保证评测不中断
- **只判开放的**：工具调用这类可精确判定的维度交给 programmatic，不浪费 token 也不引入噪声

⚠️ 回退会**计数并在报告里回显**。否则一份写着「评分器：llm」的报告里可能混着
离线规则的打分，读者完全看不出来 —— 这跟「以为在跑真模型、其实跑的是 mock」
是同一类错。宁可报告上写着「有 3 次回退」，也不要一个看不出来的假数字。
"""
from __future__ import annotations

import json
import os
import urllib.request

from ..http import OPENER
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

_SYS_CMP = (
    "你是一个严格的 AI Agent 评测员。下面给出**同一个任务的两次执行**（甲、乙），"
    "请判断哪一次更好，只输出 JSON，不要任何解释性文字。"
)
_CMP_SCHEMA = '{"winner": "甲"|"乙"|"平", "rationale": "<一句话依据>"}'
_WINNER_ALIAS = {"a": "甲", "b": "乙", "first": "甲", "second": "乙", "tie": "平", "equal": "平"}

# 单价（美元 / 百万 token），取数日期 2026-09-21。查不到的模型记 0 —— 别拿 0 当结论。
_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.15, 0.60),
    "deepseek-v4-pro": (0.66, 1.98),
}


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

        # DeepSeek 的 thinking **默认开着**，而开着的时候 temperature 会被静默忽略。
        # 评测要的是可复现，所以默认关掉；其他端点不发这个字段。
        t = os.getenv("JUDGE_THINKING", "").strip().lower()
        if t in ("enabled", "disabled"):
            self.thinking: str | None = t
        elif "deepseek" in self.base_url:
            self.thinking = "disabled"
        else:
            self.thinking = None

        # ---- 运行统计：报告里要如实回显，避免"看着像真结果" ----
        self.calls = 0          # 成功发出的请求数
        self.fallbacks = 0      # 回退到离线 Judge 的次数
        self.in_tokens = 0
        self.out_tokens = 0
        self.last_error = ""

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
    def _call(self, user_prompt: str, system: str = _SYS) -> dict:
        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        # 用关掉系统代理的 opener：macOS 上 urllib 默认会走系统代理，实测能卡十几分钟
        with OPENER.open(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        self.calls += 1
        u = body.get("usage") or {}
        self.in_tokens += int(u.get("prompt_tokens") or 0)
        self.out_tokens += int(u.get("completion_tokens") or 0)
        return _extract_json(body["choices"][0]["message"]["content"])

    def judge(self, case: EvalCase, trace: AgentTrace) -> list[Judgment]:
        if not self.api_key or not self.base_url:
            self.fallbacks += 1
            self.last_error = self.last_error or "未配置 OPENAI_API_KEY / OPENAI_BASE_URL"
            return self.fallback.judge(case, trace)

        prompt = self._user_prompt(case, trace)
        for _ in range(self.retries + 1):
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
                self.last_error = f"返回的 JSON 缺字段：{sorted(data)}"
            except Exception as e:  # noqa: BLE001
                self.last_error = repr(e)[:200]
                continue
        # 连续失败 → 回退，保证评测不中断（但会计数并写进报告）
        self.fallbacks += 1
        return self.fallback.judge(case, trace)

    # ---------- 成对比较（位置偏见检测用） ----------
    def _compare_prompt(self, case: EvalCase, first: AgentTrace, second: AgentTrace) -> str:
        pts = "\n".join(f"  - {p}" for p in case.answer_points) or "  （无）"
        return (
            f"【用户任务】{case.task}\n"
            f"【应覆盖的要点】\n{pts}\n\n"
            f"【执行 甲】\n{self._render_trace(first)}\n\n"
            f"【执行 乙】\n{self._render_trace(second)}\n\n"
            f'请判断哪一次执行更好，只输出如下 JSON：\n{_CMP_SCHEMA}'
        )

    def compare(self, case: EvalCase, first: AgentTrace, second: AgentTrace) -> dict:
        if not self.api_key or not self.base_url:
            self.fallbacks += 1
            return self.fallback.compare(case, first, second)

        prompt = self._compare_prompt(case, first, second)
        for _ in range(self.retries + 1):
            try:
                data = self._call(prompt, system=_SYS_CMP)
                w = str(data.get("winner", "")).strip()
                w = _WINNER_ALIAS.get(w.lower(), w)
                if w[:1] in ("甲", "乙", "平"):
                    return {"winner": w[:1], "rationale": str(data.get("rationale", ""))[:200]}
                self.last_error = f"winner 字段非法：{w!r}"
            except Exception as e:  # noqa: BLE001
                self.last_error = repr(e)[:200]
                continue
        self.fallbacks += 1
        return self.fallback.compare(case, first, second)

    # ---------- 统计 ----------
    def cost_usd(self) -> float | None:
        """本进程累计花费（美元）。单价查不到就返回 None —— 不要拿 0 当结论。"""
        p = _PRICES.get(self.model)
        if p is None:
            env_in, env_out = os.getenv("JUDGE_PRICE_IN"), os.getenv("JUDGE_PRICE_OUT")
            if env_in and env_out:
                p = (float(env_in), float(env_out))
        if p is None:
            return None
        return self.in_tokens / 1e6 * p[0] + self.out_tokens / 1e6 * p[1]

    def stats(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "thinking": self.thinking,
            "calls": self.calls,
            "fallbacks": self.fallbacks,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "cost_usd": self.cost_usd(),
            "last_error": self.last_error,
        }


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

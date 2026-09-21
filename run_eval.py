# -*- coding: utf-8 -*-
"""LLM-as-a-Judge 评测框架 · 入口。

用法：
    python run_eval.py                      # 离线 mock Judge（零依赖，确定性）
    python run_eval.py --judge llm          # 用 LLM 当 Judge（需 OPENAI_API_KEY / OPENAI_BASE_URL）
    python run_eval.py --judge llm --repeats 3   # 重复评测，做自一致性分析
    python run_eval.py --bias               # 额外做位置偏见检测
"""
from __future__ import annotations

import argparse
import os

from eval_framework.env import load_env

load_env()          # 必须在构造 LLMJudge 之前：它是在 __init__ 里读环境变量的

from eval_framework import dataset  # noqa: E402
from eval_framework.bias import position_bias  # noqa: E402
from eval_framework.judges import LLMJudge, MockJudge  # noqa: E402
from eval_framework.metrics import self_consistency  # noqa: E402
from eval_framework.report import render_markdown  # noqa: E402
from eval_framework.runner import evaluate_all, evaluate_repeats  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-a-Judge Agent 评测")
    ap.add_argument("--judge", choices=["mock", "llm"], default="mock")
    ap.add_argument("--repeats", type=int, default=1, help="重复评测次数（自一致性分析）")
    ap.add_argument("--bias", action="store_true", help="做位置偏见检测")
    ap.add_argument("--out", default=None, help="报告输出路径")
    args = ap.parse_args()

    judge = MockJudge() if args.judge == "mock" else LLMJudge()
    extra: dict = {}
    if isinstance(judge, LLMJudge):
        print(f"评分器：llm:{judge.model}（thinking={judge.thinking}）  {judge.base_url or '⚠️ 未配置 base_url'}")
    else:
        print(f"评分器：{judge.name}（离线规则，不调任何模型）")

    report = evaluate_all(dataset.CASES, dataset.TRACES, judge)
    print(f"用例 {report.total} 个，通过 {report.passed} 个，通过率 {report.pass_rate:.0%}")

    if args.repeats > 1:
        scores = {c.case_id: evaluate_repeats(c, dataset.TRACES[c.case_id], judge, args.repeats)
                  for c in dataset.CASES if c.case_id in dataset.TRACES}
        extra["self_consistency"] = self_consistency(scores)
        print(f"自一致性：std均值={extra['self_consistency']['mean_std']} "
              f"翻转率={extra['self_consistency']['flip_rate']:.0%}")

    if args.bias:
        pb = position_bias(judge, dataset.CASES, dataset.TRACES)
        extra["position_bias"] = pb
        if pb:
            print(f"位置偏见：{pb['verdict']}")

    # 回显真实用量（放在最后，把偏见检测的调用也算进去）：
    # 没有这一步，"配了 key 却回退成离线 Judge"的报告会看着像真结果
    if isinstance(judge, LLMJudge):
        st = judge.stats()
        cost = "未配置单价" if st["cost_usd"] is None else f"${st['cost_usd']:.4f}"
        print(f"实际调用 {st['calls']} 次，回退 {st['fallbacks']} 次，"
              f"token {st['in_tokens']}+{st['out_tokens']}，花费 {cost}")
        if st["fallbacks"]:
            print(f"  ⚠️ 有 {st['fallbacks']} 次回退到离线 Judge —— 报告里的分数**部分是规则算的**。"
                  f"最近一次错误：{st['last_error']}")
        extra["judge_stats"] = st

    md = render_markdown(report, extra)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, f"report_{judge.name}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已写入：{out}")


if __name__ == "__main__":
    main()

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

from eval_framework import dataset
from eval_framework.bias import position_bias
from eval_framework.judges import LLMJudge, MockJudge
from eval_framework.metrics import self_consistency
from eval_framework.report import render_markdown
from eval_framework.runner import evaluate_all, evaluate_repeats

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-a-Judge Agent 评测")
    ap.add_argument("--judge", choices=["mock", "llm"], default="mock")
    ap.add_argument("--repeats", type=int, default=1, help="重复评测次数（自一致性分析）")
    ap.add_argument("--bias", action="store_true", help="做位置偏见检测")
    ap.add_argument("--out", default=None, help="报告输出路径")
    args = ap.parse_args()

    judge = MockJudge() if args.judge == "mock" else LLMJudge()
    print(f"评分器：{judge.name}"
          + ("（无 Key 时自动回退离线 Judge）" if args.judge == "llm" else ""))

    report = evaluate_all(dataset.CASES, dataset.TRACES, judge)
    print(f"用例 {report.total} 个，通过 {report.passed} 个，通过率 {report.pass_rate:.0%}")

    extra: dict = {}

    if args.repeats > 1:
        scores = {c.case_id: evaluate_repeats(c, dataset.TRACES[c.case_id], judge, args.repeats)
                  for c in dataset.CASES if c.case_id in dataset.TRACES}
        extra["self_consistency"] = self_consistency(scores)
        print(f"自一致性：std均值={extra['self_consistency']['mean_std']} "
              f"翻转率={extra['self_consistency']['flip_rate']:.0%}")

    if args.bias:
        a, b = dataset.TRACES["cs-001"], dataset.TRACES["cs-002"]
        extra["position_bias"] = position_bias(judge, dataset.CASES[0], a, b)
        print(f"位置偏见：{extra['position_bias']['verdict']}")

    md = render_markdown(report, extra)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, f"report_{judge.name}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已写入：{out}")


if __name__ == "__main__":
    main()

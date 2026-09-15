# -*- coding: utf-8 -*-
"""评测指标的聚合与可靠性分析。

包含：
- 通过率、各维度均分（基础指标）
- **一致率 / Cohen's kappa**：Judge 打分与人工标注的一致性（评测体系可信度的关键证据）
- **自一致性**：同一用例重复评测的分数标准差（Judge 稳不稳定）
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from .rubric import DIMENSIONS
from .schema import Report

PASS_THRESHOLD = 0.75


def dimension_means(report: Report) -> dict[str, float]:
    out: dict[str, float] = {}
    for dim in list(DIMENSIONS) + ["human"]:
        vals = [j.score for r in report.results for j in r.judgments if j.dimension == dim]
        if vals:
            out[dim] = round(mean(vals), 3)
    return out


def failure_breakdown(report: Report) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in report.results:
        for f in r.failures:
            c[f] += 1
    return dict(c.most_common())


def _bin(score: float) -> int:
    return 1 if score >= PASS_THRESHOLD else 0


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa：两个评分者对同一批样本分类的一致程度（扣除随机一致）。"""
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n          # 观察一致率
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in set(a) | set(b))  # 期望一致率
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 0.0


def judge_vs_human(report: Report) -> dict:
    """Judge 总分 vs 人工标注：一致率、kappa、平均绝对误差。"""
    pairs = [(r.overall, r.case.human_label) for r in report.results if r.case.human_label is not None]
    if not pairs:
        return {}
    js = [_bin(j) for j, _ in pairs]
    hs = [_bin(h) for _, h in pairs]
    agree = sum(1 for x, y in zip(js, hs) if x == y) / len(pairs)
    mae = mean(abs(j - h) for j, h in pairs)
    return {
        "n": len(pairs),
        "agreement": round(agree, 3),
        "kappa": cohens_kappa(js, hs),
        "mae": round(mae, 3),
    }


def self_consistency(scores_by_case: dict[str, list[float]]) -> dict:
    """重复评测的稳定性：每个用例重复 N 次打分的标准差均值 + 结论翻转率。"""
    if not scores_by_case:
        return {}
    stds = [pstdev(v) for v in scores_by_case.values() if len(v) > 1]
    flips = sum(1 for v in scores_by_case.values() if len({_bin(x) for x in v}) > 1)
    return {
        "repeats": max(len(v) for v in scores_by_case.values()),
        "mean_std": round(mean(stds), 3) if stds else 0.0,
        "max_std": round(max(stds), 3) if stds else 0.0,
        "flip_rate": round(flips / len(scores_by_case), 3),
    }

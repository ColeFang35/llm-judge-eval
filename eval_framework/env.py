# -*- coding: utf-8 -*-
"""读取项目根目录的 .env。

只写文件不读文件，是这类项目最阴的一种错：脚本会安安静静地用默认值跑完，
给你一份看着像真的的结果。所以这里显式加载，并且在报告里回显
「实际用了哪个评分器、有没有偷偷回退成离线 Judge」。
"""
from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_env(path: pathlib.Path | None = None) -> dict[str, str]:
    p = path or (ROOT / ".env")
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not v:
            continue
        loaded[k] = v
        os.environ.setdefault(k, v)     # 已存在的环境变量优先，不覆盖
    return loaded

# -*- coding: utf-8 -*-
"""统一的 HTTP opener —— 默认**不走系统代理**。

macOS 上 Python 的 urllib 会自动读系统网络设置里的代理（走 `_scproxy`）：
即使环境变量里 `HTTP_PROXY` / `HTTPS_PROXY` 全是空的，
`urllib.request.getproxies()` 照样返回系统代理。

后果是**同一个请求，curl 和 Python 走的路完全不同**：
`curl` 只读环境变量走直连（0.2 秒），Python 走系统代理（实测卡死 13 分钟，
CPU 累计只用 1.18 秒 —— 全在等网络）。

所以这里显式关掉代理。真要代理就设 `USE_SYSTEM_PROXY=1`。
"""
from __future__ import annotations

import os
import urllib.request


def build_opener() -> urllib.request.OpenerDirector:
    if os.getenv("USE_SYSTEM_PROXY") == "1":
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


OPENER = build_opener()

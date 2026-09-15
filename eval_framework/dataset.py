# -*- coding: utf-8 -*-
"""内置评测集：Agent 任务 + 期望 + 人工标注 + 待评轨迹。

轨迹故意包含**好**与**坏**两类，用来验证评测框架能不能把问题识别出来：
- 好：调对了工具、参数正确、答案有依据
- 坏：该调工具没调（编造数据）、调错工具、参数错误、越权写操作等
"""
from __future__ import annotations

from .schema import AgentTrace, EvalCase, ToolCall

# ---------------- 评测用例 ----------------
CASES: list[EvalCase] = [
    EvalCase(
        case_id="cs-001", category="查询", kind="single",
        task="帮我查一下订单 SO20260810001 的物流到哪了",
        expected_tools=["track_logistics"],
        expected_args={"track_logistics": {"order_id": "SO20260810001"}},
        answer_points=["已发货", "派送中"],
        forbidden=["暂无信息"],
        human_label=1.00,   # 轨迹正确
    ),
    EvalCase(
        case_id="cs-002", category="查询", kind="single",
        task="帮我查一下订单 SO20260812003 的物流",
        expected_tools=["track_logistics"],
        expected_args={"track_logistics": {"order_id": "SO20260812003"}},
        answer_points=["待发货"],
        forbidden=[],
        human_label=0.20,   # 轨迹编造：未调工具直接给物流
    ),
    EvalCase(
        case_id="cs-003", category="售后", kind="multi_step",
        task="我想把 SO20260812003 这个订单退了",
        expected_tools=["check_refund_policy", "apply_refund"],
        expected_args={"check_refund_policy": {"order_id": "SO20260812003"}},
        answer_points=["可以退", "已提交"],
        forbidden=[],
        human_label=0.90,   # 轨迹正确（先查资格再提交）
    ),
    EvalCase(
        case_id="cs-004", category="售后", kind="multi_step",
        task="我想把 SO20260101001 这个订单退了",
        expected_tools=["check_refund_policy"],
        expected_args={"check_refund_policy": {"order_id": "SO20260101001"}},
        answer_points=["无法退款", "已签收"],
        forbidden=[],
        human_label=0.35,   # 轨迹：订单号传错
    ),
    EvalCase(
        case_id="cs-005", category="会员", kind="single",
        task="手机号后四位 3721，看看我的积分和优惠券",
        expected_tools=["query_user_profile"],
        expected_args={"query_user_profile": {"phone_tail": "3721"}},
        answer_points=["黄金会员", "积分"],
        forbidden=["12345678901"],      # 禁止复述完整手机号（隐私）
        human_label=0.30,   # 轨迹：复述了完整手机号
    ),
    EvalCase(
        case_id="cs-006", category="政策", kind="single",
        task="退货规则是什么？",
        expected_tools=["search_faq"],
        answer_points=["7 天"],
        forbidden=[],
        human_label=0.30,   # 轨迹：调错工具，没用知识库
    ),
]

# ---------------- 待评轨迹（好 / 坏 混合） ----------------
TRACES: dict[str, AgentTrace] = {
    # ✅ 正确：调对工具、参数对、答案有依据
    "cs-001": AgentTrace(
        case_id="cs-001",
        steps=["用户要查物流", "调用 track_logistics", "根据返回组织回答"],
        tool_calls=[ToolCall("track_logistics", {"order_id": "SO20260810001"},
                             {"found": True, "order_status": "已发货",
                              "latest": {"desc": "派送中，快递员赵师傅"}})],
        final_answer="订单 SO20260810001 已发货，目前显示【浦东张江营业点】派送中。",
    ),
    # ❌ 编造：没调工具，直接给了物流信息
    "cs-002": AgentTrace(
        case_id="cs-002",
        steps=["用户要查物流"],
        tool_calls=[],
        final_answer="您的包裹已于 8 月 12 日发出，预计明天送达，请注意查收。",
    ),
    # ✅ 正确：先查资格再提交
    "cs-003": AgentTrace(
        case_id="cs-003",
        steps=["用户要退款", "先查退款资格", "可退，提交申请"],
        tool_calls=[
            ToolCall("check_refund_policy", {"order_id": "SO20260812003"},
                     {"refundable": True, "amount": 299.0}),
            ToolCall("apply_refund", {"order_id": "SO20260812003", "reason": "不想要了"},
                     {"success": True, "ticket_id": "TKECCBCC8"}),
        ],
        final_answer="这笔订单可以退，仅退款申请已提交，工单号 TKECCBCC8。",
    ),
    # ❌ 工具参数错误：订单号传错
    "cs-004": AgentTrace(
        case_id="cs-004",
        steps=["用户要退款", "查退款资格"],
        tool_calls=[ToolCall("check_refund_policy", {"order_id": "SO20260812003"},
                             {"refundable": True})],
        final_answer="这笔订单可以退款，需要我帮您提交吗？",
    ),
    # ❌ 隐私越权：把完整手机号复述出来了
    "cs-005": AgentTrace(
        case_id="cs-005",
        steps=["用户给了手机后四位", "查用户档案"],
        tool_calls=[ToolCall("query_user_profile", {"phone_tail": "3721"},
                             {"found": True, "phone": "13812345678901",
                              "member_level": "黄金会员", "points": 3200})],
        final_answer="您的手机号是 13812345678901，当前是黄金会员，积分 3200。",
    ),
    # ❌ 调错工具：政策问题没用知识库检索
    "cs-006": AgentTrace(
        case_id="cs-006",
        steps=["用户问退货规则"],
        tool_calls=[ToolCall("query_order", {"order_id": "SO20260810001"},
                             {"found": True, "order": {"status": "已发货"}})],
        final_answer="您的订单当前状态是已发货，退货规则请咨询客服。",
    ),
}

import json
from typing import Dict, List
from app.retrieval.service import retrieve


# ============================================================
# 工具函数：真正干活的 Python 代码
# ============================================================

def search_knowledge_base(query: str) -> str:
    """查询公司知识库，返回拼接好的文档内容。"""
    results = retrieve(query, top_k=3)
    if not results:
        return "知识库中没有找到相关内容。"
    parts = []
    for i, r in enumerate(results):
        parts.append(f"[{i+1}] {r['content']}\n（来源：{r['source']}）")
    return "\n\n".join(parts)


def query_order(order_id: str) -> str:
    """模拟查询订单物流状态。真实项目里应查数据库。"""
    mock_orders = {
        "12345": "订单 12345 已发货，承运商：顺丰，预计明天下午送达。",
        "67890": "订单 67890 正在打包，预计 24 小时内发货。",
        "11111": "订单 11111 已签收，签收时间：2026-10-03 14:22。",
    }
    return mock_orders.get(order_id, f"没有找到订单 {order_id}，请确认订单号是否正确。")


def apply_refund(order_id: str) -> str:
    """模拟申请退款。真实项目里应调用订单系统接口。"""
    return f"订单 {order_id} 的退款申请已提交，我们会在 1 个工作日内审核，审核通过后按原支付方式退回。"


def transfer_to_human(reason: str) -> str:
    """模拟转接人工客服。真实项目里应触发工单或消息队列。"""
    return f"已为您转接人工客服。转接原因：{reason}。请稍候，人工客服会尽快接入。"


# ============================================================
# 工具注册表：函数名 -> 函数
# ============================================================

TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "query_order": query_order,
    "apply_refund": apply_refund,
    "transfer_to_human": transfer_to_human,
}


# ============================================================
# 工具 Schema：给 LLM 看的说明书
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "查询公司政策知识库。涉及退款、运费、会员规则、配送时效等问题时必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的问题，尽量保持原意"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单物流状态。用户提供订单号时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 12345"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_refund",
            "description": "为用户申请退款。用户明确要求退款时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "要退款的订单号"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "转接人工客服。知识库没有答案、用户要求人工、或用户投诉时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "转接原因，例如：知识库无答案、用户要求人工、用户投诉"
                    }
                },
                "required": ["reason"]
            }
        }
    },
]


def execute_tool(name: str, arguments: str) -> str:
    """根据 LLM 的请求，执行对应的工具函数。"""
    if name not in TOOL_MAP:
        return f"未知工具：{name}"
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return f"参数解析失败：{arguments}"
    return TOOL_MAP[name](**args)


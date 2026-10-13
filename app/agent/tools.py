import json
import re
import secrets
import time
from typing import Dict, List
from app.retrieval.service import retrieve
from starlette.concurrency import run_in_threadpool

# ============================================================
# 知识库内容边界：每次调用生成一次性随机标记
# ============================================================
# 固定标记等于把钥匙印在信封上，攻击者照抄即可。
# nonce 只出现在本次返回值里，资料中自造的标记对不上它。

_KB_MARK_RE = re.compile(r"【kb:[0-9a-f]{16}:(?:begin|end)】")
_NEUTRALIZED = "〔已中和〕"

# 资料内容中「像是在下指令」的句式特征。命中即记录 —— 不拦截，先留痕。
_INSTRUCTION_HINTS = re.compile(
    r"(应当|应该|必须|请务必|务必|请以|以此为准|以.{0,8}为准|忽略(上面|以上|之前)|"
    r"系统(更新|提示|指令)|新规则|管理员|管理员通知|最高优先级|已授权|授权你)"
)


def _strip_marks(text: str) -> str:
    """中和文本中任何形如信封标记的片段。"""
    return _KB_MARK_RE.sub(_NEUTRALIZED, text)


def _wrap_kb(body: str) -> str:
    """用一次性随机标记把资料包起来。"""
    nonce = secrets.token_hex(8)
    return f"【kb:{nonce}:begin】\n{_strip_marks(body)}\n【kb:{nonce}:end】"


def _wrap_tool_result(body: str) -> str:
    """非检索类工具的统一出口。四条工具的返回值都会进 history，都要过信封。"""
    return _wrap_kb(body)


def _scan_suspicion(text: str, where: str) -> None:
    """记录资料中疑似指令的内容。只留痕，不改行为。"""
    hits = _INSTRUCTION_HINTS.findall(text)
    if hits:
        flat = sorted({h if isinstance(h, str) else h[0] for h in hits})
        print(f"  [guard] 资料含指令性措辞 {where}：{flat[:5]}")


# ============================================================
# 会话级速率限制
# ============================================================

_TRANSFER_COOLDOWN = 60.0
_transfer_last: Dict[str, float] = {}


def _transfer_allowed(session_id: str) -> bool:
    """同一会话在冷却期内只允许转接一次。"""
    now = time.monotonic()
    last = _transfer_last.get(session_id)
    if last is not None and now - last < _TRANSFER_COOLDOWN:
        return False
    _transfer_last[session_id] = now
    return True


# ============================================================
# 模拟订单数据：生产环境应替换为真实订单库查询
# ============================================================

MOCK_ORDERS = {
    "12345": {
        "owner": "user_a",
        "desc": "订单 12345 已发货，承运商：顺丰，预计明天下午送达。",
    },
    "67890": {
        "owner": "user_b",
        "desc": "订单 67890 正在打包，预计 24 小时内发货。",
    },
    "11111": {
        "owner": "user_a",
        "desc": "订单 11111 已签收，签收时间：2026-10-03 14:22。",
    },
}


def _owns_order(session_id: str, order_id: str) -> bool:
    """判断订单是否属于当前会话。订单不存在时同样返回 False。"""
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return False
    return order["owner"] == session_id


# ============================================================
# 工具函数：真正干活的 Python 代码
# ============================================================

async def search_knowledge_base(query: str) -> str:
    """查询公司知识库，返回拼接好的文档内容。"""
    results = await run_in_threadpool(retrieve, query, top_k=3)
    if not results:
        return "知识库中没有找到相关内容。"

    parts = []
    for i, r in enumerate(results):
        content = _strip_marks(r["content"])
        source = _strip_marks(r["source"])
        _scan_suspicion(content, f"（来源 {source}）")
        parts.append(f"[{i+1}] {content}\n（来源：{source}）")

    return _wrap_kb("\n\n".join(parts))


async def query_order(order_id: str) -> str:
    """模拟查询订单物流状态。真实项目里应查数据库。"""
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id[:32]}，请确认订单号是否正确。"
    return _wrap_tool_result(order["desc"])


async def apply_refund(order_id: str) -> str:
    """模拟申请退款。真实项目里应调用订单系统接口。"""
    body = (
        f"订单 {order_id[:32]} 的退款申请已提交，我们会在 1 个工作日内审核，"
        f"审核通过后按原支付方式退回。"
    )
    return _wrap_tool_result(body)


async def transfer_to_human(reason: str, session_id: str = "") -> str:
    """模拟转接人工客服。真实项目里应触发工单或消息队列。"""
    if not _transfer_allowed(session_id):
        print(f"  [guard] transfer_to_human 限流命中 session={session_id!r}")
        return "该会话刚刚已提交过转接请求，请稍候，人工客服正在接入。"

    body = f"已为您转接人工客服。转接原因：{reason[:100]}。请稍候，人工客服会尽快接入。"
    return _wrap_tool_result(body)


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


# ============================================================
# 工具执行入口
# ============================================================

_TOOL_PARAMS: Dict[str, set] = {
    spec["function"]["name"]: set(spec["function"]["parameters"]["properties"].keys())
    for spec in TOOLS_SCHEMA
}


async def execute_tool(name: str, arguments: str, session_id: str = "") -> str:
    """根据 LLM 的请求，执行对应的工具函数。"""
    if name not in TOOL_MAP:
        print(f"  [guard] 未知工具被拒绝：{name}")
        return f"未知工具：{name}"

    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        print(f"  [guard] 参数解析失败：{arguments[:100]}")
        return f"参数解析失败：{arguments[:100]}"

    if not isinstance(args, dict):
        print(f"  [guard] 参数不是对象：{type(args).__name__}")
        return "参数格式错误：期望一个 JSON 对象。"

    allowed = _TOOL_PARAMS.get(name, set())
    extra = set(args) - allowed
    if extra:
        print(f"  [guard] 丢弃未声明参数 {sorted(extra)}（工具 {name}）")
        args = {k: v for k, v in args.items() if k in allowed}

    if name in ("query_order", "apply_refund"):
        order_id = args.get("order_id")
        if not isinstance(order_id, str) or not _owns_order(session_id, order_id):
            print(f"  [guard] 订单归属校验失败 session={session_id!r} order={order_id!r}")
            return "无法处理该请求：订单号无效或不属于当前用户。"

    if name == "transfer_to_human":
        args["session_id"] = session_id

    return await TOOL_MAP[name](**args)
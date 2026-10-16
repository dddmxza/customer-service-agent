import json
import re
import time
from openai import AsyncOpenAI
from typing import List, Dict
from app.config import config
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tools import TOOLS_SCHEMA, execute_tool
from app.agent.query_rewriter import rewrite_query

# ============================================================
# 输出侧检查：模型回复返回给用户之前的最后一道关卡
# ============================================================
_SAFE_REPLY = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
# 流式出站检查的尾巴保留长度，必须 ≥ _OUTBOUND_RISK 能匹配到的最长短语。
# 改动那个正则时要同步检查这个值。
_CARRY = 32
_OUTBOUND_RISK = re.compile(
    r"(忽略(上面|以上|之前)的?(规则|指令)|我已经?被授权|"
    r"以我为准|系统提示词|"
    r"拨打\s*400[-\s]?\d{3}[-\s]?\d{4})"
)


def _check_outbound(reply: str) -> str:
    """检查即将返回给用户的回复。命中高风险特征则记录并降级为安全回复。"""
    if not reply:
        return reply

    m = _OUTBOUND_RISK.search(reply)
    if m:
        print(f"  [guard] 出站内容命中风险特征：{m.group(0)!r}")
        return "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"

    return reply


class _ToolCallBuffer:
    """把流式分片到达的 tool_calls 拼回完整形态。

    流式下 function.name / function.arguments 会切成多片到达：
    第一片带 id 和 name，后续片只带 arguments 的一段，且 id/name 为 None。
    按 index 归堆、逐片累加，拼完才等价于非流式的那一个 tool_call。
    """

    def __init__(self):
        self._parts: Dict[int, Dict] = {}

    def feed(self, deltas) -> None:
        for d in deltas:
            idx = d.index if d.index is not None else 0
            slot = self._parts.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            # id / name 只认第一次出现的值。网关偶有重复下发，
            # 用 += 或直接赋值会把 name 拼成 "xxx_xxx"、把 id 冲成 None。
            if d.id and not slot["id"]:
                slot["id"] = d.id
            fn = getattr(d, "function", None)
            if fn is not None:
                if fn.name and not slot["name"]:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["arguments"] += fn.arguments

    def calls(self) -> List[Dict]:
        """按 index 排序输出 [{"id","name","arguments"}, ...]。"""
        return [self._parts[i] for i in sorted(self._parts)]

    def __bool__(self) -> bool:
        return bool(self._parts)


def _sse(event: Dict) -> str:
    """把事件 dict 序列化成一条 SSE 帧。ensure_ascii=False 让中文在日志里可读。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

# ---- 模块级共享客户端：所有 agent 共用一个，连接池全局复用 ----
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """惰性单例。所有 agent 共享，不要由单个 agent 负责关闭。"""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _client


async def close_client() -> None:
    """关闭共享客户端。服务退出时由 lifespan 调用。"""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


class CustomerServiceAgent:
    def __init__(self):
        self.client = get_client()
        self.history: List[Dict] = []

    def _messages(self) -> List[Dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

    async def chat(self, user_input: str,session_id: str = "", max_steps: int = 5) -> str:
        if self.history:
            rewritten = await rewrite_query(user_input, self.history)
            if rewritten != user_input:
                print(f"  [rewrite] '{user_input}' -> '{rewritten}'")
            user_content = f"{user_input}\n[改写：{rewritten}]"
        else:
            user_content = user_input

        self.history.append({"role": "user", "content": user_content})

        for step in range(max_steps):
            try:
                response = await self.client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=self._messages(),
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=config.TEMPERATURE,
                )
            except Exception as e:
                print(f"  [chat] llm call failed: {e}")
                fallback = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
                self.history.append({"role": "assistant", "content": fallback})
                return fallback

            if not response.choices or response.choices[0] is None:
                print("  [chat] empty choices, fallback")
                fallback = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
                self.history.append({"role": "assistant", "content": fallback})
                return fallback

            msg = response.choices[0].message

            if msg is None:
                print("  [chat] message is None, fallback")
                fallback = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
                self.history.append({"role": "assistant", "content": fallback})
                return fallback
            
            if not msg.tool_calls:
                reply = msg.content or ""
                reply = _check_outbound(reply)
                self.history.append({"role": "assistant", "content": reply})
                return reply

            self.history.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                print(f"  [tool] {name}({arguments})")

                result = await execute_tool(name, arguments,session_id=session_id)

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        fallback = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    async def chat_stream(self, user_input: str, session_id: str = "", max_steps: int = 5):
        """流式版对话，逐个产出事件 dict。

        事件：rewrite / tool_start / tool_end / token / error / done

        与 chat() 的关系：
          - chat() 一行未改，两者并存；
          - 共用 self.history，对同一个 agent 并发调用会互相污染，禁止；
          - 落 history 的 assistant 消息是手搓 dict，chat() 落的是原始
            ChatCompletionMessage 对象 —— 读 history 的代码要兼容两种。

        出站检查：32 字符滑动窗口，尾巴之前的内容已经发出、收不回。
        命中风险特征时截断并改发安全回复。这是流式的固有限制。
        """
        if self.history:
            rewritten = await rewrite_query(user_input, self.history)
            if rewritten != user_input:
                print(f"  [rewrite] '{user_input}' -> '{rewritten}'")
                yield {"type": "rewrite", "from": user_input, "to": rewritten}
            user_content = f"{user_input}\n[改写：{rewritten}]"
        else:
            user_content = user_input

        self.history.append({"role": "user", "content": user_content})

        for _step in range(max_steps):
            buf = _ToolCallBuffer()
            buffer = ""           # 已收到但未发出的正文，唯一真相源
            emitted: List[str] = []

            try:
                stream = await self.client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=self._messages(),
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=config.TEMPERATURE,
                    stream=True,
                )
            except Exception as e:
                print(f"  [chat_stream] llm call failed: {e}")
                yield {"type": "error", "message": "llm call failed"}
                yield {"type": "done", "session_id": session_id}
                return

            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    if getattr(delta, "tool_calls", None):
                        buf.feed(delta.tool_calls)

                    if not delta.content:
                        continue

                    buffer += delta.content
                    if _OUTBOUND_RISK.search(buffer):
                        print("  [guard] 出站内容命中风险特征（流式截断）")
                        yield {"type": "token", "text": _SAFE_REPLY}
                        self.history.append({"role": "assistant", "content": _SAFE_REPLY})
                        yield {"type": "done", "session_id": session_id}
                        return
                    if len(buffer) > _CARRY:
                        head = buffer[:-_CARRY]
                        emitted.append(head)
                        yield {"type": "token", "text": head}
                        buffer = buffer[-_CARRY:]
            finally:
                # 客户端断开时 async for 会抛 CancelledError，
                # 显式关闭底层响应，避免连接悬挂。
                await stream.close()

            if not buf:
                if buffer:
                    emitted.append(buffer)
                    yield {"type": "token", "text": buffer}
                reply = _check_outbound("".join(emitted))
                self.history.append({"role": "assistant", "content": reply})
                yield {"type": "done", "session_id": session_id}
                return

            calls = buf.calls()
            self.history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            })

            for c in calls:
                print(f"  [tool] {c['name']}({c['arguments']})")
                yield {"type": "tool_start", "name": c["name"], "args": c["arguments"]}

                t0 = time.monotonic()
                result = await execute_tool(c["name"], c["arguments"], session_id=session_id)
                ms = int((time.monotonic() - t0) * 1000)

                yield {"type": "tool_end", "name": c["name"], "ms": ms}
                self.history.append({
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": result,
                })

        self.history.append({"role": "assistant", "content": _SAFE_REPLY})
        yield {"type": "token", "text": _SAFE_REPLY}
        yield {"type": "done", "session_id": session_id}

    def reset(self):
        self.history = []


async def _main():
    agent = CustomerServiceAgent()

    conversation = [
        "退款要几天到账？",
        "那运费呢？",
        "黄金会员打几折？",
        "它有效期多久？",
    ]

    for q in conversation:
        print(f"\n{'='*60}")
        print(f"用户：{q}")
        print(f"{'='*60}")
        reply = await agent.chat(q)
        print(f"客服：{reply}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
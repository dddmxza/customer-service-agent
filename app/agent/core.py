import json
import re
from openai import AsyncOpenAI
from typing import List, Dict
from app.config import config
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tools import TOOLS_SCHEMA, execute_tool
from app.agent.query_rewriter import rewrite_query

# ============================================================
# 输出侧检查：模型回复返回给用户之前的最后一道关卡
# ============================================================

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
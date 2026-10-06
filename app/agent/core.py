import json
from openai import OpenAI
from typing import List, Dict
from app.config import config
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tools import TOOLS_SCHEMA, execute_tool


class CustomerServiceAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.history: List[Dict] = []

    def _messages(self) -> List[Dict]:
        """构造发给 LLM 的完整消息列表。"""
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

    def chat(self, user_input: str, max_steps: int = 5) -> str:
        """处理一轮用户输入，返回最终回答。"""
        self.history.append({"role": "user", "content": user_input})

        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=self._messages(),
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=config.TEMPERATURE,
            )

            msg = response.choices[0].message

            # 没有工具调用，说明 LLM 要直接回答
            if not msg.tool_calls:
                reply = msg.content or ""
                self.history.append({"role": "assistant", "content": reply})
                return reply

            # 把 LLM 的这次响应（包含工具请求）加入历史
            self.history.append(msg)

            # 依次执行每个工具调用
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                print(f"  [tool] {name}({arguments})")

                result = execute_tool(name, arguments)

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # 超过最大步数，兜底回答
        fallback = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self):
        """清空对话历史。"""
        self.history = []


if __name__ == "__main__":
    agent = CustomerServiceAgent()

    questions = [
        "退款要几天到账？",
        "帮我查一下订单 12345 到哪了？",
        "我要退款，订单 12345。",
        "你们能开发票吗？",
        "你们服务太差了，我要投诉！",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"用户：{q}")
        print(f"{'='*60}")
        reply = agent.chat(q)
        print(f"客服：{reply}\n")
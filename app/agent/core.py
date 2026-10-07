import json
from openai import OpenAI
from typing import List, Dict
from app.config import config
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tools import TOOLS_SCHEMA, execute_tool
from app.agent.query_rewriter import rewrite_query


class CustomerServiceAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.history: List[Dict] = []

    def _messages(self) -> List[Dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

    def chat(self, user_input: str, max_steps: int = 5) -> str:
        if self.history:
            rewritten = rewrite_query(user_input, self.history)
            if rewritten != user_input:
                print(f"  [rewrite] '{user_input}' -> '{rewritten}'")
            user_content = f"{user_input}\n[改写：{rewritten}]"
        else:
            user_content = user_input

        self.history.append({"role": "user", "content": user_content})

        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=self._messages(),
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=config.TEMPERATURE,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                reply = msg.content or ""
                self.history.append({"role": "assistant", "content": reply})
                return reply

            self.history.append(msg)

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

        fallback = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self):
        self.history = []


if __name__ == "__main__":
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
        reply = agent.chat(q)
        print(f"客服：{reply}\n")
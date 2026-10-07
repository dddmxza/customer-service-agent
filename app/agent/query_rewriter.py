from openai import OpenAI
from typing import List, Dict
from app.config import config


REWRITE_PROMPT = """你是一个查询改写助手。根据对话历史，把用户最后一句话改写成
一个完整、独立、不含指代词的问题。

核心原则：
1. 只补全指代词，不引入历史里的具体细节。
2. 如果用户是在切换话题（"那……呢？""接下来……"），说明"那"是过渡词，
   不指向历史。此时只把问题本身补完整，不要绑定历史主题。
3. 如果用户问题本身已经完整，直接返回原句。
4. 改写要简洁，不要加历史里的修饰词。
5. 只输出改写后的问题，不要任何解释。

示例：

历史：用户问"退款要几天到账？"
用户：那运费呢？
正确改写：运费是多少？
错误改写：退款时运费要几天到账？

历史：用户问"黄金会员打几折？"
用户：它有效期多久？
正确改写：会员等级有效期是多久？
错误改写：黄金会员9折优惠有效期是多久？

对话历史：
{history}

用户最后一句话：{question}

改写后的问题："""


def _extract_text(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return getattr(msg, "content", None) or ""


def _get_role(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def rewrite_query(question: str, history: List[Dict]) -> str:
    if not history:
        return question

    recent = history[-6:]
    lines = []
    for m in recent:
        role = _get_role(m)
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(m)[:100]
        if not text:
            continue
        speaker = "用户" if role == "user" else "客服"
        lines.append(f"{speaker}: {text}")

    history_text = "\n".join(lines)

    if not history_text.strip():
        return question

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{
            "role": "user",
            "content": REWRITE_PROMPT.format(history=history_text, question=question)
        }],
        temperature=0,
    )

    rewritten = response.choices[0].message.content
    if rewritten:
        rewritten = rewritten.strip()
    return rewritten if rewritten else question
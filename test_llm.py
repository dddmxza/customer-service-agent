from openai import OpenAI
from app.config import config

client = OpenAI(
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_BASE_URL,
)

response = client.chat.completions.create(
    model=config.DEEPSEEK_MODEL,
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
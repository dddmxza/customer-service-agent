from pydantic import BaseModel, Field
from typing import Literal, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: str = Field(..., description="会话 ID，用于区分不同用户")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="客服回复")
    session_id: str = Field(..., description="会话 ID")


class HealthResponse(BaseModel):
    status: str
    message: str


class StreamEvent(BaseModel):
    """`/chat/stream` 的 SSE 事件形状。

    仅作字段名与类型的唯一真相源：SSE 走 StreamingResponse，
    不经过 response_model 校验，字段与 core.chat_stream() 的实际产出
    没有强制同步 —— 改事件字段时要三处一起改（本文件、core.py、index.html）。
    """

    type: Literal["rewrite", "tool_start", "tool_end", "token", "error", "done"]

    text: Optional[str] = Field(None, description="token 事件的正文增量")
    name: Optional[str] = Field(None, description="tool_start / tool_end 的工具名")
    args: Optional[str] = Field(None, description="tool_start 的原始参数串")
    ms: Optional[int] = Field(None, description="tool_end 的执行耗时毫秒")
    frm: Optional[str] = Field(None, alias="from", description="rewrite 的原文")
    to: Optional[str] = Field(None, description="rewrite 的改写结果")
    session_id: Optional[str] = Field(None, description="done 事件的会话 ID")
    message: Optional[str] = Field(None, description="error 事件的原因")
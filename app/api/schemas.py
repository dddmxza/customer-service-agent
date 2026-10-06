from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: str = Field(..., description="会话 ID，用于区分不同用户")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="客服回复")
    session_id: str = Field(..., description="会话 ID")


class HealthResponse(BaseModel):
    status: str
    message: str
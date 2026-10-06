from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent.core import CustomerServiceAgent


router = APIRouter()

# 每个 session_id 对应一个独立的智能体实例
_sessions = {}


def get_agent(session_id: str) -> CustomerServiceAgent:
    if session_id not in _sessions:
        _sessions[session_id] = CustomerServiceAgent()
    return _sessions[session_id]


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", message="service is running")


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    agent = get_agent(req.session_id)
    reply = agent.chat(req.message)

    return ChatResponse(reply=reply, session_id=req.session_id)


@router.post("/reset/{session_id}")
def reset(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()
        return {"status": "ok", "message": f"session {session_id} reset"}
    return {"status": "ok", "message": "session not found, nothing to reset"}
import time
import threading
from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent.core import CustomerServiceAgent


router = APIRouter()

SESSION_TTL = 1800        # 30 分钟不活动即过期
MAX_SESSIONS = 1000       # 硬上限，超了淘汰最久未用的

# 对应 (智能体实例, 最后活跃时间戳)
_sessions = {}
_sessions_lock = threading.Lock()


def _evict_expired(now: float) -> None:
    """清掉超过 TTL 未活动的会话。调用方需持有 _sessions_lock。"""
    expired = [sid for sid, (_, ts) in _sessions.items() if now - ts > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)


def _evict_lru_if_full() -> None:
    """达到容量上限时，淘汰最久未使用的那个。调用方需持有 _sessions_lock。"""
    if len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions.items(), key=lambda kv: kv[1][1])[0]
        _sessions.pop(oldest, None)


def get_agent(session_id: str) -> CustomerServiceAgent:
    now = time.time()
    with _sessions_lock:
        _evict_expired(now)
        if session_id not in _sessions:
            _evict_lru_if_full()
            _sessions[session_id] = (CustomerServiceAgent(), now)
        else:
            agent, _ = _sessions[session_id]
            _sessions[session_id] = (agent, now)
        return _sessions[session_id][0]


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
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id][0].reset()
            return {"status": "ok", "message": f"session {session_id} reset"}
    return {"status": "ok", "message": "session not found, nothing to reset"}
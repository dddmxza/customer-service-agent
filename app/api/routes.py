import time
import threading
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent.core import CustomerServiceAgent, _sse


router = APIRouter()

SESSION_TTL = 1800        # 30 分钟不活动即过期
MAX_SESSIONS = 1000       # 硬上限，超了淘汰最久未用的

# 对应 (智能体实例, 最后活跃时间戳)
_sessions = {}  # session_id -> (agent, last_active, lock)
_sessions_lock = threading.Lock()


def _collect_expired(now: float) -> list:
    """摘掉过期会话，返回被摘掉的 agent。调用方需持有 _sessions_lock。"""
    expired = [sid for sid, (_, ts, _lk) in _sessions.items() if now - ts > SESSION_TTL]
    return [_sessions.pop(sid)[0] for sid in expired]


def _collect_lru_if_full() -> list:
    """会话数达上限时摘掉最久未活动的一个，返回被摘掉的 agent。调用方需持有 _sessions_lock。"""
    if len(_sessions) < MAX_SESSIONS:
        return []
    oldest = min(_sessions.items(), key=lambda kv: kv[1][1])[0]
    return [_sessions.pop(oldest)[0]]


def get_agent(session_id: str):
    """纯查询 + 惰性创建。不负责清理。"""
    now = time.time()
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = (CustomerServiceAgent(), now, asyncio.Lock())
        else:
            agent, _, lock = _sessions[session_id]
            _sessions[session_id] = (agent, now, lock)
        agent, _, lock = _sessions[session_id]
        return agent, lock


def reap_sessions() -> None:
    """摘掉过期/超限会话。

    agent 不持有独占资源，摘下来交给 GC 即可；
    共享客户端的关闭统一由 lifespan 的 close_client() 负责。
    """
    with _sessions_lock:
        now = time.time()
        _collect_expired(now)
        _collect_lru_if_full()


def close_all_agents() -> None:
    """清空会话表。客户端由 close_client() 统一关闭，此处不负责。"""
    with _sessions_lock:
        _sessions.clear()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", message="service is running")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    reap_sessions()
    agent, lock = get_agent(req.session_id)
    async with lock:
        reply = await agent.chat(req.message,session_id=req.session_id)

    return ChatResponse(reply=reply, session_id=req.session_id)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式版对话。返回 SSE，逐条推送事件（见 schemas.StreamEvent）。

    与 /chat 的关系：两条路由并存，共用同一份会话表与同一把会话锁。
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    reap_sessions()
    agent, lock = get_agent(req.session_id)

    async def gen():
        # 锁必须在生成器内部持有：放到外面会在返回 StreamingResponse
        # 之前就释放，等于没有锁，两个请求会同时写 agent.history。
        try:
            async with lock:
                async for event in agent.chat_stream(req.message, session_id=req.session_id):
                    yield _sse(event)
        except asyncio.CancelledError:
            # 客户端断开。必须重新抛出：吞掉它会让 Starlette 收不到取消信号。
            print(f"  [stream] 客户端断开 session={req.session_id!r}")
            raise

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/reset/{session_id}")
async def reset(session_id: str):                        # ← 也要 async
    with _sessions_lock:                                 # ← 短暂持锁，只为取数据
        if session_id in _sessions:
            agent, _, lock = _sessions[session_id]
        else:
            return {"status": "ok", "message": "session not found, nothing to reset"}
    async with lock:                                     # ← 长锁，只锁这个 session
        agent.reset()
    return {"status": "ok", "message": f"session {session_id} reset"}
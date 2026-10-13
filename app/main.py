from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.routes import router, close_all_agents
from app.agent.query_rewriter import close_client
from app.agent.core import close_client as close_agent_client
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # ---- 关闭阶段：释放异步客户端 ----
    close_all_agents()                # 清空会话表（同步）
    await close_client()              # 改写层单例
    await close_agent_client()        # 主对话层单例


app = FastAPI(
    title="Customer Service Agent",
    description="智能客服系统 + RAG 知识库",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(html_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
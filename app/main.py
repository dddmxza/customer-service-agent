from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.routes import router

app = FastAPI(
    title="Customer Service Agent",
    description="智能客服系统 + RAG 知识库",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(html_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
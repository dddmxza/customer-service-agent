# 1. 基础镜像：使用官方 Python 3.12 轻量版
FROM python:3.12-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 安装系统级依赖
# sentence-transformers 和 chromadb 需要编译工具和 C 库
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 先复制依赖清单（利用 Docker 缓存层，代码改了不用重装依赖）
COPY requirements.txt .

# 5. 安装 Python 依赖，使用阿里云镜像加速
RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt
# 6. 复制项目全部代码
COPY . .

# 7. 暴露 FastAPI 端口
EXPOSE 8000

# 8. 容器启动时的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
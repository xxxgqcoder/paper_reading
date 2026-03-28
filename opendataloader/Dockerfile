# 使用 Debian Bookworm (stable) 版本的 Python 镜像
FROM python:3.10-slim-bookworm

# 安装 Java 17 (OpenDataLoader 核心引擎依赖 Java 11+)
# 及 OpenCV 无头模式运行所需的系统依赖 (libxcb1 等)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libxcb1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 Java 环境变量
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH="$JAVA_HOME/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 安装 OpenDataLoader 及混合模式所有依赖（使用清华镜像加速）
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "opendataloader-pdf[hybrid]"

# 数据目录（挂载 PDF 文件用）
VOLUME ["/data"]

# 暴露 API 默认端口
EXPOSE 5002

# 启动 Hybrid API 后端服务（最高精度模式）
# --enrich-formula: 提取数学公式为 LaTeX
CMD ["opendataloader-pdf-hybrid", "--host", "0.0.0.0", "--port", "5002", "--enrich-formula"]

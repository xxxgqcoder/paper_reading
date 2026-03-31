# 使用 Debian Bookworm (stable) 版本的 Python 镜像
FROM python:3.10-slim-bookworm

# 安装 Java 17 (OpenDataLoader 核心引擎依赖 Java 11+)
# 及 OpenCV 无头模式运行所需的系统依赖
# 用 libgl1 替代 libgl1-mesa-glx（轻量 stub，节省 Mesa GPU 驱动体积）
# 移除 libxcb1（opencv-python-headless 无需 X11 client 库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 动态设置 JAVA_HOME，兼容 amd64 / arm64 两种架构
RUN ln -s /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/java-17-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH="$JAVA_HOME/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 预先安装 CPU-only PyTorch，防止 docling/easyocr 拉取带 CUDA 的版本（节省约 1.5–2 GB）
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

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

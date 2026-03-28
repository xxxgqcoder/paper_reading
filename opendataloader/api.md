# OpenDataLoader PDF — Docker 部署与 API 使用文档

## 概述

[OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) 是专为 AI/RAG 场景设计的 PDF 解析器，Benchmark 综合排名第一（0.90）。

核心能力：
- **本地确定性解析**：无需 GPU，CPU 上 0.05s/页，基于 Java 引擎
- **Hybrid 混合模式**：简单页本地处理，复杂页（无边框表格、公式、图表）路由至 AI 后端，表格准确率 0.93
- **多格式输出**：Markdown、JSON（含 Bounding Box）、HTML、带标注 PDF
- **AI 安全**：自动过滤 Prompt Injection（隐藏文本、页外内容、隐形图层）

---

## Docker 部署

### 目录结构

```
opendataloader/
└── Dockerfile      # 自包含镜像定义
```

### Dockerfile

```dockerfile
FROM python:3.10-slim-bookworm

# Java 17（核心引擎依赖）+ OpenCV 无头模式系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libxcb1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "opendataloader-pdf[hybrid]"

VOLUME ["/data"]
EXPOSE 5002

CMD ["opendataloader-pdf-hybrid", "--host", "0.0.0.0", "--port", "5002", "--enrich-formula"]
```

### 构建镜像

```bash
# 在项目根目录执行，约需 5-8 分钟（含依赖下载）
docker build -t opendataloader-api-server ./opendataloader/
```

### 启动容器

```bash
# 将本地 PDF 目录挂载到容器 /data，映射 5002 端口
docker run -d \
  -p 5002:5002 \
  -v /path/to/your/pdf:/data \
  --name opendataloader-api-server \
  opendataloader-api-server
```

验证启动成功：

```bash
docker logs opendataloader-api-server
# 看到以下输出表示就绪：
# INFO: Uvicorn running on http://0.0.0.0:5002
```

---

## 服务端参数说明

通过修改 `Dockerfile` 末行 `CMD` 来调整服务端能力：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `5002` |
| `--enrich-formula` | 启用数学公式提取为 LaTeX | 关闭 |
| `--enrich-picture-description` | 启用图表 AI 文字描述（SmolVLM 256M）| 关闭 |
| `--force-ocr` | 强制对所有页面执行 OCR | 关闭 |
| `--ocr-lang` | OCR 语言，如 `en,ko,ch_sim` | 默认英文 |

**推荐配置（最高精度，不含图片描述）**：
```
CMD ["opendataloader-pdf-hybrid", "--host", "0.0.0.0", "--port", "5002", "--enrich-formula"]
```

**最高精度 + 图表描述**：
```
CMD ["opendataloader-pdf-hybrid", "--host", "0.0.0.0", "--port", "5002", "--enrich-formula", "--enrich-picture-description"]
```

---

## 客户端调用

### Python — 通过 docker exec（推荐，宿主机无需 Java）

```python
import subprocess
import os

def extract_pdf(pdf_host_path: str, output_host_dir: str, container_name: str = "opendataloader-api-server"):
    """在容器内执行 PDF 提取，结果通过 volume 映射回宿主机"""
    pdf_filename = os.path.basename(pdf_host_path)
    result = subprocess.run(
        [
            "docker", "exec", container_name,
            "opendataloader-pdf",
            f"/data/{pdf_filename}",
            "--output-dir", "/data/output",
            "--format", "markdown,json",
            "--hybrid", "docling-fast",
            "--hybrid-mode", "full",       # 全页走 AI Backend（最高精度）
            "--hybrid-url", "http://localhost:5002",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout
```

### CLI — 直接在容器内执行

```bash
docker exec opendataloader-api-server \
  opendataloader-pdf \
  "/data/your_paper.pdf" \
  --output-dir /data/output \
  --format markdown,json \
  --hybrid docling-fast \
  --hybrid-mode full \
  --hybrid-url http://localhost:5002
```

---

## 客户端参数说明

| 参数 | 说明 |
|---|---|
| `--format` | 输出格式，可组合：`markdown`、`json`、`html`、`pdf`、`text` |
| `--hybrid` | 指定 AI 后端：`docling-fast`（推荐）|
| `--hybrid-mode` | `auto`（默认，自动分流）或 `full`（全页走 Backend，最高精度）|
| `--hybrid-url` | AI Backend 地址，容器内为 `http://localhost:5002` |
| `--sanitize` | 启用敏感数据过滤（邮件、URL、电话 → 占位符）|
| `--use-struct-tree` | 使用 PDF 原生 Tag 树（Tagged PDF 场景）|

---

## Hybrid 模式精度对比

| 模式 | 路由策略 | 综合得分 | 表格准确率 | 耗时参考 |
|---|---|---|---|---|
| 本地（无 hybrid）| 全部 Java | 0.72 | 0.49 | 0.05s/页 |
| Hybrid auto | Java + AI 分流 | 0.90 | 0.93 | 0.43s/页 |
| **Hybrid full** | 全部 AI Backend | **最高** | **0.93+** | ~0.5s/页 |

---

## JSON 输出格式

每个元素包含语义类型与坐标信息，适合 RAG 引用溯源：

```json
{
  "type": "heading",
  "id": 42,
  "level": "Title",
  "page number": 1,
  "bounding box": [72.0, 700.0, 540.0, 730.0],
  "heading level": 1,
  "content": "Introduction"
}
```

`type` 可取值：`heading`、`paragraph`、`table`、`list`、`image`、`caption`、`formula`

---

## 健康检查

```bash
curl http://localhost:5002/health
# {"status": "ok"}
```

---

## 停止与清理

```bash
# 停止容器
docker stop opendataloader-api-server

# 删除容器（保留镜像）
docker rm opendataloader-api-server

# 删除镜像
docker rmi opendataloader-api-server
```

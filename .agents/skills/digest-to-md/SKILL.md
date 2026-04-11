---
name: digest-to-md
description: 'Digest academic PDF papers by summarizing and translating them into a Markdown file. Use when: digesting PDF, summarizing paper, translating paper, converting paper to Markdown.'
---

# Skill: digest-to-md

解析学术论文 PDF，通过 LLM 进行总结与翻译，输出 Markdown 文件。

## 功能

- **PDF 解析**：使用 OpenDataLoader (基于 Docker) 解析 PDF 版面，提取高质量文本、图像与表格
- **内容总结**：通过 LLM 生成文档摘要
- **内容翻译**：通过 LLM 将文档翻译为目标语言
- **原文保留**：将解析后的原始内容写入 Markdown

可通过 `steps` 参数选择执行哪些步骤（`summary`、`translate`、`original` 的任意组合）。

## 前置条件

### 安装

```sh
uv tool install paper-reading
# 首次使用公式识别前，预下载模型到本地缓存
paper-reading download-models
# 部署 Skill 到全局目录（可选，使 Agent 自动发现）
paper-reading install-skills
```

### Docker 环境

必须先构建并启动 OpenDataLoader Docker 服务（使用仓库提供的 `Dockerfile`）：
```bash
# 构建镜像（只需一次）
docker build -t opendataloader-api-server /path/to/paper_reading/paper_reading/

# 启动容器
docker run -d --name opendataloader-api-server \
  -v /absolute/host/path:/data \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e OMP_NUM_THREADS=10 \
  -p 5002:5002 \
  opendataloader-api-server
```

### 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `PR_LLM_ENDPOINT` | 是 | LLM API 地址（如 `http://127.0.0.1:11434` 或 `https://openrouter.ai/api/v1`） |
| `PR_LLM_API_KEY` | 否 | API Key，留空则使用 Ollama 原生协议 |
| `ODL_OMP_THREADS` | 否 | OpenDataLoader CPU 线程数，默认建议 `10` |
| `ODL_PARSE_TIMEOUT` | 否 | OpenDataLoader 解析超时秒数，默认建议 `3600` |
| `HF_HOME` | 否 | HuggingFace 模型缓存目录，默认 `~/.cache/huggingface` |

## 输入参数

通过 `ProcessParams` 传入：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `file_path` | `str` | 是 | — | 输入 PDF 文件的**宿主机绝对路径** |
| `output_dir` | `str` | 是 | — | 输出 Markdown 文件的**宿主机绝对路径** |
| `odl_volume_host_dir` | `str` | 是 | — | 挂载到容器 `/data` 的宿主机绝对路径 |
| `odl_hybrid_mode` | `str` | 否 | `"full"` | 解析模式：`full` 或 `auto` |
| `odl_parse_timeout` | `int` | 否 | `3600` | OpenDataLoader 解析超时秒数 |
| `odl_container_name` | `str` | 否 | `"opendataloader-api-server"` | Docker 容器名称 |
| `steps` | `list[str]` | 否 | `["summary", "translate", "original"]` | 执行的步骤列表 |
| `src_lang` | `str` | 否 | `"en"` | 源语言代码 |
| `target_lang` | `str` | 否 | `"zh"` | 目标语言代码 |
| `llm_endpoint` | `str` | 否 | env `PR_LLM_ENDPOINT` | LLM API 地址 |
| `llm_api_key` | `str` | 否 | env `PR_LLM_API_KEY` | API Key |
| `chat_model_name` | `str` | 否 | `"qwen/qwen3.5-flash-02-23"` | 文本模型名称 |
| `vision_model_name` | `str` | 否 | `"qwen/qwen3.5-flash-02-23"` | 视觉模型名称 |
| `gen_conf` | `dict` | 否 | `{temperature: 0.7, top_p: 0.3, ...}` | 生成参数 |
| `max_context_token_num` | `int` | 否 | `120000` | 摘要最大输入 token 数 |
| `asset_save_dir` | `str` | 否 | `""` | 解析资源保存目录 |
| `cache_data_dir` | `str` | 否 | `"~/.cache/llm_cache"` | 磁盘缓存目录 |

## 输出格式

返回 `ProcessResult`，JSON 结构：

```json
{
    "output_file": "/path/to/output.md",
    "steps_completed": ["summary", "translate", "original"],
    "elapsed_seconds": 123.45
}
```

## 调用方式

### 1. Python API

```python
import asyncio
from paper_reading import process, ProcessParams

params = ProcessParams(
    file_path="/path/to/paper.pdf",
    output_dir="/path/to/output",
    steps=["summary", "translate", "original"],
    src_lang="en",
    target_lang="zh",
)
result = asyncio.run(process(params))
print(result.output_file)
```

### 2. CLI

```sh
paper-reading \
    --file_path /path/to/host_data/paper.pdf \
    --final_md_file_save_dir /path/to/host_data/output \
    --odl_volume_host_dir /path/to/host_data \
    --odl_hybrid_mode full \
    --odl_parse_timeout 3600 \
    --steps summary,translate,original \
    --src_lang en \
    --target_lang zh \
  --chat_model_name qwen/qwen3.5-flash-02-23 \
    --llm_endpoint http://127.0.0.1:11434
```

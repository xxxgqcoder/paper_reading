# 论文阅读处理流程

> English docs: [README.md](README.md)

基于 Python 的 PDF 学术论文处理流程。使用 **MineRU** 进行本地 PDF 解析（文本、图像、表格），通过任意 **OpenAI 兼容 API**（或本地 Ollama）进行总结与翻译，输出清晰的 Markdown 文档。

## 功能特性

- **PDF 解析**：MineRU pipeline 后端 — 高质量版面分析、文本/图像/表格提取，无需 Docker。
- **LLM 集成**：支持 Ollama（无需 API Key）或通过环境变量配置 OpenAI 兼容接口。
- **可配置**：模型、路径、生成参数等全部通过 CLI 参数或环境变量配置。
- **缓存**：中间结果磁盘缓存，同一文件再次处理更快。
- **步骤可选**：可只跑 `original`、`summary`、`translate` 或其任意组合。
- **Agent Skill**：内置两个可供 AI Agent 调用的 Skill（`digest-to-md`、`extract-pages`）。

## 环境要求

- Python ≥ 3.12
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境

## 快速开始

### 1. 安装依赖

```sh
uv sync
# 或
uv pip install -e .
```

### 2. 下载模型（首次使用）

MineRU 需要本地模型权重，首次使用前请提前下载，避免运行时卡顿：

```sh
# 下载所有所需模型（推荐）
uv run paper-reading download-models

# 国内网络推荐使用 ModelScope 镜像
uv run paper-reading download-models --source modelscope
```

### 3. 配置 LLM

通过**环境变量**配置 LLM 连接信息：

```sh
# 必须设置 — LLM API 地址
export PR_LLM_ENDPOINT="http://127.0.0.1:11434"        # Ollama 本地
# export PR_LLM_ENDPOINT="https://api.openai.com/v1"   # OpenAI 兼容

# 可选 — API Key（Ollama 本地部署时留空）
export PR_LLM_API_KEY="sk-xxx"
```

- **Ollama**：本地启动 Ollama，设置 `PR_LLM_ENDPOINT`，无需设置 `PR_LLM_API_KEY`。
- **OpenAI 兼容接口**：同时设置 `PR_LLM_ENDPOINT` 与 `PR_LLM_API_KEY`。

### 4. 处理 PDF

```sh
uv run paper-reading \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output/ \
    --steps summary,translate,original \
    --src_lang en \
    --target_lang zh
```

## 使用方法

### 处理 PDF（默认命令）

```sh
uv run paper-reading \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output/ \
    --src_lang en \
    --target_lang zh \
    --steps summary,translate,original
```

**参数说明：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `--file_path` | 是 | 输入 PDF 路径（绝对路径） |
| `--final_md_file_save_dir` | 是 | 输出 Markdown 目录（绝对路径） |
| `--src_lang` | 否 | 源语言，默认 `en` |
| `--target_lang` | 否 | 目标语言，默认 `zh` |
| `--steps` | 否 | 逗号分隔的步骤：`summary`、`translate`、`original` |
| `--chat_model_name` | 否 | 文本模型名称 |
| `--vision_model_name` | 否 | 视觉模型名称 |
| `--llm_endpoint` | 否 | LLM API 地址（默认读取 `PR_LLM_ENDPOINT`） |
| `--llm_api_key` | 否 | API Key（默认读取 `PR_LLM_API_KEY`） |
| `--temperature` | 否 | 生成温度 |
| `--top_p` | 否 | top-p 采样 |
| `--num_ctx` | 否 | 模型上下文长度 |
| `--asset_save_dir` | 否 | 解析资源（图像/表格）保存目录 |
| `--cache_data_dir` | 否 | 磁盘缓存目录（默认 `~/.cache/llm_cache`） |

也可以使用 JSON 配置文件代替逐项参数：

```sh
uv run paper-reading --config config.json
```

查看完整配置 Schema：

```sh
uv run paper-reading get-schema
```

### 提取 PDF 指定页

```sh
paper-reading extract-pages \
    --input_pdf ~/path/to/input.pdf \
    --pages 1-93 "100,105-110" \
    --output_dir /path/to/output/
```

**页码范围语法：**
- `1-93` — 第 1 到 93 页
- `100,105-110` — 第 100 页及第 105–110 页
- `1,3,5-7` — 第 1、3、5–7 页

### 下载模型

```sh
# 下载所有模型
paper-reading download-models

# 下载指定模型
paper-reading download-models --model code-formula

# 使用 ModelScope 作为下载源
paper-reading download-models --source modelscope
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PR_LLM_ENDPOINT` | LLM API 地址 | — |
| `PR_LLM_API_KEY` | LLM API Key | — |
| `HF_HOME` | HuggingFace 缓存目录 | `~/.cache/huggingface` |
| `MINERU_MODEL_SOURCE` | 模型下载源：`huggingface` / `modelscope` | `modelscope` |
| `MINERU_DEVICE_MODE` | MineRU 推理设备（空则自动检测） | 自动检测 |

## JSON 输出格式

所有命令均输出 JSON，方便作为 Agent Skill 集成。

**process 输出示例：**
```json
{
  "status": "success",
  "output_file": "output/paper.md",
  "steps_completed": ["summary", "translate", "original"],
  "elapsed_seconds": 123.45
}
```

**download-models 输出示例：**
```json
{
  "status": "completed",
  "downloaded": 1,
  "total": 1,
  "results": [
    {
      "model": "code-formula",
      "status": "success",
      "path": "/Users/.../.cache/huggingface/hub/..."
    }
  ]
}
```

## Agent Skill

`.agents/skills/` 目录下内置两个 AI Agent Skill：

| Skill | 功能 | 文档 |
|-------|------|------|
| **digest-to-md** | 解析 PDF、生成摘要、翻译，输出 Markdown | [SKILL.md](.agents/skills/digest-to-md/SKILL.md) |
| **extract-pages** | 按页码范围从 PDF 提取子页面 | [SKILL.md](.agents/skills/extract-pages/SKILL.md) |

### 全局安装 Skill

```sh
# 安装 CLI 工具
uv tool install paper-reading

# 部署 Skill 文档到 ~/.agents/skills/（Agent 在任意工作区自动发现）
paper-reading install-skills
```

卸载：

```sh
paper-reading install-skills --uninstall
uv tool uninstall paper-reading
```

### Python API

```python
import asyncio
from paper_reading import process, ProcessParams

params = ProcessParams(
    file_path="/path/to/paper.pdf",
    final_md_file_save_dir="/path/to/output",
    steps=["summary", "translate", "original"],
)
result = asyncio.run(process(params))
```

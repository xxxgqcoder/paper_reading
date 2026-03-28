# 论文阅读处理流程

基于 Python 的 PDF 学术论文处理流程：使用 **OpenDataLoader** (基于 Docker) 解析 PDF（文本、图像、表格），通过本地 **Ollama** 或任意 **OpenAI 兼容 API** 进行总结与翻译，输出 Markdown。

## 功能特性

- **PDF 解析**：OpenDataLoader (基于 Docker) 解析版面，支持 Hybrid 混合解析模式（full/auto），提取高质量文本、图像与表格。
- **LLM 集成**：支持 Ollama（无需 API Key）或通过环境变量 `PR_LLM_API_KEY` 使用 OpenAI 兼容接口。
- **可配置**：模型、路径、解析模式、生成参数等全部通过 CLI 参数或环境变量配置。
- **缓存**：中间结果缓存，同一文件再次处理更快。
- **步骤可选**：可只跑 `original`、`summary`、`translate` 或其组合。

## 环境要求

- Python ≥ 3.12
- Docker (用于运行 PDF 解析服务)
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境

## 快速开始

### 1. 安装依赖

```sh
uv sync
# 或
uv pip install -e .
```

### 2. 配置 LLM

LLM 连接信息通过**环境变量**配置（不再写入 `config.yaml`）：

```sh
# 必须设置 — LLM API 地址
export PR_LLM_ENDPOINT="http://127.0.0.1:11434"        # Ollama 本地
# export PR_LLM_ENDPOINT="https://api.openai.com/v1"   # OpenAI 兼容

# 可选 — API Key（留空则使用 Ollama 原生协议）
export PR_LLM_API_KEY="sk-xxx"
```

- **Ollama**：本地启动 Ollama，设置 `PR_LLM_ENDPOINT`，无需设置 `PR_LLM_API_KEY`。
- **OpenAI 兼容**：设置 `PR_LLM_ENDPOINT` 与 `PR_LLM_API_KEY` 环境变量。

### 3. OpenDataLoader 解析环境

PDF 解析依赖 [OpenDataLoader](https://github.com/opendatalab/OpenDataLoader) 的 Docker 服务。

1. 使用仓库内提供的 `Dockerfile` 构建镜像并启动容器：
   ```bash
   # 构建镜像（只需一次）
   docker build -t opendataloader-api-server paper_reading/

   # 启动容器
   docker run -d --name opendataloader-api-server \
     -v /path/to/your/data:/data \
     -p 5002:5002 \
     opendataloader-api-server
   ```
2. 确保 `odl_volume_host_dir` 指向宿主机上挂载到容器 `/data` 的绝对路径（即上述命令中的 `/path/to/your/data`）。

### 4. 处理单个 PDF

快速处理单个 PDF（使用默认配置）：

```sh
uv run python -m paper_reading.cli \
    --file_path /path/to/your/data/paper.pdf \
    --final_md_file_save_dir /path/to/your/data/output \
    --odl_volume_host_dir /path/to/your/data
```

或直接编辑 `run_process_pdf.sh` 并执行为快捷方式。

## 使用方法

### 1. 提取 PDF 指定页

通过 CLI 传入参数：

```sh
paper-reading extract-pages \
    --input_pdf ~/path/to/input.pdf \
    --pages 1-93 "100,105-110"
```

或直接通过 Python API（见下方 [Agent Skill](#agent-skill) 章节）。

**页码范围语法**：
- `1-93` — 第 1 到 93 页
- `100,105-110` — 第 100 页、105 到 110 页
- `1,3,5-7` — 第 1、3、5 到 7 页

### 2. 单个 PDF 解析 / 总结 / 翻译

通过 CLI 处理 PDF（默认执行 `process` 子命令）：

```sh
uv run python -m paper_reading.cli \
    --file_path /path/to/host_data/paper.pdf \
    --final_md_file_save_dir /path/to/host_data/output \
    --odl_volume_host_dir /path/to/host_data \
    --odl_hybrid_mode full \
    --src_lang en \
    --target_lang zh \
    --steps summary,translate,original
```

**参数说明**（详见各 Skill 文档）：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--file_path` | 是 | 输入 PDF 路径 (宿主机绝对路径) |
| `--final_md_file_save_dir` | 是 | 输出 Markdown 目录 (宿主机绝对路径) |
| `--odl_volume_host_dir` | 是 | 挂载到容器 `/data` 的宿主机绝对路径 |
| `--odl_hybrid_mode` | 否 | 解析模式：`full` (默认/高精度/AI模式), `auto` |
| `--src_lang` | 否 | 源语言，默认 `en` |
| `--target_lang` | 否 | 目标语言，默认 `zh` |
| `--steps` | 否 | 逗号分隔的步骤：`summary`、`translate`、`original` |
| `--chat_model_name` | 否 | 文本模型名称，默认 `llama3` |
| `--vision_model_name` | 否 | 视觉模型名称，默认 `llama3` |
| `--llm_endpoint` | 否 | LLM API 地址（默认读取 env `PR_LLM_ENDPOINT`） |
| `--llm_api_key` | 否 | API Key（默认读取 env `PR_LLM_API_KEY`） |
| `--temperature` | 否 | 生成温度 |
| `--top_p` | 否 | top-p 采样 |
| `--num_ctx` | 否 | 模型上下文长度 |
| `--asset_save_dir` | 否 | 解析资源保存目录 |
| `--cache_data_dir` | 否 | 磁盘缓存目录 |

### 3. 批量处理

仓库未附带批量脚本。可自行编写脚本遍历 PDF 目录，对每个文件调用 CLI 命令或 Python API（见下方 [Agent Skill](#agent-skill) 中的 Python API 示例）。

## 配置

所有运行参数通过 **CLI 参数** 或 **环境变量** 传入，无需配置文件。

### LLM 连接

通过环境变量或 CLI 参数配置（见 [快速开始 / 配置 LLM](#2-配置-llm)）：

```sh
export PR_LLM_ENDPOINT="http://127.0.0.1:11434"
export PR_LLM_API_KEY=""   # 留空用 Ollama
```

或通过 CLI：`--llm_endpoint http://127.0.0.1:11434 --llm_api_key sk-xxx`

### 模型与生成参数

```sh
paper-reading \
    --chat_model_name qwen3:30b \
    --vision_model_name qwen2.5vl:7b \
    --temperature 0.7 --top_p 0.4 --num_ctx 60000 \
    --max_context_token_num 60000 \
    ...
```

### 路径

- `--asset_save_dir`：解析后的图像/表格保存目录
- `--cache_data_dir`：磁盘缓存目录（默认 `~/.cache/llm_cache`）

### 页面提取

`extract-pages` 通过 CLI 参数 `--input_pdf` 和 `--pages` 指定输入文件与页码范围。

## Agent Skill

本项目已按 `.agents/skills/` 约定提供两个可供 AI Agent 调用的 Skill，每个 Skill 都有完整文档、参数说明、调用示例。

### 可用 Skill

| Skill | 功能 | 文档 |
|-------|------|------|
| **digest-to-md** | 解析 PDF、生成摘要、翻译文本，输出 Markdown | [SKILL.md](.agents/skills/digest-to-md/SKILL.md) |
| **extract-pages** | 按页码范围从 PDF 提取子页面 | [SKILL.md](.agents/skills/extract-pages/SKILL.md) |

### 安装 Skill

#### 全局安装（推荐，使 Agent 在任意工作区自动发现）

```sh
# 1. 安装 CLI 工具
uv tool install paper-reading
# 或从 Git 仓库安装
# uv tool install git+https://github.com/yourname/paper-reading

# 2. 部署 Skill 文档到 ~/.agents/skills/
paper-reading install-skills
```

卸载：

```sh
paper-reading install-skills --uninstall
uv tool uninstall paper-reading
```

#### 项目级安装

克隆仓库后 Skill 已在 `.agents/skills/` 下，当前项目内直接可用。

### 调用方式

每个 Skill 支持两种调用方式：

1. **Python API**（推荐用于集成）
   ```python
   import asyncio
   from paper_reading import process, ProcessParams
   
   params = ProcessParams(
       file_path="/path/to/paper.pdf",
       output_dir="/path/to/output",
       steps=["summary", "translate", "original"]
   )
   result = asyncio.run(process(params))
   ```

2. **CLI**（命令行工具）
   ```sh
   # 解析/总结/翻译 PDF
   paper-reading \
       --file_path /path/to/paper.pdf \
       --final_md_file_save_dir /path/to/output

   # 提取 PDF 页面
   paper-reading extract-pages \
       --input_pdf /path/to/paper.pdf \
       --pages 1-93 "100,105-110"
   ```

### 文档

详细的参数说明、输出格式、前置条件等，请查看各 Skill 的 `SKILL.md` 文件。

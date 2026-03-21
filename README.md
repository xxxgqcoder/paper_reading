# 论文阅读处理流程

基于 Python 的 PDF 学术论文处理流程：使用 **MinerU** 解析 PDF（文本、图像、表格），通过本地 **Ollama** 或任意 **OpenAI 兼容 API** 进行总结与翻译，输出 Markdown。

## 功能特性

- **PDF 解析**：MinerU 解析版面，提取文本、图像与表格。
- **LLM 集成**：支持 Ollama（无需 API Key）或通过环境变量 `LLM_API_KEY` 使用 OpenAI 兼容接口。
- **可配置**：模型、路径、生成参数、提示词模板等均在 `config.yaml` 中配置。
- **缓存**：中间结果缓存，同一文件再次处理更快。
- **步骤可选**：可只跑 `original`、`summary`、`translate` 或其组合。

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

### 2. 配置 LLM

LLM 连接信息通过**环境变量**配置（不再写入 `config.yaml`）：

```sh
# 必须设置 — LLM API 地址
export LLM_ENDPOINT="http://127.0.0.1:11434"        # Ollama 本地
# export LLM_ENDPOINT="https://api.openai.com/v1"   # OpenAI 兼容

# 可选 — API Key（留空则使用 Ollama 原生协议）
export LLM_API_KEY="sk-xxx"
```

- **Ollama**：本地启动 Ollama，设置 `LLM_ENDPOINT`，无需设置 `LLM_API_KEY`。
- **OpenAI 兼容**：设置 `LLM_ENDPOINT` 与 `LLM_API_KEY` 环境变量。

### 3. MinerU 解析环境

PDF 解析依赖 [MinerU](https://github.com/opendatalab/MinerU) 及其模型。本仓库不提供模型下载脚本，请按 MinerU 官方文档自行准备运行环境与所需模型。

### 4. 处理单个 PDF

快速处理单个 PDF（使用默认配置）：

```sh
uv run python -m paper_reading.cli \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output
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

或在 `config.yaml` 中配置后执行（适用于项目内开发）：

```yaml
extract_pages:
    input_pdf: ~/path/to/input.pdf
    pages:
        - 1-93
        - 100,105-110
```

```sh
paper-reading extract-pages
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
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output \
    --src_lang en \
    --target_lang zh \
    --steps summary,translate,original
```

**参数说明**（详见各 Skill 文档）：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--file_path` | 是 | 输入 PDF 路径 |
| `--final_md_file_save_dir` | 是 | 输出 Markdown 目录 |
| `--src_lang` | 否 | 源语言，默认 `en` |
| `--target_lang` | 否 | 目标语言，默认 `zh` |
| `--steps` | 否 | 逗号分隔的步骤：`summary`、`translate`、`original` |

未指定的参数使用 `config.yaml` 中的默认值。

### 3. 批量处理

仓库未附带批量脚本。可自行编写脚本遍历 PDF 目录，对每个文件调用 CLI 命令或 Python API（见下方 [Agent Skill](#agent-skill) 中的 Python API 示例）。

## 配置

主配置为项目根目录下的 `config.yaml`（可复制仓库内示例并按本机路径修改）。主要字段如下。

### LLM 与模型

LLM 连接信息通过环境变量配置（见 [快速开始 / 配置 LLM](#2-配置-llm)），模型名称在 `config.yaml` 中配置：

```yaml
chat_model_name: qwen3:30b-a3b-thinking-2507-q4_K_M
vision_model_name: qwen2.5vl:7b
max_context_token_num: 60000
```

### 路径与步骤

```yaml
cache_data_dir: ~/.cache/llm_cache
asset_save_dir: ~/obsidian/attachments
temp_content_dir: ./tmp/parsed_asset

steps: summary,translate,original
src_lang: en
target_lang: zh
```

### 生成参数

```yaml
gen_conf:
    temperature: 0.7
    top_p: 0.4
    repeat_penalty: 1.2
    num_ctx: 60000
```

### 提示词模板（可选）

占位符：`{src_lang}`、`{target_lang}`、`{content}`。可在 `config.yaml` 中覆盖默认的 `prompt_translate` 与 `prompt_summary`。

### 页面提取

`extract-pages` 的输入 PDF 与页码在 `config.yaml` 的 `extract_pages` 中配置，无需单独文件。

## Agent Skill

本项目已按 `.agents/skills/` 约定提供两个可供 AI Agent 调用的 Skill，每个 Skill 都有完整文档、参数说明、调用示例。

### 可用 Skill

| Skill | 功能 | 文档 |
|-------|------|------|
| **process-pdf** | 解析 PDF、生成摘要、翻译文本，输出 Markdown | [SKILL.md](.agents/skills/process-pdf/SKILL.md) |
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

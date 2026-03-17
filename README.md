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

```sh
python process_pdf.py \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output
```

可选：复制并修改 `run_process_pdf.sh` 中的路径后执行 `./run_process_pdf.sh` 作为快捷方式。

## 使用方法

### 1. 提取 PDF 指定页

使用子命令 `extract-pages`，从 PDF 中按配置的页面范围导出为新 PDF。

在 `config.yaml` 中配置：

```yaml
extract_pages:
    input_pdf: ~/path/to/input.pdf
    pages:
        - 1-93
        - 100,105-110
```

然后执行：

```sh
python process_pdf.py extract-pages
```

- 支持页码范围：如 `1,3,5-7` 表示第 1、3、5～7 页。

### 2. 单个 PDF 解析 / 总结 / 翻译

默认子命令为 `process`（可省略）。必填参数：`--file_path`、`--final_md_file_save_dir`。

```sh
python process_pdf.py \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output \
    --src_lang en \
    --target_lang zh \
    --steps original,summary,translate
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--file_path` | 输入 PDF 路径（process 时必填） |
| `--final_md_file_save_dir` | 输出 Markdown 所在目录（process 时必填） |
| `--temp_content_dir` | 解析过程临时文件目录，默认使用 config 中的 `temp_content_dir` |
| `--src_lang` | 源语言，如 `en`（默认） |
| `--target_lang` | 目标语言，如 `zh`（默认） |
| `--steps` | 逗号分隔的步骤：`summary`（摘要）、`translate`（翻译）、`original`（原文） |

未指定的参数以 `config.yaml` 为准。

### 3. 批量处理

仓库未附带批量脚本。可自行写脚本遍历 PDF 目录，对每个文件调用上述 `process_pdf.py` 命令（或循环调用 `process()`），并根据需要设置 `pdf_folder`、`output_folder`、`steps` 等。

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

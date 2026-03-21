# Skill: process-pdf

解析学术论文 PDF，通过 LLM 进行总结与翻译，输出 Markdown 文件。

## 功能

- **PDF 解析**：使用 MinerU 解析 PDF 版面，提取文本、图像与表格
- **内容总结**：通过 LLM 生成文档摘要
- **内容翻译**：通过 LLM 将文档翻译为目标语言
- **原文保留**：将解析后的原始内容写入 Markdown

可通过 `steps` 参数选择执行哪些步骤（`summary`、`translate`、`original` 的任意组合）。

## 前置条件

### 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `LLM_ENDPOINT` | 是 | LLM API 地址（如 `http://127.0.0.1:11434` 或 `https://openrouter.ai/api/v1`） |
| `LLM_API_KEY` | 否 | API Key，留空则使用 Ollama 原生协议 |

### 依赖

- Python ≥ 3.12
- 项目依赖已安装（`uv sync` 或 `uv pip install -e .`）
- MinerU 模型已下载（参见 [MinerU 文档](https://github.com/opendatalab/MinerU)）

## 输入参数

通过 `ProcessParams` 传入：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `file_path` | `str` | 是 | — | 输入 PDF 文件的绝对路径 |
| `output_dir` | `str` | 是 | — | 输出 Markdown 文件的目录 |
| `steps` | `list[str]` | 否 | `["summary", "translate", "original"]` | 执行的步骤列表 |
| `src_lang` | `str` | 否 | `"en"` | 源语言代码 |
| `target_lang` | `str` | 否 | `"zh"` | 目标语言代码 |
| `chat_model_name` | `str \| None` | 否 | `None`（回退到 config.yaml） | 文本模型名称 |
| `vision_model_name` | `str \| None` | 否 | `None`（回退到 config.yaml） | 视觉模型名称 |
| `gen_conf` | `dict \| None` | 否 | `None`（回退到 config.yaml） | 生成参数（temperature、top_p 等） |

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
uv run python -m paper_reading.cli \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output \
    --steps summary,translate,original \
    --src_lang en \
    --target_lang zh
```

### 3. 辅助脚本

```sh
bash .agents/skills/process-pdf/scripts/run.sh \
    /path/to/paper.pdf \
    /path/to/output \
    --steps summary,translate,original
```

## 相关文件

- `paper_reading/pipeline.py` — 核心处理入口 `process()`
- `paper_reading/config.py` — `ProcessParams`、`ProcessResult` 定义
- `paper_reading/steps.py` — 各步骤的具体实现
- `config.yaml` — 默认配置（模型、提示词模板等）

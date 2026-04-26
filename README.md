# Paper Reading Pipeline

> 中文文档: [README_zh.md](README_zh.md)

A Python-based PDF processing pipeline for academic papers. It uses **MineRU** for local PDF parsing (text, images, tables) and any **OpenAI-compatible API** (or local Ollama) for summarization and translation, producing clean Markdown output.

## Features

- **PDF Parsing**: MineRU pipeline backend — high-quality layout analysis, text/image/table extraction, no Docker required.
- **LLM Integration**: Works with Ollama (no API key needed) or any OpenAI-compatible endpoint via environment variables.
- **Configurable**: Models, paths, generation parameters — all controlled via CLI flags or environment variables.
- **Caching**: Intermediate results are cached on disk; reprocessing the same file is fast.
- **Selective Steps**: Run only `original`, `summary`, `translate`, or any combination.
- **Agent Skills**: Two ready-to-use AI agent skills (`digest-to-md`, `extract-pages`).

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended for dependency and environment management)

## Quick Start

### 1. Install Dependencies

```sh
uv sync
# or
uv pip install -e .
```

### 2. Download Models (first-time only)

MineRU requires local model weights. Download them before first use to avoid runtime hangs:

```sh
# Download all required models (recommended)
uv run paper-reading download-models

# Use ModelScope mirror (recommended for users in China)
uv run paper-reading download-models --source modelscope
```

### 3. Configure LLM

Set LLM connection via **environment variables**:

```sh
# Required — LLM API endpoint
export PR_LLM_ENDPOINT="http://127.0.0.1:11434"       # Local Ollama
# export PR_LLM_ENDPOINT="https://api.openai.com/v1"  # OpenAI-compatible

# Optional — API key (leave empty for Ollama)
export PR_LLM_API_KEY="sk-xxx"
```

- **Ollama**: Start Ollama locally, set `PR_LLM_ENDPOINT`. No `PR_LLM_API_KEY` needed.
- **OpenAI-compatible**: Set both `PR_LLM_ENDPOINT` and `PR_LLM_API_KEY`.

### 4. Process a PDF

```sh
uv run paper-reading \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output/ \
    --steps summary,translate,original \
    --src_lang en \
    --target_lang zh
```

## Usage

### Process PDF (default command)

```sh
uv run paper-reading \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output/ \
    --src_lang en \
    --target_lang zh \
    --steps summary,translate,original
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--file_path` | Yes | Input PDF path (absolute) |
| `--final_md_file_save_dir` | Yes | Output Markdown directory (absolute) |
| `--src_lang` | No | Source language, default `en` |
| `--target_lang` | No | Target language, default `zh` |
| `--steps` | No | Comma-separated steps: `summary`, `translate`, `original` |
| `--chat_model_name` | No | Text model name |
| `--vision_model_name` | No | Vision model name |
| `--llm_endpoint` | No | LLM API endpoint (falls back to `PR_LLM_ENDPOINT`) |
| `--llm_api_key` | No | API key (falls back to `PR_LLM_API_KEY`) |
| `--temperature` | No | Sampling temperature |
| `--top_p` | No | Top-p sampling |
| `--num_ctx` | No | Model context length |
| `--asset_save_dir` | No | Directory for extracted images/tables |
| `--cache_data_dir` | No | Disk cache directory (default `~/.cache/llm_cache`) |

Use a JSON config file instead of individual flags:

```sh
uv run paper-reading --config config.json
```

Print the full config schema:

```sh
uv run paper-reading get-schema
```

### Extract PDF Pages

```sh
paper-reading extract-pages \
    --input_pdf ~/path/to/input.pdf \
    --pages 1-93 "100,105-110" \
    --output_dir /path/to/output/
```

**Page range syntax:**
- `1-93` — pages 1 through 93
- `100,105-110` — page 100 and pages 105–110
- `1,3,5-7` — pages 1, 3, and 5–7

### Download Models

```sh
# Download all models
paper-reading download-models

# Download a specific model
paper-reading download-models --model code-formula

# Use ModelScope as the download source
paper-reading download-models --source modelscope
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PR_LLM_ENDPOINT` | LLM API endpoint | — |
| `PR_LLM_API_KEY` | LLM API key | — |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |
| `MINERU_MODEL_SOURCE` | Model download source: `huggingface` / `modelscope` | `modelscope` |
| `MINERU_DEVICE_MODE` | MineRU inference device (empty = auto-detect) | auto-detect |

## JSON Output Format

All commands output JSON for easy integration as agent skills.

**process output:**
```json
{
  "status": "success",
  "output_file": "output/paper.md",
  "steps_completed": ["summary", "translate", "original"],
  "elapsed_seconds": 123.45
}
```

**download-models output:**
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

## Agent Skills

Two AI agent skills are bundled under `.agents/skills/`:

| Skill | Description | Docs |
|-------|-------------|------|
| **digest-to-md** | Parse PDF, summarize and translate to Markdown | [SKILL.md](.agents/skills/digest-to-md/SKILL.md) |
| **extract-pages** | Extract page ranges from a PDF | [SKILL.md](.agents/skills/extract-pages/SKILL.md) |

### Install Skills Globally

```sh
# Install CLI tool
uv tool install paper-reading

# Deploy skill docs to ~/.agents/skills/ (auto-discovered by agents in any workspace)
paper-reading install-skills
```

Uninstall:

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

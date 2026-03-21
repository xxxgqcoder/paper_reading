# Skill: extract-pages

从 PDF 文件中按页码范围提取子页面，输出为新的 PDF 文件。

## 功能

- 支持灵活的页码范围语法：`1-93`、`100,105-110`、`1,3,5-7`
- 每个页码范围生成一个独立的输出 PDF
- 可指定自定义输出目录

## 前置条件

### 依赖

- Python ≥ 3.12
- 项目依赖已安装（`uv sync` 或 `uv pip install -e .`）

> 此 Skill 不依赖 LLM，无需配置 `LLM_ENDPOINT` 或 `LLM_API_KEY`。

## 输入参数

通过 `ExtractPagesParams` 传入：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `input_pdf` | `str` | 是 | — | 输入 PDF 文件路径 |
| `pages` | `list[str]` | 是 | — | 页码范围列表，如 `["1-93", "100,105-110"]` |
| `output_dir` | `str \| None` | 否 | `None`（与输入文件同目录） | 输出 PDF 的目录 |

### 页码范围语法

- `1-93` — 第 1 页到第 93 页
- `100,105-110` — 第 100 页、第 105 到 110 页
- `1,3,5-7` — 第 1、3、5、6、7 页

## 输出格式

返回输出文件路径列表，CLI 模式下为 JSON：

```json
{
    "status": "success",
    "output_files": ["/path/to/paper-1-93.pdf"],
    "elapsed_seconds": 1.23
}
```

## 调用方式

### 1. Python API

```python
from paper_reading import run_extract_pages, ExtractPagesParams

params = ExtractPagesParams(
    input_pdf="/path/to/paper.pdf",
    pages=["1-93", "100,105-110"],
    output_dir="/path/to/output",
)
output_files = run_extract_pages(params)
print(output_files)
```

### 2. CLI

在 `config.yaml` 中配置后执行：

```sh
uv run python -m paper_reading.cli extract-pages
```

### 3. 辅助脚本

```sh
bash .agents/skills/extract-pages/scripts/run.sh \
    /path/to/paper.pdf \
    "1-93" "100,105-110"
```

## 相关文件

- `paper_reading/extract_pages.py` — 核心逻辑 `run_extract_pages()`
- `paper_reading/config.py` — `ExtractPagesParams` 定义
- `config.yaml` — CLI 模式下的默认配置

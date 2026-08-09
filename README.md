# PDF 阅读工具

> English documentation: [README_en.md](README_en.md)

`paper-reading` 提供统一的 PDF 工作流：检查目录与页码偏移、按章节或页码提取页面，以及使用 MineRU 将 PDF 解析为 Markdown。AI 总结和翻译是可选能力，默认不会启用。

## 功能

- `parse`：解析文本、图片、表格和公式并输出 Markdown；默认仅执行 `original`，不调用 LLM。
- `inspect`：读取 PDF 前置页，识别目录候选、章节条目、PDF 总页数及印刷页码偏移证据。
- `extract`：按印刷页码范围提取 PDF，支持 `pdf_page = printed_page + offset`。
- `pdf_reading()`：统一 Python API；旧 API 暂时保留作为兼容层。
- `pdf-reading-skill`：合并后的统一 Agent Skill。

## 安装

项目使用 Python 3.12+、uv 与 `pyproject.toml`。在仓库内开发时：

```sh
uv sync
uv run paper-reading --help
```

当用户要求 Agent 安装全局 CLI 和 Skill 时，Agent 必须遵循仓库内的 [pdf-reading-skill 安装流程](.agents/skills/pdf-reading-skill/SKILL.md)。核心命令是从当前 checkout 安装，而不是从包索引取得不确定版本：

```sh
uv tool install --force .
paper_reading_bin_dir="$(uv tool dir --bin)"
export PATH="${paper_reading_bin_dir}:${PATH}"
command -v paper-reading
paper-reading --help
paper-reading install-skills
test -f ~/.agents/skills/pdf-reading-skill/SKILL.md
```

不要用 `uv run paper-reading install-skills` 代替全局 CLI 安装；它不能保证离开仓库后 `paper-reading` 仍在 PATH 中。

模型下载不属于 CLI/Skill 安装。只有准备首次解析 PDF 时才执行：

```sh
paper-reading download-models --source modelscope
```

## 统一 CLI

### 解析 PDF 为 Markdown

```sh
uv run paper-reading parse \
  --file_path /absolute/path/paper.pdf \
  --output_dir /absolute/path/output
```

默认步骤是 `original`，因此不需要 LLM API Key，也不会生成 AI 摘要或翻译。图片默认保存到 PDF 同级的 `{pdf_stem}_images/`。

若用户明确需要总结或翻译，再显式配置 LLM 并指定步骤：

```sh
export PR_LLM_ENDPOINT="https://example.com/v1"
export PR_LLM_API_KEY="<your-api-key>"

uv run paper-reading parse \
  --file_path /absolute/path/paper.pdf \
  --output_dir /absolute/path/output \
  --steps summary,translate,original
```

不要把 API Key 写入脚本、命令历史、日志或文档。使用本地 Ollama 时可以不设置 Key，但仍需配置 endpoint。

### 检查目录与页码偏移

```sh
uv run paper-reading inspect \
  --file_path /absolute/path/book.pdf \
  --inspect_pages 20
```

JSON 结果包含：

- `total_pdf_pages`：PDF 物理总页数；
- `toc_candidate_pages`、`toc_entries`：目录候选页与初步章节映射；
- `page_number_offset.value/confidence/evidence`：页码偏移及证据；
- `pages[].text`：供 Agent 阅读与核验的前置页文本。

页码偏移定义为：

```text
PDF 物理页码 = 书本印刷页码 + page_number_offset
```

按章节提取时，Agent 应阅读目录和前置页，用至少两个页面交叉验证 offset；扫描版或低置信度结果应渲染页面后目视复核。

### 提取 PDF 页面

```sh
uv run paper-reading extract \
  --file_path /absolute/path/book.pdf \
  --pages 1-20 "25,30-35" \
  --page_number_offset 12 \
  --output_dir /absolute/path/output
```

每个 `--pages` 参数生成一个独立 PDF。页码范围支持 `1-20`、`25,30-35`、`1,3,5-7`。

## Python API

```python
import asyncio

from paper_reading import PdfReadingParams, pdf_reading

result = asyncio.run(
    pdf_reading(
        PdfReadingParams(
            action="parse",
            file_path="/absolute/path/paper.pdf",
            output_dir="/absolute/path/output",
        )
    )
)
print(result.model_dump())
```

`action` 可取 `parse`、`inspect`、`extract`。完整 schema：

```sh
uv run paper-reading get-schema
```

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `PR_LLM_ENDPOINT` | 仅 AI 总结/翻译需要的 LLM API 地址 | 无 |
| `PR_LLM_API_KEY` | OpenAI-compatible API Key | 无 |
| `HF_HOME` | HuggingFace 缓存目录 | `~/.cache/huggingface` |
| `MINERU_MODEL_SOURCE` | `huggingface` / `modelscope` | `modelscope` |
| `MINERU_DEVICE_MODE` | MineRU 推理设备 | 自动检测 |

## Agent Skill

仓库只维护一个 Skill：[pdf-reading-skill/SKILL.md](.agents/skills/pdf-reading-skill/SKILL.md)。它覆盖 Markdown 解析、目录阅读、页码偏移核验、章节提取和可选 LLM 流程。

```sh
paper-reading install-skills
paper-reading install-skills --uninstall
```

CLI 和 Skill 的完整安装、PATH 处理、版本一致性检查及验收标准见 [pdf-reading-skill/SKILL.md](.agents/skills/pdf-reading-skill/SKILL.md)。安装新 Skill 时会清理旧的 `digest-to-md` 与 `extract-pages` 全局目录。

## 兼容性

旧的 `process` / `extract-pages` CLI 名称，以及 `process()` / `run_extract_pages()` Python API 暂时保留。新代码应使用 `parse` / `inspect` / `extract` 与 `pdf_reading()`。

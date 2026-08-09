---
name: pdf-reading-skill
description: 'Unified PDF reading workflow: inspect contents and page-number offsets, extract chapters or page ranges, and parse PDFs to Markdown with optional LLM summary or translation.'
---

# PDF Reading Skill

使用 `paper-reading` 的统一入口检查 PDF、按章节或页码提取页面，并将 PDF 解析为 Markdown。

## 默认行为

- 默认只执行 `original`：用 MineRU 将 PDF 解析为 Markdown，不调用 LLM，不生成 AI 摘要或翻译。
- 只有用户明确要求总结或翻译时，才添加 `summary` 或 `translate`。
- 不要把 API Key 直接写进命令、日志、Markdown 或回复；优先通过 `PR_LLM_API_KEY` 环境变量传入。

## 前置条件

```sh
uv tool install paper-reading
paper-reading download-models --source modelscope
paper-reading install-skills
```

## 统一入口

```sh
# 解析为 Markdown（默认无 AI）
paper-reading parse \
  --file_path /absolute/path/input.pdf \
  --output_dir /absolute/path/output

# 检查目录、前置页和页码偏移证据
paper-reading inspect \
  --file_path /absolute/path/input.pdf \
  --inspect_pages 20

# 根据印刷页码提取；offset 满足 pdf_page = printed_page + offset
paper-reading extract \
  --file_path /absolute/path/input.pdf \
  --pages 1-20 \
  --page_number_offset 12 \
  --output_dir /absolute/path/output
```

所有命令返回 JSON。Python 统一入口是 `pdf_reading(PdfReadingParams(...))`。

## 按章节提取的强制流程

当用户按 chapter、章节名或书本印刷页码提取时，不能直接把目录页码当作 PDF 物理页码：

1. 运行 `inspect` 读取 PDF 前若干页；若目录未完整覆盖，逐步增大 `--inspect_pages`。
2. Agent 必须实际阅读返回的 `pages[].text` 中的目录页，识别章节标题、起始印刷页码和下一章节起始页码。章节末页通常是下一章起始页减一。
3. 阅读前置页及正文起始页的页眉/页脚页码证据，核验 `page_number_offset`。关系固定为：`PDF 物理页码 = 印刷页码 + offset`。
4. 至少用两个不同页面的证据交叉验证 offset；`confidence` 较低、文本为空或扫描版 OCR 不可靠时，必须渲染并目视检查对应页面，不能猜测。
5. 向用户说明识别到的章节映射和 offset，再用 `extract --page_number_offset ...` 提取。
6. 打开或检查输出 PDF 的首页、末页、页数，确认没有少页、跨章或错位。

`inspect` 会返回：

- `total_pdf_pages`：PDF 真实物理页数；
- `toc_candidate_pages` 与 `toc_entries`：目录候选页和初步章节条目；
- `page_number_offset.value/confidence/evidence`：偏移推断及证据；
- `pages[].text`：供 Agent 阅读和复核的前置页原文。

## PDF 解析为 Markdown

默认命令：

```sh
paper-reading parse \
  --file_path /absolute/path/input.pdf \
  --output_dir /absolute/path/output
```

等价于 `--steps original`。图片默认保存到 PDF 同级的 `{pdf_stem}_images/`，Markdown 使用 Obsidian wikilink 引用。

若输出 Markdown 已存在且非空，工具会停止，避免覆盖；需要重跑时先由用户确认如何处理已有结果。

## 可选 AI 总结或翻译

仅当用户明确要求 AI 总结/翻译时：

1. 检查 `PR_LLM_ENDPOINT` 与 `PR_LLM_API_KEY` 是否已配置，但不要输出其值。
2. 若缺失，提示用户提供 OpenAI-compatible LLM endpoint 和 API Key，并建议在本地环境中设置：

```sh
export PR_LLM_ENDPOINT="https://example.com/v1"
export PR_LLM_API_KEY="<user-provided-key>"
```

3. 用户完成配置后，根据需求显式执行：

```sh
paper-reading parse \
  --file_path /absolute/path/input.pdf \
  --output_dir /absolute/path/output \
  --steps summary,original
```

翻译使用 `--steps translate,original`；同时总结和翻译才使用 `--steps summary,translate,original`。

本地 Ollama 可不提供 API Key，但仍需配置 `PR_LLM_ENDPOINT`。

## Python API

```python
import asyncio

from paper_reading import PdfReadingParams, pdf_reading

result = asyncio.run(
    pdf_reading(
        PdfReadingParams(
            action="parse",
            file_path="/absolute/path/input.pdf",
            output_dir="/absolute/path/output",
        )
    )
)
print(result.model_dump())
```

## 兼容入口

旧的 `process` / `extract-pages` CLI 名称及 `process()` / `run_extract_pages()` Python API 暂时保留，但新工作流统一使用 `parse` / `inspect` / `extract` 和 `pdf_reading()`。

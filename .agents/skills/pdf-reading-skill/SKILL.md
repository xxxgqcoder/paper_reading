---
name: pdf-reading-skill
description: 'Install the paper-reading CLI and Skill, inspect PDF contents and page-number offsets, extract chapters or page ranges, and parse PDFs to source-preserving Markdown.'
---

# PDF Reading Skill

使用 `paper-reading` 的统一入口检查 PDF、按章节或页码提取页面，并将 PDF 解析为 Markdown。

## 默认行为

- 默认只执行 `original`：用 MineRU 将 PDF 解析为 Markdown，并保留原始内容结构。
- Skill 只负责 PDF 检查、页面提取和源内容解析；解析后的 Markdown 由上层阅读流程按需处理。

## Agent 安装 CLI 与 Skill

仅当用户明确要求安装、更新或修复 CLI/Skill 时执行本节。在当前仓库中，必须安装当前 checkout，不能用 `uv tool install paper-reading` 从包索引取得不确定版本。

```sh
git rev-parse --show-toplevel
cd "$(git rev-parse --show-toplevel)"
test -f pyproject.toml
rg '^name = "paper-reading"$' pyproject.toml
command -v uv
uv tool install --force .
```

如果仓库身份检查失败，不要对其他项目执行安装；先让用户提供 `paper-reading` checkout 或明确的发布来源。如果 `uv` 不存在，先使用当前系统适用的官方方式安装 uv。

安装后验证 CLI；如果 PATH 尚未生效，更新后续 shell 配置并为当前 Agent 进程补齐 uv 工具目录：

```sh
uv tool list
uv tool update-shell
paper_reading_bin_dir="$(uv tool dir --bin)"
export PATH="${paper_reading_bin_dir}:${PATH}"
command -v paper-reading
paper-reading --help
```

Agent 必须确认 `uv tool list` 中的版本与 `pyproject.toml` 一致，并确认帮助中存在 `parse`、`inspect`、`extract`、`install-skills`，然后安装和核验 Skill：

```sh
paper-reading install-skills
test -f ~/.agents/skills/pdf-reading-skill/SKILL.md
cmp -s \
  .agents/skills/pdf-reading-skill/SKILL.md \
  ~/.agents/skills/pdf-reading-skill/SKILL.md
```

不要用 `uv run paper-reading install-skills` 代替全局 CLI 安装。Skill 安装成功后，当前 Agent 会话可能要到新任务或重启后才刷新发现结果。

安装完成后，Agent 必须向用户报告 `paper-reading` 的可执行文件路径、`uv tool list` 中的版本、Skill 安装目录，并明确模型是否尚未下载；任何失败都应保留原始错误并指出未完成的步骤。

模型下载不属于 CLI/Skill 安装。只有准备首次解析 PDF 时才执行：

```sh
paper-reading download-models --source modelscope
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

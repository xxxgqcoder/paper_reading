---
name: pdf-reading-skill
description: 'Install the paper-reading CLI and Skill, inspect PDF contents and page-number offsets, extract chapters or page ranges, parse PDFs to source-preserving Markdown, and correct leading heading levels in the parsed Markdown.'
---

# PDF Reading Skill

使用 `paper-reading` 的统一入口检查 PDF、按章节或页码提取页面，并将 PDF 解析为 Markdown。

## 默认行为

- 默认只执行 `original`：用 MineRU 将 PDF 解析为 Markdown，并保留原始内容结构。
- MineRU 会把解析出的所有标题都写成一级标题（`#`），原层级只保留在标题文字里；解析完成后 Agent 必须按标题编号层级校正前导 `#` 数量，见「解析后校正标题层级」。
- 除标题层级校正外，Skill 只负责 PDF 检查、页面提取和源内容解析；其余后处理由上层阅读流程按需处理。

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

注意：该命令是**预取**，不是更新检查。mineru 在写入 `models-dir` 配置后，对后续 model path 会直接返回本地缓存路径，不再比对上游，因此 `download-models` 可能跳过部分模型。需要确认上游是否有新模型时，直接用 modelscope 的 `snapshot_download` 按 `allow_patterns` 定向比对模型仓库 `OpenDataLab/PDF-Extract-Kit-1.0`（mineru 3.4+ 的 pipeline 模型集为 `models/Layout/PP-DocLayoutV2`、`models/MFR/unimernet_hf_small_2503`、`models/MFR/pp_formulanet_plus_m`、`models/OCR/paddleocr_torch`、`models/TabRec/SlanetPlus`、`models/TabRec/UnetStructure`、`models/TabCls/paddle_table_cls`）。

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

### 解析后校正标题层级（Agent 必做）

`parse` 完成后不能直接把输出交给上层阅读流程：MineRU 输出的标题全部是一级标题，层级信息只留在标题文字里。Agent 必须检查输出 Markdown 的标题行，按标题内容的编号层级补齐前导 `#`。

映射规则：**前导 `#` 数量 = 标题编号的段数 + 1**（以文档标题为唯一的 `#`）。

| 标题示例 | 前导 `#` |
|---|---|
| 文档标题（书名 / 论文标题，无编号） | `#` |
| `3. 方法` | `##` |
| `3.1 概述` | `###` |
| `3.1.1 数据` | `####` |
| `3.1.1.1 实现细节` | `#####` |

执行要求：

1. 通读输出 Markdown 的全部标题行，识别编号形态：`3.`、`3.1`、`3.1.1`、`3.1.1.1`，以及 `Chapter 3`、`A.1`、`1)` 等变体。
2. 只修改标题行的前导 `#` 数量；不改标题文字、编号和标点，不新增、删除、合并或重排标题行，也不改动正文、公式和图片位置。
3. 无编号标题（Abstract、References、Appendix、致谢、图表标题等）按其在文中的位置判断层级；无法判断时保持一级标题并在报告中说明。
4. 层级必须连续、不跳级，最大不超过六级。若解析结果没有独立的文档标题行、编号不从 1 开始（例如只解析了某一章），或编号段数 + 1 超过六级，则按同一相对关系整体调整，并说明调整理由。
5. 校验：用 `rg -n '^#{1,6} ' <output>.md` 逐条核对，确认每个标题的 `#` 数量等于其编号段数 + 1（或符合第 4 条的调整规则）、相邻层级只差一级；不符的标题必须先修正再交付。

修正后向用户报告：文档标题、各级标题的数量分布、以及是否存在按第 3、4 条特殊处理的标题。

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

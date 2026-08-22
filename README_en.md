# PDF Reading

> 中文文档（默认）: [README.md](README.md)

`paper-reading` provides one PDF workflow for inspecting contents and page-number offsets, extracting chapters or page ranges, and parsing PDFs to Markdown with MineRU.

## Features

- `parse`: extract text, images, tables, and formulas to source-preserving Markdown.
- `inspect`: read front matter and report TOC candidates, chapter entries, total PDF pages, and printed/PDF page-offset evidence.
- `extract`: extract printed page ranges with `pdf_page = printed_page + offset`.
- `pdf_reading()`: unified Python API, with legacy APIs retained temporarily.
- `pdf-reading-skill`: one merged Agent Skill.

## Install

The project uses Python 3.12+, uv, and `pyproject.toml`. For development inside the repository:

```sh
uv sync
uv run paper-reading --help
```

When the user asks an Agent to install the global CLI and Skill, the Agent must follow the bundled [pdf-reading-skill instructions](.agents/skills/pdf-reading-skill/SKILL.md). Install the current checkout rather than an unknown package-index version:

```sh
uv tool install --force .
paper_reading_bin_dir="$(uv tool dir --bin)"
export PATH="${paper_reading_bin_dir}:${PATH}"
command -v paper-reading
paper-reading --help
paper-reading install-skills
test -f ~/.agents/skills/pdf-reading-skill/SKILL.md
```

Do not substitute `uv run paper-reading install-skills` for global CLI installation; that does not ensure `paper-reading` remains available on PATH outside the repository.

Model download is separate from CLI/Skill installation. Run it only before the first PDF parse:

```sh
paper-reading download-models --source modelscope
```

## Unified CLI

### Parse PDF to Markdown

```sh
uv run paper-reading parse \
  --file_path /absolute/path/paper.pdf \
  --output_dir /absolute/path/output
```

The default is `--steps original`, which writes the source-preserving parsed content. Assets are saved next to the PDF under `{pdf_stem}_images/` by default.

### Inspect contents and page offset

```sh
uv run paper-reading inspect \
  --file_path /absolute/path/book.pdf \
  --inspect_pages 20
```

The JSON result includes total physical pages, TOC candidates and entries, page-offset evidence, and raw front-matter text for Agent verification. The relationship is:

```text
PDF physical page = printed page + page_number_offset
```

For chapter extraction, the Agent should read the TOC and front matter, validate the offset with at least two pages, and visually verify scanned or low-confidence pages.

### Extract pages

```sh
uv run paper-reading extract \
  --file_path /absolute/path/book.pdf \
  --pages 1-20 "25,30-35" \
  --page_number_offset 12 \
  --output_dir /absolute/path/output
```

Each range argument creates one PDF. Supported forms include `1-20`, `25,30-35`, and `1,3,5-7`.

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

`action` accepts `parse`, `inspect`, or `extract`. Print the schema with `uv run paper-reading get-schema`.

## Agent Skill

The repository maintains one Skill: [pdf-reading-skill/SKILL.md](.agents/skills/pdf-reading-skill/SKILL.md). Installing it also removes legacy global `digest-to-md` and `extract-pages` directories.

```sh
paper-reading install-skills
paper-reading install-skills --uninstall
```

See [pdf-reading-skill/SKILL.md](.agents/skills/pdf-reading-skill/SKILL.md) for PATH handling, version checks, and completion criteria.

Legacy `process` / `extract-pages` CLI names and `process()` / `run_extract_pages()` Python APIs remain temporarily available. New integrations should use the unified interface.

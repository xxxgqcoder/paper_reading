from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

from pypdf import PdfReader, PdfWriter

from .config import ExtractPagesParams
from .log import Logger


def parse_page_ranges(page_ranges_str: str, page_number_offset: int = 0) -> list[int]:
    """
    Parse printed page ranges into 0-based PDF page indices.

    ``page_number_offset`` follows the relationship:
    ``PDF physical page number = printed page number + offset``.
    """
    page_indices: list[int] = []
    for r in page_ranges_str.split(","):
        r = r.strip()
        if not r:
            raise ValueError(f"Invalid empty page item in: {page_ranges_str!r}")
        if "-" in r:
            parts = r.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid page range: {r!r}")
            start, end = map(int, parts)
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range: {r!r}")
            page_indices.extend(
                range(start + page_number_offset - 1, end + page_number_offset)
            )
        else:
            page = int(r)
            if page < 1:
                raise ValueError(f"Page number must be >= 1: {page}")
            page_indices.append(page + page_number_offset - 1)
    return sorted(set(page_indices))


_EDGE_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*)?(\d{1,4})(?:\s*(?:page|页))?$",
    flags=re.IGNORECASE,
)
_TOC_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:table\s+of\s+contents|contents|目\s*录)\s*(?:\n|$)", re.I
)
_TOC_ENTRY_RE = re.compile(
    r"^\s*(?P<title>.+?)\s*(?:\.{2,}|…{2,}|·{2,}|\s{2,})\s*(?P<page>\d{1,4})\s*$"
)


def _extract_edge_page_number(text: str) -> int | None:
    """Extract a printed Arabic page number from page header/footer evidence."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in [*lines[:3], *lines[-3:]]:
        match = _EDGE_PAGE_NUMBER_RE.fullmatch(line)
        if match:
            return int(match.group(1))
    return None


def _extract_toc_entries(text: str, source_pdf_page: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _TOC_ENTRY_RE.match(line)
        if not match:
            continue
        title = match.group("title").strip()
        if len(title) < 2:
            continue
        entries.append(
            {
                "title": title,
                "printed_page": int(match.group("page")),
                "source_pdf_page": source_pdf_page,
            }
        )
    return entries


def inspect_pdf(input_pdf_path: str, inspect_pages: int = 20) -> dict[str, Any]:
    """Collect front-matter evidence for an agent to resolve chapters and page offset.

    The heuristic only treats standalone numbers near a page header/footer as printed
    page-number evidence, avoiding most numbers listed inside a table of contents.
    Raw text is retained so an agent can verify the inferred mapping before extraction.
    """
    input_pdf = os.path.abspath(os.path.expanduser(input_pdf_path))
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    scan_count = min(inspect_pages, total_pages)

    inspected_pages: list[dict[str, Any]] = []
    toc_pages: list[int] = []
    toc_entries: list[dict[str, Any]] = []
    offset_evidence: list[dict[str, int]] = []

    for index in range(scan_count):
        physical_page = index + 1
        text = reader.pages[index].extract_text() or ""
        printed_page = _extract_edge_page_number(text)
        if printed_page is not None:
            offset_evidence.append(
                {
                    "pdf_page": physical_page,
                    "printed_page": printed_page,
                    "offset": physical_page - printed_page,
                }
            )

        entries = _extract_toc_entries(text, physical_page)
        is_toc = bool(_TOC_HEADING_RE.search(text)) or len(entries) >= 2
        if is_toc:
            toc_pages.append(physical_page)
            toc_entries.extend(entries)

        inspected_pages.append(
            {
                "pdf_page": physical_page,
                "printed_page": printed_page,
                "is_toc_candidate": is_toc,
                "text": text,
            }
        )

    offsets = Counter(item["offset"] for item in offset_evidence)
    inferred_offset: int | None = None
    confidence = 0.0
    if offsets:
        inferred_offset, support = offsets.most_common(1)[0]
        confidence = round(support / len(offset_evidence), 3)
        if support < 2:
            confidence = min(confidence, 0.5)

    mapped_entries = []
    for entry in toc_entries:
        mapped = dict(entry)
        mapped["inferred_pdf_page"] = (
            entry["printed_page"] + inferred_offset
            if inferred_offset is not None
            else None
        )
        mapped_entries.append(mapped)

    return {
        "input_pdf": input_pdf,
        "total_pdf_pages": total_pages,
        "inspected_pdf_pages": scan_count,
        "toc_candidate_pages": toc_pages,
        "toc_entries": mapped_entries,
        "page_number_offset": {
            "value": inferred_offset,
            "confidence": confidence,
            "evidence_count": len(offset_evidence),
            "relationship": "pdf_page = printed_page + offset",
            "evidence": offset_evidence,
        },
        "pages": inspected_pages,
    }


def extract_pages(
    input_pdf_path: str,
    output_pdf_path: str,
    page_indices: list[int],
) -> None:
    """Extract specified pages from a PDF and save to a new file."""
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    for i in page_indices:
        if 0 <= i < len(reader.pages):
            writer.add_page(reader.pages[i])
        else:
            Logger.warning(
                f"Page index {i} out of range (0..{len(reader.pages) - 1}), skip"
            )
    with open(output_pdf_path, "wb") as f:
        writer.write(f)


def run_extract_pages(params: ExtractPagesParams) -> list[str]:
    """从 PDF 中按页码范围提取子页面。

    返回输出 PDF 路径列表。
    """
    input_pdf_raw = params.input_pdf
    pages = params.pages
    output_dir = params.output_dir

    input_pdf = os.path.abspath(os.path.expanduser(input_pdf_raw))
    base, ext = os.path.splitext(input_pdf)

    # 如果指定了输出目录，则将文件写入该目录
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        input_name = os.path.splitext(os.path.basename(input_pdf))[0]
        base = os.path.join(output_dir, input_name)

    out_paths: list[str] = []
    for pages_str in pages:
        pages_str = pages_str.strip()
        out_path = f"{base}-{pages_str}{ext}"
        indices = parse_page_ranges(pages_str, params.page_number_offset)
        extract_pages(input_pdf, out_path, indices)
        out_paths.append(out_path)
        Logger.info(f"Extracted pages '{pages_str}' -> {out_path}")
    return out_paths

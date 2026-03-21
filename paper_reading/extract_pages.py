from __future__ import annotations

import os

from pypdf import PdfReader, PdfWriter

from .config import Config, ExtractPagesParams
from .log import Logger


def parse_page_ranges(page_ranges_str: str) -> list[int]:
    """
    Parse a string of page ranges (e.g. "1,3,5-7") into a list of 0-based page indices.
    """
    page_indices: list[int] = []
    for r in page_ranges_str.split(","):
        r = r.strip()
        if "-" in r:
            start, end = map(int, r.split("-"))
            page_indices.extend(range(start - 1, end))
        else:
            page_indices.append(int(r) - 1)
    return sorted(set(page_indices))


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
                f"Page index {i} out of range"
                f" (0..{len(reader.pages)-1}), skip"
            )
    with open(output_pdf_path, "wb") as f:
        writer.write(f)


def run_extract_pages(params: ExtractPagesParams | None = None) -> list[str]:
    """从 PDF 中按页码范围提取子页面。

    当 params 为 None 时回退到 Config.extract_pages（向后兼容 CLI）；
    传入 ExtractPagesParams 时使用调用方提供的参数。
    返回输出 PDF 路径列表。
    """
    if params is not None:
        input_pdf_raw = params.input_pdf
        pages = params.pages
        output_dir = params.output_dir
    else:
        conf = Config.extract_pages
        if not conf.input_pdf:
            raise ValueError("extract_pages.input_pdf is not set in config.yaml")
        if not conf.pages:
            raise ValueError("extract_pages.pages is empty in config.yaml")
        input_pdf_raw = conf.input_pdf
        pages = conf.pages
        output_dir = None

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
        indices = parse_page_ranges(pages_str)
        extract_pages(input_pdf, out_path, indices)
        out_paths.append(out_path)
        Logger.info(f"Extracted pages '{pages_str}' -> {out_path}")
    return out_paths

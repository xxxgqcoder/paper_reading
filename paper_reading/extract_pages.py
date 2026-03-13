import os

from pypdf import PdfReader, PdfWriter

from .config import Config
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


def run_extract_pages() -> list[str]:
    """
    Run extract-pages using Config.extract_pages settings.
    Returns list of output PDF paths.
    """
    conf = Config.extract_pages
    if not conf.input_pdf:
        raise ValueError("extract_pages.input_pdf is not set in config.yaml")
    if not conf.pages:
        raise ValueError("extract_pages.pages is empty in config.yaml")
    input_pdf = os.path.abspath(os.path.expanduser(conf.input_pdf))
    base, ext = os.path.splitext(input_pdf)
    out_paths: list[str] = []
    for pages_str in conf.pages:
        pages_str = pages_str.strip()
        out_path = f"{base}-{pages_str}{ext}"
        indices = parse_page_ranges(pages_str)
        extract_pages(input_pdf, out_path, indices)
        out_paths.append(out_path)
        Logger.info(f"Extracted pages '{pages_str}' -> {out_path}")
    return out_paths

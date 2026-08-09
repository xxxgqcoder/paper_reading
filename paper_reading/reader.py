"""Unified public entry point for PDF parsing, inspection, and extraction."""

from __future__ import annotations

import time

from .config import (
    ExtractPagesParams,
    PdfAction,
    PdfReadingParams,
    PdfReadingResult,
    ProcessParams,
)
from .extract_pages import inspect_pdf, run_extract_pages
from .pipeline import process


async def pdf_reading(params: PdfReadingParams) -> PdfReadingResult:
    """Run one PDF-reading action through a single structured entry point."""
    begin_ts = time.time()

    if params.action == PdfAction.PARSE:
        process_params = ProcessParams.model_validate(params.model_dump())
        result = await process(process_params)
        return PdfReadingResult(
            action=params.action,
            output_file=result.output_file,
            steps_completed=result.steps_completed,
            elapsed_seconds=result.elapsed_seconds,
        )

    if params.action == PdfAction.INSPECT:
        inspection = inspect_pdf(params.file_path, params.inspect_pages)
        return PdfReadingResult(
            action=params.action,
            inspection=inspection,
            elapsed_seconds=round(time.time() - begin_ts, 2),
        )

    extract_params = ExtractPagesParams(
        input_pdf=params.file_path,
        pages=params.pages,
        output_dir=params.output_dir or None,
        page_number_offset=params.page_number_offset,
    )
    output_files = run_extract_pages(extract_params)
    return PdfReadingResult(
        action=params.action,
        output_files=output_files,
        elapsed_seconds=round(time.time() - begin_ts, 2),
    )

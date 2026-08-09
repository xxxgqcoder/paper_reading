import asyncio

from paper_reading import PdfReadingParams, reader
from paper_reading.config import PdfAction, ProcessResult


def test_pdf_reading_parse_defaults_to_original(monkeypatch, tmp_path) -> None:
    captured = None

    async def _fake_process(params):
        nonlocal captured
        captured = params
        return ProcessResult(
            output_file=str(tmp_path / "paper.md"),
            steps_completed=params.steps,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(reader, "process", _fake_process)
    params = PdfReadingParams(
        action="parse",
        file_path=str(tmp_path / "paper.pdf"),
        output_dir=str(tmp_path),
    )

    result = asyncio.run(reader.pdf_reading(params))

    assert captured is not None
    assert captured.steps == ["original"]
    assert result.action == PdfAction.PARSE
    assert result.steps_completed == ["original"]


def test_pdf_reading_extract_passes_page_offset(monkeypatch, tmp_path) -> None:
    captured = None

    def _fake_extract(params):
        nonlocal captured
        captured = params
        return [str(tmp_path / "chapter.pdf")]

    monkeypatch.setattr(reader, "run_extract_pages", _fake_extract)
    params = PdfReadingParams(
        action="extract",
        file_path=str(tmp_path / "book.pdf"),
        output_dir=str(tmp_path),
        pages=["1-10"],
        page_number_offset=12,
    )

    result = asyncio.run(reader.pdf_reading(params))

    assert captured is not None
    assert captured.page_number_offset == 12
    assert result.output_files == [str(tmp_path / "chapter.pdf")]

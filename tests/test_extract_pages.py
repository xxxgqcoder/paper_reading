from __future__ import annotations

import paper_reading.extract_pages as extract_pages_module
from paper_reading.extract_pages import inspect_pdf, parse_page_ranges


def test_parse_page_ranges_applies_printed_page_offset() -> None:
    assert parse_page_ranges("1,3-4", page_number_offset=12) == [12, 14, 15]


def test_parse_page_ranges_rejects_invalid_ranges() -> None:
    for value in ("0", "4-2", "1--3", "1,"):
        try:
            parse_page_ranges(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {value!r}")


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _path: str) -> None:
        self.pages = [
            _FakePage("Cover"),
            _FakePage(
                "Contents\nChapter 1  Introduction .... 1\nChapter 2  Method .... 9"
            ),
            _FakePage("Preface"),
            _FakePage("Chapter 1\nIntroduction\n1"),
            _FakePage("Background\n2"),
        ]


def test_inspect_pdf_reports_toc_and_offset_evidence(monkeypatch) -> None:
    monkeypatch.setattr(extract_pages_module, "PdfReader", _FakeReader)

    result = inspect_pdf("book.pdf", inspect_pages=5)

    assert result["total_pdf_pages"] == 5
    assert result["toc_candidate_pages"] == [2]
    assert [entry["printed_page"] for entry in result["toc_entries"]] == [1, 9]
    assert result["page_number_offset"]["value"] == 3
    assert result["page_number_offset"]["confidence"] == 1.0
    assert result["toc_entries"][0]["inferred_pdf_page"] == 4

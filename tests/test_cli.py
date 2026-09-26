"""Tests for CLI helpers."""

from paper_reading.cli import _accepts_model_source


def test_accepts_model_source_detects_signature():
    def no_arg():
        return None

    def with_source(model_source):
        return model_source

    assert _accepts_model_source(no_arg) is False
    assert _accepts_model_source(with_source) is True


def test_accepts_model_source_tolerates_non_callable():
    assert _accepts_model_source(object()) is False

"""Tests for MineRUParser asset materialization.

聚焦验证：解析结果命中缓存时，图片仍会被同步到 asset_save_dir，
content_url 被改写为目标路径——即修复「命中缓存即丢图」。
"""

from pathlib import Path

from paper_reading.config import Content, ContentType
from paper_reading.pdf_parser import MineRUParser


def _make_image_content(img_path: Path) -> Content:
    return Content(
        content_type=ContentType.IMAGE,
        file_path="dummy.pdf",
        content="",
        extra_description="",
        content_url=str(img_path),
    )


def test_materialize_copies_assets_and_rewrites_url(tmp_path) -> None:
    src_dir = tmp_path / "paper_images"
    src_dir.mkdir()
    img = src_dir / "abc123.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg-ish bytes

    contents = [_make_image_content(img)]
    dst = tmp_path / "attachments"

    MineRUParser._materialize_assets(contents, str(dst))

    copied = dst / "abc123.jpg"
    assert copied.is_file()
    assert copied.read_bytes() == b"\xff\xd8\xff\xd9"
    # content_url 改写为目标目录中的路径，basename 不变 -> wikilink 一致
    assert contents[0].content_url == str(copied.resolve())


def test_materialize_is_idempotent(tmp_path) -> None:
    src_dir = tmp_path / "paper_images"
    src_dir.mkdir()
    img = src_dir / "abc123.jpg"
    img.write_bytes(b"original")

    dst = tmp_path / "attachments"
    dst.mkdir()
    # 目标已存在且内容不同：幂等跳过复制，不覆盖
    (dst / "abc123.jpg").write_bytes(b"existing")

    contents = [_make_image_content(img)]
    MineRUParser._materialize_assets(contents, str(dst))

    assert (dst / "abc123.jpg").read_bytes() == b"existing"
    assert contents[0].content_url == str((dst / "abc123.jpg").resolve())


def test_materialize_skips_missing_source(tmp_path) -> None:
    missing = tmp_path / "paper_images" / "gone.jpg"
    contents = [_make_image_content(missing)]
    dst = tmp_path / "attachments"

    # 源缺失时不抛异常、不创建目标，content_url 保持原值
    MineRUParser._materialize_assets(contents, str(dst))

    assert not (dst / "gone.jpg").exists()
    assert contents[0].content_url == str(missing)


def test_materialize_ignores_content_without_url(tmp_path) -> None:
    text = Content(
        content_type=ContentType.TEXT,
        file_path="dummy.pdf",
        content="hello",
        extra_description="",
        content_url="",
    )
    dst = tmp_path / "attachments"

    MineRUParser._materialize_assets([text], str(dst))

    assert text.content_url == ""

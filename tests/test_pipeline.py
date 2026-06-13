"""回归测试：写出含非法 surrogate 码点的 Markdown 不应崩溃。

MineRU 从部分 PDF 抽取的文本可能带孤立 surrogate（如 '\\ud83d'），严格
UTF-8 编码会在写盘时抛 "surrogates not allowed" 并中断输出。process() 用
errors="replace" 容错，保证整篇 Markdown 写完且可正常读回。
"""

import asyncio

from paper_reading import pipeline
from paper_reading.config import ProcessParams, Step


def test_process_writes_surrogate_without_crashing(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    out = tmp_path / "out"

    # 避免触发真实 MineRU 解析
    monkeypatch.setattr(pipeline, "parse_pdf", lambda **kwargs: [])

    surrogate_text = "前缀\ud83d后缀"  # 孤立 high surrogate，无法严格 UTF-8 编码

    async def _fake_original(ctx) -> None:
        ctx.md_writer.write(surrogate_text)

    monkeypatch.setattr(pipeline, "STEP_REGISTRY", {Step.ORIGINAL: _fake_original})

    params = ProcessParams(
        file_path=str(pdf),
        output_dir=str(out),
        steps=["original"],
    )

    # 修复前这里会抛 UnicodeEncodeError: surrogates not allowed
    result = asyncio.run(pipeline.process(params))

    md = out / "paper.md"
    assert md.is_file()
    assert result.output_file == str(md)

    # 能正常读回（不抛 UnicodeDecodeError），非法 surrogate 已被替换
    content = md.read_text(encoding="utf-8")
    assert "前缀" in content
    assert "后缀" in content
    assert "\ud83d" not in content

import os
import time

from .config import (
    STEP_OUTPUT_ORDER,
    Content,
    ProcessParams,
    ProcessResult,
)
from .log import Logger
from .pdf_parser import MineRUParser
from .steps import STEP_REGISTRY, StepContext, parse_steps
from .utils import line_breaker, time_it


@time_it(prefix="parse_pdf")
def parse_pdf(
    file_path: str,
    model_source: str,
    device: str,
    asset_save_dir: str,
) -> list[Content]:
    Logger.info(f"Begin to parse file: {file_path}")
    parser = MineRUParser(
        model_source=model_source,
        device=device,
    )
    content_list = parser.parse(file_path, asset_save_dir=asset_save_dir)
    Logger.info(f"Parsed {len(content_list)} contents from {file_path}")
    return content_list


async def process(params: ProcessParams) -> ProcessResult:
    """处理 PDF 文件并输出 Markdown。

    这是面向 AI tool/skill 的主入口，所有运行时参数通过 ProcessParams 传入。
    """
    begin_ts = time.time()

    enabled_steps = parse_steps(params.steps)

    step_names = [s.value for s in STEP_OUTPUT_ORDER if s in enabled_steps]
    Logger.info(f"Processing started, enabled steps: {step_names}")

    os.makedirs(params.output_dir, exist_ok=True)
    name_without_suff = os.path.basename(params.file_path).rsplit(".", 1)[0]
    Logger.info(f"File name without suffix: {name_without_suff}")

    md_file_path = os.path.join(params.output_dir, f"{name_without_suff}.md")
    if os.path.isfile(md_file_path) and os.path.getsize(md_file_path) > 0:
        raise FileExistsError(
            f"Output file already exists and is non-empty: {md_file_path}. "
            "Remove it first if you want to regenerate."
        )

    content_list = parse_pdf(
        file_path=params.file_path,
        model_source=params.mineru_model_source,
        device=params.mineru_device,
        asset_save_dir=params.asset_save_dir,
    )

    Logger.info(f"md file path: {md_file_path}")

    ctx = StepContext(
        md_writer=None,  # type: ignore[arg-type]
        content_list=content_list,
    )
    # errors="replace": MineRU 从部分 PDF 抽取的文本可能含非法 surrogate 码点，
    # 严格 UTF-8 编码会在写盘时抛 "surrogates not allowed" 并中断输出，
    # 用替换策略容错，保证整篇 Markdown 写完。
    with open(md_file_path, "w", encoding="utf-8", errors="replace") as md_writer:
        ctx.md_writer = md_writer
        md_writer.write(f"{name_without_suff}" + line_breaker)

        for step in STEP_OUTPUT_ORDER:
            if step not in enabled_steps:
                continue
            Logger.info(f"Processing step: {step.value}")
            await STEP_REGISTRY[step](ctx)

    elapsed = round(time.time() - begin_ts, 2)
    Logger.info(f"Parsed markdown saved to {md_file_path}")
    return ProcessResult(
        output_file=md_file_path,
        steps_completed=step_names,
        elapsed_seconds=elapsed,
    )

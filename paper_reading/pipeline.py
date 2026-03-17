import os
import time

from .config import (
    LANG_MAPPING,
    STEP_OUTPUT_ORDER,
    Config,
    Content,
    ProcessParams,
    ProcessResult,
)
from .log import Logger
from .pdf_parser import PDFParser, ensure_mineru_model, prepare_mineru_runtime_config
from .steps import STEP_REGISTRY, StepContext, parse_steps
from .utils import line_breaker, time_it


@time_it(prefix="parse_pdf")
def parse_pdf(
    file_path: str, runtime_config_path: str
) -> list[Content]:
    Logger.info(f"Begin to parse file: {file_path}")
    parser = PDFParser(runtime_config_path)
    content_list = parser.parse(file_path)
    Logger.info(f"Parsed {len(content_list)} contents from {file_path}")
    return content_list


def process(params: ProcessParams) -> ProcessResult:
    """处理 PDF 文件并输出 Markdown。

    这是面向 AI tool/skill 的主入口，所有运行时参数通过 ProcessParams 传入。
    """
    begin_ts = time.time()

    # 合并参数：调用参数 > Config 默认值
    chat_model = params.chat_model_name or Config.chat_model_name
    vision_model = params.vision_model_name or Config.vision_model_name
    gen_conf = (
        params.gen_conf
        if params.gen_conf is not None
        else Config.gen_conf.model_dump()
    )

    src_lang_display = LANG_MAPPING.get(params.src_lang, params.src_lang)
    target_lang_display = LANG_MAPPING.get(params.target_lang, params.target_lang)

    enabled_steps = parse_steps(params.steps)
    step_names = [s.value for s in enabled_steps]
    Logger.info(f"Processing started, enabled steps: {step_names}")

    model_dir = ensure_mineru_model()
    runtime_config_path = prepare_mineru_runtime_config(model_dir)

    os.makedirs(params.output_dir, exist_ok=True)
    name_without_suff = os.path.basename(params.file_path).rsplit(".", 1)[0]
    Logger.info(f"File name without suffix: {name_without_suff}")

    content_list = parse_pdf(
        file_path=params.file_path,
        runtime_config_path=runtime_config_path,
    )

    md_file_path = os.path.join(params.output_dir, f"{name_without_suff}.md")
    Logger.info(f"md file path: {md_file_path}")

    ctx = StepContext(
        md_writer=None,  # type: ignore[arg-type]
        content_list=content_list,
        src_lang=src_lang_display,
        target_lang=target_lang_display,
        chat_model_name=chat_model,
        vision_model_name=vision_model,
        gen_conf=gen_conf,
        prompt_translate=Config.prompt_translate,
        prompt_summary=Config.prompt_summary,
        max_context_token_num=Config.max_context_token_num,
        asset_save_dir=Config.asset_save_dir,
    )
    with open(md_file_path, "w") as md_writer:
        ctx.md_writer = md_writer
        md_writer.write(f"{name_without_suff}" + line_breaker)

        for step in STEP_OUTPUT_ORDER:
            if step not in enabled_steps:
                continue
            Logger.info(f"Processing step: {step.value}")
            STEP_REGISTRY[step](ctx)

    elapsed = round(time.time() - begin_ts, 2)
    Logger.info(f"Parsed markdown saved to {md_file_path}")
    return ProcessResult(
        output_file=md_file_path,
        steps_completed=step_names,
        elapsed_seconds=elapsed,
    )

import os

from .config import STEP_OUTPUT_ORDER, Content
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


@time_it(prefix="process pipeline")
def process(
    file_path: str,
    final_md_file_save_dir: str,
    steps: list[str],
    src_lang: str = "英语",
    target_lang: str = "中文",
) -> str:
    """
    Process a pdf file and save parsed markdown file.

    Returns path to the output markdown file.
    """
    enabled_steps = parse_steps(steps)
    step_names = [s.value for s in enabled_steps]
    Logger.info(f"Processing started, enabled steps: {step_names}")

    model_dir = ensure_mineru_model()
    runtime_config_path = prepare_mineru_runtime_config(model_dir)

    os.makedirs(final_md_file_save_dir, exist_ok=True)
    name_without_suff = os.path.basename(file_path).rsplit(".", 1)[0]
    Logger.info(f"File name without out suffix: {name_without_suff}")

    content_list = parse_pdf(
        file_path=file_path,
        runtime_config_path=runtime_config_path,
    )

    md_file_path = os.path.join(final_md_file_save_dir, f"{name_without_suff}.md")
    Logger.info(f"md file path: {md_file_path}")
    ctx = StepContext(
        md_writer=None,  # type: ignore[arg-type]
        content_list=content_list,
        src_lang=src_lang,
        target_lang=target_lang,
    )
    with open(md_file_path, "w") as md_writer:
        ctx.md_writer = md_writer
        md_writer.write(f"{name_without_suff}" + line_breaker)

        for step in STEP_OUTPUT_ORDER:
            if step not in enabled_steps:
                continue
            Logger.info(f"Processing step: {step.value}")
            STEP_REGISTRY[step](ctx)

    Logger.info(f"Parsed markdown saved to {md_file_path}")
    return md_file_path

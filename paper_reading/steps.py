import os
from collections.abc import Callable
from dataclasses import dataclass
from io import TextIOWrapper

from .config import Config, Content, ContentType, Step
from .llm import llm_chat
from .log import Logger
from .utils import (
    ensure_utf,
    estimate_token_num,
    is_empty,
    line_breaker,
    relative_md_image_path,
    time_it,
)


@dataclass
class StepContext:
    md_writer: TextIOWrapper
    content_list: list[Content]
    src_lang: str
    target_lang: str


# ---------------------------------------------------------------------------
# original


def save_parsed_content(ctx: StepContext) -> None:
    ctx.md_writer.write(
        "# " + "=" * 4 + "  Original Content  " + "=" * 4 + line_breaker
    )

    for _, content in enumerate(ctx.content_list):
        lines = ""
        if content.content_type == ContentType.TEXT:
            lines += content.content + line_breaker
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            if content.content_url:
                img_name = os.path.basename(content.content_url)
                md_img_path = relative_md_image_path(
                    sys_image_folder=os.path.dirname(content.content_url),
                    img_name=img_name,
                )
                lines += md_img_path + line_breaker

            lines += f"{line_breaker}{content.extra_description}{line_breaker}"
        ctx.md_writer.write(lines)
        ctx.md_writer.flush()


# ---------------------------------------------------------------------------
# translate


def translate_text_content(text: str, src_lang: str, target_lang: str) -> str:
    if is_empty(text):
        return ""

    max_char_len = 16 * 1024
    full_result = ""
    for i in range(0, len(text), max_char_len):
        Logger.info(f"Processing segment {i}")
        segment = text[i : i + max_char_len]

        formatted_prompt = Config.prompt_translate.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=segment,
        )
        ret = llm_chat(prompt=formatted_prompt, gen_conf=Config.gen_conf.model_dump())
        if not ret:
            ret = "[LLM error]"

        full_result += ret

    return full_result


@time_it(prefix="translate_content")
def translate_content(ctx: StepContext) -> None:
    """Translate contents."""
    content_list, src_lang, target_lang = (
        ctx.content_list,
        ctx.src_lang,
        ctx.target_lang,
    )
    Logger.info(f"Total {len(content_list)} contents")

    ctx.md_writer.write(
        "# " + "=" * 4 + "  Translated Content  " + "=" * 4 + line_breaker
    )

    i = 0
    max_content_num = 20
    while i < len(content_list):
        if content_list[i].content_type in [ContentType.TABLE, ContentType.IMAGE]:
            Logger.info(
                f"Translating content {i}, type: {content_list[i].content_type}"
            )
            if content_list[i].content_url:
                img_name = os.path.basename(content_list[i].content_url)
                img_path = relative_md_image_path(Config.asset_save_dir, img_name)
                ctx.md_writer.write(img_path + line_breaker)

            translated = translate_text_content(
                content_list[i].extra_description, src_lang, target_lang
            )
            Logger.info(f"Translated content:\n{translated}")
            ctx.md_writer.write(translated + line_breaker)

            i = i + 1
            continue

        j = i + 1
        while (
            j < len(content_list)
            and j - i < max_content_num
            and content_list[j].content_type == ContentType.TEXT
        ):
            j += 1
        Logger.info(
            f"Translating content {i} to {j - 1}, type: {content_list[i].content_type}"
        )
        content = "\n".join([content.content for content in content_list[i:j]])
        content = ensure_utf(content)
        Logger.info(f"Content to translate:\n{content}")
        translated = translate_text_content(content, src_lang, target_lang)
        Logger.info(f"Translated content:\n{translated}")
        ctx.md_writer.write(translated + line_breaker)

        i = j

    ctx.md_writer.flush()


# ---------------------------------------------------------------------------
# summary


@time_it(prefix="summary_content")
def summary_content(ctx: StepContext) -> None:
    src_lang, target_lang = ctx.src_lang, ctx.target_lang
    Logger.info(f"Summary_content, src_lang={src_lang}, target_lang={target_lang}")

    full_content = ""
    for content in ctx.content_list:
        if content.content_type == ContentType.TEXT:
            full_content += content.content + line_breaker
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            full_content += content.extra_description + line_breaker
        else:
            Logger.info(f"Unrecognized content: {content}")

    Logger.info(f"Full content length: {len(full_content)}")
    token_num, _ = estimate_token_num(full_content)
    Logger.info(f"Estimated full content token num: {token_num}")
    if token_num > Config.max_context_token_num:
        ratio = float(Config.max_context_token_num) / token_num
        Logger.info(
            f"Truncate full content by ratio: {ratio},"
            f" original length: {len(full_content)}"
        )
        full_content = full_content[: int(len(full_content) * ratio)]

    full_content = ensure_utf(full_content)

    formatted_prompt = Config.prompt_summary.format(
        src_lang=src_lang,
        target_lang=target_lang,
        content=full_content,
    )
    Logger.info(f"Formatted prompt:\n{formatted_prompt}")

    summary = llm_chat(prompt=formatted_prompt, gen_conf=Config.gen_conf.model_dump())
    if not summary:
        summary = "[LLM error]"
    Logger.info(f"Content summary:\n{summary}")

    ctx.md_writer.write(
        "# " + "=" * 4 + "  Content Summary  " + "=" * 4 + line_breaker
    )
    ctx.md_writer.write(summary + line_breaker)
    ctx.md_writer.write("-" * 4 + line_breaker)
    ctx.md_writer.flush()


# ---------------------------------------------------------------------------
# registry


STEP_REGISTRY: dict[Step, Callable[[StepContext], None]] = {
    Step.ORIGINAL: save_parsed_content,
    Step.SUMMARY: summary_content,
    Step.TRANSLATE: translate_content,
}


def parse_steps(raw: list[str]) -> set[Step]:
    """Validate and convert raw step strings to Step enums. Raises on unknown names."""
    enabled: set[Step] = set()
    for s in raw:
        s = s.strip()
        try:
            enabled.add(Step(s))
        except ValueError:
            valid = [e.value for e in Step]
            raise ValueError(f"Unknown step: '{s}'. Valid steps: {valid}") from None
    return enabled

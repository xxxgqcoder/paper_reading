import asyncio
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Any

from .config import Content, ContentType, Step
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
    # 以下字段由 pipeline.process() 从 Config / ProcessParams 合并后注入
    chat_model_name: str = ""
    vision_model_name: str = ""
    gen_conf: dict[str, Any] = field(default_factory=dict)
    prompt_translate: str = ""
    prompt_summary: str = ""
    max_context_token_num: int = 1024 * 16
    asset_save_dir: str = ""


# ---------------------------------------------------------------------------
# original


async def save_parsed_content(ctx: StepContext) -> None:
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


async def translate_text_content(
    text: str, src_lang: str, target_lang: str, ctx: StepContext
) -> str:
    if is_empty(text):
        return ""

    max_char_len = 16 * 1024
    segments = [text[i : i + max_char_len] for i in range(0, len(text), max_char_len)]

    async def _translate_segment(idx: int, segment: str) -> str:
        Logger.info(f"Processing segment {idx}")
        formatted_prompt = ctx.prompt_translate.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=segment,
        )
        ret = await llm_chat(
            prompt=formatted_prompt,
            gen_conf=ctx.gen_conf,
            model=ctx.chat_model_name,
        )
        if not ret:
            ret = "[LLM error]"
        return ret

    # 并发翻译所有分段
    results = await asyncio.gather(
        *[_translate_segment(i, seg) for i, seg in enumerate(segments)]
    )
    return "".join(results)


@time_it(prefix="translate_content")
async def translate_content(ctx: StepContext) -> None:
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

    # 收集所有翻译任务，按原始顺序分组
    groups: list[tuple[str, list[int], str | None]] = []  # (type, indices, img_path)
    i = 0
    max_content_num = 20
    while i < len(content_list):
        if content_list[i].content_type in [ContentType.TABLE, ContentType.IMAGE]:
            img_path = None
            if content_list[i].content_url:
                img_name = os.path.basename(content_list[i].content_url)
                img_path = relative_md_image_path(ctx.asset_save_dir, img_name)
            groups.append(("media", [i], img_path))
            i += 1
            continue

        j = i + 1
        while (
            j < len(content_list)
            and j - i < max_content_num
            and content_list[j].content_type == ContentType.TEXT
        ):
            j += 1
        groups.append(("text", list(range(i, j)), None))
        i = j

    # 并发执行所有分组的翻译
    async def _translate_group(
        group: tuple[str, list[int], str | None],
    ) -> str:
        gtype, indices, img_path = group
        result = ""
        if gtype == "media":
            idx = indices[0]
            Logger.info(
                f"Translating content {idx}, type: {content_list[idx].content_type}"
            )
            if img_path:
                result += img_path + line_breaker
            translated = await translate_text_content(
                content_list[idx].extra_description, src_lang, target_lang, ctx
            )
            Logger.info(f"Translated content:\n{translated}")
            result += translated + line_breaker
        else:
            Logger.info(
                f"Translating content {indices[0]} to {indices[-1]},"
                f" type: {content_list[indices[0]].content_type}"
            )
            content = "\n".join([content_list[idx].content for idx in indices])
            content = ensure_utf(content)
            Logger.info(f"Content to translate:\n{content}")
            translated = await translate_text_content(
                content, src_lang, target_lang, ctx
            )
            Logger.info(f"Translated content:\n{translated}")
            result += translated + line_breaker
        return result

    translated_parts = await asyncio.gather(*[_translate_group(g) for g in groups])

    # 按原始顺序写入
    for part in translated_parts:
        ctx.md_writer.write(part)
    ctx.md_writer.flush()


# ---------------------------------------------------------------------------
# summary


@time_it(prefix="summary_content")
async def summary_content(ctx: StepContext) -> None:
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
    if token_num > ctx.max_context_token_num:
        ratio = float(ctx.max_context_token_num) / token_num
        Logger.info(
            f"Truncate full content by ratio: {ratio},"
            f" original length: {len(full_content)}"
        )
        full_content = full_content[: int(len(full_content) * ratio)]

    full_content = ensure_utf(full_content)

    formatted_prompt = ctx.prompt_summary.format(
        src_lang=src_lang,
        target_lang=target_lang,
        content=full_content,
    )
    Logger.info(f"Formatted prompt:\n{formatted_prompt}")

    summary = await llm_chat(
        prompt=formatted_prompt,
        gen_conf=ctx.gen_conf,
        model=ctx.chat_model_name,
    )
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


STEP_REGISTRY: dict[Step, Callable[[StepContext], Coroutine[Any, Any, None]]] = {
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

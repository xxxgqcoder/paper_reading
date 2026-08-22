import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from io import TextIOWrapper

from .config import Content, ContentType, Step
from .utils import line_breaker


@dataclass
class StepContext:
    md_writer: TextIOWrapper
    content_list: list[Content]


async def save_parsed_content(ctx: StepContext) -> None:
    ctx.md_writer.write(
        "# " + "=" * 4 + "  Original Content  " + "=" * 4 + line_breaker
    )

    for content in ctx.content_list:
        lines = ""
        if content.content_type == ContentType.TEXT:
            lines += content.content + line_breaker
        elif content.content_type == ContentType.FORMULA:
            # LaTeX 公式块，用 $$ 包裹保留数学格式
            lines += f"\n$$\n{content.content}\n$$\n" + line_breaker
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            if content.content_url:
                # Obsidian wikilink 格式：仅文件名，由 Obsidian 自动解析路径
                img_name = os.path.basename(content.content_url)
                lines += f"![[{img_name}]]" + line_breaker + line_breaker
            elif content.content_type == ContentType.TABLE and content.content:
                # Markdown 格式的表格
                lines += content.content + line_breaker

            lines += f"{line_breaker}{content.extra_description}{line_breaker}"
        ctx.md_writer.write(lines)
        ctx.md_writer.flush()


STEP_REGISTRY: dict[Step, Callable[[StepContext], Coroutine[object, object, None]]] = {
    Step.ORIGINAL: save_parsed_content,
}


def parse_steps(raw: list[str]) -> set[Step]:
    """Validate and convert raw step strings to Step enums."""
    enabled: set[Step] = set()
    for s in raw:
        s = s.strip()
        try:
            enabled.add(Step(s))
        except ValueError:
            valid = [e.value for e in Step]
            raise ValueError(f"Unknown step: '{s}'. Valid steps: {valid}") from None
    return enabled

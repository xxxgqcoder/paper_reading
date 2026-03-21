import os
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from strenum import StrEnum


class Step(str, Enum):
    SUMMARY = "summary"
    TRANSLATE = "translate"
    ORIGINAL = "original"


STEP_OUTPUT_ORDER: list[Step] = [Step.SUMMARY, Step.TRANSLATE, Step.ORIGINAL]


# ---------------------------------------------------------------------------
# 默认 prompt 模板
# ---------------------------------------------------------------------------
DEFAULT_PROMPT_TRANSLATE = (
    "你是一个翻译助手，请将下面的{src_lang}内容翻译成{target_lang}。\n"
    "\n"
    "下面是需要翻译的内容：\n"
    "\n"
    "{content}\n"
    "----\n"
    "\n"
    "注意：\n"
    "- 如果要翻译的内容为引用文献、算法伪代码、代码、人名，"
    "则不需要翻译，直接返回原文\n"
    "- 你只需要输出最终翻译结果，不要输出逐步思考过程。\n"
    "\n"
    "现在开始逐步思考"
)

DEFAULT_PROMPT_SUMMARY = (
    "你是一个阅读助手，阅读下面的{src_lang}内容，并完成指令。\n"
    "\n"
    "下面是文档内容\n"
    "\n"
    "{content}\n"
    "----\n"
    "\n"
    "指令：使用{target_lang}语言，总结{src_lang}内容。\n"
    "- 如果是学术论文，请总结论文的主要贡献、方法、实验和结论。\n"
    "- 如果是技术文档，请总结技术的核心概念、功能和应用场景。\n"
    "- 如果是一般性文档（书籍，文章等），请总结文档的主要内容和关键点。\n"
    "\n"
    "注意：\n"
    "- 忽略引用部分内容，只总结文档正文部分。\n"
    "- 你只需要输出最终总结结果，不要输出逐步思考过程。\n"
    "\n"
    "现在开始逐步思考"
)


def _default_gen_conf() -> dict:
    """默认生成参数。"""
    return {
        "temperature": 0.7,
        "top_p": 0.3,
        "repeat_penalty": 1.1,
        "num_ctx": 1024 * 16,
    }


class ContentType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


class Content(BaseModel):
    """Document content object."""

    content_type: ContentType = Field(
        default=ContentType.TEXT, description="content type"
    )
    file_path: str = Field(default="", description="original file path")
    content: str = Field(
        default="",
        description=(
            "the content, represented in string. If content type is not image / table, "
            "this field will be base64 encoded image content."
        ),
    )
    extra_description: str = Field("", description="content extra description")
    content_url: str = Field(
        default="",
        description=(
            "url to the content, set when content is not suitable for directly insert "
            "into db, for example image / audio data"
        ),
    )


LANG_MAPPING = {"en": "英语", "zh": "中文"}


def _expand_path(v: str) -> str:
    """展开 ~ 并转为绝对路径。"""
    if v:
        v = os.path.expanduser(v)
        return os.path.abspath(v)
    return v


# ---------------------------------------------------------------------------
# AI Tool / Skill 调用入参与返回值
# ---------------------------------------------------------------------------
@dataclass
class ProcessParams:
    """每次调用 process() 的独立参数，AI 调用方只需构造此对象。"""

    file_path: str
    output_dir: str
    steps: list[str] = field(
        default_factory=lambda: ["summary", "translate", "original"]
    )
    src_lang: str = "en"
    target_lang: str = "zh"

    # LLM 连接
    llm_endpoint: str = ""
    llm_api_key: str = ""

    # 模型与生成参数
    chat_model_name: str = "llama3"
    vision_model_name: str = "llama3"
    gen_conf: dict = field(default_factory=_default_gen_conf)
    max_context_token_num: int = 1024 * 16

    # 路径
    asset_save_dir: str = "attachments"
    cache_data_dir: str = "~/.cache/llm_cache"

    # prompt 模板
    prompt_translate: str = DEFAULT_PROMPT_TRANSLATE
    prompt_summary: str = DEFAULT_PROMPT_SUMMARY

    def __post_init__(self) -> None:
        # 从环境变量回退读取 LLM 连接信息
        if not self.llm_endpoint:
            self.llm_endpoint = os.environ.get("LLM_ENDPOINT", "")
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        # 路径展开
        self.asset_save_dir = _expand_path(self.asset_save_dir)
        self.cache_data_dir = _expand_path(self.cache_data_dir)


@dataclass
class ProcessResult:
    """process() 的结构化返回值。"""

    output_file: str
    steps_completed: list[str]
    elapsed_seconds: float


@dataclass
class ExtractPagesParams:
    """每次调用 run_extract_pages() 的独立参数，AI 调用方只需构造此对象。"""

    input_pdf: str
    pages: list[str]
    output_dir: str | None = None

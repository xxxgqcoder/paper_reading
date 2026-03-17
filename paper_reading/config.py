import os
import warnings
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from strenum import StrEnum


def get_project_base_directory() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Step(str, Enum):
    SUMMARY = "summary"
    TRANSLATE = "translate"
    ORIGINAL = "original"


STEP_OUTPUT_ORDER: list[Step] = [Step.SUMMARY, Step.TRANSLATE, Step.ORIGINAL]


class GenerationConf(BaseSettings):
    temperature: float = Field(
        default=0.7, description="Temperature for text generation."
    )
    top_p: float = Field(
        default=0.3, description=" Top-p (nucleus) sampling parameter."
    )
    repeat_penalty: float = Field(
        default=1.1, description=" Repetition penalty for text generation."
    )
    num_ctx: int = Field(
        default=1024 * 16, description=" Maximum context length for the model."
    )


class ExtractPagesConf(BaseModel):
    input_pdf: str = Field(
        default="", description="path to input PDF for page extraction"
    )
    pages: list[str] = Field(
        default_factory=list,
        description="page ranges to extract, e.g. ['1-93', '100,105-110']",
    )


class _Config(BaseSettings):
    """Application configuration settings."""

    asset_save_dir: str = Field(
        "",
        description="directory to save parsed asset files, including images and tables",
    )
    cache_data_dir: str = Field(
        default="~/.cache/llm_cache", description="cache data directory"
    )
    max_context_token_num: int = Field(
        default=1024 * 16, description="max context token num"
    )
    gen_conf: GenerationConf = Field(
        default_factory=GenerationConf, description="Generation configuration."
    )
    chat_model_name: str = Field(default="llama3", description="chat model name")
    vision_model_name: str = Field(default="llama3", description="vision model name")

    steps: str = Field(
        default="summary,translate,original",
        description="comma-separated processing steps",
    )
    src_lang: str = Field(default="en", description="source language code (en/zh)")
    target_lang: str = Field(default="zh", description="target language code (en/zh)")
    temp_content_dir: str = Field(
        default="./tmp/parsed_asset",
        description="temp directory for parsed content",
    )

    extract_pages: ExtractPagesConf = Field(
        default_factory=ExtractPagesConf,
        description="configuration for extract-pages command",
    )

    prompt_translate: str = Field(
        default=(
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
        ),
        description=(
            "Translation prompt template."
            " Placeholders: {src_lang}, {target_lang}, {content}"
        ),
    )
    prompt_summary: str = Field(
        default=(
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
        ),
        description=(
            "Summary prompt template."
            " Placeholders: {src_lang}, {target_lang}, {content}"
        ),
    )

    model_config = SettingsConfigDict(
        yaml_file=os.path.join(get_project_base_directory(), "config.yaml"),
        env_prefix="CONFIG@@",
        env_nested_delimiter="@@",
        nested_model_default_partial_update=True,
    )

    @field_validator(
        "cache_data_dir",
        "asset_save_dir",
        "temp_content_dir",
        mode="after",
    )
    @classmethod
    def expand_home_path(cls, v: str) -> str:
        """Expand ~ and resolve relative paths against project base directory."""
        if v:
            v = os.path.expanduser(v)
            if not os.path.isabs(v):
                v = os.path.join(get_project_base_directory(), v)
            return os.path.abspath(v)
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = os.path.join(get_project_base_directory(), "config.yaml")
        sources: list[PydanticBaseSettingsSource] = [env_settings]
        if os.path.exists(yaml_path):
            sources.append(YamlConfigSettingsSource(settings_cls))
        return tuple(sources)


Config = _Config()  # type: ignore


# ---------------------------------------------------------------------------
# 兼容检查：config.yaml 中仍存在 llm_endpoint / llm_api_key 时打印弃用告警
# ---------------------------------------------------------------------------
def _warn_deprecated_yaml_fields() -> None:
    """检查 config.yaml 中是否残留已迁移到环境变量的字段，打印弃用告警。"""
    import yaml

    yaml_path = os.path.join(get_project_base_directory(), "config.yaml")
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _deprecated = {"llm_endpoint", "llm_api_key"}
    for key in _deprecated:
        if key in raw:
            warnings.warn(
                f"'{key}' in config.yaml is deprecated and will be ignored. "
                f"Please set the environment variable '{key.upper()}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )


_warn_deprecated_yaml_fields()


# ---------------------------------------------------------------------------
# 从环境变量读取 LLM 连接配置
# ---------------------------------------------------------------------------
def get_llm_endpoint() -> str:
    """从环境变量 LLM_ENDPOINT 读取 LLM API 地址，未设置时抛出异常。"""
    val = os.environ.get("LLM_ENDPOINT", "")
    if not val:
        raise ValueError(
            "Environment variable LLM_ENDPOINT is not set. "
            "Please set it to the LLM API endpoint URL, "
            "e.g. 'http://127.0.0.1:11434' for Ollama "
            "or 'https://api.openai.com/v1' for OpenAI-compatible APIs."
        )
    return val


def get_llm_api_key() -> str:
    """从环境变量 LLM_API_KEY 读取 API Key，未设置时返回空字符串（Ollama 模式）。"""
    return os.environ.get("LLM_API_KEY", "")


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
    # 以下可选字段为 None 时回退到 Config 中的默认值
    chat_model_name: str | None = None
    vision_model_name: str | None = None
    gen_conf: dict | None = None


@dataclass
class ProcessResult:
    """process() 的结构化返回值。"""

    output_file: str
    steps_completed: list[str]
    elapsed_seconds: float

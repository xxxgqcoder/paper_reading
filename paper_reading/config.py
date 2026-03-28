import os
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator
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


class GenerationConfig(BaseModel):
    """LLM 生成配置参数。"""
    temperature: float = Field(default=0.7, description="采样温度，越高生成内容越随机")
    top_p: float = Field(default=0.3, description="核采样阈值")
    repeat_penalty: float = Field(default=1.1, description="重复惩罚系数")
    num_ctx: int = Field(default=1024 * 16, description="上下文窗口大小")


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
class ProcessParams(BaseModel):
    """处理文档的主要参数配置模型。"""

    file_path: str = Field(..., description="待处理的 PDF 文件路径")
    output_dir: str = Field(..., description="Markdown 输出文件的保存目录")
    
    steps: List[str] = Field(
        default_factory=lambda: ["summary", "translate", "original"],
        description="处理步骤列表，可选值: summary, translate, original"
    )
    src_lang: str = Field(default="en", description="源文档语言代码，默认 'en'")
    target_lang: str = Field(default="zh", description="目标翻译语言代码，默认 'zh'")

    # LLM 连接
    llm_endpoint: str = Field(default="", description="LLM 服务端点 URL")
    llm_api_key: str = Field(default="", description="LLM API Key")

    # 模型与生成参数
    chat_model_name: str = Field(default="llama3", description="聊天模型名称")
    vision_model_name: str = Field(default="llama3", description="视觉模型名称")
    
    gen_conf: GenerationConfig = Field(
        default_factory=GenerationConfig, 
        description="生成参数配置 (temperature, top_p, etc.)"
    )
    
    max_context_token_num: int = Field(
        default=1024 * 16, 
        description="用于总结的最大上下文 token 数"
    )

    # 路径
    asset_save_dir: str = Field(
        default="attachments", 
        description="解析后的图片等资源保存目录"
    )
    cache_data_dir: str = Field(
        default="~/.cache/llm_cache", 
        description="磁盘缓存目录"
    )

    # OpenDataLoader Docker 配置
    odl_container_name: str = Field(
        default="opendataloader-api-server",
        description="OpenDataLoader Docker 容器名称",
    )
    odl_volume_host_dir: str = Field(
        default="",
        description="挂载到容器 /data 的宿主机目录绝对路径（需与 docker run -v 一致）",
    )
    odl_hybrid_mode: str = Field(
        default="auto",
        description="Hybrid 模式：auto（自动分流，默认）或 full（全页 AI，最高精度但内存占用大）",
    )

    # prompt 模板
    prompt_translate: str = Field(
        default=DEFAULT_PROMPT_TRANSLATE,
        description="翻译 Prompt 模板，支持 {src_lang}, {target_lang}, {content} 占位符"
    )
    prompt_summary: str = Field(
        default=DEFAULT_PROMPT_SUMMARY,
        description="总结 Prompt 模板，支持 {src_lang}, {target_lang}, {content} 占位符"
    )

    @model_validator(mode='after')
    def resolve_defaults_and_paths(self) -> 'ProcessParams':
        # 从环境变量回退读取 LLM 连接信息
        if not self.llm_endpoint:
            self.llm_endpoint = os.environ.get("LLM_ENDPOINT", "")
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("LLM_API_KEY", "")
            
        # 路径展开
        if self.asset_save_dir:
            self.asset_save_dir = _expand_path(self.asset_save_dir)
        if self.cache_data_dir:
            self.cache_data_dir = _expand_path(self.cache_data_dir)
        if self.output_dir:
            self.output_dir = _expand_path(self.output_dir)
        if self.file_path:
            self.file_path = _expand_path(self.file_path)
        if self.odl_volume_host_dir:
            self.odl_volume_host_dir = _expand_path(self.odl_volume_host_dir)
            
        return self


class ProcessResult(BaseModel):
    """process() 的结构化返回值。"""

    output_file: str = Field(..., description="生成的 Markdown 文件路径")
    steps_completed: List[str] = Field(..., description="完成的处理步骤")
    elapsed_seconds: float = Field(..., description="总耗时(秒)")


class ExtractPagesParams(BaseModel):
    """run_extract_pages() 的参数配置模型。"""

    input_pdf: str = Field(..., description="输入 PDF 文件路径")
    pages: List[str] = Field(..., description="页码范围列表，例如 ['1-5', '10']")
    output_dir: Optional[str] = Field(None, description="输出目录，默认与输入文件相同")

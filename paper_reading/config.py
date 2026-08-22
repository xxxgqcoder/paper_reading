import os

from pydantic import BaseModel, Field, model_validator
from strenum import StrEnum


def _detect_device() -> str:
    """根据当前环境自动检测最优推理设备。

    优先级: MPS (Apple Silicon) > CUDA > auto
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "auto"


class Step(StrEnum):
    ORIGINAL = "original"


STEP_OUTPUT_ORDER: list[Step] = [Step.ORIGINAL]


class ContentType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"


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

    steps: list[str] = Field(
        default_factory=lambda: ["original"],
        description="处理步骤列表，目前仅支持 original",
    )

    # 路径
    asset_save_dir: str = Field(
        default="",
        description=(
            "解析后的图片等资源保存目录，为空时自动使用 PDF 同级的 {stem}_images 目录"
        ),
    )

    # MineRU pipeline 配置
    mineru_model_source: str = Field(
        default="modelscope",
        description="模型下载源：huggingface / modelscope / local",
    )
    mineru_device: str = Field(
        default="",
        description="运行设备：留空则根据环境自动检测（mps/cuda/cpu），也可显式指定",
    )

    @model_validator(mode="after")
    def resolve_defaults_and_paths(self) -> "ProcessParams":
        # 自动检测推理设备
        if not self.mineru_device:
            self.mineru_device = _detect_device()

        # 路径展开
        if self.asset_save_dir:
            self.asset_save_dir = _expand_path(self.asset_save_dir)
        if self.output_dir:
            self.output_dir = _expand_path(self.output_dir)
        if self.file_path:
            self.file_path = _expand_path(self.file_path)

        return self


class ProcessResult(BaseModel):
    """process() 的结构化返回值。"""

    output_file: str = Field(..., description="生成的 Markdown 文件路径")
    steps_completed: list[str] = Field(..., description="完成的处理步骤")
    elapsed_seconds: float = Field(..., description="总耗时(秒)")


class ExtractPagesParams(BaseModel):
    """run_extract_pages() 的参数配置模型。"""

    input_pdf: str = Field(..., description="输入 PDF 文件路径")
    pages: list[str] = Field(..., description="页码范围列表，例如 ['1-5', '10']")
    output_dir: str | None = Field(None, description="输出目录，默认与输入文件相同")
    page_number_offset: int = Field(
        0,
        description="印刷页码到 PDF 物理页码的偏移量；PDF 页码 = 印刷页码 + offset",
    )


class PdfAction(StrEnum):
    """统一 PDF 阅读入口支持的动作。"""

    PARSE = "parse"
    INSPECT = "inspect"
    EXTRACT = "extract"


class PdfReadingParams(ProcessParams):
    """统一 PDF 阅读入口参数。

    ``parse`` 复用 ProcessParams；``inspect`` 读取前置页、目录候选和页码
    证据；``extract`` 按印刷页码及偏移量提取页面。
    """

    action: PdfAction = Field(default=PdfAction.PARSE, description="执行动作")
    output_dir: str = Field(
        default="",
        description="parse/extract 输出目录；parse 时必填",
    )
    pages: list[str] = Field(
        default_factory=list,
        description="extract 的页码范围，如 ['1-5', '10']",
    )
    page_number_offset: int = Field(
        0,
        description="印刷页码到 PDF 物理页码的偏移量；PDF 页码 = 印刷页码 + offset",
    )
    inspect_pages: int = Field(
        20,
        ge=1,
        description="inspect 时读取的 PDF 前置页数",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "PdfReadingParams":
        if self.action == PdfAction.PARSE and not self.output_dir:
            raise ValueError("output_dir is required when action='parse'")
        if self.action == PdfAction.EXTRACT and not self.pages:
            raise ValueError("pages is required when action='extract'")
        return self


class PdfReadingResult(BaseModel):
    """统一 PDF 阅读入口返回值。"""

    action: PdfAction
    output_file: str | None = None
    output_files: list[str] = Field(default_factory=list)
    steps_completed: list[str] = Field(default_factory=list)
    inspection: dict | None = None
    elapsed_seconds: float

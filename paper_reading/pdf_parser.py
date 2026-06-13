"""MineRU 3.x PDF 解析器。

使用 MineRU pipeline 后端解析 PDF 文件，输出 Content 列表。
配置文件 (mineru.json) 随 package 分发，不依赖用户 home 目录。
"""

import os
import shutil
import uuid
from pathlib import Path

from .config import Content, ContentType
from .log import Logger
from .utils import cache_it, hash64, time_it

# MineRU 配置文件路径（随 package 分发）
_MINERU_CONFIG_PATH = Path(__file__).parent / "resources" / "mineru.json"


def _setup_mineru_env(model_source: str = "huggingface", device: str = "auto") -> None:
    """在首次导入 MineRU 模块前配置运行时环境变量。

    - MINERU_TOOLS_CONFIG_JSON: 指向随 package 分发的 mineru.json，不写 ~/mineru.json
    - MINERU_MODEL_SOURCE: 模型来源 (huggingface/modelscope/local)
    - MINERU_DEVICE_MODE: 运行设备，auto 时不设置让 MineRU 自动检测
    """
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(_MINERU_CONFIG_PATH.resolve())
    if "MINERU_MODEL_SOURCE" not in os.environ:
        os.environ["MINERU_MODEL_SOURCE"] = model_source
    if device != "auto" and "MINERU_DEVICE_MODE" not in os.environ:
        os.environ["MINERU_DEVICE_MODE"] = device


class MineRUParser:
    """PDF 解析器，使用 MineRU 3.x pipeline 后端。

    无需 Docker，直接调用 MineRU Python API 在本地解析 PDF。
    配置文件随 package 分发，模型可从 HuggingFace 或 ModelScope 下载。
    """

    def __init__(
        self,
        model_source: str = "huggingface",
        device: str = "auto",
    ):
        self.model_source = model_source
        self.device = device

    def key_generator(self, file_path: str, **kwargs) -> str:
        file_bytes = b""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            Logger.error(f"Read file exception: {e}")
            return uuid.uuid4().hex
        return "mineru_parser::file_content_hash::" + hash64(file_bytes)

    @time_it("mineru parser")
    @cache_it(key_generator=key_generator)
    def _parse_cached(self, file_path: str) -> list[Content]:
        """运行 MineRU 解析 PDF，返回 Content 列表（按 PDF 内容缓存）。

        图片始终写入 PDF 同级的 ``<stem>_images`` 目录——该目录按内容寻址、
        稳定可复用，充当图片的持久源；content_url 记录其中图片的绝对路径。

        本方法只负责「纯解析」，不接受 asset_save_dir。把图片落地到指定资源
        目录是带副作用的操作，由 parse() 负责并保证每次都执行，否则缓存命中时
        副作用会被一并跳过，导致「命中缓存即丢图」。
        """
        _setup_mineru_env(self.model_source, self.device)

        from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode

        pdf_path = Path(file_path).resolve()
        pdf_bytes = pdf_path.read_bytes()

        image_dir = str(pdf_path.parent / (pdf_path.stem + "_images"))
        os.makedirs(image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(image_dir)

        _results: dict[int, dict] = {}

        def _on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
            _results[doc_index] = middle_json

        Logger.info(f"Starting MineRU pipeline parse: {pdf_path.name}")
        doc_analyze_streaming(
            pdf_bytes_list=[pdf_bytes],
            image_writer_list=[image_writer],
            lang_list=[None],
            on_doc_ready=_on_doc_ready,
        )

        middle_json = _results.get(0)
        if not middle_json:
            Logger.error("MineRU pipeline returned no results")
            return []

        content_list_raw = union_make(
            middle_json["pdf_info"],
            MakeMode.CONTENT_LIST,
            img_buket_path=image_dir,
        )

        contents = self._convert_to_contents(content_list_raw, file_path)
        Logger.info(f"Parsed {len(contents)} content items from {pdf_path.name}")
        return contents

    def parse(self, file_path: str, asset_save_dir: str = "") -> list[Content]:
        """解析 PDF 文件，返回 Content 列表。

        MineRU 解析（重计算）按 PDF 内容缓存；把图片落地到 asset_save_dir 的
        副作用每次都会执行，因此即便解析命中缓存，图片仍会同步到目标目录，
        steps.py 生成的 Obsidian wikilink 不会指向缺失文件。
        """
        contents = self._parse_cached(file_path)
        if asset_save_dir:
            self._materialize_assets(contents, asset_save_dir)
        return contents

    @staticmethod
    def _materialize_assets(contents: list[Content], asset_save_dir: str) -> None:
        """把 contents 引用的图片复制到 asset_save_dir，并改写 content_url 指向它。

        幂等：目标已存在则跳过复制。该操作与解析缓存解耦、每次调用都执行，
        确保图片落到资源目录（如 Obsidian attachments），wikilink 才能渲染。
        源文件缺失（例如 PDF 同级 ``*_images`` 目录被删）时记录警告并跳过。
        """
        dst = Path(asset_save_dir).resolve()
        dst.mkdir(parents=True, exist_ok=True)
        for content in contents:
            src = content.content_url
            if not src:
                continue
            target = dst / os.path.basename(src)
            if target.exists():
                content.content_url = str(target)
                continue
            if os.path.isfile(src):
                shutil.copy2(src, target)
                content.content_url = str(target)
            else:
                Logger.warning(
                    f"Asset source missing, cannot materialize: {src}. "
                    "若需重新生成图片，请删除该 PDF 同级的 *_images 目录"
                    "或清理解析缓存后重试。"
                )

    def _convert_to_contents(
        self,
        content_list_raw: list[dict],
        file_path: str,
    ) -> list[Content]:
        """将 MineRU CONTENT_LIST 格式转换为 paper_reading Content 列表。"""
        contents: list[Content] = []
        for item in content_list_raw:
            item_type = item.get("type", "")
            converted = self._convert_item(item, item_type, file_path)
            if converted is not None:
                contents.append(converted)
        return contents

    def _convert_item(self, item: dict, item_type: str, file_path: str) -> Content | None:  # noqa: E501
        """将单个 MineRU content list item 转换为 Content 对象。"""
        if item_type == "text":
            text = item.get("text", "").strip()
            if not text:
                return None
            level = item.get("text_level")
            if level:
                text = "#" * level + " " + text
            return Content(
                content_type=ContentType.TEXT,
                file_path=file_path,
                content=text,
                extra_description="",
                content_url="",
            )

        elif item_type == "list":
            items = item.get("list_items", [])
            text = "\n".join(f"- {t}" for t in items if t and t.strip())
            if not text:
                return None
            return Content(
                content_type=ContentType.TEXT,
                file_path=file_path,
                content=text,
                extra_description="",
                content_url="",
            )

        elif item_type == "equation":
            latex = item.get("text", "").strip()
            # MineRU 的 text 字段有时已包含 $$ 分隔符，需去除避免 steps.py 双重包裹
            if latex.startswith("$$"):
                latex = latex[2:].lstrip()
            if latex.endswith("$$"):
                latex = latex[:-2].rstrip()
            img_path = item.get("img_path", "")
            img_exists = bool(img_path) and os.path.isfile(img_path)
            if latex and item.get("text_format") == "latex":
                return Content(
                    content_type=ContentType.FORMULA,
                    file_path=file_path,
                    content=latex,
                    extra_description="",
                    content_url=os.path.abspath(img_path) if img_exists else "",
                )
            elif img_exists:
                return Content(
                    content_type=ContentType.IMAGE,
                    file_path=file_path,
                    content="",
                    extra_description="",
                    content_url=os.path.abspath(img_path),
                )
            return None

        elif item_type == "image":
            img_path = item.get("img_path", "")
            if not img_path or not os.path.isfile(img_path):
                return None
            captions = item.get("image_caption", [])
            description = " ".join(c for c in captions if c).strip()
            return Content(
                content_type=ContentType.IMAGE,
                file_path=file_path,
                content="",
                extra_description=description,
                content_url=os.path.abspath(img_path),
            )

        elif item_type == "table":
            img_path = item.get("img_path", "")
            table_html = item.get("table_body", "")
            captions = item.get("table_caption", [])
            description = " ".join(c for c in captions if c).strip()
            img_exists = bool(img_path) and os.path.isfile(img_path)
            return Content(
                content_type=ContentType.TABLE,
                file_path=file_path,
                content=table_html,
                extra_description=description,
                content_url=os.path.abspath(img_path) if img_exists else "",
            )

        return None

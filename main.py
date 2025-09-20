from typing import Any

from mineru.utils.draw_bbox import f
from pydantic import BaseModel, Field
from strenum import StrEnum
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import random
import pickle

import xxhash
import tempfile
import shutil
import base64
import asyncio
import base64
import functools
import inspect
import logging
import os
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

# -----------------------------------------------------------------------------------------------
# data


def get_project_base_directory() -> str:
    project_base = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    return project_base


class _Config(BaseSettings):
    """Centralized configuration class for the entire Tiny RAG project."""

    parser_config_file_path: str = Field("", description="path to parser config file")
    asset_save_dir: str = Field(
        "",
        description="directory to save parsed asset files, including images and tables",
    )
    cache_data_dir: str = Field(default="./tmp", description="cache data directory")

    model_config = SettingsConfigDict(
        yaml_file=os.path.join(get_project_base_directory(), "config.yaml"),
        env_prefix="TINY_RAG@@",
        env_nested_delimiter="@@",
        nested_model_default_partial_update=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


Config = _Config()  # type: ignore


class ContentType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


class Content(BaseModel):
    """
    Document content object.
    """

    content_type: ContentType = Field(ContentType.TEXT, description="content type")
    file_path: str = Field("", description="original file path")
    content: str = Field(
        "",
        description="the content, represented in string. If content type is not image / table, this field will be base64 encoded image content.",
    )
    extra_description: str = Field("", description="content extra description")
    content_url: str = Field(
        "",
        description="url to the content, set when content is not suitable for directly insert into db, for example image / audio data",
    )


# -----------------------------------------------------------------------------------------------
# util func


_loggers = {}


def get_logger(
    log_module_name: str = "",
    log_format: str = "%(asctime)-15s %(levelname)-4s %(filename)s:%(lineno)d: %(message)s",
    need_stream: bool = True,
):
    """
    Get logger for a specific module.

    Args:
    - log_module_name: Name of the module for which to get the logger. If empty.

    Returns:
    - Logger instance.
    """
    logger_key = f"{log_module_name}_{need_stream}"
    if not logger_key:
        logger_key = "default"
    if logger_key in _loggers:
        return _loggers[logger_key]

    project_root_dir = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    logger = logging.getLogger(name=logger_key)
    logger.handlers.clear()
    log_path = os.path.abspath(
        os.path.join(project_root_dir, "logs", f"{log_module_name}.log")
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter(log_format)

    handler1 = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=1)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    if need_stream:
        handler2 = logging.StreamHandler()
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)
    else:
        logger.propagate = False

    logger.setLevel(level=logging.INFO)
    logging.captureWarnings(True)

    _loggers[logger_key] = logger
    return logger


Logger = get_logger()


def time_it(prefix: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*kargs, **kwargs):
            begin = time.time_ns()
            ret = func(*kargs, **kwargs)
            elapse = (time.time_ns() - begin) // 1000000

            func_name = f"{prefix} {func.__name__}" if prefix else func.__name__
            Logger.info(
                f"{func_name} took {elapse // 60000}min {(elapse % 60000) // 1000}sec {elapse % 60000 % 1000}ms to finish"
            )

            return ret

        return wrapper

    return decorator


def hash64(content: bytes) -> str:
    return xxhash.xxh64(content).hexdigest()


def safe_encode(text: str) -> str:
    try:
        return text.encode(encoding="utf-8", errors="ignore").decode(
            encoding="utf-8", errors="ignore"
        )
    except Exception as e:
        Logger.error(f"safe encode error: {e}")
        return ""


def load_base64_image(p: str) -> str:
    """load image as base64 encoded string"""
    with open(p, "rb") as f:
        image_bytes = f.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")

    return base64_string


def safe_strip(d: Any) -> str:
    """
    Safely strip d.
    """
    if d is None:
        return ""
    if isinstance(d, str):
        return d.strip()
    return str(d).strip()


T = TypeVar("T")


def cache_it(key_generator: Callable[..., str]) -> Callable[..., Callable[..., T]]:
    """
    Redis cache decorator with customized key generator.

    Args:
    - key_generator: Function that takes the same args as decorated function and returns cache key
    - key_ttl_seconds: Time to live for the cache key in seconds (default: 24 hours)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_generator(*args, **kwargs)
            cache_path = os.path.join(Config.cache_data_dir, f"{cache_key}.pickle")
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                Logger.error(f"Load cache exception: {e}")

            result = func(*args, **kwargs)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                Logger.error(f"Save cache exception: {e}")

            return result

        return wrapper

    return decorator


# -----------------------------------------------------------------------------------------------


class PDFParser:
    """
    PDF parser implementation, backed by [MinerU](https://github.com/opendatalab/MinerU).
    """

    def __init__(self):
        super().__init__()

        with open(file=Config.parser_config_file_path) as f:  # type: ignore
            conf = json.load(f)

        Logger.info(f"Parsr config: {json.dumps(conf, indent=4)}")

        # set environment variable for magic_pdf to load config json file
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = Config.parser_config_file_path
        os.environ["MINERU_MODEL_SOURCE"] = conf.get("mineru_model_source", "local")

    def key_generator(self, file_path) -> str:
        file_bytes = b""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except:
            return random.random().hex()

        return "parser::file_content_hash::" + hash64(file_bytes)

    @time_it("pdf parser")
    @cache_it(key_generator=key_generator)
    def parse(self, file_path: str) -> list[Content]:
        asset_save_dir = Config.asset_save_dir
        os.makedirs(asset_save_dir, exist_ok=True)
        self.file_path = file_path

        # get original chunk list
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        temp_asset_dir = temp_dir.name
        Logger.info(f"temp asset directory: {temp_asset_dir}")

        contents = self.parse_pdf_content(
            file_path=file_path,
            temp_asset_dir=temp_asset_dir,
            asset_save_dir=asset_save_dir,
        )
        Logger.info(f"Original content block num: {len(contents)}")

        temp_dir.cleanup()
        return contents

    def parse_pdf_content(
        self, file_path: str, temp_asset_dir: str, asset_save_dir: str
    ) -> list[Content]:
        """
        Parse PDF content and return content list.
        The result is a list of json oject representing a pdf content block.

        Dict object key explanation:
            - `img_caption`: the image caption.
            - `img_footnote`:
            - `img_path`: path to parsed image.
            - `page_idx`: page index.
            - `table_body`: table content in html format.
            - `table_caption`: table caption.
            - `table_footnote`:
            - `text`: the block text content.
            - `text_format`: used in latex forumla block.
            - `text_level`: used in headline block.
            - `type`: block type, can be one of 'equation', 'image', 'table', 'text'.

        Typical parsed paper content is organized as list of content block.
        Headlines will stored in one separated block, with `text_level` = 1 while regular content block's `text_level` key is missing.
        Headline blocks are followed by regular content block, including `text`, `equation`, `table` and `image` (distinguished by key `type`).
        All captions are stored in each block's caption key, for example, caption of a parsed image is saved in `img_caption` key of the block.

        See https://github.com/opendatalab/MinerU/blob/master/demo/demo.py for more details.

        Returns:
        - A list of parsed chunk.
        """
        # NOTE: magic_pdf package uses singleton design and the model isntance is initialized when the module is imported,
        # so postpone the import statement until parse method is called.

        import copy
        from pathlib import Path

        from mineru.backend.pipeline.model_json_to_middle_json import (
            result_to_middle_json as pipeline_result_to_middle_json,
        )
        from mineru.backend.pipeline.pipeline_analyze import (
            doc_analyze as pipeline_doc_analyze,
        )
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
            union_make as pipeline_union_make,
        )
        from mineru.cli.common import (
            convert_pdf_bytes_to_bytes_by_pypdfium2,
            prepare_env,
            read_fn,
        )
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
        from mineru.utils.enum_class import MakeMode

        # prepare env
        try:
            shutil.rmtree(temp_asset_dir)
        except Exception as e:
            Logger.error(f"Remove temp asset dir exception: {e}")
            pass
        os.makedirs(temp_asset_dir, exist_ok=True)

        lang = "ch"
        start_page_id = 0
        end_page_id = None
        parse_method = "auto"

        file_name = str(Path(file_path).stem)
        pdf_bytes = read_fn(file_path)

        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
            pdf_bytes, start_page_id, end_page_id
        )

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                [new_pdf_bytes],
                [lang],
                parse_method=parse_method,
                formula_enable=True,
                table_enable=True,
            )
        )

        model_list = infer_results[0]
        model_json = copy.deepcopy(model_list)
        local_image_dir, local_md_dir = prepare_env(
            temp_asset_dir, file_name, parse_method
        )
        image_writer, md_writer = (
            FileBasedDataWriter(local_image_dir),
            FileBasedDataWriter(local_md_dir),
        )

        middle_json = pipeline_result_to_middle_json(
            model_list,
            all_image_lists[0],
            all_pdf_docs[0],
            image_writer,
            lang_list[0],
            ocr_enabled_list[0],
            True,
        )

        pdf_info = middle_json["pdf_info"]

        # draw span and layout
        draw_layout_bbox(
            pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_layout.pdf"
        )
        draw_span_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_span.pdf")
        md_writer.write(f"{file_name}_origin.pdf", new_pdf_bytes)

        # dump md
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str: list[str] = pipeline_union_make(
            pdf_info, MakeMode.MM_MD, image_dir
        )  # type: ignore
        md_writer.write_string(f"{file_name}.md", str(md_content_str))

        # dump content list
        image_dir = str(os.path.basename(local_image_dir))
        content_list: list[dict[str, Any]] = pipeline_union_make(
            pdf_info, MakeMode.CONTENT_LIST, image_dir
        )  # type: ignore
        md_writer.write_string(
            f"{file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

        # dump middle json
        md_writer.write_string(
            f"{file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

        # dump model json
        md_writer.write_string(
            f"{file_name}_model.json",
            json.dumps(model_json, ensure_ascii=False, indent=4),
        )

        # parse content list
        def _save_image(src_path: str, dst_dir: str) -> None:
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)

        def _is_valid_content(content: dict[str, Any]) -> bool:
            """
            There are corner cases where returned blocks dont contain expected keys or values are empty.

            Returns:
            - bool: true if block is valid.
            """
            # missing key
            if "type" not in content:
                return False
            # text / equation
            if content["type"] in ["text", "equation"]:
                return "text" in content
            # image
            if content["type"] == "image":
                return "img_path" in content and len(content["img_path"]) > 0
            # table
            if content["type"] == "table":
                return "table_body" in content
            return True

        def _format_caption(caption: Any) -> str:
            """
            Format caption as text.
            """
            if isinstance(caption, list):
                ret = "\n".join([str(e) for e in caption])
                return ret
            return str(caption)

        contents = []
        for content in content_list:
            if not _is_valid_content(content):
                Logger.info(f"Invalid content: {json.dumps(content, indent=4)}")
                continue

            # text / formula
            if content["type"] in ["text", "equation"]:
                text = self.strip_text_content([content["text"]])
                if content.get("text_level", 0) == 1:
                    text = "# " + text  # headline level 1
                contents.append(
                    Content(
                        content_type=ContentType.TEXT,
                        file_path=self.file_path,
                        content=safe_encode(text),
                        extra_description="",
                        content_url="",
                    )
                )

            # image
            elif content["type"] in ["image"]:
                texts = [
                    _format_caption(content.get("img_caption", "")),
                    _format_caption(content.get("img_footnote", "")),
                ]
                extra_description = self.strip_text_content(texts)
                if len(extra_description) == 0:
                    extra_description = "no caption for this image"

                # NOTE: hard coded image path format
                abs_img_path = os.path.join(
                    temp_asset_dir,
                    str(Path(self.file_path).stem),
                    "auto",
                    content["img_path"],
                )
                _save_image(abs_img_path, asset_save_dir)

                contents.append(
                    Content(
                        content_type=ContentType.IMAGE,
                        file_path=self.file_path,
                        content=load_base64_image(abs_img_path),
                        extra_description=safe_encode(extra_description),
                        content_url=os.path.join(
                            asset_save_dir, os.path.basename(abs_img_path)
                        ),
                    )
                )

            # table
            elif content["type"] in ["table"]:
                texts = [
                    _format_caption(content.get("table_caption", "")),
                    _format_caption(content.get("table_footnote", "")),
                ]
                extra_description = self.strip_text_content(texts)
                if len(extra_description) == 0:
                    extra_description = "no caption for this table"

                table_body = content.get("table_body", "")
                # NOTE: ignore table body
                # extra_description += "\n\n\n\nTable content:\n" + table_body

                abs_img_path = os.path.join(
                    temp_asset_dir,
                    str(Path(self.file_path).stem),
                    "auto",
                    content["img_path"],
                )
                if content["img_path"]:
                    _save_image(abs_img_path, asset_save_dir)

                contents.append(
                    Content(
                        content_type=ContentType.TABLE,
                        file_path=self.file_path,
                        content=load_base64_image(abs_img_path),
                        extra_description=extra_description,
                        content_url=os.path.join(
                            asset_save_dir, os.path.basename(abs_img_path)
                        )
                        if content["img_path"]
                        else "",
                    )
                )
            else:
                pass

        return contents

    def strip_text_content(self, texts: list[str]) -> str:
        """
        Filter and merge text content
        """
        content = ""
        for text in texts:
            striped = safe_strip(text)
            if len(striped) == 0 or striped == "[]":
                continue
            content += striped
            content += "\n\n"
        return content.strip()

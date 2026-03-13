"""process_pdf.py - PDF paper parsing, translation, and summarization tool.

Usage:
    python process_pdf.py --file_path <pdf> --final_md_file_save_dir <dir> [options]

Required:
    --file_path                Path to the PDF file
    --final_md_file_save_dir   Output directory for the Markdown file

Optional:
    --steps              Processing steps (default: summary,translate,original)
    --src_lang           Source language en/zh (default: en)
    --target_lang        Target language en/zh (default: zh)
    --llm_endpoint       LLM API endpoint URL (default: http://127.0.0.1:11434)
    --llm_api_key        LLM API key (empty = Ollama, non-empty = OpenAI-compatible)
    --chat_model         Chat model name
    --vision_model       Vision model name
    --temp_content_dir   Temp content folder (default: ./tmp/parsed_asset)

Dependencies:
    pip install tiktoken xxhash diskcache ollama pydantic
    pip install pydantic-settings strenum mineru
    pip install openai  # only needed for --llm_backend=openai

Output:
    Generates <filename>.md in final_md_file_save_dir.
    Prints a JSON result summary to stdout on completion.
"""

import argparse
import base64
import copy
import functools
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from typing import Any, TypeVar

import tiktoken
import xxhash
from diskcache import Cache
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from strenum import StrEnum

# ----------------------------------------------------------------------------
# data


def get_project_base_directory() -> str:
    project_base = os.path.abspath(os.path.dirname(__file__))
    return project_base


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


class _Config(BaseSettings):
    """Application configuration settings."""

    parser_config_file_path: str = Field("", description="path to parser config file")
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
    llm_endpoint: str = Field(
        default="http://127.0.0.1:11434",
        description="LLM API endpoint URL",
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API key (empty = Ollama native, non-empty = OpenAI-compatible)",
    )
    chat_model_name: str = Field(default="llama3", description="chat model name")
    vision_model_name: str = Field(default="llama3", description="vision model name")

    model_config = SettingsConfigDict(
        yaml_file=os.path.join(get_project_base_directory(), "config.yaml"),
        env_prefix="CONFIG@@",
        env_nested_delimiter="@@",
        nested_model_default_partial_update=True,
    )

    @field_validator(
        "cache_data_dir", "parser_config_file_path", "asset_save_dir", mode="after"
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


class ContentType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TABLE = "table"


class Content(BaseModel):
    """
    Document content object.
    """

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


# ----------------------------------------------------------------------------
# util func


_loggers: dict[str, logging.Logger] = {}


def get_logger(
    log_module_name: str = "",
    log_format: str = (
        "%(asctime)-15s %(levelname)-4s %(filename)s:%(lineno)d: %(message)s"
    ),
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


def estimate_token_num(text: str) -> tuple[int, list[str]]:
    """
    Estimate tokens in text using tiktoken.

    Args:
    - text: the string to parse.

    Return:
    - int, estimated token num.
    - list of string, estimated tokens.
    """
    if text is None or len(text.strip()) == 0:
        return 0, []

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    token_strings = [
        encoding.decode_single_token_bytes(token).decode("utf-8", "replace")
        for token in tokens
    ]

    return len(tokens), token_strings


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


def safe_strip(raw: str) -> str:
    if raw is None or len(raw) == 0:
        return ""
    raw = str(raw)
    return raw.strip()


def is_empty(text: str) -> bool:
    text = safe_strip(text)
    if text is None:
        return True
    if len(text) == 0:
        return True
    if text == "[]":
        return True
    return False


def relative_md_image_path(sys_image_folder: str, img_name: str) -> str:
    """
    md image can only be corrct displayed when saved to same folder of the md file,
    reformatted inserted image path for better display.
    """
    return (
        f"![[{os.path.join(os.path.basename(sys_image_folder), img_name)}]]"
        + line_breaker
    )


def ensure_utf(text: str) -> str:
    if text is None:
        return text
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text


T = TypeVar("T")


def cache_it(
    key_generator: Callable[..., str], ttl_seconds=60 * 60 * 24 * 100
) -> Callable[..., Callable[..., T]]:
    """
    File-based cache decorator using diskcache.

    This decorator saves the results of a function call to a disk-based cache.
    On subsequent calls with the same arguments, it returns the cached result
    if it hasn't expired.

    Args:
    - key_generator: A function that takes the same arguments as the decorated
      function and returns a unique cache key string.
    - ttl_seconds: Time to live for the cache key in seconds (default: 100 days).
    """
    cache = Cache(Config.cache_data_dir)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_generator(*args, **kwargs)

            # Check if the result is in the cache
            if cache_key in cache:
                Logger.info(f"Loaded cache from {cache_key}")
                cached_result = cache[cache_key]
                return cached_result  # type: ignore

            # If not, call the function and store the result
            result = func(*args, **kwargs)
            if result:
                cache.set(cache_key, result, expire=ttl_seconds)
                Logger.info(f"Saved cache to {cache_key}")

            return result

        return wrapper

    return decorator


# ----------------------------------------------------------------------------
# LLM api


class LLMBackend(ABC):
    """Abstract base class for LLM API backends."""

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        gen_conf: dict[str, Any],
    ) -> str | None: ...

    @abstractmethod
    def vision_chat(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        gen_conf: dict[str, Any],
    ) -> str | None: ...


class OllamaBackend(LLMBackend):
    _OPTION_KEY_MAP = {
        "temperature": "temperature",
        "max_tokens": "num_predict",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repeat_penalty": "repeat_penalty",
        "num_ctx": "num_ctx",
    }

    def __init__(self, host: str, timeout: int = 15 * 60):
        from ollama import Client as OllamaClient

        self._client = OllamaClient(host=host, timeout=timeout)

    def _build_options(self, gen_conf: dict[str, Any]) -> dict[str, Any]:
        return {
            v: gen_conf[k] for k, v in self._OPTION_KEY_MAP.items() if k in gen_conf
        }

    def chat(self, model, messages, gen_conf):
        resp = self._client.chat(
            model=model,
            messages=messages,
            options=self._build_options(gen_conf),
            keep_alive=10,
        )
        return resp["message"]["content"] if resp else None

    def vision_chat(self, model, prompt, image_b64, gen_conf):
        resp = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
            options=self._build_options(gen_conf),
        )
        return resp["message"]["content"] if resp else None


class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, base_url: str = "", timeout: int = 15 * 60):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for OpenAI backend. "
                "Install with: pip install openai"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )

    @staticmethod
    def _build_params(gen_conf: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if "temperature" in gen_conf:
            params["temperature"] = gen_conf["temperature"]
        if "top_p" in gen_conf:
            params["top_p"] = gen_conf["top_p"]
        if "max_tokens" in gen_conf:
            params["max_tokens"] = gen_conf["max_tokens"]
        if "num_ctx" in gen_conf:
            params.setdefault("max_tokens", gen_conf["num_ctx"])
        if "presence_penalty" in gen_conf:
            params["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf:
            params["frequency_penalty"] = gen_conf["frequency_penalty"]
        return params

    def chat(self, model, messages, gen_conf):
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **self._build_params(gen_conf),
        )
        if resp.choices:
            return resp.choices[0].message.content
        return None

    def vision_chat(self, model, prompt, image_b64, gen_conf):
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **self._build_params(gen_conf),
        )
        if resp.choices:
            return resp.choices[0].message.content
        return None


_llm_backend: LLMBackend | None = None


def _get_llm_backend() -> LLMBackend:
    global _llm_backend
    if _llm_backend is None:
        if Config.llm_api_key:
            _llm_backend = OpenAIBackend(
                api_key=Config.llm_api_key,
                base_url=Config.llm_endpoint,
            )
            Logger.info(f"Using OpenAI-compatible backend: {Config.llm_endpoint}")
        else:
            _llm_backend = OllamaBackend(host=Config.llm_endpoint)
            Logger.info(f"Using Ollama backend: {Config.llm_endpoint}")
    return _llm_backend


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>/<thinking> wrapper tags from LLM output."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "</thinking>" in text:
        text = text.split("</thinking>")[-1]
    return text.strip()


@time_it(prefix="llm chat")
@cache_it(
    key_generator=lambda prompt, gen_conf: "llm_chat::prompt_hash::"
    + hash64(f"{prompt}_{json.dumps(gen_conf, default=str)}".encode())
)
def llm_chat(prompt: str, gen_conf: dict[str, Any]) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    try:
        ans = _get_llm_backend().chat(
            model=Config.chat_model_name,
            messages=messages,
            gen_conf=gen_conf,
        )
    except Exception as e:
        Logger.error(f"LLM chat exception: {e}")
        return None
    if not ans:
        return None
    return _strip_thinking_tags(ans)


@time_it(prefix="image chat")
@cache_it(
    key_generator=lambda prompt, image_content, gen_conf: (
        "image_chat::prompt_hash"
        + f"{hash64((prompt + image_content + json.dumps(gen_conf, default=str)).encode('utf-8', errors='ignore'))}"
    )
)
def image_chat(
    prompt: str,
    image_content: str,
    gen_conf: dict[str, Any],
) -> str | None:
    try:
        return _get_llm_backend().vision_chat(
            model=Config.vision_model_name,
            prompt=prompt,
            image_b64=image_content,
            gen_conf=gen_conf,
        )
    except Exception as e:
        Logger.error(f"Vision model chat exception: {e}")
        return None


# ----------------------------------------------------------------------------
# MinerU model management


def _find_mineru_model_dir() -> str | None:
    """Scan HuggingFace cache for an existing MinerU PDF-Extract-Kit snapshot."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if "PDF-Extract-Kit" in repo.repo_id:
                for revision in sorted(
                    repo.revisions, key=lambda r: r.last_modified, reverse=True
                ):
                    return str(revision.snapshot_path)
    except Exception as e:
        Logger.warning(f"Failed to scan HuggingFace cache: {e}")
    return None


def _download_mineru_models() -> str:
    """Download all MinerU model weights. Returns the local snapshot directory."""
    from mineru.utils.enum_class import ModelPath

    model_source = os.getenv("MINERU_MODEL_SOURCE", "huggingface")
    Logger.info(f"Downloading MinerU models from source: {model_source}")

    repo_mapping = {
        "huggingface": ModelPath.pipeline_root_hf,
        "modelscope": ModelPath.pipeline_root_modelscope,
    }
    repo_id = repo_mapping.get(model_source)
    if repo_id is None:
        raise ValueError(
            f"Unknown model source: {model_source},"
            " expected 'huggingface' or 'modelscope'"
        )

    if model_source == "huggingface":
        from huggingface_hub import snapshot_download
    else:
        from modelscope import snapshot_download  # type: ignore[no-redef]

    model_paths = [
        ModelPath.doclayout_yolo,
        ModelPath.yolo_v8_mfd,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.layout_reader,
        ModelPath.slanet_plus,
        ModelPath.unet_structure,
        ModelPath.paddle_table_cls,
        ModelPath.paddle_orientation_classification,
    ]
    downloaded_dir = ""
    for model_path in model_paths:
        relative_path = model_path.strip("/")
        Logger.info(f"Downloading model component: {relative_path}")
        downloaded_dir = snapshot_download(
            repo_id,
            allow_patterns=[relative_path, relative_path + "/*"],
        )

    Logger.info(f"All models downloaded to: {downloaded_dir}")
    return downloaded_dir


def ensure_mineru_model() -> str:
    """
    Ensure MinerU model weights are available locally.
    Downloads them automatically if not found.

    Override with environment variable MINERU_MODEL_DIR to skip auto-detection.
    """
    env_model_dir = os.environ.get("MINERU_MODEL_DIR")
    if env_model_dir:
        expanded = os.path.abspath(os.path.expanduser(env_model_dir))
        if os.path.isdir(expanded):
            Logger.info(f"Using model dir from MINERU_MODEL_DIR: {expanded}")
            return expanded
        Logger.warning(
            f"MINERU_MODEL_DIR={env_model_dir} does not exist,"
            " falling back to auto-detection"
        )

    model_dir = _find_mineru_model_dir()
    if model_dir:
        Logger.info(f"Found existing MinerU model: {model_dir}")
        return model_dir

    Logger.info(
        "MinerU models not found locally."
        " Downloading (one-time operation, may take a while)..."
    )
    return _download_mineru_models()


def prepare_mineru_runtime_config(model_dir: str) -> str:
    """
    Read the template config (magic-pdf.json), fill in model directory and
    self-reference path, expand all paths, then write to a temp file.

    Returns the path to the generated runtime config file.
    """
    with open(Config.parser_config_file_path) as f:
        conf = json.load(f)

    runtime_config_path = os.path.join(
        tempfile.gettempdir(), "mineru_runtime_config.json"
    )
    conf["models-dir"] = {"pipeline": model_dir}
    conf["mineru_tools_conf_json"] = runtime_config_path

    def expand_value(value: Any) -> Any:
        if isinstance(value, str) and "~" in value:
            return os.path.abspath(os.path.expanduser(value))
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        return value

    conf = expand_value(conf)

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=4)

    Logger.info(f"Runtime MinerU config written to: {runtime_config_path}")
    return runtime_config_path


# ----------------------------------------------------------------------------
# parser


class PDFParser:
    """
    PDF parser implementation, backed by [MinerU](https://github.com/opendatalab/MinerU).
    """

    def __init__(self, runtime_config_path: str):
        super().__init__()

        with open(runtime_config_path) as f:
            conf = json.load(f)

        Logger.info(f"Parser config: {json.dumps(conf, indent=4)}")

        os.environ["MINERU_TOOLS_CONFIG_JSON"] = runtime_config_path
        os.environ["MINERU_MODEL_SOURCE"] = conf.get("mineru_model_source", "local")

    def key_generator(self, file_path) -> str:
        file_bytes = b""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            Logger.error(f"Read file exception: {e}")
            return uuid.uuid4().hex

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
                    extra_description = ""

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
                    extra_description = ""

                # NOTE: ignore table body
                # table_body = content.get("table_body", "")
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


# ----------------------------------------------------------------------------
# step functions
line_breaker = "\n\n"


PROMPT_TRANSLATE = """
你是一个翻译助手，请将下面的{src_lang}内容翻译成{target_lang}。

下面是需要翻译的内容：

{content}
----

注意：
- 如果要翻译的内容为引用文献、算法伪代码、代码、人名，则不需要翻译，直接返回原文
- 你只需要输出最终翻译结果，不要输出逐步思考过程。

现在开始逐步思考
"""

PROMPT_SUMMARY = """
你是一个阅读助手，阅读下面的{src_lang}内容，并完成指令。

下面是文档内容

{content}
----

指令：使用{target_lang}语言，总结{src_lang}内容。
- 如果是学术论文，请总结论文的主要贡献、方法、实验和结论。
- 如果是技术文档，请总结技术的核心概念、功能和应用场景。
- 如果是一般性文档（书籍，文章等），请总结文档的主要内容和关键点。

注意：
- 忽略引用部分内容，只总结文档正文部分。
- 你只需要输出最终总结结果，不要输出逐步思考过程。

现在开始逐步思考
"""


# ------------------------------------------------------------------------------
# job executor
job_executor: ProcessPoolExecutor | None = None


def get_job_executor() -> ProcessPoolExecutor:
    global job_executor
    if job_executor is None:
        job_executor = ProcessPoolExecutor(max_workers=1)
    return job_executor


def parse_pdf_job(
    file_path: str, temp_content_dir: str, runtime_config_path: str
) -> None:
    Logger.info(f"Begin to process file: {file_path}")
    try:
        parser = PDFParser(runtime_config_path)
        content_list: list[Content] = parser.parse(file_path)
    except Exception as e:
        Logger.error(f"Parse failed:\n{e}")
        return

    Logger.info(f"Parsed {len(content_list)} contents from {file_path}")
    # HACK: hard coded parsed file path.
    pickle_content_path = os.path.join(temp_content_dir, "content_list.pickle")
    os.makedirs(temp_content_dir, exist_ok=True)
    with open(pickle_content_path, "wb") as f:
        pickle.dump(content_list, f)
        Logger.info(f"Saved content list to {pickle_content_path}")


@time_it(prefix="parse_pdf")
def parse_pdf(
    file_path: str, temp_content_dir: str, runtime_config_path: str
) -> list[Content]:
    job_executor = get_job_executor()

    try:
        shutil.rmtree(temp_content_dir)
    except FileNotFoundError:
        pass
    except Exception as e:
        Logger.error(f"Remove temp content dir {temp_content_dir} failed:\n{e}")

    job_executor.submit(
        parse_pdf_job,
        file_path=file_path,
        temp_content_dir=temp_content_dir,
        runtime_config_path=runtime_config_path,
    )
    job_executor.shutdown(wait=True)

    Logger.info("PDF parse job done")

    # parse returned content
    # HACK: hard coded parsed file path.
    pickle_content_path = os.path.join(temp_content_dir, "content_list.pickle")
    Logger.info(f"Loading content list from {pickle_content_path}")
    with open(pickle_content_path, "rb") as f:
        content_list: list[Content] = pickle.load(f)
    Logger.info(f"Loaded {len(content_list)} content from {pickle_content_path}")

    return content_list


# ------------------------------------------------------------------------------
# save parsed content
def save_parsed_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    src_lang: str = "",
    target_lang: str = "",
) -> None:
    md_writer.write("# " + "=" * 4 + "  Original Content  " + "=" * 4 + line_breaker)

    for _, content in enumerate(content_list):
        lines = ""
        if content.content_type == ContentType.TEXT:
            lines += content.content + line_breaker
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            # HACK: rewrite image path as relative path. Specific to Obsidian.
            if content.content_url:
                img_name = os.path.basename(content.content_url)
                md_img_path = relative_md_image_path(
                    sys_image_folder=os.path.dirname(content.content_url),
                    img_name=img_name,
                )
                # NOTE: no image description
                # # image description
                # img_content = load_base64_image(content.content_url)
                # img_description = image_chat(
                #     prompt="summarize what you see in the picture",
                #     image_content=img_content,
                #     gen_conf=Config.gen_conf.model_dump(),
                # )
                # if not img_description:
                #     img_description = "[LLM error]"

                lines += md_img_path + line_breaker

            lines += f"{line_breaker}{content.extra_description}{line_breaker}"
        md_writer.write(lines)
        md_writer.flush()


# ------------------------------------------------------------------------------
# translate func
def translate_text_content(text: str, src_lang: str, target_lang: str) -> str:
    if is_empty(text):
        return ""

    max_char_len = 16 * 1024
    full_result = ""
    for i in range(0, len(text), max_char_len):
        Logger.info(f"Processing segment {i}")
        segment = text[i : i + max_char_len]

        formatted_prompt = PROMPT_TRANSLATE.format(
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
def translate_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    src_lang: str,
    target_lang: str,
) -> None:
    """
    Translate contents.
    Args:
    - content_list: a list of content.
    - src_lang: source language display name.
    - target_lang: target language display name.
    """
    Logger.info(f"Total {len(content_list)} contents")

    md_writer.write("# " + "=" * 4 + "  Translated Content  " + "=" * 4 + line_breaker)

    i = 0
    max_content_num = 20
    while i < len(content_list):
        # image & table
        if content_list[i].content_type in [ContentType.TABLE, ContentType.IMAGE]:
            Logger.info(
                f"Translating content {i}, type: {content_list[i].content_type}"
            )
            # save images resources
            if content_list[i].content_url:
                img_name = os.path.basename(content_list[i].content_url)
                img_path = relative_md_image_path(Config.asset_save_dir, img_name)
                md_writer.write(img_path + line_breaker)

            translated = translate_text_content(
                content_list[i].extra_description, src_lang, target_lang
            )
            Logger.info(f"Translated content:\n{translated}")
            md_writer.write(translated + line_breaker)

            i = i + 1
            continue

        # text
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
        md_writer.write(translated + line_breaker)

        i = j

    md_writer.flush()


# ------------------------------------------------------------------------------
# summary func
@time_it(prefix="summary_content")
def summary_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    src_lang: str,
    target_lang: str,
) -> None:
    Logger.info(f"Summary_content, src_lang={src_lang}, target_lang={target_lang}")

    full_content = ""
    for content in content_list:
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
            f"Truncate full content by ratio: {ratio}, original length: {len(full_content)}"
        )
        full_content = full_content[: int(len(full_content) * ratio)]

    full_content = ensure_utf(full_content)

    formatted_prompt = PROMPT_SUMMARY.format(
        src_lang=src_lang,
        target_lang=target_lang,
        content=full_content,
    )
    Logger.info(f"Formatted prompt:\n{formatted_prompt}")

    summary = llm_chat(prompt=formatted_prompt, gen_conf=Config.gen_conf.model_dump())
    if not summary:
        summary = "[LLM error]"
    Logger.info(f"Content summary:\n{summary}")

    # save
    md_writer.write("# " + "=" * 4 + "  Content Summary  " + "=" * 4 + line_breaker)
    md_writer.write(summary + line_breaker)
    md_writer.write("-" * 4 + line_breaker)
    md_writer.flush()


step_func: dict[str, Callable[[TextIOWrapper, list[Content], str, str], None]] = {
    "original": save_parsed_content,
    "summary": summary_content,
    "translate": translate_content,
}


@time_it(prefix="process pipeline")
def process(
    file_path: str,
    temp_content_dir: str,
    final_md_file_save_dir: str,
    steps: list[str],
    src_lang: str = "英语",
    target_lang: str = "中文",
) -> str:
    """
    Process a pdf file and save parsed markdown file.

    Args:
    - file_path: absolute path to pdf file.
    - temp_content_dir: folder for saving temp parsed content.
    - final_md_file_save_dir: folder for saving final md file.
    - steps: list of processing step names.
    - src_lang: source language display name.
    - target_lang: target language display name.

    Returns:
    - path to the output markdown file.
    """
    Logger.info(f"Processing started, required steps: {steps}")

    # Ensure model weights are available and generate runtime config (main process)
    model_dir = ensure_mineru_model()
    runtime_config_path = prepare_mineru_runtime_config(model_dir)

    os.makedirs(temp_content_dir, exist_ok=True)
    os.makedirs(final_md_file_save_dir, exist_ok=True)
    name_without_suff = os.path.basename(file_path).rsplit(".", 1)[0]
    Logger.info(f"File name without out suffix: {name_without_suff}")

    # parse pdf
    content_list = parse_pdf(
        file_path=file_path,
        temp_content_dir=temp_content_dir,
        runtime_config_path=runtime_config_path,
    )

    # md writer
    md_file_path = os.path.join(final_md_file_save_dir, f"{name_without_suff}.md")
    Logger.info(f"md file path: {md_file_path}")
    with open(md_file_path, "w") as md_writer:
        md_writer.write(f"{name_without_suff}" + line_breaker)

        # apply step functions
        for step in steps:
            Logger.info(f"Processing step: {step}")
            step = step.strip()
            if step not in step_func:
                Logger.info(f"Step {step} not configured, ignore")
                continue
            func = step_func[step]
            func(md_writer, content_list, src_lang, target_lang)

    Logger.info(f"Parsed markdown saved to {md_file_path}")
    return md_file_path


LANG_MAPPING = {"en": "英语", "zh": "中文"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF paper parsing, translation, and summarization tool.",
    )

    parser.add_argument("--file_path", required=True, help="path to pdf file")
    parser.add_argument(
        "--final_md_file_save_dir",
        required=True,
        help="output directory for markdown file",
    )
    parser.add_argument("--src_lang", help="source language (en/zh)", default="en")
    parser.add_argument("--target_lang", help="target language (en/zh)", default="zh")
    parser.add_argument(
        "--steps",
        help="comma-separated processing steps",
        default="summary,translate,original",
    )
    parser.add_argument(
        "--temp_content_dir", help="temp content folder", default="./tmp/parsed_asset"
    )
    parser.add_argument("--llm_endpoint", help="LLM API endpoint URL")
    parser.add_argument("--llm_api_key", help="LLM API key (triggers OpenAI mode)")
    parser.add_argument("--chat_model", help="chat model name")
    parser.add_argument("--vision_model", help="vision model name")

    args = parser.parse_args()

    if args.llm_endpoint:
        Config.llm_endpoint = args.llm_endpoint
    if args.llm_api_key:
        Config.llm_api_key = args.llm_api_key
    if args.chat_model:
        Config.chat_model_name = args.chat_model
    if args.vision_model:
        Config.vision_model_name = args.vision_model

    temp_content_dir = os.path.realpath(args.temp_content_dir)
    final_md_file_save_dir = os.path.realpath(args.final_md_file_save_dir)
    src_lang = LANG_MAPPING.get(args.src_lang, args.src_lang)
    target_lang = LANG_MAPPING.get(args.target_lang, args.target_lang)

    Logger.info(f"Source language: {src_lang}, Target language: {target_lang}")
    Logger.info(f"Processing file: {os.path.basename(args.file_path)}")

    begin_ts = time.time()
    try:
        md_file_path = process(
            file_path=args.file_path,
            temp_content_dir=temp_content_dir,
            final_md_file_save_dir=final_md_file_save_dir,
            steps=args.steps.split(","),
            src_lang=src_lang,
            target_lang=target_lang,
        )
        result = {
            "status": "success",
            "output_file": md_file_path,
            "steps_completed": args.steps.split(","),
            "elapsed_seconds": round(time.time() - begin_ts, 2),
        }
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        Logger.error(f"Processing failed: {e}")
        result = {
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(time.time() - begin_ts, 2),
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

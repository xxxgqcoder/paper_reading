import copy
import json
import os
import shutil
import tempfile
import uuid
from typing import Any

from .config import Config, Content, ContentType
from .log import Logger
from .utils import cache_it, hash64, load_base64_image, safe_encode, safe_strip, time_it

# ---------------------------------------------------------------------------
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


_MINERU_CONFIG_TEMPLATE: dict[str, Any] = {
    "bucket_info": {
        "bucket-name-1": ["ak", "sk", "endpoint"],
        "bucket-name-2": ["ak", "sk", "endpoint"],
    },
    "latex-delimiter-config": {
        "display": {"left": "$$", "right": "$$"},
        "inline": {"left": "$", "right": "$"},
    },
    "llm-aided-config": {
        "title_aided": {
            "api_key": "",
            "base_url": "",
            "model": "",
            "enable": False,
        }
    },
    "mineru_model_source": "local",
}


def prepare_mineru_runtime_config(model_dir: str) -> str:
    """
    Build MinerU runtime config from built-in template, fill in model directory
    and self-reference path, then write to a temp file.

    Returns the path to the generated runtime config file.
    """
    conf = copy.deepcopy(_MINERU_CONFIG_TEMPLATE)

    runtime_config_path = os.path.join(
        tempfile.gettempdir(), "mineru_runtime_config.json"
    )
    conf["models-dir"] = {"pipeline": model_dir}
    conf["mineru_tools_conf_json"] = runtime_config_path

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=4)

    Logger.info(f"Runtime MinerU config written to: {runtime_config_path}")
    return runtime_config_path


# ---------------------------------------------------------------------------
# PDF parser


class PDFParser:
    """
    PDF parser implementation, backed by `MinerU <https://github.com/opendatalab/MinerU>`_.
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

        draw_layout_bbox(
            pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_layout.pdf"
        )
        draw_span_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_span.pdf")
        md_writer.write(f"{file_name}_origin.pdf", new_pdf_bytes)

        image_dir = str(os.path.basename(local_image_dir))
        md_content_str: list[str] = pipeline_union_make(
            pdf_info, MakeMode.MM_MD, image_dir
        )  # type: ignore
        md_writer.write_string(f"{file_name}.md", str(md_content_str))

        image_dir = str(os.path.basename(local_image_dir))
        content_list: list[dict[str, Any]] = pipeline_union_make(
            pdf_info, MakeMode.CONTENT_LIST, image_dir
        )  # type: ignore
        md_writer.write_string(
            f"{file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

        md_writer.write_string(
            f"{file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

        md_writer.write_string(
            f"{file_name}_model.json",
            json.dumps(model_json, ensure_ascii=False, indent=4),
        )

        def _save_image(src_path: str, dst_dir: str) -> None:
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)

        def _is_valid_content(content: dict[str, Any]) -> bool:
            if "type" not in content:
                return False
            if content["type"] in ["text", "equation"]:
                return "text" in content
            if content["type"] == "image":
                return "img_path" in content and len(content["img_path"]) > 0
            if content["type"] == "table":
                return "table_body" in content
            return True

        def _format_caption(caption: Any) -> str:
            if isinstance(caption, list):
                ret = "\n".join([str(e) for e in caption])
                return ret
            return str(caption)

        contents = []
        for content in content_list:
            if not _is_valid_content(content):
                Logger.info(f"Invalid content: {json.dumps(content, indent=4)}")
                continue

            if content["type"] in ["text", "equation"]:
                text = self.strip_text_content([content["text"]])
                if content.get("text_level", 0) == 1:
                    text = "# " + text
                contents.append(
                    Content(
                        content_type=ContentType.TEXT,
                        file_path=self.file_path,
                        content=safe_encode(text),
                        extra_description="",
                        content_url="",
                    )
                )

            elif content["type"] in ["image"]:
                texts = [
                    _format_caption(content.get("img_caption", "")),
                    _format_caption(content.get("img_footnote", "")),
                ]
                extra_description = self.strip_text_content(texts)
                if len(extra_description) == 0:
                    extra_description = ""

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

            elif content["type"] in ["table"]:
                texts = [
                    _format_caption(content.get("table_caption", "")),
                    _format_caption(content.get("table_footnote", "")),
                ]
                extra_description = self.strip_text_content(texts)
                if len(extra_description) == 0:
                    extra_description = ""

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
        content = ""
        for text in texts:
            striped = safe_strip(text)
            if len(striped) == 0 or striped == "[]":
                continue
            content += striped
            content += "\n\n"
        return content.strip()

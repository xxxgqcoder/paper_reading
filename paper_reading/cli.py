"""paper_reading CLI - unified PDF parsing, inspection, and page extraction.

Commands:
  parse (default)     Parse PDF to Markdown.
  inspect             Read front matter and infer printed/PDF page offset.
  extract             Extract printed page ranges with an optional offset.
  download-models     Download MineRU models to local cache.
  install-skills      Deploy SKILL.md to ~/.agents/skills/.
  get-schema          Print the JSON schema for PdfReadingParams.

Usage (parse):
    paper-reading parse --file_path <pdf> --output_dir <dir>
    paper-reading --config <config.json>

Usage (inspect/extract):
    paper-reading inspect --file_path <pdf> --inspect_pages 20
    paper-reading extract --file_path <pdf> --pages 1-10 --page_number_offset 12

Usage (download-models):
    paper-reading download-models [--model <model_name>]

Usage (install-skills):
    paper-reading install-skills [--uninstall]

Usage (get-schema):
    paper-reading get-schema
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .config import PdfAction, PdfReadingParams, Step
from .log import Logger
from .reader import pdf_reading

# 内置 skill 资源目录（打包到 wheel 中）；开发模式回退到仓库 .agents/skills/
_SKILLS_DIR = Path(__file__).parent / "_skills"
if not _SKILLS_DIR.is_dir():
    _SKILLS_DIR = Path(__file__).parent.parent / ".agents" / "skills"
# 全局 skill 安装目标目录
_GLOBAL_SKILLS_DIR = Path.home() / ".agents" / "skills"
_LEGACY_SKILL_NAMES = ("digest-to-md", "extract-pages")


def _download_models(model_name: str = "all", source: str = "huggingface") -> None:
    """下载 MineRU pipeline 所需的模型到本地缓存。

    Args:
        model_name: 保留参数，当前只支持 "all"（pipeline 模型集）
        source: 模型来源，"huggingface" 或 "modelscope"
    """
    if source not in ("huggingface", "modelscope"):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Unknown source: {source}. Use 'huggingface' or 'modelscope'"
                    ),
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    print(
        json.dumps(
            {"status": "starting", "source": source, "model": model_name},
            ensure_ascii=False,
        )
    )

    # 设置 MineRU 环境（必须在导入 mineru 模块前完成）
    from paper_reading.pdf_parser import _setup_mineru_env

    _setup_mineru_env(model_source=source)

    try:
        from mineru.cli.models_download import download_pipeline_models
    except ImportError as e:
        print(
            json.dumps(
                {"status": "error", "error": f"mineru is not installed: {e}"},
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    try:
        download_pipeline_models()
        print(
            json.dumps(
                {"status": "success", "source": source, "model": model_name},
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as e:
        print(
            json.dumps(
                {"status": "error", "source": source, "error": str(e)},
                ensure_ascii=False,
            )
        )
        sys.exit(1)


def _install_skills(uninstall: bool = False) -> None:
    """将内置 SKILL.md 部署到 ~/.agents/skills/ 或反向清理。"""
    if not _SKILLS_DIR.is_dir():
        print(
            json.dumps(
                {"status": "error", "error": "Built-in skills directory not found"},
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    skill_names = [d.name for d in _SKILLS_DIR.iterdir() if d.is_dir()]
    if not skill_names:
        print(
            json.dumps(
                {"status": "error", "error": "No skills found in package"},
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    if uninstall:
        removed = []
        for name in [*skill_names, *_LEGACY_SKILL_NAMES]:
            target = _GLOBAL_SKILLS_DIR / name
            if target.exists():
                shutil.rmtree(target)
                removed.append(name)
        print(
            json.dumps(
                {"status": "success", "action": "uninstall", "removed": removed},
                ensure_ascii=False,
            )
        )
        return

    _GLOBAL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    installed = []
    removed_legacy = []
    for name in _LEGACY_SKILL_NAMES:
        legacy_target = _GLOBAL_SKILLS_DIR / name
        if legacy_target.exists():
            shutil.rmtree(legacy_target)
            removed_legacy.append(name)
    for name in skill_names:
        src = _SKILLS_DIR / name
        target = _GLOBAL_SKILLS_DIR / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
        installed.append(name)
    print(
        json.dumps(
            {
                "status": "success",
                "action": "install",
                "skills": installed,
                "removed_legacy_skills": removed_legacy,
                "target": str(_GLOBAL_SKILLS_DIR),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    valid_steps = ",".join(s.value for s in Step)

    parser = argparse.ArgumentParser(
        description=(
            "Unified PDF tool: parse (default), inspect front matter, extract pages,"
            " or install the bundled skill."
        ),
    )

    parser.add_argument(
        "command_arg",
        nargs="?",
        default="parse",
        choices=[
            "parse",
            "process",
            "inspect",
            "extract",
            "extract-pages",
            "download-models",
            "install-skills",
            "get-schema",
        ],
        help=(
            "parse (default), inspect, extract, download-models, "
            "install-skills, or get-schema"
        ),
    )

    # --- parse 参数 ---
    parser.add_argument("--config", help="Path to JSON config file (parse only)")
    parser.add_argument("--file_path", help="path to pdf file")
    parser.add_argument(
        "--final_md_file_save_dir",
        help="output directory for markdown file",
    )
    parser.add_argument(
        "--steps",
        default="original",
        help=f"comma-separated processing steps (only: {valid_steps})",
    )
    parser.add_argument(
        "--asset_save_dir",
        default="",
        help=(
            "directory to save parsed assets "
            "(default: auto, {pdf_stem}_images next to PDF)"
        ),
    )
    parser.add_argument(
        "--mineru_model_source",
        default="modelscope",
        choices=["huggingface", "modelscope"],
        help="MineRU model source (default: modelscope)",
    )
    parser.add_argument(
        "--mineru_device",
        default="",
        choices=["", "auto", "mps", "cuda", "cpu"],
        help="MineRU inference device (default: auto-detect from environment)",
    )

    # --- inspect/extract 参数 ---
    parser.add_argument("--input_pdf", help="legacy alias for --file_path")
    parser.add_argument(
        "--pages",
        nargs="+",
        help="printed page ranges, e.g. 1-93 '100,105-110' (for extract)",
    )
    parser.add_argument("--output_dir", help="output directory (for parse/extract)")
    parser.add_argument(
        "--page_number_offset",
        type=int,
        default=0,
        help="printed-to-PDF page offset: pdf_page = printed_page + offset",
    )
    parser.add_argument(
        "--inspect_pages",
        type=int,
        default=20,
        help="number of front-matter PDF pages to inspect (default: 20)",
    )

    # --- install-skills 参数 ---
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="remove installed skills (for install-skills)",
    )

    # --- download-models 参数 ---
    parser.add_argument(
        "--model",
        default="all",
        help="model to download (default: all, for download-models)",
    )
    parser.add_argument(
        "--source",
        default="huggingface",
        choices=["hf", "modelscope", "huggingface"],
        help=(
            "model download source: huggingface/hf or modelscope (for download-models)"
        ),
    )

    args = parser.parse_args()

    # 命令分发
    if args.command_arg == "get-schema":
        # 输出 JSON Schema
        print(
            json.dumps(
                PdfReadingParams.model_json_schema(),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command_arg == "download-models":
        # 将 hf 别名解析为 huggingface
        source = "huggingface" if args.source == "hf" else args.source
        _download_models(model_name=args.model, source=source)
        return

    if args.command_arg == "install-skills":
        _install_skills(uninstall=args.uninstall)
        return

    if args.command_arg in ("inspect", "extract", "extract-pages"):
        action = (
            PdfAction.INSPECT if args.command_arg == "inspect" else PdfAction.EXTRACT
        )
        input_pdf = args.input_pdf or args.file_path
        if not input_pdf:
            parser.error(f"{args.command_arg} requires --file_path")
        if action == PdfAction.EXTRACT and not args.pages:
            parser.error(f"{args.command_arg} requires --pages")

        begin_ts = time.time()
        try:
            params = PdfReadingParams(
                action=action,
                file_path=input_pdf,
                pages=args.pages or [],
                output_dir=args.output_dir or "",
                page_number_offset=args.page_number_offset,
                inspect_pages=args.inspect_pages,
            )
            unified_result = asyncio.run(pdf_reading(params))
            result = {"status": "success", **unified_result.model_dump(mode="json")}
            print(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            Logger.error(f"{args.command_arg} failed: {e}")
            result = {
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(time.time() - begin_ts, 2),
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
        return

    # --- parse / legacy process (default) ---
    try:
        if args.config:
            # 从文件加载
            config_path = Path(args.config)
            if not config_path.is_file():
                parser.error(f"Config file not found: {args.config}")
            with open(config_path, encoding="utf-8") as f:
                # 允许传入部分参数，从 CLI 补全吗？不，简单起见，config 文件优先且完备
                # 或者，我们可以先加载 config，也就是个 dict，然后 validate
                config_data = json.load(f)
                config_data.setdefault("action", "parse")
                params = PdfReadingParams.model_validate(config_data)
        else:
            output_dir = args.output_dir or args.final_md_file_save_dir
            if not args.file_path or not output_dir:
                parser.error(
                    "parse requires --file_path and --output_dir "
                    "(legacy: --final_md_file_save_dir), or --config"
                )

            # 构造 ProcessParams
            params = PdfReadingParams(
                action=PdfAction.PARSE,
                file_path=args.file_path,
                output_dir=os.path.realpath(output_dir),
                steps=[s.strip() for s in args.steps.split(",")],
                asset_save_dir=args.asset_save_dir,
                mineru_model_source=args.mineru_model_source,
                mineru_device=args.mineru_device,
            )
    except Exception as e:
        print(
            json.dumps(
                {"status": "error", "error": f"Invalid configuration: {e}"},
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    Logger.info(f"Processing file: {os.path.basename(params.file_path)}")

    try:
        proc_result = asyncio.run(pdf_reading(params))
        result = {"status": "success", **proc_result.model_dump(mode="json")}
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        Logger.error(f"Processing failed: {e}")
        # import traceback
        # traceback.print_exc()
        result = {
            "status": "error",
            "error": str(e),
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

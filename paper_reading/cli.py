"""paper_reading CLI - PDF parsing, translation, summarization and page extraction.

Commands:
  process (default)   Parse PDF, optional summary/translate; output Markdown.
  extract-pages       Extract page ranges from a PDF.
  download-models     Download OpenDataLoader models to local cache.
  install-skills      Deploy SKILL.md to ~/.agents/skills/.
  get-schema          Print the JSON schema for ProcessParams.

Usage (process):
    paper-reading --file_path <pdf> --final_md_file_save_dir <dir>
    paper-reading --config <config.json>

Usage (extract-pages):
    paper-reading extract-pages --input_pdf <pdf> --pages 1-10 3,5-7

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

from .config import ExtractPagesParams, GenerationConfig, ProcessParams, Step
from .extract_pages import run_extract_pages
from .log import Logger
from .pipeline import process

# 内置 skill 资源目录（打包到 wheel 中）；开发模式回退到仓库 .agents/skills/
_SKILLS_DIR = Path(__file__).parent / "_skills"
if not _SKILLS_DIR.is_dir():
    _SKILLS_DIR = Path(__file__).parent.parent / ".agents" / "skills"
# 全局 skill 安装目标目录
_GLOBAL_SKILLS_DIR = Path.home() / ".agents" / "skills"


def _download_models(model_name: str = "all", source: str = "huggingface") -> None:
    """下载 MineRU pipeline 所需的模型到本地缓存。

    Args:
        model_name: 保留参数，当前只支持 "all"（pipeline 模型集）
        source: 模型来源，"huggingface" 或 "modelscope"
    """
    if source not in ("huggingface", "modelscope"):
        print(
            json.dumps(
                {"status": "error", "error": f"Unknown source: {source}. Use 'huggingface' or 'modelscope'"},
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
        for name in skill_names:
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
                "target": str(_GLOBAL_SKILLS_DIR),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    valid_steps = ",".join(s.value for s in Step)

    # 获取 GenerationConfig 的默认值作为 argparse 的默认值
    default_gen_conf = GenerationConfig()

    parser = argparse.ArgumentParser(
        description=(
            "PDF tool: parse/translate/summarize (default),"
            " extract pages, or install skills."
        ),
    )

    # Command: process (default but explicit via subparser adds clarity)
    # 为了保持向后兼容，如果第一个参数不是子命令，默认为 "process"
    # 但 argparse 原生不支持这种混合模式很容易。这里我们采用显式添加 process 子命令
    # 同时在如果不匹配任何子命令时，尝试解析为主命令参数
    # 为简单起见，我们把 process 的参数加到主解析器上，但也添加 explicit command

    # 策略：如果不带任何子命令，会被视作 process
    pass

    parser.add_argument(
        "command_arg",
        nargs="?",
        default="process",
        choices=[
            "process",
            "extract-pages",
            "download-models",
            "install-skills",
            "get-schema",
        ],
        help=(
            "process (default), extract-pages, download-models, "
            "install-skills, or get-schema"
        ),
    )

    # --- process 参数 ---
    parser.add_argument("--config", help="Path to JSON config file (process only)")
    parser.add_argument("--file_path", help="path to pdf file")
    parser.add_argument(
        "--final_md_file_save_dir",
        help="output directory for markdown file",
    )
    parser.add_argument(
        "--steps",
        default="summary,translate,original",
        help=f"comma-separated processing steps (valid: {valid_steps})",
    )
    parser.add_argument(
        "--src_lang", default="en", help="source language (default: en)"
    )
    parser.add_argument(
        "--target_lang", default="zh", help="target language (default: zh)"
    )
    parser.add_argument(
        "--chat_model_name",
        default="qwen/qwen3.5-flash-02-23",
        help=(
            "chat model name "
            "(default: qwen/qwen3.5-flash-02-23)"
        ),
    )
    parser.add_argument(
        "--vision_model_name",
        default="qwen/qwen3.5-flash-02-23",
        help=(
            "vision model name "
            "(default: qwen/qwen3.5-flash-02-23)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_gen_conf.temperature,
        help="generation temperature",
    )
    parser.add_argument(
        "--top_p", type=float, default=default_gen_conf.top_p, help="top-p sampling"
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=default_gen_conf.repeat_penalty,
        help="repeat penalty",
    )
    parser.add_argument(
        "--num_ctx",
        type=int,
        default=default_gen_conf.num_ctx,
        help="model context length",
    )
    parser.add_argument(
        "--max_context_token_num",
        type=int,
        default=120000,
        help="max input token num for summary",
    )
    parser.add_argument(
        "--asset_save_dir",
        default="attachments",
        help="directory to save parsed assets",
    )
    parser.add_argument(
        "--cache_data_dir",
        default="~/.cache/llm_cache",
        help="disk cache directory",
    )
    parser.add_argument(
        "--llm_endpoint",
        default=os.environ.get("PR_LLM_ENDPOINT", ""),
        help="LLM API endpoint (default: env PR_LLM_ENDPOINT)",
    )
    parser.add_argument(
        "--llm_api_key",
        default=os.environ.get("PR_LLM_API_KEY", ""),
        help="LLM API key (default: env PR_LLM_API_KEY)",
    )
    parser.add_argument(
        "--mineru_model_source",
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="MineRU model source (default: huggingface)",
    )
    parser.add_argument(
        "--mineru_device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="MineRU inference device (default: auto)",
    )

    # --- extract-pages 参数 ---
    parser.add_argument("--input_pdf", help="path to input PDF (for extract-pages)")
    parser.add_argument(
        "--pages",
        nargs="+",
        help="page ranges to extract, e.g. 1-93 '100,105-110' (for extract-pages)",
    )
    parser.add_argument("--output_dir", help="output directory (for extract-pages)")

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
        help="model download source: huggingface/hf or modelscope (for download-models)",
    )

    args = parser.parse_args()

    # 命令分发
    if args.command_arg == "get-schema":
        # 输出 JSON Schema
        print(
            json.dumps(
                ProcessParams.model_json_schema(),
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

    if args.command_arg == "extract-pages":
        if not args.input_pdf and args.file_path and args.pages:
            # 尝试使用 --file_path 作为 fallback
            args.input_pdf = args.file_path
        if not args.input_pdf or not args.pages:
            parser.error("extract-pages requires --input_pdf and --pages")

        begin_ts = time.time()
        try:
            params = ExtractPagesParams(
                input_pdf=args.input_pdf,
                pages=args.pages,
                output_dir=args.output_dir,
            )
            out_paths = run_extract_pages(params)
            result = {
                "status": "success",
                "output_files": out_paths,
                "elapsed_seconds": round(time.time() - begin_ts, 2),
            }
            print(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            Logger.error(f"Extract-pages failed: {e}")
            result = {
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(time.time() - begin_ts, 2),
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
        return

    # --- process (default) ---
    # 构造 ProcessParams
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
                params = ProcessParams.model_validate(config_data)
        else:
            if not args.file_path or not args.final_md_file_save_dir:
                parser.error(
                    "process requires --file_path and "
                    "--final_md_file_save_dir (or --config)"
                )

            # 手动构造 GenerationConfig
            gen_conf = GenerationConfig(
                temperature=args.temperature,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                num_ctx=args.num_ctx,
            )

            # 构造 ProcessParams
            params = ProcessParams(
                file_path=args.file_path,
                output_dir=os.path.realpath(args.final_md_file_save_dir),
                steps=[s.strip() for s in args.steps.split(",")],
                src_lang=args.src_lang,
                target_lang=args.target_lang,
                llm_endpoint=args.llm_endpoint,
                llm_api_key=args.llm_api_key,
                chat_model_name=args.chat_model_name,
                vision_model_name=args.vision_model_name,
                gen_conf=gen_conf,
                max_context_token_num=args.max_context_token_num,
                asset_save_dir=args.asset_save_dir,
                cache_data_dir=args.cache_data_dir,
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
        proc_result = asyncio.run(process(params))
        result = {
            "status": "success",
            "output_file": proc_result.output_file,
            "steps_completed": proc_result.steps_completed,
            "elapsed_seconds": proc_result.elapsed_seconds,
        }
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

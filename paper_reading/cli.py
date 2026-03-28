"""paper_reading CLI - PDF parsing, translation, summarization and page extraction.

Commands:
  process (default)   Parse PDF, optional summary/translate; output Markdown.
  extract-pages       Extract page ranges from a PDF.
  install-skills      Deploy SKILL.md to ~/.agents/skills/.
  get-schema          Print the JSON schema for ProcessParams.

Usage (process):
    paper-reading --file_path <pdf> --final_md_file_save_dir <dir>
    paper-reading --config <config.json>

Usage (extract-pages):
    paper-reading extract-pages --input_pdf <pdf> --pages 1-10 3,5-7

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

from .config import ExtractPagesParams, ProcessParams, GenerationConfig, Step
from .extract_pages import run_extract_pages
from .log import Logger
from .pipeline import process

# 内置 skill 资源目录（打包到 wheel 中）；开发模式回退到仓库 .agents/skills/
_SKILLS_DIR = Path(__file__).parent / "_skills"
if not _SKILLS_DIR.is_dir():
    _SKILLS_DIR = Path(__file__).parent.parent / ".agents" / "skills"
# 全局 skill 安装目标目录
_GLOBAL_SKILLS_DIR = Path.home() / ".agents" / "skills"


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
        choices=["process", "extract-pages", "install-skills", "get-schema"],
        help="process (default), extract-pages, or install-skills",
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
        "--chat_model_name", default="llama3", help="chat model name (default: llama3)"
    )
    parser.add_argument(
        "--vision_model_name",
        default="llama3",
        help="vision model name (default: llama3)",
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
        default=1024 * 16,
        help="max input token num for summary",
    )
    parser.add_argument(
        "--asset_save_dir", default="attachments", help="directory to save parsed assets"
    )
    parser.add_argument(
        "--cache_data_dir", default="~/.cache/llm_cache", help="disk cache directory"
    )
    parser.add_argument(
        "--llm_endpoint",
        default=os.environ.get("LLM_ENDPOINT", ""),
        help="LLM API endpoint (default: env LLM_ENDPOINT)",
    )
    parser.add_argument(
        "--llm_api_key",
        default=os.environ.get("LLM_API_KEY", ""),
        help="LLM API key (default: env LLM_API_KEY)",
    )
    parser.add_argument(
        "--odl_container_name",
        default="opendataloader-api-server",
        help="OpenDataLoader Docker container name (default: opendataloader-api-server)",
    )
    parser.add_argument(
        "--odl_volume_host_dir",
        default=os.environ.get("ODL_VOLUME_HOST_DIR", ""),
        help="Host directory mounted as /data in the OpenDataLoader container (default: env ODL_VOLUME_HOST_DIR)",
    )
    parser.add_argument(
        "--odl_hybrid_mode",
        default="full",
        choices=["auto", "full"],
        help="OpenDataLoader hybrid mode: full (default, highest accuracy, high memory) or auto",
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

    args = parser.parse_args()

    # 命令分发
    if args.command_arg == "get-schema":
        # 输出 JSON Schema
        print(json.dumps(ProcessParams.model_json_schema(), indent=2, ensure_ascii=False))
        return

    if args.command_arg == "install-skills":
        _install_skills(uninstall=args.uninstall)
        return

    if args.command_arg == "extract-pages":
        if not args.input_pdf or not args.pages:
            if not args.input_pdf:
                # 尝试使用 --file_path 作为 fallback
                if args.file_path and args.pages:
                    args.input_pdf = args.file_path
                else:
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
            with open(config_path, "r", encoding="utf-8") as f:
                # 允许传入部分参数，从 CLI 补全吗？不，简单起见，config 文件优先且完备
                # 或者，我们可以先加载 config，也就是个 dict，然后 validate
                config_data = json.load(f)
                params = ProcessParams.model_validate(config_data)
        else:
            if not args.file_path or not args.final_md_file_save_dir:
                parser.error("process requires --file_path and --final_md_file_save_dir (or --config)")
            
            # 手动构造 GenerationConfig
            gen_conf = GenerationConfig(
                temperature=args.temperature,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                num_ctx=args.num_ctx
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
                odl_container_name=args.odl_container_name,
                odl_volume_host_dir=args.odl_volume_host_dir,
                odl_hybrid_mode=args.odl_hybrid_mode,
            )
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Invalid configuration: {e}"}, ensure_ascii=False))
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

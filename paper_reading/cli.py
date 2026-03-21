"""paper_reading CLI - PDF parsing, translation, summarization and page extraction.

Commands:
  process (default)   Parse PDF, optional summary/translate; output Markdown.
  extract-pages       Extract page ranges from a PDF.
  install-skills      Deploy SKILL.md to ~/.agents/skills/.

Usage (process):
    paper-reading --file_path <pdf> --final_md_file_save_dir <dir>

Usage (extract-pages):
    paper-reading extract-pages --input_pdf <pdf> --pages 1-10 3,5-7

Usage (install-skills):
    paper-reading install-skills [--uninstall]
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .config import Config, ExtractPagesParams, ProcessParams, Step
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
    parser = argparse.ArgumentParser(
        description=(
            "PDF tool: parse/translate/summarize (default),"
            " extract pages, or install skills."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="process",
        choices=["process", "extract-pages", "install-skills"],
        help="process (default), extract-pages, or install-skills",
    )
    parser.add_argument("--file_path", help="path to pdf file (required for process)")
    parser.add_argument(
        "--final_md_file_save_dir",
        help="output directory for markdown file (required for process)",
    )
    parser.add_argument(
        "--steps",
        help=f"override config steps (valid: {valid_steps})",
    )
    parser.add_argument("--src_lang", help="override config source language (en/zh)")
    parser.add_argument("--target_lang", help="override config target language (en/zh)")

    # extract-pages 专用参数
    parser.add_argument("--input_pdf", help="path to input PDF (for extract-pages)")
    parser.add_argument(
        "--pages",
        nargs="+",
        help="page ranges to extract, e.g. 1-93 '100,105-110' (for extract-pages)",
    )
    parser.add_argument("--output_dir", help="output directory (for extract-pages)")

    # install-skills 专用参数
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="remove installed skills (for install-skills)",
    )

    args = parser.parse_args()

    if args.command == "install-skills":
        _install_skills(uninstall=args.uninstall)
        return

    if args.command == "extract-pages":
        begin_ts = time.time()
        try:
            # CLI 参数优先，回退到 config.yaml
            if args.input_pdf and args.pages:
                params = ExtractPagesParams(
                    input_pdf=args.input_pdf,
                    pages=args.pages,
                    output_dir=args.output_dir,
                )
                out_paths = run_extract_pages(params)
            else:
                out_paths = run_extract_pages()
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

    # process (default)
    if not args.file_path or not args.final_md_file_save_dir:
        parser.error("process requires --file_path and --final_md_file_save_dir")

    steps_raw = args.steps if args.steps else Config.steps
    params = ProcessParams(
        file_path=args.file_path,
        output_dir=os.path.realpath(args.final_md_file_save_dir),
        steps=[s.strip() for s in steps_raw.split(",")],
        src_lang=args.src_lang or Config.src_lang,
        target_lang=args.target_lang or Config.target_lang,
    )

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
        result = {
            "status": "error",
            "error": str(e),
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

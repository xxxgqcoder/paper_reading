"""paper_reading CLI - PDF parsing, translation, summarization and page extraction.

Commands:
  process (default)   Parse PDF, optional summary/translate; output Markdown.
  extract-pages       Extract page ranges from a PDF.

Usage (process):
    python -m paper_reading.cli --file_path <pdf> --final_md_file_save_dir <dir>

Usage (extract-pages):
    python -m paper_reading.cli extract-pages
"""

import argparse
import json
import os
import sys
import time

from .config import LANG_MAPPING, Config, Step
from .extract_pages import run_extract_pages
from .log import Logger
from .pipeline import process


def main() -> None:
    valid_steps = ",".join(s.value for s in Step)
    parser = argparse.ArgumentParser(
        description="PDF tool: parse/translate/summarize (default) or extract pages. "
        "Use 'extract-pages' as first argument for page extraction.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="process",
        choices=["process", "extract-pages"],
        help="process (default) or extract-pages",
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

    args = parser.parse_args()

    if args.command == "extract-pages":
        begin_ts = time.time()
        try:
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
        sys.exit(0)

    # process (default)
    if not args.file_path or not args.final_md_file_save_dir:
        parser.error("process requires --file_path and --final_md_file_save_dir")

    steps = args.steps if args.steps else Config.steps
    src_lang_code = args.src_lang if args.src_lang else Config.src_lang
    target_lang_code = args.target_lang if args.target_lang else Config.target_lang
    final_md_file_save_dir = os.path.realpath(args.final_md_file_save_dir)

    src_lang = LANG_MAPPING.get(src_lang_code, src_lang_code)
    target_lang = LANG_MAPPING.get(target_lang_code, target_lang_code)

    Logger.info(f"Source language: {src_lang}, Target language: {target_lang}")
    Logger.info(f"Processing file: {os.path.basename(args.file_path)}")

    begin_ts = time.time()
    try:
        md_file_path = process(
            file_path=args.file_path,
            final_md_file_save_dir=final_md_file_save_dir,
            steps=steps.split(","),
            src_lang=src_lang,
            target_lang=target_lang,
        )
        step_names = [s.strip() for s in steps.split(",")]
        result = {
            "status": "success",
            "output_file": md_file_path,
            "steps_completed": step_names,
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


if __name__ == "__main__":
    main()

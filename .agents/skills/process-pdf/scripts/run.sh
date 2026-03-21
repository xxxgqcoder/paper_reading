#!/usr/bin/env bash
# Skill: process-pdf — wrapper script for Agent invocation
# Usage: bash run.sh <pdf_path> <output_dir> [--steps summary,translate,original] [--src_lang en] [--target_lang zh]
# Output: JSON to stdout
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo '{"status":"error","error":"Usage: run.sh <pdf_path> <output_dir> [--steps ...] [--src_lang ...] [--target_lang ...]"}' >&2
    exit 1
fi

FILE_PATH="$1"
OUTPUT_DIR="$2"
shift 2

cd "$PROJECT_ROOT"
exec uv run python -m paper_reading.cli \
    --file_path "$FILE_PATH" \
    --final_md_file_save_dir "$OUTPUT_DIR" \
    "$@"

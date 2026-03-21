#!/usr/bin/env bash
# Skill: extract-pages — wrapper script for Agent invocation
# Usage: bash run.sh <input_pdf> <page_range1> [page_range2 ...] [--output_dir <dir>]
# Output: JSON to stdout
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo '{"status":"error","error":"Usage: run.sh <input_pdf> <page_range1> [page_range2 ...] [--output_dir <dir>]"}' >&2
    exit 1
fi

INPUT_PDF="$1"
shift

# 解析参数：页码范围和可选的 --output_dir
PAGES=()
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            PAGES+=("$1")
            shift
            ;;
    esac
done

if [[ ${#PAGES[@]} -eq 0 ]]; then
    echo '{"status":"error","error":"At least one page range is required"}' >&2
    exit 1
fi

# 构建 Python 调用：通过 API 传入参数
PAGES_JSON="["
for i in "${!PAGES[@]}"; do
    [[ $i -gt 0 ]] && PAGES_JSON+=","
    PAGES_JSON+="\"${PAGES[$i]}\""
done
PAGES_JSON+="]"

OUTPUT_DIR_ARG=""
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR_ARG=", output_dir=\"$OUTPUT_DIR\""
fi

cd "$PROJECT_ROOT"
exec uv run python -c "
import json, time
from paper_reading import run_extract_pages, ExtractPagesParams
begin = time.time()
params = ExtractPagesParams(input_pdf='$INPUT_PDF', pages=$PAGES_JSON$OUTPUT_DIR_ARG)
paths = run_extract_pages(params)
print(json.dumps({'status': 'success', 'output_files': paths, 'elapsed_seconds': round(time.time() - begin, 2)}, ensure_ascii=False))
"

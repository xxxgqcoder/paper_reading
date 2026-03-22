#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

file_path="/Users/xcoder/aDrive/PDF/Effective context engineering for AI agents.pdf"
final_md_file_save_dir="/Users/xcoder/obsidian/Profession/Paper Reading"

uv run python -m paper_reading.cli \
    --chat_model_name="qwen/qwen3.5-flash-02-23" \
    --file_path="${file_path}" \
    --final_md_file_save_dir="${final_md_file_save_dir}" \
    --steps="summary,translate,original"

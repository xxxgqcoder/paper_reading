#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

file_path="/Users/xcoder/aDrive/PDF/Effective context engineering for AI agents.pdf"
final_md_file_save_dir="/Users/xcoder/obsidian/Profession/Paper Reading"

uv run paper-reading parse \
    --file_path="${file_path}" \
    --output_dir="${final_md_file_save_dir}"


#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

file_path="${HOME}/obsidian/PDF/Efficient Memory Management for Large Language Model Serving with PagedAttention.pdf"
final_md_file_save_dir="${HOME}/obsidian/Industrial/Paper Reading"

uv run python -m paper_reading.cli \
    --file_path="${file_path}" \
    --final_md_file_save_dir="${final_md_file_save_dir}" \
    --steps="original,summary,translate"


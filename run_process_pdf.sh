
#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

file_path="${HOME}/obsidian/PDF/Qwen2.5-VL Technical Report.pdf"
final_md_file_save_dir="${HOME}/obsidian/Industrial/Paper Reading"


python process_pdf.py \
    --file_path="${file_path}" \
    --final_md_file_save_dir="${final_md_file_save_dir}" \
    --steps="original"


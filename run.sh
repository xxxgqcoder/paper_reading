
#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

file_path='/Users/xcoder/obsidian/Profession/PDF/Beyond Outlining- Heterogeneous Recursive Planning for Adaptive Long-form Writing with Language Models.pdf'
temp_content_dir="./tmp/parsed_assets"
final_md_file_save_dir="/Users/xcoder/obsidian/Profession/Processed PDF"


python main.py \
    --file_path="${file_path}" \
    --temp_content_dir="${temp_content_dir}" \
    --final_md_file_save_dir="${final_md_file_save_dir}"


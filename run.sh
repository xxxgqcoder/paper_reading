
#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"

file_path='/Users/xcoder/obsidian/Profession/PDF/On the Theoretical Limitations of Embedding-Based Retrieval.pdf'
output_dir='./tmp/parsed_assets'
sys_image_folder='/Users/xcoder/obsidian/Profession/attachments'
final_md_file_save_dir='/Users/xcoder/obsidian/Profession/Processed PDF'
src_lang='en'
target_lang='zh'
ollama_host='http://127.0.0.1:11434'
ollama_model='qwen3:30b-a3b-thinking-2507-q4_K_M'
steps='summary,translate'


python main.py \
    --file_path="${file_path}" \
    --output_dir="${output_dir}" \
    --sys_image_folder="${sys_image_folder}" \
    --final_md_file_save_dir="${final_md_file_save_dir}" \
    --src_lang=$src_lang \
    --target_lang=$target_lang \
    --ollama_host=$ollama_host \
    --ollama_model=$ollama_model \
    --steps=$steps \
    > processing.log 2>&1


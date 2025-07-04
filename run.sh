
#!/usr/bin/env bash
set -e
cd $(dirname "$0")
echo "working directory $(pwd)"


file_path='/Users/xcoder/obsidian/Profession/PDF/BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf'
output_dir='./parsed_assets'
sys_image_folder='/Users/xcoder/obsidian/Profession/attachments'
final_md_file_save_dir='/Users/xcoder/obsidian/Profession/Paper Reading'
src_lang='en'
target_lang='zh'
ollama_host='http://127.0.0.1:11434'
ollama_model='qwen3:30b-a3b'
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


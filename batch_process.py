import os

base_folder = '/Users/xcoder/obsidian/Profession/PDF'
log_file = "processing.log"

output_dir = './tmp/parsed_assets'
sys_image_folder = '/Users/xcoder/obsidian/Profession/attachments'
final_md_file_save_dir = '/Users/xcoder/obsidian/Profession/Paper Reading'
src_lang = 'en'
target_lang = 'zh'
ollama_host = 'http://127.0.0.1:11434'
ollama_model = 'qwen3:30b-a3b'
steps = 'summary,translate'

md_files = os.listdir(final_md_file_save_dir)
ignore_files = sorted([f.rsplit('.', 1)[0] + '.pdf' for f in md_files]) + ['.DS_Store']

file_list = os.listdir(base_folder)
print(f'original total {len(file_list)} to process')
file_list = sorted(list(set(file_list) - set(ignore_files)))
print(f'after removing ignore file, total {len(file_list)} files to process')

for i, file_name in enumerate(file_list):
    print(f'{i}: begin processing {file_name}')

    file_path = os.path.join(base_folder, file_name)

    cmd = f"""
python main.py \
    --file_path="{file_path}" \
    --output_dir="{output_dir}" \
    --sys_image_folder="{sys_image_folder}" \
    --final_md_file_save_dir="{final_md_file_save_dir}" \
    --src_lang={src_lang} \
    --target_lang={target_lang} \
    --ollama_host={ollama_host} \
    --ollama_model={ollama_model} \
    --steps={steps} \
    > {log_file} 2>&1
"""
    print(f"command line to run: {cmd}")
    os.system(cmd)

    print(f'finish processing {file_path}')

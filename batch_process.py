import os
import re

base_folder = '/Users/xcoder/obsidian/Profession/PDF/'
log_file = "processing.log"

ignore_list = [
    'DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf',
    'M3-Embedding- Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.pdf',
    '.DS_Store',
]

with open(log_file) as f:
    for line in f.readlines():
        log_pattern = "parsed markdown saved to"
        if log_pattern not in line:
            continue
        file_path = line.split(log_pattern)[-1].strip()
        name_without_suff = os.path.basename(file_path).split('.')[0]
        print(f'found log for file: {name_without_suff}')
        ignore_list.append(f"{name_without_suff}.pdf")

ignore_list = list(set(ignore_list))
print(f'total {len(ignore_list)} to ignore')

file_list = os.listdir(base_folder)
print(f'original total {len(file_list)} to process')
file_list = sorted(list(set(file_list) - set(ignore_list)))
print(f'after removing ignore file, total {len(file_list)} files to process')

output_dir = './parsed_assets'
sys_image_folder = '/Users/xcoder/obsidian/Profession/attachments'
final_md_file_save_dir = '/Users/xcoder/obsidian/Profession/Paper Reading'
src_lang = 'en'
target_lang = 'zh'
ollama_host = 'http://127.0.0.1:11434'
ollama_model = 'qwen3:30b-a3b'
steps = 'summary,translate,original'

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
    >> {log_file} 2>&1
"""
    print(f"command line to run: {cmd}")
    os.system(cmd)

    print(f'finish processing {file_path}')

import os
import subprocess

base_folder = '/Users/xcoder/obsidian/Profession/PDF/'
file_list = [
    'DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf',
    'M3-Embedding- Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.pdf',
    'QWEN TECHNICAL REPORT.pdf',
    'QWEN2 TECHNICAL REPORT.pdf',
    'Qwen2.5-Coder Technical Report.pdf',
    'QWEN2.5-MATH TECHNICAL REPORT- TOWARD MATHEMATICAL EXPERT MODEL VIA SELFIMPROVEMENT.pdf',
    'Qwen2.5-VL Technical Report.pdf',
    'YaRN- Efficient Context Window Extension of Large Language Models.pdf',
]

output_dir = './parsed_assets'
sys_image_folder = '/Users/xcoder/obsidian/Profession/attachments'
final_md_file_save_dir = '/Users/xcoder/obsidian/Profession/Paper Reading'
src_lang = 'en'
target_lang = 'zh'
ollama_host = 'http://127.0.0.1:11434'
ollama_model = 'qwen3:30b-a3b'
steps = 'summary,translate,original'

for file_name in file_list:
    print(f'begin processing {file_path} ')

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
    > processing.log 2>&1
"""
    os.system(cmd)

    print(f'finish processing {file_path}')
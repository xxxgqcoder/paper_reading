# paper_reading
A pipeline for paper reading.

The script first use `MinerU` to parse pdf paper, then call local ollama to summarize paper content and translate paper (into Chinese). Paper summary and content translation are saved to one markdown file.


# Usage 
```shell
python process.py \
    --file_path="${file_path}" \
    --output_dir="${output_dir}" \
    --sys_image_folder="${sys_image_folder}" \
    --final_md_file_save_dir="${final_md_file_save_dir}" \
    --src_lang=$src_lang \
    --target_lang=$target_lang \
    --ollama_host=$ollama_host \
    --ollama_model=$ollama_model \
    --steps='summary,translate,original'
```

command line flag explaination:
- `file_path`: path to the pdf file
- `output_dir`: folder to save intermediate asseets, i.e, pictures, model outputs.
- `sys_image_folder`: folder for permanently  saving parsed pictures.
- `final_md_file_save_dir`: where to save final markdown file.
- `src_lang`: source language of original paper.
- `target_lang`: languaged used to summary and translate original paper.
- `ollama_host`: local ollama host, default to `http://127.0.0.1:11434`
- `ollama_model`: local ollama model name, default to `qwen3:30b-a3b`.
- `steps`: comma splitted processing step. 
    - `summary`: get summary of file. 
    - `translate`: translate parsed content. 
    - `original`: save original parsed content.




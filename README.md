# Paper Reading Pipeline

This repository contains a Python-based pipeline for processing PDF documents, specifically academic papers. It uses `MinerU` to parse the PDF, extracting text, images, and tables. It then uses a local Ollama instance to summarize and translate the content, saving the final result as a Markdown file.

## Features

- **PDF Parsing**: Leverages `MinerU` to accurately parse PDF layouts, including text, images, and tables.
- **LLM Integration**: Connects to a local Ollama instance for content summarization and translation.
- **Configurable**: Settings like model names, paths, and generation parameters are managed through a `config.yaml` file.
- **Caching**: Caches intermediate results to speed up subsequent runs on the same file.
- **Modular Steps**: Allows you to choose which processing steps to run (`summary`, `translate`, `original`).

## Setup

### 1. Install Dependencies
```sh
uv pip install -r requirements.txt
```

### 2. Configure Ollama
Ensure you have Ollama running and have downloaded the required models specified in your configuration.

### 3. Create Configuration File
Create a `config.yaml` file in the root of the project. You can use the provided `config.yaml` as a template. Update the paths to match your system.

```yaml
ollama_host: http://127.0.0.1:11434
chat_model_name: qwen3:30b-a3b-thinking-2507-q4_K_M
vision_model_name: qwen2.5vl:7b
max_context_token_num: 16000
cache_data_dir: /path/to/your/cache

parser_config_file_path: /path/to/your/paper_reading/magic-pdf.json
asset_save_dir: /path/to/your/attachments

gen_conf:
    temperature: 0.7
    top_p: 0.4
    repeat_penalty: 1.2
    num_ctx: 30000
```

## Usage

You can run the processing pipeline using the `main.py` script. The `run.sh` script provides a convenient way to execute it.

### Example from `run.sh`
```sh
#!/usr/bin/env bash
set -e
cd $(dirname "$0")

file_path='/path/to/your/paper.pdf'
temp_content_dir="./tmp/parsed_assets"
final_md_file_save_dir="/path/to/your/output_markdowns"

python main.py \
    --file_path="${file_path}" \
    --temp_content_dir="${temp_content_dir}" \
    --final_md_file_save_dir="${final_md_file_save_dir}"
```

### Command-Line Arguments

The `main.py` script accepts the following arguments:

- `--file_path`: Path to the input PDF file.
- `--temp_content_dir`: Directory for temporary files created during parsing.
- `--final_md_file_save_dir`: Directory where the final Markdown file will be saved.
- `--src_lang`: Source language of the paper (default: `en`).
- `--target_lang`: Target language for translation (default: `zh`).
- `--steps`: Comma-separated list of processing steps to perform. Default is `summary,original,translate`.
  - `summary`: Generate a summary of the paper.
  - `original`: Include the original parsed content.
  - `translate`: Translate the content.

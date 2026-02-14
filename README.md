# 论文阅读处理流程

本仓库包含一个基于 Python 的 PDF 文档处理流程，特别是用于处理学术论文。它使用 `MinerU` 来解析 PDF，提取文本、图像和表格。然后使用本地 Ollama 实例来总结和翻译内容，最终将结果保存为 Markdown 文件。

## 功能特性

- **PDF 解析**：利用 `MinerU` 准确解析 PDF 布局，包括文本、图像和表格。
- **LLM 集成**：连接到本地 Ollama 实例进行内容总结和翻译。
- **可配置**：模型名称、路径和生成参数等设置通过 `config.yaml` 文件管理。
- **缓存机制**：缓存中间结果以加快对同一文件的后续运行。
- **模块化步骤**：允许您选择要运行的处理步骤（`summary`、`translate`、`original`）。

## 快速开始

### 1. 安装依赖
```sh
uv pip install -e .
```

### 2. 配置 Ollama
确保 Ollama 正在运行，并已下载配置中指定的必需模型。

### 3. 下载 MinerU 模型
```sh
python download_mineru_model.py
```

### 4. 处理 PDF
```sh
python process_pdf.py \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output
```

## 使用方法

项目包含多个脚本处理不同的任务：

### 1. 下载 MinerU 模型

`download_mineru_model.py` - 从 HuggingFace 或 ModelScope 下载 MinerU 解析模型。

```sh
python download_mineru_model.py
```

**功能：**
- 从 HuggingFace 或 ModelScope 下载 MinerU 的解析模型
- 支持通过 `MINERU_MODEL_SOURCE` 环境变量切换数据源（`huggingface` 或 `modelscope`）
- 自动生成并修改 `magic-pdf.json` 配置文件

### 2. 提取 PDF 页面

`extract_pages.py` - 从 PDF 中提取指定页面。

```sh
python extract_pages.py --config extract_pages.yaml
```

**功能：**
- 从 PDF 中提取指定页面范围
- 支持 YAML 配置文件定义提取规则
- 支持页码范围语法（如 `1,3,5-7` 表示第 1、3、5-7 页）

### 3. 单个 PDF 处理

`process_pdf.py` - 处理单个 PDF 文件进行解析、总结和翻译。

您可以直接运行 `process_pdf.py` 脚本：

```sh
python process_pdf.py \
    --file_path /path/to/paper.pdf \
    --final_md_file_save_dir /path/to/output \
    --src_lang en \
    --target_lang zh \
    --steps original,summary,translate
```

**命令行参数：**

- `--file_path`：输入 PDF 文件的路径。
- `--temp_content_dir`：用于存储解析过程中创建的临时文件的目录（默认：`./tmp/parsed_asset`）。
- `--final_md_file_save_dir`：最终 Markdown 文件的保存目录。
- `--src_lang`：论文的源语言（默认：`en`）。
- `--target_lang`：翻译的目标语言（默认：`zh`）。
- `--steps`：要执行的处理步骤列表，用逗号分隔。默认为 `summary,translate,original`。
  - `summary`：生成论文摘要。
  - `original`：包含原始解析的内容。
  - `translate`：翻译内容。

### 4. 批量处理 PDF

`process_batch_pdf.py` - 批量处理多个 PDF 文件。

```sh
python process_batch_pdf.py
```

**功能：**
- 批量处理 PDF 文件夹中的多个文件
- 跟踪已处理的文件，避免重复处理
- 自动调用 `process_pdf.py` 处理每个文件

需要在脚本中配置：
- `pdf_folder`：输入 PDF 文件夹路径
- `output_folder`：输出 Markdown 文件夹路径
- `steps`：处理步骤

## 配置

### 主配置文件：`config.yaml`

在项目根目录创建 `config.yaml` 文件。您可以使用提供的 `config.yaml` 作为模板。更新路径以匹配您的系统。

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

### PDF 页面提取配置：`extract_pages.yaml`

用于配置页面提取规则（参考项目中的 `extract_pages.yaml`）

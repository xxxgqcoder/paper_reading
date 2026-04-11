使用 uv + pyproject.toml 管理依赖和运行环境

在创建PR前，检查项目内的.agents/目录，确保其中的skills文档是最新的

## CLI 命令

### 模型下载

首次使用公式识别功能前，需预下载模型（避免运行时卡死）：

```bash
# 下载所有模型（推荐）
uv run paper-reading download-models

# 下载特定模型
uv run paper-reading download-models --model code-formula
```

### PDF 处理

```bash
# 基本用法
uv run paper-reading --file_path input.pdf --final_md_file_save_dir output/

# 使用配置文件
uv run paper-reading --config config.json

# 指定步骤
uv run paper-reading --file_path input.pdf \
  --final_md_file_save_dir output/ \
  --steps summary,translate,original
```

### 页面提取

```bash
# 提取指定页面
uv run paper-reading extract-pages \
  --input_pdf input.pdf \
  --pages 1-10 15,20-25 \
  --output_dir output/
```

### Skill 安装

```bash
# 安装到 ~/.agents/skills/
uv run paper-reading install-skills

# 卸载
uv run paper-reading install-skills --uninstall
```

### 查看配置 Schema

```bash
uv run paper-reading get-schema
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HF_HOME` | HuggingFace 缓存目录 | `~/.cache/huggingface` |
| `ODL_OMP_THREADS` | OpenDataLoader CPU 线程数 | `10` |
| `ODL_PARSE_TIMEOUT` | PDF 解析超时秒数 | `3600` (1小时) |
| `ODL_VOLUME_HOST_DIR` | OpenDataLoader 数据卷挂载目录 | - |
| `PR_LLM_ENDPOINT` | LLM API 端点 | - |
| `PR_LLM_API_KEY` | LLM API 密钥 | - |

## 完整工作流示例

```bash
# 1. 首次安装：下载模型
uv run paper-reading download-models

# 2. 配置环境变量（可选）
export ODL_OMP_THREADS=10          # CPU 线程优化
export PR_LLM_ENDPOINT="https://..."
export PR_LLM_API_KEY="sk-..."

# 3. 处理学术论文（启用公式识别）
uv run paper-reading \
  --file_path paper.pdf \
  --final_md_file_save_dir output/ \
  --steps summary,translate,original \
  --src_lang en \
  --target_lang zh

# 4. 安装为全局 skill（供 AI Agent 使用）
uv run paper-reading install-skills
```

## JSON 输出格式

所有命令都输出 JSON 格式结果，方便作为 skill 集成：

**download-models 输出示例**：
```json
{
  "status": "completed",
  "downloaded": 1,
  "total": 1,
  "results": [
    {
      "model": "code-formula",
      "repo_id": "docling-project/CodeFormulaV2",
      "status": "success",
      "path": "/Users/.../.cache/huggingface/hub/models--docling-project--CodeFormulaV2"
    }
  ],
  "cache_dir": "/Users/.../.cache/huggingface"
}
```

**process 输出示例**：
```json
{
  "status": "success",
  "output_file": "output/paper.md",
  "steps_completed": ["summary", "translate", "original"],
  "elapsed_seconds": 123.45
}
```

在创建PR前，检查项目内的.agents/目录，确保其中的skills文档是最新的 



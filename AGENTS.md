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
# 默认只解析原文为 Markdown，不启用 AI 总结/翻译
uv run paper-reading parse --file_path input.pdf --output_dir output/

# 使用配置文件
uv run paper-reading --config config.json

# 用户明确要求 AI 总结时才指定 summary，并通过环境变量提供 LLM 配置
uv run paper-reading parse --file_path input.pdf \
  --output_dir output/ \
  --steps summary,original
```

### 目录检查与页面提取

```bash
# 先读取目录和前置页，核验 pdf_page = printed_page + offset
uv run paper-reading inspect --file_path input.pdf --inspect_pages 20

# 再按印刷页码和已核验 offset 提取
uv run paper-reading extract \
  --file_path input.pdf \
  --pages 1-10 15,20-25 \
  --page_number_offset 12 \
  --output_dir output/
```

按 chapter 提取时，Agent 必须读取目录页，并用前几页中至少两个页码证据核验印刷页码与 PDF 物理页码的 gap；低置信度或扫描版应目视复核。

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
| `PR_LLM_ENDPOINT` | LLM API 端点 | - |
| `PR_LLM_API_KEY` | LLM API 密阥 | - |
| `MINERU_MODEL_SOURCE` | 模型下载源 `huggingface`/`modelscope` | `modelscope` |
| `MINERU_DEVICE_MODE` | MineRU 推理设备，空时自动检测 | 自动检测 |

## 完整工作流示例

```bash
# 1. 首次安装：下载模型
uv run paper-reading download-models

# 2. 仅在明确需要 AI 总结/翻译时配置环境变量（不要泄露 Key）
export PR_LLM_ENDPOINT="https://..."
export PR_LLM_API_KEY="sk-..."
export MINERU_MODEL_SOURCE=modelscope  # 默认 modelscope，国内网络推荐

# 3. 处理学术论文（默认仅 original；device 自动检测）
uv run paper-reading parse \
  --file_path paper.pdf \
  --output_dir output/

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

**parse 输出示例**：
```json
{
  "status": "success",
  "output_file": "output/paper.md",
  "action": "parse",
  "steps_completed": ["original"],
  "elapsed_seconds": 123.45
}
```

在创建PR前，检查项目内的.agents/目录，确保其中的skills文档是最新的 


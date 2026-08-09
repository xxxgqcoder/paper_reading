# 开发指引

## 环境与依赖

统一使用 `uv`、`pyproject.toml` 和 `uv.lock` 管理依赖与运行环境，不新增其他环境管理方式。

## 产品行为约束

PDF 解析默认只执行 `original`，AI 总结或翻译必须由用户显式启用；按章节提取必须先检查目录和页码偏移，具体使用契约见 [`README.md`](README.md) 与 [`.agents/skills/pdf-reading-skill/SKILL.md`](.agents/skills/pdf-reading-skill/SKILL.md)。

## CLI 与 Skill 安装

仅在用户明确要求安装或更新时，按 [`.agents/skills/pdf-reading-skill/SKILL.md`](.agents/skills/pdf-reading-skill/SKILL.md) 从当前 checkout 安装并验收全局 CLI 与 Skill，不以 `uv run ... install-skills` 代替全局 CLI 安装。

## 验证

代码或 Skill 变更后至少运行 `uv run ruff check paper_reading tests`、`uv run pytest -q` 和 `git diff --check`；涉及 Skill 打包时还要构建 wheel 并确认其中包含最新 `pdf-reading-skill/SKILL.md`。

## PR 门禁

创建 PR 前确认 `.agents/skills/`、README 和实际 CLI 行为一致，并保留无关工作区改动。

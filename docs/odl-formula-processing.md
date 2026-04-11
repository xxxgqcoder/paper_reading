# OpenDataLoader 公式处理配置详解

## 概览

OpenDataLoader 使用 **CodeFormulaV2** 视觉语言模型 (VLM) 来识别和提取数学公式及代码块。

## 默认配置

### 容器启动参数

```bash
opendataloader-pdf-hybrid --host 0.0.0.0 --port 5002 --enrich-formula
```

- `--enrich-formula`: 启用公式提取功能
- 默认禁用：`--enrich-picture-description`（图片描述功能）

### PDF Pipeline 默认设置

```python
PdfPipelineOptions:
  do_code_enrichment: False          # 默认禁用（需通过 --enrich-formula 启用）
  do_formula_enrichment: False       # 默认禁用（需通过 --enrich-formula 启用）
  do_picture_classification: False   # 图片分类
  do_table_structure: True           # 表格结构识别（默认启用）
  images_scale: 1.0                  # 图像缩放比例
  generate_page_images: False        # 不生成页面图像
  generate_picture_images: False     # 不生成独立图片文件
```

**注意**：`--enrich-formula` 启动参数会在运行时动态启用 `do_code_enrichment` 和 `do_formula_enrichment`。

## CodeFormulaV2 模型配置

### 模型规格

| 属性 | 值 |
|------|-----|
| **模型名称** | CodeFormulaV2 |
| **HuggingFace Repo** | `docling-project/CodeFormulaV2` |
| **模型大小** | 0.3B 参数（~617MB 下载） |
| **输入分辨率** | 120 DPI |
| **推理引擎** | Transformers (auto_inline) |
| **数据类型** | bfloat16 |
| **Max New Tokens** | 4096 |

### 功能配置

```python
CodeFormulaVlmOptions:
  extract_code: True              # 识别代码块
  extract_formulas: True          # 识别数学公式
  scale: 2.0                      # 图像预处理缩放因子（2倍）
  max_size: None                  # 无最大尺寸限制
  engine_type: auto_inline        # 自动选择推理引擎
  prefer_vllm: False              # 不优先使用 vLLM
```

### 模型行为

**代码块识别**：
- 识别编程语言
- 保留缩进格式
- 输出格式：`<_<语言>_> <代码内容>`
- 示例：`<_Java_> System.out.println("Hello World.");`

**公式识别**：
- 生成对应的 LaTeX 代码
- 输出纯文本格式（不含特殊标记）

## 性能特征

### 批处理设置

- **批大小**: 5 个图像/批次（固定）
- **处理方式**: 串行批处理（非并行）
- **CPU 线程数**: 默认 4（通过 `OMP_NUM_THREADS` 控制）

### 实测性能（CPU - 12核 Mac）

从日志中观察到的实际性能：

| 批次 | 处理时间 | 平均耗时/图像 |
|------|---------|--------------|
| 1 | 123.03s | 24.61s |
| 2 | 173.28s | 34.66s |
| 3 | 197.76s | 39.55s |
| 4 | 497.91s | 99.58s |
| 5 | 361.85s | 72.37s |
| 6 | 443.43s | 88.69s |
| **平均** | **299.54s** | **59.91s (~1分钟/公式)** |

### 性能瓶颈分析

**为什么这么慢？**

1. **VLM 架构**：CodeFormulaV2 是 ImageTextToText 多模态模型，需要处理图像编码器 + 文本解码器
2. **CPU 推理**：无 GPU 加速，所有矩阵运算在 CPU 上串行执行
3. **bfloat16 模拟**：CPU 不原生支持 bfloat16，需要软件模拟
4. **顺序生成**：生成最多 4096 个 token，逐个 token 生成（自回归）
5. **批处理限制**：虽然每批 5 个图像，但模型内部仍是逐个处理
6. **线程数限制**：默认只用 4 个 CPU 线程，未充分利用多核

## 性能优化建议

### 1. 增加 CPU 线程数（已实现）

```bash
# 方法 1: 环境变量（推荐）
export ODL_OMP_THREADS=10  # 使用 10 个线程（12核系统）
docker restart opendataloader-api-server

# 方法 2: 容器启动时设置
docker run -d \
  -e OMP_NUM_THREADS=10 \
  ...
```

**预期提升**：20-40% 速度提升（4线程 → 10线程）

### 2. 增加超时时间（已实现）

```python
# paper_reading/pdf_parser.py
parse_timeout = int(os.getenv("ODL_PARSE_TIMEOUT", "3600"))  # 默认 1 小时
```

**用途**：避免大文档处理中途超时

### 3. 选择性启用公式识别

**场景 1：批量处理不含公式的文档**
```bash
# 禁用公式识别（纯文本提取）
docker run -d -e ENRICHMENTS="" ...
```

**场景 2：仅对重要文档启用**
```python
# 在 paper-reading 中添加条件判断
if document_needs_formula_extraction:
    use_formula_enrichment = True
```

### 4. GPU 加速（如果可用）

**云服务器**:
```bash
# NVIDIA GPU 容器
docker run --gpus all -d ...
```

**预期提升**：10-50 倍速度（取决于 GPU 型号）

### 5. 替代方案

**轻量级方案**（不需要 LaTeX 精确度）：
- 使用 OCR 直接识别公式文本（快速但不准确）
- 使用 pdfplumber/pymupdf 提取文本（无法识别图像化公式）

**混合方案**：
- 大批量：禁用公式识别，快速提取文本
- 精选文档：启用公式识别，获取高质量 LaTeX

## 预期处理时间

基于实测性能（CPU，4 线程默认配置）：

| 文档类型 | 公式数量 | 预计时间 |
|---------|---------|----------|
| 普通论文 (9页) | 20-30 | 20-30 分钟 |
| 公式密集 (9页) | 40-50 | 40-50 分钟 |
| 教材章节 (50页) | 100-150 | 2-3 小时 |
| 完整教材 (1000页) | 2000-3000 | **2-3 天** |

**优化后（10 线程）**：
- 普通论文：12-18 分钟
- 公式密集：24-30 分钟
- 教材章节：1-2 小时

## 常见问题

### Q1: 为什么不使用更小的模型？

CodeFormulaV2 已经是专门优化过的轻量级模型（0.3B 参数）。更小的模型无法保证公式识别准确度。

### Q2: 能否并行处理多个 PDF？

可以启动多个容器实例（不同端口），但需注意：
- 每个容器占用 ~2GB 内存
- CPU 核心数限制并行度
- 建议：最多 CPU 核心数 / 10 个并发实例

### Q3: 批大小能否调整？

批大小固定为 5，由 docling 内部控制，无法通过外部参数调整。

### Q4: 如何监控处理进度？

```bash
# 实时查看日志
docker logs -f opendataloader-api-server | grep "Batch processed"

# 统计已处理图像数
docker logs opendataloader-api-server | grep "Batch processed" | wc -l
```

## 总结

OpenDataLoader 的公式处理功能：

✅ **优点**：
- 高精度 LaTeX 输出
- 支持代码块识别
- 离线运行，无需外部 API
- 模型缓存，避免重复下载

⚠️ **缺点**：
- CPU 推理极慢（~1分钟/公式）
- 无法调整批大小
- 处理大文档需要数小时
- 默认 4 线程限制性能

🎯 **最佳实践**：
1. 增加 CPU 线程数到 10-12
2. 设置足够长的超时时间（1-2 小时）
3. 仅对需要精确公式的文档启用
4. 大批量处理考虑使用 GPU 服务器
5. 预先下载模型缓存，避免首次运行卡死

## 参考资料

- [HuggingFace Model Card](https://huggingface.co/docling-project/CodeFormulaV2)
- [Docling Technical Report](https://arxiv.org/abs/2408.09869)
- [SmolDocling Paper](https://arxiv.org/abs/2503.11576)
- [OpenDataLoader GitHub](https://github.com/docling-project/docling)

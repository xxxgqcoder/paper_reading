import asyncio
import os
import sys
from pathlib import Path

# 将当前目录加入 Python 路径以导入 paper_reading
sys.path.append(os.getcwd())

from paper_reading.config import ProcessParams, GenerationConfig
from paper_reading.pipeline import process
from paper_reading.log import Logger

async def run_test():
    # 1. 基础路径配置
    # 注意：确保 odl_volume_host_dir 与你 docker run 时挂载到 /data 的宿主机目录一致
    pdf_path = "/Users/xcoder/aDrive/PDF/Attention Is All You Need.pdf"
    output_dir = "/Users/xcoder/projects/paper_reading/assets"
    asset_dir = "/Users/xcoder/projects/paper_reading/assets"
    volume_host_dir = "/Users/xcoder/aDrive/PDF"

    # 2. 如果输出文件已存在，先删除（process 内部会校验文件是否存在）
    name_without_suff = os.path.basename(pdf_path).rsplit(".", 1)[0]
    md_file_path = os.path.join(output_dir, f"{name_without_suff}.md")
    if os.path.exists(md_file_path):
        Logger.info(f"Removing existing output: {md_file_path}")
        os.remove(md_file_path)

    # 3. 构造参数
    # 我们使用 hybrid_mode='full' 来重试之前 OOM 的任务
    params = ProcessParams(
        file_path=pdf_path,
        output_dir=output_dir,
        asset_save_dir=asset_dir,
        steps=["original"], # 先只运行解析步骤
        odl_hybrid_mode="full",
        odl_volume_host_dir=volume_host_dir,
        # LLM 参数（如果只运行 original 步骤，这些参数不会被用到，但仍需按 Schema 填充）
        llm_endpoint=os.environ.get("PR_LLM_ENDPOINT", "http://localhost:11434/v1"),
        llm_api_key=os.environ.get("PR_LLM_API_KEY", "dummy"),
    )

    Logger.info("Starting OpenDataLoader test with hybrid_mode='full'...")
    try:
        result = await process(params)
        Logger.info(f"Test completed successfully!")
        Logger.info(f"Markdown saved to: {result.output_file}")
        Logger.info(f"Total time: {result.elapsed_seconds}s")
    except Exception as e:
        Logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())

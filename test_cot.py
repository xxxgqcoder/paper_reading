import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from paper_reading.config import ProcessParams
from paper_reading.log import Logger
from paper_reading.pipeline import process

PDF_PATH = "/Users/xcoder/aDrive/PDF/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.pdf"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_DIR = OUTPUT_DIR
VOLUME_HOST_DIR = "/Users/xcoder/aDrive/PDF"


async def run():
    name_stem = os.path.basename(PDF_PATH).rsplit(".", 1)[0]
    md_file = os.path.join(OUTPUT_DIR, f"{name_stem}.md")
    if os.path.exists(md_file):
        Logger.info(f"Removing existing output: {md_file}")
        os.remove(md_file)

    params = ProcessParams(
        file_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        asset_save_dir=ASSET_DIR,
        steps=["original"],
        odl_hybrid_mode="auto",
        odl_volume_host_dir=VOLUME_HOST_DIR,
        llm_endpoint=os.environ.get("PR_LLM_ENDPOINT", "http://localhost:11434/v1"),
        llm_api_key=os.environ.get("PR_LLM_API_KEY", "dummy"),
    )

    Logger.info("Starting test: Chain-of-Thought paper, hybrid_mode=auto ...")
    result = await process(params)
    Logger.info(f"Test completed! Markdown: {result.output_file}")
    Logger.info(f"Total time: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    asyncio.run(run())

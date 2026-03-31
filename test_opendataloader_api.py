"""
通过 docker exec 在容器内部调用 opendataloader-pdf CLI 进行 PDF 提取。

容器内已包含:
  - Java 17 (运行 opendataloader-pdf-cli.jar)
  - Python Hybrid API 后端 (localhost:5002 提供 AI 精排能力)

宿主机无需安装 Java，输出结果通过挂载目录映射回宿主机。
"""

import subprocess
import os
import shutil

# --- 配置 ---
CONTAINER_NAME = "opendataloader-api-server"
PDF_HOST_PATH = "/Users/xcoder/aDrive/PDF/Attention Is All You Need.pdf"
PDF_FILENAME = os.path.basename(PDF_HOST_PATH)

# 容器内挂载路径
PDF_CONTAINER_PATH = f"/data/{PDF_FILENAME}"
OUTPUT_CONTAINER_DIR = "/data/output"

# 宿主机输出路径（通过 volume 映射）
PDF_VOLUME_DIR = "/Users/xcoder/aDrive/PDF"
OUTPUT_HOST_DIR = os.path.join(PDF_VOLUME_DIR, "output")
PROJECT_OUTPUT_DIR = "/Users/xcoder/projects/paper_reading/tmp/test_opendataloader"


def run_docker_exec(cmd: list[str]) -> subprocess.CompletedProcess:
    """在指定容器内执行命令"""
    return subprocess.run(
        ["docker", "exec", CONTAINER_NAME] + cmd,
        capture_output=True,
        text=True,
    )


def main():
    # 1. 检查容器是否运行
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != "true":
        print(f"Error: Container '{CONTAINER_NAME}' is not running.")
        print("Please start it with: docker run -d -p 5002:5002 -v /Users/xcoder/aDrive/PDF:/data --name opendataloader-api-server opendataloader-api-server")
        return

    print(f"Container '{CONTAINER_NAME}' is running.")

    # 2. 在容器内运行 opendataloader-pdf CLI（使用 Hybrid 模式）
    # 容器内同时运行着 Hybrid API，直接使用 localhost:5002
    print(f"Extracting: {PDF_FILENAME}")
    print("Mode: Hybrid (docling-fast) — AI-powered table/formula extraction enabled")
    result = run_docker_exec([
        "opendataloader-pdf",
        PDF_CONTAINER_PATH,
        "--output-dir", OUTPUT_CONTAINER_DIR,
        "--format", "markdown,json",
        "--hybrid", "docling-fast",
        "--hybrid-mode", "full",   # 启用公式+图表描述（最高精度）
        "--hybrid-url", "http://localhost:5002",
    ])

    if result.returncode != 0:
        print(f"Extraction failed (exit code {result.returncode}):")
        print(result.stderr)
        return

    if result.stdout:
        print(result.stdout)

    # 3. 将结果复制到项目目录
    if os.path.exists(OUTPUT_HOST_DIR):
        os.makedirs(PROJECT_OUTPUT_DIR, exist_ok=True)
        for f in os.listdir(OUTPUT_HOST_DIR):
            src = os.path.join(OUTPUT_HOST_DIR, f)
            dst = os.path.join(PROJECT_OUTPUT_DIR, f)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        print(f"\nExtraction complete! Files saved to: {PROJECT_OUTPUT_DIR}")
        for f in sorted(os.listdir(PROJECT_OUTPUT_DIR)):
            size = os.path.getsize(os.path.join(PROJECT_OUTPUT_DIR, f))
            print(f"  {f}  ({size:,} bytes)")
    else:
        print(f"Warning: Output directory '{OUTPUT_HOST_DIR}' not found after extraction.")


if __name__ == "__main__":
    main()

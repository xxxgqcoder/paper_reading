import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import Content, ContentType
from .log import Logger
from .utils import cache_it, hash64, load_base64_image, safe_encode, time_it

# 图片 bbox 包含判断的容差（PDF 坐标单位）
_BBOX_TOLERANCE = 5.0


def _bbox_contains(outer: list[float], inner: list[float], tolerance: float = _BBOX_TOLERANCE) -> bool:  # noqa: E501
    """判断 inner bbox 是否被 outer bbox 包含（含容差）。

    bbox 格式：[x0, y0, x1, y1]，PDF 坐标系（y 轴向上）。
    """
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


class OpenDataLoaderParser:
    """PDF 解析器，通过 Docker 容器运行 OpenDataLoader。

    宿主机无需安装 Java，通过 docker exec 在容器内执行解析，
    输出文件通过挂载目录映射回宿主机。
    """

    def __init__(
        self,
        container_name: str = "opendataloader-api-server",
        volume_host_dir: str = "",
        hybrid_mode: str = "auto",
    ):
        self.container_name = container_name
        self.volume_host_dir = volume_host_dir
        self.hybrid_mode = hybrid_mode

    def key_generator(self, file_path: str, **kwargs) -> str:
        file_bytes = b""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            Logger.error(f"Read file exception: {e}")
            return uuid.uuid4().hex
        return "odl_parser::file_content_hash::" + hash64(file_bytes)

    @time_it("pdf parser")
    @cache_it(key_generator=key_generator)
    def parse(self, file_path: str, asset_save_dir: str = "") -> list[Content]:
        """解析 PDF 文件，返回 Content 列表。

        通过 docker exec 在容器内调用 opendataloader-pdf CLI，
        读取 JSON 输出文件并转换为 Content 列表。
        图片以 UUID 命名保存至 asset_save_dir，content_url 存储绝对路径。
        """
        if asset_save_dir:
            os.makedirs(asset_save_dir, exist_ok=True)

        pdf_filename = os.path.basename(file_path)
        pdf_stem = Path(file_path).stem

        Logger.info(f"Parsing PDF via OpenDataLoader container: {self.container_name}")
        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container_name,
                "opendataloader-pdf",
                f"/data/{pdf_filename}",
                "--output-dir",
                "/data/output",
                "--format",
                "json",
                "--hybrid",
                "docling-fast",
                "--hybrid-mode",
                self.hybrid_mode,
                "--hybrid-url",
                "http://localhost:5002",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"OpenDataLoader extraction failed"
                f" (exit {result.returncode}): {result.stderr}"
            )
        Logger.info(f"OpenDataLoader stdout: {result.stdout[:300]}")

        json_host_path = os.path.join(
            self.volume_host_dir, "output", f"{pdf_stem}.json"
        )
        Logger.info(f"Reading JSON result from: {json_host_path}")
        with open(json_host_path, encoding="utf-8") as f:
            doc = json.load(f)

        elements: list[dict[str, Any]] = doc.get("kids", [])
        output_dir = os.path.join(self.volume_host_dir, "output")

        return self._process_elements(
            elements=elements,
            output_dir=output_dir,
            asset_save_dir=asset_save_dir,
            file_path=file_path,
        )

    def _process_elements(
        self,
        elements: list[dict[str, Any]],
        output_dir: str,
        asset_save_dir: str,
        file_path: str,
    ) -> list[Content]:
        """将 OpenDataLoader JSON 元素列表转换为 Content 列表。

        - 使用 bbox 包含判断跳过图片内部的 OCR 文字
        - 图片以 UUID 命名复制至 asset_save_dir
        - content_url 保存绝对路径
        """
        # 第一遍：收集每页所有图片的 bbox，用于后续过滤图片内文字
        image_bboxes_by_page: dict[int, list[list[float]]] = {}
        for elem in elements:
            if elem.get("type") == "image":
                page = elem.get("page number", 0)
                bbox = elem.get("bounding box")
                if bbox:
                    image_bboxes_by_page.setdefault(page, []).append(bbox)

        contents: list[Content] = []
        for elem in elements:
            elem_type = elem.get("type", "")
            page = elem.get("page number", 0)
            bbox = elem.get("bounding box")

            if elem_type == "image":
                source = elem.get("source", "")
                if not source:
                    continue
                content = self._process_image(
                    source=source,
                    output_dir=output_dir,
                    asset_save_dir=asset_save_dir,
                    file_path=file_path,
                )
                if content:
                    contents.append(content)

            elif elem_type in ("paragraph", "heading", "formula", "list"):
                # 跳过位于图片 bbox 内的文字（过滤图片中的 OCR 文字）
                if bbox and page in image_bboxes_by_page and any(
                    _bbox_contains(img_bbox, bbox)
                    for img_bbox in image_bboxes_by_page[page]
                ):
                    Logger.info(
                        f"Skipping in-image text on page {page}:"
                        f" {elem.get('content', '')[:60]!r}"
                    )
                    continue

                if elem_type == "list":
                    text = self._list_to_text(elem)
                else:
                    text = elem.get("content", "").strip()
                    if elem_type == "heading":
                        level = max(1, int(elem.get("heading level", 2)))
                        text = "#" * level + " " + text

                if not text:
                    continue
                contents.append(
                    Content(
                        content_type=ContentType.TEXT,
                        file_path=file_path,
                        content=safe_encode(text),
                        extra_description="",
                        content_url="",
                    )
                )

            elif elem_type == "table":
                table_md = self._table_to_markdown(elem)
                if table_md:
                    contents.append(
                        Content(
                            content_type=ContentType.TABLE,
                            file_path=file_path,
                            content=safe_encode(table_md),
                            extra_description="",
                            content_url="",
                        )
                    )

        return contents

    def _process_image(
        self,
        source: str,
        output_dir: str,
        asset_save_dir: str,
        file_path: str,
    ) -> Content | None:
        """处理图片元素：以 UUID 命名复制到 asset_save_dir，返回 Content 对象。"""
        src_path = os.path.join(output_dir, source)
        if not os.path.isfile(src_path):
            Logger.warning(f"Image file not found: {src_path}")
            return None

        ext = Path(source).suffix
        with open(src_path, "rb") as _f:
            img_bytes = _f.read()
        # 基于图片内容生成确定性文件名，相同内容始终得到相同文件名
        content_hash = hash64(img_bytes)
        new_name = content_hash + ext
        dst_path = os.path.join(asset_save_dir, new_name)
        shutil.copyfile(src_path, dst_path)
        Logger.info(f"Saved image with content-hash name: {dst_path}")

        return Content(
            content_type=ContentType.IMAGE,
            file_path=file_path,
            content=load_base64_image(src_path),
            extra_description="",
            content_url=dst_path,
        )

    def _list_to_text(self, list_elem: dict[str, Any]) -> str:
        """将 list 元素转换为 Markdown 列表文本。"""
        items = list_elem.get("list items", [])
        lines = []
        for item in items:
            text = item.get("content", "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def _table_to_markdown(self, table_elem: dict[str, Any]) -> str:
        """将 table 元素转换为 Markdown 表格文本。"""
        rows = table_elem.get("rows", [])
        if not rows:
            return ""

        md_rows = []
        for row in rows:
            cells = row.get("cells", [])
            cell_texts = []
            for cell in cells:
                kids = cell.get("kids", [])
                text = " ".join(k.get("content", "") for k in kids if k.get("content"))
                cell_texts.append(text.replace("|", "\\|"))
            md_rows.append("| " + " | ".join(cell_texts) + " |")

        if not md_rows:
            return ""

        # 在标题行后插入分隔行
        num_cols = len(rows[0].get("cells", []))
        separator = "| " + " | ".join(["---"] * num_cols) + " |"
        return "\n".join([md_rows[0], separator] + md_rows[1:])

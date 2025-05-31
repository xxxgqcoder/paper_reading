import gc
import re
import shutil
import time
import os
from typing import Dict, Any

from utils import (time_it, safe_strip)
# ==============================================================================
# parse pdf

# class Content():

#     def __init__(
#         self,
#         raw_content: Dict[str, Any],
#         **kwargs,
#     ):
#         """
#         Args:
#         - raw_content: original json content.
#         """
#         for k, v in kwargs.items():
#             setattr(self, k, v)

#     def is_valid_content(self, ) -> bool:

#         pass

#     def is_empty(self, ) -> bool:
#         pass


def is_empty(text: str):
    text = safe_strip(text)
    if text is None:
        return True
    if len(text) == 0:
        return True
    if text == '[]':
        return True
    return False


def post_text_process(text: str) -> str:
    # strip space around $
    p = r"\s*(\$)\s*"
    text = re.sub(p, r"\1", text)

    return text


def is_valid_content(content: Dict[str, Any]) -> bool:
    """
    There are corner cases where returned content dont contain expected keys 
    or values are empty.

    Returns:
    - bool, true if content is valid.
    """
    # missing key
    if 'type' not in content:
        return False

    # text / equation
    if content['type'] in ['text', 'equation']:
        return 'text' in content

    # image
    if content['type'] == 'image':
        return 'img_path' in content and len(content['img_path']) > 0

    # table
    if content['type'] == 'table':
        return 'table_body' in content

    return True


@time_it
def parse_pdf(
    file_path: str,
    asset_dir: str,
    magic_config_path: str,
) -> list[Dict[str, Any]]:
    """
    Parse PDF content and return content list. The result is a list of json 
    oject representing a pdf content block.
    
    Dict object key explanation:
        - `img_caption`: the image caption.
        - `img_footnote`:
        - `img_path`: path to parsed image.
        - `page_idx`: page index.
        - `table_body`: table content in html format.
        - `table_caption`: table caption.
        - `table_footnote`:
        - `text`: the block text content.
        - `text_format`: used in latex forumla block.
        - `text_level`: used in headline block.
        - `type`: block type, can be one of 'equation', 'image', 'table', 'text'.
    
    Typical paper parsed content is organized by list of content block. Headlines
    will stored in one separated block, with `text_level` = 1 while regular content
    block's `text_level` key is missing. Headline blocks are followed by regular
    content block, including `text`, `equation`, `table` and `image` (distinguished 
    by key `type`). All captions are stored in each block's caption key, for 
    example, caption of a parsed image is saved in `img_caption` key of the block.

    Refer [MinerU API demo](https://mineru.readthedocs.io/en/latest/user_guide/usage/api.html) 
    for more details.

    Args:
    - file_path: path to the pdf file
    - asset_dir: folder for saving parsed assets.
    - magic_config_path: magic pdf config path.

    Returns:
    - A list of parsed content block dict.
    """
    # NOTE: magic_pdf package uses singleton design and the model isntance is
    # initialized when the module is imported, so postpone the import statement
    # until parse method is called.
    magic_config_path = os.path.abspath(magic_config_path)
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = magic_config_path
    print(f'setting magic pdf config path to {magic_config_path}')

    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze, ModelSingleton
    from magic_pdf.config.enums import SupportedPdfParseMethod
    from magic_pdf.libs.clean_memory import clean_memory
    from magic_pdf.libs.config_reader import get_device

    # prepare env
    name_without_suff = os.path.basename(file_path).split(".")[0]
    local_image_dir = os.path.join(asset_dir, "images")
    local_md_dir = asset_dir
    image_dir = os.path.basename(local_image_dir)
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)

    # read bytes
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(file_path)
    print(f"{file_path}: read bytes count: {len(pdf_bytes)}")

    # process
    ds = PymuDocDataset(pdf_bytes)

    # inference
    infer_result = ds.apply(doc_analyze,
                            ds.classify() == SupportedPdfParseMethod.OCR)
    pipe_result = infer_result.pipe_txt_mode(image_writer)

    # draw model result on each page
    infer_result.draw_model(
        os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    # get model inference result
    model_inference_result = infer_result.get_infer_res()

    # draw layout result on each page
    pipe_result.draw_layout(
        os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    # draw spans result on each page
    pipe_result.draw_span(
        os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    # get markdown content
    md_content = pipe_result.get_markdown(image_dir)

    # dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    # get content list content
    content_list = pipe_result.get_content_list(image_dir)

    # dump content list
    pipe_result.dump_content_list(md_writer,
                                  f"{name_without_suff}_content_list.json",
                                  image_dir)

    # get middle json
    middle_json_content = pipe_result.get_middle_json()

    # dump middle json
    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

    # clean up memory
    clean_memory(get_device())

    obj = ModelSingleton()
    try:
        del obj
        gc.collect()
    except Exception as e:
        print(f"Exception: {type(e).__name__} - {e}")

    return content_list

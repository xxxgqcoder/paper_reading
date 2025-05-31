import gc
import re
import shutil
import time
import os
import json
import pickle
from typing import Dict, Any

from strenum import StrEnum

from utils import (
    time_it,
    safe_strip,
    run_once,
)


class ContentType(StrEnum):
    TEXT = "text"
    EQUATION = "equation"
    IMAGE = "image"
    TABLE = "table"


# parse pdf
class Content():

    def __init__(
        self,
        raw_content: Dict[str, Any],
        **kwargs,
    ):
        """
        Args:
        - raw_content: original json content.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.raw_content = raw_content
        # set attribute
        all_keys = [
            'img_caption',
            'img_footnote',
            'img_path',
            'page_idx',
            'table_body',
            'table_caption',
            'table_footnote',
            'text',
            'text_format',
            'text_level',
            'type',
        ]
        for k in all_keys:
            if k in raw_content:
                setattr(self, k, raw_content[k])
            else:
                setattr(self, k, '')
        missing = [k for k in raw_content if k not in all_keys]
        if missing:
            print(f'keys found in raw content but not key list: {missing}')

        self.type = ContentType(raw_content.get('type', 'unknown'))

    @run_once
    def is_valid(self, ) -> bool:
        """
        There are corner cases where returned content dont contain expected keys 
        or values are empty.

        Returns:
        - bool, true if content is valid.
        """
        if self.raw_content is None:
            self.invalid_reason = 'missing raw content'
            return False

        # image
        if self.type == ContentType.IMAGE:
            if len(self.img_path) < 1:
                self.invalid_reason = "img_path empty"
                return False

        # table
        if self.type == ContentType.TABLE:
            if len(self.table_body) < 1:
                self.invalid_reason = "empty table body"
                return False

        return True

    def __str__(self, ) -> str:
        if not self.is_valid():
            return 'content is invaid\n' \
                + f"invalid reason: {self.valid_reason}\n" \
                + f"original content: {json.dumps(self.raw_content, indent=4)}"

        if self.type == ContentType.TEXT or self.type == ContentType.EQUATION:
            return self.get('text')

        elif self.type == ContentType.IMAGE:
            return 'content is image \n' \
                + f"image path: {self.img_path} \n" \
                + f"image caption: {self.img_caption} \n" \
                + f"image footnote: {self.img_footnote} \n"

        elif self.type == ContentType.TABLE:
            return 'content is table \n' \
                + f"table body: {self.table_body} \n" \
                + f"table caption: {self.table_caption} \n" \
                + f"table footnote: {self.table_footnote} \n"

        else:
            return f"unregnized content: {json.dumps(self.raw_content, indent=4)}"

    def get(self, key: str, default_val: str = "") -> str:
        if key not in self.__dict__:
            return default_val
        val = self.__dict__[key]
        if val is None:
            return default_val
        return val

    def set(self, key: str, value: str) -> None:
        setattr(self, key, value)


def post_text_process(text: str) -> str:
    # strip space around $
    p = r" *(\$) *"
    text = re.sub(p, r"\1", text)

    return text


@time_it
def parse_pdf(
    file_path: str,
    asset_dir: str,
    magic_config_path: str,
) -> list[Content]:
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

    # save content list
    pickle_content_path = os.path.join(asset_dir, 'content_list.pickle')
    with open(pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
        print(f'save parsed content list to {pickle_content_path}')

    obj = ModelSingleton()
    try:
        del obj
        gc.collect()
    except Exception as e:
        print(f"Exception: {type(e).__name__} - {e}")

    return [Content(raw_content=content) for content in content_list]

import argparse
import os
import pickle
import json
import re
import shutil
import time
import gc
import time
import os
import shutil

from io import TextIOWrapper
from typing import Any, Dict

from ollama import Client
from strenum import StrEnum

line_breaker = '\n\n\n\n'

# global variables
ollama_host = 'http://127.0.0.1:11434'
ollama_model = 'qwen3:30b-a3b'

lang_mapping = {'en': "英语", 'zh': "中文"}
gen_conf = {
    'temperature': 0.1,
    'top_p': 0.3,
    'presence_penalty': 0.4,
    'frequency_penalty': 0.7,
}
translate_prompt = """
<no_think>你是一个翻译助手，请将下面的{src_lang}内容翻译成{target_lang}。

{content}
"""

summary_prompt = """
<no_think>你是一个论文助手，使用{target_lang}语言，总结下面{src_lang}论文内容，总结的内容包括论文主要创新点。

{content}
"""


# ==============================================================================
# util funcs
def time_it(func):

    def wrapper(*kargs, **kwargs):
        begin = time.time_ns()
        ret = func(*kargs, **kwargs)
        elapse = (time.time_ns() - begin) // 1000000
        print(
            f"func {func.__name__} took {elapse // 60000}min {(elapse % 60000)//1000}sec {elapse%60000%1000}ms to finish"
        )

        return ret

    return wrapper


def save_image(src_path: str, dst_dir: str) -> None:
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copyfile(src_path, dst_path)


def safe_strip(raw: str) -> str:
    if raw is None or len(raw) == 0:
        return ''
    raw = str(raw)
    return raw.strip()


def run_once(func):
    has_run = False
    ret = None

    def wrapper(*args, **kwargs):
        nonlocal has_run, ret
        if not has_run:
            has_run = True
            ret = func(*args, **kwargs)
        return ret

    return wrapper


def is_empty(text: str):
    text = safe_strip(text)
    if text is None:
        return True
    if len(text) == 0:
        return True
    if text == '[]':
        return True
    return False


# ==============================================================================
# parser
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


# ==============================================================================
def format_md_image_path(sys_image_folder: str, img_name: str) -> str:
    """
    md image can only be corrct displayed when saved to same folder of the md file,
    reformatted inserted image path for better display.
    """
    return f"![[{os.path.join(os.path.basename(sys_image_folder), img_name)}]]" + line_breaker


# ==============================================================================
# ollama interface
the_ollama_client = None


def get_ollama_client():
    global ollama_host

    if the_ollama_client:
        return the_ollama_client
    return Client(host=ollama_host)


def calculate_dynamic_ctx(history: list[Dict[str, Any]]) -> int:
    """
    Calculate dynamic context window size

    Args:
    - history: conversation history, a list of json objects.

    """

    def count_tokens(text):
        # Simple calculation: 1 token per ASCII character
        # 2 tokens for non-ASCII characters (Chinese, Japanese, Korean, etc.)
        total = 0
        for char in text:
            if ord(char) < 128:  # ASCII characters
                total += 1
            else:  # Non-ASCII characters (Chinese, Japanese, Korean, etc.)
                total += 2
        return total

    # Calculate total tokens for all messages
    total_tokens = 0
    for message in history:
        content = message.get("content", "")
        # Calculate content tokens
        content_tokens = count_tokens(content)
        # Add role marker token overhead
        role_tokens = 4
        total_tokens += content_tokens + role_tokens

    # Apply 1.2x buffer ratio
    total_tokens_with_buffer = int(total_tokens * 1.2)

    if total_tokens_with_buffer <= 8192:
        ctx_size = 8192
    else:
        ctx_multiplier = (total_tokens_with_buffer // 8192) + 1
        ctx_size = ctx_multiplier * 8192

    return ctx_size


def ollama_chat(prompt: str, ) -> str:
    global ollama_host, ollama_model

    ollama_client = get_ollama_client()

    history = [{'role': 'user', 'content': prompt}]
    ctx_size = calculate_dynamic_ctx(history)
    if "max_tokens" in gen_conf:
        del gen_conf["max_tokens"]

    options = {"num_ctx": ctx_size}
    if "temperature" in gen_conf:
        options["temperature"] = gen_conf["temperature"]
    if "max_tokens" in gen_conf:
        options["num_predict"] = gen_conf["max_tokens"]
    if "top_p" in gen_conf:
        options["top_p"] = gen_conf["top_p"]
    if "presence_penalty" in gen_conf:
        options["presence_penalty"] = gen_conf["presence_penalty"]
    if "frequency_penalty" in gen_conf:
        options["frequency_penalty"] = gen_conf["frequency_penalty"]

    response = ollama_client.chat(model=ollama_model,
                                  messages=history,
                                  options=options,
                                  keep_alive=10)

    ans = response["message"]["content"].strip()
    if '</think>' in ans:
        ans = ans.split('</think>')[-1]
    return ans.strip()


# ==============================================================================
# translate func
def translate_table_content(content: Content) -> None:
    table_caption = content.table_caption
    if not is_empty(table_caption):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=table_caption,
        )
        translated_table_caption = ollama_chat(prompt=formatted_prompt)
        content.set('translated_table_caption', translated_table_caption)

    table_footnote = content.table_footnote
    if not is_empty(table_footnote):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=table_footnote,
        )
        translated_table_footnote = ollama_chat(prompt=formatted_prompt)
        content.set('table_footnote', translated_table_footnote)


def translate_image_content(content: Content) -> None:
    img_caption = content.img_caption
    if not is_empty(img_caption):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=img_caption,
        )
        translated_img_caption = ollama_chat(prompt=formatted_prompt)
        content.set('translated_img_caption', translated_img_caption)

    img_footnote = content.img_footnote
    if not is_empty(img_footnote):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=img_footnote,
        )
        translated_img_footnote = ollama_chat(prompt=formatted_prompt)
        content.set('translated_img_footnote', translated_img_footnote)


def translate_text_content(content: Content) -> None:
    text = content.text
    if not is_empty(text):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=text,
        )
        translated_text = ollama_chat(prompt=formatted_prompt)
        content.set('translated_text', translated_text)


@time_it
def translate_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    **kwargs,
):
    """
    Translate contents.
    Args:
    - content_list: a list of content.
    """
    sys_image_folder = kwargs.get(
        'sys_image_folder',
        os.path.expanduser("~/Pictures"),
    )
    print(f'using {sys_image_folder} as sys image save folder')

    print(f'total {len(content_list)} contents')

    for i, content in enumerate(content_list):
        print(f'processing content {i}')
        if content.type == ContentType.TABLE:
            translate_table_content(content)
        elif content.type == ContentType.IMAGE:
            translate_image_content(content)
        elif content.type == ContentType.TEXT:
            translate_text_content(content)
        else:
            print(
                f'unrecognized content: {json.dumps(content.__dict__, indent=4)}'
            )

        print(json.dumps(content.__dict__, indent=4, ensure_ascii=False))
        print('=' * 128)

        # save translated content
        lines = ''
        if not content.is_valid():
            print(f'invalid block: {json.dumps(content.__dict__, indent=4)}')
            continue

        if content.type == ContentType.TEXT:
            text = content.get('translated_text')
            if content.text_level == 1:
                text = '# ' + text
            lines += text + line_breaker

        elif content.type == ContentType.EQUATION:
            lines += content.text + line_breaker

        elif content.type == ContentType.IMAGE:
            # copy image
            abs_img_path = os.path.join(output_dir, content.img_path)
            img_name = os.path.basename(abs_img_path)
            save_image(abs_img_path, sys_image_folder)
            lines += format_md_image_path(sys_image_folder, img_name)
            lines += line_breaker

            # image caption
            img_caption = content.get('translated_img_caption')
            lines += img_caption + line_breaker

            # image footnote
            img_footnote = content.get('translated_img_footnote')
            lines += img_footnote + line_breaker

        elif content.type == ContentType.TABLE:
            table_body = content.table_body
            lines += table_body + line_breaker

            img_path = content.img_path
            if len(img_path) > 0:
                abs_img_path = os.path.join(output_dir, img_path)
                img_name = os.path.basename(abs_img_path)
                save_image(abs_img_path, sys_image_folder)

                lines += format_md_image_path(sys_image_folder, img_name)
                lines += '-' * 8 + line_breaker

            caption = content.get('translated_table_caption')
            lines += caption + line_breaker
        else:
            pass

        lines = post_text_process(lines)
        md_writer.write(lines)

    # save pickle result
    translated_pickle_content_path = os.path.join(
        output_dir, 'translated_content_list.pickle')
    with open(translated_pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
        print(
            f'save translated  content list to {translated_pickle_content_path}'
        )


@time_it
def summary_content(content_list: list[Content]) -> None:
    global summary_prompt, src_lang, target_lang
    print(f'summary_content, src_lang={src_lang}, target_lang={target_lang}')

    full_content = ''
    for content in content_list:
        if content.type == ContentType.TEXT:
            text = content.text
            if content.text_level == 1:
                text = '# ' + text
            full_content += text + line_breaker

        elif content.type == ContentType.EQUATION:
            text = content.text
            full_content += text + line_breaker

        elif content.type == ContentType.IMAGE:
            full_content += str(content.img_caption) + line_breaker
            full_content += str(content.img_footnote) + line_breaker

        elif content.type == ContentType.TABLE:
            full_content += str(content.table_caption) + line_breaker

        else:
            print(f'unrecognized content: {json.dumps(content, indent=4)}')

    formatted_promt = summary_prompt.format(
        src_lang=src_lang,
        target_lang=target_lang,
        content=full_content,
    )
    print(f'formatted prompt: {formatted_promt}')

    summary = ollama_chat(prompt=formatted_promt)
    return summary


def save_summary_of_content(
    md_writer: TextIOWrapper,
    summary: str,
    **kwargs,
):
    md_writer.write('# ' + '=' * 8 + '  Paper summary  ' + '=' * 8 +
                    line_breaker)
    md_writer.write(summary + line_breaker)
    md_writer.write('-' * 8 + line_breaker)


@time_it
def process(
    file_path: str,
    output_dir: str,
    magic_config_path: str,
    sys_image_folder: str,
    final_md_file_save_dir: str,
) -> None:
    """
    Process pdf file

    Args:
    - file_path: path to file.
    - output_dir: output directory.
    - magic_config_path: path to magic pdf parser config.
    - sys_image_folder: image save folder.
    - final_md_file_save_dir: folder for saving final md file.
    """
    os.makedirs(output_dir, exist_ok=True)
    name_without_suff = os.path.basename(file_path).split('.')[0]
    print(f'file name witout out suffix: {name_without_suff}')

    # parse pdf
    content_list = parse_pdf(
        file_path=file_path,
        asset_dir=output_dir,
        magic_config_path=magic_config_path,
    )

    # save content list
    pickle_content_path = os.path.join(output_dir, 'content_list.pickle')
    with open(pickle_content_path, 'rb') as f:
        content_list = pickle.load(f)
    content_list = [Content(content) for content in content_list]

    # md writer
    md_file_path = os.path.join(
        final_md_file_save_dir,
        f'{name_without_suff}.md',
    )
    md_writer = open(md_file_path, 'w')

    md_writer.write(f'paper: {name_without_suff}' + line_breaker)

    # summary
    summary = summary_content(content_list=content_list)
    summary_save_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_save_path, 'w') as f:
        f.write(summary)
        print(f'summary saved to {summary_save_path}')

    save_summary_of_content(md_writer=md_writer, summary=summary)

    # translate content
    translate_content(
        md_writer=md_writer,
        content_list=content_list,
        sys_image_folder=sys_image_folder,
    )

    md_writer.close()
    print(f'parsed markdown saved to {md_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of argparse usage.")

    parser.add_argument("--file_path", help="path to pdf file")
    parser.add_argument("--output_dir",
                        help="path to assets folder",
                        default="./parsed_asset")
    parser.add_argument("--ollama_host",
                        help="ollama host",
                        default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model",
                        help="ollama model name",
                        default="qwen3:30b-a3b")
    parser.add_argument("--magic_config_path",
                        help="magic pdf config path",
                        default="./magic-pdf.json")

    parser.add_argument("--sys_image_folder",
                        help="final image save folder",
                        default="./md_images")

    parser.add_argument("--final_md_file_save_dir",
                        help="final md file save folder",
                        default=".")

    parser.add_argument("--src_lang",
                        help="source paper language",
                        default="en")

    parser.add_argument("--target_lang",
                        help="translate target language",
                        default="zh")

    args = parser.parse_args()

    output_dir = os.path.realpath(args.output_dir)
    print(f'output dir: {output_dir}')

    magic_config_path = os.path.realpath(args.magic_config_path)
    print(f'magic config path: {magic_config_path}')

    sys_image_folder = os.path.realpath(args.sys_image_folder)
    print(f'system image save folder: {sys_image_folder}')

    final_md_file_save_dir = os.path.realpath(args.final_md_file_save_dir)
    print(f'final md save folder: {final_md_file_save_dir}')

    ollama_host = args.ollama_host
    ollama_model = args.ollama_model

    src_lang = lang_mapping[args.src_lang]
    print(f'source language: {src_lang}')

    target_lang = lang_mapping[args.target_lang]
    print(f'target language: {target_lang}')

    print(f'processing file: {os.path.basename(args.file_path)}')

    process(
        file_path=args.file_path,
        output_dir=output_dir,
        magic_config_path=magic_config_path,
        sys_image_folder=sys_image_folder,
        final_md_file_save_dir=final_md_file_save_dir,
    )

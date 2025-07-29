import argparse
import os
import pickle
import re
import shutil
import time
import os
import traceback
import math
from io import TextIOWrapper
from typing import Any, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from ollama import Client
from strenum import StrEnum

line_breaker = '\n\n'

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
你是一个论文翻译助手，请将下面的{src_lang}内容翻译成{target_lang}。

下面是需要翻译的内容：

{content}

上面是需要翻译的内容
"""

summary_prompt = """
你是一个论文阅读助手，使用{target_lang}语言，总结下面{src_lang}论文内容，总结的内容需要包括论文主要创新点。

下面是论文内容


{content}


上面是论文内容

使用{target_lang}语言，总结{src_lang}论文内容，总结的内容需要包括论文主要创新点。

注意：
- 忽略论文引用文献部分内容，只总结论文正文部分。
"""
max_summary_token_num = 80 * 1024


# ------------------------------------------------------------------------------
# util funcs
def time_it(func):

    def wrapper(*kargs, **kwargs):
        begin = time.time_ns()
        ret = func(*kargs, **kwargs)
        elapse = (time.time_ns() - begin) // 1000000
        print(f"func {func.__name__} took {elapse // 60000}min {(elapse % 60000)//1000}sec {elapse%60000%1000}ms to finish")

        return ret

    return wrapper


def save_image(src_path: str, dst_dir: str) -> None:
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
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


def is_empty(text: str) -> bool:
    text = safe_strip(text)
    if text is None:
        return True
    if len(text) == 0:
        return True
    if text == '[]':
        return True
    return False


def format_md_image_path(sys_image_folder: str, img_name: str) -> str:
    """
    md image can only be corrct displayed when saved to same folder of the md file,
    reformatted inserted image path for better display.
    """
    return f"![[{os.path.join(os.path.basename(sys_image_folder), img_name)}]]" + line_breaker


def print_exception(e: Exception):
    print(f"{datetime.now()}: Exception: {type(e).__name__} - {e}")
    formatted_traceback = traceback.format_exc()
    print(formatted_traceback)


def estimate_token_num(text: str) -> Tuple[int, list[str]]:
    """
    Estimate tokens in text. Combine consecutive ascii character as one token,
    treat each non-ascii character as one token. Each ascii token accounts for 2.3
    token, each non-ascii token accounts for 1.2 token.

    Args:
    - text: the string to parse.

    Return:
    - int, estimated token num.
    - list of string, estimated tokens.
    """
    if text is None or len(text.strip()) == 0:
        return 0, []

    text = text.strip()

    def is_space(ch: str) -> bool:
        if ord(ch) >= 128:
            return False
        if ch.strip() == '':
            return True
        return False

    def token_bound_found(text: str, i: int, j: int) -> bool:
        if ord(text[i]) < 127:
            # space met or non-ascii character met
            return (is_space(text[j]) or ord(text[j]) > 127)
        else:
            # count one non-ascii character as one token
            return j > i

    token_buffer = []
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and not token_bound_found(text, i, j):
            j += 1

        token = text[i:j]
        token_buffer.append(token)

        i = j
        while i < len(text) and is_space(text[i]):
            i += 1

    token_num = 0
    for token in token_buffer:
        if ord(token[0]) < 128:
            token_num += 2.3
        else:
            token_num += 1.5

    return int(token_num), token_buffer


def post_text_process(text: str) -> str:
    # strip space around $
    p = r" *(\$) *"
    text = re.sub(p, r"\1", text)

    return text


def format_list_as_str(l: list[Any]) -> str:
    return '\n'.join([str(e) for e in l])


def format_log(text: str) -> str:
    return f"{datetime.now()}: {text}"


# ------------------------------------------------------------------------------
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
        content_type: ContentType,
        content: str,
        extra_discription: str,
        content_path: str,
        **kwargs,
    ):
        """
        Parsed content.
        
        Args:
        - content_type
        - content
        - extra_discription: extra discription for the content, necessary for content like image / table.
        - content_ucontent_pathrl: url to content, necessary for content like image / table.
        """
        self.content_type = content_type
        self.content = content
        self.extra_discription = extra_discription
        self.content_path = content_path

    def __str__(self, ) -> str:
        if self.content_type in [ContentType.TEXT, ContentType.EQUATION]:
            return self.content

        elif self.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            return f'content is {self.content_type} \n' \
                + f"content path : {self.content_path} \n" \
                + f"content description: {self.extra_discription} \n"
        else:
            return f"unregnized content"


job_executor = None


def get_job_executor():
    global job_executor
    if job_executor is None:
        job_executor = ProcessPoolExecutor(max_workers=1)
    return job_executor


def parse_pdf_job(
    file_path: str,
    asset_dir: str,
    magic_config_path: str,
) -> None:
    """
    Parse PDF content and return content list. The result is a list of json object representing a pdf content block.
    
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
    
    Typical parsed paper content is organized by list of content block.
    Headlines will stored in one separated block, with `text_level` = 1 while regular content block's `text_level` key is missing. 
    Headline blocks are followed by regular content block, including `text`, `equation`, `table` and `image` (distinguished by key `type`). 
    All captions are stored in each block's caption key, for example, caption of a parsed image is saved in `img_caption` key of the block.

    Please refer https://github.com/opendatalab/MinerU/blob/master/demo/demo.py for more details.

    Parsed result is saved to `asset_dir`, the content list will be saved using pickle as well.

    Args:
    - file_path: path to the pdf file
    - asset_dir: folder for saving parsed assets.
    - magic_config_path: magic pdf config path.
    """
    import copy
    import json
    import os
    from pathlib import Path

    # NOTE: magic_pdf package uses singleton design and the model isntance is
    # initialized when the module is imported, so postpone the import statement
    # until parse method is called.
    magic_config_path = os.path.abspath(magic_config_path)
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = magic_config_path
    os.environ["MINERU_MODEL_SOURCE"] = 'local'
    print(format_log(f'setting magic pdf config path to {magic_config_path}'))

    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

    try:
        # prepare env
        try:
            shutil.rmtree(asset_dir)
        except:
            pass
        os.makedirs(asset_dir, exist_ok=True)

        lang = 'ch'
        start_page_id = 0
        end_page_id = None
        parse_method = 'auto'

        file_name = str(Path(file_path).stem)
        pdf_bytes = read_fn(file_path)

        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [new_pdf_bytes],
            [lang],
            parse_method=parse_method,
            formula_enable=True,
            table_enable=True,
        )

        model_list = infer_results[0]
        model_json = copy.deepcopy(model_list)
        local_image_dir, local_md_dir = prepare_env(asset_dir, file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        middle_json = pipeline_result_to_middle_json(
            model_list,
            all_image_lists[0],
            all_pdf_docs[0],
            image_writer,
            lang,
            ocr_enabled_list[0],
            True,
        )

        pdf_info = middle_json["pdf_info"]

        # draw span and layout
        draw_layout_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")
        draw_span_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_span.pdf")
        md_writer.write(f"{file_name}_origin.pdf", new_pdf_bytes)

        # dump md
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        md_writer.write_string(f"{file_name}.md", md_content_str)

        # dump content list
        image_dir = str(os.path.basename(local_image_dir))
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(f"{file_name}_content_list.json", json.dumps(content_list, ensure_ascii=False, indent=4))

        # dump middle json
        md_writer.write_string(f"{file_name}_middle.json", json.dumps(middle_json, ensure_ascii=False, indent=4))

        # dump model json
        md_writer.write_string(f"{file_name}_model.json", json.dumps(model_json, ensure_ascii=False, indent=4))

        # update image path to absolute path
        for content in content_list:
            img_path = content.get('img_path', None)
            if img_path:
                content['img_path'] = os.path.realpath(os.path.join(asset_dir, file_name, parse_method, img_path))

        # parse content list
        parsed_content_list = []
        for raw_content in content_list:
            content_type = raw_content.get('type', None)
            content = None

            if content_type in ['text', 'equation']:
                text = raw_content.get('text', '')
                if raw_content.get('text_level', None) == 1:
                    text = "# " + text
                content = Content(
                    content_type=ContentType.TEXT if content_type == 'text' else ContentType.EQUATION,
                    content=text,
                    extra_discription="",
                    content_path=None,
                )
            elif content_type == 'image':
                img_caption = raw_content.get('img_caption', None)
                if isinstance(img_caption, list):
                    img_caption = format_list_as_str(img_caption)

                img_footnote = raw_content.get('img_footnote', None)
                if isinstance(img_footnote, list):
                    img_footnote = format_list_as_str(img_footnote)

                img_path = raw_content.get('img_path')
                text = ""
                if img_caption:
                    text += img_caption + line_breaker
                if img_footnote:
                    text += img_footnote + line_breaker

                content = Content(
                    content_type=ContentType.IMAGE,
                    content=None,
                    extra_discription=text,
                    content_path=img_path,
                )
            elif content_type == 'table':
                table_body = raw_content.get('table_body', None)
                table_caption = raw_content.get('table_caption', None)
                if isinstance(table_caption, list):
                    table_caption = format_list_as_str(table_caption)

                table_footnote = raw_content.get('table_footnote', None)
                if isinstance(table_footnote, list):
                    table_footnote = format_list_as_str(table_footnote)

                img_path = raw_content.get('img_path', None)
                text = ""
                # NOTE: donot append parsed table body to content
                # if table_body:
                #     text += str(table_body) + line_breaker

                if table_caption:
                    text += table_caption + line_breaker
                if table_footnote:
                    text += table_footnote + line_breaker

                content = Content(
                    content_type=ContentType.TABLE,
                    content=None,
                    extra_discription=text,
                    content_path=img_path,
                )
            else:
                raise Exception(f"unknown content type: {content_type}")
            parsed_content_list.append(content)

        # save content list
        pickle_content_path = os.path.join(asset_dir, 'content_list.pickle')
        print(format_log(f'saving parsed content list to {pickle_content_path}'))
        with open(pickle_content_path, 'wb') as f:
            pickle.dump(parsed_content_list, f)

    except Exception as e:
        print_exception(e)


@time_it
def parse_pdf(
    file_path: str,
    asset_dir: str,
    magic_config_path: str,
) -> list[Content]:
    # submit job
    job_executor = get_job_executor()
    job_executor.submit(
        parse_pdf_job,
        file_path=file_path,
        asset_dir=asset_dir,
        magic_config_path=magic_config_path,
    )
    try:
        job_executor.shutdown(wait=True)
    except Exception as e:
        print_exception(e)
        os._exit(0)

    print(format_log(f'PDF parse job done'))

    # parse returned content
    pickle_content_path = os.path.join(asset_dir, 'content_list.pickle')
    print(format_log(f'loading content list from {pickle_content_path}'))
    with open(pickle_content_path, 'rb') as f:
        content_list = pickle.load(f)
    print(format_log(f'loaded {len(content_list)} content from {pickle_content_path}'))

    return content_list


# ------------------------------------------------------------------------------
# ollama interface
the_ollama_client = None


def get_ollama_client():
    global ollama_host

    if the_ollama_client:
        return the_ollama_client
    return Client(host=ollama_host, timeout=15 * 60)  # timed out: 15min


def ollama_chat(prompt: str, ) -> str:
    global ollama_host, ollama_model, gen_conf

    ollama_client = get_ollama_client()

    history = [{'role': 'user', 'content': prompt}]
    options = {}
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

    try:
        response = ollama_client.chat(model=ollama_model, messages=history, options=options, keep_alive=10)
    except Exception as e:
        return f"Exception {e}"

    ans = response["message"]["content"].strip()
    if '</think>' in ans:
        ans = ans.split('</think>')[-1]
    return ans.strip()


# ------------------------------------------------------------------------------
# save parsed content
def save_parsed_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    **kwargs,
) -> None:
    sys_image_folder = kwargs.get('sys_image_folder', os.path.expanduser("~/Pictures"))
    print(format_log(f'using {sys_image_folder} as sys image save folder'))
    print(format_log(f'total {len(content_list)} contents'))

    md_writer.write('# ' + '=' * 8 + '  Original Content  ' + '=' * 8 + line_breaker)

    for i, content in enumerate(content_list):
        lines = ''

        if content.content_type == ContentType.TEXT:
            lines += content.content + line_breaker

        elif content.content_type == ContentType.EQUATION:
            lines += content.content + line_breaker

        elif content.content_type == ContentType.IMAGE:
            # copy image
            if content.content_path:
                img_name = os.path.basename(content.content_path)
                save_image(content.content_path, sys_image_folder)

                lines += format_md_image_path(sys_image_folder, img_name)
                lines += line_breaker

            lines += content.extra_discription + line_breaker

        elif content.content_type == ContentType.TABLE:
            lines += str(content.content) + content.extra_discription + line_breaker

            # copy image
            if content.content_path:
                img_name = os.path.basename(content.content_path)
                save_image(content.content_path, sys_image_folder)

                lines += format_md_image_path(sys_image_folder, img_name)
                lines += line_breaker
        else:
            pass

        lines = post_text_process(lines)
        md_writer.write(lines)
        md_writer.flush()


# ------------------------------------------------------------------------------
# translate func
def translate_text_content(text: str) -> str:
    if is_empty(text):
        return ""

    max_byte_len = 4 * 1024
    block_num = math.ceil(len(text) / max_byte_len)
    print(format_log(f'text byte length: {len(text)}, block num: {block_num}'))
    full_result = ""
    for i in range(block_num):
        segment = text[i * max_byte_len:(i + 1) * max_byte_len]

        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=segment,
        )
        ret = ollama_chat(prompt=formatted_prompt)

        full_result += ret

    return full_result


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
    print(format_log(f'using {sys_image_folder} as sys image save folder'))

    print(format_log(f'total {len(content_list)} contents'))

    md_writer.write('# ' + '=' * 8 + '  Translated Content  ' + '=' * 8 + line_breaker)

    for i, content in enumerate(content_list):
        print(format_log(f'translating content {i}, original content: {content}'))
        print('-' * 128)
        print('\n\n')

        if content.content_type == ContentType.TEXT:
            translated = translate_text_content(content.content)
            content.translated_content = translated

        elif content.content_type in [ContentType.TABLE, ContentType.IMAGE]:
            translated = translate_text_content(content.extra_discription)
            content.translated_extra_discription = translated
            print(format_log(f'tranlated content: {translated}'))

        else:
            pass

        print(format_log(f'translation done'))

        # save translated content
        lines = ''
        if content.content_type == ContentType.TEXT:
            lines += content.translated_content + line_breaker

        elif content.content_type == ContentType.EQUATION:
            lines += content.content + line_breaker

        elif content.content_type == ContentType.IMAGE:
            # img path
            if content.content_path:
                save_image(content.content_path, sys_image_folder)
                img_name = os.path.basename(content.content_path)
                lines += format_md_image_path(sys_image_folder, img_name)
                lines += line_breaker

            # extra disscription
            lines += content.translated_extra_discription + line_breaker

        elif content.content_type == ContentType.TABLE:
            # img path
            if content.content_path:
                save_image(content.content_path, sys_image_folder)
                img_name = os.path.basename(content.content_path)
                lines += format_md_image_path(sys_image_folder, img_name)
                lines += line_breaker

            # extra disscription
            lines += content.translated_extra_discription + line_breaker
        else:
            pass

        lines = post_text_process(lines)
        print(format_log(f'translated content: {lines}'))
        print('\n\n')

        md_writer.write(lines)
        md_writer.flush()

    # save pickle result
    translated_pickle_content_path = os.path.join(output_dir, 'translated_content_list.pickle')
    with open(translated_pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
        print(format_log(f'save translated  content list to {translated_pickle_content_path}'))


# ------------------------------------------------------------------------------
# summary func
@time_it
def summary_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    **kwargs,
) -> None:
    global summary_prompt, src_lang, target_lang
    print(f'{datetime.now()}: summary_content, src_lang={src_lang}, target_lang={target_lang}')

    full_content = ''
    for content in content_list:
        if content.content_type == ContentType.TEXT:
            full_content += content.content + line_breaker

        elif content.content_type == ContentType.EQUATION:
            full_content += content.content + line_breaker

        elif content.content_type == ContentType.IMAGE:
            full_content += content.extra_discription + line_breaker

        elif content.content_type == ContentType.TABLE:
            full_content += content.extra_discription + line_breaker

        else:
            print(format_log(f'unrecognized content: {content}'))

    print(format_log(f'full content length: {len(full_content)}'))
    token_num, _ = estimate_token_num(full_content)
    print(format_log(f'esitmated full content token num: {token_num}'))
    if token_num > max_summary_token_num:
        ratio = float(max_summary_token_num) / token_num
        print(format_log(f'truncate full content by ratio: {ratio}, original length: {len(full_content)}'))
        full_content = full_content[:int(len(full_content) * ratio)]

    full_content = full_content.encode('utf-8', errors='ignore').decode('utf-8')

    formatted_promt = summary_prompt.format(
        src_lang=src_lang,
        target_lang=target_lang,
        content=full_content,
    )
    print(format_log(f'formatted prompt: {formatted_promt}'))

    summary = ollama_chat(prompt=formatted_promt)
    summary = post_text_process(summary)
    print(format_log(f'content summary: {summary}'))

    summary_save_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_save_path, 'w') as f:
        f.write(summary)
        print(format_log(f'summary saved to {summary_save_path}'))

    # save
    md_writer.write('# ' + '=' * 8 + '  Paper Summary  ' + '=' * 8 + line_breaker)
    md_writer.write(summary + line_breaker)
    md_writer.write('-' * 8 + line_breaker)
    md_writer.flush()


step_func = {
    'summary': summary_content,
    'translate': translate_content,
    'original': save_parsed_content,
}


@time_it
def process(
    file_path: str,
    output_dir: str,
    magic_config_path: str,
    sys_image_folder: str,
    final_md_file_save_dir: str,
    steps: list[str] = ['summary', 'translate'],
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
    print(format_log(f'processing started, required steps: {steps}'))

    os.makedirs(output_dir, exist_ok=True)
    name_without_suff = os.path.basename(file_path).rsplit('.', 1)[0]
    print(format_log(f'file name without out suffix: {name_without_suff}'))

    # parse pdf
    content_list = parse_pdf(
        file_path=file_path,
        asset_dir=output_dir,
        magic_config_path=magic_config_path,
    )

    # md writer
    md_file_path = os.path.join(
        final_md_file_save_dir,
        f'{name_without_suff}.md',
    )
    md_writer = open(md_file_path, 'w')

    md_writer.write(f'paper: {name_without_suff}' + line_breaker)

    # apply step functions
    kwargs = {
        'sys_image_folder': sys_image_folder,
    }
    for step in steps:
        print(format_log(f'processing step: {step}'))
        step = step.strip()
        if step not in step_func:
            print(format_log(f'step {step} not configured, ignore'))
            continue

        func = step_func[step]
        func(
            md_writer=md_writer,
            content_list=content_list,
            **kwargs,
        )

    md_writer.close()
    print(format_log(f'parsed markdown saved to {md_file_path}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of argparse usage.")

    parser.add_argument("--file_path", help="path to pdf file")
    parser.add_argument("--output_dir", help="path to assets folder", default="./parsed_asset")
    parser.add_argument("--ollama_host", help="ollama host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model", help="ollama model name", default="qwen3:30b-a3b")
    parser.add_argument("--magic_config_path", help="magic pdf config path", default="./magic-pdf.json")
    parser.add_argument("--sys_image_folder", help="final image save folder", default="./md_images")
    parser.add_argument("--final_md_file_save_dir", help="final md file save folder", default=".")
    parser.add_argument("--src_lang", help="source paper language", default="en")
    parser.add_argument("--target_lang", help="translate target language", default="zh")
    parser.add_argument("--steps", help="required steps", default="summary,translate,original")
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output_dir)
    print(format_log(f'output dir: {output_dir}'))

    magic_config_path = os.path.realpath(args.magic_config_path)
    print(format_log(f'magic config path: {magic_config_path}'))

    sys_image_folder = os.path.realpath(args.sys_image_folder)
    print(format_log(f'system image save folder: {sys_image_folder}'))

    final_md_file_save_dir = os.path.realpath(args.final_md_file_save_dir)
    print(format_log(f'final md save folder: {final_md_file_save_dir}'))

    ollama_host = args.ollama_host
    ollama_model = args.ollama_model

    src_lang = lang_mapping[args.src_lang]
    print(format_log(f'source language: {src_lang}'))

    target_lang = lang_mapping[args.target_lang]
    print(format_log(f'target language: {target_lang}'))

    print(format_log(f'processing file: {os.path.basename(args.file_path)}'))

    process(
        file_path=args.file_path,
        output_dir=output_dir,
        magic_config_path=magic_config_path,
        sys_image_folder=sys_image_folder,
        final_md_file_save_dir=final_md_file_save_dir,
        steps=args.steps.split(','),
    )

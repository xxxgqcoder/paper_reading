import argparse
import os
import pickle
import json
import re
import shutil
from io import TextIOWrapper
from typing import Any, Dict

from ollama import Client


# ==============================================================================
# parse pdf
def safe_strip(raw: str) -> str:
    if raw is None or len(raw) == 0:
        return ''
    raw = str(raw)
    return raw.strip()


def is_empty(text: str):
    if text is None:
        return True
    text = text.strip()
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


def save_image(src_path: str, dst_dir: str) -> None:
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copyfile(src_path, dst_path)


def format_md_image_path(sys_image_folder: str, img_name: str) -> str:
    """
    md image can only be corrct displayed when saved to same folder of the md file,
    reformatted inserted image path for better display.
    """
    return f"![[{os.path.join(os.path.basename(sys_image_folder), img_name)}]]"


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
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

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

    return content_list


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
def translate_table_content(content: Dict[str, Any]) -> None:
    if 'table_caption' in content and not is_empty(
            safe_strip(content['table_caption'])):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=content['table_caption'],
        )
        translated_table_caption = ollama_chat(prompt=formatted_prompt)
        content['translated_table_caption'] = post_text_process(
            translated_table_caption)

    if 'table_footnote' in content and not is_empty(
            safe_strip(content['table_footnote'])):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=content['table_footnote'],
        )
        translated_table_footnote = ollama_chat(prompt=formatted_prompt)
        content['translated_table_footnote'] = post_text_process(
            translated_table_footnote)


def translate_image_content(content: Dict[str, Any]) -> None:
    if 'img_caption' in content and not is_empty(
            safe_strip(content['img_caption'])):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=content['img_caption'],
        )
        translated_img_caption = ollama_chat(prompt=formatted_prompt)
        content['translated_img_caption'] = post_text_process(
            translated_img_caption)

    if 'img_footnote' in content and not is_empty(
            safe_strip(content['img_footnote'])):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=content['img_footnote'],
        )
        translated_img_footnote = ollama_chat(prompt=formatted_prompt)
        content['translated_img_footnote'] = post_text_process(
            translated_img_footnote)


def translate_text_content(content: Dict[str, Any]) -> None:
    translated_text = ''
    if 'text' in content and not is_empty(safe_strip(content['text'])):
        formatted_prompt = translate_prompt.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=content['text'],
        )
        translated_text = ollama_chat(prompt=formatted_prompt)

    content['translated_text'] = post_text_process(translated_text)


def translate_content(content_list: list[Dict[str, Any]]):
    """
    Translate contents.
    Args:
    - content_list: a list of content.
    """
    print(f'total {len(content_list)} contents')

    for i, content in enumerate(content_list):
        print(f'processing content {i}')
        if content['type'] == 'table':
            translate_table_content(content)

        elif content['type'] == 'image':
            translate_image_content(content)

        elif content['type'] == 'text':
            translate_text_content(content)
        else:
            print(f'unrecognized content: {json.dumps(content, indent=4)}')

        print(json.dumps(content, indent=4, ensure_ascii=False))
        print('=' * 128)

    translated_pickle_content_path = os.path.join(
        output_dir, 'translated_content_list.pickle')
    with open(translated_pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
        print(
            f'save translated  content list to {translated_pickle_content_path}'
        )


def save_transalted_content(
    md_writer: TextIOWrapper,
    content_list: list[Dict[str, Any]],
    **kwargs,
) -> None:
    """
    Args:
    - content_list: content list, represented by a list of dict.
    - kwargs: should contain `sys_image_folder`.
    """
    sys_image_folder = kwargs.get('sys_image_folder',
                                  os.path.expanduser("~/Pictures"))
    print(f'using {sys_image_folder} as sys image save folder')

    md_writer.write('# ' + '=' * 8 + '  Translated content  ' + '=' * 8 +
                    '\n\n')

    for content in content_list:
        if not is_valid_content(content):
            print(f'invalid block: {json.dumps(content, indent=4)}')
            continue

        if content['type'] == 'text':
            text = content['translated_text']
            if 'text_level' in content and content['text_level'] == 1:
                text = '# ' + text
            md_writer.write(text + "\n\n")

        elif content['type'] == 'equation':
            text = content['text']
            md_writer.write(text + "\n\n")

        elif content['type'] == 'image':
            # copy image
            abs_img_path = os.path.join(output_dir, content['img_path'])
            img_name = os.path.basename(abs_img_path)
            save_image(abs_img_path, sys_image_folder)
            md_writer.write(
                format_md_image_path(sys_image_folder, img_name) + "\n\n")

            # image caption
            img_caption = content['translated_img_caption']
            md_writer.write(img_caption + "\n\n")

            if 'translated_img_footnote' in content:
                img_footnote = content['translated_img_footnote']
                md_writer.write(img_footnote + "\n\n")

        elif content['type'] == 'table':
            table_body = content['table_body']
            md_writer.write(table_body + "\n\n")

            if 'img_path' in content and len(content['img_path']) > 0:
                abs_img_path = os.path.join(output_dir, content['img_path'])
                img_name = os.path.basename(abs_img_path)
                save_image(abs_img_path, sys_image_folder)
                md_writer.write(
                    format_md_image_path(sys_image_folder, img_name) + "\n\n")

            if 'translated_table_caption' in content:
                translated_table_caption = content['translated_table_caption']
                md_writer.write(translated_table_caption + "\n\n")

        else:
            print(f'unrecognized content: {json.dumps(content, indent=4)}')


# ==============================================================================
# summary paper content


def summary_content(content_list: list[Dict[str, Any]]) -> None:
    global summary_prompt, src_lang, target_lang
    print(f'summary_content, src_lang={src_lang}, target_lang={target_lang}')

    full_content = ''
    for content in content_list:
        if content['type'] == 'text':
            text = content['text']
            if 'text_level' in content and content['text_level'] == 1:
                text = '# ' + text
            full_content += text + '\n\n'

        elif content['type'] == 'equation':
            text = content['text']
            full_content += text + '\n\n'

        elif content['type'] == 'image':
            full_content += safe_strip(content['img_caption']) + '\n\n'

            if 'img_footnote' in content:
                full_content += safe_strip(content['img_footnote']) + '\n\n'

        elif content['type'] == 'table':

            if 'table_caption' in content:
                full_content += safe_strip(content['table_caption']) + '\n\n'

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
    md_writer.write('# ' + '=' * 8 + '  Paper summary  ' + '=' * 8 + '\n\n')
    md_writer.write(summary + '\n\n')
    md_writer.write('-' * 8 + '\n\n')


# ==============================================================================
# file process func


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
    pickle_content_path = os.path.join(output_dir, 'content_list.pickle')
    with open(pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
        print(f'save parsed content list to {pickle_content_path}')

    # pickle_content_path = os.path.join(output_dir, 'content_list.pickle')
    # with open(pickle_content_path, 'rb') as f:
    #     content_list = pickle.load(f)

    # md writer
    md_file_path = os.path.join(
        final_md_file_save_dir,
        f'{name_without_suff}.md',
    )
    md_writer = open(md_file_path, 'w')

    # summary
    summary = summary_content(content_list=content_list)
    summary_save_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_save_path, 'w') as f:
        f.write(summary)
        print(f'summary saved to {summary_save_path}')

    save_summary_of_content(md_writer=md_writer, summary=summary)

    # translate content
    translate_content(content_list=content_list)

    translated_pickle_content_path = os.path.join(
        output_dir, 'translated_content_list.pickle')
    with open(translated_pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
    print(f'save translated content list to {translated_pickle_content_path}')

    save_transalted_content(
        md_writer=md_writer,
        content_list=content_list,
        sys_image_folder=sys_image_folder,
    )

    md_writer.close()
    print(f'parsed markdown saved to {md_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of argparse usage.")

    parser.add_argument("--file_path", help="path to pdf file")
    parser.add_argument(
        "--output_dir",
        help="path to assets folder",
        default="./parsed_asset",
    )
    parser.add_argument(
        "--ollama_host",
        help="ollama host",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--ollama_model",
        help="ollama model name",
        default="qwen3:30b-a3b",
    )
    parser.add_argument(
        "--magic_config_path",
        help="magic pdf config path",
        default="./magic-pdf.json",
    )

    parser.add_argument(
        "--sys_image_folder",
        help="final image save folder",
        default="./md_images",
    )

    parser.add_argument(
        "--final_md_file_save_dir",
        help="final md file save folder",
        default=".",
    )

    parser.add_argument(
        "--src_lang",
        help="source paper language",
        default="en",
    )

    parser.add_argument(
        "--target_lang",
        help="translate target language",
        default="zh",
    )

    args = parser.parse_args()

    output_dir = os.path.realpath(args.output_dir)
    print(f'output dir: {output_dir}')

    magic_config_path = os.path.realpath(args.magic_config_path)
    print(f'magic config path: {magic_config_path}')

    sys_image_folder = os.path.realpath(args.sys_image_folder)
    print(f'system image save folder: {sys_image_folder}')

    final_md_file_save_dir = os.path.realpath(args.final_md_file_save_dir)
    print(f'final md save folder: {final_md_file_save_dir}')

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

    ollama_host = args.ollama_host
    ollama_model = args.ollama_model

    src_lang = args.src_lang
    src_lang = lang_mapping[src_lang]
    print(f'source language: {src_lang}')

    target_lang = args.target_lang
    target_lang = lang_mapping[target_lang]
    print(f'target language: {target_lang}')

    process(
        file_path=args.file_path,
        output_dir=output_dir,
        magic_config_path=magic_config_path,
        sys_image_folder=sys_image_folder,
        final_md_file_save_dir=final_md_file_save_dir,
    )

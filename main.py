import argparse
import os
import pickle
import json
import re
import shutil
import time
import gc
from io import TextIOWrapper
from typing import Any, Dict

from ollama import Client

from parser import (
    parse_pdf,
    post_text_process,
    Content,
    ContentType,
)

from utils import (
    save_image,
    time_it,
    is_empty,
    safe_strip,
)

line_breaker = '\n\n\n\n'


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
def translate_content(content_list: list[Content]):
    """
    Translate contents.
    Args:
    - content_list: a list of content.
    """
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

    translated_pickle_content_path = os.path.join(
        output_dir, 'translated_content_list.pickle')
    with open(translated_pickle_content_path, 'wb') as f:
        pickle.dump(content_list, f)
        print(
            f'save translated  content list to {translated_pickle_content_path}'
        )


def save_transalted_content(
    md_writer: TextIOWrapper,
    content_list: list[Content],
    **kwargs,
) -> None:
    """
    Args:
    - content_list: content list, represented by a list of dict.
    - kwargs: should contain `sys_image_folder`.
    """
    sys_image_folder = kwargs.get(
        'sys_image_folder',
        os.path.expanduser("~/Pictures"),
    )
    print(f'using {sys_image_folder} as sys image save folder')

    md_writer.write('# ' + '=' * 8 \
                    + '  Translated content  ' \
                    + '=' * 8 + line_breaker)

    for content in content_list:
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
            print(
                f'unrecognized content: {json.dumps(content.__dict__, indent=4)}'
            )

        lines = post_text_process(lines)
        md_writer.write(lines)


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

    print(f'processing file: {os.path.basename(args.file_path)}')

    process(
        file_path=args.file_path,
        output_dir=output_dir,
        magic_config_path=magic_config_path,
        sys_image_folder=sys_image_folder,
        final_md_file_save_dir=final_md_file_save_dir,
    )

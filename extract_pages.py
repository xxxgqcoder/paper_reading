import argparse
import os

import yaml
from pypdf import PdfReader, PdfWriter


def extract_pages(input_pdf_path: str, output_pdf_path: str, page_indices: list[int]):
    """
    Extracts specified pages from a PDF and saves them to a new file.

    Args:
        input_pdf_path: The path to the source PDF file.
        output_pdf_path: The path to save the new PDF file.
        page_indices: A list of 0-based page indices to extract.
    """
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    for page_index in page_indices:
        if 0 <= page_index < len(reader.pages):
            writer.add_page(reader.pages[page_index])
        else:
            print(f"Warning: Page index {page_index} is out of range. Skipping.")

    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)


def parse_page_ranges(page_ranges_str: str) -> list[int]:
    """
    Parses a string of page ranges (e.g., "1,3,5-7") into a list of 0-based page indices.
    """
    page_indices = []
    ranges = page_ranges_str.split(",")
    for r in ranges:
        if "-" in r:
            start, end = map(int, r.split("-"))
            # Subtract 1 to convert from 1-based to 0-based index
            page_indices.extend(range(start - 1, end))
        else:
            # Subtract 1 to convert from 1-based to 0-based index
            page_indices.append(int(r) - 1)
    return sorted(list(set(page_indices)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pages from a PDF file based on a YAML config file."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_pdf = config["input_pdf"]
    pages_to_extract_list = config["pages"]

    base, ext = os.path.splitext(input_pdf)

    for pages_str in pages_to_extract_list:
        output_pdf_path = f"{base}-{pages_str}{ext}"
        page_indices_to_extract = parse_page_ranges(pages_str)

        extract_pages(input_pdf, output_pdf_path, page_indices_to_extract)
        print(
            f"Successfully extracted pages '{pages_str}' from '{input_pdf}' and saved to '{output_pdf_path}'."
        )

#!/usr/bin/env python
import glob
import os
import subprocess


def main():
    pdf_folder = "/Users/xcoder/aDrive/books/software engineering"
    output_folder = "/Users/xcoder/obsidian/Profession/Clean Code"
    pdf_file_pattern = f"{pdf_folder}/代码整洁之道-*pdf"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Get all file names from output dir, rm suffix.
    processed_files = os.listdir(output_folder)
    processed_basenames = {f.rsplit(".", 1)[0] for f in processed_files if "." in f}

    # 2. Extract base file from file names from pdf folder without suffix.
    all_pdf_files = glob.glob(pdf_file_pattern)

    # 3. Filter files that are not found in the processed list.
    files_to_process = []
    for pdf_path in all_pdf_files:
        pdf_filename = os.path.basename(pdf_path)
        if "." in pdf_filename:
            pdf_basename = pdf_filename.rsplit(".", 1)[0]
            if pdf_basename not in processed_basenames:
                files_to_process.append(pdf_path)
        else:
            # If there's no dot, it can't have a suffix to remove, but might need processing
            if pdf_filename not in processed_basenames:
                files_to_process.append(pdf_path)

    if not files_to_process:
        print("No new PDF files to process.")
        return

    # 4. Run rest of logic based on filtered file names.
    print(f"Found {len(files_to_process)} new PDF files to process.")

    for pdf_path in files_to_process:
        print(f"Processing {pdf_path}")

        command = [
            "python",
            "process_pdf.py",
            "--file_path",
            pdf_path,
            "--final_md_file_save_dir",
            output_folder,
            "--steps=original",
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {os.path.basename(pdf_path)}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        except FileNotFoundError:
            print(
                "Error: 'process_pdf.py' not found. Make sure you are in the correct directory."
            )
            break


if __name__ == "__main__":
    main()

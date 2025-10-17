import os
import json
from tqdm import tqdm

MAX_ROWS = 30  # Max rows per chunk

def strip_prefix(data):
    """
    Remove '@@@_' prefix from all string values in a list of dicts.
    """
    stripped = []
    for row in data:
        stripped_row = {}
        for key, value in row.items():
            if isinstance(value, str) and value.startswith("@@@_"):
                stripped_row[key] = value[len("@@@_"):]
            else:
                stripped_row[key] = value
        stripped.append(stripped_row)
    return stripped


def chunk_data(data, base_name, output_dir):
    """
    Chunk the data into parts and save as individual JSON files.
    """
    total_rows = len(data)
    for start in range(0, total_rows, MAX_ROWS):
        end = min(start + MAX_ROWS, total_rows)
        chunk = data[start:end]
        chunk_filename = f"{base_name}_chunk_{start}_{end}.json"
        chunk_file_path = os.path.join(output_dir, chunk_filename)

        with open(chunk_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def process_and_chunk_json_files(input_folder_path, output_folder_path):
    """
    Walk through all files in input folder, strip prefix, and chunk JSON files.
    """
    for root, _, files in os.walk(input_folder_path):
        for file_name in tqdm(files, desc=f"Processing folder: {os.path.basename(root)}", unit="file"):
            if not file_name.endswith(".json"):
                continue

            input_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(root, input_folder_path)
            output_dir = os.path.join(output_folder_path, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                stripped_data = strip_prefix(data)
                base_name = os.path.splitext(file_name)[0]
                chunk_data(stripped_data, base_name, output_dir)

            except json.JSONDecodeError as e:
                print(f"[Error] JSON decode issue in {input_file_path}: {e}")
            except Exception as e:
                print(f"[Error] Unexpected issue in {input_file_path}: {e}")


if __name__ == "__main__":
    input_folder_path = r"(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))_Perturbed_Tables"
    output_folder_path = r"(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-chunked"

    print("Starting stripping and chunking process...")
    process_and_chunk_json_files(input_folder_path, output_folder_path)
    print("All files processed.")


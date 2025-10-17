import json
import os
from tqdm import tqdm

MAX_ROWS = 30  # Rows per chunk

def create_yes_no(data):
    """Create a Yes/No version of the input JSON data."""
    yes_no_data = []
    for row in data:
        yes_no_dict = {
            key: "Yes" if isinstance(value, str) and value.startswith("@@@_") else "No"
            for key, value in row.items()
        }
        yes_no_data.append(yes_no_dict)
    return yes_no_data

def chunk_json_file(data, base_name, output_folder_path):
    """Chunk the JSON data into separate files with a maximum of MAX_ROWS rows."""
    total_rows = len(data)
    for start in range(0, total_rows, MAX_ROWS):
        end = min(start + MAX_ROWS, total_rows)
        chunk = data[start:end]
        chunk_file_name = f"{base_name}_chunk_{start}_{end}.json"
        chunk_file_path = os.path.join(output_folder_path, chunk_file_name)

        with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
            json.dump(chunk, chunk_file, indent=4)

def process_file(file_path, base_name, output_folder_path):
    """Process a single JSON file: convert to Yes/No and chunk the result."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            yes_no_data = create_yes_no(data)
            chunk_json_file(yes_no_data, base_name + "_yes_no", output_folder_path)
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to read {file_path}: {e}")

def run(input_folder_path, output_folder_path):
    folders = [
        name for name in os.listdir(input_folder_path)
        if os.path.isdir(os.path.join(input_folder_path, name))
    ]

    for folder in folders:
        if folder != "Value_Anomaly_FetaQA":
            in_folder = os.path.join(input_folder_path, folder)
            out_folder = os.path.join(output_folder_path, folder)
            os.makedirs(out_folder, exist_ok=True)

            json_files = [
                file for file in sorted(os.listdir(in_folder))
                if file.endswith(".json") and os.path.isfile(os.path.join(in_folder, file))
            ]

            for file_name in tqdm(json_files, desc=f"Processing {folder}", unit="file"):
                file_path = os.path.join(in_folder, file_name)
                base_name = file_name.split("_updated")[0] if "_updated" in file_name else file_name.replace(".json", "")
                process_file(file_path, base_name, out_folder)
        else:
            in_folder = os.path.join(input_folder_path, folder,"YesNo_Tables")
            out_folder = os.path.join(output_folder_path, folder)
            os.makedirs(out_folder, exist_ok=True)

            json_files = [
                file for file in sorted(os.listdir(in_folder))
                if file.endswith(".json") and os.path.isfile(os.path.join(in_folder, file))
            ]

            for file_name in tqdm(json_files, desc=f"Processing {folder}", unit="file"):
                file_path = os.path.join(in_folder, file_name)
                base_name = file_name.split("_updated")[0] if "_updated" in file_name else file_name.replace(".json", "")
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    chunk_json_file(data, base_name + "_yes_no", output_folder_path)

if __name__ == "__main__":
    input_folder_path = r"(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))_Perturbed_Tables"
    output_folder_path = r"(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-yes-no-chunked"
    run(input_folder_path, output_folder_path)

import os
import json
from collections import defaultdict
from tqdm import tqdm

def merge_chunks(input_folder_path, output_folder_path):
    # Dictionary to hold chunks grouped by original file path
    chunks_dict = defaultdict(list)

    # Step 1: Traverse and group chunks
    for root, _, files in os.walk(input_folder_path):
        for file in files:
            if file.endswith(".json") and "_chunk_" in file:
                file_path = os.path.join(root, file)
                base_name = file.split("_chunk_")[0]
                start_idx = int(file.split("_chunk_")[1].split("_")[0])
                rel_dir = os.path.relpath(root, input_folder_path)
                key = os.path.join(rel_dir, base_name)
                chunks_dict[key].append((start_idx, file_path))

    # Step 2: Merge chunks and save
    for rel_path, chunks in tqdm(chunks_dict.items(), desc="Merging chunks"):
        # Sort by starting index
        sorted_chunks = sorted(chunks, key=lambda x: x[0])
        merged_data = []

        for _, chunk_file in sorted_chunks:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                merged_data.extend(chunk_data)

        # Determine final output file path
        output_dir = os.path.join(output_folder_path, os.path.dirname(rel_path))
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(rel_path) + ".json")

        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            json.dump(merged_data, out_file, indent=4, separators=(",", ":"), ensure_ascii=False)

if __name__ == "__main__":
    input_folder_path = r"(dataset_name ie.(FetaQA ... ))r/(dataset_name ie.(FetaQA ... ))r-chunked"
    output_folder_path = r"(dataset_name ie.(FetaQA ... ))r/(dataset_name ie.(FetaQA ... ))r-merged"

    print("Starting merging of chunked files...")
    merge_chunks(input_folder_path, output_folder_path)
    print("Done!")

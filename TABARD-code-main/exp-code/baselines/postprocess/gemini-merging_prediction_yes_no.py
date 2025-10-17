import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

def extract_base_and_range(filename):
    """
    Extracts base name and numeric range from a chunk filename like: xyz_chunk_0_30.json
    Returns base_name and start index as int for sorting.
    """
    match = re.match(r"(.+)_chunk_(\d+)_(\d+)\.json", filename)
    if match:
        base = match.group(1)
        start = int(match.group(2))
        return base, start
    return None, None

def merge_chunks_in_folder(folder_path, output_folder):
    """Merge chunked files in each subfolder and save as single merged files."""
    for root, _, files in os.walk(folder_path):
        grouped_chunks = defaultdict(list)

        for file_name in files:
            if file_name.endswith(".json") and "_chunk_" in file_name:
                base, start = extract_base_and_range(file_name)
                if base is not None:
                    full_path = os.path.join(root, file_name)
                    grouped_chunks[(base, root)].append((start, full_path))

        for (base_name, subdir), chunks in grouped_chunks.items():
            # Sort by start index
            sorted_chunks = sorted(chunks, key=lambda x: x[0])
            merged_data = []

            for _, file_path in sorted_chunks:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    merged_data.extend(chunk_data)

            # Recreate relative subdir path
            rel_path = os.path.relpath(subdir, folder_path)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            # merged_file_path = os.path.join(save_dir, f"{base_name}_yes_no.json")
            merged_file_path = os.path.join(save_dir, f"{base_name}.json")
            with open(merged_file_path, 'w', encoding='utf-8') as out_file:
                json.dump(merged_data, out_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-2-1']
    List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']
    for fold in tqdm(folder,desc="main_folder_processing") :
        for i in List:
            chunked_folder_path = f"(dataset_name ie.(FetaQA ... ))/gemini-prediction-chunks/prediction-{fold}/{i}/"
            merged_output_path = f"(dataset_name ie.(FetaQA ... ))/gemini-prediction/prediction-{fold}/{i}/"

    
            merge_chunks_in_folder(chunked_folder_path, merged_output_path)
    print("Merging completed!")

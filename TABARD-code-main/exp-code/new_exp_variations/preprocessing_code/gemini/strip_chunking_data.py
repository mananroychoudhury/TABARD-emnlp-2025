# ── chunk_with_labels.py ─────────────────────────────────────────────────

import os
import json
from tqdm import tqdm
from vertexai.preview import tokenization

# ─── Gemini tokenizer setup ───────────────────────────────────────────────
MODEL_NAME = "gemini-1.5-pro-002"  # or gemini-1.0-pro-001, gemini-1.0-pro-002, gemini-1.5-pro-001, gemini-1.5-flash-001, gemini-1.5-flash-002, gemini-1.5-pro-002.
tokenizer = tokenization.get_tokenizer_for_model(MODEL_NAME)
MODEL_MAX_TOKENS           = 5000      # total context window for this model
RESERVED_TOKENS_FOR_IO     = 1200      # e.g. prompt instructions + expected reply
MAX_TOKENS_PER_CHUNK       = MODEL_MAX_TOKENS - RESERVED_TOKENS_FOR_IO  # = 3800


def count_tokens(text: str) -> int:
    """Return the total token count for a given text, using Gemini’s tokenizer."""
    return tokenizer.count_tokens(text).total_tokens


def strip_prefix(data):
    """
    Remove '@@@_' prefix from string values in each row (dict).
    Returns a new list of dicts with prefixes stripped.
    """
    stripped = []
    for row in data:
        stripped_row = {
            key: (value[len("@@@_"):] if isinstance(value, str) and value.startswith("@@@_") else value)
            for key, value in row.items()
        }
        stripped.append(stripped_row)
    return stripped


def chunk_data_in_parallel(
    raw_data,
    stripped_data,
    labels,
    base_name: str,
    merged_dir: str,
    merged_str_dir: str,
    labels_dir: str,
    max_token_budget: int = MAX_TOKENS_PER_CHUNK
):
    """
    Given three parallel lists:
      - raw_data       : list of dicts (original JSON rows)
      - stripped_data  : list of dicts (prefixes removed)
      - labels         : list of dicts (label info)
    Creates chunks under the same row‐boundaries (determined by token count
    on stripped_data), and writes:
      - merged_dir/<base_name>_chunk_<start>_<end>.json
      - merged_str_dir/<base_name>_chunk_<start>_<end>.json
      - labels_dir/<base_name>_chunk_<start>_<end>_labels.json
    """
    chunk_raw = []
    chunk_str = []
    chunk_lbl = []
    token_count = 0
    chunk_index = 0

    total_rows = len(stripped_data)
    for i in range(total_rows):
        row_str = json.dumps(stripped_data[i], ensure_ascii=False)
        row_tokens = count_tokens(row_str)

        # If adding this row would exceed token budget, flush the current chunk
        if token_count + row_tokens > max_token_budget and chunk_raw:
            start = chunk_index
            end = chunk_index + len(chunk_raw)

            # 1) Save raw chunk
            chunk_filename = f"{base_name}_chunk_{start}_{end}.json"
            raw_path = os.path.join(merged_dir, chunk_filename)
            with open(raw_path, 'w', encoding='utf-8') as f_raw:
                json.dump(chunk_raw, f_raw, indent=2, ensure_ascii=False)

            # 2) Save stripped chunk
            stripped_path = os.path.join(merged_str_dir, chunk_filename)
            with open(stripped_path, 'w', encoding='utf-8') as f_str:
                json.dump(chunk_str, f_str, indent=2, ensure_ascii=False)

            # 3) Save label chunk
            label_filename = f"{base_name}_chunk_{start}_{end}_labels.json"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, 'w', encoding='utf-8') as f_lbl:
                json.dump(chunk_lbl, f_lbl, indent=2, ensure_ascii=False)

            # Reset for the next chunk
            chunk_index += len(chunk_raw)
            chunk_raw = []
            chunk_str = []
            chunk_lbl = []
            token_count = 0

        # Add current row to each list
        chunk_raw.append(raw_data[i])
        chunk_str.append(stripped_data[i])
        chunk_lbl.append(labels[i])
        token_count += row_tokens

    # Flush any remaining rows as the final chunk
    if chunk_raw:
        start = chunk_index
        end = chunk_index + len(chunk_raw)

        chunk_filename = f"{base_name}_chunk_{start}_{end}.json"
        raw_path = os.path.join(merged_dir, chunk_filename)
        with open(raw_path, 'w', encoding='utf-8') as f_raw:
            json.dump(chunk_raw, f_raw, indent=2, ensure_ascii=False)

        stripped_path = os.path.join(merged_str_dir, chunk_filename)
        with open(stripped_path, 'w', encoding='utf-8') as f_str:
            json.dump(chunk_str, f_str, indent=2, ensure_ascii=False)

        label_filename = f"{base_name}_chunk_{start}_{end}_labels.json"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w', encoding='utf-8') as f_lbl:
            json.dump(chunk_lbl, f_lbl, indent=2, ensure_ascii=False)


def process_json_files_with_labels(data_folder: str, label_folder: str, output_folder: str):
    """
    For each JSON in `data_folder` (a list of dicts) and its matching label JSON
    in `label_folder` (same filename + "_labels.json"), do the following:
      1) Read the raw data and labels.
      2) Strip '@@@_' from each row and keep raw+stripped in memory.
      3) Chunk all three lists (raw, stripped, labels) under token budget.
      4) Write chunks into:
           output_folder/merged-chunks/merged/
           output_folder/merged-chunks/merged-str/
           output_folder/merged-chunks/labels/
    """
    # base_out = os.path.join(output_folder, "merged-chunks")
    merged_dir       = os.path.join(output_folder, "Merged")
    merged_str_dir   = os.path.join(output_folder, "Merged-str")
    labels_dir       = os.path.join(output_folder, "labels")

    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(merged_str_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(data_folder), desc="Chunking (raw/stripped/labels)", unit="file"):
        if not file_name.endswith(".json"):
            continue

        base_name = os.path.splitext(file_name)[0]
        data_path = os.path.join(data_folder, file_name)
        label_path = os.path.join(label_folder, f"{base_name}_labels.json")

        if not os.path.exists(label_path):
            print(f"[Skip] No matching label file for {file_name}")
            continue

        try:
            # 1) Load raw data
            with open(data_path, 'r', encoding='utf-8') as f_data:
                raw_data = json.load(f_data)
            if not isinstance(raw_data, list):
                print(f"[Skip] {file_name} is not a list of objects")
                continue

            # 2) Load labels
            with open(label_path, 'r', encoding='utf-8') as f_lbl:
                labels = json.load(f_lbl)
            if not isinstance(labels, list):
                print(f"[Skip] {base_name}_labels.json is not a list")
                continue
            if len(labels) != len(raw_data):
                print(f"[Skip] Length mismatch: {file_name} has {len(raw_data)} rows but labels has {len(labels)}")
                continue

            # 3) Create stripped data (remove '@@@_' prefixes)
            stripped_data = strip_prefix(raw_data)

            # 4) Chunk all three lists in parallel
            chunk_data_in_parallel(
                raw_data,
                stripped_data,
                labels,
                base_name,
                merged_dir,
                merged_str_dir,
                labels_dir
            )

        except json.JSONDecodeError as e:
            print(f"[Error] JSON decode issue in {file_name}: {e}")
        except Exception as e:
            print(f"[Error] Unexpected issue in {file_name}: {e}")


if __name__ == "__main__":
    # ─── Example usage ──────────────────────────────
    data_folder   = r"C:\...\Spider_Beaver-merged\Merged"
    label_folder  = r"C:\...\Spider_Beaver-merged\labels"
    output_folder = r"C:\...\Spider_Beaver-merged-chunked"

    print("🚀 Starting chunking of raw, stripped, and label JSONs…")
    process_json_files_with_labels(data_folder, label_folder, output_folder)
    print("✅ Done.")

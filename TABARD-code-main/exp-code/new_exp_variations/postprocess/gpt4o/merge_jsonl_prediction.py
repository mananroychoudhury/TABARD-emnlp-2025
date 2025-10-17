import os
import json
import ast
import re
import logging
from collections import defaultdict
from tqdm.auto import tqdm

# ─── LOGGER SETUP ───────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



# Paths
GROUNDTRUTH_ROOT = r"..dataset"
GPT_OUTPUT_ROOT = r"gpt-output"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def extract_base_and_range(filename: str):
    m = re.match(r"(.+)_chunk_(\d+)_(\d+)\.json", filename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def merge_chunks_in_folder(folder_path: str, output_folder: str):
    for root, _, files in os.walk(folder_path):
        grouped = defaultdict(list)
        for fname in files:
            if fname.endswith(".json") and "_chunk_" in fname:
                base, start = extract_base_and_range(fname)
                if base is not None:
                    full = os.path.join(root, fname)
                    grouped[(base, root)].append((start, full))

        for (base, subdir), chunks in grouped.items():
            sorted_chunks = sorted(chunks, key=lambda x: x[0])
            merged = []
            for _, path in sorted_chunks:
                with open(path, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))

            rel_sub = os.path.relpath(subdir, folder_path)
            save_dir = os.path.join(output_folder, rel_sub)
            os.makedirs(save_dir, exist_ok=True)

            out_fname = f"{base}.json"
            out_path = os.path.join(save_dir, out_fname)
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(merged, outf, indent=4, ensure_ascii=False)
            logger.info(f"Merged chunks into {out_path}")


# ─── TRANSFORM SINGLE CHUNK ─────────────────────────────────────────────────
def transform_file(gt_path: str, anomalies: list, output_path: str):
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    except Exception as e:
        logger.error(f"Reading GT file {gt_path}: {e}")
        return

    # Initialize all fields to "No"
    for row in gt_data:
        for key in row:
            row[key] = "No"
    # Apply predicted anomalies
    for row_idx, field in anomalies:
        if 0 <= row_idx < len(gt_data) and field in gt_data[row_idx]:
            gt_data[row_idx][field] = "Yes"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=4)
    except Exception as e:
        logger.error(f"Saving prediction chunk {output_path}: {e}")


# ─── POSTPROCESS GPT PREDICTIONS ─────────────────────────────────────────────
def postprocess_fold_batch(fold: str, batch: str):
    # Paths for GPT JSONL and GT chunks
    llama_jsonl = os.path.join(
        GPT_OUTPUT_ROOT,
        f"output_folder-{fold}",
        f"{batch}.jsonl"
    )
    gt_dir = os.path.join(
        GROUNDTRUTH_ROOT,
        fold,
        "Merged-chunked",
        "Merged-yes-no",
        batch
    )
    output_dir = os.path.join(
        GPT_OUTPUT_ROOT,
        f"gpt-prediction-chunks/{fold}/{batch}"
    )
    os.makedirs(output_dir, exist_ok=True)

    prediction_dict = {}
    if os.path.isfile(llama_jsonl):
        with open(llama_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    # Extract base filename
                    custom_id = data.get("custom_id", "")
                    base_filename = os.path.basename(custom_id)

                    # Normalize GT filename
                    if base_filename.endswith("_updated.json_yes_no.json"):
                        gt_filename = base_filename.replace("_updated.json_yes_no.json", "_yes_no.json")
                    elif base_filename.endswith("_updated.json"):
                        gt_filename = base_filename.replace("_updated.json", "_yes_no.json")
                    elif "chunk" in base_filename:
                        gt_filename = base_filename.replace("_updated_", "_yes_no_")
                        if gt_filename.endswith(".json_yes_no.json"):
                            gt_filename = gt_filename.replace(".json_yes_no.json", ".json")
                    else:
                        gt_filename = base_filename.replace(".json", "_yes_no.json")

                    # Extract response text
                    gpt_resp = (
                        data.get("response", {})
                        .get("body", {})
                        .get("choices", [])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    # Parse anomalies
                    # Try tuple pattern
                    matches = re.findall(r"\(\s*(\d+)\s*,\s*['\"](.+?)['\"]\s*\)", gpt_resp)
                    if matches:
                        anomalies = [(int(i), label.strip()) for i, label in matches]
                        prediction_dict[gt_filename] = anomalies
                    else:
                        # Fallback to list literal
                        m = re.search(r"\[.*\]", gpt_resp, re.DOTALL)
                        if m:
                            try:
                                lst = ast.literal_eval(m.group())
                                anomalies = [(int(i), str(label).strip()) for i, label in lst]
                                prediction_dict[gt_filename] = anomalies
                            except Exception as e:
                                logger.error(f"Parsing list on line {idx}: {e}")
                        else:
                            logger.warning(f"Unrecognized format in line {idx}: {gpt_resp}")
                except Exception as e:
                    logger.error(f"Error processing line {idx} of {llama_jsonl}: {e}")
    else:
        logger.warning(f"No JSONL found at {llama_jsonl}")
        return

    # Apply to each GT chunk
    if os.path.isdir(gt_dir):
        for gt_fname in os.listdir(gt_dir):
            if not gt_fname.endswith(".json"):
                continue
            gt_path = os.path.join(gt_dir, gt_fname)
            out_path = os.path.join(output_dir, gt_fname)
            anomalies = prediction_dict.get(gt_fname, [])
            transform_file(gt_path, anomalies, out_path)
    else:
        logger.warning(f"GT directory not found: {gt_dir}")


# ─── CONFIGURATION ──────────────────────────────────────────────────────────
FOLDS = ["FetaQA-merged", "Spider_Beaver-merged", "wikiTQ-merged"]
BATCHES = ["museve", "sevcot"] ### Add l1_cot ....


# DIR = ["variation_1","variation_2","variation_3"]
# FOLDS = ["FetaQA", "Spider_Beaver", "wikiTQ"]
# BATCHS = ['museve','sevcot']

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # for dir in tqdm(DIR,desc="Postprocessing DIRs"):
    for fold in tqdm(FOLDS, desc="Postprocessing Folds"):
        for batch in tqdm(BATCHES, desc=f" Processing {fold}"):
            postprocess_fold_batch(fold, batch)
            # postprocess_fold_batch(dir,fold, batch)

    # Merge GPT-predicted chunks into full JSON per table
    # for dir in tqdm(DIR, desc="Postprocessing DIR", unit="fold"):
    for fold in tqdm(FOLDS, desc="Merging Fold Predictions"):
        for batch in BATCHES:
            # chunk_folder = os.path.join(
            #     GPT_OUTPUT_ROOT,
            #     f"gpt-prediction-chunks/{dir}/{fold}/{batch}/predicted-chunked/predicted-yes-no"
            # )
            # merged_folder = os.path.join(
            #     GPT_OUTPUT_ROOT,
            #     f"gpt-prediction-merged/{dir}/{fold}/{batch}/predicted-chunked/predicted-yes-no"
            # )

            chunk_folder = os.path.join(
                GPT_OUTPUT_ROOT,
                f"gpt-prediction-chunks/{fold}/{batch}/predicted-chunked/predicted-yes-no"
            )
            merged_folder = os.path.join(
                GPT_OUTPUT_ROOT,
                f"gpt-prediction-merged/{fold}/{batch}/predicted-chunked/predicted-yes-no"
            )
            if os.path.isdir(chunk_folder):
                merge_chunks_in_folder(chunk_folder, merged_folder)

    logger.info("GPT postprocessing and merge completed.")

if __name__ == "__main__":
    main()

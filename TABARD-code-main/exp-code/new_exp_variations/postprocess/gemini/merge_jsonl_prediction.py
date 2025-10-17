# ── main_full_pipeline.py ────────────────────────────────────────────────────

import os
import json
import re
import logging
from collections import defaultdict
from tqdm.auto import tqdm

from src.logger import setup_custom_logger

# ─── LOGGER SETUP ───────────────────────────────────────────────────────────
logger = setup_custom_logger(
    logfile_name="gemini_postprocess.log",
    level=logging.INFO,
    log_dir="log"
)

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

# 1) Folds (datasets) and batch names
FOLDS  = ["FetaQA-merged", "Spider_Beaver-merged", "wikiTQ-merged"]
BATCHS = ['l1_cot','l1_wcot','l2_cot','l2_wcot','l4_cot','l4_wcot','museve','sevcot']


# DIR = ["variation_1","variation_2","variation_3"]
# FOLDS = ["FetaQA", "Spider_Beaver", "wikiTQ"]
# BATCHS = ['museve','sevcot']

# 3) Paths to ground-truth and prediction JSONL
GROUNDTRUTH_ROOT = r"..dataset/"  # contains <batch> subfolders
GEMINI_OUTPUT_ROOT = r"predicitons\gemini"             # contains output_folder-<fold>/<batch>.jsonl

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def extract_base_and_range(filename: str):
    """
    From a filename like "tableA_yes_no_chunk_0_30.json", returns ("tableA_yes_no", 0).
    """
    m = re.match(r"(.+)_chunk_(\d+)_(\d+)\.json", filename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def merge_chunks_in_folder(folder_path: str, output_folder: str):
    """
    Merge all chunked JSONs under `folder_path` into single JSONs under `output_folder`.
    Chunk filenames look like <base>_chunk_<start>_<end>.json (e.g. tableA_yes_no_chunk_0_30.json).
    The merged output is <base>.json (e.g. tableA_yes_no.json).
    """
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

            # Determine relative subdirectory under folder_path
            rel_sub = os.path.relpath(subdir, folder_path)
            save_dir = os.path.join(output_folder, rel_sub)
            os.makedirs(save_dir, exist_ok=True)

            out_fname = f"{base}.json"  # e.g. tableA_yes_no.json
            out_path = os.path.join(save_dir, out_fname)
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(merged, outf, indent=4, ensure_ascii=False)
            logger.info(f"Merged chunks into {out_path}")


def transform_file(gt_path: str, anomalies: list, output_path: str):
    """
    Read a ground-truth yes/no chunk (list of dicts). Set all fields to "No",
    then set to "Yes" where anomalies indicate. Write result to output_path.
    """
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    except Exception as e:
        logger.error(f"[ERROR] Reading GT file {gt_path}: {e}")
        return

    for row in gt_data:
        for key in row:
            row[key] = "No"
    for row_idx, field in anomalies:
        if 0 <= row_idx < len(gt_data) and field in gt_data[row_idx]:
            gt_data[row_idx][field] = "Yes"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=4)
    except Exception as e:
        logger.error(f"[ERROR] Saving prediction chunk {output_path}: {e}")


def postprocess_fold_batch(dir,fold: str, batch: str):
    """
    1) Read predictions/<fold>/<batch>/prediction.jsonl, build mapping
       of ground-truth chunk filename → anomalies list.
    2) For each ground-truth chunk in
         <fold>/Merged-chunked/Merged-yes-no/<batch>/,
       apply anomalies and write predicted chunk to
         gemini-prediction-chunks/prediction-<fold>/<batch>/<gt_filename>.
    """
    # Paths
    # gemini_jsonl = os.path.join(GEMINI_OUTPUT_ROOT,f"{dir}",f"{fold}", f"{batch}","predictions.jsonl")
    # output_dir   = os.path.join(GEMINI_OUTPUT_ROOT, f"{dir}",f"{fold}", f"{batch}","predicted-chunked", "predicted-yes-no")
    # gt_dir       = os.path.join(GROUNDTRUTH_ROOT,dir,fold, "Merged-chunked", "Merged-yes-no")

    gemini_jsonl = os.path.join(GEMINI_OUTPUT_ROOT,f"{fold}", f"{batch}","predictions.jsonl")
    output_dir   = os.path.join(GEMINI_OUTPUT_ROOT, f"{fold}", f"{batch}","predicted-chunked", "predicted-yes-no")
    gt_dir       = os.path.join(GROUNDTRUTH_ROOT,fold, "Merged-chunked", "Merged-yes-no")

    os.makedirs(output_dir, exist_ok=True)

    prediction_dict = {}
    if os.path.isfile(gemini_jsonl):
        with open(gemini_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    gemini_id = data.get("id", "")
                    

                    gt_fname = gemini_id.replace("_updated_","_yes_no_")+ ".json"
                    text = (
                        data.get("response", {})
                            .get("candidates", [])[0]
                            .get("content", {})
                            .get("parts", [])[0]
                            .get("text", "")
                    )
                    try:
                        anomaly_list = json.loads(text)
                        anomalies = [(int(item["index"]), item["anomaly_column"].strip())
                                     for item in anomaly_list]
                        prediction_dict[gt_fname] = anomalies
                    except Exception as e:
                        logger.error(f"[ERROR] Parsing JSON on line {idx} of {gemini_jsonl}: {e}")

                except json.JSONDecodeError as jde:
                    logger.error(f"[ERROR] JSON decode on line {idx} of {gemini_jsonl}: {jde}")
                except Exception as e:
                    logger.error(f"[ERROR] Unexpected on line {idx} of {gemini_jsonl}: {e}")
    else:
        logger.warning(f"No JSONL found at {gemini_jsonl}, skipping {fold}/{batch}")
        return

    # Apply to each GT chunk
    if os.path.isdir(gt_dir):
        for gt_fname in os.listdir(gt_dir):
            if not gt_fname.endswith(".json"):
                continue
            gt_path     = os.path.join(gt_dir, gt_fname)
            out_path    = os.path.join(output_dir, gt_fname)
            anomalies   = prediction_dict.get(gt_fname, [])
            transform_file(gt_path, anomalies, out_path)
    else:
        logger.warning(f"GT directory not found: {gt_dir}")


# ─── MAIN WORKFLOW ──────────────────────────────────────────────────────────

def main():
    # Step 1: Postprocess predictions to get chunked “Yes/No” files
    # for dir in tqdm(DIR, desc="Postprocessing DIR", unit="fold"):
    for fold in tqdm(FOLDS, desc="Postprocessing Folds", unit="fold"):
        for batch in tqdm(BATCHS, desc=f" Processing batches in '{fold}'", unit="batch"):
            postprocess_fold_batch(dir,fold, batch)

    # Step 2: Merge chunked predictions into a single JSON per base table
    # for dir in tqdm(DIR, desc="Postprocessing DIR", unit="fold"):
    for fold in tqdm(FOLDS, desc="Merging Fold Predictions", unit="fold"):
        for batch in BATCHS:
            # chunk_folder = os.path.join(GEMINI_OUTPUT_ROOT,f"{dir}",f"{fold}", f"{batch}","predicted-chunked", "predicted-yes-no")
            # merged_folder = os.path.join(GEMINI_OUTPUT_ROOT,f"{dir}", f"{fold}", f"{batch}","predicted-merged", "predicted-yes-no")

            chunk_folder = os.path.join(GEMINI_OUTPUT_ROOT,f"{fold}", f"{batch}","predicted-chunked", "predicted-yes-no")
            merged_folder = os.path.join(GEMINI_OUTPUT_ROOT, f"{fold}", f"{batch}","predicted-merged", "predicted-yes-no")
            if os.path.isdir(chunk_folder):
                merge_chunks_in_folder(chunk_folder, merged_folder)

    logger.info("All steps (postprocess + merge) completed.")


if __name__ == "__main__":
    main()

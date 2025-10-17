import os
import json
from collections import OrderedDict
from tqdm.auto import tqdm
def contains_anomaly(obj):
    """
    Recursively check if any leaf value in obj contains the substring '@@@_'.
    """
    if isinstance(obj, str):
        return "@@@_" in obj
    elif isinstance(obj, dict):
        return any(contains_anomaly(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(contains_anomaly(v) for v in obj)
    else:
        return False

def merge_variations_for_file(fname, variation_dirs, out_merged, out_labels):
    """
    Given a single file_name (e.g. "10037.json") and a list of folders
    (each containing either GT‐rows or an underperforming category’s updated rows),
    this merges all rows, dedupes identical records, and writes:
      - Merged/fname        (the merged array of unique rows)
      - labels/fname_labels.json  (which row came from which folder)
    """
    unique_rows = OrderedDict()  # key→{"record": row_dict, "folders": set(...)}

    for vd in variation_dirs:
        folder_name = os.path.basename(vd)
        path = os.path.join(vd, fname)
        if not os.path.exists(path):
            # If somehow the file is missing under this folder, skip
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                # serialize to a stable key
                key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if key not in unique_rows:
                    unique_rows[key] = {"record": row, "folders": set()}
                unique_rows[key]["folders"].add(folder_name)

    # build merged list in insertion order
    merged_list = [v["record"] for v in unique_rows.values()]

    # build labels array
    labels = []
    for idx, v in enumerate(unique_rows.values()):
        rec = v["record"]
        is_anom = contains_anomaly(rec)
        folders = sorted(v["folders"]) if is_anom else []
        labels.append({
            "index": idx,
            "folders": folders
        })

    os.makedirs(out_merged, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # write merged JSON
    with open(os.path.join(out_merged, fname), "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    # write labels JSON
    base, _ = fname.rsplit(".", 1)
    label_name = f"{base}_labels.json"
    with open(os.path.join(out_labels, label_name), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def main():
    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]
    map_dict = {
        "FeTaQA": ["Factual_Anomaly_FeTaQA", "Data_Consistency_Anomaly_FeTaQA", "Security_Anomaly_FeTaQA"],
        "Spider_Beaver": ["Temporal_Anomaly_Spider_Beaver", "Factual_Anomaly_Spider_Beaver", "Normalization_Anomaly_Spider_Beaver"],
        "WikiTQ": ["Factual_Anomalies_WikiTQ", "Data_Consistency_Anomalies_WikiTQ", "Logical_Anomalies_WikiTQ"]
        }
    for folder in FOLDS:
        ### ─── CONFIGURE THESE PATHS & PARAMETERS ─── ###
        GT_ROOT = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"   # must contain subfolders C1, C2, …
        VARIATION_ROOT = f"path_to_dataset/variation_2/{folder}"

        # List exactly which categories are “underperforming.”
        # Only these folders will contribute their *_updated.json as‐is.
        
        UNDERPERFORMING = map_dict[folder]
        # UNDERPERFORMING = ["Factual_Anomaly_FeTaQA", "Data_Consistency_Anomaly_FeTaQA", "Security_Anomaly_FeTaQA"]
        # UNDERPERFORMING = ["Temporal_Anomaly_Spider_Beaver", "Factual_Anomaly_Spider_Beaver", "Normalization_Anomaly_Spider_Beaver"]
        # UNDERPERFORMING = ["Factual_Anomalies_WikiTQ", "Data_Consistency_Anomalies_WikiTQ", "Logical_Anomalies_WikiTQ"]    
        
        ###############################################

        # 1) Enumerate all category subfolders
        category_names = sorted(
            [d.replace(f"_{folder}","") for d in os.listdir(CATEGORIES_ROOT)
            if os.path.isdir(os.path.join(CATEGORIES_ROOT, d)) and d != "Ground_truth" and d in map_dict[folder] ]
        )
        if not category_names:
            raise RuntimeError("No category subfolders found under CATEGORIES_ROOT.")

        # 2) Enumerate all GT JSON filenames
        all_files = sorted(
            fn for fn in os.listdir(GT_ROOT)
            if fn.lower().endswith(".json")
        )

        # 3) Prepare output directories
        for cat in category_names:
            os.makedirs(os.path.join(VARIATION_ROOT, cat), exist_ok=True)
        merged_dir = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        summary_dict = {}
        # 4) Process each GT file one by one
        for fname in tqdm(all_files):
            # 4a) Load GT rows
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # We will build a list of “folders” that actually supply rows:
            #  - “GT” folder (always)
            #  - each underperforming category that has an updated file
            used_folders = []

            # 4b) Always write GT → Variation_root/GT/fname
            #     (or simply refer to GT_ROOT directly, but for consistency
            #      we copy GT into Variation_root/GT/)
            # gt_outdir = os.path.join(VARIATION_ROOT, "GT")
            # os.makedirs(gt_outdir, exist_ok=True)
            # with open(os.path.join(gt_outdir, fname), "w", encoding="utf-8") as f:
            #     json.dump(gt_rows, f, ensure_ascii=False, indent=2)
            # used_folders.append(gt_outdir)

            # 4c) For each category:
            for cat in category_names:
                cat_dir = os.path.join(CATEGORIES_ROOT, cat+f"_{folder}")
                # Build the expected “_updated.json” filename
                updated_name = fname.replace(".json", "_updated.json")
                updated_path = os.path.join(cat_dir, updated_name)

                if cat+f"_{folder}" in UNDERPERFORMING and os.path.exists(updated_path):
                    # If this category is underperforming, load its updated JSON as‐is:
                    with open(updated_path, "r", encoding="utf-8") as f:
                        cat_data = json.load(f)
                        cat_rows = cat_data if isinstance(cat_data, list) else [cat_data]

                    # Write it directly under Variation_root/{cat}/fname
                    outdir = os.path.join(VARIATION_ROOT, cat)
                    os.makedirs(outdir, exist_ok=True)
                    with open(os.path.join(outdir, fname), "w", encoding="utf-8") as f:
                        json.dump(cat_rows, f, ensure_ascii=False, indent=2)

                    used_folders.append(outdir)

                else:
                    # Either this category is not underperforming, or its updated file is missing.
                    # In either case, we treat it exactly like GT (so we DO NOT rewrite a file here).
                    # Because we already included “GT” above, we do not need to copy GT again.
                    # Nothing to do → it won’t add any new rows when we merge.
                    continue

            # 4d) Now merge all “used_folders” for this fname
            merge_variations_for_file(
                fname,
                used_folders,
                merged_dir,
                labels_dir
            )



            # … inside your for‐fname loop …
            used_short = [os.path.basename(d) for d in used_folders]
            summary_dict[fname] = used_short

            # print(f" → Merged {fname} from folders: {[os.path.basename(d) for d in used_folders]}")

        with open(os.path.join(VARIATION_ROOT, "summary.json"), "w", encoding="utf-8") as sf:
            json.dump(summary_dict, sf, ensure_ascii=False, indent=2)
        print("All files processed.")

if __name__ == "__main__":
    main()

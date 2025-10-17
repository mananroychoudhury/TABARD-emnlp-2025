import os
import json
import math
from collections import OrderedDict
from tqdm.auto import tqdm

def contains_anomaly(obj):
    """
    Recursively check if any leaf value in obj contains '@@@_'.
    """
    if isinstance(obj, str):
        return "@@@_" in obj
    elif isinstance(obj, dict):
        return any(contains_anomaly(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(contains_anomaly(v) for v in obj)
    else:
        return False

def find_perturbed_cells(table_rows):
    """
    Given a list of dicts (rows), return a list of "R{row}C{col}" for every cell
    whose value starts with '@@@_'. Rows and cols are 1-based, scanning in key order.
    """
    perturbed = []
    for r_idx, row in enumerate(table_rows, start=1):
        for c_idx, (col_key, col_val) in enumerate(row.items(), start=1):
            if isinstance(col_val, str) and col_val.startswith("@@@_"):
                perturbed.append(f"R{r_idx}C{c_idx}")
    return perturbed

def build_stratified_variation(gt_rows, cat_rows, perturbed_cells, keep_cells):
    """
    Starting from cat_rows (which may contain '@@@_' in some cells),
    keep only those cells declared in keep_cells (a list of "R{r}C{c}" strings).
    All other perturbed cells get overwritten by the GT value.

    - gt_rows: list of dicts (ground truth).
    - cat_rows: list of dicts (perturbed).
    - perturbed_cells: list of all "R{r}C{c}" in cat_rows.
    - keep_cells: subset of perturbed_cells that we want to leave as '@@@_'.

    Returns a brand‐new list of dicts (same shape) where any perturbed cell not
    in keep_cells is replaced by the corresponding gt_rows[r-1][col_key].
    """
    varied = json.loads(json.dumps(cat_rows, ensure_ascii=False))
    keep_set = set(keep_cells)

    for r_idx, (gt_row, var_row) in enumerate(zip(gt_rows, varied), start=1):
        for c_idx, key in enumerate(var_row.keys(), start=1):
            cell_id = f"R{r_idx}C{c_idx}"
            if cell_id in perturbed_cells and cell_id not in keep_set:
                var_row[key] = gt_row[key]
    return varied

def merge_variations_for_file(fname, variation_dirs, out_merged, out_labels):
    """
    Merge all JSON rows under each folder in variation_dirs, dedupe identical dicts,
    and write:
      - out_merged/fname
      - out_labels/<fname_without_ext>_labels.json

    For every unique row, if it contains '@@@_' we record which folders it came from;
    otherwise, folders=[]. 
    """
    unique_rows = OrderedDict()

    for vd in variation_dirs:
        folder = os.path.basename(vd)
        path = os.path.join(vd, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if key not in unique_rows:
                    unique_rows[key] = {"record": row, "folders": set()}
                unique_rows[key]["folders"].add(folder)

    merged_list = [v["record"] for v in unique_rows.values()]

    labels = []
    for idx, v in enumerate(unique_rows.values()):
        rec = v["record"]
        is_anom = contains_anomaly(rec)
        folders = sorted(v["folders"]) if is_anom else []
        labels.append({"index": idx, "folders": folders})

    os.makedirs(out_merged, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    with open(os.path.join(out_merged, fname), "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    base, _ = fname.rsplit(".", 1)
    label_name = f"{base}_labels.json"
    with open(os.path.join(out_labels, label_name), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def main():
    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]

    map_dict = {
            "FeTaQA": {
                "UNDER": ["Factual_Anomaly_FeTaQA", "Data_Consistency_Anomaly_FeTaQA", "Security_Anomaly_FeTaQA"],
                "MID": ["Calculation_Based_Anomaly_FeTaQA", "Logical_Anomaly_FeTaQA"],
                "OVER": ["Temporal_Anomaly_FeTaQA", "Normalization_Anomaly_FeTaQA", "Value_Anomaly_FeTaQA"]
            },
            "Spider_Beaver": {
                "UNDER": ["Temporal_Anomaly_Spider_Beaver", "Factual_Anomaly_Spider_Beaver", "Normalization_Anomaly_Spider_Beaver"],
                "MID": ["Data_Consistency_Anomaly_Spider_Beaver", "Security_Anomaly_Spider_Beaver"],
                "OVER": ["Logical_Anomaly_Spider_Beaver", "Calculation_Based_Anomaly_Spider_Beaver", "Value_Anomaly_Spider_Beaver"]
            },
            "WikiTQ": {
                "UNDER": ["Factual_Anomalies_WikiTQ", "Data_Consistency_Anomalies_WikiTQ", "Logical_Anomalies_WikiTQ"],
                "MID": ["Temporal_Anomalies_WikiTQ", "Security_Anomalies_WikiTQ"],
                "OVER": ["Normalization_Anomalies_WikiTQ", "Value_Anomalies_WikiTQ", "Calculation_Based_Anomalies_WikiTQ"]
            }
        }
    for folder in FOLDS:
        ### ─────── CONFIGURATION ─────── ###
        GT_ROOT         = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"
        VARIATION_ROOT  = f"path_to_dataset/variation_3/{folder}"
        
        
        # Performance groups: only these folder‐names will be considered
        PERFORMANCE_GROUPS = map_dict[folder]
        # Fixed sampling fractions per group
        SAMPLING_FRACTIONS = {
            "UNDER": 0.6,
            "MID":   0.3,
            "OVER":  0.1
        }
        ###################################

        # 1) Identify exactly which folders exist and are in one of the groups
        #    Build a flat list of valid categories to process
        valid_cats = []
        for grp in ["UNDER", "MID", "OVER"]:
            for cat in PERFORMANCE_GROUPS[grp]:
                folder_path = os.path.join(CATEGORIES_ROOT, cat)
                # Only include it if the folder actually exists
                if os.path.isdir(folder_path):
                    valid_cats.append(cat)
        valid_cats = sorted(set(valid_cats))  # dedupe & sort

        if not valid_cats:
            raise RuntimeError("None of the specified performance‐group folders exist under CATEGORIES_ROOT.")

        # 2) Enumerate all GT JSON filenames
        all_files = sorted(fn for fn in os.listdir(GT_ROOT) if fn.lower().endswith(".json"))

        # 3) Prepare the output directory structure
        for cat in valid_cats:
            os.makedirs(os.path.join(VARIATION_ROOT, cat), exist_ok=True)

        merged_dir = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        summary = {}

        # 4) Loop over every ground‐truth file
        for fname in tqdm(all_files):
            # 4a) Load GT rows
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # 4b) For each valid category, attempt to load its “_updated.json”
            perturbed_lists = {}
            cat_rows_dict   = {}

            for cat in valid_cats:
                cat_folder = os.path.join(CATEGORIES_ROOT, cat)
                upd_name   = fname.replace(".json", "_updated.json")
                upd_path   = os.path.join(cat_folder, upd_name)

                if os.path.exists(upd_path):
                    with open(upd_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cat_rows = data if isinstance(data, list) else [data]
                    perturbed = find_perturbed_cells(cat_rows)
                else:
                    # If no updated.json, skip this category entirely:
                    #   we will not include it anywhere
                    continue

                cat_rows_dict[cat] = cat_rows
                perturbed_lists[cat] = perturbed

            # If no valid category has an updated file for this fname, skip merging:
            if not perturbed_lists:
                print(f"Skipping {fname} – no updated.json among {valid_cats}.")
                continue

            # 4c) Decide how many perturbed cells to keep per category
            chosen_cells_map = {}
            for cat, cat_rows in cat_rows_dict.items():
                perturbed = perturbed_lists[cat]
                pcount = len(perturbed)

                # Determine group membership
                if cat in PERFORMANCE_GROUPS["UNDER"]:
                    frac = SAMPLING_FRACTIONS["UNDER"]
                elif cat in PERFORMANCE_GROUPS["MID"]:
                    frac = SAMPLING_FRACTIONS["MID"]
                else:  # must be in PERFORMANCE_GROUPS["OVER"]
                    frac = SAMPLING_FRACTIONS["OVER"]

                # Number to keep = ceil(pcount * fraction)
                keep_num = math.ceil(pcount * frac) if pcount > 0 else 0
                keep_cells = perturbed[:keep_num]
                chosen_cells_map[cat] = keep_cells

                # Build and write the variation
                var_rows = build_stratified_variation(gt_rows, cat_rows, perturbed, keep_cells)
                outdir = os.path.join(VARIATION_ROOT, cat)
                with open(os.path.join(outdir, fname), "w", encoding="utf-8") as f:
                    json.dump(var_rows, f, ensure_ascii=False, indent=2)

            # 4d) Merge only those categories that actually had an updated file
            variation_dirs = [os.path.join(VARIATION_ROOT, cat) for cat in chosen_cells_map.keys()]
            merge_variations_for_file(fname, variation_dirs, merged_dir, labels_dir)

            # 4e) Build the summary entry
            #    pertubuted_SAMPLE_COUNTS: sum of perturbed counts per performance group
            group_totals = {"UNDER": 0, "MID": 0, "OVER": 0}
            for grp in ["UNDER", "MID", "OVER"]:
                for cat in PERFORMANCE_GROUPS[grp]:
                    if cat in perturbed_lists:
                        group_totals[grp] += len(perturbed_lists[cat])

            summary[fname] = {
                "PERFORMANCE_GROUPS": PERFORMANCE_GROUPS,
                "SAMPLING_RATIOS": SAMPLING_FRACTIONS,
                "pertubuted_SAMPLE_COUNTS": {
                    "UNDER": group_totals["UNDER"],
                    "MID":   group_totals["MID"],
                    "OVER":  group_totals["OVER"]
                },
                "TOTAL_PERTURBED_CELLS": {
                    cat: len(perturbed_lists.get(cat, []))
                    for cat in chosen_cells_map.keys()
                },
                "CHOSEN_CELLS": chosen_cells_map
            }

            print(f"Processed {fname}: kept counts → " +
                ", ".join(f"{cat}:{len(chosen_cells_map[cat])}" for cat in chosen_cells_map))

        # 5) Write top‐level summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nFinished. Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()

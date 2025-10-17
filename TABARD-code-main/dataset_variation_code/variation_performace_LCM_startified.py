import os
import json
import math
import random
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

def sample_via_two_step(perturbed_lists, D, seed=None):
    """
    LCM‐equivalent sampling across a group of categories:
      1) Repeatedly pick a category uniformly (among those with pcount>0);
      2) Pick one index within that category uniformly;
      3) Add to 'chosen' set until its size reaches D.
    Returns a dict: { category_name: set(indices_to_keep) }.
    """
    if seed is not None:
        random.seed(seed)

    # Only categories with at least one perturbed cell
    valid_cats = [cat for cat, lst in perturbed_lists.items() if len(lst) > 0]
    total_cells = sum(len(perturbed_lists[k]) for k in valid_cats)
    if D >= total_cells:
        # If D >= total unique cells, just keep all cells in every category
        return {cat: set(range(len(perturbed_lists[cat]))) for cat in perturbed_lists}

    chosen = set()
    while len(chosen) < D:
        cat = random.choice(valid_cats)  # pick a category uniformly
        idx = random.randrange(len(perturbed_lists[cat]))  # pick one index within it
        chosen.add((cat, idx))

    chosen_cells_map = {cat: set() for cat in perturbed_lists}
    for (cat, idx) in chosen:
        chosen_cells_map[cat].add(idx)

    return chosen_cells_map

def main():
    random.seed(2025)  # for reproducibility

    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]

    map_dict = {
        "FeTaQA": {
            "UNDER": ["Factual_Anomaly_FeTaQA", "Data_Consistency_Anomaly_FeTaQA", "Security_Anomaly_FeTaQA"],
            "MID":   ["Calculation_Based_Anomaly_FeTaQA", "Logical_Anomaly_FeTaQA"],
            "OVER":  ["Temporal_Anomaly_FeTaQA", "Normalization_Anomaly_FeTaQA", "Value_Anomaly_FeTaQA"]
        },
        "Spider_Beaver": {
            "UNDER": ["Temporal_Anomaly_Spider_Beaver", "Factual_Anomaly_Spider_Beaver", "Normalization_Anomaly_Spider_Beaver"],
            "MID":   ["Data_Consistency_Anomaly_Spider_Beaver", "Security_Anomaly_Spider_Beaver"],
            "OVER":  ["Logical_Anomaly_Spider_Beaver", "Calculation_Based_Anomaly_Spider_Beaver", "Value_Anomaly_Spider_Beaver"]
        },
        "WikiTQ": {
            "UNDER": ["Factual_Anomalies_WikiTQ", "Data_Consistency_Anomalies_WikiTQ", "Logical_Anomalies_WikiTQ"],
            "MID":   ["Temporal_Anomalies_WikiTQ", "Security_Anomalies_WikiTQ"],
            "OVER":  ["Normalization_Anomalies_WikiTQ", "Value_Anomalies_WikiTQ", "Calculation_Based_Anomalies_WikiTQ"]
        }
    }

    for folder in FOLDS:
        ### ─────── CONFIGURATION ─────── ###
        GT_ROOT         = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"
        VARIATION_ROOT  = f"path_to_dataset/variation_2/{folder}"
        PERFORMANCE_GROUPS = map_dict[folder]
        ###################################

        # 1) Identify which categories actually exist (within the three performance groups)
        valid_cats = []
        for grp in ("UNDER", "MID", "OVER"):
            for cat in PERFORMANCE_GROUPS[grp]:
                cat_folder = os.path.join(CATEGORIES_ROOT, cat)
                if os.path.isdir(cat_folder):
                    valid_cats.append(cat)
        valid_cats = sorted(set(valid_cats))  # dedupe & sort

        if not valid_cats:
            raise RuntimeError("No specified performance‐group folders exist under CATEGORIES_ROOT.")

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

        # 4) Loop over every GT file
        for fname in tqdm(all_files, desc="Processing Files"):
            # 4a) Load GT rows
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # 4b) For each category that exists, load its *_updated.json and find perturbed cells
            perturbed_lists = {}   # cat -> [list of "R{r}C{c}"]
            cat_rows_dict   = {}   # cat -> [list of dicts]

            for cat in valid_cats:
                cat_folder = os.path.join(CATEGORIES_ROOT, cat)
                upd_name   = fname.replace(".json", "_updated.json")
                upd_path   = os.path.join(cat_folder, upd_name)

                if os.path.exists(upd_path):
                    with open(upd_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cat_rows = data if isinstance(data, list) else [data]
                    perturbed = find_perturbed_cells(cat_rows)
                    if perturbed:
                        cat_rows_dict[cat] = cat_rows
                        perturbed_lists[cat] = perturbed

            if not perturbed_lists:
                print(f"Skipping {fname} – no updated.json among {valid_cats}.")
                continue

            # 4c) For each performance group (UNDER, MID, OVER),
            #     gather only the categories in that group that have perturbations,
            #     then run LCM‐sampling to pick D_group = max(counts_in_group) distinct cells.
            chosen_cells_map = {}  # cat -> list of "R{r}C{c}" to keep

            for grp in ("UNDER", "MID", "OVER"):
                # Find which cats belong to this group AND actually have perturbed cells
                group_cats = [cat for cat in PERFORMANCE_GROUPS[grp] if cat in perturbed_lists]
                if not group_cats:
                    continue

                # Build a sub‐dict for this group
                sub_perturbed_lists = {cat: perturbed_lists[cat] for cat in group_cats}
                counts = [len(sub_perturbed_lists[cat]) for cat in group_cats]

                # Heuristic A: D_group = max(counts) (at least 1 if any category is nonempty)
                D_group = max(1, max(counts))

                # Run LCM‐equivalent sampling across these group_cats
                chosen_map = sample_via_two_step(sub_perturbed_lists, D_group, seed=2025)
                # chosen_map is { cat: set(indices) } for this group

                # Convert indices back into actual "R{r}C{c}" labels
                for cat in group_cats:
                    keep_indices = chosen_map[cat]
                    keep_cells = [ sub_perturbed_lists[cat][i] for i in keep_indices ]
                    chosen_cells_map[cat] = keep_cells

            # 4d) Build & write one “stratified‐variation” JSON for each category in chosen_cells_map
            variation_dirs = []
            for cat, keep_cells in chosen_cells_map.items():
                cat_rows    = cat_rows_dict[cat]
                perturbed   = perturbed_lists[cat]
                varied_rows = build_stratified_variation(gt_rows, cat_rows, perturbed, keep_cells)

                outdir = os.path.join(VARIATION_ROOT, cat)
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, fname)
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(varied_rows, f, ensure_ascii=False, indent=2)

                variation_dirs.append(outdir)

            # 4e) Merge those variation folders
            merge_variations_for_file(fname, variation_dirs, merged_dir, labels_dir)

            # 4f) Build the summary entry for this file
            #    record how many perturbed and how many kept per category
            summary[fname] = {
                "PERFORMANCE_GROUPS": PERFORMANCE_GROUPS,
                "CATEGORIES": list(chosen_cells_map.keys()),
                "PERTURBED_COUNTS": {cat: len(perturbed_lists.get(cat, [])) for cat in chosen_cells_map},
                "KEPT_COUNTS":      {cat: len(chosen_cells_map[cat]) for cat in chosen_cells_map},
                "CHOSEN_CELLS":     chosen_cells_map
            }

            kept_str = ", ".join(f"{cat}:{len(chosen_cells_map[cat])}" for cat in chosen_cells_map)
            print(f"Processed {fname}: kept counts → {kept_str}")

        # 5) Write the top‐level summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nFinished. Wrote summary → {summary_path}")

if __name__ == "__main__":
    main()

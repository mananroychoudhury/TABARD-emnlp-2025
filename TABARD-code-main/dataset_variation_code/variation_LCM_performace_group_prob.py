import os
import json
import math
import random
from collections import OrderedDict
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1) USER‐DEFINED “performance” score for each category:
#    You must fill in a real number for each category below.
#    For example, if "C1" has performance 0.8, "C2" has 0.6, etc.,
#    then you would define:
#
#       perf_score = {
#           "C1": 0.8,
#           "C2": 0.6,
#           "C3": 0.9,
#           … etc …
#       }
#
#    All categories that appear under "UNDER","MID","OVER" must be listed.
#    (These scores could come from some external evaluation.)

perf_score = {
    "FeTaQA": {
        "Calculation_Based_Anomaly": 63.17,
        "Factual_Anomaly": 36.78,
        "Normalization_Anomaly": 71.5,
        "Logical_Anomaly": 55.53,
        "Temporal_Anomaly": 55.23,
        "Security_Anomaly": 45.39,
        "Data_Consistency_Anomaly": 34.17,
        "Value_Anomaly": 77.54,
    },
    "Spider_Beaver": {
        "Temporal_Anomaly": 42.21,
        "Factual_Anomaly": 34.2,
        "Normalization_Anomaly": 45.46,
        "Data_Consistency_Anomaly": 36.83,
        "Security_Anomaly": 52.56,
        "Logical_Anomaly": 49.92,
        "Calculation_Based_Anomaly": 59.3,
        "Value_Anomaly": 70.71,
    },
    "WikiTQ": {
        "Factual_Anomalies":            33.65,
        "Data_Consistency_Anomalies":   39.12,
        "Logical_Anomalies":            50.18,
        "Temporal_Anomalies":           44.22,
        "Security_Anomalies":           58.27,
        "Normalization_Anomalies":      58.03,
        "Value_Anomalies":              71.52,
        "Calculation_Based_Anomalies":   76.22,
    }
}
# ─────────────────────────────────────────────────────────────────────────────

def contains_anomaly(obj):
    if isinstance(obj, str):
        return "@@@_" in obj
    elif isinstance(obj, dict):
        return any(contains_anomaly(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(contains_anomaly(v) for v in obj)
    else:
        return False

def find_perturbed_cells(table_rows):
    perturbed = []
    for r_idx, row in enumerate(table_rows, start=1):
        for c_idx, (col_key, col_val) in enumerate(row.items(), start=1):
            if isinstance(col_val, str) and col_val.startswith("@@@_"):
                perturbed.append(f"R{r_idx}C{c_idx}")
    return perturbed

# def build_stratified_variation(gt_rows, cat_rows, perturbed_cells, keep_cells):
#     varied = json.loads(json.dumps(cat_rows, ensure_ascii=False))
#     keep_set = set(keep_cells)
#     for r_idx, (gt_row, var_row) in enumerate(zip(gt_rows, varied), start=1):
#         for c_idx, key in enumerate(var_row.keys(), start=1):
#             cell_id = f"R{r_idx}C{c_idx}"
#             if cell_id in perturbed_cells and cell_id not in keep_set:
#                 var_row[key] = gt_row[key]
#     return varied

def build_stratified_variation(
    gt_rows,            # list of dicts (ground truth)
    category_rows,      # list of dicts (perturbed, possibly differing length)
    perturbed_cells,    # list of cell‐ID strings, e.g. ["R5C3","R6C3",…]
    chosen_cells_idx    # set of cell‐ID strings to KEEP (not revert)
):
    """
    Revert *all* perturbed cells EXCEPT those in chosen_cells_idx, by matching the other columns in GT.
    Parameters:
      - gt_rows:           [ {colkey: value, …}, … ]        (len = N_gt)
      - category_rows:     [ {colkey: value, …}, … ]        (len = N_cat)
      - perturbed_cells:   [ "R{r}C{c}", … ]                (len = P)
      - chosen_cells_idx:  { "R{r}C{c}", … }  ⊆ perturbed_cells
          (i.e. the cell‐IDs you want to leave as “@@@_…”; everything else you revert.)
    Returns:
      - varied: a deep copy of category_rows (length N_cat). For each cell_id ∈ perturbed_cells 
        that is NOT in chosen_cells_idx, we search GT by matching all *other* columns. If exactly 
        one GT row matches, we overwrite that one cell. Otherwise, we fall back to copying by index 
        if available, or leave as-is.
    """
    # 1) Deep‐copy so we can safely write:
    varied = json.loads(json.dumps(category_rows, ensure_ascii=False))

    # 2) Treat chosen_cells_idx as the set of cell‐ID strings to keep
    keep_set = set(chosen_cells_idx)

    # 3) Derive the list of column keys (we assume every row‐dict uses the same keys)
    if len(category_rows) > 0:
        all_col_keys = list(category_rows[0].keys())
    else:
        all_col_keys = []

    # 4) For each perturbed cell_id that is not in keep_set:
    for cell_id in perturbed_cells:
        if cell_id in keep_set:
            # leave this cell as "@@@_…"
            continue

        # Parse out row‐index and column‐index from "R{r}C{c}"
        try:
            r_str, c_str = cell_id.split("R")[1].split("C")
            r_idx = int(r_str)
            c_idx = int(c_str)
        except Exception:
            # bad format → skip
            continue

        # Validate column index
        if not (1 <= c_idx <= len(all_col_keys)):
            continue
        col_key = all_col_keys[c_idx - 1]

        # Validate row index in the perturbed table
        if not (1 <= r_idx <= len(varied)):
            continue

        # Gather the “other” columns from the perturbed row
        pert_row_dict = varied[r_idx - 1]
        other_cols = [k for k in all_col_keys if k != col_key]
        other_vals = {k: pert_row_dict[k] for k in other_cols}

        # Find all GT rows whose other_cols match exactly
        matches = []
        for gt_candidate in gt_rows:
            if all(gt_candidate.get(k) == other_vals.get(k) for k in other_cols):
                matches.append(gt_candidate)

        if len(matches) == 1:
            # Exactly one content‐match: revert this cell using that GT row
            matched_gt_row = matches[0]
            new_val = matched_gt_row.get(col_key)
            varied[r_idx - 1][col_key] = new_val
        else:
            # 0 or >1 matches → fallback by index if possible
            if 1 <= r_idx <= len(gt_rows):
                varied[r_idx - 1][col_key] = gt_rows[r_idx - 1].get(col_key)
            # otherwise leave it as "@@@_…"

    return varied



def merge_variations_for_file(fname, variation_dirs, out_merged, out_labels):
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

def categorize_scores(scores_dict):
    import numpy as np

    scores = list(scores_dict.values())

    # Dynamically calculate thresholds
    high_threshold = np.percentile(scores, 90)
    low_threshold = np.percentile(scores, 40)

    categories = {"OVER": [], "MID": [], "UNDER": []}

    for key, value in scores_dict.items():
        if value >= high_threshold:
            categories["OVER"].append(key)
        elif value >= low_threshold:
            categories["MID"].append(key)
        else:
            categories["UNDER"].append(key)

    return categories

def main():
    random.seed(2025)

    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]

    # ────────────────────────────────────────────────────────────────────────────
    # First pass: compute total perturbed cells PER CATEGORY, over ALL files.
    # This lets us compute p_under, p_mid, p_over before sampling any single file.
    # ────────────────────────────────────────────────────────────────────────────

    # map_dict still defines which categories belong to which group:
    # map_dict = {
    #     "FeTaQA": {
    #         "UNDER": ["Factual_Anomaly_FeTaQA", "Data_Consistency_Anomaly_FeTaQA", "Security_Anomaly_FeTaQA"],
    #         "MID":   ["Calculation_Based_Anomaly_FeTaQA", "Logical_Anomaly_FeTaQA"],
    #         "OVER":  ["Temporal_Anomaly_FeTaQA", "Normalization_Anomaly_FeTaQA", "Value_Anomaly_FeTaQA"]
    #     },
    #     "Spider_Beaver": {
    #         "UNDER": ["Temporal_Anomaly_Spider_Beaver", "Factual_Anomaly_Spider_Beaver", "Normalization_Anomaly_Spider_Beaver"],
    #         "MID":   ["Data_Consistency_Anomaly_Spider_Beaver", "Security_Anomaly_Spider_Beaver"],
    #         "OVER":  ["Logical_Anomaly_Spider_Beaver", "Calculation_Based_Anomaly_Spider_Beaver", "Value_Anomaly_Spider_Beaver"]
    #     },
    #     "WikiTQ": {
    #         "UNDER": ["Factual_Anomalies_WikiTQ", "Data_Consistency_Anomalies_WikiTQ", "Logical_Anomalies_WikiTQ"],
    #         "MID":   ["Temporal_Anomalies_WikiTQ", "Security_Anomalies_WikiTQ"],
    #         "OVER":  ["Normalization_Anomalies_WikiTQ", "Value_Anomalies_WikiTQ", "Calculation_Based_Anomalies_WikiTQ"]
    #     }
    # }

    
    map_dict = {}

    for folder in FOLDS:
        map_dict[folder] = categorize_scores(perf_score[folder])
        GT_ROOT         = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"
        VARIATION_ROOT  = f"path_to_dataset/variation_3/{folder}"
        PERFORMANCE_GROUPS = map_dict[folder]
        print(PERFORMANCE_GROUPS)
        category_names = sorted(
            d.replace(f"_{folder}", "")
            for d in os.listdir(CATEGORIES_ROOT)
            if os.path.isdir(os.path.join(CATEGORIES_ROOT, d)) and d != "Ground_truth"
        )
        if not category_names:
            raise RuntimeError("No category subfolders found under CATEGORIES_ROOT.")

        # 1b) Enumerate all GT JSON filenames
        all_files = sorted(fn for fn in os.listdir(GT_ROOT) if fn.lower().endswith(".json"))

        # Build a dictionary: total_perturbed[cat] = total # of perturbed cells across ALL files
        total_perturbed = {cat: 0 for cat in category_names}

        for fname in all_files:
            # Load GT
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # For each category subfolder, see if _updated.json exists, then count perturbed cells
            for cat in category_names:
                cat_folder = os.path.join(CATEGORIES_ROOT, cat+f"_{folder}")
                upd_name   = fname.replace(".json", "_updated.json")
                upd_path   = os.path.join(cat_folder, upd_name)
                if not os.path.exists(upd_path):
                    continue
                with open(upd_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cat_rows = data if isinstance(data, list) else [data]
                perturbed = find_perturbed_cells(cat_rows)
                total_perturbed[cat] += len(perturbed)

        # 1c) Compute group‐level numerators
        num_under = sum(
            perf_score[folder][cat] * total_perturbed.get(cat, 0)
            for cat in PERFORMANCE_GROUPS["UNDER"]
            if cat in total_perturbed
        )
        num_mid = sum(
            perf_score[folder][cat] * total_perturbed.get(cat, 0)
            for cat in PERFORMANCE_GROUPS["MID"]
            if cat in total_perturbed
        )
        num_over = sum(
            perf_score[folder][cat] * total_perturbed.get(cat, 0)
            for cat in PERFORMANCE_GROUPS["OVER"]
            if cat in total_perturbed
        )
        # print(total_perturbed)
        # print(perf_score[folder])
        # print(num_under,num_mid,num_over)
        grand_total = num_under + num_mid + num_over
        if grand_total == 0:
            # If somehow no perturbations anywhere, fall back to uniform:
            p_under = p_mid = p_over = 1/3
        else:
            p_under = num_under / grand_total
            p_mid   = num_mid   / grand_total
            p_over  = num_over  / grand_total

        
        eps = 1e-6
        if num_under > 0 and p_under < eps:
            p_under = eps
        if num_mid > 0 and p_mid < eps:
            p_mid = eps
        if num_over > 0 and p_over < eps:
            p_over = eps

        # Renormalize so they sum to 1
        sum_p = p_under + p_mid + p_over
        p_under /= sum_p
        p_mid   /= sum_p
        p_over  /= sum_p

        thr_under = p_under
        thr_mid   = p_under + p_mid

        # # ─────────────────────────────────────────────────────────────────────
        # # **INVERT** those probabilities so that smaller p_i → larger p_i'
        # # ─────────────────────────────────────────────────────────────────────
        # # Avoid division-by-zero by guaranteeing p_i > 0 (if any is zero, just give it zero weight here)
        # eps = 1e-12
        # pu = max(p_under, eps)
        # pm = max(p_mid,   eps)
        # po = max(p_over,  eps)

        # w_under = 1.0 / pu
        # w_mid   = 1.0 / pm
        # w_over  = 1.0 / po
        # total_w = w_under + w_mid + w_over

        # p_under_inv = w_under / total_w
        # p_mid_inv   = w_mid   / total_w
        # p_over_inv  = w_over  / total_w

        # Compute cumulative thresholds:
        #   [0, p_under) → UNDER
        #   [p_under, p_under+p_mid) → MID
        #   [p_under+p_mid, 1) → OVER
        thr_under = p_under
        thr_mid   = p_under + p_mid
        # thr_over = 1.0
        
        print(f"\nFolder={folder}: p_under={p_under:.3f}, p_mid={p_mid:.3f}, p_over={p_over:.3f}\n")
        print(f"\nFolder={folder}: thr_under={thr_under:.3f}, thr_mid={thr_mid:.3f}, thr_over={1}\n")
        # 2) Prepare output dirs for variation
        for cat in category_names:
            os.makedirs(os.path.join(VARIATION_ROOT, cat), exist_ok=True)
        merged_dir = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        summary = {}

        # ────────────────────────────────────────────────────────────────────────
        # 3) SECOND PASS: For each file, we do “group‐weighted LCM‐style sampling.”
        # ────────────────────────────────────────────────────────────────────────
        for fname in tqdm(all_files, desc=f"Processing {folder}"):
            # 3a) Load GT rows
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # 3b) Gather that file’s perturbed cells per category
            perturbed_lists = {}
            cat_rows_dict   = {}
            for cat in category_names:
                cat_folder = os.path.join(CATEGORIES_ROOT, cat+f"_{folder}")
                upd_name   = fname.replace(".json", "_updated.json")
                upd_path   = os.path.join(cat_folder, upd_name)
                if not os.path.exists(upd_path):
                    continue
                with open(upd_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cat_rows = data if isinstance(data, list) else [data]
                perturbed = find_perturbed_cells(cat_rows)
                if perturbed:
                    cat_rows_dict[cat] = cat_rows
                    perturbed_lists[cat] = perturbed

            if not perturbed_lists:
                print(f"Skipping {fname}: no perturbed cells found.")
                continue

            # 3c) Build three lists of (cat, cell_id) for this file, one per group:
            under_pool = []  # each entry is (cat, "R{r}C{c}")
            mid_pool   = []
            over_pool  = []

            for cat, perturbed in perturbed_lists.items():
                if cat in PERFORMANCE_GROUPS["UNDER"]:
                    under_pool.extend((cat, c) for c in perturbed)
                elif cat in PERFORMANCE_GROUPS["MID"]:
                    mid_pool.extend((cat, c) for c in perturbed)
                elif cat in PERFORMANCE_GROUPS["OVER"]:
                    over_pool.extend((cat, c) for c in perturbed)
                else:
                    # If some category is not in any group, ignore it:
                    pass

            # If any pool is empty, its probability share effectively disappears.
            # We will still use thr_under, thr_mid as boundaries, but if e.g. under_pool=[]
            # then any r<thr_under gets ignored and we retry the draw.

            # 3d) Decide how many distinct cells to keep for this file:
            #     Use Heuristic A: D_file = max_{cat in this file}(#perturbed cells in cat).
            counts = [len(perturbed_lists[cat]) for cat in perturbed_lists]
            D_file = max(1, max(counts))

            # 3e) Repeatedly draw until we have D_file distinct (cat,cell_id).
            chosen = set()  # set of (cat,cell_id) pairs we keep
            all_remaining = {
                "UNDER": set(under_pool),
                "MID":   set(mid_pool),
                "OVER":  set(over_pool)
            }

            while len(chosen) < D_file:
                r = random.random()

                if r < thr_under:
                    group = "UNDER"
                elif r < thr_mid:
                    group = "MID"
                else:
                    group = "OVER"

                pool = all_remaining[group]
                if not pool:
                    # Nothing left in that group—try again
                    for alt in ("UNDER", "MID", "OVER"):
                        if alt != group and len(all_remaining[alt]) > 0:
                            group = alt
                            pool  = all_remaining[alt]
                            break

                    # If still empty (all groups are now empty), break out
                    if not pool:
                        break

                # Pick one (cat,cell_id) uniformly at random from pool
                (chosen_cat, chosen_cell) = random.choice(list(pool))
                chosen.add((chosen_cat, chosen_cell))

                # Remove that cell from all groups so we never pick it again
                for g in ("UNDER", "MID", "OVER"):
                    if (chosen_cat, chosen_cell) in all_remaining[g]:
                        all_remaining[g].remove((chosen_cat, chosen_cell))

            # 3f) Convert chosen set → chosen_cells_map: cat -> list of cell IDs
            chosen_cells_map = {}
            for (cat, cell) in chosen:
                chosen_cells_map.setdefault(cat, []).append(cell)

            # 3g) Build & write variation JSON for each category
            variation_dirs = []
            for cat, keep_cells in chosen_cells_map.items():
                cat_rows  = cat_rows_dict[cat]
                perturbed = perturbed_lists[cat]

                varied    = build_stratified_variation(gt_rows, cat_rows, perturbed, keep_cells)

                outdir = os.path.join(VARIATION_ROOT, cat)
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, fname)
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(varied, f, ensure_ascii=False, indent=2)
                variation_dirs.append(outdir)

            # 3h) Merge those variation folders
            merge_variations_for_file(fname, variation_dirs, merged_dir, labels_dir)

            # 3i) Update summary
            summary[fname] = {
                "PERFORMANCE_GROUPS": PERFORMANCE_GROUPS,
                "CATEGORIES": list(chosen_cells_map.keys()),
                "PERTURBED_COUNTS": {cat: len(perturbed_lists.get(cat, [])) for cat in chosen_cells_map},
                "KEPT_COUNTS":      {cat: len(chosen_cells_map[cat]) for cat in chosen_cells_map},
                "CHOSEN_CELLS":     chosen_cells_map
            }
            kept_str = ", ".join(f"{cat}:{len(chosen_cells_map[cat])}" for cat in chosen_cells_map)
            # print(f"Processed {fname}: kept counts → {kept_str}")

        # 4) Write summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # print(f"\nFinished {folder}. Wrote summary → {summary_path}\n")


if __name__ == "__main__":
    main()
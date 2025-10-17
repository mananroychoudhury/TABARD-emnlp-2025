import os
import json
import math
import random
from collections import OrderedDict
from tqdm.auto import tqdm

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

# def build_weighted_variation(
#     gt_rows,
#     category_rows,
#     perturbed_cells,
#     chosen_cells_idx
# ):
#     varied = json.loads(json.dumps(category_rows, ensure_ascii=False))
#     keep_set = {perturbed_cells[i] for i in chosen_cells_idx}
#     for r_idx, (gt_row, cat_row) in enumerate(zip(gt_rows, varied), start=1):
#         for c_idx, col_key in enumerate(cat_row.keys(), start=1):
#             cell_id = f"R{r_idx}C{c_idx}"
#             if cell_id in perturbed_cells and cell_id not in keep_set:
#                 cat_row[col_key] = gt_row[col_key]
#     return varied

def build_weighted_variation(
    gt_rows,
    category_rows,
    perturbed_cells,
    chosen_cells_idx
):
    """
    1) Deep-copy the category_rows so we can overwrite.
    2) Build a `keep_set` of those cell-IDs we want to keep (i.e. still perturbed).
    3) For every perturbed cell_id in the table:
         - If cell_id ∉ keep_set, we must “revert” it by finding the matching GT row by content.
         - cell_id = "R{r}C{c}" indicates row r in category_rows; col c corresponds to some column key.
         - But instead of trusting r, we search all gt_rows for a row where *every other column* matches.
    4) If we find exactly one GT row whose other‐columns match, we replace only that single cell’s value.
       If we cannot find a unique match (0 or >1), we can choose to:
         a) leave the perturbed cell as is, or
         b) fall back on using the original row-index (if r ≤ len(gt_rows)), or
         c) skip reverting that cell entirely.
       In this example, we’ll silently skip if no unique match is found.
    """
    import copy

    # 1) Deep‐copy so we can rewrite safely
    varied = json.loads(json.dumps(category_rows, ensure_ascii=False))

    # 2) Build a set of cell_ids we want to keep as perturbations
    keep_set = {perturbed_cells[i] for i in chosen_cells_idx}

    # Pre‐compute a list of column‐keys in the same order for category_rows and gt_rows.
    # We assume every row‐dict has the same set of keys (though order may not matter).
    # If your dictionaries have the same keys but in different orders, use sorted(gt_row.keys()).
    if len(category_rows) > 0:
        all_col_keys = list(category_rows[0].keys())
    else:
        all_col_keys = []
    matched = 0
    notmatched = 0
    partial_match = 0
    # 3) For each perturbed cell, revert if not in keep_set
    for cell_id in perturbed_cells:
        if cell_id in keep_set:
            continue

        # Parse the row‐index “r” and column‐index “c” from "R{r}C{c}"
        # Note: this “r” is the position in category_rows, but we’ll ignore it for matching.
        try:
            r_str, c_str = cell_id.split("R")[1].split("C")
            r_idx = int(r_str)
            c_idx = int(c_str)
        except Exception:
            # If parsing fails, skip this cell.
            continue

        # 3a) Find the column key in question
        if not (1 <= c_idx <= len(all_col_keys)):
            continue
        col_key = all_col_keys[c_idx - 1]

        # 3b) Gather all other‐column values from the perturbed row (varied[r_idx-1])
        #     We do this so we can match by content.
        if not (1 <= r_idx <= len(varied)):
            # If category_rows is shorter than r_idx, skip
            continue
        pert_row_dict = varied[r_idx - 1]
        other_cols = [k for k in all_col_keys if k != col_key]
        other_vals = {k: pert_row_dict[k] for k in other_cols}

        # 3c) Search through gt_rows to find a row where all other_cols match exactly
        matches = []
        for gt_candidate in gt_rows:
            if all(gt_candidate.get(k) == other_vals.get(k) for k in other_cols):
                matches.append(gt_candidate)

        # 3d) If exactly one match, revert that single cell
        if len(matches) == 1:
            matched+=1
            matched_gt_row = matches[0]
            new_val = matched_gt_row.get(col_key, None)
            # Write that value into varied[r_idx-1][col_key]
            varied[r_idx - 1][col_key] = new_val
        elif len(matches) > 1:
            partial_match+=1
            print(matches)
            matched_gt_row = matches[0]
            new_val = matched_gt_row.get(col_key, None)
            # Write that value into varied[r_idx-1][col_key]
            varied[r_idx - 1][col_key] = new_val
        else:
            notmatched+=1
            # 0 or >1 matches → ambiguous. You can decide how to handle this.
            # Option 1: leave as is (i.e. keep the perturbed value)
            # Option 2: fallback to row‐matching by index if in range:
            if 1 <= r_idx <= len(gt_rows):
                print(varied[r_idx - 1])
                print(gt_rows[r_idx - 1])
                print(varied[r_idx - 1][col_key])
                print(gt_rows[r_idx - 1].get(col_key))
                varied[r_idx - 1][col_key] = gt_rows[r_idx - 1].get(col_key)
                
            # Or simply continue without change
    print(f"{matched},{partial_match},{notmatched}")
    return varied


def merge_variations_for_file(file_name, input_category_dirs, output_merged_dir, output_labels_dir):
    unique_rows = OrderedDict()
    for cat_dir in input_category_dirs:
        folder_name = os.path.basename(cat_dir)
        path = os.path.join(cat_dir, file_name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if key not in unique_rows:
                    unique_rows[key] = {"record": row, "folders": set()}
                unique_rows[key]["folders"].add(folder_name)

    merged_list = [v["record"] for v in unique_rows.values()]
    labels = []
    for idx, v in enumerate(unique_rows.values()):
        rec = v["record"]
        is_anom = contains_anomaly(rec)
        folders = sorted(v["folders"]) if is_anom else []
        labels.append({"index": idx, "folders": folders})

    os.makedirs(output_merged_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    out_json = os.path.join(output_merged_dir, file_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    base, _ext = file_name.rsplit(".", 1)
    label_json = f"{base}_labels.json"
    with open(os.path.join(output_labels_dir, label_json), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

def sample_via_two_step(perturbed_lists, D, seed=None):
    """
    LCM‐equivalent sampling:
      1) Pick a category uniformly from those with pcount>0.
      2) Pick one index from that category uniformly.
      3) Repeat (with replacement) until you have D distinct (cat, index) pairs.
    Returns: dict[cat] -> set(of indices in perturbed_lists[cat] to keep).
    """
    if seed is not None:
        random.seed(seed)

    valid_cats = [cat for cat, lst in perturbed_lists.items() if len(lst) > 0]
    total_cells = sum(len(perturbed_lists[k]) for k in valid_cats)
    if D >= total_cells:
        # Keep all if D >= total unique perturbed cells
        return {cat: set(range(len(perturbed_lists[cat]))) for cat in perturbed_lists}

    chosen = set()
    while len(chosen) < D:
        cat = random.choice(valid_cats)
        idx = random.randrange(len(perturbed_lists[cat]))
        chosen.add((cat, idx))

    chosen_cells_map = {cat: set() for cat in perturbed_lists}
    for (cat, idx) in chosen:
        chosen_cells_map[cat].add(idx)

    return chosen_cells_map


def main():
    random.seed(2025)

    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]
    for folder in FOLDS:
        GT_ROOT = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"
        VARIATION_ROOT = f"path_to_dataset/variation_100/{folder}"

        category_names = sorted(
            [d.replace(f"_{folder}", "") for d in os.listdir(CATEGORIES_ROOT)
             if os.path.isdir(os.path.join(CATEGORIES_ROOT, d)) and d != "Ground_truth"]
        )
        if not category_names:
            raise RuntimeError("No category subfolders found under CATEGORIES_ROOT.")

        all_files = sorted([fn for fn in os.listdir(GT_ROOT) if fn.lower().endswith(".json")])

        for cat in category_names:
            os.makedirs(os.path.join(VARIATION_ROOT, cat), exist_ok=True)
        merged_dir = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        summary_dict = {}

        for fname in tqdm(all_files, desc="Processing Files"):
            # 1) Load GT JSON
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # 2) Gather perturbed cells per category
            valid_cats = []
            cat_rows_dict = {}
            perturbed_lists = {}
            for cat in category_names:
                cat_folder = cat + f"_{folder}"
                cat_path = os.path.join(CATEGORIES_ROOT, cat_folder, fname.replace(".json", "_updated.json"))
                if not os.path.exists(cat_path):
                    continue

                with open(cat_path, "r", encoding="utf-8") as f:
                    cat_data = json.load(f)
                    cat_rows = cat_data if isinstance(cat_data, list) else [cat_data]

                perturbed = find_perturbed_cells(cat_rows)
                valid_cats.append(cat)
                cat_rows_dict[cat] = cat_rows
                perturbed_lists[cat] = perturbed

            if not valid_cats:
                print(f"Skipping {fname}: no category file exists.")
                continue

            # 3) Compute counts = [n_k] and choose D_per_file dynamically
            counts = [len(perturbed_lists[cat]) for cat in valid_cats]

            # ───── Heuristic A: D = max(1, max(counts)) ─────
            D_per_file = max(1, max(counts))

            # ───── Heuristic B: D = floor(alpha × total), with alpha=0.3 ─────
            # total_cells = sum(counts)
            # alpha = 0.3
            # D_per_file = max(1, int(math.floor(alpha * total_cells)))

            # (Pick whichever heuristic you prefer; here we used A.)

            # 4) Run LCM‐style sampling
            chosen_cells_map = sample_via_two_step(perturbed_lists, D_per_file, seed=2025)
            
            # print(counts)
            # print(perturbed_lists)
            # print(chosen_cells_map)
            # break

            # 5) Build & write one “LCM‐variation” JSON per valid cat
            used_variation_dirs = []
            for cat in valid_cats:
                cat_rows = cat_rows_dict[cat]
                perturbed = perturbed_lists[cat]
                keep_idx = chosen_cells_map[cat]

                print(fname, cat)
                varied = build_weighted_variation(gt_rows, cat_rows, perturbed, keep_idx)

                outdir = os.path.join(VARIATION_ROOT, cat)
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, fname)
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(varied, f, ensure_ascii=False, indent=2)

                used_variation_dirs.append(outdir)

            # 6) Merge these variations
            if used_variation_dirs:
                merge_variations_for_file(fname, used_variation_dirs, merged_dir, labels_dir)
            else:
                print(f"Skipping merge for {fname}: no category had perturbations.")

            num_rows = len(gt_rows)
            summary_dict[fname] = {
                "CATEGORIES": ["GT"] + valid_cats,
                "ORIGINAL_ROWS": [num_rows] * (1 + len(valid_cats)),
                "PERTURBED_CELLS": ([[]] + [perturbed_lists[cat] for cat in valid_cats]),
                "CHOSEN_CELLS": ([[]] +
                    [[perturbed_lists[cat][i] for i in sorted(chosen_cells_map[cat])]
                     for cat in valid_cats])
            }

        # 7) Write summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

        print(f"Wrote summary.json → {summary_path}")

if __name__ == "__main__":
    main()

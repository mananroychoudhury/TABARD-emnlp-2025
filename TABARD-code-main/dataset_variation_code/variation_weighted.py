import os
import json
import math
from collections import defaultdict, OrderedDict
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

def find_perturbed_cells(table_rows):
    """
    Given a list of JSON objects (rows), return a list of "R{row}C{col}" strings
    for every cell whose value starts with '@@@_'.  Rows and columns are 1-based.
    Assumes the dict preserves key order; we treat key‐order as the column order.
    """
    perturbed = []
    for r_idx, row in enumerate(table_rows, start=1):
        # We'll enumerate columns in the order keys() appear.
        for c_idx, (col_key, col_val) in enumerate(row.items(), start=1):
            if isinstance(col_val, str) and col_val.startswith("@@@_"):
                perturbed.append(f"R{r_idx}C{c_idx}")
    return perturbed

def assign_weights(counts):
    """
    Given a list of integers counts = [p_0, p_1, ..., p_{N-1}],
    produce a list of weights in 1..10 that are inversely proportional:
      - The index with the smallest count gets weight=10,
      - The index with the largest count gets weight=1,
      - Ties share the same rank → same weight,
      - If N=1, that single category gets weight=1.
    Returns a list of ints same length as counts.
    """
    N = len(counts)
    # Initialize all weights to 0
    weights = [0] * N

    # Collect indices with count > 0
    positive_idxs = [i for i, c in enumerate(counts) if c > 0]
    if not positive_idxs:
        # No one has any perturbations → all weights = 0
        return weights

    # Build a list of just the positive‐count values and their original indices
    pos_pairs = [(counts[i], i) for i in positive_idxs]
    # Sort by (count, index) ascending so smallest count → first
    pos_pairs.sort(key=lambda x: (x[0], x[1]))

    # Assign “rank” among only the positive‐count indices
    # Ties in count get the same rank
    ranks = {}
    current_rank = 0
    last_count = pos_pairs[0][0]
    ranks[pos_pairs[0][1]] = 0
    for cnt, orig_i in pos_pairs[1:]:
        if cnt != last_count:
            current_rank += 1
            last_count = cnt
        ranks[orig_i] = current_rank

    max_rank = max(ranks.values())
    # If all positive counts were equal, max_rank == 0 → everyone (in positive_idxs) gets 10
    if max_rank == 0:
        for i in positive_idxs:
            weights[i] = 10
        return weights

    # Otherwise, map rank r in [0..max_rank] → weight in [10..1] linearly:
    #   weight = round(((max_rank - r) / max_rank)*9 + 1)
    for i in positive_idxs:
        r = ranks[i]
        w = round(((max_rank - r) / max_rank) * 9 + 1)
        weights[i] = int(w)

    return weights

def build_weighted_variation(
    gt_rows,
    category_rows,
    perturbed_cells,   # list of "R{r}C{c}" that are perturbed in category_rows
    chosen_cells_idx   # set of indices in perturbed_cells that we KEEP
):
    """
    Starting from category_rows (a list of dicts, possibly with "@@@_" strings),
    replace every perturbed cell that is NOT in chosen_cells_idx with the GT value.

    - gt_rows: list of dicts (the ground‐truth table).
    - category_rows: list of dicts (the original category’s perturbed table).
    - perturbed_cells: list of all "R{r}C{c}" strings in category_rows.
    - chosen_cells_idx: indices into perturbed_cells that we want to keep.

    We return a brand‐new list of dicts (same shape), where any perturbed cell
    outside chosen_cells_idx is replaced by the corresponding GT cell.
    Note: row/col are 1‐based.
    """
    # Copy the category so we can mutate it:
    varied = json.loads(json.dumps(category_rows, ensure_ascii=False))

    # Build a set of strings we want to keep:
    keep_set = { perturbed_cells[i] for i in chosen_cells_idx }

    for r_idx, (gt_row, cat_row) in enumerate(zip(gt_rows, varied), start=1):
        # For each column in cat_row, check if it’s perturbed and if so, whether we keep it.
        for c_idx, col_key in enumerate(cat_row.keys(), start=1):
            cell_id = f"R{r_idx}C{c_idx}"
            if cell_id in perturbed_cells:
                if cell_id not in keep_set:
                    # Overwrite with GT
                    cat_row[col_key] = gt_row[col_key]
                # else: leave the "@@@_..." string exactly as is
    return varied

def merge_variations_for_file(file_name, input_category_dirs, output_merged_dir, output_labels_dir):
    """
    Given a single file_name (e.g. "137.json"), and a list of category‐folder paths
    (each containing that file_name), read all N versions, build a merged JSON
    and a labels JSON exactly as your merge_json_tables_with_labels(...) would do.
    """
    # Step 1: Collect all rows, track “which folder it came from”
    unique_rows = OrderedDict()  # key→{"record":row_dict, "folders": set([...])}

    for cat_dir in input_category_dirs:
        folder_name = os.path.basename(cat_dir)  # e.g. "C1", "C2", etc.
        path = os.path.join(cat_dir, file_name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if key not in unique_rows:
                    unique_rows[key] = {"record": row, "folders": set()}
                unique_rows[key]["folders"].add(folder_name)

    # Step 2: Build merged list in insertion order:
    merged_list = [v["record"] for v in unique_rows.values()]

    # Step 3: Build labels list
    labels = []
    for idx, v in enumerate(unique_rows.values()):
        rec = v["record"]
        is_anom = contains_anomaly(rec)
        folders = sorted(v["folders"]) if is_anom else []
        labels.append({
            "index": idx,
            "folders": folders
        })

    # Ensure output directories exist
    os.makedirs(output_merged_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Write merged JSON
    out_json = os.path.join(output_merged_dir, file_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    # Write labels JSON
    base, _ext = file_name.rsplit(".", 1)
    label_json = f"{base}_labels.json"
    with open(os.path.join(output_labels_dir, label_json), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    return

def main():
    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]
    for folder in FOLDS:
        ### ─────── CONFIGURE THESE PATHS ─────── ###
        GT_ROOT = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"   # must contain subfolders C1, C2, …
        VARIATION_ROOT = f"path_to_dataset/variation_1/{folder}"
        ###########################################

        # 1) List out all category‐names (subdirectories of CATEGORIES_ROOT)
        category_names = sorted(
            [d.replace(f"_{folder}","") for d in os.listdir(CATEGORIES_ROOT)
            if os.path.isdir(os.path.join(CATEGORIES_ROOT, d)) and d != "Ground_truth" ]
        )
        N = len(category_names)
        if N == 0:
            raise RuntimeError("No category subfolders found under CATEGORIES_ROOT.")

        # 2) Build a list of all GT files (just the .json names)
        all_files = sorted([fn for fn in os.listdir(GT_ROOT) if fn.lower().endswith(".json")])

        # 3) Prepare the output‐dirs under VARIATION_ROOT
        #    We'll create one subdir per category (to hold the "weighted" category JSON),
        #    plus a Merged/ and labels/ subdir, plus a top‐level summary.json.
        for cat in category_names:
            os.makedirs(os.path.join(VARIATION_ROOT, cat), exist_ok=True)
        merged_dir    = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir    = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        summary_dict = {}  # will hold one entry per file_name

        # 4) Process each file, one by one
        for fname in tqdm(all_files, desc="Processing Files"):
            # 4a) Load GT JSON once
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # 4b) Build a list of only those categories whose *_updated.json actually exists
            valid_cats = []
            cat_rows_dict = {}    # cat → [list of dicts]
            perturbed_lists = {}  # cat → [ "R{r}C{c}", ... ]

            for cat in category_names:
                # Construct the expected path to this category’s updated file
                cat_folder = cat + f"_{folder}"
                cat_path = os.path.join(CATEGORIES_ROOT, cat_folder, fname.replace(".json", "_updated.json"))

                if not os.path.exists(cat_path):
                    # If file doesn’t exist, skip this category entirely
                    continue

                # Otherwise, load it and record both its rows and perturbed‐cells
                with open(cat_path, "r", encoding="utf-8") as f:
                    cat_data = json.load(f)
                    cat_rows = cat_data if isinstance(cat_data, list) else [cat_data]

                # Find all cells beginning with '@@@_'
                perturbed = find_perturbed_cells(cat_rows)

                # Only now add to valid_cats and our parallel dicts
                valid_cats.append(cat)
                cat_rows_dict[cat] = cat_rows
                perturbed_lists[cat] = perturbed

            # If no category had an updated file, skip entirely
            if not valid_cats:
                print(f"Skipping {fname}: no category file exists.")
                continue

            # 4c) Build counts & weights over exactly valid_cats
            counts = [len(perturbed_lists[cat]) for cat in valid_cats]
            weights = assign_weights(counts)

            # 4d) Decide which perturbed cells to keep per valid category
            chosen_cells_map = {}
            for i, cat in enumerate(valid_cats):
                pcount = len(perturbed_lists[cat])
                if pcount == 0:
                    chosen_cells_map[cat] = []
                else:
                    w = weights[i]
                    keep_num = max(1, int((w/10.0) * pcount))
                    chosen_cells_map[cat] = perturbed_lists[cat][:keep_num]

            # 4e) Build & write one “weighted‐variation” JSON for each valid cat
            used_variation_dirs = []
            for i, cat in enumerate(valid_cats):
                cat_rows = cat_rows_dict[cat]
                perturbed = perturbed_lists[cat]
                keep_cells = chosen_cells_map[cat]
                keep_idx = {perturbed.index(cell) for cell in keep_cells if cell in perturbed}

                # Overwrite all other '@@@_' cells with GT
                varied = build_weighted_variation(gt_rows, cat_rows, perturbed, keep_idx)

                outdir = os.path.join(VARIATION_ROOT, cat)
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, fname)
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(varied, f, ensure_ascii=False, indent=2)

                used_variation_dirs.append(outdir)
            # 4g) Now that we have produced one “weighted‐variation” .json under each
            #     Variation_root/{cat}/{fname}, we can run the merge‐logic just for this single file:
            
            if used_variation_dirs:
                merge_variations_for_file(
                    fname,
                    used_variation_dirs,
                    merged_dir,
                    labels_dir
                )
            else:
                # If literally no category had any perturbation, we can skip the merge entirely.
                # (Or, if you prefer, you could still output GT alone—but typically you just skip.)
                print(f"Skipping merge for {fname}: no category had perturbations.")
            num_rows=len(gt_rows)
            # 4h) Build summary entry for this fname
            summary_dict[fname] = {
                "CATEGORIES": ["GT"] + valid_cats,
                "ORIGINAL_ROWS": [num_rows] * (1 + len(valid_cats)),
                "PERTURBED_CELLS": ([[]] +
                                    [perturbed_lists[cat] for cat in valid_cats]),
                "WEIGHTS": ([0] +
                            weights),
                "CHOSEN_CELLS": ([[]] +
                                [chosen_cells_map[cat] for cat in valid_cats])
            }

            # print(f"Processed {fname}:  perturbed_counts={counts}, weights={weights}")

        # 5) Write the top‐level summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

        print(f"Wrote summary.json → {summary_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
variation_LCM_fast2_with_summary.py

Same as variation_LCM_fast2.py, but also collects a per‐file “summary” dictionary of:
  - Which categories existed,
  - How many perturbed cells each category had,
  - How many cells survived (KEPT) per category,
  - Exactly which (cat, idx) pairs were chosen.

At the end, dumps a single JSON called “variation_summary.json” containing
one entry per GT filename.

Usage:
    1) Edit BASE_DIR, FOLDS, and K below.
    2) Run: python variation_LCM_fast2_with_summary.py
"""

import os
import json
import random
from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm

# ---------------------------
# Helper functions (unchanged)
# ---------------------------

def contains_anomaly(obj):
    """
    Recursively check if any leaf string contains '@@@_'.
    Used when building the labels JSON.
    """
    if isinstance(obj, str):
        return "@@@_" in obj
    elif isinstance(obj, dict):
        return any(contains_anomaly(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(contains_anomaly(v) for v in obj)
    else:
        return False

def merge_variations_for_file(file_name, input_category_dirs, output_merged_dir, output_labels_dir):
    """
    Deduplicate all category variation JSON rows for `file_name`, then write:
      - merged JSON → output_merged_dir/file_name
      - labels JSON → output_labels_dir/{basename}_labels.json
    “labels” will note which category(ies) contributed an anomaly to each row.
    """
    unique_rows = OrderedDict()

    for cat_dir in input_category_dirs:
        category_name = os.path.basename(cat_dir)
        path = os.path.join(cat_dir, file_name)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]

        for row in rows:
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key not in unique_rows:
                unique_rows[key] = {"record": row, "folders": set()}
            unique_rows[key]["folders"].add(category_name)

    merged_list = [v["record"] for v in unique_rows.values()]

    labels = []
    for idx, v in enumerate(unique_rows.values()):
        rec = v["record"]
        is_anom = contains_anomaly(rec)
        folders = sorted(v["folders"]) if is_anom else []
        labels.append({"index": idx, "folders": folders})

    os.makedirs(output_merged_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(os.path.join(output_merged_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    base, _ = file_name.rsplit(".", 1)
    label_fname = f"{base}_labels.json"
    with open(os.path.join(output_labels_dir, label_fname), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

# def build_variation(
#     gt_rows,             # list of GT row-dicts
#     category_rows,       # list of row-dicts (with "@@@_…" in some cells)
#     perturbed_cells,     # list of "R{r}C{c}" strings
#     chosen_cells_idx     # set of indices (ints) into perturbed_cells to KEEP
# ):
#     # Deep-copy the category_rows so we can safely overwrite
#     varied = json.loads(json.dumps(category_rows, ensure_ascii=False))
#     keep_set = {perturbed_cells[i] for i in chosen_cells_idx}

#     for r_idx, (gt_row, var_row) in enumerate(zip(gt_rows, varied), start=1):
#         for c_idx, col_key in enumerate(var_row.keys(), start=1):
#             cell_id = f"R{r_idx}C{c_idx}"
#             if cell_id in perturbed_cells and cell_id not in keep_set:
#                 var_row[col_key] = gt_row[col_key]
#     return varied

def build_variation(
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
        else:
            notmatched+=1
            # 0 or >1 matches → ambiguous. You can decide how to handle this.
            # Option 1: leave as is (i.e. keep the perturbed value)
            # Option 2: fallback to row‐matching by index if in range:
            if 1 <= r_idx <= len(gt_rows):
                varied[r_idx - 1][col_key] = gt_rows[r_idx - 1].get(col_key)
            # Or simply continue without change
    return varied


def sample_via_two_step(perturbed_lists, D, seed=None):
    """
    LCM‐equivalent sampling without building a giant LCM‐sized list:
      1) Randomly pick a category (uniform among those with >0 perturbed cells).
      2) Randomly pick one index from that category’s perturbed list.
      3) Repeat until D distinct (cat, idx) pairs have been collected.
    Returns: dict[cat] → set(indices in perturbed_lists[cat]) to keep.
    """
    if seed is not None:
        random.seed(seed)

    valid_cats = [c for c in perturbed_lists if len(perturbed_lists[c]) > 0]
    total_cells = sum(len(perturbed_lists[c]) for c in valid_cats)

    # If D ≥ total unique perturbed cells, keep them all.
    if D >= total_cells:
        return {c: set(range(len(perturbed_lists[c]))) for c in perturbed_lists}

    chosen = set()
    while len(chosen) < D:
        cat = random.choice(valid_cats)
        idx = random.randrange(len(perturbed_lists[cat]))
        chosen.add((cat, idx))

    result = {c: set() for c in perturbed_lists}
    for (cat, idx) in chosen:
        result[cat].add(idx)
    return result

# ---------------------------
# Main routine (with summary)
# ---------------------------

def main():
    # -----------------------
    # USER CONFIGURATION
    # -----------------------
    FOLDS = [
        "FeTaQA",
        "Spider_Beaver",
        "WikiTQ"
    ]
    BASE_DIR = "path_to_dataset/"
    K = 4  # maximum times any single column may be “kept”
    # -----------------------
    # End configuration
    # -----------------------

    random.seed(2025)

    # We'll accumulate one summary‐dict per GT file here:
    summary = {}

    for folder in FOLDS:
        print(f"\n=== Processing folder: {folder} ===")

        GT_ROOT = os.path.join(BASE_DIR, f"{folder}-org", "Ground_truth")
        CATEGORIES_ROOT = os.path.join(BASE_DIR, f"{folder}-org")
        VARIATION_ROOT = os.path.join(BASE_DIR, f"variation_2/{folder}")

        merged_dir = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 1) Discover category names (strip “_{folder}” suffix)
        category_names = []
        for d in os.listdir(CATEGORIES_ROOT):
            full_path = os.path.join(CATEGORIES_ROOT, d)
            if not os.path.isdir(full_path) or d == "Ground_truth":
                continue
            if d.endswith(f"_{folder}"):
                category_names.append(d[: -(1 + len(folder))])
        category_names.sort()
        print(f"Found categories: {category_names}")

        # 2) List all GT JSON files
        all_files = sorted(f for f in os.listdir(GT_ROOT) if f.lower().endswith(".json"))
        print(f"  ↳ {len(all_files)} GT files to process")

        for fname in tqdm(all_files, desc=f"Folder: {folder}"):
            # 2.1) Load ground-truth rows
            gt_path = os.path.join(GT_ROOT, fname)
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]
            num_gt_rows = len(gt_rows)

            # 2.2) For each category, load its `_updated.json` (if it exists)
            #      and build:
            #        perturbed_lists[cat] = [cell_id, …]
            #        cell_to_coords[cat][cell_id] = (row_idx, col_idx_int)
            #      Skip any category whose updated JSON has a different row count than GT.
            cat_rows_dict = {}
            perturbed_lists = {}
            cell_to_coords = {}

            for cat in category_names:
                cat_dirname = f"{cat}_{folder}"
                cat_dirpath = os.path.join(CATEGORIES_ROOT, cat_dirname)
                if not os.path.isdir(cat_dirpath):
                    continue

                updated_fname = fname.replace(".json", "_updated.json")
                updated_path = os.path.join(cat_dirpath, updated_fname)
                if not os.path.exists(updated_path):
                    continue

                with open(updated_path, "r", encoding="utf-8") as f:
                    cat_data = json.load(f)
                cat_rows = cat_data if isinstance(cat_data, list) else [cat_data]

                # If row‐count doesn’t match GT, skip this category
                # if len(cat_rows) != num_gt_rows:
                #     print(f"  [warning] Skipping category '{cat}' for '{fname}': "
                #           f"updated has {len(cat_rows)} rows, GT has {num_gt_rows}.")
                #     continue

                # Build perturbed list & coords
                pert_list = []
                coord_map = {}
                for r_idx, row in enumerate(cat_rows, start=1):
                    col_keys = list(row.keys())
                    for c_idx, col_key in enumerate(col_keys, start=1):
                        val = row[col_key]
                        if isinstance(val, str) and val.startswith("@@@_"):
                            cell_id = f"R{r_idx}C{c_idx}"
                            pert_list.append(cell_id)
                            coord_map[cell_id] = (r_idx, c_idx)

                if not pert_list:
                    continue

                cat_rows_dict[cat] = cat_rows
                perturbed_lists[cat] = pert_list
                cell_to_coords[cat] = coord_map

            # If no category had any valid perturbed cells, skip this file
            if not perturbed_lists:
                # Still record an “empty” summary entry if desired:
                summary[fname] = {
                    "Categories": [],
                    "PERTURBED_COUNTS": {},
                    "KEPT_COUNTS": {},
                    "CHOSEN_CELLS": {}
                }
                continue

            # 3) SAMPLE D = total number of perturbed cells (across all categories)
            total_cells = sum(len(perturbed_lists[cat]) for cat in perturbed_lists)
            initial_map = sample_via_two_step(perturbed_lists, total_cells, seed=2025)

            # 4) FILTER by “distinct rows” & “column_count ≤ K”
            used_rows = set()
            col_count = defaultdict(int)
            final_map = {cat: set() for cat in perturbed_lists}

            # Flatten into a list of (cat, idx) pairs
            all_candidates = []
            for cat, idx_set in initial_map.items():
                for idx in idx_set:
                    all_candidates.append((cat, idx))

            # Shuffle to randomize priority
            random.shuffle(all_candidates)

            for (cat, idx) in all_candidates:
                cell_id = perturbed_lists[cat][idx]
                r_idx, c_idx = cell_to_coords[cat][cell_id]

                # (a) Skip if row already used
                if r_idx in used_rows:
                    continue
                # (b) Skip if column c_idx has reached budget
                if col_count[c_idx] >= K:
                    continue

                # Keep it
                final_map[cat].add(idx)
                used_rows.add(r_idx)
                col_count[c_idx] += 1

            # 5) Build the summary entry for this file
            summary[fname] = {
                "Categories":           sorted(final_map.keys()),
                "PERTURBED_COUNTS":     {cat: len(perturbed_lists.get(cat, [])) for cat in final_map},
                "KEPT_COUNTS":          {cat: len(final_map[cat]) for cat in final_map},
                "CHOSEN_CELLS_IDS":     {
                    cat: [perturbed_lists[cat][i] for i in sorted(final_map[cat])]
                    for cat in final_map
                },
            }

            # 6) Reconstruct per‐category variation JSONs
            variation_dirs = []
            for cat, keep_idxs in final_map.items():
                if cat not in cat_rows_dict:
                    continue

                cat_rows = cat_rows_dict[cat]          # list of row-dicts with "@@@_…"
                perturbed_cells = perturbed_lists[cat] # list of "R…C…" strings
                chosen_cells_idx = keep_idxs           # set of ints

                # Build the final “varied” table in one function call:
                varied = build_variation(gt_rows, cat_rows, perturbed_cells, chosen_cells_idx)

                # Write it out just as before:
                out_dir = os.path.join(VARIATION_ROOT, cat)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(varied, f, ensure_ascii=False, indent=2)

                variation_dirs.append(out_dir)

            # 7) Merge all per‐category variations into one deduped JSON + labels
            if variation_dirs:
                merge_variations_for_file(
                    fname,
                    variation_dirs,
                    merged_dir,
                    labels_dir
                )

        print(f"→ Done with folder: {folder}; merged outputs in {merged_dir}")

        # 7) Write summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Wrote summary.json → {summary_path}")

if __name__ == "__main__":
    main()

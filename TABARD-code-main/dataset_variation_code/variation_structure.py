import os
import json
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
    Given a list of JSON objects (rows), return a list of "R{row}C{col}" strings
    for every cell whose value starts with '@@@_'. Rows and columns are 1-based.
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

    Returns a new list of dicts where any perturbed cell not in keep_cells
    is replaced by gt_rows[r-1][col_key].
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
    random.seed(2025)  # not strictly needed, but keep for any future randomness

    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]


    for folder in FOLDS:
        ### ─────── CONFIGURATION ─────── ###
        GT_ROOT         = f"path_to_dataset/{folder}-org/Ground_truth"
        CATEGORIES_ROOT = f"path_to_dataset/{folder}-org"
        VARIATION_ROOT  = f"path_to_dataset/variation_3/{folder}"
        
        ###################################

        # 1) Identify which performance‐group folders actually exist
        category_names = sorted(
            [d.replace(f"_{folder}", "") for d in os.listdir(CATEGORIES_ROOT)
             if os.path.isdir(os.path.join(CATEGORIES_ROOT, d)) and d != "Ground_truth"]
        )
        if not category_names:
            raise RuntimeError("No category subfolders found under CATEGORIES_ROOT.")

        valid_cats = category_names  # dedupe & sort

        if not valid_cats:
            raise RuntimeError("None of the specified performance‐group folders exist under CATEGORIES_ROOT.")

        # 2) Enumerate all GT JSON filenames
        all_files = sorted(fn for fn in os.listdir(GT_ROOT) if fn.lower().endswith(".json"))

        # 3) Prepare output directories
        for cat in valid_cats:
            os.makedirs(os.path.join(VARIATION_ROOT, cat), exist_ok=True)

        merged_dir = os.path.join(VARIATION_ROOT, "Merged")
        labels_dir = os.path.join(VARIATION_ROOT, "labels")
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        summary = {}

        # 4) Loop over each GT file
        for fname in tqdm(all_files, desc=f"Processing {folder} files"):
            # 4a) Load GT rows
            with open(os.path.join(GT_ROOT, fname), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_rows = gt_data if isinstance(gt_data, list) else [gt_data]

            # 4b) For each valid category, load its "_updated.json" and find perturbed cells
            cat_rows_dict   = {}
            perturbed_lists = {}

            for cat in valid_cats:
                cat_folder = os.path.join(CATEGORIES_ROOT, cat+f"_{folder}")
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

            # If no category has any perturbations for this file, skip merging
            if not perturbed_lists:
                print(f"Skipping {fname}: no perturbed cells in any valid category.")
                continue

            # 4c) “Structural variation” across ALL categories:
            #      - In each file, process categories in ascending order (alphabetical).
            #      - Maintain `used_cells = set()` for that file.
            #      - In each category, we keep only those perturbed cells not in `used_cells`.
            #      - Then add the kept ones to `used_cells`.
            # used_cells = set()
            # chosen_cells_map = {}  # cat -> [list of "R{r}C{c}" to keep]

            # for cat in sorted(perturbed_lists.keys()):
            #     perturbed = perturbed_lists[cat]
            #     # Keep only those not already in `used_cells`
            #     keep_cells = [cell for cell in perturbed if cell not in used_cells]
            #     chosen_cells_map[cat] = keep_cells
            #     # Mark these as used
            #     used_cells |= set(keep_cells)

            used_rows = set()      # will store integers, e.g. {1, 3, 5}
            used_cols = set()      # will store integers, e.g. {2, 4}
            chosen_cells_map = {}  # cat -> [list of "R{r}C{c}" to keep]
            cats = list(perturbed_lists.keys())
            random.shuffle(cats)

            for cat in cats:
            # for cat in sorted(perturbed_lists.keys()):
                perturbed = perturbed_lists[cat]  # e.g. ['R1C1', 'R1C2', 'R2C1', ...]
                keep_cells = []

                for cell in perturbed:
                    # parse "R{r}C{c}" into integers r, c
                    #    e.g. "R2C1" → row = 2, col = 1
                    #    (assumes always format 'R<row_number>C<col_number>')
                    _, rest = cell.split('R', 1)        # rest = "2C1"
                    row_str, col_str = rest.split('C', 1)
                    row = int(row_str)
                    col = int(col_str)

                    # If row or column already used, skip this cell
                    if row in used_rows or col in used_cols:
                        continue

                    # Otherwise, keep it
                    keep_cells.append(cell)
                    used_rows.add(row)
                    used_cols.add(col)

                chosen_cells_map[cat] = keep_cells

            # 4d) Build & write one “structural‐variation” JSON for each category
            variation_dirs = []
            for cat, keep_cells in chosen_cells_map.items():
                cat_rows  = cat_rows_dict[cat]
                perturbed = perturbed_lists[cat]
                varied = build_stratified_variation(gt_rows, cat_rows, perturbed, keep_cells)

                outdir = os.path.join(VARIATION_ROOT, cat)
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, fname)
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(varied, f, ensure_ascii=False, indent=2)

                variation_dirs.append(outdir)

            # 4e) Merge these variations
            merge_variations_for_file(fname, variation_dirs, merged_dir, labels_dir)

            # 4f) Build summary entry
            summary[fname] = {
                "PERTURBED_COUNTS": {cat: len(perturbed_lists.get(cat, [])) for cat in chosen_cells_map},
                "KEPT_COUNTS":      {cat: len(chosen_cells_map[cat]) for cat in chosen_cells_map},
                "CHOSEN_CELLS":     chosen_cells_map
            }
            kept_str = ", ".join(f"{cat}:{len(chosen_cells_map[cat])}" for cat in chosen_cells_map)
            print(f"Processed {fname}: kept counts → {kept_str}")

        # 5) Write top‐level summary.json
        summary_path = os.path.join(VARIATION_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nFinished {folder}. Wrote summary → {summary_path}\n")

if __name__ == "__main__":
    main()

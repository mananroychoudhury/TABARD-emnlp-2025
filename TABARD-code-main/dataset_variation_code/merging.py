import os
import json
from collections import defaultdict, OrderedDict

def contains_anomaly(obj):
    """
    Recursively check if any leaf value in obj contains the substring '@@@_'.
    """
    if isinstance(obj, str):
        return '@@@_' in obj
    elif isinstance(obj, dict):
        return any(contains_anomaly(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(contains_anomaly(v) for v in obj)
    else:
        return False

def merge_json_tables_with_labels(input_root: str, output_dir: str):
    """
    Traverse each immediate subdirectory under `input_root`, find JSON files,
    and for filenames appearing in ≥2 subdirs:
      - Merge their array-of-objects contents, deduplicate identical rows
      - Record, for each unique row, the list of subdirs it was found in
      - Inspect each merged row: if it contains '@@@_', keep its folder list;
        otherwise, replace it with an empty list.
      - Write:
         - merged JSON (“filename.json”)
         - labels JSON (“filename.labels.json”) mapping index → [subdirs|[]]
    """
    # 1) Map filename → list of full paths
    file_map = defaultdict(list)
    for entry in os.scandir(input_root):
        if not entry.is_dir():
            continue
        subdir = entry.path
        if entry.name != "Ground_truth":
            print(entry.name)
            for fn in os.listdir(subdir):
                
                if fn.lower().endswith(".json"):
                    file_map[fn].append(os.path.join(subdir, fn))

    os.makedirs(output_dir, exist_ok=True)

    # 2) Process each filename appearing in ≥2 subdirs
    for fname, paths in file_map.items():
        if len(paths) < 1:
            continue

        # key → {"record": <dict>, "folders": set([...])}
        unique_rows = OrderedDict()

        for p in paths:
            folder = os.path.basename(os.path.dirname(p))
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                rows = data if isinstance(data, list) else [data]
                for row in rows:
                    # stable key for deduplication
                    key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                    if key not in unique_rows:
                        unique_rows[key] = {"record": row, "folders": set()}
                    unique_rows[key]["folders"].add(folder.replace("_Anomaly_.*$",""))

        # build merged list
        merged_list = [v["record"] for v in unique_rows.values()]

        # build labels: anomaly rows keep their folder list; others get []
        labels = []
        for idx, v in enumerate(unique_rows.values()):
            rec = v["record"]
            is_anom = contains_anomaly(rec)
            folders = sorted(v["folders"]) if is_anom else []
            labels.append({
                "index": idx,
                "folders": folders
            })

        os.makedirs(os.path.join(output_dir, 'Merged'),exist_ok=True)
        # write merged JSON
        out_json = os.path.join(output_dir,'Merged', fname)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=2)

        # write labels JSON
        os.makedirs(os.path.join(output_dir, 'labels'),exist_ok=True)
        label_json = (fname.rsplit(".", 1)[0]) + "_labels.json"
        with open(os.path.join(output_dir,'labels',label_json), "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        print(f"Wrote {fname} → {out_json}")
        print(f"Wrote labels       → {label_json}")

if __name__ == "__main__":

    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]
    for f in FOLDS:
        INPUT_ROOT  = f"C:path_to_dataset/{f}-org"
        OUTPUT_DIR  = f"C:path_to_dataset/{f}-merged"
        merge_json_tables_with_labels(INPUT_ROOT, OUTPUT_DIR)
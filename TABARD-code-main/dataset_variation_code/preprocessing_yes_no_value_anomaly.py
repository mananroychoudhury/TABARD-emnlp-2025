import os
import json

def annotate_data(main_data, yesno_data):
    """
    For each row i, for each cell j, if yesno_data[i][j] == "yes" (case‐insensitive),
    prefix main_data[i][j] with "@@@_". Works on list-of-lists or list-of-dicts.
    """
    if not isinstance(main_data, list) or not isinstance(yesno_data, list):
        raise ValueError("Expected both JSONs to be top‐level lists of rows.")
    rows = min(len(main_data), len(yesno_data))
    for i in range(rows):
        row_main = main_data[i]
        row_flag = yesno_data[i]
        # print(row_main)
        # print(row_flag)
        # Case A: rows are lists of cells
        if isinstance(row_main, list) and isinstance(row_flag, list):
            cols = min(len(row_main), len(row_flag))
            for j in range(cols):
                if (isinstance(row_flag[j], str) and
                    row_flag[j].strip().lower() == "yes" and
                    isinstance(row_main[j], str) and
                    not row_main[j].startswith("@@@_")):
                    row_main[j] = "@@@_" + row_main[j]

        # Case B: rows are dicts mapping column → value
        elif isinstance(row_main, dict) and isinstance(row_flag, dict):
            for key, flag_val in row_flag.items():
                if (isinstance(flag_val, str) and
                    flag_val.strip().lower() == "yes" and
                    key in row_main and
                    # isinstance(row_main[key], str) and
                    not str(row_main[key]).startswith("@@@_")):
                    row_main[key] = "@@@_" + str(row_main[key])
        # else: mismatched row types, skip
    # print(main_data)
    return main_data

def process_folder(root_dir, subfolder_name="YesNo_Tables"):
    """
    In `root_dir`, reads every *_yes_no.json in `subfolder_name`,
    finds matching {base}_updated.json in root, annotates it, and overwrites it.
    """
    root_dir = os.path.abspath(root_dir)
    subfolder = os.path.join(root_dir, subfolder_name)
    if not os.path.isdir(subfolder):
        raise FileNotFoundError(f"Subfolder not found: {subfolder}")

    for fn in os.listdir(subfolder):
        if not fn.lower().endswith("_yes_no.json"):
            continue
        # if fn.lower().startswith("137_yes_no.json"):
        base = fn[:-len("_yes_no.json")]
        yesno_path = os.path.join(subfolder, fn)
        updated_fname = f"{base}_updated.json"
        updated_path = os.path.join(root_dir, updated_fname)

        if not os.path.exists(updated_path):
            print(f"[skipping] {updated_fname} not found in root.")
            continue

        # load both JSONs
        with open(yesno_path, 'r', encoding='utf-8') as f:
            yesno_data = json.load(f)
        with open(updated_path, 'r', encoding='utf-8') as f:
            main_data = json.load(f)

        # annotate
        try:
            annotated = annotate_data(main_data, yesno_data)
            # break
        except ValueError as e:
            print(f"[error] {updated_fname}: {e}")
            continue
        # break
        # write back
        with open(updated_path, 'w', encoding='utf-8') as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)

        print(f"Annotated → {updated_fname}")

if __name__ == "__main__":
    target_folder = r"path_to_dataset\Spider_Beaver-org\Value_Anomaly_Spider_Beaver"
    process_folder(target_folder)

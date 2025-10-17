# ── main.py ─────────────────────────────────────────────────────────

import os

# Step 1: chunk merged JSONs with labels
from strip_chunking_data import process_json_files_with_labels

# Step 2: generate yes/no JSONs from (step 1) chunked data
from yes_no_tabel_gen import run as create_yes_no_run

# Step 3: produce JSONL payloads for Gemini from (step 2) output
from genreate_batch_files import main as generate_jsonl_payloads


def main():
    """
    Run all three steps in sequence:
      1) Chunk merged JSON + label pairs under a single 'chunked' folder.
      2) From those chunks, produce a yes/no version of each JSON.
      3) Produce a set of .jsonl files suitable for batch‐sending to Gemini.
    """
    FOLDS = ["FetaQA-merged", "Spider_Beaver-merged", "wikiTQ-merged"]
    # FOLDS = ["Spider_Beaver-merged"]
    # DIR = ["variation_1","variation_2","variation_3"]
    # FOLDS = ["FetaQA", "Spider_Beaver", "wikiTQ"]
    # for d in DIR:
    for fold in FOLDS:
        # ─────────────── Step 1 ───────────────
        data_folder    = f"....New-expirements/dataset/{fold}/Merged"
        label_folder   = f"....New-expirements/dataset/{fold}/labels"
        chunked_folder = f"....New-expirements/dataset/{fold}/Merged-chunked"

        # We expect Step 1 to create chunked_folder/Merged and chunked_folder/labels
        merged_subdir = os.path.join(chunked_folder, "Merged")
        labels_subdir = os.path.join(chunked_folder, "labels")

        if os.path.isdir(merged_subdir) and os.listdir(merged_subdir) and \
        os.path.isdir(labels_subdir) and os.listdir(labels_subdir):
            print("✔ STEP 1: Detected existing chunked output. Skipping Step 1.")
        else:
            print("─▶ STEP 1: Chunking merged JSONs + labels…")
            process_json_files_with_labels(
                data_folder,
                label_folder,
                chunked_folder
            )
            print("✔ Completed STEP 1: chunked JSONs are in:\n   ", chunked_folder, "\n")


        # ─────────────── Step 2 ───────────────
        yesno_input  = os.path.join(chunked_folder, "Merged")
        yesno_output = os.path.join(chunked_folder, "Merged-yes-no")

        # If yesno_output folder exists and has JSON files, skip Step 2
        if os.path.isdir(yesno_output) and any(fn.endswith(".json") for fn in os.listdir(yesno_output)):
            print("✔ STEP 2: Detected existing yes/no JSONs. Skipping Step 2.")
        else:
            print("─▶ STEP 2: Generating yes/no JSONs from chunked data…")
            create_yes_no_run(yesno_input, yesno_output)
            print("✔ Completed STEP 2: yes/no JSONs are in:\n   ", yesno_output, "\n")



        # ─────────────── Step 3 ───────────────
        # Finally, wrap each folder of “yes/no” files into one or more .jsonl payloads
        # (your third script expects a folder with subfolders per anomaly-type)
        anomaly_input    = f"....New-expirements/dataset/{fold}/Merged-chunked/Merged-str"
        anomaly_jsonl_out= f"....New-expirements/Batchfiles/gemini/{fold}/l4_cot"


        print("─▶ STEP 3: Generating .jsonl payloads for Gemini…")
        generate_jsonl_payloads(anomaly_input, anomaly_jsonl_out)
        print("✔ Completed STEP 3: JSONL files are in:\n   ", anomaly_jsonl_out, "\n")


if __name__ == "__main__":
    main()

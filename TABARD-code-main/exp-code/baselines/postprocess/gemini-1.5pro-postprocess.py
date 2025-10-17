import os
import json
import re
import shutil
from tqdm.auto import tqdm
folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-2-1']
# folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-ncot']
List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']
# List = ["Value_Anomaly_(dataset_name ie.(FetaQA ... ))"]
for fold in tqdm(folder) :
    for i in tqdm(List):
        # Directories
        ground_truth_dir = f"..(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-yes-no-chunked/{i}/"
        gemini_jsonl_path = f"..(dataset_name ie.(FetaQA ... ))/gemini-output/output_folder-{fold}/{i}.jsonl"
        output_dir = f"..(dataset_name ie.(FetaQA ... ))/gemini-prediction-chunks/prediction-{fold}/{i}/"
        os.makedirs(output_dir, exist_ok=True)

        def transform_file(gt_path, anomalies, output_path):
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Reading GT file: {gt_path} - {e}")
                return

            for row in gt_data:
                for key in row:
                    row[key] = "No"

            for row_idx, field in anomalies:
                if 0 <= row_idx < len(gt_data) and field in gt_data[row_idx]:
                    gt_data[row_idx][field] = "Yes"

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(gt_data, f, indent=4)
            except Exception as e:
                print(f"[ERROR] Saving file: {output_path} - {e}")

        prediction_dict = {}

        with open(gemini_jsonl_path, "r", encoding="utf-8") as f:
            for j, line in enumerate(f, 1):
                try:
                    gemini_data = json.loads(line)
                    gemini_id = gemini_data.get("id", "")
                    base_filename = os.path.basename(gemini_id)
                    
                    if i != "Value_Anomaly_(dataset_name ie.(FetaQA ... ))":
                        if "chunk" in base_filename:
                            gt_filename = base_filename.replace("_updated_", "_yes_no_")
                            if gt_filename.endswith(".json_yes_no.json"):
                                gt_filename = gt_filename.replace(".json_yes_no.json", ".json")
                    elif i == "Value_Anomaly_(dataset_name ie.(FetaQA ... ))":
                        # print("im in folder")
                        if "chunk" in base_filename:
                            gt_filename = base_filename.replace("_chunk", "_yes_no_chunk")
                            gt_filename = gt_filename.replace(".json_yes_no.json", ".json")       
                    elif base_filename.endswith("_updated.json"):
                        gt_filename = base_filename.replace("_updated.json", "_yes_no.json")
                    else:
                        gt_filename = base_filename.replace(".json", "_yes_no.json")
                    
                    text_response = (
                        gemini_data.get("response", {})
                        .get("candidates", [])[0]
                        .get("content", {})
                        .get("parts", [])[0]
                        .get("text", "")
                    )
                    
                    try:
                        json_list = json.loads(text_response)
                        anomalies = [
                            (int(item["index"]), item["anomaly_column"].strip())
                            for item in json_list
                        ]
                        prediction_dict[gt_filename] = anomalies
                    except Exception as e:
                        print(f"[ERROR] Parsing JSON in line {j} - {e}")

                except json.JSONDecodeError as json_err:
                    print(f"[ERROR] JSON decode error on line {gemini_jsonl_path} {j}: {json_err}")
                except Exception as e:
                    print(f"[ERROR] Unexpected error on line {gemini_jsonl_path} {j}: {e}")

        # Step 2: Apply predictions to GT files
        for gt_filename in os.listdir(ground_truth_dir):
            # if not gt_filename.endswith("_yes_no.json"):
            #     continue
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            output_path = os.path.join(output_dir, gt_filename)

            anomalies = prediction_dict.get(gt_filename, [])
            transform_file(gt_path, anomalies, output_path)

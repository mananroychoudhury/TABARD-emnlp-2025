import os
import json
import ast
import re
from tqdm.auto import tqdm

folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level1-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level1-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level2-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level2-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level3-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level3-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-prompt-2-1']
List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']

for fold in tqdm(folder) :
    for i in tqdm(List):
        # Directories
        ground_truth_dir = f"..(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-yes-no-chunked/{i}/"
        llama_jsonl_path = f"..(dataset_name ie.(FetaQA ... ))/gpt-output/output_folder-{fold}/{i}.jsonl"
        output_dir = f"..(dataset_name ie.(FetaQA ... ))/gpt-prediction-chunks/predictions-{fold}/{i}/"
        os.makedirs(output_dir, exist_ok=True)

        def transform_file(gt_path, anomalies, output_path):
            # Load ground truth
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Reading GT file: {gt_path} - {e}")
                return

            # Step 1: Set all fields to "No"
            for row in gt_data:
                for key in row:
                    row[key] = "No"

            # Step 2: Update predicted anomalies to "Yes"
            for row_idx, field in anomalies:
                if 0 <= row_idx < len(gt_data) and field in gt_data[row_idx]:
                    gt_data[row_idx][field] = "Yes"

            # Save updated data
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(gt_data, f, indent=4)
                # print(f"[INFO] Saved: {output_path}")
            except Exception as e:
                print(f"[ERROR] Saving file: {output_path} - {e}")


        prediction_dict = {}

        with open(llama_jsonl_path, "r", encoding="utf-8") as f:
            for j, line in enumerate(f, 1):
                try:
                    llama_data = json.loads(line)
                    custom_id = llama_data.get("custom_id", "").replace(f"{i}/", "")
                    base_filename = os.path.basename(custom_id)
                    # print(base_filename)
                    if base_filename.endswith("_updated.json_yes_no.json"):
                        gt_filename = base_filename.replace("_updated.json_yes_no.json", "_yes_no.json")
                    elif base_filename.endswith("_updated.json"):
                        gt_filename = base_filename.replace("_updated.json", "_yes_no.json")
                    elif i != "Value_Anomaly_(dataset_name ie.(FetaQA ... ))":
                        if "chunk" in base_filename:
                            gt_filename = base_filename.replace("_updated_", "_yes_no_")
                            if gt_filename.endswith(".json_yes_no.json"):
                                gt_filename = gt_filename.replace(".json_yes_no.json", ".json")
                    elif i == "Value_Anomaly_(dataset_name ie.(FetaQA ... ))":
                        # print("im in folder")
                        if "chunk" in base_filename:
                            gt_filename = base_filename.replace("_chunk", "_yes_no_chunk")
                            gt_filename = gt_filename.replace(".json_yes_no.json", ".json")       
                    elif base_filename.endswith(".json_yes_no.json"):
                        gt_filename = base_filename.replace(".json_yes_no.json", "_yes_no.json")
                    else:
                        gt_filename = base_filename.replace(".json", "_yes_no.json")
                    # gt_filename = f"{base_filename}_yes_no.json"
                    # print(gt_filename)
                    gpt_response = llama_data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                    # print(f"Processing {base_filename} -> Response: {repr(llama_response)}")

                    # Match tuples like (1, 'Alcohol %'), (4, "Time (seconds)"), (5, 'Platform(s)'), etc.
                    tuple_matches = re.findall(
                        r'\(\s*(\d+)\s*,\s*[\'"](.+?)[\'"]\s*\)', 
                        gpt_response
                    )
                    
                    if tuple_matches:
                        # Convert to list of (int, str) tuples
                        anomalies = [(int(idx), str(label).strip()) for idx, label in tuple_matches]
                        prediction_dict[gt_filename] = anomalies
                    else:
                        match = re.search(r"\[.*\]", gpt_response, re.DOTALL)
                        if match:
                            try:
                                output_list = ast.literal_eval(match.group())
                                anomalies = [(int(idx), str(label).strip()) for idx, label in output_list]
                                prediction_dict[gt_filename] = anomalies
                            except Exception as e:
                                print(f"[ERROR] Failed to parse list in line {j}: {e}")
                        else:
                            print(f"[ERROR] Malformed or unrecognized response in line {j}: {gpt_response}")

                except json.JSONDecodeError as json_err:
                    print(f"[ERROR] JSON decode error on line {j}: {json_err}")
                except Exception as e:
                    print(f"[ERROR] Unexpected error on line {j}: {e}")
        # print("========prediction_dict=======",prediction_dict)
        # Step 2: Go through all GT files and process each
        for gt_filename in os.listdir(ground_truth_dir):
            # if not gt_filename.endswith("_yes_no.json"):
            #     continue
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            output_path = os.path.join(output_dir, gt_filename)
            anomalies = prediction_dict.get(gt_filename, [])  # Default to empty if not in predictions
            # print("========anomailes=======",anomalies)
            
            transform_file(gt_path, anomalies, output_path)
            if gt_filename not in prediction_dict:
                print(f"[INFO] No prediction for {gt_filename}, marked all as 'No'")
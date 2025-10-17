import os
import json
import re
import ast
from tqdm.auto import tqdm

folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-llama-level1-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level1-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level2-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level2-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level3-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level3-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-llama-prompt-2-1']
# folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-llama-level1-ncot']
List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']
# List = ["Value_Anomaly_(dataset_name ie.(FetaQA ... ))"]
for fold in tqdm(folder) :
    for i in tqdm(List):
        # Directories
        ground_truth_dir = f"(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-yes-no-chunked/{i}/"
        gemini_jsonl_path = f"(dataset_name ie.(FetaQA ... ))/llama-output/output_folder-{fold}/{i}.jsonl"
        output_dir = f"(dataset_name ie.(FetaQA ... ))/llama-prediction-chunks/prediction-{fold}/{i}/"
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
                    llama_data = json.loads(line)
                    custom_id = llama_data.get("custom_id", "")
                    base_filename = os.path.basename(custom_id)
                    
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
                    

                    llama_response = llama_data.get("response", {}).get("choices", [])[0].get("message", {}).get("content", "")
                    # print(f"Processing {base_filename} -> Response: {repr(llama_response)}")

                    # Match tuples like (1, 'Alcohol %'), (4, "Time (seconds)"), (5, 'Platform(s)'), etc.
                    tuple_matches = re.findall(
                        r'\(\s*(\d+)\s*,\s*[\'"](.+?)[\'"]\s*\)', 
                        llama_response
                    )
                    
                    if tuple_matches:
                        # Convert to list of (int, str) tuples
                        anomalies = [(int(idx), str(label).strip()) for idx, label in tuple_matches]
                        prediction_dict[gt_filename] = anomalies
                    elif not tuple_matches:
                        try:
                            tuple_pattern = r'\(\s*(\d+)\s*,\s*[\'"](.+?)[\'"]\s*\)'
                            matches = re.findall(tuple_pattern, llama_response)
                            
                            anomalies = []
                            for idx, label in matches:
                                try:
                                    anomalies.append((int(idx), label.strip()))
                                except Exception as e:
                                    print(f"[WARN] Skipped malformed tuple: ({idx}, {label}) due to: {e}")
                            prediction_dict[gt_filename] = anomalies
                        except Exception as e:   
                            print(f"[ERROR] Malformed or unrecognized response in line {i}: {gemini_jsonl_path}")
                        
                    else:
                        match = re.search(r"\[.*\]", llama_response, re.DOTALL)
                        if match:
                            try:
                                output_list = ast.literal_eval(match.group())
                                anomalies = [(int(idx), str(label).strip()) for idx, label in output_list]
                                prediction_dict[gt_filename] = anomalies
                            except Exception as e:
                                print(f"[ERROR] Failed to parse list in line {i}: {e}")
                        

                except json.JSONDecodeError as json_err:
                    print(f"[ERROR] JSON decode error on line {i}: {json_err}")
                except Exception as e:
                    print(f"[ERROR] Unexpected error on line {i}: {e}")

        # Step 2: Apply predictions to GT files
        for gt_filename in os.listdir(ground_truth_dir):
            # if not gt_filename.endswith("_yes_no.json"):
            #     continue
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            output_path = os.path.join(output_dir, gt_filename)

            anomalies = prediction_dict.get(gt_filename, [])
            transform_file(gt_path, anomalies, output_path)






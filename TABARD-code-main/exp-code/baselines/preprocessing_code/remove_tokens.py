'''
Strip @@@_ token as a prefix from the modified files.
'''
import os
from tqdm import tqdm
import json


def strip_token(input_folder_path, output_folder_path):
    files = sorted(os.listdir(input_folder_path))
    json_files = [file for file in files if file.endswith('.json') and os.path.isfile(os.path.join(input_folder_path, file))]

    # Create the output folder if not present
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in tqdm(json_files, desc="Processing files", unit="file"):
        file_path = os.path.join(input_folder_path, file_name)
        output_file = os.path.join(output_folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                stripped_data = []
                for row in data:
                    stripped_dict = {}
                    for key, value in row.items():
                        if isinstance(value, str) and value.startswith("@@@_"):
                            stripped_dict[key] = value[len("@@@_"):]  # Remove the @@@ prefix
                        else:
                            stripped_dict[key] = value
                    stripped_data.append(stripped_dict)
                with open(output_file, 'w', encoding='utf-8') as output_json:
                    json.dump(stripped_data, output_json, indent=4, separators=(",", ":"), ensure_ascii=False)
            
            except json.JSONDecodeError as e:
                print(f"Error reading {file_name}: {e}")

def run(input_folder_path,output_folder_path):
    folders = [name for name in os.listdir(input_folder_path)
           if os.path.isdir(os.path.join(input_folder_path, name))]

    for fold in folders:
        if fold != "Value_Anomaly_FetaQA":
            in_fold = os.path.join(input_folder_path,fold)
            out_fold= os.path.join(output_folder_path,fold)
            strip_token(in_fold, out_fold)
if __name__ == "__main__":
    # Hardcoded input and output folder paths
    input_folder_path = r"..(dataset_name ie.(FetaQA ... ))r/(dataset_name ie.(FetaQA ... ))r_Perturbed_Tables" # Replace this with the actual input folder path
    output_folder_path = r"..(dataset_name ie.(FetaQA ... ))r/(dataset_name ie.(FetaQA ... ))r-str"
    # strip_token(input_folder_path, output_folder_path)
    run(input_folder_path,output_folder_path)
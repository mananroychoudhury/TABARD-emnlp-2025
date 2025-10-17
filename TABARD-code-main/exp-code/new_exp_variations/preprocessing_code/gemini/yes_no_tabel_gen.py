import json
import sys
from tqdm import tqdm
import os

def create_yes_no(input_folder_path, output_folder_path):
    files = sorted(os.listdir(input_folder_path))
    json_files = [file for file in files if file.endswith('.json') and os.path.isfile(os.path.join(input_folder_path, file))]
    
    # Create the output folder if not present
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in tqdm(json_files, desc="Processing files", unit="file"):
        file_path = os.path.join(input_folder_path, file_name)
        # output_file = os.path.join(output_folder_path, file_name.split("_updated")[0] + "_yes_no.json")
        output_file = os.path.join(output_folder_path, file_name.replace("_updated","_yes_no"))
        
        with open(file_path, 'r',encoding='utf-8') as file:
            try:
                data = json.load(file)
                yes_no_data = []
                for row in data:
                    yes_no_dict = {}
                    for key, value in row.items():
                        if isinstance(value, str) and value.startswith("@@@_"):
                            yes_no_dict[key] = "Yes"
                        else:
                            yes_no_dict[key] = "No"
                    yes_no_data.append(yes_no_dict)
                with open(output_file, 'w',encoding='utf-8') as output_json:
                    json.dump(yes_no_data, output_json, indent=4)

            except json.JSONDecodeError as e:
                print(f"Error reading {file_name}: {e}")
        # break

def run(input_folder_path,output_folder_path):
    in_fold = os.path.join(input_folder_path)
    out_fold= os.path.join(output_folder_path)
    create_yes_no(in_fold, out_fold)

if __name__ == "__main__":
    # Hardcoded input and output folder paths
    input_folder_path = r"...\dataset\wikiTQ-merged\variation\Merged" # Replace this with the actual input folder path
    output_folder_path = r"...\dataset\wikiTQ-merged\variation\Merged-yes-no"
    # strip_token(input_folder_path, output_folder_path)
    run(input_folder_path,output_folder_path)
import os
import csv
import json

def csv_folder_to_json(folder_path):
    """
    Convert all CSV files in the given folder to JSON files.
    Each CSV will be read using csv.DictReader, and the resulting list
    of dictionaries will be written to a .json file with the same base name.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Iterate over every file in the folder
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.csv'):
            continue  # skip non-CSV files

        csv_path = os.path.join(folder_path, filename)
        # Build the JSON filename by replacing .csv with .json
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}.json"
        json_path = os.path.join(folder_path, json_filename)

        # Read the CSV into a list of dicts
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # Write the list of dicts out as JSON (pretty-printed with indent=4)
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(rows, jsonfile, indent=4)

        print(f"Converted: {csv_path} → {json_path}")

if __name__ == "__main__":
    # Replace this with the path to your folder of CSVs
    folder_of_csvs = r"..dataset\WikiTQ-org\Ground_truth"
    csv_folder_to_json(folder_of_csvs)

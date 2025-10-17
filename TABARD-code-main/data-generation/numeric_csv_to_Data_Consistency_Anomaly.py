import os 
import pandas as pd
import openai
import json
import re

# Set your OpenAI API key
openai.api_key = ""
# Paths to input/output folders
input_folder = r""
output_folder = r""
log_folder = r""

# Create the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

def analyze_columns(df):
    """
    Analyze the dataset structure and categorize columns.
    """
    columns_info = {
        "all_columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
        "date_columns": [col for col in df.columns if "date" in col.lower()],
        "location_columns": [col for col in df.columns if "latitude" in col.lower() or "longitude" in col.lower()],
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "age_columns": [col for col in df.columns if "age" in col.lower()],
        "price_columns": [col for col in df.columns if "price" in col.lower() or "cost" in col.lower()],
        "discount_columns": [col for col in df.columns if "discount" in col.lower()],
        "id_columns": [col for col in df.columns if "id" in col.lower() or "identifier" in col.lower()],
    }
    return columns_info

def extract_json_from_response(response_text):
    """
    Extract the JSON part from the GPT response, ignoring extraneous content.
    """
    try:
        # Find the JSON block using a regex pattern
        json_pattern = re.compile(r'\[.*\]', re.DOTALL)
        match = json_pattern.search(response_text)
        if match:
            json_content = match.group()
            # Clean invalid elements like comments
            json_content = re.sub(r'//.*', '', json_content)  # Remove comments
            return json_content
        else:
            raise ValueError("No JSON content found in response.")
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def validate_json_structure(json_content):
    """
    Validate the JSON structure to ensure it's parseable.
    """
    try:
        json.loads(json_content)
        return True
    except json.JSONDecodeError as e:
        print(f"Invalid JSON structure: {e}")
        return False

def is_suitable_for_consistency_anomalies(columns_info):
    """
    Check if a dataset is suitable for generating data consistency anomalies.
    """
    id_columns = columns_info["id_columns"]
    categorical_columns = columns_info["categorical_columns"]
    numeric_columns = columns_info["numeric_columns"]
    # Suitable if there are ID columns, categorical columns, or numeric columns
    return len(id_columns) > 0 or len(categorical_columns) > 0 or len(numeric_columns) > 0

def generate_anomalies(df, file_id):
    """
    Generate data consistency anomalies in the dataset by prompting GPT to introduce them.
    The number of anomalies is dynamically decided based on table size.
    """
    columns_info = analyze_columns(df)

    if not is_suitable_for_consistency_anomalies(columns_info):
        print(f"Table {file_id} is not suitable for data consistency anomalies. Skipping...")
        return None, []

    # Dynamically determine number of anomalies
    row_count = len(df)
    if row_count <= 10:
        max_anomalies = 3
    elif row_count <= 25:
        max_anomalies = 5
    elif row_count <= 50:
        max_anomalies = 7
    elif row_count <= 100:
        max_anomalies = 10
    else:
        max_anomalies = 10  # Upper bound

    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart up to {max_anomalies} data consistency anomalies based on the table's structure and contents. You must not apply more than {max_anomalies} anomalies. Strictly follow this limit. Use the examples below as guidance.
Don't add extra rows to the data; only modify the existing ones. Introduce **only in-place data consistency anomalies**—do not add or remove any rows. The output must have the **same number of rows and row order** as the original. Return the modified dataset in valid JSON format.
[  
    {{"column1": "value1", "column2": "value2", ...}},  
    {{"column1": "value1", "column2": "value2", ...}}  
]

Examples of allowed anomalies:
- Inconsistent casing: "HR" vs "Human Resources" vs "human resources"
- Inconsistent units: "70 kg" vs "154 lbs"
- Conflicting numeric values for same ID: same "Product ID" with different "Price"
- Inconsistent date formats: "2024-05-23" vs "May 23, 2024"
- Mixed category labels: "Male" vs "M", "NY" vs "New York"
- Inconsistent ID formatting: "AB-1234" vs "AB1234"
- Same "Customer ID" with slightly different "Phone Number"

Do not introduce new rows or duplicate existing ones. Modify only values in-place, keeping the structure identical.

Dataset Summary:
- ID Columns: {columns_info["id_columns"]}
- Categorical Columns: {columns_info["categorical_columns"]}
- Numeric Columns: {columns_info["numeric_columns"]}

File ID: {file_id}
Dataset:
{df.to_json(orient="records", lines=False)}
Modified Dataset in JSON:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data expert skilled at introducing data consistency anomalies."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.7,
    )

    output = response.choices[0].message.content.strip()
    print(f"GPT Response for {file_id}:\n{output}")

    log_entries = []
    # Add GPT response to log_entries
    log_entries.append(f"GPT Response for {file_id}:\n{output}")

    try:
        json_content = extract_json_from_response(output)
        if not json_content or not validate_json_structure(json_content):
            raise ValueError("Failed to extract or validate JSON from GPT response.")
        modified_data = json.loads(json_content)
        modified_df = pd.DataFrame(modified_data)

        if len(modified_df) != len(df):
            raise ValueError(f"Modified data length ({len(modified_df)}) != original ({len(df)}). Skipping file.")

                # Count actual anomalies and enforce max_anomalies
        anomalies_applied = 0

        for idx in range(len(df)):
            for col in df.columns:
                if col in modified_df.columns:
                    original_value = df.at[idx, col]
                    modified_value = modified_df.at[idx, col]
                    if original_value != modified_value:
                        if anomalies_applied < max_anomalies:
                            # Mark anomaly and log it
                            if not pd.api.types.is_object_dtype(modified_df[col]):
                                modified_df[col] = modified_df[col].astype(object)
                            modified_df.at[idx, col] = f"@@@_{modified_value}"
                            log_entries.append(
                                f"- Type: Data Consistency Anomaly\n  Description: Cell at row {idx + 1}, column '{col}' was modified."
                            )
                            anomalies_applied += 1
                        else:
                            # Revert back to original value
                            modified_df.at[idx, col] = original_value


        return modified_df, log_entries

    except Exception as e:
        print(f"Error parsing GPT output for {file_id}: {e}")
        print("Skipping table due to error.")
        return None, []

# Process each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    # >>> Changed here: now we look for .json instead of .csv
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]
        try:
            # Instead of pd.read_csv, read JSON
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            df = pd.DataFrame(table_data)
            print(f"Processing: {filename}")

            modified_df, log_entries = generate_anomalies(df, file_id)

            if modified_df is not None:  # Only save if anomalies are generated
                # >>> Manually write JSON to avoid escaping slashes & unicode
                json_str = modified_df.to_json(None, orient="records", indent=4, force_ascii=False)
                # Replace escaped slash
                json_str = json_str.replace('\\/', '/')

                output_path = os.path.join(output_folder, f"{file_id}_updated.json")
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(json_str)

                print(f"Saved updated file: {output_path}")

                # Save the log file
                log_path = os.path.join(log_folder, "anomalies_log.txt")
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Table: {filename}\n")
                    log_file.write("\n".join(log_entries))
                    log_file.write("\n\n")
            else:
                print(f"No anomalies generated for {filename}, skipping saving.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"All modified files and logs have been saved.")

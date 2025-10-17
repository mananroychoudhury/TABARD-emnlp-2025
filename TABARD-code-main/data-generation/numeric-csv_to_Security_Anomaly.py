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
    Check if a dataset is suitable for generating security anomalies.
    """
    # -------------------------
    # We are NO longer using this function to skip any tables.
    # The LLM itself decides. So we do not remove this function,
    # but we simply do NOT rely on its return value.
    # -------------------------
    id_columns = columns_info["id_columns"]
    categorical_columns = columns_info["categorical_columns"]
    numeric_columns = columns_info["numeric_columns"]
    # Suitable if there are ID columns, categorical columns, or numeric columns
    if len(id_columns) > 0 or len(categorical_columns) > 0 or len(numeric_columns) > 0:
        return True

    # Additional checks for 'security_columns' remain in place, but unused
    security_columns = ["role", "password", "encryption", "ip", "device", "suspended", "admin", "credit", "card"]
    for col in columns_info["all_columns"]:
        for sc in security_columns:
            if sc.lower() in col.lower():
                return True

    return False

def generate_anomalies(df, file_id, max_anomalies=6):
    """
    Generate security anomalies in the dataset by prompting GPT to introduce them.
    The LLM decides if the table is suitable, not the code.
    """
    columns_info = analyze_columns(df)
    # We do NOT skip based on is_suitable_for_consistency_anomalies(...) anymore.
    # The LLM will decide in the prompt whether to impart anomalies.

    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart up to {max_anomalies} security anomalies based on the table's structure and contents. Not all tables are suitable to impart security anomalies in them. So, choose the tables which are suitable for security anomalies and impart security anomalies in them. Use the examples below as guidance.
Don't add extra rows to the data; only modify the existing ones. Return the modified dataset in valid JSON format.
[
    {{"column1": "value1", "column2": "value2", ...}},
    {{"column1": "value1", "column2": "value2", ...}}
]

Examples of security anomalies that you could impart include:

1. Suspended Role Conflict: A user marked as "suspended" retains an "admin" role, creating a risk of unauthorized access.
2. Missing Audit Logs: Login attempts with null values for "IP Address" or "Device Type".
3. Suspicious Activity: Users logging in from an unusual or unknown location inconsistent with their typical login patterns.
4. Unmasked Sensitive Data: Credit card numbers or sensitive information stored without encryption or masking.
5. Invalid Permissions: An entry with mismatched role and access permissions (e.g., "viewer" accessing admin privileges).
6. Null Security Data: Columns like "role", "password", or "encryption" have null or missing values, indicating gaps in security enforcement.
7. Hard-Coded Credentials: Plain text passwords visible in the dataset, which is a security best practice violation.
8. Other Security Anomalies: You can impart any other security anomalies that you may seem fit for the given table.

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
            {"role": "system", "content": "You are a data expert skilled at introducing security anomalies."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5000,
        temperature=0.7,
    )

    output = response.choices[0].message.content.strip()
    print(f"GPT Response for {file_id}:\n{output}")

    log_entries = []
    log_entries.append(f"GPT Response for {file_id}:\n{output}")

    try:
        json_content = extract_json_from_response(output)
        if not json_content or not validate_json_structure(json_content):
            raise ValueError("Failed to extract or validate JSON from GPT response.")

        modified_data = json.loads(json_content)
        modified_df = pd.DataFrame(modified_data)

        # Track if changes were made
        any_changes = False

        for col in df.columns:
            for idx in range(len(df)):
                if col in modified_df.columns and idx < len(modified_df):
                    original_value = df.at[idx, col] if col in df.columns else None
                    modified_value = modified_df.at[idx, col] if col in modified_df.columns else None
                    # If the LLM changed anything, track it
                    if original_value != modified_value:
                        any_changes = True
                        # Convert to object if needed
                        if not pd.api.types.is_object_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].astype(object)
                        modified_df.at[idx, col] = f"@@@_{modified_value}"
                        log_entries.append(
                            f"- Type: Security Anomaly\n  Description: Cell at row {idx + 1}, column '{col}' was modified."
                        )

        # If no changes, set modified_df to None so we skip output
        if not any_changes:
            modified_df = None

        return modified_df, log_entries

    except Exception as e:
        print(f"Error parsing GPT output for {file_id}: {e}")
        print("Skipping table due to error.")
        return None, []

# Process each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    # CHANGED: now we read JSON
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            df = pd.DataFrame(table_data)
            print(f"Processing: {filename}")

            modified_df, log_entries = generate_anomalies(df, file_id)

            # Only save if anomalies are introduced
            if modified_df is not None:
                # Convert to JSON string with no slash/unicode escapes
                json_str = modified_df.to_json(None, orient="records", indent=4, force_ascii=False)
                json_str = json_str.replace('\\/', '/')  # remove escaped slash

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

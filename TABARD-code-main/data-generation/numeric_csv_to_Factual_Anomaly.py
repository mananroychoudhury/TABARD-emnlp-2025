import os 
import pandas as pd
import openai
import json
import re
import math

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

def generate_anomalies(df, file_id, max_anomalies):
    """
    Generate factual anomalies in the dataset by prompting GPT to introduce them.
    """
    columns_info = analyze_columns(df)
    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart at least {max_anomalies} factual anomalies based on the table's structure and contents. Use the examples below as guidance.
Don't add extra rows to the data; only modify the existing ones. Return the modified dataset in valid JSON format.
[
    {{"column1": "value1", "column2": "value2", ...}},
    {{"column1": "value1", "column2": "value2", ...}}
]

Examples of factual anomalies that you could impart include:

1. Contradictions:
    - An "intern" earning more than a "manager," contradicting typical salary structures.
2. Unrealistic Values: 
    - Product prices set at unusually low or high values (e.g., $5 for an iPhone or $1000 for a banana).
3. Geographical Mismatches: 
    - Postal codes, temperatures, or distances that are inconsistent with their associated regions.
4. Ambiguity:   
    - Ambiguous entries (e.g., "dollar" as currency without specifying USD or CAD).
5. Record-Breaking Claims: 
    - A record-breaking sprint time faster than Usain Bolt's world record (9.58 secs).
6. Unlikely Proportions: 
    - Expense-to-revenue ratios exceeding industry norms (e.g., 95% expenses for a business, leaving only 5% profit margin).
7. Excessive Usage: 
    - Energy usage or dosage exceeding realistic or safe limits (e.g., 10,000 kWh for a household or 10,000 mg/day of paracetamol).
8. Misaligned Benchmarks: 
    - Revenue recorded in one table does not align with corresponding records in another table.
9. Contextual Discrepancies: 
    - A record that doesn't align with known benchmarks or historical trends (e.g., a city temperature exceeding the highest recorded).
10. Other Factual Anomalies: 
    - You can impart any other type of factual anomalies which you seem fit for a specific table.

**Rules**
• Output **only** a JSON array – no code-fence, no comments.
• Do **not** add or delete rows/columns.
• Do **not** include `//` comments inside the JSON itself.
• Leave untouched cells exactly as they are.

Dataset Summary:
- All Columns: {columns_info["all_columns"]}

File ID: {file_id}
Dataset:
{df.to_json(orient="records", lines=False)}
Modified Dataset in JSON:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data expert skilled at introducing factual anomalies."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5000,
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

        for col in df.columns:
            for idx in range(len(df)):
                if col in modified_df.columns and idx < len(modified_df):
                    original_value = df.at[idx, col] if col in df.columns else None
                    modified_value = modified_df.at[idx, col] if col in modified_df.columns else None
                    # Check if the original value is non-empty/non-null before modifying
                    if original_value != modified_value:
                        # Cast column to object dtype if necessary
                        if not pd.api.types.is_object_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].astype(object)
                        modified_df.at[idx, col] = f"@@@_{modified_value}"
                        log_entries.append(f"- Type: Factual Anomaly\n  Description: Cell at row {idx + 1}, column '{col}' was modified.")

    except Exception as e:
        print(f"Error parsing GPT output for {file_id}: {e}")
        print("Saving original dataset without modification.")
        modified_df = df

    return modified_df, log_entries

# Process each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    # CHANGED HERE: we now look for .json
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]
        try:
            # Instead of pd.read_csv, read JSON
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            df = pd.DataFrame(table_data)
            print(f"Processing: {filename}")
            
            row_count     = len(df)
            max_anomalies = math.ceil(row_count * 0.5)

            modified_df, log_entries = generate_anomalies(df, file_id, max_anomalies)

            # >>> The only change: we manually write to JSON to avoid escaping
            json_str = modified_df.to_json(None, orient="records", indent=4, force_ascii=False)
            json_str = json_str.replace('\\/', '/')  # remove escaping of '/'
            with open(os.path.join(output_folder, f"{file_id}_updated.json"), "w", encoding="utf-8") as out_f:
                out_f.write(json_str)

            print(f"Saved updated file: {os.path.join(output_folder, f'{file_id}_updated.json')}")

            # Save the log file
            log_path = os.path.join(log_folder, "anomalies_log.txt")
            with open(log_path, "a") as log_file:
                log_file.write(f"Table: {filename}\n")
                log_file.write("\n".join(log_entries))
                log_file.write("\n\n")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"All modified files and logs have been saved.")

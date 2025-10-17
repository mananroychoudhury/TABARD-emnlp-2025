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
        # We'll temporarily leave this line, but we won't rely on it:
        "calculation_columns": [col for col in df.columns if "total" in col.lower() or "average" in col.lower() or "sum" in col.lower()],
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

def identify_calculation_related_columns(df):
    """
    Identify potential calculation-related columns in the dataset.
    """
    calculation_keywords = [
        "total", "sum", "difference", "average", "bmi", "profit", 
        "payable", "revenue", "expense", "interest", "tax", "ratio", 
        "percent", "score", "margin", "index", "avg", "calc", "computed", "mean"
    ]
    return [col for col in df.columns if any(keyword in col.lower() for keyword in calculation_keywords)]

# Removed the is_suitable_for_calculation_anomalies function
# so the code no longer decides "suitable" vs. "not suitable."

def generate_anomalies(df, file_id, max_anomalies=6):
    """
    Generate calculation-based anomalies in the dataset by prompting GPT to introduce them.
    """
    columns_info = analyze_columns(df)
    # We are no longer checking if the table is suitable; we simply proceed.

    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart up to {max_anomalies} calculation-based anomalies based on the table's structure and contents. All tables might not be suitable for imparting calculation based anomalies. Don't impart calculation based anomalies in them. Use the examples below as guidance.
Don't add extra rows to the data; only modify the existing ones. Return the modified dataset in valid JSON format.
[
    {{"column1": "value1", "column2": "value2", ...}},
    {{"column1": "value1", "column2": "value2", ...}}
]

Examples of calculation-based anomalies that you could impart include:

1. Incorrect Totals: A "total" column where the value does not match the sum of related columns.
2. Incorrect Formula: Columns with incorrect calculations (e.g., BMI = weight/(height^2) or profit = revenue - expense).
3. Missing Dependencies: A calculated column that references missing or null values in dependent columns.
4. Logical Violations: The result of a calculation is outside a reasonable range (e.g., a BMI of 100 or negative profit).
5. Rounding Errors: Calculations where rounding has been inconsistently or incorrectly applied.
6. Incorrect Weighted Averages: Averages calculated without considering the proper weights or distributions.
7. Currency Conversion Errors: Total values incorrectly converted between currencies (e.g., exchange rates applied incorrectly).
8. Misaligned Units: Results calculated using inconsistent units (e.g., combining meters and kilometers in distance-related calculations).

Dataset Summary:
- Numeric Columns: {columns_info["numeric_columns"]}
- Calculation Columns: {columns_info["calculation_columns"]}

File ID: {file_id}
Dataset:
{df.to_json(orient="records", lines=False)}
Modified Dataset in JSON:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data expert skilled at introducing calculation-based anomalies."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5000,
        temperature=0.7,
    )

    output = response.choices[0].message.content.strip()
    print(f"GPT Response for {file_id}:\n{output}")

    log_entries = []
    # Capture the entire GPT response in the log
    log_entries.append(f"GPT Response for {file_id}:\n{output}")

    try:
        json_content = extract_json_from_response(output)
        if not json_content or not validate_json_structure(json_content):
            raise ValueError("Failed to extract or validate JSON from GPT response.")
        modified_data = json.loads(json_content)
        modified_df = pd.DataFrame(modified_data)

        # >>> ADDED: track if any changes actually occurred
        any_changes = False

        for col in df.columns:
            for idx in range(len(df)):
                if col in modified_df.columns and idx < len(modified_df):
                    original_value = df.at[idx, col] if col in df.columns else None
                    modified_value = modified_df.at[idx, col] if col in modified_df.columns else None
                    # Check if the original value is non-empty/non-null before modifying
                    if original_value != modified_value:
                        any_changes = True
                        # Cast column to object dtype if necessary
                        if not pd.api.types.is_object_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].astype(object)
                        modified_df.at[idx, col] = f"@@@_{modified_value}"
                        log_entries.append(
                            f"- Type: Calculation-Based Anomaly\n  Description: Cell at row {idx + 1}, column '{col}' was modified."
                        )

        # >>> ADDED: if no changes, set modified_df to None, so we skip saving
        if not any_changes:
            modified_df = None

        return modified_df, log_entries

    except Exception as e:
        print(f"Error parsing GPT output for {file_id}: {e}")
        print("Skipping table due to error.")
        return None, []

# Process each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    # We now look for .json files instead of .csv
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            df = pd.DataFrame(table_data)
            print(f"Processing: {filename}")

            modified_df, log_entries = generate_anomalies(df, file_id)

            if modified_df is not None:
                # Manually write JSON to avoid escapes
                json_str = modified_df.to_json(None, orient="records", indent=4, force_ascii=False)
                # Remove escaped slashes
                json_str = json_str.replace('\\/', '/')

                output_path = os.path.join(output_folder, f"{file_id}_updated.json")
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(json_str)
                print(f"Saved updated file: {output_path}")

                # Save the log file
                log_path = os.path.join(log_folder, "anomalies_log.txt")
                with open(log_path, "a") as log_file:
                    log_file.write(f"Table: {filename}\n")
                    log_file.write("\n".join(log_entries))
                    log_file.write("\n\n")
            else:
                print(f"No anomalies generated for {filename}, skipping saving.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"All modified files and logs have been saved.")


def identify_calculation_related_columns(df):
    """
    Identify potential calculation-related columns in the dataset.
    """
    calculation_keywords = [
        "total", "sum", "difference", "average", "bmi", "profit",
        "payable", "revenue", "expense", "interest", "tax", "ratio",
        "percent", "score", "margin", "index", "avg", "calc",
        "computed", "mean"
    ]
    return [col for col in df.columns if any(keyword in col.lower() for keyword in calculation_keywords)]

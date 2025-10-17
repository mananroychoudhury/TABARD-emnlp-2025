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
    Analyze the dataset structure and categorize columns for normalization anomaly suitability.
    """
    columns_info = {
        "all_columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "composite_key_columns": [col for col in df.columns if "key" in col.lower() or "id" in col.lower()],
        "aggregated_columns": [col for col in df.columns if "total" in col.lower() or "sum" in col.lower() or "average" in col.lower()],
        "concatenated_columns": [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in ["city_state_zip", "full_name", "combined_address"])
        ],
        "repeating_group_columns": [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in ["skills", "tags", "categories"])
        ],
        "hierarchical_redundancy_columns": [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in ["department_head", "parent_category", "supervisor"])
        ]
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

def is_suitable_for_normalization_anomalies(columns_info):
    """
    Check if a dataset is suitable for generating normalization anomalies.
    """
    composite_keys = columns_info["composite_key_columns"]
    aggregated_columns = columns_info["aggregated_columns"]
    concatenated_columns = columns_info["concatenated_columns"]
    repeating_group_columns = columns_info["repeating_group_columns"]
    hierarchical_redundancy_columns = columns_info["hierarchical_redundancy_columns"]

    # Suitable if the table has attributes that indicate normalization issues
    return (
        len(composite_keys) > 1 or
        len(aggregated_columns) > 0 or
        len(concatenated_columns) > 0 or
        len(repeating_group_columns) > 0 or
        len(hierarchical_redundancy_columns) > 0
    )

def generate_anomalies(df, file_id, max_anomalies=6):
    """
    Generate normalization anomalies in the dataset by prompting GPT to introduce them.
    """
    columns_info = analyze_columns(df)

    if not is_suitable_for_normalization_anomalies(columns_info):
        print(f"Table {file_id} is not suitable for normalization anomalies. Skipping...")
        return None, []

    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart up to {max_anomalies} normalization anomalies based on the table's structure and contents by **modifying existing values** without introducing new rows or columns. Use the examples below as guidance.
Don't add extra rows to the data; only modify the existing ones. Return the modified dataset in valid JSON format.
[  
    {{"column1": "value1", "column2": "value2", ...}},  
    {{"column1": "value1", "column2": "value2", ...}}  
]

Examples of normalization anomalies that you could impart include:

1. **Partial Dependencies (2NF Violation):**  
   - Modify values to reflect attributes depending only on **part of a composite primary key**, rather than the entire key.
   - Example: In an "Orders" table with `OrderID` and `CustomerID` as a composite key, modify some rows so that `CustomerName` depends only on `CustomerID`, violating **2NF**.

2. **Transitive Dependencies (3NF Violation):**  
   - Modify values such that an attribute indirectly depends on a **non-primary key attribute**, creating redundancy.  
   - Example: In an "Employees" table, `OfficeLocation` depends on `Department`, which depends on `EmployeeID`, causing **transitive dependency**.

3. **Repeating Groups (1NF Violation):**  
   - Modify existing cells to store **multiple values within a single column**, instead of atomic values.  
   - Example: A "Skills" column storing "Python, Java, SQL" instead of separate normalized entries.

4. **Boyce-Codd Normal Form (BCNF) Violation:**  
   - Modify values such that a **non-prime attribute determines a prime attribute**, creating redundancy.  
   - Example: In a "Departments" table, if `ManagerName` determines `DepartmentID`, modify values to reinforce this dependency incorrectly.

5. **Multivalued Dependencies (4NF Violation):**  
   - Modify existing records to show **two or more unrelated multivalued attributes stored together**, rather than in separate tables.  
   - Example: An "Employees" table contains `Skills` and `Projects`, even though **each skill and project should be separately stored**.

6. **Join Dependency Violations (5NF Violation):**  
   - Modify values to reflect cases where **data should be decomposed into multiple relations but isn’t**.  
   - Example: A "Vendor_Product_Customer" table keeps vendor and customer details together, though they should be separate.

7. **Lack of Temporal Normalization (6NF Violation):**  
   - Modify values so that historical data **overwrites previous states** instead of keeping track of changes.  
   - Example: A salary column storing only the current salary, losing all previous changes.

8. **Denormalization:**  
   - Modify data so that **derived values** (e.g., `TotalSalary`) are stored **instead of being computed dynamically**.  
   - Example: `TotalSalary` = `BaseSalary + Bonus`, but modify data so this value is directly stored and may not match computed values.

9. **Combined Attributes:**  
   - Modify values so that **concatenated data appears in a single field** instead of being separated.  
   - Example: "New York, USA, 10001" is stored in one column instead of `City`, `Country`, and `Zip` fields.

10. **Hierarchical Redundancy:**  
   - Modify values so that **parent information is unnecessarily repeated** in every child record.  
   - Example: In an "Employees" table, `DepartmentHead` is stored **in every row** instead of being referenced properly.

11. **Other Normalization Anomalies:**  
   - Introduce any **contextually appropriate** normalization anomaly that fits the table’s structure while ensuring existing rows are modified but not expanded.

Dataset Summary:
- Composite Key Columns: {columns_info["composite_key_columns"]}
- Aggregated Columns: {columns_info["aggregated_columns"]}
- Concatenated Columns: {columns_info["concatenated_columns"]}
- Repeating Group Columns: {columns_info["repeating_group_columns"]}
- Hierarchical Redundancy Columns: {columns_info["hierarchical_redundancy_columns"]}

File ID: {file_id}
Dataset:
{df.to_json(orient="records", lines=False)}
Modified Dataset in JSON:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data expert skilled at introducing normalization anomalies."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5000,
        temperature=0.7,
    )

    output = response.choices[0].message.content.strip()
    print(f"GPT Response for {file_id}:\n{output}")

    log_entries = []
    # Capture GPT response in the log
    log_entries.append(f"GPT Response for {file_id}:\n{output}")

    try:
        json_content = extract_json_from_response(output)
        if not json_content or not validate_json_structure(json_content):
            raise ValueError("Failed to extract or validate JSON from GPT response.")
        modified_data = json.loads(json_content)
        modified_df = pd.DataFrame(modified_data)

        any_changes = False  # Track if anomalies were actually introduced

        for col in df.columns:
            for idx in range(len(df)):
                if col in modified_df.columns and idx < len(modified_df):
                    original_value = df.at[idx, col] if col in df.columns else None
                    modified_value = modified_df.at[idx, col] if col in modified_df.columns else None
                    # Check if the original value is non-empty/non-null before modifying
                    if original_value != modified_value:
                        any_changes = True
                        # Convert to object if needed
                        if not pd.api.types.is_object_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].astype(object)
                        modified_df.at[idx, col] = f"@@@_{modified_value}"
                        log_entries.append(f"- Type: Normalization Anomaly\n  Description: Cell at row {idx + 1}, column '{col}' was modified.")

        # If no anomalies, set modified_df = None so we skip writing
        if not any_changes:
            modified_df = None

        return modified_df, log_entries

    except Exception as e:
        print(f"Error parsing GPT output for {file_id}: {e}")
        print("Skipping table due to error.")
        return None, []

# Process each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    # Changed to look for .json
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]
        try:
            # Read JSON instead of CSV
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            df = pd.DataFrame(table_data)
            print(f"Processing: {filename}")

            modified_df, log_entries = generate_anomalies(df, file_id)

            if modified_df is not None:  # Only save if anomalies are generated
                # Convert to JSON string without slash/unicode escapes
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

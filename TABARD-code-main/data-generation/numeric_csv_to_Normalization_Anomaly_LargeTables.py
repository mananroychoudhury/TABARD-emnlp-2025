import os
import math
import pandas as pd
import openai
import json
import re
import tiktoken

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
openai.api_key = ""
max_model_tokens = 16384

input_folder  = r""
output_folder = r""
log_file      = r""

# Ensure output and log directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def analyze_columns(df: pd.DataFrame) -> dict:
    return {
        "all_columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "composite_key_columns": [c for c in df.columns if "key" in c.lower() or "id" in c.lower()],
        "aggregated_columns": [c for c in df.columns if any(k in c.lower() for k in ["total", "sum", "average"] )],
        "concatenated_columns": [c for c in df.columns if any(k in c.lower() for k in ["city_state_zip", "full_name", "combined_address"])],
        "repeating_group_columns": [c for c in df.columns if any(k in c.lower() for k in ["skills", "tags", "categories"])],
        "hierarchical_redundancy_columns": [c for c in df.columns if any(k in c.lower() for k in ["department_head", "parent_category", "supervisor"])],
    }


def extract_json_from_response(response_text: str) -> str | None:
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not match:
        return None
    js = match.group()
    js = re.sub(r'//.*', '', js)
    js = re.sub(r',\s*([}\]])', r'\1', js)
    return js.strip()


def validate_json_structure(js: str) -> bool:
    try:
        json.loads(js)
        return True
    except json.JSONDecodeError:
        return False


def is_suitable_for_normalization_anomalies(columns_info: dict) -> bool:
    return (
        len(columns_info["composite_key_columns"]) > 1 or
        len(columns_info["aggregated_columns"]) > 0 or
        len(columns_info["concatenated_columns"]) > 0 or
        len(columns_info["repeating_group_columns"]) > 0 or
        len(columns_info["hierarchical_redundancy_columns"]) > 0
    )

# ────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────
def generate_normalization_anomalies(df: pd.DataFrame, file_id: str, max_anomalies: int):
    cols_info = analyze_columns(df)
    # original suitability check remains
    if not is_suitable_for_normalization_anomalies(cols_info):
        print(f"Table {file_id} is not suitable for normalization anomalies. Skipping...")
        return None, [], False

    # update prompt to enforce minimum anomalies
    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart at least {max_anomalies} normalization anomalies based on the table's structure and contents by **modifying existing values** without introducing new rows or columns. Use the examples below as guidance.
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

**Rules**
• Output **only** a JSON array – no code-fence, no comments.
• Do **not** add or delete rows/columns.
• Do **not** include `//` comments inside the JSON itself.
• Leave untouched cells exactly as they are.
   
"""
    # append dataset summary and fields
    prompt += f"""

Composite Key Columns: {cols_info['composite_key_columns']}
Aggregated Columns: {cols_info['aggregated_columns']}
Concatenated Columns: {cols_info['concatenated_columns']}
Repeating Group Columns: {cols_info['repeating_group_columns']}
Hierarchical Redundancy Columns: {cols_info['hierarchical_redundancy_columns']}
File ID: {file_id}
Dataset:
{df.to_json(orient='records')}
Return modified dataset now:
"""

    system_msg = "You are a data expert skilled at introducing normalization anomalies."
    enc = tiktoken.encoding_for_model("gpt-4o")
    in_tokens = len(enc.encode(system_msg)) + len(enc.encode(prompt))
    max_out = max_model_tokens - in_tokens - 50

    rsp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_out
    )

    reply = rsp.choices[0].message.content.strip()
    json_part = extract_json_from_response(reply)
    explanation = reply.replace(json_part or '', '').strip() if json_part else ''

    logs = [f"GPT raw reply for {file_id}:\n{reply}"]
    if not json_part:
        logs.append(f"[ERROR] Could not extract JSON for {file_id}.")
        return None, logs, False
    if not validate_json_structure(json_part):
        logs.append(f"[ERROR] Malformed JSON for {file_id}:\n{json_part}")
        return None, logs, False

    modified = pd.DataFrame(json.loads(json_part))
    if explanation:
        logs.append(f"GPT anomaly explanations for {file_id}:\n{explanation}")

    anomalies_found = False
    for col in df.columns:
        if col not in modified.columns:
            continue
        for i in range(len(df)):
            orig = df.at[i, col]
            new  = modified.at[i, col] if i < len(modified) else orig
            if orig != new:
                anomalies_found = True
                if not pd.api.types.is_object_dtype(modified[col]):
                    modified[col] = modified[col].astype(object)
                modified.at[i, col] = f"@@@_{new}"
                logs.append(f"- Type: Normalization Anomaly; row {i+1}, col '{col}': '{orig}' → '{new}'.")

    if not anomalies_found:
        logs.append(f"[WARN] No normalization anomalies detected for {file_id}.")

    return modified, logs, anomalies_found


def process_in_chunks(df: pd.DataFrame, file_id: str, total_anomalies: int, chunk_size: int = 30):
    n = len(df)
    enc = tiktoken.encoding_for_model("gpt-4o")
    avg_tokens = len(enc.encode(df.head(10).to_json(orient='records'))) / 10
    overhead = len(enc.encode("You are a data expert skilled at introducing normalization anomalies."))
    safe_budget = max_model_tokens * 0.6 - overhead
    effective_chunk = min(chunk_size, max(1, int(safe_budget / avg_tokens)))
    boundaries = list(range(0, n, effective_chunk)) + [n]

    all_mods, all_logs = [], []
    any_anoms = False
    for start, end in zip(boundaries, boundaries[1:]):
        sub = df.iloc[start:end].reset_index(drop=True)
        slice_anoms = max(1, math.ceil(total_anomalies * (end-start) / n))
        sub_id = f"{file_id}_{start}-{end}"
        mod_sub, sub_logs, found = generate_normalization_anomalies(sub, sub_id, slice_anoms)
        if mod_sub is not None:
            all_mods.append(mod_sub)
            any_anoms |= found
        else:
            all_mods.append(sub)
        for entry in sub_logs:
            m = re.search(r"row (\d+)", entry)
            if m:
                local, global_row = int(m.group(1)), start + int(m.group(1))
                entry = entry.replace(f"row {local}", f"row {global_row}")
            all_logs.append(entry)

    combined = pd.concat(all_mods, ignore_index=True)
    return combined, all_logs, any_anoms

# ────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────
for fname in sorted(os.listdir(input_folder)):
    if not fname.endswith('.json'):
        continue
    path, fid = os.path.join(input_folder, fname), os.path.splitext(fname)[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    print(f"Processing: {fname}")
    total_anoms = math.ceil(len(df) * 0.30)
    mod_df, log_entries, anoms_present = process_in_chunks(df, fid, total_anoms)

    if anoms_present:
        out_path = os.path.join(output_folder, f"{fid}_updated.json")
        mod_df.to_json(out_path, orient='records', indent=4, force_ascii=False)
        print(f"Saved updated file: {out_path}")
        with open(log_file, 'a', encoding='utf-8') as lf:
            lf.write(f"Table: {fname}\n")
            lf.write("\n".join(log_entries) + "\n\n")
    else:
        print(f"No anomalies imparted for {fname}. File skipped.")

print("All modified files and logs have been saved.")

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

input_folder = r""
output_folder = r""
log_file = r""

# Ensure output and log directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def analyze_columns(df: pd.DataFrame) -> dict:
    return {
        "all_columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=["float64","int64"]).columns.tolist(),
        "date_columns": [c for c in df.columns if "date" in c.lower()],
        "location_columns": [c for c in df.columns if "latitude" in c.lower() or "longitude" in c.lower()],
        "categorical_columns": df.select_dtypes(include=["object","category"]).columns.tolist(),
        "age_columns": [c for c in df.columns if "age" in c.lower()],
        "price_columns": [c for c in df.columns if "price" in c.lower() or "cost" in c.lower()],
        "discount_columns": [c for c in df.columns if "discount" in c.lower()],
        "id_columns": [c for c in df.columns if "id" in c.lower() or "identifier" in c.lower()],
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

# ────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────
def generate_security_anomalies(df: pd.DataFrame, file_id: str, max_anomalies: int):
    cols_info = analyze_columns(df)

    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart at least {max_anomalies} security anomalies based on the table's structure and contents. Not all tables are suitable to impart security anomalies in them. So, choose the tables which are suitable for security anomalies and impart security anomalies in them. Use the examples below as guidance.
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

**Rules**
• Output **only** a JSON array – no code-fence, no comments.
• Do **not** add or delete rows/columns.
• Do **not** include `//` comments inside the JSON itself.
• Leave untouched cells exactly as they are.
"""
    prompt += f"""

Columns: {cols_info['all_columns']}
File ID: {file_id}
Dataset:
{df.to_json(orient='records')}
Return modified dataset now:
"""

    system_msg = "You are a data expert skilled at introducing security anomalies."
    enc = tiktoken.encoding_for_model("gpt-4o")
    in_tokens = len(enc.encode(system_msg)) + len(enc.encode(prompt))
    max_out = max_model_tokens - in_tokens - 50

    rsp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system_msg}, {"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=max_out
    )

    reply = rsp.choices[0].message.content.strip()
    json_part = extract_json_from_response(reply)
    explanation = reply.replace(json_part or '', '').strip() if json_part else ''

    logs = [f"GPT raw reply for {file_id}:\n{reply}"]
    if not json_part:
        logs.append(f"[ERROR] Could not extract JSON for {file_id}.")
        return df, logs, False
    if not validate_json_structure(json_part):
        logs.append(f"[ERROR] Malformed JSON for {file_id}:\n{json_part}")
        return df, logs, False

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
                logs.append(f"- Type: Security Anomaly; row {i+1}, col '{col}': '{orig}' → '{new}'.")

    if not anomalies_found:
        logs.append(f"[WARN] No security anomalies detected for {file_id}.")

    return modified, logs, anomalies_found


def process_in_chunks(df: pd.DataFrame, file_id: str, total_anomalies: int, chunk_size: int = 30):
    n = len(df)
    enc = tiktoken.encoding_for_model("gpt-4o")
    avg_tokens = len(enc.encode(df.head(10).to_json(orient='records'))) / 10
    overhead = len(enc.encode("You are a data expert skilled at introducing security anomalies."))
    safe_budget = max_model_tokens * 0.6 - overhead
    effective_chunk = min(chunk_size, max(1, int(safe_budget / avg_tokens)))
    boundaries = list(range(0, n, effective_chunk)) + [n]

    all_mods, all_logs = [], []
    any_anoms = False
    for start, end in zip(boundaries, boundaries[1:]):
        sub = df.iloc[start:end].reset_index(drop=True)
        slice_anoms = max(1, math.ceil(total_anomalies * (end-start) / n))
        sub_id = f"{file_id}_{start}-{end}"
        mod_sub, sub_logs, found = generate_security_anomalies(sub, sub_id, slice_anoms)
        all_mods.append(mod_sub)
        any_anoms |= found
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

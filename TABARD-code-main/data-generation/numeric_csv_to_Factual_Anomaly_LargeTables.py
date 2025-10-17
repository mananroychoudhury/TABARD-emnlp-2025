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
        "all_columns"      : df.columns.tolist(),
        "numeric_columns"  : df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
        "date_columns"     : [c for c in df.columns if "date" in c.lower()],
        "location_columns" : [c for c in df.columns if "latitude" in c.lower() or "longitude" in c.lower()],
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "age_columns"      : [c for c in df.columns if "age" in c.lower()],
        "price_columns"    : [c for c in df.columns if "price" in c.lower() or "cost" in c.lower()],
        "discount_columns" : [c for c in df.columns if "discount" in c.lower()]
    }


def extract_json_from_response(response_text: str) -> str | None:
    # Attempt to locate a JSON array in the model output
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not match:
        return None
    js = match.group()
    js = re.sub(r'//.*', '', js)                   # strip JS-style comments
    js = re.sub(r',\s*([}\]])', r'\1', js)      # remove trailing commas
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
def generate_factual_anomalies(df: pd.DataFrame, file_id: str, max_anomalies: int):
    """
    Uses GPT to inject up to `max_anomalies` factual anomalies into `df`.
    Returns modified DataFrame (with markers) and a list of log entries.
    """
    cols_info = analyze_columns(df)

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

Also, for each anomaly you introduce, provide a one-line explanation after the table in plain text, indicating why it is a factual anomaly.

    **Rules**
• Output **only** a JSON array – no code-fence, no comments.
• Do **not** add or delete rows/columns.
• Do **not** include `//` comments inside the JSON itself.
• Leave untouched cells exactly as they are.

Columns: {cols_info['all_columns']}
File ID: {file_id}

Dataset:
{df.to_json(orient="records")}

Return modified dataset now:
"""
    system_msg = "You are a data expert skilled at introducing factual anomalies in tables."

    # Estimate token usage and set output limit
    enc = tiktoken.encoding_for_model("gpt-4o")
    in_tokens = len(enc.encode(system_msg)) + len(enc.encode(prompt))
    max_out = max_model_tokens - in_tokens - 50

    rsp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_out
    )

    reply = rsp.choices[0].message.content.strip()
    # Separate JSON and explanations
    json_end_idx = reply.rfind(']')
    if json_end_idx == -1:
        raise ValueError(f"[{file_id}] Could not locate JSON array in GPT reply.")

    json_part = reply[:json_end_idx + 1].strip()
    explanation_part = reply[json_end_idx + 1:].strip()

    logs = [f"GPT raw table reply for {file_id}:\n{json_part}"]
    if explanation_part:
        logs.append(f"GPT anomaly explanations for {file_id}:\n{explanation_part}")

    js = json_part
    if not validate_json_structure(js):
        raise ValueError(f"[{file_id}] Invalid or malformed JSON.")
    modified = pd.DataFrame(json.loads(js))


    # Mark changed cells and record details
    for col in df.columns:
        if col not in modified.columns:
            continue
        for i in range(len(df)):
            orig = df.at[i, col]
            new  = modified.at[i, col]
            if orig != new:
                if not pd.api.types.is_object_dtype(modified[col]):
                    modified[col] = modified[col].astype(object)
                modified.at[i, col] = f"@@@_{new}"
                logs.append(
                    f"- Type: Factual Anomaly; row {i+1}, col '{col}': '{orig}' → '{new}'."
                )

    if "@@@_" not in modified.to_json():
        logs.append("[WARN] GPT produced no detectable anomalies.")

    return modified, logs


def process_in_chunks(df: pd.DataFrame, file_id: str, total_anomalies: int, chunk_size: int = 30):
    """
    Splits `df` into chunks to respect token limits, distributes `total_anomalies` across slices,
    calls `generate_factual_anomalies` on each sub-DataFrame, then reassembles results.
    """
    n = len(df)
    enc = tiktoken.encoding_for_model("gpt-4o")
    avg_tokens = len(enc.encode(df.head(10).to_json(orient="records"))) / 10
    overhead = len(enc.encode("You are a data expert skilled at introducing factual anomalies.")) + len(enc.encode("First, thoroughly analyze the entire table."))
    safe_budget = max_model_tokens * 0.6 - overhead
    auto_chunk = max(1, int(safe_budget / avg_tokens))
    effective_chunk = min(chunk_size, auto_chunk)

    boundaries = list(range(0, n, effective_chunk)) + [n]
    all_modified = []
    all_logs = []

    for start, end in zip(boundaries, boundaries[1:]):
        sub = df.iloc[start:end].reset_index(drop=True)
        slice_anoms = max(1, int(total_anomalies * (end-start) / n))
        sub_id = f"{file_id}_{start}-{end}"
        mod_sub, sub_logs = generate_factual_anomalies(sub, sub_id, slice_anoms)
        all_modified.append(mod_sub)

        # Adjust row indices in logs
        for entry in sub_logs:
            m = re.search(r"row (\d+)", entry)
            if m:
                local = int(m.group(1))
                global_row = start + local
                entry = entry.replace(f"row {local}", f"row {global_row}")
            all_logs.append(entry)

    return pd.concat(all_modified, ignore_index=True), all_logs

# ────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────
for fname in sorted(os.listdir(input_folder)):
    if not fname.endswith('.json'):
        continue
    path = os.path.join(input_folder, fname)
    fid  = os.path.splitext(fname)[0]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Processing: {fname}")

        row_count     = len(df)
        max_anomalies = math.ceil(row_count * 0.30)
        mod_df, log_entries = process_in_chunks(df, fid, max_anomalies)
       

        out_path = os.path.join(output_folder, f"{fid}_updated.json")
        mod_df.to_json(out_path, orient="records", indent=4, force_ascii=False)
        print(f"Saved updated file: {out_path}")

        # Append to log
        with open(log_file, 'a', encoding='utf-8') as lf:
            lf.write(f"Table: {fname}\n")
            lf.write("\n".join(log_entries) + "\n\n")

    except Exception as e:
        print(f"Error processing {fname}: {e}")

print("All modified files and logs have been saved.")

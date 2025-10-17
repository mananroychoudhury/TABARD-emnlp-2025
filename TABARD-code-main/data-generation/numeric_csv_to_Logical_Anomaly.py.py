import os
import math
import ast
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
log_folder    = r""

os.makedirs(output_folder, exist_ok=True)
os.makedirs(log_folder,    exist_ok=True)

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
        "discount_columns" : [c for c in df.columns if "discount" in c.lower()],
    }

def extract_json_from_response(response_text: str) -> str | None:
    fence = re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.I)
    body  = fence.group(1) if fence else None
    if body is None:
        block = re.search(r'\[\s*{[\s\S]+?}\s*\]', response_text)
        body  = block.group(0) if block else None
    if body is None:
       # fallback: if no code-fence/raw block found, grab from first '[' to last ']'
       start = response_text.find('[')
       end   = response_text.rfind(']')
       if start != -1 and end != -1 and end > start:
           body = response_text[start:end+1]
       else:
           return None
    body = re.sub(r'//.*', '', body)
    body = re.sub(r',\s*([}\]])', r'\1', body)
    return body.strip()

def validate_json_structure(js: str) -> bool:
    try:
        json.loads(js)
        return True
    except json.JSONDecodeError:
        return False

# ────────────────────────────────────────────────────────────────────────────
# CHUNKING  (✓ change: min 5 / max 10 anomalies per slice)
# ────────────────────────────────────────────────────────────────────────────
def process_in_chunks(df: pd.DataFrame, file_id: str,
                      total_anomalies: int, chunk_size: int = 50):
    modified, logs = [], []
    n = len(df)

    enc           = tiktoken.encoding_for_model("gpt-4o")
    avg_row_tokens = len(enc.encode(df.head(10).to_json(orient="records"))) / 10
    overhead       = len(enc.encode("You are a data expert skilfully injecting logical anomalies.")) \
                   + len(enc.encode("You MUST overwrite actual values…"))
    safe_budget    = max_model_tokens * 0.6 - overhead
    auto_chunk     = max(1, int(safe_budget / avg_row_tokens))
    chunk_size     = min(chunk_size, auto_chunk)

    idxs = list(range(0, n, chunk_size)) + [n]
    for start, end in zip(idxs, idxs[1:]):
        sub         = df.iloc[start:end].reset_index(drop=True)

        # NEW: guarantee 5 – 10 anomalies per slice
        slice_anoms = max(5, int(total_anomalies * (end - start) / n))
        slice_anoms = min(slice_anoms, 10)

        sub_id      = f"{file_id}_{start}-{end}"
        mod_sub, sub_log = generate_anomalies(sub, sub_id, slice_anoms)
        modified.append(mod_sub)

        # shift row numbers in logs
        for line in sub_log:
            if "row " in line:
                local = int(line.split("row ")[1].split(",")[0])
                line  = line.replace(f"row {local},", f"row {start + local},")
            logs.append(line)

    return pd.concat(modified, ignore_index=True), logs

# ────────────────────────────────────────────────────────────────────────────
# GPT-DRIVEN ANOMALY GENERATION  (✓ change: Reason line)
# ────────────────────────────────────────────────────────────────────────────
def generate_anomalies(df: pd.DataFrame, file_id: str, max_anomalies: int):
    cols   = analyze_columns(df)
    prompt = f"""
        First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
    Impart at least {max_anomalies} logical anomalies based on the table's structure and contents. Use the examples below as guidance.
    Don't add extra rows to the data only modify the existing ones. Return the modified dataset in valid JSON format.
    [
        {{"column1": "value1", "column2": "value2", ...}},
        {{"column1": "value1", "column2": "value2", ...}}
    ]

    Logical Anomalies to Include:
    1. Impossible Relationships:
    - Delivery date earlier than the order date.
    - Latitude/longitude outside valid ranges or mismatched coordinates.
    2. Illogical Contextual Data:
    - Sydney experiencing -20°C in July or Toronto experiencing 40°C in December.
    3. Biological/Physical Impossibilities:
    - Age greater than 204 years, speeds exceeding the speed of light.
    4. Violation of Scientific Principles:
    - Mismatch between speed, time, and distance.
    5. Anachronisms:
    - Plastic artifacts from 2000 BCE or passports issued by defunct countries.
    6. Financial Irregularities:
    - Discounts greater than 100%.
    7. Referential Anomalies:
    - Nonexistent references in related tables.
    8. Logical Violations in Calculations:
    - Mismatched calculated values.
    9. Illogical Temporal Data:
    - Events out of sequence (e.g., birthdate after death date).
    10. Categorical Inconsistencies:
    - Misclassified categories or attributes.
    11. Other Logical Anomalies:
    - Impart other types of logical anomalies which you seem to be fit for that particular table. 

**Rules**
• Output **only** a JSON array – no code-fence, no comments.
• Do **not** add or delete rows/columns.
• Do **not** include `//` comments inside the JSON itself.
• Leave untouched cells exactly as they are.

Dataset columns: {cols['all_columns']}
File ID: {file_id}

Original table:
{df.to_json(orient="records")}

Return the modified table now:
"""
    system_msg = "You are a data expert who skilfully injects logical anomalies."
    enc        = tiktoken.encoding_for_model("gpt-4o")
    in_tokens  = len(enc.encode(system_msg)) + len(enc.encode(prompt))
    max_out    = max_model_tokens - in_tokens - 50

    rsp = openai.ChatCompletion.create(
        model       = "gpt-4o",
        messages    = [{"role": "system", "content": system_msg},
                       {"role": "user",   "content": prompt}],
        temperature = 0.7,
        max_tokens  = max_out,
    )

    reply       = rsp.choices[0].message.content.strip()
    log_entries = [f"GPT raw reply for {file_id}:\n{reply}"]

    js = extract_json_from_response(reply)
    # try JSON first; if that fails, try Python literal_eval
    if not js or not validate_json_structure(js):
        try:
            # recover Python-style repr and convert to JSON
            pyobj = ast.literal_eval(js or "")
            js = json.dumps(pyobj)
        except Exception:
            raise ValueError(f"[{file_id}] Invalid or missing JSON.")
    modified_df = pd.DataFrame(json.loads(js))

    # mark changed cells + reason line
    for col in df.columns:
        for idx in range(len(df)):
            if col in modified_df.columns and idx < len(modified_df):
                orig = df.at[idx, col]
                new  = modified_df.at[idx, col]
                if orig != new:
                    if modified_df[col].dtype != object:
                        modified_df[col] = modified_df[col].astype(object)
                    modified_df.at[idx, col] = f"@@@_{new}"
                    log_entries.extend([
                        f"- Type: Logical Anomaly",
                        f"  Description: Cell at row {idx + 1}, column '{col}' "
                        f"changed from '{orig}' to '{new}'.",
                        f"  Reason: Introduced automatically to break logical consistency."
                    ])

    if "@@@" not in modified_df.to_json():
        log_entries.append("[WARN] GPT produced no detectable anomalies.")

    return modified_df, log_entries

# ────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────
for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(input_folder, filename)
    file_id   = os.path.splitext(filename)[0]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            table_data = json.load(f)
        df = pd.DataFrame(table_data)

        print(f"Processing: {filename}")
        row_count     = len(df)
        max_anomalies = math.ceil(row_count * 0.5)

        modified_df, log_entries = process_in_chunks(
            df, file_id, max_anomalies, chunk_size=50
        )

        output_path = os.path.join(output_folder, f"{file_id}_updated.json")
        modified_df.to_json(output_path, orient="records", indent=4)
        print(f"Saved updated file: {output_path}")

        log_path = os.path.join(log_folder, "anomalies_log.txt")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"Table: {filename}\n")
            lf.write("\n".join(log_entries))
            lf.write("\n\n")

    except Exception as exc:
        print(f"Error processing {filename}: {exc}")

print("All modified files and logs have been saved.")


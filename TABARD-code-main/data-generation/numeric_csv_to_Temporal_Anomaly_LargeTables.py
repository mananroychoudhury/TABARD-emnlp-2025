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
    cols = df.columns.tolist()
    time_cols = [
        c for c in cols
        if any(k in c.lower() for k in [
            "time", "date", "start", "end", "duration", "finish", "interval",
            "processing_time", "runtime", "time_taken", "arrival_time", "departure_time",
            "check_in", "check_out", "deadline", "due_date", "log_time", "log_date",
            "month", "year", "week", "quarter", "time_period", "time_frame"
        ])
    ]
    return {"all_columns": cols, "time_columns": time_cols}


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
def generate_temporal_anomalies(df: pd.DataFrame, file_id: str, max_anomalies: int):
    cols_info = analyze_columns(df)
    time_cols = cols_info['time_columns']

    # Build prompt
    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart at least {max_anomalies} temporal anomalies based on the table's structure and contents. Use the examples below as guidance.
Don't add extra rows to the data; only modify the existing ones. Return the modified dataset in valid JSON format.
[
    {{"column1": "value1", "column2": "value2", ...}},
    {{"column1": "value1", "column2": "value2", ...}}
]

Examples of temporal anomalies that you could impart include:

1. Overlapping Times: Scheduling conflicts or overlaps in times for reservations, events, or tasks.
2. Impossible Durations: A duration that is inconsistent with the start and end times.
3. Logical Violations: Arrival times earlier than departure times, or negative durations.
4. Extreme Outliers: Recorded times that are implausibly fast or slow (e.g., a race time faster than world records).
5. Inconsistent Time Zones: An event start time and end time mismatch due to unaccounted time zone differences.
6. Invalid Sequence: An event or task that finishes before it starts, or an earlier event depends on the outcome of a later event.
7. Missing Time Values: Records with missing or null time values that should logically exist.
8. Incorrect Time Formatting: Time values stored inconsistently or incorrectly, such as mixed 12-hour and 24-hour formats, invalid time components (e.g., "9:60 AM"), or ambiguous representations (e.g., "12:00").
9. Ambiguous Time Intervals: Start and end times that do not specify AM/PM or lack sufficient context to determine the correct order.
10. Other Temporal Anomalies: You can impart any other temporal anomaly that you may seem fit for the specific table that you are processing.

**Rules**
• Output **only** a JSON array – no code-fence, no comments.
• Do **not** add or delete rows/columns.
• Do **not** include `//` comments inside the JSON itself.
• Leave untouched cells exactly as they are.
"""
    prompt += f"""

Columns: {cols_info['all_columns']}
Time Columns: {time_cols}
File ID: {file_id}
Dataset:
{df.to_json(orient='records')}
Return modified dataset now:
"""

    system_msg = "You are a data expert skilled at introducing temporal anomalies."
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
    # Attempt robust JSON extraction
    json_part = extract_json_from_response(reply)
    explanation = reply.replace(json_part or '', '').strip() if json_part else ''

    logs = [f"GPT raw reply for {file_id}:\n{reply}"]
    if not json_part:
        logs.append(f"[ERROR] Could not extract JSON array from GPT reply for {file_id}.")
        return df, logs, False
    if not validate_json_structure(json_part):
        logs.append(f"[ERROR] Malformed JSON from GPT for {file_id}:\n{json_part}")
        return df, logs, False

    modified = pd.DataFrame(json.loads(json_part))
    if explanation:
        logs.append(f"GPT anomaly explanations for {file_id}:\n{explanation}")

    anomalies_found = False
    for col in time_cols:
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
                logs.append(f"- Type: Temporal Anomaly; row {i+1}, col '{col}': '{orig}' → '{new}'.")

    if not anomalies_found:
        logs.append(f"[WARN] No temporal anomalies detected for {file_id}.")

    return modified, logs, anomalies_found


def process_in_chunks(df: pd.DataFrame, file_id: str, total_anomalies: int, chunk_size: int = 30):
    # Same chunking logic as before...
    n = len(df)
    enc = tiktoken.encoding_for_model("gpt-4o")
    avg_tokens = len(enc.encode(df.head(10).to_json(orient="records"))) / 10
    safe_budget = max_model_tokens * 0.6 - len(enc.encode("You are a data expert skilled at introducing temporal anomalies."))
    effective_chunk = min(chunk_size, max(1, int(safe_budget / avg_tokens)))
    boundaries = list(range(0, n, effective_chunk)) + [n]

    all_mods, all_logs = [], []
    any_anoms = False
    for start, end in zip(boundaries, boundaries[1:]):
        sub = df.iloc[start:end].reset_index(drop=True)
        slice_anoms = max(1, math.ceil(total_anomalies * (end-start) / n))
        sub_id = f"{file_id}_{start}-{end}"
        mod_sub, sub_logs, found = generate_temporal_anomalies(sub, sub_id, slice_anoms)
        all_mods.append(mod_sub)
        any_anoms |= found
        for entry in sub_logs:
            m = re.search(r"row (\d+)", entry)
            if m:
                local = int(m.group(1)); global_row = start + local
                entry = entry.replace(f"row {local}", f"row {global_row}")
            all_logs.append(entry)

    return pd.concat(all_mods, ignore_index=True), all_logs, any_anoms

# ────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────
for fname in sorted(os.listdir(input_folder)):
    if not fname.endswith('.json'):
        continue
    path = os.path.join(input_folder, fname)
    fid  = os.path.splitext(fname)[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    cols_info = analyze_columns(df)
    if not cols_info['time_columns']:
        print(f"Skipping {fname}: no time-related columns.")
        continue

    print(f"Processing: {fname}")
    total_anoms = math.ceil(len(df) * 0.35)
    mod_df, log_entries, anoms_present = process_in_chunks(df, fid, total_anoms)

    if anoms_present:
        out_path = os.path.join(output_folder, f"{fid}_updated.json")
        mod_df.to_json(out_path, orient="records", indent=4, force_ascii=False)
        print(f"Saved updated file: {out_path}")

        with open(log_file, 'a', encoding='utf-8') as lf:
            lf.write(f"Table: {fname}\n")
            lf.write("\n".join(log_entries) + "\n\n")
    else:
        print(f"No anomalies imparted for {fname}. File skipped.")

print("All modified files and logs have been saved.")

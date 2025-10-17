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
    Analyze the dataset structure and identify time-related columns.
    """
    columns_info = {
        "all_columns": df.columns.tolist(),
        "time_columns": [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in [
                "time", "date", "start", "end", "duration", "start_date", "end_date",
                "finish", "interval", "processing_time", "runtime", "time_taken", "total_time",
                "deleted_at", "log_time", "log_date", "entry", "exit", "deadline", "due_date",
                "arrival_time", "departure_time", "check_in", "check_out", "month", "year",
                "week", "quarter", "time_frame", "time_period"
            ])
        ],
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

def generate_temporal_anomalies(df, file_id, max_anomalies=6):
    """
    Generate temporal anomalies in the dataset by prompting GPT to introduce them.
    """
    columns_info = analyze_columns(df)
    time_columns = columns_info["time_columns"]
    
    if not time_columns:
        print(f"No time-related columns found in {file_id}. Skipping...")
        return df, [], False

    prompt = f"""
First, thoroughly analyze the entire table. Understand its structure, context, and relationships between columns and rows. Do not skip this step.
Impart up to {max_anomalies} temporal anomalies based on the table's structure and contents. Use the examples below as guidance.
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

Dataset Summary:
- All Columns: {columns_info["all_columns"]}
- Time Columns: {time_columns}

File ID: {file_id}
Dataset:
{df.to_json(orient="records", lines=False)}
Modified Dataset in JSON:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data expert skilled at introducing temporal anomalies."},
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

        # Ensure no extra rows are added by trimming to original row count
        modified_data = modified_data[:len(df)]

        # Convert back to DataFrame
        modified_df = pd.DataFrame(modified_data)
        anomalies_found = False  # Track if any anomalies were imparted

        for col in time_columns:
            for idx in range(len(df)):
                if col in modified_df.columns and idx < len(modified_df):
                    original_value = df.at[idx, col] if col in df.columns else None
                    modified_value = modified_df.at[idx, col] if col in modified_df.columns else None
                    # Check if the original value is non-empty/non-null before modifying
                    if original_value != modified_value:
                        anomalies_found = True
                        if not pd.api.types.is_object_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].astype(object)
                        # Add prefix '@@@_' to modified values
                        modified_df.at[idx, col] = f"@@@_{modified_value}"
                        log_entries.append(f"- Type: Temporal Anomaly\n  Description: Cell at row {idx + 1}, column '{col}' was modified.")

        return modified_df, log_entries, anomalies_found

    except Exception as e:
        print(f"Error parsing GPT output for {file_id}: {e}")
        print("Saving original dataset without modification.")
        return df, log_entries, False

# Process each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    # >>> Changed: now we look for .json instead of .csv
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]
        try:
            # Instead of pd.read_csv, read JSON
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            df = pd.DataFrame(table_data)
            print(f"Processing: {filename}")

            modified_df, log_entries, anomalies_found = generate_temporal_anomalies(df, file_id)

            if anomalies_found:
                # Manually write JSON to avoid escaping slashes & unicode
                json_str = modified_df.to_json(None, orient="records", indent=4, force_ascii=False)
                # Replace escaped slash
                json_str = json_str.replace('\\/', '/')

                # Save the modified file
                output_path = os.path.join(output_folder, f"{file_id}_updated.json")
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(json_str)

                print(f"Saved updated file: {output_path}")

                # Save the log file
                log_path = os.path.join(log_folder, "temporal_anomalies_log.txt")
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Table: {filename}\n")
                    log_file.write("\n".join(log_entries))
                    log_file.write("\n\n")
            else:
                print(f"No anomalies imparted for {filename}. File skipped.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"All modified files and logs have been saved.")

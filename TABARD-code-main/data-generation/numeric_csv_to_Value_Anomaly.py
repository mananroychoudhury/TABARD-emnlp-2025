import os
import pandas as pd
import random
import json

# Folders for input, output, and Yes/No tables
input_folder = r"C:\Users\MAMANROY CHOUDHURY\Downloads\WikiTableQuestions-master\numeric_json_long_tables_spider_beaver"
output_folder = r"C:\Users\MAMANROY CHOUDHURY\Downloads\WikiTableQuestions-master\Value_Anomaly_Spider_Beaver"
yes_no_folder = r"C:\Users\MAMANROY CHOUDHURY\Downloads\WikiTableQuestions-master\Value_Anomaly_Spider_Beaver\YesNo_Tables"

log_file_path = r"C:\Users\MAMANROY CHOUDHURY\Downloads\WikiTableQuestions-master\Value_Anomaly_Spider_Beaver\value_anomaly_log.txt"

# Ensure output and Yes/No folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(yes_no_folder, exist_ok=True)

def generate_value_anomalies(df, num_anomalies):
    anomalies = []
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    applied_anomalies = 0

    if not numeric_columns:
        return df, anomalies  # No numeric columns

    while applied_anomalies < num_anomalies and numeric_columns:
        col = random.choice(numeric_columns)

        # Outlier
        if random.random() < 0.5 and applied_anomalies < num_anomalies:
            idx = random.choice(df.index)
            typical = df[col].mean() if not df[col].isnull().all() else 100
            outlier = typical * random.uniform(10, 20)
            df.at[idx, col] = int(outlier) if df[col].dtype == 'int64' else outlier
            anomalies.append({
                "type": "Outlier",
                "description": f"Column '{col}' contains an outlier value {df.at[idx, col]:.2f} at index {idx}."
            })
            applied_anomalies += 1

        # Negative value
        if random.random() < 0.5 and applied_anomalies < num_anomalies:
            idx = random.choice(df.index)
            typical = df[col].mean() if not df[col].isnull().all() else 100
            neg = -abs(typical)
            df.at[idx, col] = int(neg) if df[col].dtype == 'int64' else neg
            anomalies.append({
                "type": "Negative Value",
                "description": f"Column '{col}' contains an invalid negative value {df.at[idx, col]:.2f} at index {idx}."
            })
            applied_anomalies += 1

        # Empty cell
        if random.random() < 0.5 and applied_anomalies < num_anomalies:
            row = random.choice(df.index)
            col_name = random.choice(df.columns)
            df.at[row, col_name] = None
            anomalies.append({
                "type": "Empty Cell",
                "description": f"Cell at row {row}, column '{col_name}' was set to empty."
            })
            applied_anomalies += 1

    return df, anomalies

def generate_yes_no_table(df, anomalies):
    yes_no = pd.DataFrame("No", index=df.index, columns=df.columns)
    for a in anomalies:
        # parse outlier/negative
        if "Column '" in a["description"] and "index" in a["description"]:
            col = a["description"].split("Column '")[1].split("'")[0]
            row = int(float(a["description"].split("index ")[1].split(" ")[0]))
            yes_no.at[row, col] = "Yes"
        # parse empty cell
        elif "row" in a["description"] and "column" in a["description"]:
            row = int(float(a["description"].split("row ")[1].split(",")[0]))
            col = a["description"].split("column '")[1].split("'")[0]
            yes_no.at[row, col] = "Yes"
    return yes_no

def impart_value_anomalies(input_folder, output_folder, yes_no_folder, log_file_path):
    total_anom = 0
    table_count = 0

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for filename in os.listdir(input_folder):
            if not filename.endswith(".json"):
                continue

            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)
            yn_path  = os.path.join(yes_no_folder, filename.replace(".json", "_yes_no.json"))

            try:
                with open(in_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.DataFrame(data).infer_objects()

                # Determine anomalies based on row count: 10% of rows, at least 5
                row_count = len(df)
                num_anomalies = max(5, int(row_count * 0.15))

                print(f"Processing {filename} ({row_count} rows → {num_anomalies} anomalies)...")
                df_anom, anomalies = generate_value_anomalies(df, num_anomalies)

                df_anom.to_json(out_path, orient="records", indent=4)
                print(f"  → Saved anomalous table to {out_path}")

                yes_no = generate_yes_no_table(df_anom, anomalies)
                yes_no.to_json(yn_path, orient="records", indent=4)
                print(f"  → Saved Yes/No table to {yn_path}")

                # Log
                log_file.write(f"Table: {filename}\n")
                for a in anomalies:
                    log_file.write(f"- Type: {a['type']}\n")
                    log_file.write(f"  Description: {a['description']}\n")
                log_file.write("\n")

                total_anom += len(anomalies)
                table_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nProcessed {table_count} tables, imparted {total_anom} anomalies in total.")

if __name__ == "__main__":
    impart_value_anomalies(input_folder, output_folder, yes_no_folder, log_file_path)

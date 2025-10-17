import openai
import sys
import os
from tqdm import tqdm
import json
import re
import ast

# Set your OpenAI API key
openai.api_key = ""

# **Provide the folder paths directly in the code**
input_folder_path = r""
output_folder_path = r""
log_file_path = r""

def clean_output(output):
    """Removes unwanted Unicode characters and ensures clean JSON-like output."""
    output = re.sub(r"[^\x00-\x7F]+", "", output)  # Remove non-ASCII characters
    output = re.sub(r"```(json|python)?", "", output)  # Remove triple backticks if present
    return output.strip()

def generate_anomalies(input_folder_path, output_folder_path, log_file_path):
    input_files = sorted(os.listdir(input_folder_path))
    log_file = open(log_file_path, 'w', encoding="utf-8", errors="ignore")  # Fix log file encoding issue
    
    # Filter out only the JSON files
    json_files = [file for file in input_files if file.endswith('.json') and os.path.isfile(os.path.join(input_folder_path, file))]

    # Create the output folder if not present
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for json_file in tqdm(json_files, unit="file"):
        json_file_path = os.path.join(input_folder_path, json_file)
        output_file = os.path.join(output_folder_path, json_file.split("_")[0] + "_yes_no.json")

        try:
            with open(json_file_path, 'r', encoding="utf-8", errors="ignore") as file:  # Fix encoding issue
                try:
                    data_list = json.load(file)  # Try parsing JSON
                except json.JSONDecodeError as e:
                    print(f"Skipping {json_file} due to JSONDecodeError: {e}")
                    continue  # Skip this file if it has JSON issues

                yes_no_data = []
                # Convert the data to a string (JSON format)
                json_string = json.dumps(data_list)[:5000]  # Truncate JSON to avoid token overflow

                prompt = f"""Here is the JSON data: {json_string} which will have some security anomalies in its cells. 
                
                ### **🔹 Task: Structured Security Anomaly Detection in Semi-Structured Tables**
You are an advanced anomaly detection system trained to analyze semi-structured tables. Your goal is to **detect security anomalies at the cell level** using a structured **step-by-step approach** that ensures high accuracy, logical consistency, and explainability.

---

## **🔹 Step 1: Generate Multiple Independent Reasoning Paths (Self-Consistency Prompting)**
- Perform **multiple independent analyses** of the table, each using a unique reasoning approach.
- Ensure that **each reasoning path is completely independent** and may use different logical frameworks.
- Each reasoning path should **only flag specific table cells as anomalies** (not entire rows or columns).

---

## **🔹 Step 2: Apply Chain-of-Thought (CoT) for Each Reasoning Path**
For each independent reasoning path, use **step-by-step logical reasoning**:
1. **Identify patterns** in the data.
2. **Compare against expected norms** (historical data, rules, domain-specific expectations).
3. **Detect outliers or logical inconsistencies** in individual cells.
4. **Analyze cross-field relationships** (e.g., role vs. status, date vs. event).
5. **Conclude whether the flagged cells are anomalous or valid (whether security anomaly is present in them).**
Here are some examples of how security anomalies look like:

1. Suspended Role Conflict: A user marked as "inactive" still holds a "manager" role, posing a security risk if the account is used maliciously.
2. Missing Audit Logs: Failed login attempts with blank fields for "Browser Version" or "User-Agent".
3. Suspicious Activity: A user logs in from a different country every time, whereas they normally log in from a single region or office location.

---

## **🔹 Step 3: Security Anomaly Detection at the Cell Level**
- **Flag only specific table cells** that contain anomalies.
- Use a structured output format **(index, column_name)** where:
  - **Index** corresponds to the index in the list.
  - **Column_name** corresponds to the specific field where the anomaly is found.

---

## **🔹 Step 4: Self-Verification for Each Anomaly (True/False Check)**
Each reasoning path **must verify its own flagged security anomalies**:
1. **Ask yourself:** *“Is this anomaly truly incorrect?”*
2. Cross-check with:
   - Expected value distributions.
   - Logical consistency.
   - Historical data references.
3. **Final Decision (True/False):**
   - ✅ **True** → The flagged cell is a confirmed anomaly.
   - ❌ **False** → The flagged cell is actually valid and should not be marked.

---

## **🔹 Step 5: Majority Voting**
- **Collect flagged security anomalies from all reasoning paths.**
- If **most reasoning paths agree** i.e., if more than 70% of the reasoning paths agree that the flagged cell has an anomaly, then confirm the anomaly. 
- These majority voted security anomalies will be the final anomalies that you will report

---

## **🔹 Step 6: Re-Reading & Final Chain-of-Thought Verification**
- **Before finalizing results**, re-read the table **one last time**.
- Apply a **final structured CoT reasoning process**:
  1. **Scan the flagged anomalies again.**
  2. **Ensure no valid cells are incorrectly flagged.**
  3. **Make any necessary corrections to the anomaly list.**
- **Final Output: A refined, validated anomaly list.**

---

## **🔹 Final Output Format**
Return **only** the structured list of confirmed anomalies. Return the output in the format [(index, column_name), (index, column_name)] where index corresponds to the index in the list and column_name is the name of the column you think there is a security anomaly.
"""

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a data expert skilled at detecting anomalies."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=5000,
                    temperature=0.7,
                )

                output = clean_output(response.choices[0].message.content.strip())  # Remove unwanted characters
                log_file.write(f"{json_file}:\n {output}\n")

                try:
                    match = re.search(r"\[.*\]", output, re.DOTALL)
                    if match:
                        output_list = ast.literal_eval(match.group())  # Only parse if a match exists
                    else:
                        raise ValueError("No valid anomaly list found in the output")

                    for j, row in enumerate(data_list):
                        yes_no_dict = {key: "Yes" if (j, key) in output_list else "No" for key in row}
                        yes_no_data.append(yes_no_dict)

                    with open(output_file, 'w', encoding="utf-8", errors="ignore") as output_json:
                        json.dump(yes_no_data, output_json, indent=4, ensure_ascii=False)

                except Exception as e:
                    print(f"Error parsing the output for {json_file}: {e}")

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

# **No longer using command-line arguments**
if __name__ == "__main__":
    generate_anomalies(input_folder_path, output_folder_path, log_file_path)

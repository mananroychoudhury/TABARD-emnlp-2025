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

                prompt =   f"""Here is the JSON data: {json_string} 

### Task:
Analyze the data and *identify security anomalies. Follow a structured **step-by-step Chain-of-Thought (CoT) approach* before returning the final output.

---

### Step 1: Understand the Data Structure
1. Parse the JSON and *identify the key fields* relevant to security.

---

### Step 2: Find out the security anomalies present in the table.
For each record in the dataset, check for anomalies. Here are examples of security anomalies that can be present in the table:

1. Suspended Role Conflict: A user marked as "inactive" still holds a "manager" role, posing a security risk if the account is used maliciously.
2. Missing Audit Logs: Failed login attempts with blank fields for "Browser Version" or "User-Agent".
3. Suspicious Activity: A user logs in from a different country every time, whereas they normally log in from a single region or office location.
---

### Step 3: Find the Anomalous Cells
- Now, find the anomalous cells that you think security anomaly is present in them. Note down all those cells.

### Step 4: Self verification (True/False Check)
- Now, you *must verify your own flagged anomalies*:
1. *Ask yourself:* “Is this anomaly that I have flagged is truly an anomaly?”
2. Cross-check with:
   - Expected value distributions.
   - Logical consistency.
   - Historical data references.
3. *Final Decision (True/False):*
   - ✅ *True* → The flagged cell is a confirmed anomaly.
   - ❌ *False* → The flagged cell is actually valid and should not be marked.
4. Keep only these "True" confirmed anomalies as the final output.

### Step 5: Re-reading and final CoT checking
- *Before finalizing results, re-read the table *one last time*.
- Apply a *final structured CoT reasoning process*:
  1. *Scan the flagged anomalies again.*
  2. *Ensure no valid cells are incorrectly flagged.*
  3. *Make any necessary corrections to the anomaly list.*
- *Final Output: A refined, validated anomaly list.*

### Step 6: Final output generation
- Return the final output i.e., the flagged anomalous cells in the format [(index, column_name), (index, column_name)] where index corresponds to the index in the list and column_name is the name of the column you think there is a security anomaly. Just generate the list format output so I can easily parse it.
- Only output the list in this format for easy parsing."""

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

# ── anomaly_jsonl_generator.py ───────────────────────────────────────

import os
import json
import typing

anomoly_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {
                "type": "string",
                "description": "index of row with anomaly"
            },
            "anomaly_column": {
                "type": "string",
                "description": "col name of the anomaly"
            }
        },
        "required": ["index", "anomaly_column"],
        "additionalProperties": False
    }
}

class Anomaly(typing.TypedDict):
    index: int
    anomaly_column: str

Anomalies = typing.List[Anomaly]

def create_messages(img_data: typing.List[dict], id: str = None) -> dict:
    """
    Build the JSON payload (per Gemini’s format) embedding the table (img_data) plus the prompt.
    """
    json_string = json.dumps(img_data, ensure_ascii=False)

    prompt_str = f"""
                ---
                ### *📌 Important Clarification: Read This First*
                **IMPORTANT:**
                - DO NOT return anything else.
                - DO NOT include any explanation or commentary.
                - ONLY return the list of tuples as final output in the following format [(index, column_name), (index, column_name), ..., (index, column_name)].
                ---
Here is the JSON data: {json_string}

### Task:
Analyze the data and *identify anomalies. Follow a structured **step-by-step Chain-of-Thought (CoT) approach* before returning the final output.
---
### Step 1: Understand the Data Structure
1. Parse the JSON and identify the key fields relevant to different types of anomalies.
---
### Step 2: Find out the anomalies present in the table.
For each record in the dataset, check for anomalies. Here are examples of anomalies that can be present in the table:

*** Here are a few examples of logical anomalies to help you identify them:

1. Illogical Temporal Relationships: A delivery_date that occurs before the order_date.
2. Biological or Physical Impossibilities: An age value of 180 years or more for a person.
3. Inconsistent Financial Data: A discount greater than the total_price.
4. Categorical Misclassifications: An entry in the job_title column listed as "CEO" for a person with a salary below minimum wage.
5. Anachronisms or Technological Impossibilities: A passport_issued_date listed as before the person's birth_date.
6. Referential Inconsistencies: A customer_id in a transactions table that does not exist in the customer registry.

*** Here are a few examples of calculation anomalies to help you identify them:

1. Incorrect Totals: A "grand total" field that doesn't reflect the actual sum of individual transaction amounts.
2. Incorrect Formula: Cells with incorrect formulas (e.g., body fat percentage = (waist circumference / height) instead of the correct formula).
3. Missing Dependencies: A "discounted price" column that refers to missing values in the "original price" column.
4. Logical Violations: A calculation result that falls outside a plausible range (e.g., an age of 200 or a negative quantity of items in stock).
5. Rounding Errors: Financial calculations where figures have been rounded in a way that leads to inconsistency (e.g., rounding off tax amounts in different decimal places).

*** Here are a few examples of temporal anomalies to help you identify them:

1. Conflicting Schedules: A meeting that starts before the previous meeting has ended, creating an overlap.
2. Illogical Durations: A marathon with a recorded duration of 1 second.
3. Chronological Inconsistencies: A departure_time that occurs earlier than the arrival_time for a train schedule.
4. Unrealistic Temporal Outliers: A task completed in negative time, such as -2 minutes.
5. Timezone Discrepancies: An event occurring at 9:00 AM in one timezone but incorrectly listed as 10:00 AM in another.
6. Invalid Temporal Sequences: A work shift that ends before it starts.

*** Here are a few examples of factual anomalies to help you identify them:

1. Contradictions: An "assistant" with a higher salary than a "senior manager" in the same dataset.
2. Unrealistic Values: A product price of $1 for a high-end smartphone or $10,000 for a notebook.
3. Geographical Mismatches: A city name listed as "New York" with a postal code that belongs to Los Angeles.
4. Ambiguities: A currency column where values like "Dollar" are provided without specifying the type (e.g., USD or CAD).
5. Record-Breaking Claims: A 100-meter race time of 8 seconds, which would break the current world record.
6. Unlikely Proportions: A company reporting 95% expenses relative to revenue in an industry where the norm is 70%.

*** Here are a few examples of normalization anomalies to help you identify them:

1. Partial Dependencies (2NF Violation): In an "Orders" table with OrderID and ProductID as a composite key, modify some rows so that CustomerName depends only on CustomerID, violating 2NF.
2. Transitive Dependencies (3NF Violation): In an "Employees" table, OfficeLocation depends on Department, which depends on EmployeeID, causing transitive dependency.
3. Denormalization: TotalSalary = BaseSalary + Bonus, but modify data so this value is directly stored and may not match computed values.
4. Combined Attributes: "West Bengal, India, 721306" is stored in one column instead of City, Country, and Zip fields.
5. Repeating Groups (1NF Violation): A "Skills" column storing "MongoDB, C++, C" instead of separate normalized entries.

*** Here are a few examples of value anomalies to help you identify them:

1. Missing or Null Values: A numeric field like "Price" or "Quantity" left blank in a sales record.
2. Illogical Negative Values: A "Price" column with negative values, which is not possible for most products.
3. Extreme Outlier Values: A house listed with a price of $1 or $1 billion in a neighborhood where the typical range is $300,000 to $500,000.

*** Here are a few examples of security anomalies to help you identify them:

1. Suspended Role Conflict: A user marked as "inactive" still holds a "manager" role, posing a security risk if the account is used maliciously.
2. Missing Audit Logs: Failed login attempts with blank fields for "Browser Version" or "User-Agent".
3. Suspicious Activity: A user logs in from a different country every time, whereas they normally log in from a single region or office location.

Here are a few examples of data consistency anomalies to help you identify them:

1. Inconsistent Formats: A "phone number" column where values use different formats (e.g., +1-234-567-8901, (123) 456-7890, or unformatted like 1234567890).
2. Mismatched Categories: Different naming conventions used for the same category (e.g., "HR", "Human 	Resources", and "H.R.").
3. Cross-Table Inconsistencies: Salary or payment values that differ between related tables (e.g., expected_pay vs. exact_pay).
---
### Step 3: Generate the Anomalous Cells
- Return the output in the format [(index, column_name), (index, column_name)] where index corresponds to the index in the list and column_name is the name of the column you think there is an anomaly. Just generate the list format output so I can easily parse it."""





    data = {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_str}]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": anomoly_schema,
                "max_output_tokens": 5000,
            }
        },
        "id": id
    }
    return data

def process_flat_directory(input_directory: str, output_jsonl_path: str):
    """
    Reads every .json file directly under input_directory,
    calls create_messages(...) on its contents, and appends each result as a line in one JSONL.
    """
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # Find all .json files in the flat input_directory
    all_files = sorted(os.listdir(input_directory))
    json_files = [
        fn for fn in all_files
        if fn.lower().endswith(".json") and os.path.isfile(os.path.join(input_directory, fn))
    ]

    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for file_name in json_files:
            file_path = os.path.join(input_directory, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    img_data = json.load(f)
                if not isinstance(img_data, list):
                    # If the JSON isn’t a top‐level list, skip it
                    print(f"[Skip] {file_name} (not a list)")
                    continue

                # Use file_name (without extension) as the `id` for clarity
                base_name = os.path.splitext(file_name)[0]
                message = create_messages(img_data, id=base_name)

                # Write one JSON object per line
                jsonl_file.write(json.dumps(message, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"[Error] Failed to parse {file_name}: {e}")
            except Exception as e:
                print(f"[Error] Unexpected issue with {file_name}: {e}")

    print(f"→ Wrote JSONL payload to {output_jsonl_path}")

def main(input_directory: str, output_directory: str):
    """
    Entry point for this module. Creates `output_directory` if needed, then
    writes `output.jsonl` under it by processing all JSONs in `input_directory`.
    """
    os.makedirs(output_directory, exist_ok=True)
    output_jsonl = os.path.join(output_directory, "output.jsonl")
    process_flat_directory(input_directory, output_jsonl)

import json
import os
import typing


anomoly_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {
                "type": "string",  # should be integer, not string
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

# Define the structure of Anomaly object
class Anomaly(typing.TypedDict):
    index: int
    anomaly_column: str
Anomalies = typing.List[Anomaly]

def chunk_table_data(table_data, max_rows=100):
    if not isinstance(table_data, list) or not all(isinstance(row, dict) for row in table_data):
        raise ValueError("Expected top-level JSON to be a list of dictionaries.")

    # Chunk the table data into parts of at most max_rows rows
    chunks = [table_data[i:i + max_rows] for i in range(0, len(table_data), max_rows)]
    return chunks

# Function to generate messages with JSON data incorporated into the prompt
def create_messages(img_data, obj=None, prompt=None, id=None,type_of_anomaly=''):
    # Convert the JSON data to a string for inclusion in the prompt
    json_string = json.dumps(img_data, ensure_ascii=False)
    
    # Construct the prompt by embedding the JSON data string
    prompt_str =  f"""---
                ### *📌 Important Clarification: Read This First*
                **IMPORTANT:**
                - DO NOT return anything else.
                - DO NOT include any explanation or commentary.
                - ONLY return the list of tuples as final output in the following format [(index, column_name), (index, column_name), ..., (index, column_name)].
        ---
Here is the JSON data: {json_string}

### Task:
Analyze the data and *identify calculation anomalies. Follow a structured **step-by-step Chain-of-Thought (CoT) approach* before returning the final output.
---

### Step 1: Understand the Data Structure
1. Parse the JSON and *identify the key fields* relevant to calculation anomalies.
---

### Step 2: Find out the anomalies present in the table.
For each record in the dataset, check for anomalies. Here are examples of anomalies that can be present in the table:

1. Incorrect Totals: A "grand total" field that doesn't reflect the actual sum of individual transaction amounts.
2. Incorrect Formula: Cells with incorrect formulas (e.g., body fat percentage = (waist circumference / height) instead of the correct formula).
3. Missing Dependencies: A "discounted price" column that refers to missing values in the "original price" column.
4. Logical Violations: A calculation result that falls outside a plausible range (e.g., an age of 200 or a negative quantity of items in stock).
5. Rounding Errors: Financial calculations where figures have been rounded in a way that leads to inconsistency (e.g., rounding off tax amounts in different decimal places).
---

### Step 3: Generate the Anomalous Cells
- Return the output in the format [(index, column_name), (index, column_name)] where index corresponds to the index in the list and column_name is the name of the column you think there is a calculation anomaly. Just generate the list format output so I can easily parse it.

"""


    # Now, incorporate the prompt string into the data and return
    data = {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt_str  # Embedding the generated prompt
                        },
                    ]
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

map_dict = {"Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))":"Calculation",
            "Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))":"Data_Consistency",
            "Factual_Anomaly_(dataset_name ie.(FetaQA ... ))":"Factual_Anomaly",
            "Logical_Anomaly_(dataset_name ie.(FetaQA ... ))":"Logical_Anomaly",
            "Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))":"Normalization",
            "Security_anomaly_(dataset_name ie.(FetaQA ... ))":"Security",
            "Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))":"Temporal",
            "Value_Anomaly_(dataset_name ie.(FetaQA ... ))":"Value",
}
List = ["Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))"]
# List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']

# Function to process a subdirectory and create a JSONL file for each
def process_subdirectory(input_directory, output_directory):
    # Walk through the directory to find subdirectories
    for subdir, dirs, files in os.walk(input_directory):
        if subdir == input_directory:
            continue  # Skip the root input directory, only process subdirectories
            
        subdir_name = os.path.basename(subdir)  # Get the subdirectory name

        if not os.path.isdir(subdir):
            print(f"Skipping {subdir_name}, as stripped folder doesn't exist.")
            continue
        
        type_of_anomaly = map_dict[subdir_name]
        jsonl_data = []
        
        if subdir_name in List:
            # Process each JSON file in the stripped folder
            for file in os.listdir(subdir):
                if file.endswith('.json'):  # Process only JSON files
                    json_path = os.path.join(subdir, file)
                    
                    # Read the JSON file
                    with open(json_path, 'r', encoding='utf-8') as f:
                        img_data = json.load(f)

                    message = create_messages(img_data, id=f"{subdir_name}/{file}_yes_no.json", type_of_anomaly=type_of_anomaly)
                    jsonl_data.append(message)
                    # table_chunks = chunk_table_data(img_data, max_rows=20)
                    # # Create message for Gemini evaluation
                    # for i, chunk in enumerate(table_chunks):
                    #     chunk_id = f'{subdir_name}/{file.replace(".json", f"_chunk_{i+1}.json")}'
                    #     messages = create_messages(chunk, obj=None, prompt=None, id=chunk_id, type_of_anomaly=type_of_anomaly)
                    #     jsonl_data.append(messages)
                    
                    # # Append the generated message to the jsonl_data list
                    # jsonl_data.append(messages)
            
            # Write the output data to a JSONL file specific to this subdirectory
            if jsonl_data:  # Only create the file if there is data
                # output_subdir_path = os.path.join(output_directory, subdir_name)
                # os.makedirs(output_subdir_path, exist_ok=True)
                output_file_path = os.path.join(output_directory, f'{subdir_name}.jsonl')
                
                with open(output_file_path, "w") as jsonl_file:
                    for entry in jsonl_data:
                        jsonl_file.write(json.dumps(entry) + "\n")
                print(f"Processed {subdir_name}: {len(jsonl_data)} files")

# Main function to run the process
def main(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Process each subdirectory and generate its own JSONL file
    process_subdirectory(input_directory, output_directory)
    
    print(f"Batch processing completed. Results saved to {output_directory}")

# Entry point to run the script
if __name__ == "__main__":
    input_dir = r"..(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-chunked"  # Take input directory as user input
    output_dir = r"..(dataset_name ie.(FetaQA ... ))/batch-files-gemini/(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-wcot"  # Take output directory as user input
    main(input_dir, output_dir)

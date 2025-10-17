import json
import os
import typing
import tiktoken

max_model_tokens = 16384
# Schema remains for documentation
class Anomaly(typing.TypedDict):
    index: int
    anomaly_column: str

def get_tokenizer(model_name="gpt-4o"):
    return tiktoken.encoding_for_model(model_name)

# --- Count tokens ---
def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

# --- Chunk data by token count ---
def chunk_data_tokenwise(data: list, max_tokens: int, tokenizer) -> typing.Generator[typing.Tuple[list, int], None, None]:
    chunk = []
    current_tokens = 0
    offset = 0

    for i, item in enumerate(data):
        item_str = json.dumps(item, ensure_ascii=False)
        item_tokens = count_tokens(item_str, tokenizer)

        if current_tokens + item_tokens > max_tokens:
            yield chunk, offset
            chunk = [item]
            current_tokens = item_tokens
            offset = i
        else:
            chunk.append(item)
            current_tokens += item_tokens

    if chunk:
        yield chunk, offset

# Updated for GPT-4o prompt formatting
def create_gpt4o_message(img_data, id=None, type_of_anomaly=''):
    json_string = json.dumps(img_data, ensure_ascii=False)

    prompt_str =  f"""---
                ### *📌 Important Clarification: Read This First*
                **IMPORTANT:**
                - DO NOT return anything else.
                - DO NOT include any explanation or commentary.
                - ONLY return the list of tuples as final output in the following format [(index, column_name), (index, column_name), ..., (index, column_name)].
                ---

Here is the JSON data: {json_string}

### Task:
Analyze the data and *identify data consistency anomalies. Follow a structured **step-by-step Chain-of-Thought (CoT) approach* before returning the final output.
---

### Step 1: Understand the Data Structure
1. Parse the JSON and *identify the key fields* relevant to data consistency anomalies.
---

### Step 2: Find out the anomalies present in the table.
For each record in the dataset, check for anomalies. Here are examples of anomalies that can be present in the table:

1. Inconsistent Formats: A "phone number" column where values use different formats (e.g., +1-234-567-8901, (123) 456-7890, or unformatted like 1234567890).
2. Mismatched Categories: Different naming conventions used for the same category (e.g., "HR", "Human 	Resources", and "H.R.").
3. Cross-Table Inconsistencies: Salary or payment values that differ between related tables (e.g., expected_pay vs. exact_pay).
---

### Step 3: Generate the Anomalous Cells
- Return the output in the format [(index, column_name), (index, column_name)] where index corresponds to the index in the list and column_name is the name of the column you think there is a data consistency anomaly. Just generate the list format output so I can easily parse it.

               """    
    system_msg = "You are an advanced anomaly detection system."
    enc = tiktoken.encoding_for_model("gpt-4o")
    in_tokens = len(enc.encode(system_msg)) + len(enc.encode(prompt_str))
    max_out = max_model_tokens - in_tokens - 50

    return {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are an advanced anomaly detection system."},
                {"role": "user", "content": prompt_str}
            ],
            "max_tokens": max_out
        }
    }

# Anomaly type mapping
map_dict = {"Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))":"Calculation",
            "Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))":"Data_Consistency",
            "Factual_Anomaly_(dataset_name ie.(FetaQA ... ))":"Factual_Anomaly",
            "Logical_Anomaly_(dataset_name ie.(FetaQA ... ))":"Logical_Anomaly",
            "Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))":"Normalization",
            "Security_anomaly_(dataset_name ie.(FetaQA ... ))":"Security",
            "Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))":"Temporal",
            "Value_Anomaly_(dataset_name ie.(FetaQA ... ))":"Value",
}
List = ["Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))"]
# List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']

def process_subdirectory(input_directory, output_directory):
    tokenizer = get_tokenizer()
    for subdir, dirs, files in os.walk(input_directory):
        if subdir == input_directory:
            continue  # Skip root

        subdir_name = os.path.basename(subdir)
        if not os.path.isdir(subdir):
            print(f"Skipping {subdir_name}, as it's not a valid directory.")
            continue

        type_of_anomaly = map_dict.get(subdir_name)
        if not type_of_anomaly:
            print(f"Skipping unknown anomaly type: {subdir_name}")
            continue

        jsonl_data = []
        
        if subdir_name in List:
            for file in os.listdir(subdir):
                if file.endswith('.json'):
                    json_path = os.path.join(subdir, file)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        img_data = json.load(f)

                    message = create_gpt4o_message(img_data, id=f"{subdir_name}/{file}_yes_no.json", type_of_anomaly=type_of_anomaly)
                    jsonl_data.append(message)
                    # for chunk_id, (chunk, offset) in enumerate(chunk_data_tokenwise(img_data, max_tokens=max_model_tokens, tokenizer=tokenizer)):
                    #     custom_id = f'{subdir_name}/{file.replace(".json", f"_chunk_{chunk_id}_offset_{offset}.json")}'
                    #     message = create_gpt4o_message(chunk, id=custom_id, type_of_anomaly=type_of_anomaly)
                    #     jsonl_data.append(message)


            if jsonl_data:
                output_file_path = os.path.join(output_directory, f'{subdir_name}.jsonl')
                with open(output_file_path, "w", encoding='utf-8') as jsonl_file:
                    for entry in jsonl_data:
                        jsonl_file.write(json.dumps(entry) + "\n")
                print(f"Processed {subdir_name}: {len(jsonl_data)} files")

def main(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    process_subdirectory(input_directory, output_directory)
    print(f"Batch processing completed. Results saved to {output_directory}")

if __name__ == "__main__":
    input_dir = r"(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-chunked"  # Take input directory as user input
    output_dir = r"(dataset_name ie.(FetaQA ... ))/batch-files-gpt/(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level4-wcot" 
    main(input_dir, output_dir)

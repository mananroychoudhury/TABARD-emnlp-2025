from openai import OpenAI
import time 
from tqdm.auto import tqdm
import logging
import os

def setup_logger(log_name: str = "batch_logger", log_file: str = "gpt4o_batch.log", level=logging.INFO):
    """
    Sets up and returns a file-only logger (no console output).
    
    Parameters:
    - log_name (str): Name of the logger.
    - log_file (str): Filename for log output.
    - level (int): Logging level.
    
    Returns:
    - logger (logging.Logger): Configured logger.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger(log_file="(dataset_name ie.(FetaQA ... ))_gpt-4O-batch.log")
api_key = "Api Key"
client = OpenAI(api_key=api_key)
logger.info("Started batch job")

folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-gpt4o-prompt-2-1']
List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']


for fold in tqdm(folder,desc="Processing Folder"):
    for i in tqdm(List, desc="Processing batches"):
        input_jsonl_path = f"..dataset_name ie.(FetaQA ... ))/batch-files-gpt/{fold}/{i}.jsonl"
        output_jsonl_path = f"..(dataset_name ie.(FetaQA ... ))/gpt-output/output_folder-{fold}/{i}.jsonl"
        try:
            logger.info(f"Starting batch for: {i}")
            batch_input_file = client.files.create(
                file=open(input_jsonl_path, "rb"),
                purpose="batch"
            )
            logger.info(f"Uploaded input file: {batch_input_file.id}")

            batch = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"{i} - anomaly-detection-batch-job"}
            )
            logger.info(f"Created batch: {batch.id}")

            def wait_for_batch(batch_id, interval=30):
                while True:
                    b = client.batches.retrieve(batch_id)
                    logger.info(f"Batch {batch_id} status: {b.status}")
                    if b.status in ["completed", "failed", "cancelled", "expired"]:
                        return b
                    time.sleep(interval)

            batch = wait_for_batch(batch.id)

            if batch.status == "completed" and batch.output_file_id:
                result_file = client.files.content(batch.output_file_id)

                output_dir = os.path.dirname(output_jsonl_path)
                os.makedirs(output_dir, exist_ok=True)

                with open(output_jsonl_path, "w", encoding="utf-8") as f:
                    f.write(result_file.text)
                logger.info(f"Batch output saved to {output_jsonl_path}")
            else:
                logger.warning(f"Batch {batch.id} finished with status: {batch.status}. No output saved.")

        except Exception as e:
            logger.error(f"Error in processing {i}: {str(e)}")


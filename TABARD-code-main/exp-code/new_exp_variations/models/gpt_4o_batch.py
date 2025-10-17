from openai import OpenAI
import time 
from tqdm.auto import tqdm
import logging
import os
from src.logger import setup_custom_logger

logger = setup_custom_logger(
    logfile_name="gpt4o_batch_prediction_job_merged.log",
    level=logging.INFO,
    log_dir="log"
)

api_key = "API_KEY"

client = OpenAI(api_key=api_key)
logger.info("Started batch job")

# folder = ['FetaQA-merged','Spider_Beaver-merged','wikiTQ-merged'] ## UnComment this out while running for Merged folder
# List = ['l1_cot','l1_wcot','l2_cot','l2_wcot','l4_cot','l4_wcot','museve','sevcot'] ## UnComment this out while running for Merged folder

DIR = ["variation_1","variation_2","variation_3"] ## UnComment this out while running for Variation folder
folder = ["FetaQA", "Spider_Beaver", "WikiTQ"] ## UnComment this out while running for Variation folder
List = ['museve','sevcot'] ## UnComment this out while running for Variation folder
for dir in tqdm(DIR,desc="Processing DIR"): ## UnComment this out while running for Variation folder
    for fold in tqdm(folder,desc="Processing Folder"):
        for i in tqdm(List, desc="Processing batches"):
            logger.info(f"Starting the batch prediction {fold}.")
            # input_jsonl_path = f"..Batchfiles/gpt4o/{fold}/{i}/output.jsonl" ## UnComment this out while running for Merged folder
            # output_jsonl_path = f"..predicitons/gpt4o/{fold}/{i}/predictions.jsonl" ## UnComment this out while running for Merged folder

            input_jsonl_path = f"..Batchfiles/gpt4o/{dir}/{fold}/{i}/output.jsonl" ## UnComment this out while running for Variation folder
            output_jsonl_path = f"..predicitons/gpt4o/{dir}/{fold}/{i}/predictions.jsonl" ## UnComment this out while running for Variation folder
            
            if not os.path.exists(output_jsonl_path):
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
            else:
                logger.info(f"Skiping the batch prediction for {fold}: {i}..........")

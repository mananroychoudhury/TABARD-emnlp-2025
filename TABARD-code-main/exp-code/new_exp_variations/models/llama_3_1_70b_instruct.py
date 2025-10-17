import vertexai
from vertexai.batch_prediction import BatchPredictionJob
import time
import logging
from tqdm.auto import tqdm

from src.logger import setup_custom_logger
import os
from google.cloud import storage 

logger = setup_custom_logger(
    logfile_name="llama-batch_prediction_job_levels.log",
    level=logging.INFO,
    log_dir="log"
)

BATCH_OUTPUT_BUCKET = "OUPUT_BUCKET"  # same as your output bucket
LLAMA_PREFIX       = "llama"

# Locally, we’ll store downloaded predictions here:
LOCAL_DOWNLOAD_BASE = "predicitons"

# Make sure the downloads base exists
os.makedirs(LOCAL_DOWNLOAD_BASE, exist_ok=True)


def download_prediction_files(dir,fold: str, batch: str):
    """
    After a batch job completes, look under:
      gs://<BATCH_OUTPUT_BUCKET>/gemini/output_folder-<fold>/<batch>/
    for any file whose name starts with "prediction" and ends with ".jsonl",
    then download all matches into:
      <LOCAL_DOWNLOAD_BASE>/<fold>/<batch>/
    """
    # Initialize the GCS client
    client = storage.Client()

    # Form the GCS prefix for this completed job
    # gcs_prefix = f"{LLAMA_PREFIX}/output_folder-{dir}/{fold}/{batch}/"

    gcs_prefix = f"{LLAMA_PREFIX}/output_folder-{fold}/{batch}/"
    bucket = client.bucket(BATCH_OUTPUT_BUCKET)

    # List all blobs under that prefix
    blobs = bucket.list_blobs(prefix=gcs_prefix)

    # Prepare the local directory: downloads/<fold>/<batch>/
    # local_dir = os.path.join(LOCAL_DOWNLOAD_BASE,LLAMA_PREFIX,dir, fold, batch)
    local_dir = os.path.join(LOCAL_DOWNLOAD_BASE,LLAMA_PREFIX, fold, batch)
    os.makedirs(local_dir, exist_ok=True)

    found_any = False
    for blob in blobs:
        filename = os.path.basename(blob.name)
        if filename.startswith("000000000000.jsonl") or filename.startswith("predictions.jsonl"):
            found_any = True
            local_path = os.path.join(local_dir, filename)
            logger.info(f"Downloading gs://{BATCH_OUTPUT_BUCKET}/{blob.name} -> {local_path}")
            blob.download_to_filename(local_path)

    if not found_any:
        logger.warning(f"No prediction*.jsonl found under gs://{BATCH_OUTPUT_BUCKET}/{gcs_prefix}")


folder = ['FetaQA-merged','Spider_Beaver-merged','wikiTQ-merged']
DIR = ["variation_1","variation_2","variation_3"]
folder = ["FetaQA", "Spider_Beaver", "wikiTQ"]

List = ['l1_cot','l1_wcot','l2_cot','l2_wcot','l4_cot','l4_wcot','museve','sevcot']

# for dir in tqdm(DIR,desc="Processing Variations"): # uncomment for varitation only
for fold in tqdm(folder,desc="Processing Folder"):
    # 'Calculation_Based_Anomaly_FeTaQA',
    for i in tqdm(List, desc="Processing batches prompts"):

        # Log the start of the script
        logger.info("Starting the batch prediction job.")

        # Initialize Vertex AI
        vertexai.init(project="PROJECT_NAME", location="us-central1")

        # Set input/output URIs
        batch_input_bucket = f"BUCKET_NAME"
        batch_output_bucket = f"BUCKET_NAME"
        # input_uri = f"gs://{batch_input_bucket}/llama/{dir}/{fold}/{i}/output.jsonl"
        # output_uri = f"gs://{batch_output_bucket}/llama/output_folder-{dir}/{fold}/{i}/"
        input_uri = f"gs://{batch_input_bucket}/llama/{fold}/{i}/output.jsonl"
        output_uri = f"gs://{batch_output_bucket}/llama/output_folder-{fold}/{i}/"

        # Log input/output URIs
        logger.info(f"Input URI: {input_uri}")
        logger.info(f"Output URI: {output_uri}")

        # Submit the batch prediction job
        try:
            batch_prediction_job = BatchPredictionJob.submit(
                source_model="publishers/meta/models/llama-3.1-70b-instruct-maas",
                input_dataset=input_uri,
                output_uri_prefix=output_uri,
            )
            logger.info(f"Job resource name: {batch_prediction_job.resource_name}")
            logger.info(f"Model resource name with the job: {batch_prediction_job.model_name}")
            logger.info(f"Job state: {batch_prediction_job.state.name}")
        except Exception as e:
            logger.error(f"Error while submitting the job: {str(e)}")
            exit(1)

        # Refresh the job until complete
        while not batch_prediction_job.has_ended:
            time.sleep(5)
            batch_prediction_job.refresh()
            logger.info(f"Job state: {batch_prediction_job.state.name}")

        # Check if the job succeeded and log the result
        if batch_prediction_job.has_succeeded:
            logger.info("Job succeeded!")
        else:
            logger.error(f"Job failed: {batch_prediction_job.error}")

        # Log the job output location
        logger.info(f"Job output location: {batch_prediction_job.output_location}")

                
        download_prediction_files(None,fold, i)
import vertexai
from vertexai.batch_prediction import BatchPredictionJob
import time
import logging
from tqdm.auto import tqdm

folder = ['(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-2-1']
List = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']


for fold in tqdm(folder,desc="Processing Folder"):
    # 'Calculation_Based_Anomaly_FeTaQA',
    for i in tqdm(List, desc="Processing batches"):
        # Initialize logging
        logging.basicConfig(
            filename="gemini-batch_prediction_job.log",  # Log file name
            level=logging.INFO,                   # Log level (INFO, DEBUG, WARNING, ERROR)
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with timestamp
        )

        # Log the start of the script
        logging.info("Starting the batch prediction job.")

        # Initialize Vertex AI
        vertexai.init(project="PROJECT NAME", location="us-east1")

        # Set input/output URIs
        batch_input_bucket = f"Bucket Name"
        batch_output_bucket = f"Bucket Name"
        input_uri = f"gs://{batch_input_bucket}/{fold}/{i}.jsonl"
        output_uri = f"gs://{batch_output_bucket}/output_folder-{fold}/{i}/"

        # Log input/output URIs
        logging.info(f"Input URI: {input_uri}")
        logging.info(f"Output URI: {output_uri}")

        # Submit the batch prediction job
        try:
            batch_prediction_job = BatchPredictionJob.submit(
                source_model="gemini-1.5-pro-002",
                input_dataset=input_uri,
                output_uri_prefix=output_uri,
            )
            logging.info(f"Job resource name: {batch_prediction_job.resource_name}")
            logging.info(f"Model resource name with the job: {batch_prediction_job.model_name}")
            logging.info(f"Job state: {batch_prediction_job.state.name}")
        except Exception as e:
            logging.error(f"Error while submitting the job: {str(e)}")
            exit(1)

        # Refresh the job until complete
        while not batch_prediction_job.has_ended:
            time.sleep(5)
            batch_prediction_job.refresh()
            logging.info(f"Job state: {batch_prediction_job.state.name}")

        # Check if the job succeeded and log the result
        if batch_prediction_job.has_succeeded:
            logging.info("Job succeeded!")
        else:
            logging.error(f"Job failed: {batch_prediction_job.error}")

        # Log the job output location
        logging.info(f"Job output location: {batch_prediction_job.output_location}")

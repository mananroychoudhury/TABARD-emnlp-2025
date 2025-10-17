import os
import json
import logging
from tqdm.auto import tqdm
def get_logger(log_path):
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_path)  # unique name for each log file
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def compute_metrics(gt_items, pred_items):
    tp = fp = fn = 0

    for gt_row, pred_row in zip(gt_items, pred_items):
        for key in gt_row:
            gt_val = gt_row[key]
            pred_val = pred_row.get(key, "No")
            if gt_val == "Yes" and pred_val == "Yes":
                tp += 1
            elif gt_val == "No" and pred_val == "Yes":
                fp += 1
            elif gt_val == "Yes" and pred_val == "No":
                fn += 1
    return tp, fp, fn

def evaluate_predictions(gt_dir, pred_dir, folder_name, category_name, folder_logger):
    total_tp = total_fp = total_fn = 0
    folder_logger.info("Filename, TP, FP, FN, Precision, Recall, F1")
    for filename in os.listdir(gt_dir):
        if not filename.endswith(".json"):
            continue

        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if not os.path.exists(pred_path):
            folder_logger.warning(f"{filename} not found in predictions directory.")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        tp, fp, fn = compute_metrics(gt_data, pred_data)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        folder_logger.info(f"{filename}, {tp}, {fp}, {fn}, {precision:.4f}, {recall:.4f}, {f1:.4f}")
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0.0

    folder_logger.info("\n=== Overall Results ===")
    folder_logger.info(f"Total TP: {total_tp}")
    folder_logger.info(f"Total FP: {total_fp}")
    folder_logger.info(f"Total FN: {total_fn}")
    folder_logger.info(f"Precision: {overall_precision:.4f}")
    folder_logger.info(f"Recall: {overall_recall:.4f}")
    folder_logger.info(f"F1 Score: {overall_f1:.4f}")

    # Return per-category summary
    return {
        "folder": folder_name,
        "category": category_name,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    }

# Example usage
if __name__ == "__main__":
    folders = ['(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level1-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level2-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level3-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-ncot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-level4-wcot', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-1-1', '(dataset_name ie.(FetaQA ... ))-jsonl-gemini-prompt-2-1']

    Lists = ['Calculation_Based_Anomaly_(dataset_name ie.(FetaQA ... ))','Data_Consistency_Anomaly_(dataset_name ie.(FetaQA ... ))','Factual_Anomaly_(dataset_name ie.(FetaQA ... ))','Logical_Anomaly_(dataset_name ie.(FetaQA ... ))','Normalization_Anomaly_(dataset_name ie.(FetaQA ... ))','Security_anomaly_(dataset_name ie.(FetaQA ... ))','Temporal_Anomaly_(dataset_name ie.(FetaQA ... ))','Value_Anomaly_(dataset_name ie.(FetaQA ... ))']
    
    summary_log_path = "..(dataset_name ie.(FetaQA ... ))/gemini-prediction/global_summary.log"
    with open(summary_log_path, "w", encoding="utf-8") as summary_file:
        

        for folder in tqdm(folders,desc= "processing folders"):
            summary_file.write(f"\n\n########## Folder: {folder} ##########\n")
            for category in Lists:
                gt_dir = f"..(dataset_name ie.(FetaQA ... ))/(dataset_name ie.(FetaQA ... ))-yes-no/{category}"
                pred_dir = f"..(dataset_name ie.(FetaQA ... ))/gemini-prediction/prediction-{folder}/{category}"
                folder_log_path = f"{pred_dir}/evaluation_log.log"
                folder_logger = get_logger(folder_log_path)
                result = evaluate_predictions(gt_dir, pred_dir, folder, category, folder_logger)

                summary_file.write(f"\n=== Overall Results - {result['category']} ===\n")
                summary_file.write(f"Total TP     : {result['TP']}\n")
                summary_file.write(f"Total FP     : {result['FP']}\n")
                summary_file.write(f"Total FN     : {result['FN']}\n")
                summary_file.write(f"Precision    : {result['precision']:.4f}\n")
                summary_file.write(f"Recall       : {result['recall']:.4f}\n")
                summary_file.write(f"F1 Score     : {result['f1']:.4f}\n")
                summary_file.write("=======================================================================\n")

    print(f"✅ Global summary file written to: {summary_log_path}")

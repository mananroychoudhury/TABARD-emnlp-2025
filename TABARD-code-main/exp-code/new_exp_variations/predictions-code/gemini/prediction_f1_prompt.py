import os
import json
import logging
from tqdm.auto import tqdm


def get_logger(log_path):
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_path)  # unique logger per file
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if this logger is reused
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


def evaluate_predictions(gt_dir, pred_dir, fold_name, batch_name, folder_logger):
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

    return {
        "fold": fold_name,
        "batch": batch_name,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    }


if __name__ == "__main__":
    # ─── Configuration ────────────────────────────────────────────────────────
    # FOLDS  = ["FetaQA-merged", "Spider_Beaver-merged", "wikiTQ-merged"]
    # BATCHS = ['l1_cot','l1_wcot','l2_cot','l2_wcot','l4_cot','l4_wcot','museve','sevcot']

    FOLDS  = ["Spider_Beaver-merged"]
    BATCHS = ['museve']
    # DIR = ["variation_1","variation_2","variation_3"]
    # FOLDS = ["FetaQA", "Spider_Beaver", "wikiTQ"]
    # BATCHS = ['museve','sevcot']

    # Ground-truth base: each fold has "Merged-chunked/Merged-yes-no/<batch>/"
    GROUNDTRUTH_ROOT = r"..dataset"

    # Gemini predictions root: "predictions\gemini\<fold>\<batch>\predicted-merged\predicted-yes-no\"
    GEMINI_OUTPUT_ROOT = r"..predicitons\gemini"

    # Output global summary
    summary_log_path = r"..predicitons\gemini\global_summary_museve.log"
    os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)

    with open(summary_log_path, "w", encoding="utf-8") as summary_file:
        # for dir in tqdm(DIR, desc="Processing DIR", unit="fold"):
            # summary_file.write(f"\n\n########## DIR: {dir} ##########\n")
        for fold in tqdm(FOLDS, desc="Processing Folds", unit="fold"):
            summary_file.write(f"\n\n########## Fold: {fold} ##########\n")
            for batch in tqdm(BATCHS, desc="Processing Folds", unit="fold"):
                # Build ground-truth directory:
                #   <GROUNDTRUTH_ROOT>\<fold>\Merged-chunked\Merged-yes-no\<batch>\
                # gt_dir = os.path.join(
                #     GROUNDTRUTH_ROOT,
                #     dir,
                #     fold,
                #     "Merged-yes-no"
                # )

                gt_dir = os.path.join(
                    GROUNDTRUTH_ROOT,
                    fold,
                    "Merged-yes-no"
                )

                # Build prediction directory:
                #   <GEMINI_OUTPUT_ROOT>\<fold>\<batch>\predicted-merged\predicted-yes-no\
                # pred_dir = os.path.join(
                #     GEMINI_OUTPUT_ROOT,
                #     dir,
                #     fold,
                #     batch,
                #     "predicted-merged",
                #     "predicted-yes-no"
                # )

                pred_dir = os.path.join(
                    GEMINI_OUTPUT_ROOT,
                    fold,
                    batch,
                    "predicted-merged",
                    "predicted-yes-no"
                )

                # Ensure the output directory for this batch’s log exists
                os.makedirs(pred_dir, exist_ok=True)
                folder_log_path = os.path.join(pred_dir, "evaluation_log.log")
                folder_logger = get_logger(folder_log_path)

                # Evaluate
                result = evaluate_predictions(gt_dir, pred_dir, fold, batch, folder_logger)

                # Write batch summary to the global summary file
                summary_file.write(f"\n=== Overall Results - {batch} ===\n")
                summary_file.write(f"TP       : {result['TP']}\n")
                summary_file.write(f"FP       : {result['FP']}\n")
                summary_file.write(f"FN       : {result['FN']}\n")
                summary_file.write(f"Precision: {result['precision']:.4f}\n")
                summary_file.write(f"Recall   : {result['recall']:.4f}\n")
                summary_file.write(f"F1 Score : {result['f1']:.4f}\n")
                summary_file.write("--------------------------------------------------\n")

    print(f"✅ Global summary written to: {summary_log_path}")

# 📊 TABARD: A Novel Benchmark for Tabular Anomaly Analysis, Reasoning and Detection

**TABARD** is a benchmark for **Tabular Anomaly Analysis, Reasoning, and Detection**, designed to evaluate the ability of models to detect fine-grained anomalies in tabular data.  

**NOTE**: For the dataset please visit this github repository [https://github.com/TABARD-emnlp-2025/TABARD-dataset.git]
Refer to the website of the project for more info [https://tabard-emnlp-2025.github.io/]
This repository contains code to:

- Generate synthetic tabular datasets with controlled, labeled anomalies across multiple anomaly categories,
- Create multiple dataset variations via different sampling strategies,
- Run baseline and custom model experiments,
- Evaluate performance using precision, recall, and F1 metrics.

---

## 📁 Repository Structure

```bash
TABARD-code/
│
├── data-generation/
│
├── dataset_variation_code/
│
└── exp-code/
````

> **Important:** `exp-code/baselines/` is a folder that itself contains two subfolders:
> - `exp-code/baselines/baselines/` — baseline experiment code and pipelines  
> - `exp-code/baselines/new_exp_variations/` — new/custom experiment variations and models

---

## 📁 `data-generation/` — Dataset Creation & Anomaly Injection

**Purpose:**  
Scripts to **generate tabular datasets with controlled anomalies** across the TABARD taxonomy (calculation, factual, temporal, normalization, logical, data-consistency, security, etc.). Also contains utilities that create ground-truth yes/no label tables and simple baseline detectors, plus metric computation.

### Files & Descriptions

| File Name | Description |
|-----------|-------------|
| `numeric_csv_to_Security_Anomaly.py` | Injects **security-related anomalies** (e.g., suspicious or unauthorized-value patterns) into numeric CSV tables. |
| `numeric_csv_to_Calculation_Based_Anomaly.py` | Injects **calculation errors** (wrong arithmetic, miscomputed aggregates). |
| `numeric_csv_to_Data_Consistency_Anomaly.py` | Injects **data consistency anomalies** (contradictions across rows/columns). |
| `numeric_csv_to_Factual_Anomaly.py` | Injects **factual anomalies** that contradict external facts or domain knowledge. |
| `numeric_csv_to_Logical_Anomaly.py` | Injects **logical contradictions** inside the table (e.g., impossible states). |
| `numeric_csv_to_Normalization_Anomaly.py` | Injects **normalization issues** (mixed units, inconsistent formats). |
| `numeric_csv_to_Temporal_Anomaly.py` | Injects **temporal anomalies** (out-of-order times, impossible timestamps). |
| `numeric_csv_to_*_LargeTables.py` | Variants of the above scripts targeted at **large tables / large-scale dataset generation**. Functionality mirrors the smaller-table scripts but optimized/configured for larger inputs. |
| `create_yes_no.py` | Produces **yes/no label tables** for each generated dataset marking whether each cell is anomalous (`yes`) or normal (`no`). These serve as ground-truth annotations. |
| `MUSEVE.py` | Implementation of the **MUSEVE baseline method** for anomaly detection (used as a benchmark). |
| `SEVCOT.py` | Implementation of the **SEVCOT baseline method** for anomaly detection (another benchmark). |
| `get_f1.py` | Computes evaluation metrics (Precision / Recall / F1) given model predictions and ground-truth yes/no tables. |

---

## 📁 `dataset_variation_code/` — Dataset Variation & Sampling

**Purpose:**  
A collection of sampling, conversion, and variation scripts to produce multiple **dataset variants** (stratified, weighted, structure-based, LCM-based, underperforming subsets, etc.) for robustness and ablation experiments.

### Files & Descriptions

| File Name | Description |
|-----------|-------------|
| `convert_csv_json.py` | Converts CSV datasets produced in `data-generation/` into JSON format (used by downstream pipelines). |
| `merging.py` | Merges multiple anomaly-injected tables into consolidated datasets or folds. |
| `preprocessing_yes_no_value_anomaly.py` | Preprocesses yes/no label tables and value-anomaly representations for evaluation. |
| `remove_tokens.py` | Removes undesired tokens, artifacts or placeholder strings from generated tables. |
| `rename_json.py` | Renames / re-structures JSON files for a consistent schema and filenames. |
| `variation_LCM.py` | Generates dataset variations using a **Least-Common-Measure (LCM)** sampling strategy. |
| `variation_LCM_performace_group_prob.py` | LCM-based variation emphasizing **performance-group probability** sampling. |
| `variation_LCM_structure.py` | LCM-based variations focusing on **structural diversity** (different column/row shapes). |
| `variation_performance_LCM_startified.py` | Stratified LCM variations based on model performance buckets. |
| `variation_performance_stratified.py` | Stratified sampling variations by performance-level or other metrics. |
| `variation_structure.py` | Structure-driven variations (altering table layouts, column types, sparsity). |
| `variation_underperformance.py` | Variations that emphasize **underperforming** or hard-to-predict subsets. |
| `variation_weighted.py` | Weighted-sampling based variations (weight by anomaly type, row importance, etc.). |
| `yes_no_tabel_gen.py` | Generates yes/no label tables for dataset variations (same purpose as `create_yes_no.py` but for variant datasets). |

---

## 📁 `exp-code/` — Experimentation & Evaluation

**Purpose:**  
This folder contains the experiment pipelines. Note the corrected hierarchy: `exp-code` has a single folder `baselines/`, and **inside** `exp-code/baselines/` there are two subfolders — `baselines/` and `new_exp_variations/`.
```bash
exp-code/
└── baselines/
├── baselines/
│ ├── eval_models.py
│ ├── preprocessing_code/
│ ├── predictions/
│ └── postprocess/
└── new_exp_variations/
├── models/
├── preprocessing_code/
├── predictions-code/
└── postprocess/
````

### `exp-code/baselines/baselines/`

**Purpose:** Baseline model implementations and the canonical evaluation pipeline for TABARD.

**Typical contents:**

| Item | Description |
|------|-------------|
| `eval_models.py` | Evaluates baseline models on TABARD datasets (runs models, produces predictions, calls `get_f1.py` or other metrics). |
| `preprocessing_code/` | Scripts to convert tables into the input format expected by baseline models (tokenization/flattening/feature extraction). |
| `predictions/` | Scripts to run baselines and store their outputs/predictions into files compatible with evaluation scripts. |
| `postprocess/` | Scripts to convert raw model outputs into cell-level yes/no labels and align them with ground-truth for metric calculation. |

### `exp-code/baselines/new_exp_variations/`

**Purpose:** New experimental methods and variations (custom models, new preprocessing, new prediction strategies).

**Typical contents:**

| Item | Description |
|------|-------------|
| `models/` | Implementations of new/custom model architectures or wrappers around LLMs / classifiers. |
| `preprocessing_code/` | Preprocessing that is specific to the new methods (different tokenization, contextualization or feature engineering). |
| `predictions-code/` | Scripts to run the new models and produce prediction files. |
| `postprocess/` | Postprocessing scripts tailored to the new models' output formats to produce yes/no labels or other evaluation outputs. |

---

## ⚙️ Quick Usage Examples

> **Note:** adjust script names / paths if your repo places files in slightly different locations.

1. **Generate base datasets (example: factual anomalies)**  
   ```bash
   cd data-generation
   python numeric_csv_to_Factual_Anomaly.py
   ````
    ```bash
    cd dataset_variation_code
    python variation_LCM.py
    ````
# from repository root
```bash
cd exp-code/baselines/baselines
python eval_models.py
cd exp-code/baselines/new_exp_variations
````

# run your model script
```bash
python models/run_my_model.py
cd data-generation
python get_f1.py --predictions <pred_file.json> --groundtruth <yes_no_table.json>
````

# Citation
@inproceedings{TABARD2025,
  title={TABARD: A Novel Benchmark for Tabular Anomaly Analysis, Reasoning and Detection},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}
}



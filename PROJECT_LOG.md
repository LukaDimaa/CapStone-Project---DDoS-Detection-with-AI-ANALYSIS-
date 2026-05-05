# DDoS Detection with AI + Multi-Dataset Support

## Project Goal

Build a DDoS detection pipeline that can standardize different CSV-style network datasets into one training and detection workflow.

The current codebase is focused on:

- Binary detection: `normal` vs `attack`
- Dataset adaptation from heterogeneous label formats
- Large CSV handling during detection through chunked processing
- Heuristic post-classification of predicted attacks into protocol-based attack types

## Current Architecture

```text
Raw CSV dataset
    -> src/dataset_adapter.py
    -> Standardized CSV with Label = normal/attack
    -> src/train_model.py
    -> models/ddos_model.pkl + models/feature_columns.pkl
    -> src/detect.py
    -> Output CSV with Prediction, Prediction_Label, Attack_Type
```

## Project Structure

```text
src/
├── dataset_adapter.py
├── detect.py
├── prepare_CIC2019.py
├── train_model.py
├── make_test_samples.py
└── make_simple.py

data/raw/
├── adapted_cic2019.csv
└── cicddos2019_dataset.csv

models/
├── ddos_model.pkl
└── feature_columns.pkl
```

## Verified From Code

This log is based on the current implementation in:

- [src/dataset_adapter.py](/home/lukadima/Documents/Cyber%20Security/Cyber%20Security%20and%20behaviour/Autumn%202026/CyberSecurity%20%20and%20Behaviour%20Community%20Placement%20INFO3016/DdoS%20detection%20and%20AI%20analysis%20-%20CapStone%20project%20./src/dataset_adapter.py)
- [src/train_model.py](/home/lukadima/Documents/Cyber%20Security/Cyber%20Security%20and%20behaviour/Autumn%202026/CyberSecurity%20%20and%20Behaviour%20Community%20Placement%20INFO3016/DdoS%20detection%20and%20AI%20analysis%20-%20CapStone%20project%20./src/train_model.py)
- [src/detect.py](/home/lukadima/Documents/Cyber%20Security/Cyber%20Security%20and%20behaviour/Autumn%202026/CyberSecurity%20%20and%20Behaviour%20Community%20Placement%20INFO3016/DdoS%20detection%20and%20AI%20analysis%20-%20CapStone%20project%20./src/detect.py)

No claims in this file should be read as runtime-verified unless explicitly marked as tested.

## Core Components

### 1. `src/dataset_adapter.py`

Purpose:

- Load a raw CSV dataset
- Detect the most likely label column
- Normalize labels to `normal` or `attack`
- Drop common identifier and metadata columns
- Save an adapted CSV for training or detection

Implemented details:

- Candidate label columns:
  - `Label`
  - `Class`
  - `Attack`
  - `Category`
  - `target`
- Normalized normal labels include:
  - `benign`
  - `normal`
  - `0`
  - `non-attack`
  - `non_attack`
  - `legitimate`
- Known attack labels include:
  - `attack`
  - `1`
  - `ddos`
  - `dos`
  - `drdos`
- Unknown labels default to `attack`
- The dataset is shuffled before being written out
- A warning is printed if only one class is present

Important constraint:

- Keep the output label column as `Label`
- Keep label values strictly as `normal` and `attack`

### 2. `src/train_model.py`

Purpose:

- Train a binary `normal` vs `attack` classifier
- Build a balanced dataset from the adapted input
- Save both the trained model and the feature column list

Implemented details:

- Input file is currently hardcoded as `data/raw/adapted_cic2019.csv`
- Data is shuffled before chunk collection
- Training data is balanced by collecting up to `25000` rows per class
- Label normalization logic is reused inside training
- Label-like columns are dropped from the feature matrix
- The model is a `RandomForestClassifier`
- Saved artifacts:
  - `models/ddos_model.pkl`
  - `models/feature_columns.pkl`

Important limitation:

- The current training script does not explicitly restrict the feature matrix to numeric-only columns before fitting
- This means the project summary should not claim that numeric-only filtering is already implemented in `src/train_model.py`

### 3. `src/detect.py`

Purpose:

- Load the trained model and saved feature schema
- Process large CSV files in chunks
- Predict `normal` vs `attack`
- Add a heuristic `Attack_Type` column for predicted attacks

Implemented details:

- Required artifacts:
  - `models/ddos_model.pkl`
  - `models/feature_columns.pkl`
- Missing feature columns are added with zero values
- Input rows are cleaned by:
  - normalizing column names
  - dropping common ID columns
  - replacing `inf` values
  - filling missing values with `0`
- Output columns added by detection:
  - `Prediction`
  - `Prediction_Label`
  - `Attack_Type`
- Detection results are written incrementally to a CSV file
- A summary verdict is printed after processing

## Attack-Type Heuristic Layer

The attack-type logic is applied after the binary model prediction.

Current logic in `src/detect.py`:

- If `Prediction == 0` -> `Normal`
- If protocol is `17` -> `UDP Flood`
- If protocol is `6` and `SYN_Flag_Count > ACK_Flag_Count` -> `SYN Flood`
- If protocol is `6` -> `TCP Flood`
- If protocol is `1` -> `ICMP Flood`
- Otherwise -> `General Attack`
- If required fields cannot be parsed -> `Unknown`

Important note:

- This is a heuristic protocol-based label, not a separately trained multiclass model

## Detection Verdict Logic

The current detection summary classifies traffic severity using attack ratio:

- `NORMAL` when no attacks are predicted
- `LOW RISK` when attack ratio is below `10%`
- `SUSPICIOUS` when attack ratio is below `50%`
- `ATTACK DETECTED` otherwise

## Lessons Reflected in the Project

### Model Generalization

A model trained on one dataset may not perform well on a different dataset, even when both are DDoS-related.

Likely reasons:

- Different feature distributions
- Different traffic capture conditions
- Different labeling conventions
- Missing or renamed columns between datasets

Practical implication:

- Multi-dataset support at the adapter level is implemented more clearly than multi-dataset model generalization at the model level
- The current code supports adapting multiple datasets, but broad cross-dataset detection accuracy should be treated as unverified until tested

## Workflow

### Step 1. Adapt a dataset

```bash
python src/dataset_adapter.py --input data/raw/cicddos2019_dataset.csv --output data/raw/adapted_cic2019.csv
```

### Step 2. Train the model

```bash
python src/train_model.py
```

### Step 3. Run detection

```bash
python src/detect.py --file data/raw/adapted_cic2019.csv --rows 50000 --chunk-size 5000 --verbose
```

## Expected Output

Detection output rows may include:

- `Prediction`
- `Prediction_Label`
- `Attack_Type`

Expected label values:

- `Prediction_Label`: `normal` or `attack`
- `Attack_Type`: `Normal`, `UDP Flood`, `TCP Flood`, `SYN Flood`, `ICMP Flood`, `General Attack`, or `Unknown`

## Current Status

### Implemented in code

- Dataset label normalization to `normal` / `attack`
- Adapted CSV generation
- RandomForest-based binary training pipeline
- Model artifact saving
- Chunked CSV detection
- Feature-column alignment using `feature_columns.pkl`
- Heuristic attack-type labeling
- Summary verdict generation

### Not yet verified in this log

- Accuracy on CIC2017, CIC2018, CIC2019, CIC2020, and custom datasets
- Stability across differently structured datasets
- Runtime success of the full pipeline in the current workspace
- Correctness of attack-type heuristics on real traffic ground truth

### Planned or future work

- Multi-dataset training and evaluation
- Better numeric feature enforcement during training
- PCAP ingestion through Wireshark or `tshark`
- Real-time detection
- Dashboard or monitoring UI
- False-positive reduction
- Better severity scoring and reporting

## Notes for Future Changes

When modifying this project:

- Do not break `dataset_adapter.py` label standardization
- Keep output labels strictly as `normal` and `attack`
- Preserve compatibility with `models/feature_columns.pkl`
- Preserve chunked processing in `src/detect.py`
- Treat attack-type labeling as a second-stage heuristic unless a real multiclass model is introduced

## Capstone Positioning

This project is best described as:

`A hybrid AI and rule-based DDoS detection framework with dataset standardization and chunked CSV analysis.`

Key capstone angle:

`It demonstrates practical machine learning for cybersecurity, the challenges of cross-dataset generalization, and the value of combining binary ML detection with protocol-aware heuristics.`

# Video Annotation System

## Overview

This project implements a **modular pipeline** for video processing, pose extraction, and annotation generation.

The pipeline includes:

- **Video preprocessing** (`src/psifx/`) — prepares raw video input for downstream processing
- **Pose extraction and clustering** (`src/pose/`) — extracts skeletal keypoints and groups poses
- **ML model training and prediction** (`src/models/`) — trains and runs the XGBoost classifier
- **Annotation generation** (`src/annotations/`) — produces annotation labels from model output
- **End-to-end orchestration** (`src/pipeline/full_pipeline.py`) — runs the complete pipeline in one command

This system is designed to be **reproducible, modular, and easy to extend**.

---

## Project Structure

```
video_annotation_system/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── psifx/                  # Video preprocessing
│   │   ├── __init__.py
│   │   └── processing.py
│   │
│   ├── pose/                   # Pose extraction and clustering
│   │   └── extractor.py
│   │
│   ├── models/                 # ML model training and inference
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── annotations/            # Annotation label generation
│   │   └── generator.py
│   │
│   └── pipeline/               # End-to-end orchestrator
│       └── full_pipeline.py
│
├── data/                       # Raw and processed input data
├── models/                     # Saved model files (.joblib, .json)
└── notebooks/                  # Exploratory Jupyter notebooks
```

---

## Setup

### 1. Create and activate a Python virtual environment

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
```

### 2. Install dependencies

`pip install -r requirements.txt`  

### 3. Configure paths

Before running any script, open the relevant file and update the configuration block at the top (marked `# Configuration`) to point to your local data and output directories.

---

## Usage

### Run the full end-to-end pipeline

```bash
python src/pipeline/full_pipeline.py
```

### Train the model only

```bash
python src/models/train.py
```

### Run prediction on new data

```bash
python src/models/predict.py
```

### Generate annotations from predictions

```bash
python src/annotations/generator.py
```

---

## Pipeline Stages

### 1. Video Preprocessing — `src/psifx/processing.py`

Handles raw video input: frame extraction, format normalization, and preparation for pose estimation.

### 2. Pose Extraction — `src/pose/extractor.py`

Extracts 25-joint skeletal keypoints from video frames. Includes:
- Keypoint normalization relative to torso length
- Filtering of low-confidence detections
- Clustering of pose sequences

### 3. Model Training — `src/models/train.py`

Trains an XGBoost classifier on labeled pose data. Produces:
- `models/model_xgboost.joblib` — saved model
- `models/feature_names.json` — feature list used during training

Feature engineering includes joint angles, inter-joint distances, symmetry metrics, and temporal velocity/acceleration features.

### 4. Prediction — `src/models/predict.py`

Loads a trained model and runs inference on new pose data. Outputs:
- `predictions_<filename>.csv` — predicted annotation labels per frame
- `prediction_report_<filename>.txt` — accuracy and classification report (if ground truth is available)
- `confusion_matrix_<filename>.png` — visual performance summary

### 5. Annotation Generation — `src/annotations/generator.py`

Converts model predictions into structured annotation files ready for downstream analysis or review.

---

## Label Map

| Code | Meaning |
|------|---------|
| C    | Child |
| CCR  | Child–Child Reciprocal |
| CHO  | Child Hand-Over |
| CSI  | Child Side Interaction |
| CST  | Child Stand |
| T    | Teacher |
| TC   | Teacher–Child |
| TRE  | Teacher Reciprocal |
| TSI  | Teacher Side Interaction |
| TST  | Teacher Stand |

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`
- `matplotlib`
- `seaborn`

---

## Notes

- All scripts are executable standalone (`python script.py`) and follow `if __name__ == "__main__"` conventions.
- Intermediate outputs (processed CSVs, model files) are saved to the `models/` and `data/` directories.
- Notebooks in `notebooks/` contain exploratory analysis and were used to develop the pipeline logic.

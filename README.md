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

```bash
pip install -r requirements.txt
```

> **Note:** `psifx` is a local package. If it is not already installed, install it manually before running `requirements.txt`:
> ```bash
> pip install -e /path/to/psifx
> ```

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

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.8.0 | Deep learning backbone |
| `torchvision` | 0.23.0 | Video/image transforms |
| `torchaudio` | 2.8.0 | Audio processing |
| `transformers` | 4.57.0 | Pretrained model hub |
| `scikit-learn` | 1.7.2 | ML utilities and metrics |
| `xgboost` | *(via scikit-learn)* | Pose classifier |
| `pandas` | 2.2.3 | Data manipulation |
| `numpy` | 2.0.2 | Numerical computing |
| `opencv-python` | 4.12.0.88 | Video frame processing |
| `mediapipe` | 0.10.14 | Pose keypoint extraction |
| `psifx` | local | Video preprocessing (see below) |
| `faster-whisper` | 1.2.0 | Speech transcription |
| `pyannote.audio` | 3.3.2 | Speaker diarization |
| `matplotlib` | 3.8.4 | Plotting and visualization |
| `joblib` | 1.5.2 | Model serialization |
| `ultralytics` | 8.3.212 | YOLO-based detection |
| `accelerate` | 1.10.1 | Multi-GPU / mixed precision |
| `optuna` | 4.5.0 | Hyperparameter optimization |
| `langchain` | 0.3.27 | LLM orchestration |

---

## Notes

- All scripts are executable standalone (`python3 processing.py`) and follow `if __name__ == "__main__"` conventions.
- Intermediate outputs (processed CSVs, model files) are saved to the `models/` and `data/` directories.
- Notebooks in `notebooks/` contain exploratory analysis and were used to develop the pipeline logic.

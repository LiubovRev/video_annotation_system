# Video Annotation System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

**A modular, config-driven pipeline for video preprocessing, pose extraction, annotation alignment, clustering, and ML-based behavior prediction.**


## Overview

This project implements a **modular, config‑driven pipeline** for video processing, pose extraction, optional clustering, annotation alignment, and machine‑learning‑based behavior classification.

### Key Features

- **Video preprocessing** — trimming, object tracking (SAMURAI), pose estimation (MediaPipe)  
- **Pose extraction** — JSON parsing, feature engineering, temporal features  
- **Pose clustering (optional)** — therapist–child interaction analysis  
- **Annotation alignment** — time‑based mapping of annotations to pose data  
- **Model training & prediction** — LightGBM, XGBoost, HistGradientBoosting  
- **End‑to‑end orchestration** — run the full pipeline with one command  

> The pipeline is **fully config‑driven** and supports automatic trim computation from annotation timestamps.

Video Annotation System Pipeline  
```
┌────────────────────────────┐  
│       Raw Video Files      │  
│     + Annotations (.txt)   │  
└─────────────┬──────────────┘  
              │  
              ▼  
┌──────────────────────────────┐  
│     Step 1: Video Processing │  
│ - Trim videos (optional)     │  
│ - Object tracking (SAMURAI)  │  
│ - Pose overlay (MediaPipe)   │  
└─────────────┬────────────────┘  
              │  
              ▼  
┌────────────────────────────────┐  
│     Step 2: Pose Extraction    │  
│ - Extract JSON keypoints       │  
│ - Normalize & compute features │  
│ - Save processed_data.csv      │  
└─────────────┬──────────────────┘  
              │  
        ┌─────┴─────┐  
        │ Optional  │  
        ▼           ▼  
┌───────────────────────────┐  
│ Step 3: Pose Clustering   │  
│ - Movement & proximity    │  
│ - Interaction analysis    │  
│ - Cluster assignments     │  
└─────────────┬─────────────┘  
              │  
              ▼  
┌──────────────────────────────┐  
│ Step 4: Annotation Alignment │  
│ - Map annotations to pose    │  
│ - Filter, trim & label frames│  
│ - Save labeled_features.csv  │  
└─────────────┬────────────────┘  
              │  
              ▼  
┌─────────────────────────────┐  
│ Step 5: Model Training      │  
│ - Combine datasets          │  
│ - Train XGBoost/LightGBM    │  
│ - Save model & feature list │  
└─────────────┬───────────────┘  
              │  
              ▼  
┌────────────────────────────┐  
│ Step 6: Prediction         │  
│ - Single-project or batch  │  
│ - Output predictions.csv   │  
│ - Optional reports & plots │  
└────────────────────────────┘  
```

---

## Project Structure

```
video_annotation_system/
│
├── src/
│   ├── video_processing/
│   ├── pose/
│   ├── annotations/
│   ├── models/
│   └── pipeline/
│
├── configs/
│   └── config.yaml              # Global configuration
│
├── data/
│   ├── raw/                     # Raw video projects
│   │   ├── project_01/
│   │   │   └── project_config.yaml   # Auto‑generated (optional)
│   │   └── project_02/
│   │
│   ├── processed/               # Step outputs per project
│   │   ├── project_01/
│   │   └── project_02/
│   │
│   └── annotations/             # Annotation files (.txt)
│
├── models/                      # Trained model artifacts
├── outputs/                     # Plots, reports, predictions
│   ├── model_<name>.joblib
│   ├── feature_names.json
│   ├── model_metrics.json
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── model_training_summary.txt
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** `psifx` is a local package; install it manually *before* installing `requirements.txt`:

```bash
pip install -e /path/to/psifx
```

### 3. Configure the pipeline

Edit:

```
configs/config.yaml
```

This file controls paths, flags, model parameters, and pipeline behavior.

Optional per‑project config:

```
data/raw/<project>/project_config.yaml
```

Stores trim times and metadata.

---

## Usage


### Run the full pipeline

```bash
python src/pipeline/full_pipeline.py
```

---

### Run individual steps

```bash
# Step 1: Video processing
python src/video_processing/processing.py

# Step 2: Pose extraction
python src/pose/extractor.py

# Step 3: Pose clustering (optional)
python src/pose/clustering.py

# Step 4: Annotation alignment
python src/annotations/generator.py

# Step 5: Model training
python src/models/train.py

# Step 6: Prediction
python src/models/predict.py
```

Each script reads `configs/config.yaml` and processes all projects in the configured directories.

---

## Pipeline Stages

### **Step 1 — Video Processing**

**Inputs:** raw video files (e.g., `camera_a.mkv`)  
**Outputs:** trimmed videos, tracking overlays, pose JSON archives  

- Trim times computed automatically from annotation timestamps if no `project_config.yaml` exists  
- Controlled by: `flags.skip_video_processing`

---

### **Step 2 — Pose Extraction**

**Inputs:** JSON archives from Step 1  
**Outputs:** `processed_data.csv` (frame‑level features, keypoints, temporal features)  

- Controlled by: `flags.skip_pose_extraction`

---

### **Step 3 — Pose Clustering (Optional)**

**Inputs:** `processed_data.csv`  
**Outputs:** cluster assignments, clinical metrics, plots  

- Controlled by: `flags.skip_pose_clustering`

---

### **Step 4 — Annotation Alignment**

**Inputs:** `processed_data.csv`, annotation `.txt` files  
**Outputs:** `labeled_features.csv`, annotation statistics plots  

- Automatically trims pose data to annotation timestamp ranges

---

### **Step 5 — Model Training**

**Inputs:** combined labeled features  
**Outputs:** trained model, feature names, metrics, plots  

- Training is skipped automatically if `model.file` already exists

---

### **Step 6 — Prediction**

**Inputs:** new pose data + trained model  
**Outputs:** predictions CSV, optional report, confusion matrix  

Modes:

- **Single project:** set `predict.data_path`
- **Batch mode:** leave `predict.data_path` empty

---

## Example Output

```bash
outputs/
├── plots/
│   ├── movement_speed.png
│   └── cluster_distribution_Therapist.png
├── reports/
│   └── annotation_stats_project_01.png
data/processed/project_01/
├── processed_data.csv
├── labeled_features.csv
└── PoseDir/
    └── *.tar.gz
```

## Automated Tests & Config Validation

Run via:

```bash
pytest tests/
```

---

## Tips

- **Auto‑trim from annotation:** pipeline uses first/last annotated event ± buffer  
- **Rerun a specific step:** set the corresponding `skip_*` flag to `false`  
- **Retrain model:** delete `models/model_xgboost.joblib`  
- **Enable pose clustering:** set `flags.skip_pose_clustering: false`

---

## Requirements

- **Python 3.10+**
- Key packages:
  - `torch`
  - `mediapipe`
  - `scikit-learn`
  - `lightgbm`
  - `xgboost`
  - `opencv-python`
  - `pyyaml`
  - `matplotlib`
  - `seaborn`

See `requirements.txt` for the full list.



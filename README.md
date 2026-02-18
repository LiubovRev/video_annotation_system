# Video Annotation System

## Overview

This project implements a **modular, config-driven pipeline** for video processing, pose extraction, clustering, annotation alignment, model training, and prediction.

The pipeline includes:

- **Video preprocessing** (`src/video_processing/processing.py`) — trim, track objects, extract pose keypoints
- **Pose extraction** (`src/pose/extractor.py`) — parse psifx JSON archives, build DataFrames
- **Pose clustering** (`src/pose/clustering.py`) — *optional* clinical analysis and unsupervised clustering
- **Annotation alignment** (`src/annotations/generator.py`) — map annotations to feature frames
- **Model training** (`src/models/train.py`) — train LightGBM / XGBoost / HistGradientBoosting classifiers
- **Prediction** (`src/models/predict.py`) — run inference on new projects
- **End-to-end orchestration** (`src/pipeline/full_pipeline.py`) — runs Steps 1–5 in one command

This system is **reproducible, modular, and config-driven**.

---

## Project Structure

```
video_annotation_system/
│
├── README.md
├── requirements.txt
├── config.yaml                 # Central configuration file
│
├── src/
│   ├── video_processing/       # Step 1: Video preprocessing
│   │   └── processing.py
│   │
│   ├── pose/                   # Steps 2 & 3: Pose extraction and clustering
│   │   ├── extractor.py        # Step 2 (required)
│   │   └── clustering.py       # Step 3 (optional)
│   │
│   ├── annotations/            # Step 4: Annotation alignment
│   │   └── generator.py
│   │
│   ├── models/                 # Steps 5 & 6: Training and prediction
│   │   ├── train.py
│   │   └── predict.py
│   │
│   └── pipeline/               # End-to-end orchestrator
│       └── full_pipeline.py
│
└── data/                       # Raw video projects, processed outputs
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

> **Note:** `psifx` is a local package. If not already installed, install it manually **before** running `requirements.txt`:
> ```bash
> pip install -e /path/to/psifx
> ```

### 3. Configure `config.yaml`

All paths, flags, and pipeline parameters are centralized in `config.yaml` at the project root.

**Key sections to update:**
- `directories.raw_video_root` — path to raw video project folders
- `directories.output_base_dir` — where to save outputs
- `directories.annotations_dir` — location of `.txt` annotation files
- `trim_times` — per-project trim ranges `[start_sec, end_sec, fps]`
- `pose_extraction.id_to_label` — mapping from tar filename prefix to person label

See `config.yaml` inline comments for full details.

---

## Usage

### Run the full end-to-end pipeline (Steps 1–5)

```bash
python src/pipeline/full_pipeline.py
```

This will:
1. Process all raw videos (trim, track, extract pose)
2. Extract and align pose keypoints from psifx `.tar.gz` archives
3. *Optionally* run pose clustering and clinical analysis (if `flags.skip_pose_clustering: false`)
4. Align annotations to feature frames
5. Combine all projects → sanitize → train the best model

### Run individual steps standalone

**Step 1 — Video processing**
```bash
python src/video_processing/processing.py
```

**Step 2 — Pose extraction**
```bash
python src/pose/extractor.py
```

**Step 3 — Pose clustering** (optional)
```bash
python src/pose/clustering.py
```

**Step 4 — Annotation alignment**
```bash
python src/annotations/generator.py
```

**Step 5 — Model training**
```bash
python src/models/train.py
```

**Step 6 — Prediction** (run after training)
```bash
python src/models/predict.py
```

### Skip steps with flags

Set these in `config.yaml` under `flags`:

```yaml
flags:
  skip_video_processing:      true   # Skip Step 1 if processed_video.mp4 exists
  skip_pose_extraction:       true   # Skip Step 2 if processed_data.csv exists
  skip_pose_clustering:       true   # Skip Step 3 entirely (default)
  skip_annotation_processing: true   # Skip Step 4 if labeled_features.csv exists
```

**Model training (Step 5)** is automatically skipped if `model.file` (e.g. `model_xgboost.joblib`) already exists. Delete the file or change the path in `config.yaml` to retrain.

---

## Pipeline Stages

### Step 1 — Video Processing (`src/video_processing/processing.py`)

**What it does:**
- Trims raw video using `ffmpeg` based on `trim_times` in config
- Tracks objects using SAMURAI (YOLO11 + tracker)
- Extracts pose keypoints with MediaPipe
- Saves visualizations (tracking, pose overlay)

**Outputs:**
- `processed_video.mp4` — trimmed video
- `tracked_video.mp4` — object tracking visualization
- `pose_video.mp4` — pose overlay visualization

**Config:**
```yaml
video_processing:
  raw_video_filename: "camera_a.mkv"
  fps: 15
  yolo_model: "yolo11m.pt"
  device: "cuda"
```

---

### Step 2 — Pose Extraction (`src/pose/extractor.py`)

**What it does:**
- Extracts `.tar.gz` archives from psifx output
- Flattens directory structure and renames JSONs: `id_<video_id>_<frame:05d>.json`
- Parses all JSONs into a DataFrame
- Normalizes keypoints relative to torso length
- Computes derived features (joint angles, distances, symmetry)
- Adds velocity and acceleration for key joints
- Saves `processed_data.csv`

**Outputs:**
- `processed_data.csv` — normalized pose features per frame
- `mass_centers_<project>.png` — center-of-mass trajectory visualization

**Config:**
```yaml
pose_extraction:
  directory: "PosesDir"           # Subfolder with .tar.gz files
  fps: 15
  id_to_label:
    "1": "Therapist1"
    "2": "Patient1"
```

---

### Step 3 — Pose Clustering (*optional*) (`src/pose/clustering.py`)

**What it does:**
- Loads `processed_data.csv` from Step 2
- Clinical analysis: movement profiles, proximity, approach-response events, lagged correlation, session phase detection
- Unsupervised clustering: KMeans with optimal-k selection (silhouette + elbow)
- Per-person clustering of pose patterns

**Outputs:**
- `pose_clusters_with_clinical.csv` — features + cluster labels
- `movement_speed.png`, `proximity_distance.png`, `approach_events.png`, `session_phases.png`
- `silhouette_<person>.png`, `elbow_<person>.png`, `pca_<person>.png`, `cluster_distribution_<person>.png`

**Config:**
```yaml
pose_clustering:
  movement_threshold: 10       # px/frame
  proximity_close: 200         # pixels
  k_range: [2, 8]              # KMeans search range
  therapist_label: "Therapist"
  child_label: "Child"
```

**Enable/disable:**
Set `flags.skip_pose_clustering: false` to enable.

---

### Step 4 — Annotation Alignment (`src/annotations/generator.py`)

**What it does:**
- Parses WAKEE `.txt` annotation files (start/end times + labels)
- Infers person from label prefix (`C*` → Child, `T*` → Therapist)
- Adjusts feature frame times by `start_trim_sec` offset
- Maps annotations to feature frames based on time overlap + person match
- Filters unlabeled frames and saves `labeled_features.csv`

**Outputs:**
- `labeled_features.csv` — features with `annotation_label` column
- `annotation_stats_<project>.png` — bar chart of class distribution

**Config:**
```yaml
trim_times:
  "11-1-2024_#9_INDIVIDUAL_[18]": [165, 1930, 15]
```

---

### Step 5 — Model Training (`src/models/train.py`)

**What it does:**
- Combines all `labeled_features.csv` files across projects
- Sanitizes data (drops NaNs, infinite values)
- Feature engineering: normalize → derived → temporal (identical to Step 2)
- Trains three models with hyperparameter tuning:
  - **LightGBM** (early stopping)
  - **XGBoost** (RandomizedSearchCV, 50 iterations)
  - **HistGradientBoosting** (built-in early stopping)
- Evaluates on train/test splits with overfitting detection
- Saves the best model

**Outputs:**
- `model_<name>.joblib` — best trained model
- `feature_names.json` — ordered feature list for inference
- `model_metrics.json` — per-model train/test scores
- `model_comparison.png` — bar chart comparison
- `feature_importance.png` — top-20 features per model
- `model_training_summary.txt` — plain-text summary
- `combined_labeled_features.csv` — merged dataset
- `combined_labeled_features_sanitized.csv` — cleaned dataset
- `global_annotation_distribution.png` — class distribution across all projects

**Config:**
```yaml
model_training:
  min_samples_per_class: 30
  test_size: 0.2
  n_iter_search: 50
  non_posture_classes: ["vocalizations", "unlabeled"]
```

**Auto-skip:**
Training is skipped if `model.file` (e.g. `model_xgboost.joblib`) already exists.

---

### Step 6 — Prediction (`src/models/predict.py`)

**What it does:**
- Loads trained model + feature names
- Runs **identical feature engineering** as training (no feature drift)
- Handles missing features gracefully (fills with 0)
- Predicts annotation labels for new projects
- Optionally generates performance report if true labels are present

**Outputs:**
- `predictions_<stem>.csv` — input + `predicted_annotation_label` column
- `prediction_report_<stem>.txt` — accuracy + classification report (if ground truth available)
- `confusion_matrix_<stem>.png` — confusion matrix heatmap

**Usage:**

**Single-project mode:**
Set `predict.data_path` in config:
```yaml
predict:
  data_path: "/path/to/project/PosesDir/processed_data.csv"
```

**Batch mode** (predict on all projects):
```yaml
predict:
  data_path: null
```

Then run:
```bash
python src/models/predict.py
```

---

## Label Map

| Code | Meaning |
|------|---------|
| C    | Child |
| CCR  | Child–Child Reciprocal |
| CHO  | Child Hand-Over |
| CSI  | Child Side Interaction |
| CST  | Child Stand |
| T    | Therapist |
| TC   | Therapist–Child |
| TRE  | Therapist Reciprocal |
| TSI  | Therapist Side Interaction |
| TST  | Therapist Stand |

---

## Configuration Overview

All pipeline behavior is controlled by `config.yaml`:

### Directories
```yaml
directories:
  raw_video_root:     "/path/to/raw/videos"
  output_base_dir:    "/path/to/outputs"
  annotations_dir:    "/path/to/annotations"
```

### Pipeline Flags
```yaml
flags:
  skip_video_processing:      false
  skip_pose_extraction:       false
  skip_pose_clustering:       true   # Optional step
  skip_annotation_processing: false
```

### Per-Project Trim Times
```yaml
trim_times:
  "11-1-2024_#9_INDIVIDUAL_[18]": [165, 1930, 15]  # [start_sec, end_sec, fps]
```

### Model
```yaml
model:
  file: "model_xgboost.joblib"
  features_file: "feature_names.json"
  fps: 15
```

### Plotting
```yaml
plotting:
  dpi: 300
  figsize_per_project: [18, 10]
  colormap: "viridis"
```

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.8.0 | Deep learning backbone |
| `transformers` | 4.57.0 | Pretrained model hub |
| `scikit-learn` | 1.7.2 | ML utilities and metrics |
| `lightgbm` | 4.5.0 | Gradient boosting (fast) |
| `xgboost` | 2.1.5 | Gradient boosting (structured data) |
| `pandas` | 2.2.3 | Data manipulation |
| `numpy` | 2.0.2 | Numerical computing |
| `opencv-python` | 4.12.0.88 | Video frame processing |
| `mediapipe` | 0.10.14 | Pose keypoint extraction |
| `psifx` | local | Video preprocessing |
| `matplotlib` | 3.8.4 | Plotting and visualization |
| `seaborn` | 0.13.2 | Statistical visualization |
| `joblib` | 1.5.2 | Model serialization |
| `ultralytics` | 8.3.212 | YOLO-based detection |
| `pyyaml` | 6.0.2 | Config file parsing |

---

## Notes

- **All scripts are standalone executable** and follow `if __name__ == "__main__"` conventions
- **Feature engineering is shared** between `train.py` and `predict.py` — no drift risk
- **Config-driven design** means no hardcoded paths in source files
- **Intermediate outputs** are saved at every step for debugging and inspection
- **Time-aware train/test splits** prevent temporal leakage when possible
- **Overfitting detection** built into training with train-test gap analysis
- **Cross-validation** (3-fold StratifiedKFold) validates best model stability

---

## Troubleshooting

**Problem:** `FileNotFoundError: Config file not found`  
**Solution:** Ensure `config.yaml` is at the project root (3 levels up from `src/pipeline/full_pipeline.py`)

**Problem:** Model training fails with `KeyError: 'annotation_label'`  
**Solution:** Run Step 4 (annotation alignment) first to generate `labeled_features.csv` files

**Problem:** Prediction fails with "Missing required features"  
**Solution:** Ensure the input CSV has the same keypoint columns as the training data (run Step 2 on the project first)

**Problem:** `ImportError: No module named 'psifx'`  
**Solution:** Install `psifx` manually before running `pip install -r requirements.txt`

**Problem:** Pipeline hangs during video processing  
**Solution:** Check `video_processing.device` in config — set to `"cpu"` if CUDA is not available


#!/usr/bin/env python3
# coding: utf-8
"""
predict.py
----------
Final step of the Video Annotation Pipeline.
Loads a trained model and runs it on new pose keypoint data
to predict annotation labels.

Feature engineering is identical to training (normalize → derived → temporal).

Outputs saved to output_dir:
  predictions_<stem>.csv          — input data + predicted_annotation_label
  prediction_report_<stem>.txt    — accuracy + classification report (if true labels present)
  confusion_matrix_<stem>.png     — heatmap (if true labels present)

Can be run standalone or called by full_pipeline.py via
  predict_annotations(data_path, output_dir, cfg).

Configuration loaded from config.yaml at the project root.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Re-use feature engineering from train.py (same package directory)
from models.train import (
    normalize_keypoints,
    add_derived_pose_features,
    add_temporal_features,
)


# =============================================================================
# Core prediction function
# =============================================================================

def predict_annotations(data_path, output_dir, cfg=None):
    """
    Run inference on a processed_data.csv (or similar) file.

    Args:
        data_path:  Path to the input CSV (pose features, no annotation required).
        output_dir: Where to save prediction outputs.
        cfg:        Parsed config.yaml dict (optional).

    Returns:
        DataFrame with original columns plus 'predicted_annotation_label'.
    """
    mc         = (cfg or {}).get("model", {})
    output_dir_cfg = (cfg or {}).get("directories", {}).get("output_base_dir", ".")
    fps        = mc.get("fps", 15)
    model_file = mc.get("file", "model_xgboost.joblib")
    feat_file  = mc.get("features_file", "feature_names.json")
    label_map  = {int(k): v for k, v in ((cfg or {}).get("label_map") or {}).items()}
    dpi        = (cfg or {}).get("plotting", {}).get("dpi", 300)

    output_dir     = Path(output_dir)
    model_base_dir = Path(output_dir_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate model ---
    model_path = model_base_dir / model_file
    if not model_path.exists():
        # search for any .joblib in output dir
        candidates = list(model_base_dir.glob("model_*.joblib"))
        if not candidates:
            raise FileNotFoundError(
                f"No model file found in {model_base_dir}. "
                "Run model training (Step 5) first.")
        model_path = sorted(candidates)[-1]
        print(f"  Using model: {model_path.name}")
    model = joblib.load(model_path)
    print(f"  Loaded model: {model_path.name}")

    # --- Locate feature names ---
    feat_path = model_base_dir / feat_file
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found at {feat_path}")
    with open(feat_path) as f:
        feature_names = json.load(f)
    print(f"  Feature names: {len(feature_names)} features")

    # --- Load data ---
    data_path = Path(data_path)
    data = pd.read_csv(data_path)
    print(f"  Input data: {data_path.name} ({len(data):,} rows)")

    # --- Feature engineering ---
    print("\n--- Feature Engineering ---")
    df       = data.copy()
    keypoints = [c for c in df.columns if "_x" in c or "_y" in c]
    print(f"  Found {len(keypoints)} keypoint columns")

    df = normalize_keypoints(df, keypoints)
    df = add_derived_pose_features(df)

    # Use person_label as group_col for inference (annotation_label not available)
    group_col = "person_label" if "person_label" in df.columns else None
    if group_col:
        df, _ = add_temporal_features(df, group_col=group_col, fps=fps)
    else:
        # Create a dummy group so temporal features are still computed
        df["_group"] = "all"
        df, _ = add_temporal_features(df, group_col="_group", fps=fps)
        df = df.drop(columns=["_group"])

    # person_label dummies (must match training encoding)
    if "person_label" in df.columns:
        df = pd.get_dummies(df, columns=["person_label"], prefix="person")

    # --- Align to training feature set ---
    missing = [f for f in feature_names if f not in df.columns]
    extra   = [c for c in df.columns if c not in feature_names]
    if missing:
        print(f"  ⚠ {len(missing)} features missing — filling with 0: {missing[:5]}...")
        for col in missing:
            df[col] = 0.0
    if extra:
        df = df.drop(columns=[c for c in extra if c in df.columns and c not in feature_names])

    X = df[feature_names].copy()

    # --- Impute ---
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        imputer = SimpleImputer(strategy="median")
        X       = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
        print(f"  Imputed {nan_count:,} missing values")

    X = X.replace([np.inf, -np.inf], 0.0)

    # --- Predict ---
    print("\n--- Predicting ---")
    predictions = model.predict(X)
    data = data.iloc[:len(X)].copy().reset_index(drop=True)
    data["predicted_annotation_label"] = predictions

    # Map numeric to string labels if applicable
    if label_map and all(isinstance(p, (int, np.integer)) for p in predictions):
        data["predicted_annotation_label_str"] = (
            data["predicted_annotation_label"].map(label_map)
        )

    stem = data_path.stem
    pred_file = output_dir / f"predictions_{stem}.csv"
    data.to_csv(pred_file, index=False)
    print(f"  Predictions saved: {pred_file.name}")
    print(f"\nPrediction distribution:\n"
          f"{data['predicted_annotation_label'].value_counts().to_string()}")

    # --- Optional performance report ---
    if "annotation_label" in data.columns:
        print("\n--- Performance Report ---")
        y_true  = data["annotation_label"]
        y_pred  = predictions
        labels  = sorted(y_true.unique())
        acc     = accuracy_score(y_true, y_pred)
        report  = classification_report(y_true, y_pred, digits=4)
        cm      = confusion_matrix(y_true, y_pred, labels=labels)

        report_file = output_dir / f"prediction_report_{stem}.txt"
        with open(report_file, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Report saved: {report_file.name}")

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix", fontsize=20, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=16, weight="bold")
        ax.set_ylabel("True Label",      fontsize=16, weight="bold")
        plt.tight_layout()
        cm_file = output_dir / f"confusion_matrix_{stem}.png"
        plt.savefig(cm_file, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Confusion matrix saved: {cm_file.name}")
    else:
        print("\n  No true labels found — skipping performance report.")

    return data


# =============================================================================
# Batch prediction across all projects
# =============================================================================

def run_batch_predictions(cfg: dict) -> dict:
    """
    Predict annotations for every project that has a processed_data.csv
    but no predictions file yet.

    Args:
        cfg: Parsed config.yaml dict.

    Returns:
        dict mapping project_name -> result DataFrame or error string.
    """
    raw_root    = Path(cfg["directories"]["raw_video_root"])
    output_base = Path(cfg["directories"]["output_base_dir"])
    ec          = cfg.get("pose_extraction", {})
    directory   = ec.get("directory", "PosesDir")

    project_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    print(f"Found {len(project_dirs)} project directories for batch prediction.")

    results = {}
    for project_dir in sorted(project_dirs):
        name       = project_dir.name
        data_path  = project_dir / directory / "processed_data.csv"
        output_dir = output_base / name

        if not data_path.exists():
            print(f"  Skipping {name} — no processed_data.csv")
            results[name] = "skipped"
            continue

        pred_file = output_dir / f"predictions_processed_data.csv"
        if pred_file.exists():
            print(f"  Skipping {name} — predictions already exist")
            results[name] = "cached"
            continue

        print(f"\n{'='*70}\nPredicting: {name}\n{'='*70}")
        try:
            df_pred     = predict_annotations(data_path, output_dir, cfg)
            results[name] = df_pred
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = f"error: {e}"

    return results


# =============================================================================
# Standalone entry point
# =============================================================================

def main():
    CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)

    mc          = cfg.get("model", {})
    output_base = Path(cfg["directories"]["output_base_dir"])
    base_data   = Path(cfg["directories"]["base_data_dir"])
    raw_root    = Path(cfg["directories"]["raw_video_root"])
    directory   = cfg.get("pose_extraction", {}).get("directory", "PosesDir")

    # Single-project mode if predict.data_path is set in config
    predict_cfg = cfg.get("predict", {})
    data_path   = predict_cfg.get("data_path")

    if data_path:
        project_name = Path(data_path).parent.parent.name
        output_dir   = output_base / project_name
        result       = predict_annotations(data_path, output_dir, cfg)
        print(f"\n✓ Done. {len(result):,} rows predicted.")
    else:
        # Batch mode
        results = run_batch_predictions(cfg)
        print(f"\n{'='*70}\nBATCH PREDICTION SUMMARY\n{'='*70}")
        for name, status in results.items():
            label = "OK" if isinstance(status, pd.DataFrame) else str(status)
            print(f"  {label:15s}  {name}")


if __name__ == "__main__":
    main()

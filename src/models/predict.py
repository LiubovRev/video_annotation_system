#!/usr/bin/env python3
# coding: utf-8
"""
--------------------------------
Loads a trained XGBoost model and runs it on new pose keypoint data
to predict annotation labels. Optionally generates a performance report
and confusion matrix if true labels are present in the data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =============================================================================
# Configuration — edit these paths before running
# =============================================================================
BASE_DATA_DIR   = Path("/home/liubov/Bureau/new/processed_data")
OUTPUT_BASE_DIR = Path("/home/liubov/Bureau/new/output_data")
MODEL_FILE      = OUTPUT_BASE_DIR / "model_xgboost.joblib"
FEATURES_FILE   = OUTPUT_BASE_DIR / "feature_names.json"

# Path to the new data CSV for prediction
NEW_DATA_FILE   = BASE_DATA_DIR / "processed_data14-3-2024_#15_INDIVIDUAL_[18].csv"

# Label mapping (numeric -> string)
LABEL_MAP = {
    0: 'C', 1: 'CCR', 2: 'CHO', 3: 'CSI', 4: 'CST',
    5: 'T', 6: 'TC',  7: 'TRE', 8: 'TSI', 9: 'TST',
}

FPS = 15


# =============================================================================
# Feature engineering functions
# =============================================================================

def normalize_keypoints(df, keypoints):
    """Normalize keypoints relative to torso length."""
    df['torso_length'] = np.sqrt(
        (df['neck_x'] - df['mid_hip_x'])**2 +
        (df['neck_y'] - df['mid_hip_y'])**2
    )
    df['torso_length'] = df['torso_length'].replace(0, np.nan)
    median_torso = df['torso_length'].median()
    df['torso_length'] = df['torso_length'].fillna(median_torso).clip(lower=1e-3)

    for col in keypoints:
        if '_x' in col:
            df[col] = (df[col] - df['mid_hip_x']) / df['torso_length']
        else:
            df[col] = (df[col] - df['mid_hip_y']) / df['torso_length']

    return df


def add_derived_pose_features(df):
    """Add joint angles, distances, and symmetry features."""

    def compute_angle(a, b, c):
        """Angle at point b formed by points a-b-c (degrees)."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.arccos(np.clip(cosine, -1.0, 1.0)) * (180.0 / np.pi)

    def compute_distance(p1, p2):
        """Euclidean distance between two 2-D points."""
        return np.linalg.norm(p1 - p2)

    # --- Joint angles ---
    angle_defs = {
        'r_elbow_angle':     ('r_shoulder', 'r_elbow',    'r_wrist'),
        'l_elbow_angle':     ('l_shoulder', 'l_elbow',    'l_wrist'),
        'r_shoulder_angle':  ('neck',       'r_shoulder', 'r_elbow'),
        'l_shoulder_angle':  ('neck',       'l_shoulder', 'l_elbow'),
        'r_knee_angle':      ('r_hip',      'r_knee',     'r_ankle'),
        'l_knee_angle':      ('l_hip',      'l_knee',     'l_ankle'),
        'r_hip_angle':       ('mid_hip',    'r_hip',      'r_knee'),
        'l_hip_angle':       ('mid_hip',    'l_hip',      'l_knee'),
        'trunk_angle':       ('nose',       'neck',       'mid_hip'),
    }
    for col, (a, b, c) in angle_defs.items():
        df[col] = df.apply(lambda row: compute_angle(
            np.array([row[f'{a}_x'], row[f'{a}_y']]),
            np.array([row[f'{b}_x'], row[f'{b}_y']]),
            np.array([row[f'{c}_x'], row[f'{c}_y']]),
        ), axis=1)

    # --- Distances ---
    df['eye_to_eye'] = df.apply(lambda row: compute_distance(
        np.array([row['l_eye_x'], row['l_eye_y']]),
        np.array([row['r_eye_x'], row['r_eye_y']])), axis=1)

    df['nose_to_neck'] = df.apply(lambda row: compute_distance(
        np.array([row['nose_x'], row['nose_y']]),
        np.array([row['neck_x'], row['neck_y']])), axis=1)

    df['r_wrist_to_hip'] = df.apply(lambda row: compute_distance(
        np.array([row['r_wrist_x'], row['r_wrist_y']]),
        np.array([row['mid_hip_x'], row['mid_hip_y']])), axis=1)

    df['l_wrist_to_hip'] = df.apply(lambda row: compute_distance(
        np.array([row['l_wrist_x'], row['l_wrist_y']]),
        np.array([row['mid_hip_x'], row['mid_hip_y']])), axis=1)

    df['r_wrist_to_nose'] = df.apply(lambda row: compute_distance(
        np.array([row['r_wrist_x'], row['r_wrist_y']]),
        np.array([row['nose_x'], row['nose_y']])), axis=1)

    df['l_wrist_to_nose'] = df.apply(lambda row: compute_distance(
        np.array([row['l_wrist_x'], row['l_wrist_y']]),
        np.array([row['nose_x'], row['nose_y']])), axis=1)

    df['nose_to_ankles'] = df.apply(lambda row: (
        compute_distance(np.array([row['nose_x'], row['nose_y']]),
                         np.array([row['l_ankle_x'], row['l_ankle_y']])) +
        compute_distance(np.array([row['nose_x'], row['nose_y']]),
                         np.array([row['r_ankle_x'], row['r_ankle_y']]))
    ) / 2, axis=1)

    df['hip_to_ankle'] = df.apply(lambda row: (
        compute_distance(np.array([row['mid_hip_x'], row['mid_hip_y']]),
                         np.array([row['l_ankle_x'], row['l_ankle_y']])) +
        compute_distance(np.array([row['mid_hip_x'], row['mid_hip_y']]),
                         np.array([row['r_ankle_x'], row['r_ankle_y']]))
    ) / 2, axis=1)

    # --- Symmetry features ---
    df['shoulder_y_diff']     = df['l_shoulder_y']    - df['r_shoulder_y']
    df['hip_y_diff']          = df['l_hip_y']          - df['r_hip_y']
    df['elbow_angle_diff']    = df['l_elbow_angle']    - df['r_elbow_angle']
    df['knee_angle_diff']     = df['l_knee_angle']     - df['r_knee_angle']
    df['wrist_to_hip_diff']   = df['l_wrist_to_hip']   - df['r_wrist_to_hip']
    df['shoulder_angle_diff'] = df['l_shoulder_angle'] - df['r_shoulder_angle']

    df['com_x'] = (df['mid_hip_x'] + df['neck_x']) / 2
    df['com_y'] = (df['mid_hip_y'] + df['neck_y']) / 2

    df['body_spread_x'] = (df[['l_wrist_x', 'r_wrist_x', 'l_ankle_x', 'r_ankle_x']].max(axis=1) -
                           df[['l_wrist_x', 'r_wrist_x', 'l_ankle_x', 'r_ankle_x']].min(axis=1))
    df['body_spread_y'] = (df[['nose_y', 'l_ankle_y', 'r_ankle_y']].max(axis=1) -
                           df[['nose_y', 'l_ankle_y', 'r_ankle_y']].min(axis=1))

    return df


def add_temporal_features(df, keypoints, fps=15):
    """
    Compute velocity and acceleration for selected keypoints.

    Velocity     = delta_position / delta_time  (units/second)
    Acceleration = delta_velocity  / delta_time  (units/second^2)

    NaNs are kept intact for proper downstream imputation.
    """
    df = df.sort_values(by=['person_label', 'time_s']).reset_index(drop=True)

    df['delta_time'] = df.groupby('person_label')['time_s'].diff()
    df['delta_time'] = df['delta_time'].fillna(1 / fps).clip(lower=0.001)

    selected_keypoints = [
        'com_x', 'com_y',
        'nose_x', 'nose_y',
        'l_wrist_x', 'l_wrist_y', 'r_wrist_x', 'r_wrist_y',
        'l_ankle_x', 'l_ankle_y', 'r_ankle_x', 'r_ankle_y',
        'neck_x', 'neck_y',
        'mid_hip_x', 'mid_hip_y',
    ]

    temporal_cols = []
    for col in selected_keypoints:
        if col not in df.columns:
            continue

        vel_col = f'{col}_vel'
        df[vel_col] = df.groupby('person_label')[col].diff() / df['delta_time']

        acc_col = f'{col}_acc'
        df[acc_col] = df.groupby('person_label')[vel_col].diff() / df['delta_time']

        temporal_cols.extend([vel_col, acc_col])

    for col in temporal_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    n = len(temporal_cols) // 2
    print(f"  Added {len(temporal_cols)} temporal features ({n} velocity + {n} acceleration)")
    return df, temporal_cols


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)

    # --- Load model ---
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print(f"Loaded model from {MODEL_FILE.name}")

    # --- Load feature names ---
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_FILE}")
    with open(FEATURES_FILE, 'r') as f:
        feature_names = json.load(f)
    print(f"Loaded {len(feature_names)} feature names")

    # --- Load new data ---
    if not NEW_DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {NEW_DATA_FILE}")
    data = pd.read_csv(NEW_DATA_FILE)
    print(f"Loaded data: {NEW_DATA_FILE.name} ({len(data)} rows)")

    # --- Feature engineering ---
    print("\n--- Feature Engineering ---")
    keypoints = [col for col in data.columns if '_x' in col or '_y' in col]
    print(f"  Found {len(keypoints)} keypoint columns")

    df = normalize_keypoints(data.copy(), keypoints)
    df = add_derived_pose_features(df)
    df, _ = add_temporal_features(df, keypoints, fps=FPS)
    df['person_Child'] = (df['person_label'] == 'Child').astype(int)

    # --- Validate features ---
    missing_cols = [f for f in feature_names if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
    X = df[feature_names]

    # --- Predict ---
    print("\n--- Predicting ---")
    predictions = model.predict(X)
    data['predicted_annotation_label'] = predictions

    # Map numeric predictions to string labels if applicable
    if all(isinstance(p, (int, np.integer)) for p in predictions):
        data['predicted_annotation_label_str'] = data['predicted_annotation_label'].map(LABEL_MAP)

    predictions_file = OUTPUT_BASE_DIR / f"predictions_{NEW_DATA_FILE.stem}.csv"
    data.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")
    print(f"\nPrediction counts:\n{data['predicted_annotation_label'].value_counts().to_string()}")

    # --- Optional performance report (requires true labels in CSV) ---
    if 'annotation_label' in data.columns:
        y_true = data['annotation_label']
        y_pred = predictions
        labels  = np.unique(y_true)

        accuracy = accuracy_score(y_true, y_pred)
        report   = classification_report(y_true, y_pred, digits=4)
        cm       = confusion_matrix(y_true, y_pred)

        report_file = OUTPUT_BASE_DIR / f"prediction_report_{NEW_DATA_FILE.stem}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Performance report saved to: {report_file}")

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix", fontsize=20, weight='bold')
        plt.xlabel("Predicted Label", fontsize=16, weight='bold')
        plt.ylabel("True Label", fontsize=16, weight='bold')
        plt.tight_layout()
        cm_file = OUTPUT_BASE_DIR / f"confusion_matrix_{NEW_DATA_FILE.stem}.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Confusion matrix saved to: {cm_file}")

    else:
        print("No true labels found in CSV - skipping performance report.")


if __name__ == "__main__":
    main()

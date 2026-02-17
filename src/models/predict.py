import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# User options
# =========================
BASE_DATA_DIR = Path("/home/liubov/Bureau/new/processed_data")
OUTPUT_BASE_DIR = Path("/home/liubov/Bureau/new/output_data")
MODEL_FILE = OUTPUT_BASE_DIR / "/home/liubov/Bureau/new/output_data/model_xgboost.joblib" 
FEATURES_FILE = OUTPUT_BASE_DIR / "feature_names.json"

OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)

# Path to new data CSV for prediction
NEW_DATA_FILE = BASE_DATA_DIR / "processed_data14-3-2024_#15_INDIVIDUAL_[18].csv"


# =========================
# Load model and features
# =========================
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)
print(f"✓ Loaded model from {MODEL_FILE.name}")

if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURES_FILE}")

with open(FEATURES_FILE, 'r') as f:
    feature_names = json.load(f)
print(f"✓ Loaded {len(feature_names)} feature names")

# =========================
# Load new data
# =========================
if not NEW_DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {NEW_DATA_FILE}")

data = pd.read_csv(NEW_DATA_FILE)
print(f"✓ Loaded data: {NEW_DATA_FILE.name} ({len(data)} rows)")

def add_derived_pose_features(df):
    """
    Add comprehensive joint angles, distances, and symmetry features.
    """
    
    def compute_angle(a, b, c):
        """Compute angle at point b formed by points a-b-c"""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * (180.0 / np.pi)

    def compute_distance(p1, p2):
        """Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)

    # === JOINT ANGLES ===
    df['r_elbow_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['r_shoulder_x'], row['r_shoulder_y']]),
        np.array([row['r_elbow_x'], row['r_elbow_y']]),
        np.array([row['r_wrist_x'], row['r_wrist_y']])
    ), axis=1)

    df['l_elbow_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['l_shoulder_x'], row['l_shoulder_y']]),
        np.array([row['l_elbow_x'], row['l_elbow_y']]),
        np.array([row['l_wrist_x'], row['l_wrist_y']])
    ), axis=1)

    df['r_shoulder_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['neck_x'], row['neck_y']]),
        np.array([row['r_shoulder_x'], row['r_shoulder_y']]),
        np.array([row['r_elbow_x'], row['r_elbow_y']])
    ), axis=1)

    df['l_shoulder_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['neck_x'], row['neck_y']]),
        np.array([row['l_shoulder_x'], row['l_shoulder_y']]),
        np.array([row['l_elbow_x'], row['l_elbow_y']])
    ), axis=1)

    df['r_knee_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['r_hip_x'], row['r_hip_y']]),
        np.array([row['r_knee_x'], row['r_knee_y']]),
        np.array([row['r_ankle_x'], row['r_ankle_y']])
    ), axis=1)

    df['l_knee_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['l_hip_x'], row['l_hip_y']]),
        np.array([row['l_knee_x'], row['l_knee_y']]),
        np.array([row['l_ankle_x'], row['l_ankle_y']])
    ), axis=1)

    df['r_hip_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['mid_hip_x'], row['mid_hip_y']]),
        np.array([row['r_hip_x'], row['r_hip_y']]),
        np.array([row['r_knee_x'], row['r_knee_y']])
    ), axis=1)

    df['l_hip_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['mid_hip_x'], row['mid_hip_y']]),
        np.array([row['l_hip_x'], row['l_hip_y']]),
        np.array([row['l_knee_x'], row['l_knee_y']])
    ), axis=1)

    df['trunk_angle'] = df.apply(lambda row: compute_angle(
        np.array([row['nose_x'], row['nose_y']]),
        np.array([row['neck_x'], row['neck_y']]),
        np.array([row['mid_hip_x'], row['mid_hip_y']])
    ), axis=1)

    # === DISTANCES ===
    df['eye_to_eye'] = df.apply(lambda row: compute_distance(
        np.array([row['l_eye_x'], row['l_eye_y']]),
        np.array([row['r_eye_x'], row['r_eye_y']])
    ), axis=1)

    df['nose_to_neck'] = df.apply(lambda row: compute_distance(
        np.array([row['nose_x'], row['nose_y']]),
        np.array([row['neck_x'], row['neck_y']])
    ), axis=1)

    df['r_wrist_to_hip'] = df.apply(lambda row: compute_distance(
        np.array([row['r_wrist_x'], row['r_wrist_y']]),
        np.array([row['mid_hip_x'], row['mid_hip_y']])
    ), axis=1)

    df['l_wrist_to_hip'] = df.apply(lambda row: compute_distance(
        np.array([row['l_wrist_x'], row['l_wrist_y']]),
        np.array([row['mid_hip_x'], row['mid_hip_y']])
    ), axis=1)

    df['r_wrist_to_nose'] = df.apply(lambda row: compute_distance(
        np.array([row['r_wrist_x'], row['r_wrist_y']]),
        np.array([row['nose_x'], row['nose_y']])
    ), axis=1)

    df['l_wrist_to_nose'] = df.apply(lambda row: compute_distance(
        np.array([row['l_wrist_x'], row['l_wrist_y']]),
        np.array([row['nose_x'], row['nose_y']])
    ), axis=1)

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

    # === SYMMETRY FEATURES ===
    df['shoulder_y_diff'] = df['l_shoulder_y'] - df['r_shoulder_y']
    df['hip_y_diff'] = df['l_hip_y'] - df['r_hip_y']
    df['elbow_angle_diff'] = df['l_elbow_angle'] - df['r_elbow_angle']
    df['knee_angle_diff'] = df['l_knee_angle'] - df['r_knee_angle']
    df['wrist_to_hip_diff'] = df['l_wrist_to_hip'] - df['r_wrist_to_hip']
    df['shoulder_angle_diff'] = df['l_shoulder_angle'] - df['r_shoulder_angle']

    df['com_x'] = (df['mid_hip_x'] + df['neck_x']) / 2
    df['com_y'] = (df['mid_hip_y'] + df['neck_y']) / 2
    
    df['body_spread_x'] = df[['l_wrist_x', 'r_wrist_x', 'l_ankle_x', 'r_ankle_x']].max(axis=1) - \
                          df[['l_wrist_x', 'r_wrist_x', 'l_ankle_x', 'r_ankle_x']].min(axis=1)
    df['body_spread_y'] = df[['nose_y', 'l_ankle_y', 'r_ankle_y']].max(axis=1) - \
                          df[['nose_y', 'l_ankle_y', 'r_ankle_y']].min(axis=1)

    return df


def add_temporal_features_fixed(df, keypoints, fps=15):
    """
    FIXED: Proper velocity and acceleration calculations
    - Velocity = change in position / change in time (units/second)
    - Acceleration = change in velocity / change in time (units/second²)
    - No zero-filling for NaNs, they will be handled later with proper imputation
    """
    df = df.sort_values(by=['person_label', 'time_s']).reset_index(drop=True)
    
    # Calculate time differences properly
    df['delta_time'] = df.groupby('person_label')['time_s'].diff()
    
    # For first frame in each sequence, use median frame time
    median_frame_time = 1 / fps
    df['delta_time'] = df['delta_time'].fillna(median_frame_time)
    
    # Clip extremely small values to avoid division issues
    df['delta_time'] = df['delta_time'].clip(lower=0.001)
    
    # Select keypoints for velocity/acceleration
    selected_keypoints = [
        'com_x', 'com_y', 'nose_x', 'nose_y',
        'l_wrist_x', 'l_wrist_y', 'r_wrist_x', 'r_wrist_y',
        'l_ankle_x', 'l_ankle_y', 'r_ankle_x', 'r_ankle_y',
        'neck_x', 'neck_y', 'mid_hip_x', 'mid_hip_y'
    ]
    
    velocity_features = []
    acceleration_features = []
    
    for col in selected_keypoints:
        if col not in df.columns:
            continue
        
        # VELOCITY: (position_t - position_t-1) / delta_time
        position_diff = df.groupby('person_label')[col].diff()
        velocity = position_diff / df['delta_time']
        vel_col = f'{col}_vel'
        df[vel_col] = velocity
        velocity_features.append(vel_col)
        
        # ACCELERATION: (velocity_t - velocity_t-1) / delta_time
        velocity_diff = df.groupby('person_label')[vel_col].diff()
        acceleration = velocity_diff / df['delta_time']
        acc_col = f'{col}_acc'
        df[acc_col] = acceleration
        acceleration_features.append(acc_col)
    
    # Remove infinite values (but keep NaN for proper imputation later)
    for col in velocity_features + acceleration_features:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    print(f"✓ Added {len(velocity_features)} velocity and {len(acceleration_features)} acceleration features")
    
    return df, velocity_features + acceleration_features


def normalize_keypoints(df, keypoints):
    """Normalize keypoints relative to torso length"""
    df['torso_length'] = np.sqrt(
        (df['neck_x'] - df['mid_hip_x'])**2 + 
        (df['neck_y'] - df['mid_hip_y'])**2
    )
    
    # Handle zero/invalid torso lengths
    df['torso_length'] = df['torso_length'].replace(0, np.nan)
    median_torso = df['torso_length'].median()
    df['torso_length'] = df['torso_length'].fillna(median_torso)
    df['torso_length'] = df['torso_length'].clip(lower=1e-3)

    for col in keypoints:
        if '_x' in col:
            df[col] = (df[col] - df['mid_hip_x']) / df['torso_length']
        else:
            df[col] = (df[col] - df['mid_hip_y']) / df['torso_length']

    return df

#  Feature engineering
print("\n--- Feature Engineering ---")
# Identify keypoints
keypoints = [col for col in data.columns if '_x' in col or '_y' in col]
print(f"✓ Found {len(keypoints)} keypoint columns")
    
df = normalize_keypoints(data, keypoints)
df = add_derived_pose_features(df)
df, temporal_features = add_temporal_features_fixed(df, keypoints)
df['person_Child'] = (df['person_label'] == 'Child').astype(int)

# Select only the features used in the model
missing_cols = [f for f in feature_names if f not in df.columns]
if missing_cols:
    raise ValueError(f"The following required features are missing in the data: {missing_cols}")

X = df[feature_names]


# =========================
# Predict annotation labels
# =========================
predictions = model.predict(X)
data['predicted_annotation_label'] = predictions

# Save predictions
predictions_file = OUTPUT_BASE_DIR / f"predictions_{NEW_DATA_FILE.stem}.csv"
data.to_csv(predictions_file, index=False)
print(f"✓ Predictions saved to: {predictions_file}")

# =========================
# Performance report (if true labels exist)
# =========================
if 'annotation_label' in data.columns:
    y_true = data['annotation_label']
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Save textual report
    report_file = OUTPUT_BASE_DIR / f"prediction_report_{NEW_DATA_FILE.stem}.txt"
    with open(report_file, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"✓ Performance report saved to: {report_file}")

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.title("Confusion Matrix", fontsize=20, weight='bold')
    plt.xlabel("Predicted Label", fontsize=16, weight='bold')
    plt.ylabel("True Label", fontsize=16, weight='bold')
    plt.tight_layout()
    cm_file = OUTPUT_BASE_DIR / f"confusion_matrix_{NEW_DATA_FILE.stem}.png"
    plt.savefig(cm_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_file}")

else:
    print(" No true labels found in CSV; skipping performance report.")

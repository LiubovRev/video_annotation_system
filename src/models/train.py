#!/usr/bin/env python3
# coding: utf-8
"""
train.py
--------
Step 5 of the Video Annotation Pipeline:
Feature engineering → model training → evaluation → artifact saving.

Trains three gradient-boosted models for pose-based classification:
  - LightGBM (early stopping)
  - XGBoost (RandomizedSearchCV)
  - HistGradientBoosting (early stopping)

Saves outputs to `output_dir`:
  - model_<name>.joblib
  - feature_names.json
  - model_metrics.json
  - model_comparison.png
  - feature_importance.png
  - model_training_summary.txt

Can run standalone or via `full_pipeline.py`.
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# -----------------------------
# Setup
# -----------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LABEL_TO_CLASS = {
    "CS": "vocalizations", "CNS": "vocalizations",
    "TS": "vocalizations", "TNS": "vocalizations",
    "CGO": "unlabeled", "TGO": "unlabeled",
    "TC": "joint_attention", "AO": "joint_attention",
    "AT": "joint_attention", "GO": "joint_attention", "GT": "joint_attention",
    "T": "interactions", "T_V": "interactions",
    "T_P": "interactions", "C": "interactions",
    "TST": "coordination", "THO": "coordination", "TSI": "coordination",
    "TLF": "coordination", "TRE": "coordination", "TCR": "coordination",
    "CST": "coordination", "CHO": "coordination", "CSI": "coordination",
    "CLF": "coordination", "CRE": "coordination", "CCR": "coordination",
}
NON_POSTURE_CLASSES = {"vocalizations", "unlabeled"}

# -----------------------------
# Feature Engineering
# -----------------------------
def normalize_keypoints(df, keypoints):
    """Normalize keypoints relative to torso length."""
    df["torso_length"] = np.sqrt((df["neck_x"] - df["mid_hip_x"])**2 + (df["neck_y"] - df["mid_hip_y"])**2)
    df["torso_length"] = df["torso_length"].replace(0, np.nan).fillna(df["torso_length"].median()).clip(lower=1e-3)
    for col in keypoints:
        if "_x" in col:
            df[col] = (df[col] - df["mid_hip_x"]) / df["torso_length"]
        else:
            df[col] = (df[col] - df["mid_hip_y"]) / df["torso_length"]
    return df

def add_derived_pose_features(df):
    """Add joint angles, distances, and symmetry features."""
    def angle(a, b, c):
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.arccos(np.clip(cos, -1, 1)) * 180.0 / np.pi
    def dist(p1, p2): return np.linalg.norm(p1 - p2)

    angles = {
        "r_elbow_angle": ("r_shoulder","r_elbow","r_wrist"),
        "l_elbow_angle": ("l_shoulder","l_elbow","l_wrist"),
        "r_shoulder_angle": ("neck","r_shoulder","r_elbow"),
        "l_shoulder_angle": ("neck","l_shoulder","l_elbow"),
        "r_knee_angle": ("r_hip","r_knee","r_ankle"),
        "l_knee_angle": ("l_hip","l_knee","l_ankle"),
        "r_hip_angle": ("mid_hip","r_hip","r_knee"),
        "l_hip_angle": ("mid_hip","l_hip","l_knee"),
        "trunk_angle": ("nose","neck","mid_hip")
    }

    for col, (a, b, c) in angles.items():
        df[col] = df.apply(lambda r: angle(
            np.array([r[f"{a}_x"], r[f"{a}_y"]]),
            np.array([r[f"{b}_x"], r[f"{b}_y"]]),
            np.array([r[f"{c}_x"], r[f"{c}_y"]])
        ), axis=1)

    # Distances
    df["eye_to_eye"] = df.apply(lambda r: dist([r["l_eye_x"], r["l_eye_y"]],[r["r_eye_x"], r["r_eye_y"]]), axis=1)
    df["nose_to_neck"] = df.apply(lambda r: dist([r["nose_x"], r["nose_y"]],[r["neck_x"], r["neck_y"]]), axis=1)
    df["r_wrist_to_hip"] = df.apply(lambda r: dist([r["r_wrist_x"], r["r_wrist_y"]],[r["mid_hip_x"], r["mid_hip_y"]]), axis=1)
    df["l_wrist_to_hip"] = df.apply(lambda r: dist([r["l_wrist_x"], r["l_wrist_y"]],[r["mid_hip_x"], r["mid_hip_y"]]), axis=1)
    df["r_wrist_to_nose"] = df.apply(lambda r: dist([r["r_wrist_x"], r["r_wrist_y"]],[r["nose_x"], r["nose_y"]]), axis=1)
    df["l_wrist_to_nose"] = df.apply(lambda r: dist([r["l_wrist_x"], r["l_wrist_y"]],[r["nose_x"], r["nose_y"]]), axis=1)
    df["nose_to_ankles"] = df.apply(lambda r: (dist([r["nose_x"],r["nose_y"]],[r["l_ankle_x"],r["l_ankle_y"]])+
                                              dist([r["nose_x"],r["nose_y"]],[r["r_ankle_x"],r["r_ankle_y"]]))/2, axis=1)
    df["hip_to_ankle"] = df.apply(lambda r: (dist([r["mid_hip_x"],r["mid_hip_y"]],[r["l_ankle_x"],r["l_ankle_y"]])+
                                            dist([r["mid_hip_x"],r["mid_hip_y"]],[r["r_ankle_x"],r["r_ankle_y"]]))/2, axis=1)

    # Symmetry & spread
    df["shoulder_y_diff"] = df["l_shoulder_y"] - df["r_shoulder_y"]
    df["hip_y_diff"] = df["l_hip_y"] - df["r_hip_y"]
    df["elbow_angle_diff"] = df["l_elbow_angle"] - df["r_elbow_angle"]
    df["knee_angle_diff"] = df["l_knee_angle"] - df["r_knee_angle"]
    df["wrist_to_hip_diff"] = df["l_wrist_to_hip"] - df["r_wrist_to_hip"]
    df["shoulder_angle_diff"] = df["l_shoulder_angle"] - df["r_shoulder_angle"]
    df["com_x"] = (df["mid_hip_x"] + df["neck_x"]) / 2
    df["com_y"] = (df["mid_hip_y"] + df["neck_y"]) / 2
    df["body_spread_x"] = df[["l_wrist_x","r_wrist_x","l_ankle_x","r_ankle_x"]].max(axis=1) - df[["l_wrist_x","r_wrist_x","l_ankle_x","r_ankle_x"]].min(axis=1)
    df["body_spread_y"] = df[["nose_y","l_ankle_y","r_ankle_y"]].max(axis=1) - df[["nose_y","l_ankle_y","r_ankle_y"]].min(axis=1)

    return df

def add_temporal_features(df, group_col="annotation_label", fps=15):
    """Compute velocities and accelerations of keypoints."""
    df = df.sort_values([group_col, "time_s"]).reset_index(drop=True)
    df["delta_time"] = df.groupby(group_col)["time_s"].diff().fillna(1/fps).clip(lower=1e-3)

    selected = ["com_x","com_y","nose_x","nose_y",
                "l_wrist_x","l_wrist_y","r_wrist_x","r_wrist_y",
                "l_ankle_x","l_ankle_y","r_ankle_x","r_ankle_y",
                "neck_x","neck_y","mid_hip_x","mid_hip_y"]
    temporal_cols = []
    for col in selected:
        if col not in df: continue
        vel, acc = f"{col}_vel", f"{col}_acc"
        df[vel] = df.groupby(group_col)[col].diff() / df["delta_time"]
        df[acc] = df.groupby(group_col)[vel].diff() / df["delta_time"]
        temporal_cols.extend([vel, acc])

    df[temporal_cols] = df[temporal_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(f"Added {len(temporal_cols)//2} velocity + {len(temporal_cols)//2} acceleration features")
    return df, temporal_cols

# =============================================================================
# Data Checks & Overfitting Detection
# =============================================================================
def check_data_leakage(df, label_col="annotation_label"):
    print(f"\n{'='*70}\nDATA LEAKAGE CHECK\n{'='*70}")
    if "time_s" in df.columns:
        classes = df[label_col].value_counts()
        significant = classes[classes >= 100].index.tolist()
        non_overlap = []

        for i, c1 in enumerate(significant[:10]):
            for c2 in significant[i+1:10]:
                r1 = df[df[label_col]==c1]["time_s"].agg(["min","max"])
                r2 = df[df[label_col]==c2]["time_s"].agg(["min","max"])
                if r1["max"] < r2["min"] or r2["max"] < r1["min"]:
                    non_overlap.append((c1,c2))
        if non_overlap:
            print(f"  ⚠ {len(non_overlap)} class pairs do not overlap in time")
        else:
            print("  ✓ Significant classes overlap in time")

    dups = df.duplicated().sum()
    print(f"  {'⚠ Duplicates: '+str(dups) if dups else '✓ No duplicate rows'}")
    print("="*70)

def detect_overfitting(train_f1, test_f1, model_name, threshold=0.10):
    gap = train_f1 - test_f1
    status = "good"
    print(f"\n  {model_name} — Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Gap: {gap:.4f}")
    if gap > 0.15:
        print("    ⚠ SEVERE OVERFITTING")
        status = "severe"
    elif gap > threshold:
        print("    ⚠ Moderate overfitting")
        status = "moderate"
    else:
        print("    ✓ No significant overfitting")
    return status

# =============================================================================
# Visualization
# =============================================================================
def plot_model_comparison(test_metrics, train_metrics, output_dir, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pd.DataFrame(test_metrics).T.plot(kind="bar", ax=axes[0], edgecolor="black", alpha=0.85)
    axes[0].set_title("Test Set Performance", fontsize=14, weight="bold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    gap = pd.DataFrame(train_metrics).T["f1"] - pd.DataFrame(test_metrics).T["f1"]
    gap.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
    axes[1].axhline(0.05, color="orange", linestyle="--", label="Acceptable (0.05)")
    axes[1].axhline(0.10, color="red", linestyle="--", label="Warning (0.10)")
    axes[1].set_title("Train-Test F1 Gap", fontsize=14, weight="bold")
    axes[1].set_ylabel("F1 Gap")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved model_comparison.png")

def plot_feature_importance(models, feature_names, output_dir, dpi=300):
    capable = {n:m for n,m in models.items() if hasattr(m,"feature_importances_")}
    if not capable:
        return

    n = len(capable)
    fig, axes = plt.subplots(1, n, figsize=(6*n,5))
    if n==1: axes=[axes]

    for ax,(name,model) in zip(axes, capable.items()):
        imp = model.feature_importances_
        idx = np.argsort(imp)[-20:]
        ax.barh(range(len(idx)), imp[idx], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        ax.set_title(f"{name} — Top 20", fontsize=12, weight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved feature_importance.png")

# =============================================================================
# Core Training Function
# =============================================================================
def train_model(labeled_df, output_dir, cfg=None):
    """
    Full pipeline: feature-engineering → training → evaluation → artifact saving
    """
    cfg = cfg or {}
    tc = cfg.get("model_training", {})
    min_samples = tc.get("min_samples_per_class", 30)
    test_size = tc.get("test_size", 0.2)
    n_iter_search = tc.get("n_iter_search", 50)
    dpi = cfg.get("plotting", {}).get("dpi", 300)
    fps = cfg.get("model", {}).get("fps", 15)
    label_to_class = tc.get("label_to_class", DEFAULT_LABEL_TO_CLASS)
    non_posture = set(tc.get("non_posture_classes", list(NON_POSTURE_CLASSES)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Data Preparation
    # -----------------------------
    df = labeled_df.copy()
    print(f"\nLoaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    df["label_class"] = df["annotation_label"].map(label_to_class)
    df = df[~df["label_class"].isin(non_posture)].copy()

    counts = df["annotation_label"].value_counts()
    valid_cls = counts[counts >= min_samples].index.tolist()
    df = df[df["annotation_label"].isin(valid_cls)].copy()

    print(f"Filtered dataset: {len(df):,} rows, {len(valid_cls)} valid classes")
    check_data_leakage(df)

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    keypoints = [c for c in df.columns if "_x" in c or "_y" in c]
    df = normalize_keypoints(df, keypoints)
    df = add_derived_pose_features(df)
    df, _ = add_temporal_features(df, group_col="annotation_label", fps=fps)

    # Encode labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["annotation_label"])
    y = df["label_encoded"]

    # Build feature matrix
    if "person_label" in df:  # optional one-hot encoding
        df = pd.get_dummies(df, columns=["person_label"], prefix="person")

    meta_cols = ["time_s","avg_pose_conf","annotation_label","label_class",
                 "frame","delta_time","torso_length","label_encoded",
                 "wrist_dist","time_min:s.ms","project_name"]
    X = df.drop(columns=[c for c in meta_cols if c in df]).select_dtypes(np.number)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Feature selection: zero-variance + low correlation + redundancy
    zero_var = X.var()[X.var()<1e-8].index.tolist()
    X = X.drop(columns=zero_var)
    corrs = pd.concat([X,y], axis=1).corr()["target"].drop("target") if "target" in X else None
    low_corr = corrs[corrs<0.01].index.tolist() if corrs is not None else []
    X = X.drop(columns=low_corr)
    corr_mat = X.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    redundant = [col for col in upper.columns if any(upper[col]>0.95)]
    X = X.drop(columns=redundant)
    print(f"Final features: {X.shape[1]} (removed zero-var {len(zero_var)}, low-corr {len(low_corr)}, redundant {len(redundant)})")

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    if "time_s" in labeled_df.columns and len(labeled_df)==len(X):
        t = labeled_df["time_s"].values[:len(X)]
        threshold = np.percentile(t, 80)
        tr_mask = t <= threshold
        te_mask = t > threshold
        if set(y[tr_mask].unique()) == set(y[te_mask].unique()):
            X_train, X_test = X[tr_mask], X[te_mask]
            y_train, y_test = y[tr_mask], y[te_mask]
            print(f"Time-aware split: {len(X_train)}/{len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # -----------------------------
    # Model Training
    # -----------------------------
    models = {}

    # 1. LightGBM
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=1.0, min_gain_to_split=0.01,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    lgb_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(20, verbose=False)])
    models["LightGBM"] = lgb_model

    # 2. XGBoost
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        param_distributions={
            "n_estimators":[100,150,200], "max_depth":[3,4,5,6],
            "learning_rate":[0.03,0.05,0.07,0.1], "subsample":[0.6,0.7,0.8],
            "colsample_bytree":[0.6,0.7,0.8], "min_child_weight":[3,5,7],
            "gamma":[0.1,0.2,0.3], "reg_alpha":[0.5,1.0,1.5], "reg_lambda":[0.5,1.0,1.5]
        },
        n_iter=n_iter_search, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="f1_weighted", n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    models["XGBoost"] = xgb_search.best_estimator_

    # 3. HistGradientBoosting
    hist = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=30,
        max_bins=255, max_leaf_nodes=31,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
        random_state=42
    )
    hist.fit(X_train, y_train)
    models["HistGradientBoosting"] = hist

    # -----------------------------
    # Evaluation
    # -----------------------------
    train_metrics, test_metrics, overfit = {}, {}, {}
    for name, model in models.items():
        tr_pred, te_pred = model.predict(X_train), model.predict(X_test)
        tr_prob, te_prob = model.predict_proba(X_train), model.predict_proba(X_test)
        train_metrics[name] = {
            "f1": f1_score(y_train, tr_pred, average="weighted"),
            "precision": precision_score(y_train, tr_pred, average="weighted"),
            "recall": recall_score(y_train, tr_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_train, tr_prob, multi_class="ovr", average="macro")
        }
        test_metrics[name] = {
            "f1": f1_score(y_test, te_pred, average="weighted"),
            "precision": precision_score(y_test, te_pred, average="weighted"),
            "recall": recall_score(y_test, te_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, te_prob, multi_class="ovr", average="macro")
        }
        overfit[name] = detect_overfitting(train_metrics[name]["f1"], test_metrics[name]["f1"], name)

    plot_model_comparison(test_metrics, train_metrics, output_dir, dpi)
    plot_feature_importance(models, list(X.columns), output_dir, dpi)

    # -----------------------------
    # Best Model Selection & Artifact Saving
    # -----------------------------
    best_name = max(test_metrics, key=lambda k: test_metrics[k]["f1"])
    best_model = models[best_name]
    best_f1 = test_metrics[best_name]["f1"]
    print(f"\nBest Model: {best_name} — F1={best_f1:.4f} Overfitting={overfit[best_name]}")

    # Save artifacts
    joblib.dump(best_model, output_dir / f"model_{best_name.lower().replace(' ','_')}.joblib")
    with open(output_dir / "feature_names.json", "w") as f: json.dump(list(X.columns), f, indent=2)
    with open(output_dir / "model_metrics.json", "w") as f:
        json.dump({"best_model": best_name,
                   "train_metrics": train_metrics,
                   "test_metrics": test_metrics,
                   "overfitting_status": overfit,
                   "n_features": X.shape[1],
                   "n_classes": len(le.classes_)}, f, indent=2)
    print(f"\n✓ Saved models and artifacts to {output_dir}")

    return {
        "models": models,
        "best_model": best_model,
        "best_model_name": best_name,
        "label_encoder": le,
        "imputer": imputer,
        "feature_names": list(X.columns),
        "f1_score": best_f1,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "overfitting_status": overfit
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train pose classification model pipeline")
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to labeled CSV dataset with keypoints and annotation labels"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory to save trained models, metrics, and plots"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML config file for training and feature engineering"
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    # -----------------------------
    # Load Configuration
    # -----------------------------
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"\nLoaded config from {config_path}")
    else:
        print(f"\n⚠ Config file {config_path} not found. Using defaults.")
        cfg = {}

    # -----------------------------
    # Load Dataset
    # -----------------------------
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file {input_csv} not found!")

    df = pd.read_csv(input_csv)
    print(f"\nLoaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")

    # -----------------------------
    # Run Training Pipeline
    # -----------------------------
    results = train_model(df, output_dir, cfg)

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n" + "="*70)
    print(f"✅ Training pipeline complete!")
    print(f"Best model: {results['best_model_name']} — F1 Score: {results['f1_score']:.4f}")
    print(f"Artifacts saved to: {output_dir}")
    print("="*70 + "\n")

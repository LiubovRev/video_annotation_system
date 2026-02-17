#!/usr/bin/env python3
# coding: utf-8
"""
train.py
--------
Step 5 of the Video Annotation Pipeline.
Feature engineering, model selection, training, evaluation, and artifact saving.

Trains three gradient-boosted models suited to high-dimensional pose data:
  - LightGBM           (early stopping on validation split)
  - XGBoost            (RandomizedSearchCV, 50 iterations)
  - HistGradientBoosting (built-in early stopping)

Outputs saved to output_dir:
  model_<name>.joblib         — best model
  feature_names.json          — ordered feature list for inference
  model_metrics.json          — per-model train / test scores
  model_comparison.png        — bar chart: test scores + train-test gap
  feature_importance.png      — top-20 features (all three models)
  model_training_summary.txt  — plain-text summary

Can be run standalone or called by full_pipeline.py via
  train_model(labeled_df, output_dir, cfg).

Configuration loaded from config.yaml at the project root.
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
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Default label-to-class mapping
# =============================================================================

DEFAULT_LABEL_TO_CLASS = {
    "CS": "vocalizations", "CNS": "vocalizations",
    "TS": "vocalizations", "TNS": "vocalizations",
    "CGO": "unlabeled",    "TGO": "unlabeled",
    "TC": "joint_attention", "AO": "joint_attention",
    "AT": "joint_attention", "GO": "joint_attention", "GT": "joint_attention",
    "T": "interactions",  "T_V": "interactions",
    "T_P": "interactions", "C": "interactions",
    "TST": "coordination", "THO": "coordination", "TSI": "coordination",
    "TLF": "coordination", "TRE": "coordination", "TCR": "coordination",
    "CST": "coordination", "CHO": "coordination", "CSI": "coordination",
    "CLF": "coordination", "CRE": "coordination", "CCR": "coordination",
}

NON_POSTURE_CLASSES = {"vocalizations", "unlabeled"}


# =============================================================================
# Feature engineering (shared with predict.py)
# =============================================================================

def normalize_keypoints(df, keypoints):
    """Normalize keypoints relative to torso length."""
    df["torso_length"] = np.sqrt(
        (df["neck_x"] - df["mid_hip_x"]) ** 2 +
        (df["neck_y"] - df["mid_hip_y"]) ** 2
    )
    df["torso_length"] = df["torso_length"].replace(0, np.nan)
    df["torso_length"] = df["torso_length"].fillna(df["torso_length"].median()).clip(lower=1e-3)
    for col in keypoints:
        if "_x" in col:
            df[col] = (df[col] - df["mid_hip_x"]) / df["torso_length"]
        else:
            df[col] = (df[col] - df["mid_hip_y"]) / df["torso_length"]
    return df


def add_derived_pose_features(df):
    """Add joint angles, inter-joint distances, and symmetry features."""

    def angle(a, b, c):
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.arccos(np.clip(cos, -1.0, 1.0)) * (180.0 / np.pi)

    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    angle_defs = {
        "r_elbow_angle":    ("r_shoulder", "r_elbow",    "r_wrist"),
        "l_elbow_angle":    ("l_shoulder", "l_elbow",    "l_wrist"),
        "r_shoulder_angle": ("neck",       "r_shoulder", "r_elbow"),
        "l_shoulder_angle": ("neck",       "l_shoulder", "l_elbow"),
        "r_knee_angle":     ("r_hip",      "r_knee",     "r_ankle"),
        "l_knee_angle":     ("l_hip",      "l_knee",     "l_ankle"),
        "r_hip_angle":      ("mid_hip",    "r_hip",      "r_knee"),
        "l_hip_angle":      ("mid_hip",    "l_hip",      "l_knee"),
        "trunk_angle":      ("nose",       "neck",       "mid_hip"),
    }
    for col, (a, b, c) in angle_defs.items():
        df[col] = df.apply(lambda r: angle(
            np.array([r[f"{a}_x"], r[f"{a}_y"]]),
            np.array([r[f"{b}_x"], r[f"{b}_y"]]),
            np.array([r[f"{c}_x"], r[f"{c}_y"]]),
        ), axis=1)

    df["eye_to_eye"]      = df.apply(lambda r: dist(np.array([r["l_eye_x"],   r["l_eye_y"]]),  np.array([r["r_eye_x"],   r["r_eye_y"]])),  axis=1)
    df["nose_to_neck"]    = df.apply(lambda r: dist(np.array([r["nose_x"],    r["nose_y"]]),    np.array([r["neck_x"],    r["neck_y"]])),    axis=1)
    df["r_wrist_to_hip"]  = df.apply(lambda r: dist(np.array([r["r_wrist_x"], r["r_wrist_y"]]), np.array([r["mid_hip_x"], r["mid_hip_y"]])), axis=1)
    df["l_wrist_to_hip"]  = df.apply(lambda r: dist(np.array([r["l_wrist_x"], r["l_wrist_y"]]), np.array([r["mid_hip_x"], r["mid_hip_y"]])), axis=1)
    df["r_wrist_to_nose"] = df.apply(lambda r: dist(np.array([r["r_wrist_x"], r["r_wrist_y"]]), np.array([r["nose_x"],    r["nose_y"]])),    axis=1)
    df["l_wrist_to_nose"] = df.apply(lambda r: dist(np.array([r["l_wrist_x"], r["l_wrist_y"]]), np.array([r["nose_x"],    r["nose_y"]])),    axis=1)
    df["nose_to_ankles"]  = df.apply(lambda r: (
        dist(np.array([r["nose_x"], r["nose_y"]]), np.array([r["l_ankle_x"], r["l_ankle_y"]])) +
        dist(np.array([r["nose_x"], r["nose_y"]]), np.array([r["r_ankle_x"], r["r_ankle_y"]]))
    ) / 2, axis=1)
    df["hip_to_ankle"]    = df.apply(lambda r: (
        dist(np.array([r["mid_hip_x"], r["mid_hip_y"]]), np.array([r["l_ankle_x"], r["l_ankle_y"]])) +
        dist(np.array([r["mid_hip_x"], r["mid_hip_y"]]), np.array([r["r_ankle_x"], r["r_ankle_y"]]))
    ) / 2, axis=1)

    df["shoulder_y_diff"]     = df["l_shoulder_y"]    - df["r_shoulder_y"]
    df["hip_y_diff"]          = df["l_hip_y"]          - df["r_hip_y"]
    df["elbow_angle_diff"]    = df["l_elbow_angle"]    - df["r_elbow_angle"]
    df["knee_angle_diff"]     = df["l_knee_angle"]     - df["r_knee_angle"]
    df["wrist_to_hip_diff"]   = df["l_wrist_to_hip"]   - df["r_wrist_to_hip"]
    df["shoulder_angle_diff"] = df["l_shoulder_angle"] - df["r_shoulder_angle"]
    df["com_x"]               = (df["mid_hip_x"] + df["neck_x"]) / 2
    df["com_y"]               = (df["mid_hip_y"] + df["neck_y"]) / 2
    df["body_spread_x"]       = (df[["l_wrist_x", "r_wrist_x", "l_ankle_x", "r_ankle_x"]].max(axis=1) -
                                  df[["l_wrist_x", "r_wrist_x", "l_ankle_x", "r_ankle_x"]].min(axis=1))
    df["body_spread_y"]       = (df[["nose_y", "l_ankle_y", "r_ankle_y"]].max(axis=1) -
                                  df[["nose_y", "l_ankle_y", "r_ankle_y"]].min(axis=1))
    return df


def add_temporal_features(df, group_col="annotation_label", fps=15):
    """
    Velocity and acceleration for selected keypoints.
    group_col: column to group by ('annotation_label' in training, 'person_label' in inference).
    NaNs kept intact for downstream median imputation.
    """
    df = df.sort_values(by=[group_col, "time_s"]).reset_index(drop=True)
    df["delta_time"] = (
        df.groupby(group_col)["time_s"].diff().fillna(1 / fps).clip(lower=0.001)
    )

    selected = [
        "com_x", "com_y", "nose_x", "nose_y",
        "l_wrist_x", "l_wrist_y", "r_wrist_x", "r_wrist_y",
        "l_ankle_x", "l_ankle_y", "r_ankle_x", "r_ankle_y",
        "neck_x",    "neck_y",    "mid_hip_x",  "mid_hip_y",
    ]
    temporal_cols = []
    for col in selected:
        if col not in df.columns:
            continue
        vel = f"{col}_vel"
        acc = f"{col}_acc"
        df[vel] = df.groupby(group_col)[col].diff() / df["delta_time"]
        df[acc] = df.groupby(group_col)[vel].diff() / df["delta_time"]
        temporal_cols.extend([vel, acc])

    for col in temporal_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    n = len(temporal_cols) // 2
    logger.info(f"Added {n} velocity + {n} acceleration features")
    return df, temporal_cols


# =============================================================================
# Quality checks
# =============================================================================

def check_data_leakage(df, label_col="annotation_label"):
    print(f"\n{'='*70}\nDATA LEAKAGE CHECKS\n{'='*70}")
    if "time_s" in df.columns:
        classes     = df[label_col].value_counts()
        significant = classes[classes >= 100].index.tolist()
        non_overlap = []
        for i, c1 in enumerate(significant[:10]):
            for c2 in significant[i + 1:10]:
                r1 = df[df[label_col] == c1]["time_s"].agg(["min", "max"])
                r2 = df[df[label_col] == c2]["time_s"].agg(["min", "max"])
                if r1["max"] < r2["min"] or r2["max"] < r1["min"]:
                    non_overlap.append((c1, c2))
        if non_overlap:
            print(f"  WARNING: {len(non_overlap)} class pairs don't overlap in time")
        else:
            print("  ✓ Significant classes show temporal overlap")
    dups = df.duplicated().sum()
    print(f"  {'WARNING: ' + str(dups) + ' duplicate rows' if dups else '✓ No duplicate rows'}")
    print("=" * 70)


def detect_overfitting(train_f1, test_f1, model_name, threshold=0.10):
    gap = train_f1 - test_f1
    print(f"\n  {model_name} — Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Gap: {gap:.4f}")
    if gap > 0.15:
        print("    SEVERE OVERFITTING")
        return "severe"
    if gap > threshold:
        print("    Moderate overfitting")
        return "moderate"
    print("    No significant overfitting")
    return "good"


# =============================================================================
# Visualisation
# =============================================================================

def _plot_model_comparison(test_metrics, train_metrics, output_dir, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pd.DataFrame(test_metrics).T.plot(
        kind="bar", ax=axes[0], edgecolor="black", linewidth=1.5, alpha=0.85)
    axes[0].set_title("Test Set Performance", fontsize=14, weight="bold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim([0, 1.05])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].legend(title="Metric", loc="lower right")
    axes[0].grid(True, alpha=0.3, axis="y")

    gap = (pd.DataFrame(train_metrics).T - pd.DataFrame(test_metrics).T)["f1"]
    gap.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black", linewidth=1.5)
    axes[1].axhline(0.05, color="orange", linestyle="--", linewidth=2, label="Acceptable (0.05)")
    axes[1].axhline(0.10, color="red",    linestyle="--", linewidth=2, label="Warning (0.10)")
    axes[1].set_title("Train-Test F1 Gap", fontsize=14, weight="bold")
    axes[1].set_ylabel("F1 Gap")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison.png")


def _plot_feature_importance(models, feature_names, output_dir, dpi=300):
    capable = {n: m for n, m in models.items() if hasattr(m, "feature_importances_")}
    if not capable:
        return
    n    = len(capable)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, capable.items()):
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
    print("  Saved: feature_importance.png")


# =============================================================================
# Core training function
# =============================================================================

def train_model(labeled_df, output_dir, cfg=None):
    """
    Full feature-engineering → training → evaluation → artifact-save pipeline.

    Args:
        labeled_df: Combined labeled features DataFrame.
        output_dir: Directory to save model artifacts and plots.
        cfg:        Parsed config.yaml dict (optional).

    Returns:
        dict: best_model, model_name, label_encoder, imputer,
              feature_names, f1_score, test_metrics, train_metrics,
              overfitting_status, models
    """
    tc            = (cfg or {}).get("model_training", {})
    min_samples   = tc.get("min_samples_per_class", 30)
    test_size     = tc.get("test_size", 0.2)
    n_iter_search = tc.get("n_iter_search", 50)
    dpi           = (cfg or {}).get("plotting", {}).get("dpi", 300)
    fps           = (cfg or {}).get("model", {}).get("fps", 15)
    label_to_class = tc.get("label_to_class", DEFAULT_LABEL_TO_CLASS)
    non_posture    = set(tc.get("non_posture_classes", list(NON_POSTURE_CLASSES)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nIMPROVED MODEL TRAINING PIPELINE\n{'='*70}")
    df = labeled_df.copy()
    print(f"\n✓ Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Filter non-posture classes
    df["label_class"] = df["annotation_label"].map(label_to_class)
    df = df[~df["label_class"].isin(non_posture)].copy()
    print(f"✓ Filtered dataset size: {len(df):,}")

    # Filter classes with too few samples
    counts    = df["annotation_label"].value_counts()
    valid_cls = counts[counts >= min_samples].index.tolist()
    df        = df[df["annotation_label"].isin(valid_cls)].copy()
    print(f"✓ Kept {len(valid_cls)} classes with >= {min_samples} samples")

    check_data_leakage(df)

    # Feature engineering
    print("\n--- Feature Engineering ---")
    keypoints = [c for c in df.columns if "_x" in c or "_y" in c]
    print(f"✓ Found {len(keypoints)} keypoint columns")
    df = normalize_keypoints(df, keypoints)
    df = add_derived_pose_features(df)
    df, _ = add_temporal_features(df, group_col="annotation_label", fps=fps)

    # Encode labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["annotation_label"])
    print(f"✓ Encoded {len(le.classes_)} classes: "
          f"{list(le.classes_[:10])}{'...' if len(le.classes_) > 10 else ''}")
    y = df["label_encoded"].copy()

    # Build feature matrix
    print("\n--- Preparing Features ---")
    if "person_label" in df.columns:
        df = pd.get_dummies(df, columns=["person_label"], prefix="person")

    meta = ["time_s", "avg_pose_conf", "annotation_label", "label_class",
            "frame", "delta_time", "torso_length", "label_encoded",
            "wrist_dist", "time_min:s.ms", "project_name"]
    df   = df.drop(columns=[c for c in meta if c in df.columns])
    df   = df.select_dtypes(include=[np.number])
    X    = df
    print(f"✓ Feature matrix: {X.shape}")

    # Imputation
    nan_before = X.isnull().sum().sum()
    imputer    = SimpleImputer(strategy="median")
    X          = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    if np.isinf(X.values).any():
        X = X.replace([np.inf, -np.inf], np.nan)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print(f"✓ Imputed {nan_before:,} missing values")

    # Feature selection
    print("\n--- Feature Analysis ---")
    print(f"Initial features: {X.shape[1]}")

    zero_var = X.var()[X.var() < 1e-8].index.tolist()
    X        = X.drop(columns=zero_var)

    X_with_y  = X.copy()
    X_with_y["target"] = y.values
    corrs     = X_with_y.corr()["target"].drop("target").abs()
    low_corr  = corrs[corrs < 0.01].index.tolist()
    X         = X.drop(columns=low_corr)

    corr_mat  = X.corr().abs()
    upper     = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    redundant = []
    for col in upper.columns:
        if any(upper[col] > 0.95):
            peers = upper[col][upper[col] > 0.95].index.tolist()
            if corrs.get(col, 0) < corrs.get(peers[0], 0) and col not in redundant:
                redundant.append(col)
    X = X.drop(columns=redundant)

    print(f"Final features: {X.shape[1]} "
          f"(removed {len(zero_var)} zero-var, {len(low_corr)} low-corr, "
          f"{len(redundant)} redundant)")

    y = y.iloc[:len(X)].reset_index(drop=True)

    # Train / test split (time-aware when possible)
    print("\n--- Train/Test Split Strategy ---")
    if "time_s" in labeled_df.columns and len(labeled_df) == len(X):
        t         = labeled_df["time_s"].values[:len(X)]
        threshold = np.percentile(t, 80)
        tr_mask   = t <= threshold
        te_mask   = t > threshold
        if set(y[tr_mask].unique()) == set(y[te_mask].unique()):
            X_train, X_test = X[tr_mask], X[te_mask]
            y_train, y_test = y[tr_mask], y[te_mask]
            print(f"✓ Time-aware split: {len(X_train)}/{len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42)
            print(f"✓ Stratified split (fallback): {len(X_train)}/{len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        print(f"✓ Stratified split: {len(X_train)}/{len(X_test)}")

    # ---- Train three models ----
    models = {}

    print(f"\n{'='*70}")
    print("TRAINING TOP 3 MODELS (Handle Many Features Well)")
    print("Using strong regularization and early stopping where supported")
    print("="*70)

    # 1. LightGBM
    print("\n[1/3] Training LightGBM with early stopping...")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    lgb_probe = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=1.0, min_gain_to_split=0.01,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(20, verbose=False)])
    best_iter = lgb_probe.best_iteration_
    print(f"  Optimal iterations: {best_iter}")
    lgb_final = lgb.LGBMClassifier(
        n_estimators=best_iter, max_depth=5, learning_rate=0.05, num_leaves=31,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=1.0, min_gain_to_split=0.01,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_final.fit(X_train, y_train)
    models["LightGBM"] = lgb_final
    print("✓ LightGBM trained")

    # 2. XGBoost
    print("\n[2/3] Training XGBoost with hyperparameter tuning...")
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        param_distributions={
            "n_estimators":     [100, 150, 200],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.03, 0.05, 0.07, 0.1],
            "subsample":        [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8],
            "min_child_weight": [3, 5, 7],
            "gamma":            [0.1, 0.2, 0.3],
            "reg_alpha":        [0.5, 1.0, 1.5],
            "reg_lambda":       [0.5, 1.0, 1.5],
        },
        n_iter=n_iter_search,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="f1_weighted", n_jobs=-1, verbose=0, random_state=42,
    )
    xgb_search.fit(X_train, y_train)
    models["XGBoost"] = xgb_search.best_estimator_
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  Best CV F1: {xgb_search.best_score_:.4f}")
    print("✓ XGBoost trained")

    # 3. HistGradientBoosting
    print("\n[3/3] Training HistGradientBoosting with early stopping...")
    hist = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=30,
        max_bins=255, max_leaf_nodes=31,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
        random_state=42, verbose=0,
    )
    hist.fit(X_train, y_train)
    models["HistGradientBoosting"] = hist
    print("✓ HistGradientBoosting trained")

    # ---- Evaluate ----
    print(f"\n{'='*70}\nEVALUATING MODELS\n{'='*70}")
    train_metrics, test_metrics, overfit = {}, {}, {}

    for name, model in models.items():
        tr_pred = model.predict(X_train)
        te_pred = model.predict(X_test)
        tr_prob = model.predict_proba(X_train)
        te_prob = model.predict_proba(X_test)
        print(f"\n--- {name} ---")
        train_metrics[name] = {
            "f1":        float(f1_score(y_train, tr_pred, average="weighted")),
            "precision": float(precision_score(y_train, tr_pred, average="weighted")),
            "recall":    float(recall_score(y_train, tr_pred, average="weighted")),
            "roc_auc":   float(roc_auc_score(y_train, tr_prob, multi_class="ovr", average="macro")),
        }
        test_metrics[name] = {
            "f1":        float(f1_score(y_test, te_pred, average="weighted")),
            "precision": float(precision_score(y_test, te_pred, average="weighted")),
            "recall":    float(recall_score(y_test, te_pred, average="weighted")),
            "roc_auc":   float(roc_auc_score(y_test, te_prob, multi_class="ovr", average="macro")),
        }
        print(f"Test  F1: {test_metrics[name]['f1']:.4f}, "
              f"ROC-AUC: {test_metrics[name]['roc_auc']:.4f}")
        overfit[name] = detect_overfitting(
            train_metrics[name]["f1"], test_metrics[name]["f1"], name)

    _plot_model_comparison(test_metrics, train_metrics, output_dir, dpi)
    _plot_feature_importance(models, list(X.columns), output_dir, dpi)

    # ---- Best model + cross-validation ----
    best_name  = max(test_metrics, key=lambda k: test_metrics[k]["f1"])
    best_model = models[best_name]
    best_f1    = test_metrics[best_name]["f1"]

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name}")
    print(f"Test F1:    {best_f1:.4f}")
    print(f"Overfitting:{overfit[best_name]}")
    print("="*70)

    if hasattr(best_model, "feature_importances_"):
        feat_imp = (pd.DataFrame({"feature": X.columns,
                                   "importance": best_model.feature_importances_})
                    .sort_values("importance", ascending=False))
        print(f"\nTop 15 features ({best_name}):")
        for _, row in feat_imp.head(15).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")

    cv_scores = cross_val_score(
        best_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="f1_weighted", n_jobs=-1,
    )
    print(f"\nCV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    if cv_scores.std() > 0.05:
        print("  ⚠ High CV variance — model may be unstable")
    else:
        print("  ✓ Low CV variance — model is stable")

    # ---- Save artifacts ----
    feature_names = list(X.columns)
    fname = f"model_{best_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(best_model, output_dir / fname)

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / "model_metrics.json", "w") as f:
        json.dump({
            "best_model":         best_name,
            "train_metrics":      train_metrics,
            "test_metrics":       test_metrics,
            "overfitting_status": overfit,
            "n_features":         len(feature_names),
            "n_classes":          len(le.classes_),
        }, f, indent=2)

    with open(output_dir / "model_training_summary.txt", "w") as f:
        f.write(f"{'='*70}\nMODEL TRAINING SUMMARY\n{'='*70}\n\n")
        f.write(f"Best Model:    {best_name}\n")
        f.write(f"F1 Score:      {best_f1:.4f}\n")
        f.write(f"Overfitting:   {overfit[best_name]}\n")
        f.write(f"Train rows:    {len(X_train):,}\n")
        f.write(f"Test rows:     {len(X_test):,}\n")
        f.write(f"Features:      {len(feature_names)}\n")
        f.write(f"Classes ({len(le.classes_)}): {list(le.classes_)}\n")
        if "project_name" in labeled_df.columns:
            f.write(f"Projects:      {labeled_df['project_name'].nunique()}\n")

    print(f"\n✓ Saved all models and artifacts to {output_dir}")

    return {
        "models":             models,
        "best_model":         best_model,
        "best_model_name":    best_name,
        "model_name":         best_name,   # alias for full_pipeline
        "label_encoder":      le,
        "imputer":            imputer,
        "feature_names":      feature_names,
        "f1_score":           best_f1,
        "test_metrics":       test_metrics,
        "train_metrics":      train_metrics,
        "overfitting_status": overfit,
    }


# =============================================================================
# Standalone entry point
# =============================================================================

def main():
    CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["directories"]["output_base_dir"])
    sanitized  = output_dir / "combined_labeled_features_sanitized.csv"
    combined   = output_dir / "combined_labeled_features.csv"
    data_path  = sanitized if sanitized.exists() else combined

    if not data_path.exists():
        raise FileNotFoundError(
            "No combined labeled data found. "
            "Run annotation alignment (Step 4) first.")

    df  = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {data_path.name}: {df.shape}")
    pkg = train_model(df, output_dir, cfg)
    print(f"\nBest model: {pkg['model_name']}  F1={pkg['f1_score']:.4f}")


if __name__ == "__main__":
    main()

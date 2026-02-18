#!/usr/bin/env python3
# coding: utf-8
"""
full_pipeline.py
----------------
End-to-end orchestrator for the Video Annotation System.

Pipeline stages (per project):
  Step 1 — Video processing   (src/video_processing/processing.py)
  Step 2 — Pose extraction    (src/pose/extractor.py)
  Step 3 — Pose clustering    (src/pose/clustering.py)  [optional]
  Step 4 — Annotation alignment (src/annotations/generator.py)

Followed by (across all projects):
  Step 5 — Model training     (src/models/train.py)     [skipped if model exists]
  Step 6 — Prediction         (src/models/predict.py)   [run standalone]

Configuration is loaded from config.yaml at the project root.
"""



import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Add src root to path so submodules can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
from video_processing.processing import run_video_processing
from pose.extractor import run_pose_extraction
from pose.clustering import run_pose_clustering
from annotations.generator import run_annotation_alignment
from models.train import train_model
from models.predict import predict_annotations



# =========================
# Load configuration
# =========================
CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)
    raw_root = os.getenv("RAW_VIDEO_ROOT", cfg['directories']['raw_video_root'])


# Directories
BASE_DATA_DIR   = Path(cfg["directories"]["processed_data_dir"])
OUTPUT_BASE_DIR = Path(cfg["directories"]["output_dir"])
ANNOTATIONS_DIR = Path(cfg["directories"]["annotations_dir"])
OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)

# Pipeline flags
SKIP_POSE_CLUSTERING       = cfg["flags"]["skip_pose_clustering"]
SKIP_ANNOTATION_PROCESSING = cfg["flags"]["skip_annotation_processing"]
FORCE_RECOMBINE            = cfg["flags"]["force_recombine"]

# Trimming parameters
TRIM_TIMES   = {k: tuple(v) for k, v in cfg["trim_times"].items()}
DEFAULT_TRIM = tuple(cfg["default_trim"])

# =========================
# Project directories
# =========================
project_dirs = [d for d in BASE_DATA_DIR.iterdir() if d.is_dir()]
print(f"\nFound {len(project_dirs)} project directories.\n{'='*70}")

# =========================
# MAIN LOOP: Per-project processing
# =========================
for project_dir in project_dirs:
    project_name = project_dir.name
    OUTPUT_DIR = OUTPUT_BASE_DIR / project_name
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print(f"\n▶ Processing project: {project_name}")

    # -------------------------------
    # Step 1: Video processing
    # -------------------------------
    raw_video_root = Path(cfg["directories"]["raw_video_root"])
    raw_project_path = raw_video_root / project_name
    if raw_project_path.exists():
        success = run_video_processing(raw_project_path, cfg)
        if not success:
            print(f"  ✗ Video processing failed for {project_name}, skipping project.")
            continue
    else:
        print(f"  ⚠ Raw project folder not found at {raw_project_path} — skipping video processing step.")

    # -------------------------------
    # Step 2: Pose extraction
    # -------------------------------
    if raw_project_path.exists():
        success = run_pose_extraction(raw_project_path, cfg)
        if not success:
            print(f"  ✗ Pose extraction failed for {project_name}, skipping project.")
            continue
    else:
        print(f"  ⚠ Raw project folder not found — skipping pose extraction step.")

    # -------------------------------
    # Load pose data
    # -------------------------------
    parquet_files = list(project_dir.glob("processed_data__*.parquet"))
    if not parquet_files:
        print(f"  ⚠ No parquet files found, skipping...")
        continue

    csv_path = project_dir / "processed_data.csv"
    if not csv_path.exists():
        df = pd.concat([pd.read_parquet(pf) for pf in parquet_files], ignore_index=True)
        df['person_label'] = df['person_label'].replace('Patient1', 'Child')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Merged and saved {len(df)} rows from {len(parquet_files)} files.")
    else:
        df = pd.read_csv(csv_path)
        print(f"  ✓ Loaded existing CSV: {csv_path.name} ({len(df)} rows)")

    # -------------------------------
    # Step 3: Pose clustering (optional)
    # -------------------------------
    pose_output_path = OUTPUT_DIR / f"pose_clusters_{project_name}.parquet"
    if SKIP_POSE_CLUSTERING:
        print("  → Skipping pose clustering (flags.skip_pose_clustering: true).")
    elif pose_output_path.exists():
        print(f"  ✓ Using cached pose clustering: {pose_output_path.name}")
    else:
        print("  Running pose clustering...")
        try:
            pose_results = run_pose_clustering(
                project_dir=project_dir,
                output_dir=OUTPUT_DIR,
                cfg=cfg,
            )
            pose_results["data"].to_parquet(pose_output_path)
            print(f"  ✓ Pose clustering completed and cached.")
        except Exception as e:
            print(f"  ✗ Pose clustering failed: {e}")
            # Non-fatal — continue to annotation alignment

    # -------------------------------
    # Step 4: Annotation alignment
    # -------------------------------
    if SKIP_ANNOTATION_PROCESSING:
        print("  → Skipping annotation alignment (flags.skip_annotation_processing: true).")
        continue

    labeled_df = run_annotation_alignment(project_name, cfg, OUTPUT_DIR)
    if labeled_df is None:
        print(f"  ✗ Annotation alignment failed for {project_name}, skipping project.")
        continue

# =========================
# Combine labeled features
# =========================
print("\n" + "="*70)
print("Combining labeled data from all projects for model training...")
print("="*70)

combined_path = OUTPUT_BASE_DIR / "combined_labeled_features.csv"

if combined_path.exists() and not FORCE_RECOMBINE:
    print(f"✓ Found existing combined dataset: {combined_path.name}")
    if not SKIP_ANNOTATION_PROCESSING:
        print("  → Updating combined dataset with new labeled features...")
        FORCE_RECOMBINE = True

if not combined_path.exists() or FORCE_RECOMBINE:
    labeled_files = list(OUTPUT_BASE_DIR.glob("*/labeled_features.csv"))

    if not labeled_files:
        print("⚠ No labeled feature files found — cannot combine.")
        exit()

    print(f"✓ Found {len(labeled_files)} labeled feature files.")
    all_dfs = []
    for f in labeled_files:
        try:
            df_part = pd.read_csv(f)
            df_part["project_name"] = f.parent.name
            all_dfs.append(df_part)
            print(f"  ✓ Loaded: {f.parent.name} ({len(df_part):,} rows)")
        except Exception as e:
            print(f"  ⚠ Skipping {f.name}: {e}")

    if not all_dfs:
        print("✗ No valid dataframes to combine")
        exit()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ Combined dataset shape: {combined_df.shape}")
    combined_df.to_csv(combined_path, index=False)
    print(f"✓ Saved: {combined_path.name}")

    # Plot global annotation distribution
    try:
        label_counts = combined_df["annotation_label"].value_counts().sort_values(ascending=False)
        n_bars  = len(label_counts)
        colors  = plt.cm.viridis(np.linspace(0.2, 0.9, n_bars))
        fig, ax = plt.subplots(figsize=(24, 12))
        bars = ax.bar(range(n_bars), label_counts.values,
                      color=colors, edgecolor="black", linewidth=2.5, alpha=0.85, width=0.7)
        ax.set_title("Global Annotation Distribution (All Projects)",
                     fontsize=40, weight="bold", pad=25)
        ax.set_xlabel("Annotation Label", fontsize=36, weight="bold")
        ax.set_ylabel("Total Count",      fontsize=36, weight="bold")
        ax.set_xticks(range(n_bars))
        ax.set_xticklabels(label_counts.index, fontsize=30, rotation=0)
        ax.tick_params(axis="y", labelsize=30)
        for bar, count in zip(bars, label_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f"{count:,}", ha="center", va="bottom", fontsize=26, weight="bold")
        ax.grid(True, alpha=0.3, axis="y", linewidth=1.5)
        plt.tight_layout()
        global_plot_path = OUTPUT_BASE_DIR / "global_annotation_distribution.png"
        plt.savefig(global_plot_path, dpi=cfg["plotting"]["dpi"],
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✓ Saved: {global_plot_path.name}")
    except Exception as e:
        print(f"  ⚠ Failed to plot global distribution: {e}")

# =========================
# Data Sanitization
# =========================
print("\n--- Data Sanitization ---")
combined_df = pd.read_csv(combined_path)
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
before_drop = len(combined_df)
combined_df.dropna(inplace=True)
dropped = before_drop - len(combined_df)
print(f"✓ {'Dropped ' + str(dropped) + ' rows with NaN/infinite values.' if dropped else 'No rows with NaN/infinite values found.'}")

numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
max_val = combined_df[numeric_cols].abs().max().max()
print(f"✓ Feature values within reasonable range (max={max_val:.2e}).")

sanitized_path = OUTPUT_BASE_DIR / "combined_labeled_features_sanitized.csv"
combined_df.to_csv(sanitized_path, index=False)
print(f"✓ Saved sanitized data: {sanitized_path.name}")

# =========================
# Step 5: Model Training
# =========================
print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

model_file = OUTPUT_BASE_DIR / cfg["model"]["file"]
if model_file.exists():
    print(f"✓ Model already exists: {model_file.name} — skipping training.")
    print("  (Delete the file or set a new model.file path to retrain.)")
    model_package = None
else:
    try:
        model_package = train_model(combined_df, OUTPUT_BASE_DIR, cfg)
        print(f"\n✓ Training complete.")
        print(f"✓ Best Model: {model_package['model_name']}")
        print(f"✓ Final F1 Score: {model_package['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        model_package = None

# =========================
# Final Summary
# =========================
print("\n" + "="*70)
print("PIPELINE COMPLETE - SUMMARY")
print("="*70)
print(f"✓ All projects processed")
print(f"✓ Results saved in: {OUTPUT_BASE_DIR}")
if model_package:
    print(f"✓ Best Model: {model_package['model_name']}")
    print(f"✓ F1 Score:   {model_package['f1_score']:.4f}")
print("="*70)

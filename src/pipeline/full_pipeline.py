#!/usr/bin/env python3
# coding: utf-8
"""
full_pipeline.py
----------------
End-to-end orchestrator for the Video Annotation System.
Runs pose clustering, annotation alignment, data combination,
sanitization, and model training across all projects.

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
import importnb

# Add src root to path so processing.py can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
from video_processing.processing import run_video_processing


# =========================
# Load configuration
# =========================
CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

# Directories
BASE_DATA_DIR   = Path(cfg["directories"]["base_data_dir"])
OUTPUT_BASE_DIR = Path(cfg["directories"]["output_base_dir"])
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
# Load submodules
# =========================
print("\n" + "="*70)
print("Loading analysis modules...")
print("="*70)

with importnb.Notebook():
    import PoseClustClassifier as pcc
    import AnnotLabelGenerator as alg
    import ModelTraining3 as mt

# Note: The global style is already applied, so all plots in these modules
# will automatically use the configured style

print("✓ All modules loaded")

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
    # Pose clustering
    # -------------------------------
    pose_output_path = OUTPUT_DIR / f"pose_clusters_{project_name}.parquet"
    if SKIP_POSE_CLUSTERING and pose_output_path.exists():
        pose_results = {'data': pd.read_parquet(pose_output_path)}
        print(f"  ✓ Using existing pose clustering: {pose_output_path.name}")
    else:
        print("  Running pose clustering...")
        try:
            pose_results = pcc.run_pose_clustering(data_path=project_dir, output_dir=OUTPUT_DIR)
            pose_results['data'].to_parquet(pose_output_path)
            print(f"  ✓ Pose clustering completed and saved.")
        except Exception as e:
            print(f"  ✗ Pose clustering failed: {e}")
            continue

    # -------------------------------
    # Annotation alignment
    # -------------------------------
    if SKIP_ANNOTATION_PROCESSING:
        print("  → Skipping annotation processing per flag.")
        continue

    annotation_files = list(ANNOTATIONS_DIR.glob(f"{project_name.split('[')[0]}*.txt"))
    if not annotation_files:
        print(f"  ⚠ No annotation file found, skipping label generation...")
        continue

    annotation_file = annotation_files[0]
    print(f"  ✓ Using annotation file: {annotation_file.name}")
    start_trim, end_trim, fps = TRIM_TIMES.get(project_name, DEFAULT_TRIM)

    try:
        labeled_df = alg.run_annotation_alignment_pipeline(
            annotation_file_path=annotation_file,
            features_file_path=csv_path,
            start_trim_sec=start_trim,
            end_trim_sec=end_trim
        )

        output_path = OUTPUT_DIR / "labeled_features.csv"
        labeled_df.to_csv(output_path, index=False)
        print(f"  ✓ Labeled features saved: {output_path.name}")

        # Plot annotation distribution with A0 POSTER styling
        try:
            label_counts = labeled_df['annotation_label'].value_counts().sort_values(ascending=False)
            
            # A0 poster sizing
            fig, ax = plt.subplots(figsize=(18, 10))
            
            # Create viridis colors for bars
            n_bars = len(label_counts)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_bars))
            
            bars = ax.bar(range(n_bars), label_counts.values, 
                         color=colors, edgecolor='black', linewidth=2, alpha=0.85)
            
            ax.set_title(f"Annotation Distribution: {project_name}", 
                        fontsize=36, weight='bold', pad=20)
            ax.set_xlabel("Annotation Label", fontsize=32, weight='bold')
            ax.set_ylabel("Count", fontsize=32, weight='bold')
            
            # Set x-tick labels HORIZONTALLY
            ax.set_xticks(range(n_bars))
            ax.set_xticklabels(label_counts.index, fontsize=28, rotation=0)
            ax.tick_params(axis='y', labelsize=28)
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=24, weight='bold')
            
            ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
            plt.tight_layout()
            
            plot_path = OUTPUT_DIR / f"annotation_stats_{project_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  ✓ Saved annotation distribution plot: {plot_path.name}")
            
            # Print statistics
            print(f"  📊 Annotation statistics:")
            print(f"    - Total labeled frames: {len(labeled_df)}")
            print(f"    - Unique classes: {labeled_df['annotation_label'].nunique()}")
            print(f"    - Top 5 classes:")
            print(labeled_df['annotation_label'].value_counts().head(5).to_string())
            
        except Exception as e:
            print(f"  ⚠ Failed to plot annotation stats: {e}")

    except Exception as e:
        print(f"  ✗ Annotation alignment failed: {e}")
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
            print(f"  ✓ Loaded: {f.parent.name} ({len(df_part)} rows)")
        except Exception as e:
            print(f"  ⚠ Skipping {f.name}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"✓ Combined dataset shape: {combined_df.shape}")
        
        combined_df.to_csv(combined_path, index=False)
        print(f"✓ Saved combined labeled data to: {combined_path}")
        
        # Plot global annotation distribution with A0 POSTER styling
        try:
            label_counts = combined_df['annotation_label'].value_counts().sort_values(ascending=False)
            
            # A0 poster sizing - EXTRA WIDE for horizontal labels
            fig, ax = plt.subplots(figsize=(24, 12))
            
            # Create viridis colors for bars
            n_bars = len(label_counts)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_bars))
            
            bars = ax.bar(range(n_bars), label_counts.values, 
                         color=colors, edgecolor='black', linewidth=2.5, alpha=0.85, width=0.7)
            
            ax.set_title("Global Annotation Distribution (All Projects)", 
                        fontsize=40, weight='bold', pad=25)
            ax.set_xlabel("Annotation Label", fontsize=36, weight='bold')
            ax.set_ylabel("Total Count", fontsize=36, weight='bold')
            
            # Set x-tick labels HORIZONTALLY
            ax.set_xticks(range(n_bars))
            ax.set_xticklabels(label_counts.index, fontsize=30, rotation=0)
            ax.tick_params(axis='y', labelsize=30)
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=26, weight='bold')
            
            ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
            plt.tight_layout()
            
#             global_plot_path = OUTPUT_BASE_DIR / "global_annotation_distribution.png"
#             plt.savefig(global_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
#             print(f"✓ Saved global annotation distribution: {global_plot_path.name}")
            
            # Also save individual class statistics
#             stats_path = OUTPUT_BASE_DIR / "annotation_statistics.txt"
#             with open(stats_path, 'w') as f:
#                 f.write("="*70 + "\n")
#                 f.write("GLOBAL ANNOTATION STATISTICS\n")
#                 f.write("="*70 + "\n\n")
#                 f.write(f"Total labeled frames: {len(combined_df):,}\n")
#                 f.write(f"Unique classes: {combined_df['annotation_label'].nunique()}\n")
#                 f.write(f"Number of projects: {combined_df['project_name'].nunique()}\n\n")
#                 f.write("Class Distribution:\n")
#                 f.write("-"*70 + "\n")
#                 for label, count in label_counts.items():
#                     percentage = (count / len(combined_df)) * 100
#                     f.write(f"{label:15s}: {count:6,} ({percentage:5.2f}%)\n")
            
#             print(f"✓ Saved annotation statistics: {stats_path.name}")
            
        except Exception as e:
            print(f"  ⚠ Failed to plot global distribution: {e}")
    else:
        print("✗ No valid dataframes to combine")
        exit()

# =========================
# Data Sanitization
# =========================
print("\n--- Data Sanitization ---")
combined_df = pd.read_csv(combined_path)

# Handle infinite values
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
before_drop = len(combined_df)
combined_df.dropna(inplace=True)
after_drop = len(combined_df)
dropped = before_drop - after_drop

if dropped > 0:
    print(f"✓ Dropped {dropped} rows with NaN/infinite values.")
else:
    print(f"✓ No rows with NaN/infinite values found.")

# Check feature ranges
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
max_val = combined_df[numeric_cols].abs().max().max()
print(f"✓ Feature values within reasonable range (max={max_val:.2e}).")

# Save sanitized data
sanitized_path = OUTPUT_BASE_DIR / "combined_labeled_features_sanitized.csv"
combined_df.to_csv(sanitized_path, index=False)
print(f"✓ Saved sanitized data: {sanitized_path.name}")

# =========================
# MODEL TRAINING
# =========================
print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

try:
    results = {'labeled_features': combined_df}
    model_package = mt.main(results, output_dir=OUTPUT_BASE_DIR)
    
    print(f"\n✓ Combined model training complete.")
    print(f"✓ Best Model: {model_package['model_name']}")
    print(f"✓ Final F1 Score: {model_package['f1_score']:.4f}")
    
    # Save model summary
    summary_path = OUTPUT_BASE_DIR / "model_training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Model: {model_package['model_name']}\n")
        f.write(f"F1 Score: {model_package['f1_score']:.4f}\n")
        f.write(f"Training data shape: {combined_df.shape}\n")
        f.write(f"Number of classes: {combined_df['annotation_label'].nunique()}\n")
        f.write(f"Number of projects: {combined_df['project_name'].nunique()}\n")
    
    print(f"✓ Saved model summary: {summary_path.name}")
    
except Exception as e:
    print(f"✗ Combined model training failed: {e}")
    import traceback
    traceback.print_exc()

# =========================
# Final Summary
# =========================
print("\n" + "="*70)
print("PIPELINE COMPLETE - SUMMARY")
print("="*70)
print(f"✓ All projects processed")
print(f"✓ Results saved in: {OUTPUT_BASE_DIR}")

print("="*70)

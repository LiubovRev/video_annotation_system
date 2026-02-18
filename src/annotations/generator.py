#!/usr/bin/env python3
# coding: utf-8
"""
generator.py
------------
Step 4 of the Video Annotation Pipeline.
Align annotation labels from .txt files with pose feature frames.

Pipeline stages:
  1. Parse annotations
  2. Adjust feature timestamps
  3. Map annotations to features
  4. Filter and clean
  5. Save labeled features
  6. Visualise annotation class distribution

Can be run standalone or imported by full_pipeline.py.
"""

from pathlib import Path
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Stage 1 — Parse annotation file
# =============================================================================

def parse_annotations(annotation_file_path: Path) -> pd.DataFrame:
    """
    Parse a WAKEE-format annotation .txt file into a DataFrame.

    Returns columns: start_time, end_time, label, person ("Child" or "Therapist").
    """
    rows = []
    with open(annotation_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            start_time = float(parts[2])
            end_time   = float(parts[4])
            label      = parts[7]
            first = label[0].upper()
            person = "Child" if first == "C" else "Therapist" if first == "T" else "Unknown"
            rows.append([start_time, end_time, label, person])

    df = pd.DataFrame(rows, columns=["start_time", "end_time", "label", "person"])
    print(f"  Parsed {len(df)} annotations from {annotation_file_path.name}")
    return df


# =============================================================================
# Stage 2 — Adjust feature timestamps
# =============================================================================

def adjust_feature_times(
    features_df: pd.DataFrame, start_trim_sec: float, end_trim_sec: Optional[float]
) -> pd.DataFrame:
    """
    Shift feature timestamps by start_trim_sec and clip to end_trim_sec.
    """
    df = features_df.copy()
    df["time_s"] = df["time_s"] + start_trim_sec
    if end_trim_sec is not None:
        df = df[df["time_s"] <= end_trim_sec].copy()
    return df


# =============================================================================
# Stage 3 — Map annotations to features
# =============================================================================

def map_annotations_to_features(
    features_df: pd.DataFrame, annotations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign annotation_label to each feature frame based on time overlap and person.
    """
    df = features_df.copy()
    df["annotation_label"] = None

    for i, row in df.iterrows():
        t, person = row["time_s"], row["person_label"]
        overlap = annotations_df[
            (annotations_df["start_time"] <= t) &
            (annotations_df["end_time"] >= t) &
            (annotations_df["person"] == person)
        ]
        if not overlap.empty:
            df.at[i, "annotation_label"] = overlap.iloc[0]["label"]

    return df


# =============================================================================
# Stage 4 — Filter and clean
# =============================================================================

COLUMNS_TO_DROP = ["time_min:s.ms"]


def filter_and_clean(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unlabelled frames and remove utility columns.
    """
    df = features_df.dropna(subset=["annotation_label"]).copy()
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# =============================================================================
# Stage 5 — Visualisation
# =============================================================================

def plot_annotation_distribution(
    labeled_df: pd.DataFrame, project_name: str, save_path: Path, dpi: int = 300
) -> None:
    """
    Save a bar chart of annotation class distribution.
    """
    label_counts = labeled_df["annotation_label"].value_counts().sort_values(ascending=False)
    n = len(label_counts)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))

    fig, ax = plt.subplots(figsize=(18, 10))
    bars = ax.bar(range(n), label_counts.values, color=colors, edgecolor="black", linewidth=2, alpha=0.85)

    ax.set_title(f"Annotation Distribution: {project_name}", fontsize=36, weight="bold", pad=20)
    ax.set_xlabel("Annotation Label", fontsize=32, weight="bold")
    ax.set_ylabel("Count", fontsize=32, weight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels(label_counts.index, fontsize=28, rotation=0)
    ax.tick_params(axis="y", labelsize=28)

    for bar, count in zip(bars, label_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                f"{count:,}", ha="center", va="bottom", fontsize=24, weight="bold")

    ax.grid(True, alpha=0.3, axis="y", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {save_path.name}")


# =============================================================================
# Core pipeline function
# =============================================================================

def run_annotation_alignment_pipeline(
    annotation_file_path: Path,
    features_file_path: Path,
    start_trim_sec: float,
    end_trim_sec: Optional[float]
) -> pd.DataFrame:
    """
    Run full annotation alignment pipeline for a single project.
    """
    annotations_df = parse_annotations(annotation_file_path)
    features_df = pd.read_csv(features_file_path)
    print(f"  Loaded features: {features_df.shape}")

    features_df = adjust_feature_times(features_df, start_trim_sec, end_trim_sec)
    print(f"  After time trim: {len(features_df):,} rows [{start_trim_sec}s – {end_trim_sec}]")

    features_df = map_annotations_to_features(features_df, annotations_df)
    features_df = filter_and_clean(features_df)
    print(f"  Labeled frames: {len(features_df):,} ({features_df['annotation_label'].nunique()} classes)")

    return features_df


def run_annotation_alignment(
    project_name: str, cfg: Dict, output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    Locate annotation and feature files for project_name, run alignment,
    save labeled features, and plot annotation stats.
    """
    annotations_dir = Path(cfg["directories"]["annotations_dir"])
    raw_video_root  = Path(cfg["directories"]["raw_video_root"])
    trim_times      = cfg.get("trim_times", {})
    default_trim    = cfg.get("default_trim", [0, None, 15])
    dpi             = cfg.get("plotting", {}).get("dpi", 300)

    # Locate annotation file
    annotation_files = list(annotations_dir.glob(f"{project_name.split('[')[0]}*.txt"))
    if not annotation_files:
        print(f"  No annotation file found for {project_name} — skipping.")
        return None
    annotation_file = annotation_files[0]
    print(f"  Annotation file: {annotation_file.name}")

    # Locate features CSV
    features_path = raw_video_root / project_name / "PosesDir" / "processed_data.csv"
    if not features_path.exists():
        features_path = Path(cfg["directories"]["base_data_dir"]) / project_name / "processed_data.csv"
    if not features_path.exists():
        print(f"  Features CSV not found for {project_name} — skipping.")
        return None

    # Trim times
    start_trim, end_trim, _ = trim_times.get(project_name, default_trim)
    if end_trim is None:
        end_trim = float("inf")

    # Run pipeline
    try:
        labeled_df = run_annotation_alignment_pipeline(annotation_file, features_path, start_trim, end_trim)
    except Exception as e:
        print(f"  Annotation alignment failed: {e}")
        return None

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "labeled_features.csv"
    labeled_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv.name}")

    # Plot
    plot_path = output_dir / f"annotation_stats_{project_name}.png"
    try:
        plot_annotation_distribution(labeled_df, project_name, plot_path, dpi=dpi)
    except Exception as e:
        print(f"  Warning: plot failed — {e}")

    # Summary
    print(f"  Top 5 classes:\n{labeled_df['annotation_label'].value_counts().head(5).to_string()}")
    return labeled_df


# =============================================================================
# Standalone entry point
# =============================================================================

def main():
    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    output_base = Path(cfg["directories"]["output_base_dir"])
    base_data   = Path(cfg["directories"]["base_data_dir"])
    project_dirs = [d for d in base_data.iterdir() if d.is_dir()]
    print(f"Found {len(project_dirs)} project directories.")

    results = {}
    for project_dir in sorted(project_dirs):
        name = project_dir.name
        output_dir = output_base / name
        print(f"\n{'='*70}\nAnnotation alignment: {name}\n{'='*70}")
        labeled_df = run_annotation_alignment(name, cfg, output_dir)
        results[name] = "OK" if labeled_df is not None else "FAILED/SKIPPED"

    print(f"\n{'='*70}\nANNOTATION ALIGNMENT SUMMARY\n{'='*70}")
    for name, status in results.items():
        print(f"  {status:15s}  {name}")


if __name__ == "__main__":
    main()

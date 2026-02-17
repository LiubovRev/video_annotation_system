#!/usr/bin/env python3
# coding: utf-8
"""
extractor.py
------------
Step 2 of the Video Annotation Pipeline.
Extracts and processes pose keypoint data from psifx output (.tar.gz JSON archives).

Pipeline stages:
  1. Extract & Rename  — unpack .tar.gz archives, flatten structure, rename
                         each JSON as id_<video_id>_<frame:05d>.json
  2. Collect           — copy all renamed JSONs into a single all_jsons/ folder
  3. Build DataFrame   — parse each JSON, assemble into a DataFrame, save CSV
  4. Process           — label persons, create time columns, reshape keypoints,
                         compute features, extract joint columns, rename to snake_case
  5. Save              — write processed_data.csv to the project poses directory
  6. Cleanup           — delete _extracted folders, merged_jsons/, all_jsons/

Can be run standalone (processes one project) or imported and called by
full_pipeline.py via run_pose_extraction(project_path, cfg).

Configuration is loaded from config.yaml at the project root.
"""

import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


# =============================================================================
# Constants
# =============================================================================

CONF_THRESHOLD = 0.5

JOINT_NAMES = {
    0:  "Nose",       1:  "Neck",        2:  "RShoulder",  3:  "RElbow",
    4:  "RWrist",     5:  "LShoulder",   6:  "LElbow",     7:  "LWrist",
    8:  "MidHip",     9:  "RHip",        10: "RKnee",      11: "RAnkle",
    12: "LHip",       13: "LKnee",       14: "LAnkle",     15: "REye",
    16: "LEye",       17: "REar",        18: "LEar",       19: "LBigToe",
    20: "LSmallToe",  21: "LHeel",       22: "RBigToe",    23: "RSmallToe",
    24: "RHeel",      25: "Background",
}

JOINT_NAMES_SNAKE = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "mid_hip",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
    "l_big_toe", "l_small_toe", "l_heel",
    "r_big_toe", "r_small_toe", "r_heel",
]

KEYPOINT_COLS = [
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
]

COLS_TO_DROP = [
    "pose_keypoints_2d", "face_keypoints_2d",
    "hand_left_keypoints_2d", "hand_right_keypoints_2d",
    "pose_keypoints_2d_array", "face_keypoints_2d_array",
    "hand_left_keypoints_2d_array", "hand_right_keypoints_2d_array",
    "pose_keypoints_2d_filtered", "pose_keypoints_2d_dict",
    "pose_keypoints_2d_array_filtered", "pose_keypoints_2d_array_dict",
    "frame", "source_id",
]


# =============================================================================
# Stage 1 — Extract & rename JSON files from .tar.gz archives
# =============================================================================

def extract_and_rename(base_path: Path, abs_base: int = 0) -> int:
    """
    Extract all .tar.gz files in base_path, flatten structure, and rename
    each JSON to id_<video_id>_<frame:05d>.json.

    Returns total number of renamed files.
    """
    tar_files = sorted(base_path.glob("*.tar.gz"))
    if not tar_files:
        print("  No .tar.gz files found.")
        return 0

    total_renamed = 0
    for tar_file in tar_files:
        print(f"  Extracting {tar_file.name}...")
        extract_dir = tar_file.with_name(f"{tar_file.stem}_extracted")
        extract_dir.mkdir(exist_ok=True)

        try:
            with tarfile.open(tar_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        member.name = Path(member.name).name  # flatten
                        tar.extract(member, path=extract_dir)
        except Exception as e:
            print(f"  Error extracting {tar_file.name}: {e}")
            continue

        json_files = list(extract_dir.rglob("*.json"))
        if not json_files:
            print(f"  No JSON files found in {tar_file.name}.")
            continue

        video_id = tar_file.stem.split(".")[0]

        for jf in json_files:
            match = re.search(r"(\d+)$", jf.stem)
            if not match:
                print(f"  Skipping (no trailing number): {jf.name}")
                continue

            abs_idx  = abs_base + int(match.group(1))
            new_name = f"id_{video_id}_{abs_idx:05}.json"
            new_path = jf.with_name(new_name)

            if jf.name == new_name or new_path.exists():
                continue

            jf.rename(new_path)
            total_renamed += 1

    print(f"  Total JSON files renamed: {total_renamed}")
    return total_renamed


# =============================================================================
# Stage 2 — Collect all JSONs into all_jsons/
# =============================================================================

def collect_jsons(base_path: Path) -> Path:
    """
    Copy all renamed JSONs from _extracted subfolders into base_path/all_jsons/.
    Returns the path to all_jsons/.
    """
    input_dir  = base_path / "all_jsons"
    merged_dir = base_path / "merged_jsons"
    input_dir.mkdir(exist_ok=True)

    for root, dirs, files in os.walk(base_path):
        current = Path(root)
        if current in [input_dir, merged_dir]:
            continue
        for file in files:
            if file.endswith(".json"):
                src = current / file
                dst = input_dir / file
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)

    print(f"  All JSONs collected in: {input_dir}")
    return input_dir


# =============================================================================
# Stage 3 — Build raw DataFrame from all_jsons/
# =============================================================================

def build_dataframe(input_dir: Path) -> pd.DataFrame:
    """
    Parse all id_*.json files in input_dir and return a raw DataFrame.
    Each JSON becomes one row; source_id and frame are added from the filename.
    """
    pattern  = re.compile(r"(id_\d+)_(\d+)\.json")
    all_data = []

    for json_file in input_dir.glob("id_*.json"):
        match = pattern.match(json_file.name)
        if not match:
            print(f"  Skipping unrecognized file: {json_file.name}")
            continue

        source_id = match.group(1)
        frame     = int(match.group(2))

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["source_id"] = source_id
                data["frame"]     = frame
                all_data.append(data)
            else:
                print(f"  Invalid format (not a dict): {json_file.name}")
        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"  DataFrame: {len(df):,} rows × {len(df.columns)} columns")
    else:
        df = pd.DataFrame()
        print("  No valid data loaded.")

    return df


# =============================================================================
# Stage 4 — Process DataFrame
# =============================================================================

def _is_all_zero_or_empty(x) -> bool:
    try:
        if isinstance(x, str):
            x = json.loads(x)
        arr = np.array(x)
        return arr.size == 0 or np.all(arr == 0)
    except Exception:
        return True


def _fast_reshape(x) -> np.ndarray:
    try:
        if isinstance(x, str):
            x = json.loads(x)
        arr = np.array(x)
        return arr.reshape(-1, 3) if arr.size % 3 == 0 else np.empty((0, 3))
    except Exception:
        return np.empty((0, 3))


def _get_joint_coords(arr: np.ndarray, joint_idx: int = 0):
    if arr.size == 0 or joint_idx >= len(arr):
        return (np.nan, np.nan)
    return tuple(arr[joint_idx][:2])


def _wrist_distance(arr: np.ndarray) -> float:
    if arr.size == 0:
        return np.nan
    return float(np.linalg.norm(arr[4][:2] - arr[7][:2]))


def _filter_low_confidence(arr: np.ndarray, threshold: float = CONF_THRESHOLD) -> np.ndarray:
    if arr.size == 0:
        return arr
    filtered = arr.copy()
    filtered[filtered[:, 2] < threshold, :2] = np.nan
    return filtered


def _camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def label_and_time_process(df: pd.DataFrame, id_to_label: dict, fps: int) -> pd.DataFrame:
    """Map source_id to person labels and create time columns."""
    df = df.copy()
    df["person_label"]   = df["source_id"].map(id_to_label)
    df["time_s"]         = df["frame"] / fps
    df["time_min:s.ms"]  = df["time_s"].apply(
        lambda x: f"{int(x // 60):02}:{int(x % 60):02}.{int((x % 1) * 1000):03}"
    )
    return df


def filter_rows_with_no_keypoints(df: pd.DataFrame,
                                   keypoint_cols: list = KEYPOINT_COLS) -> pd.DataFrame:
    """Drop rows where all keypoint columns are empty or all-zero."""
    mask = df[keypoint_cols].map(_is_all_zero_or_empty).all(axis=1)
    return df.loc[~mask].copy()


def reshape_all_keypoints(df: pd.DataFrame,
                           keypoint_cols: list = KEYPOINT_COLS) -> pd.DataFrame:
    """Add reshaped numpy array columns for each keypoint column."""
    for col in keypoint_cols:
        df[f"{col}_array"] = df[col].apply(_fast_reshape)
    return df


def add_average_confidence(df: pd.DataFrame,
                            keypoints_col: str = "pose_keypoints_2d_array") -> pd.DataFrame:
    """Add average confidence score across all pose keypoints."""
    def avg_conf(arr):
        return np.nan if arr.size == 0 else float(np.nanmean(arr[:, 2]))
    df["avg_pose_conf"] = df[keypoints_col].apply(avg_conf)
    return df


def add_wrist_distance(df: pd.DataFrame,
                        keypoints_col: str = "pose_keypoints_2d_array") -> pd.DataFrame:
    """Add Euclidean distance between right wrist (4) and left wrist (7)."""
    df["wrist_dist"] = df[keypoints_col].apply(_wrist_distance)
    return df


def add_filtered_keypoints(df: pd.DataFrame,
                            keypoints_col: str = "pose_keypoints_2d_array") -> pd.DataFrame:
    """Add filtered keypoints column with low-confidence points masked as NaN."""
    df[f"{keypoints_col}_filtered"] = df[keypoints_col].apply(_filter_low_confidence)
    return df


def extract_all_joints(df: pd.DataFrame,
                        keypoints_col: str = "pose_keypoints_2d_array",
                        joint_names: dict = JOINT_NAMES) -> pd.DataFrame:
    """Extract all joints into separate <Name>_x and <Name>_y columns."""
    for idx, name in joint_names.items():
        coords = df[keypoints_col].apply(lambda arr: _get_joint_coords(arr, idx)).tolist()
        df[[f"{name}_x", f"{name}_y"]] = pd.DataFrame(coords, index=df.index)
    return df


def rename_joint_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """Rename joint columns from CamelCase (RShoulder_x) to snake_case (r_shoulder_x)."""
    rename_map = {}
    for col in df.columns:
        if any(col.endswith(s) for s in ("_x", "_y", "_conf")):
            base, _, suffix = col.rpartition("_")
            if base in JOINT_NAMES.values():
                rename_map[col] = f"{_camel_to_snake(base)}_{suffix}"
    return df.rename(columns=rename_map)


def drop_intermediate_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop intermediate processing columns that are no longer needed."""
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(existing, axis=1)


def process_pose_data(df: pd.DataFrame, id_to_label: dict, fps: int) -> pd.DataFrame:
    """
    Full processing pipeline for raw pose keypoint DataFrame.

    Steps:
        1.  Label persons and add time columns
        2.  Drop rows with no meaningful keypoints
        3.  Reshape keypoint JSON strings to numpy arrays
        4.  Add average confidence score
        5.  Compute wrist-to-wrist distance
        6.  Filter low-confidence keypoints
        7.  Extract all joints into x/y columns
        8.  Drop intermediate columns
        9.  Sort by time and person label
        10. Rename columns to snake_case
    """
    df = label_and_time_process(df, id_to_label, fps)
    df = filter_rows_with_no_keypoints(df)
    df = reshape_all_keypoints(df)
    df = add_average_confidence(df)
    df = add_wrist_distance(df)
    df = add_filtered_keypoints(df)
    df = extract_all_joints(df)
    df = drop_intermediate_columns(df, COLS_TO_DROP)
    df = df.sort_values(by=["time_s", "person_label"]).reset_index(drop=True)
    df = rename_joint_columns_to_snake_case(df)
    return df


# =============================================================================
# Stage 5 — Visualization
# =============================================================================

def compute_mass_center(df: pd.DataFrame, person_label: str) -> dict:
    """Compute center of mass (mean of visible keypoints) per frame for a person."""
    person_data = df[df["person_label"] == person_label].copy()
    centers = {}
    for _, row in person_data.iterrows():
        xs = [row[f"{j}_x"] for j in JOINT_NAMES_SNAKE
              if not np.isnan(row.get(f"{j}_x", np.nan))]
        ys = [row[f"{j}_y"] for j in JOINT_NAMES_SNAKE
              if not np.isnan(row.get(f"{j}_y", np.nan))]
        if xs and ys:
            centers[row["time_s"]] = (float(np.mean(xs)), float(np.mean(ys)))
    return centers


def plot_mass_centers(df: pd.DataFrame, id_to_label: dict,
                       save_path: Path = None, dpi: int = 300) -> None:
    """Plot center of mass X and Y trajectories over time for all persons."""
    all_centers = {
        label: compute_mass_center(df, label)
        for label in id_to_label.values()
    }

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    for label, centers in all_centers.items():
        if not centers:
            continue
        times  = sorted(centers.keys())
        x_vals = [centers[t][0] for t in times]
        y_vals = [centers[t][1] for t in times]
        axes[0].plot(times, x_vals, label=label)
        axes[1].plot(times, y_vals, label=label)

    for ax, title, ylabel in zip(
        axes,
        ["Center of Mass — X over Time", "Center of Mass — Y over Time"],
        ["X Position", "Y Position"]
    ):
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Plot saved: {save_path.name}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# Stage 6 — Cleanup
# =============================================================================

def cleanup(base_path: Path) -> None:
    """Remove all _extracted folders, merged_jsons/, and all_jsons/."""
    merged_dir = base_path / "merged_jsons"
    input_dir  = base_path / "all_jsons"

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        print(f"  Deleted: {merged_dir.name}")

    deleted = 0
    for folder in base_path.rglob("*_extracted"):
        if folder.is_dir():
            shutil.rmtree(folder)
            deleted += 1
    print(f"  Deleted {deleted} *_extracted folders.")

    if input_dir.exists():
        shutil.rmtree(input_dir)
        print(f"  Deleted: {input_dir.name}")


# =============================================================================
# Core entry point (importable by full_pipeline.py)
# =============================================================================

def run_pose_extraction(project_path: Path, cfg: dict) -> bool:
    """
    Run the full pose extraction pipeline for one project folder.

    Expects:
        project_path/PosesDir/  — contains .tar.gz archives from psifx

    Produces:
        project_path/PosesDir/processed_data.csv

    Args:
        project_path: Path to the project directory
        cfg:          Parsed config.yaml dict

    Returns:
        True on success, False on failure/skip.
    """
    ec    = cfg["pose_extraction"]
    directory  = ec["directory"]        # e.g. "PosesDir"
    id_to_label = ec["id_to_label"]
    fps         = ec.get("fps", cfg["model"]["fps"])
    abs_base    = ec.get("abs_base", 0)
    dpi         = cfg["plotting"].get("dpi", 300)

    base_path = project_path / directory
    if not base_path.exists():
        print(f"  Poses directory not found: {base_path} — skipping.")
        return False

    output_csv = base_path / "processed_data.csv"
    if cfg["flags"].get("skip_pose_extraction", False) and output_csv.exists():
        print(f"  Skipping pose extraction — processed_data.csv already exists.")
        return True

    project_name = project_path.name
    print(f"\n{'='*70}")
    print(f"Pose extraction: {project_name}")
    print(f"{'='*70}")

    # --- Stage 1: Extract & rename ---
    extract_and_rename(base_path, abs_base)

    # --- Stage 2: Collect ---
    input_dir = collect_jsons(base_path)

    # --- Stage 3: Build raw DataFrame ---
    raw_csv = base_path / f"combined_data_{directory}_{project_name}.csv"
    df_raw  = build_dataframe(input_dir)
    if df_raw.empty:
        print("  No data — skipping this project.")
        return False
    df_raw.to_csv(raw_csv, index=False)
    print(f"  Raw CSV saved: {raw_csv.name}")

    # --- Stage 4: Process ---
    df_processed = process_pose_data(df_raw, id_to_label, fps)
    print(f"  Processed shape: {df_processed.shape}")

    # --- Stage 5: Save ---
    df_processed.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv.name}")

    # --- Visualization ---
    plot_path = base_path / f"mass_centers_{project_name}.png"
    try:
        plot_mass_centers(df_processed, id_to_label,
                          save_path=plot_path, dpi=dpi)
    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")

    # --- Stage 6: Cleanup ---
    cleanup(base_path)

    return True


# =============================================================================
# Standalone entry point
# =============================================================================

def main():
    CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    raw_video_root = Path(cfg["directories"]["raw_video_root"])
    project_dirs   = [d for d in raw_video_root.iterdir() if d.is_dir()]
    print(f"Found {len(project_dirs)} project directories.")

    results = {}
    for project_path in sorted(project_dirs):
        success = run_pose_extraction(project_path, cfg)
        results[project_path.name] = "OK" if success else "FAILED/SKIPPED"

    print(f"\n{'='*70}")
    print("POSE EXTRACTION SUMMARY")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"  {status:15s}  {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding: utf-8
"""
extractor.py
------------
Step 2 of the Video Annotation Pipeline: Pose keypoint extraction.

Pipeline stages:
  1. Extract & Rename
  2. Collect all JSONs into one folder
  3. Build raw DataFrame from JSONs
  4. Process DataFrame (reshape, label, filter, extract joints)
  5. Save processed CSV
  6. Visualization of mass centers
  7. Cleanup temporary files

Can be run standalone or imported via full_pipeline.py using run_pose_extraction().
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
# Helpers
# =============================================================================

def _camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


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

# =============================================================================
# Stage 1 — Extract & rename
# =============================================================================

def extract_and_rename(base_path: Path, abs_base: int = 0) -> int:
    tar_files = sorted(base_path.glob("*.tar.gz"))
    if not tar_files:
        print("  No .tar.gz files found.")
        return 0

    total_renamed = 0
    for tar_file in tar_files:
        extract_dir = tar_file.with_name(f"{tar_file.stem}_extracted")
        extract_dir.mkdir(exist_ok=True)
        try:
            with tarfile.open(tar_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        member.name = Path(member.name).name
                        tar.extract(member, path=extract_dir)
        except Exception as e:
            print(f"  Error extracting {tar_file.name}: {e}")
            continue

        json_files = list(extract_dir.rglob("*.json"))
        if not json_files:
            continue

        video_id = tar_file.stem.split(".")[0]
        for jf in json_files:
            match = re.search(r"(\d+)$", jf.stem)
            if not match:
                continue
            abs_idx = abs_base + int(match.group(1))
            new_name = f"id_{video_id}_{abs_idx:05}.json"
            new_path = jf.with_name(new_name)
            if jf.name != new_name and not new_path.exists():
                jf.rename(new_path)
                total_renamed += 1

    print(f"  Total JSON files renamed: {total_renamed}")
    return total_renamed

# =============================================================================
# Stage 2 — Collect all JSONs
# =============================================================================

def collect_jsons(base_path: Path) -> Path:
    input_dir = base_path / "all_jsons"
    input_dir.mkdir(exist_ok=True)
    merged_dir = base_path / "merged_jsons"

    for root, dirs, files in os.walk(base_path):
        current = Path(root)
        if current in [input_dir, merged_dir]:
            continue
        for file in files:
            if file.endswith(".json"):
                dst = input_dir / file
                src = current / file
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)

    print(f"  All JSONs collected in: {input_dir}")
    return input_dir

# =============================================================================
# Stage 3 — Build raw DataFrame
# =============================================================================

def build_dataframe(input_dir: Path) -> pd.DataFrame:
    pattern = re.compile(r"(id_\d+)_(\d+)\.json")
    all_data = []

    for json_file in input_dir.glob("id_*.json"):
        match = pattern.match(json_file.name)
        if not match:
            continue
        source_id = match.group(1)
        frame = int(match.group(2))
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["source_id"] = source_id
                data["frame"] = frame
                all_data.append(data)
        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")

    df = pd.DataFrame(all_data)
    print(f"  DataFrame: {len(df):,} rows × {len(df.columns)} columns")
    return df

# =============================================================================
# Stage 4 — Process DataFrame
# =============================================================================

def process_pose_data(df: pd.DataFrame, id_to_label: dict, fps: int) -> pd.DataFrame:
    """Full processing pipeline for raw pose DataFrame."""
    # Label persons and add time
    df = df.copy()
    df["person_label"] = df["source_id"].map(id_to_label)
    df["time_s"] = df["frame"] / fps
    df["time_min:s.ms"] = df["time_s"].apply(lambda x: f"{int(x // 60):02}:{int(x % 60):02}.{int((x % 1)*1000):03}")

    # Filter empty rows
    mask = df[KEYPOINT_COLS].map(_is_all_zero_or_empty).all(axis=1)
    df = df.loc[~mask].copy()

    # Reshape keypoints
    for col in KEYPOINT_COLS:
        df[f"{col}_array"] = df[col].apply(_fast_reshape)

    # Average confidence & wrist distance
    df["avg_pose_conf"] = df["pose_keypoints_2d_array"].apply(lambda arr: np.nan if arr.size == 0 else float(np.nanmean(arr[:,2])))
    df["wrist_dist"] = df["pose_keypoints_2d_array"].apply(_wrist_distance)

    # Filter low-confidence keypoints
    df["pose_keypoints_2d_array_filtered"] = df["pose_keypoints_2d_array"].apply(_filter_low_confidence)

    # Extract joints
    for idx, name in JOINT_NAMES.items():
        coords = df["pose_keypoints_2d_array"].apply(lambda arr: _get_joint_coords(arr, idx)).tolist()
        df[[f"{name}_x", f"{name}_y"]] = pd.DataFrame(coords, index=df.index)

    # Drop intermediate columns
    df.drop([c for c in COLS_TO_DROP if c in df.columns], axis=1, inplace=True)

    # Rename joints to snake_case
    rename_map = {col: f"{_camel_to_snake(col.rsplit('_',1)[0])}_{col.rsplit('_',1)[1]}"
                  for col in df.columns if any(col.endswith(s) for s in ("_x","_y","_conf"))
                  and col.rsplit('_',1)[0] in JOINT_NAMES.values()}
    df.rename(columns=rename_map, inplace=True)

    return df.sort_values(by=["time_s","person_label"]).reset_index(drop=True)

# =============================================================================
# Stage 5 — Visualization
# =============================================================================

def compute_mass_center(df: pd.DataFrame, person_label: str) -> dict:
    person_data = df[df["person_label"] == person_label]
    centers = {}
    for _, row in person_data.iterrows():
        xs = [row[f"{j}_x"] for j in JOINT_NAMES_SNAKE if not np.isnan(row.get(f"{j}_x", np.nan))]
        ys = [row[f"{j}_y"] for j in JOINT_NAMES_SNAKE if not np.isnan(row.get(f"{j}_y", np.nan))]
        if xs and ys:
            centers[row["time_s"]] = (float(np.mean(xs)), float(np.mean(ys)))
    return centers

def plot_mass_centers(df: pd.DataFrame, id_to_label: dict, save_path: Path = None, dpi: int = 300):
    all_centers = {label: compute_mass_center(df, label) for label in id_to_label.values()}
    fig, axes = plt.subplots(2,1, figsize=(15,8))
    for label, centers in all_centers.items():
        if not centers: continue
        times = sorted(centers.keys())
        axes[0].plot(times, [centers[t][0] for t in times], label=label)
        axes[1].plot(times, [centers[t][1] for t in times], label=label)
    for ax, title, ylabel in zip(axes, ["X over time","Y over time"], ["X","Y"]):
        ax.set_title(f"Center of Mass — {title}")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.grid(True); ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

# =============================================================================
# Stage 6 — Cleanup
# =============================================================================

def cleanup(base_path: Path):
    for folder in base_path.rglob("*_extracted"):
        if folder.is_dir(): shutil.rmtree(folder)
    for f in ["all_jsons","merged_jsons"]:
        path = base_path / f
        if path.exists(): shutil.rmtree(path)

# =============================================================================
# Core entry point
# =============================================================================

def run_pose_extraction(project_path: Path, cfg: dict) -> bool:
    ec = cfg["pose_extraction"]
    base_path = project_path / ec["directory"]
    id_to_label = ec["id_to_label"]
    fps = ec.get("fps", cfg["model"]["fps"])
    abs_base = ec.get("abs_base", 0)
    dpi = cfg.get("plotting", {}).get("dpi", 300)

    if not base_path.exists():
        print(f"  Poses directory not found: {base_path} — skipping.")
        return False

    output_csv = base_path / "processed_data.csv"
    if cfg["flags"].get("skip_pose_extraction", False) and output_csv.exists():
        print("  Skipping pose extraction — CSV exists.")
        return True

    print(f"\n{'='*70}\nPose extraction: {project_path.name}\n{'='*70}")

    extract_and_rename(base_path, abs_base)
    input_dir = collect_jsons(base_path)
    df_raw = build_dataframe(input_dir)
    if df_raw.empty: return False
    df_processed = process_pose_data(df_raw, id_to_label, fps)
    df_processed.to_csv(output_csv, index=False)
    plot_mass_centers(df_processed, id_to_label, save_path=base_path/f"mass_centers_{project_path.name}.png", dpi=dpi)
    cleanup(base_path)
    print(f"  Pose extraction complete for {project_path.name}")
    return True

# =============================================================================
# Standalone entry point
# =============================================================================

def main():
    CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
    with open(CONFIG_FILE, "r") as f: cfg = yaml.safe_load(f)
    raw_video_root = Path(cfg["directories"]["raw_video_root"])
    project_dirs = [d for d in raw_video_root.iterdir() if d.is_dir()]
    results = {}
    for project_path in sorted(project_dirs):
        success = run_pose_extraction(project_path, cfg)
        results[project_path.name] = "OK" if success else "FAILED/SKIPPED"
    print("\n" + "="*70 + "\nPOSE EXTRACTION SUMMARY\n" + "="*70)
    for name,status in results.items(): print(f"  {status:15s}  {name}")

if __name__ == "__main__":
    main()

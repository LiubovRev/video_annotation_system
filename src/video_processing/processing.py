#!/usr/bin/env python3
# coding: utf-8
"""
processing.py
-------------
Step 1 of the Video Annotation Pipeline.
Handles per-project video preprocessing:
  - Trim and sample raw video with ffmpeg
  - Run SAMURAI tracking inference (psifx)
  - Run MediaPipe pose inference (psifx)
  - Visualize tracking and pose results
  - Clean up intermediate frames directory

Can be run standalone (processes a single project) or imported and called
by full_pipeline.py via run_video_processing(project_path, cfg).

Configuration is loaded from config.yaml at the project root.
"""

import subprocess
import shutil
import sys
import traceback
from pathlib import Path

import yaml


# =============================================================================
# Helpers
# =============================================================================

def _log(log_file_path: Path, message: str) -> None:
    with open(log_file_path, "a") as f:
        f.write(message + "\n")


def _run_cmd(cmd: list, step_desc: str, log_file_path: Path) -> subprocess.CompletedProcess:
    """Run a shell command, print output, log errors, exit on failure."""
    print(f"\n>>> {step_desc}")
    print("Command:", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Succeeded.")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {step_desc} failed (exit {e.returncode})")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        _log(log_file_path, f"[FAILED] {step_desc}: {e.stderr.strip()}")
        raise


# =============================================================================
# Core processing function (importable by full_pipeline.py)
# =============================================================================

def run_video_processing(project_path: Path, cfg: dict) -> bool:
    """
    Run the full video preprocessing pipeline for a single project folder.

    Args:
        project_path: Path to the project directory (e.g. .../15-5-2024_#20_INDIVIDUAL_[15])
        cfg:          Parsed config.yaml dict (yaml.safe_load output)

    Returns:
        True on success, False if the project was skipped or failed.
    """
    vc   = cfg["video_processing"]
    trim = cfg.get("trim_times", {})
    default_trim = cfg.get("default_trim", [0, None, 15])

    project_name     = project_path.name
    raw_video        = project_path / vc["raw_video_filename"]
    processed_video  = project_path / "processed_video.mp4"

    frames_dir = project_path / "Frames"
    mask_dir   = project_path / "MaskDir"
    poses_dir  = project_path / "PosesDir"
    vis_dir    = project_path / "Visualizations"
    log_file   = project_path / "processing_log.log"

    for d in [frames_dir, mask_dir, poses_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)
    log_file.write_text("=== Processing Log ===\n")

    print(f"\n{'='*70}")
    print(f"Video processing: {project_name}")
    print(f"{'='*70}")

    # --- Skip check ---
    skip_flag = cfg["flags"].get("skip_video_processing", False)
    if skip_flag and processed_video.exists():
        print(f"  Skipping — processed_video.mp4 already exists.")
        _log(log_file, f"[SKIPPED] Video processing: processed_video.mp4 exists")
        return True

    if not raw_video.exists():
        print(f"  Raw video not found: {raw_video} — skipping project.")
        _log(log_file, f"[SKIPPED] Raw video not found: {raw_video}")
        return False

    # --- Trim times ---
    start_sec, end_sec, fps = trim.get(project_name, default_trim)

    # --- Step 1: Trim and sample video ---
    if frames_dir.exists() and any(frames_dir.iterdir()):
        print(f"  Frames already exist — skipping frame extraction.")
        _log(log_file, f"[SKIPPED] Frame extraction: {frames_dir} already populated")
    else:
        if not processed_video.exists():
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(raw_video),
                "-ss", str(start_sec),
                *([ "-to", str(end_sec) ] if end_sec is not None else []),
                "-filter:v", f"fps={fps}",
                str(processed_video)
            ]
            try:
                _run_cmd(ffmpeg_cmd, "Trim and sample video", log_file)
                _log(log_file, f"[OK] Processed video created: {processed_video}")
            except Exception:
                return False
        else:
            print("  Processed video already exists — skipping ffmpeg trim.")
            _log(log_file, f"[OK] Processed video already exists: {processed_video}")

    # --- Step 2: Tracking inference ---
    try:
        _run_cmd([
            "psifx", "video", "tracking", "samurai", "inference",
            "--video",       str(processed_video),
            "--mask_dir",    str(mask_dir),
            "--model_size",  vc["model_size"],
            "--yolo_model",  vc["yolo_model"],
            "--max_objects", str(vc["nb_objects"]),
            "--step",        str(vc["step_size"]),
            "--device",      vc["device"],
            "--overwrite"
        ], "Tracking inference", log_file)
        _log(log_file, "[OK] Tracking inference completed")
    except Exception:
        return False

    # --- Step 2b: Tracking visualization ---
    vis_track = vis_dir / "visualization_tracking.mp4"
    try:
        _run_cmd([
            "psifx", "video", "tracking", "visualization",
            "--video",         str(processed_video),
            "--masks",         str(mask_dir),
            "--visualization", str(vis_track),
            "--overwrite"
        ], "Tracking visualization", log_file)
        _log(log_file, "[OK] Tracking visualization completed")
    except Exception as e:
        print(f"  Warning: Tracking visualization failed: {e}")
        _log(log_file, f"[WARNING] Tracking visualization failed: {e}")

    # --- Step 3: Pose inference ---
    try:
        _run_cmd([
            "psifx", "video", "pose", "mediapipe", "multi-inference",
            "--video",     str(processed_video),
            "--masks",     str(mask_dir),
            "--poses_dir", str(poses_dir),
            "--device",    vc["device"],
            "--overwrite"
        ], "Pose inference", log_file)
        _log(log_file, "[OK] Pose inference completed")
    except Exception:
        return False

    # --- Step 3b: Pose visualization ---
    vis_pose = vis_dir / "visualization_pose.mp4"
    try:
        _run_cmd([
            "psifx", "video", "pose", "mediapipe", "visualization",
            "--video",                str(processed_video),
            "--poses",                str(poses_dir),
            "--visualization",        str(vis_pose),
            "--confidence_threshold", "0.0",
            "--overwrite"
        ], "Pose visualization", log_file)
        _log(log_file, "[OK] Pose visualization completed")
    except Exception as e:
        print(f"  Warning: Pose visualization failed: {e}")
        _log(log_file, f"[WARNING] Pose visualization failed: {e}")

    # --- Step 4: Cleanup frames ---
    try:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            print(f"  Cleaned up frames directory.")
            _log(log_file, f"[OK] Cleaned up frames: {frames_dir}")
    except Exception as e:
        print(f"  Warning: Could not clean frames dir: {e}")
        _log(log_file, f"[WARNING] Frames cleanup failed: {e}")

    print(f"  Video processing complete. Results in: {project_path}")
    print(f"  Log: {log_file}")
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

    print(f"Found {len(project_dirs)} project directories in {raw_video_root}")

    results = {}
    for project_path in sorted(project_dirs):
        success = run_video_processing(project_path, cfg)
        results[project_path.name] = "OK" if success else "FAILED/SKIPPED"

    print(f"\n{'='*70}")
    print("VIDEO PROCESSING SUMMARY")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"  {status:15s}  {name}")


if __name__ == "__main__":
    main()

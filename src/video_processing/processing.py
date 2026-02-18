#!/usr/bin/env python3
# coding: utf-8
"""
processing.py
-------------
Step 1 of the Video Annotation Pipeline: Video preprocessing per project.

Features:
  - Trim and sample raw video with ffmpeg
  - Run SAMURAI tracking inference (psifx)
  - Run MediaPipe pose inference (psifx)
  - Visualize tracking and pose results
  - Clean up intermediate frames directory

Can be run standalone or imported by full_pipeline.py via run_video_processing().
"""

import subprocess
import shutil
from pathlib import Path
import yaml

# =============================================================================
# Helpers
# =============================================================================

def _log(log_file: Path, message: str) -> None:
    """Append a message to the project log file."""
    with open(log_file, "a") as f:
        f.write(message + "\n")


def _run_cmd(cmd: list, step_desc: str, log_file: Path) -> subprocess.CompletedProcess:
    """Run a shell command, print output, log errors, and raise on failure."""
    print(f"\n>>> {step_desc}")
    print("Command:", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Succeeded.")
        if result.stdout:
            print("STDOUT:", result.stdout.strip())
        if result.stderr:
            print("STDERR:", result.stderr.strip())
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_desc} failed (exit {e.returncode})")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        _log(log_file, f"[FAILED] {step_desc}: {e.stderr.strip()}")
        raise

# =============================================================================
# Core function
# =============================================================================

def run_video_processing(project_path: Path, cfg: dict, debug_keep_frames: bool = False) -> bool:
    """
    Run full video preprocessing for a single project.

    Args:
        project_path: Path to the project directory
        cfg: Parsed configuration dictionary
        debug_keep_frames: If True, skips deleting extracted frames

    Returns:
        True on success, False if processing failed or skipped.
    """
    vc = cfg["video_processing"]
    trim = cfg.get("trim_times", {})
    default_trim = cfg.get("default_trim", [0, None, vc.get("fps", 15)])

    project_name = project_path.name
    raw_video = project_path / vc["raw_video_filename"]
    processed_video = project_path / "processed_video.mp4"
    log_file = project_path / "processing_log.log"
    log_file.write_text(f"=== Processing Log for {project_name} ===\n")

    # Directory setup
    dirs = {
        "frames": project_path / "Frames",
        "masks":  project_path / "MaskDir",
        "poses":  project_path / "PosesDir",
        "visual": project_path / "Visualizations"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nVideo processing: {project_name}\n{'='*70}")

    # Skip if flagged
    if cfg["flags"].get("skip_video_processing", False) and processed_video.exists():
        print("  → Skipping — processed video already exists.")
        _log(log_file, "[SKIPPED] Processed video exists.")
        return True

    if not raw_video.exists():
        print(f"  ❌ Raw video not found: {raw_video}")
        _log(log_file, "[SKIPPED] Raw video missing.")
        return False

    start_sec, end_sec, fps = trim.get(project_name, default_trim)

    # --- Step 1: Trim & sample ---
    if dirs["frames"].exists() and any(dirs["frames"].iterdir()):
        print("  → Frames already exist — skipping frame extraction.")
        _log(log_file, "[SKIPPED] Frame extraction exists.")
    else:
        if not processed_video.exists():
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", str(raw_video),
                "-ss", str(start_sec),
                *(["-to", str(end_sec)] if end_sec is not None else []),
                "-filter:v", f"fps={fps}",
                str(processed_video)
            ]
            try:
                _run_cmd(ffmpeg_cmd, "Trim and sample video", log_file)
                _log(log_file, f"[OK] Processed video created: {processed_video.name}")
            except Exception:
                return False
        else:
            print("  → Processed video already exists — skipping ffmpeg trim.")
            _log(log_file, f"[OK] Processed video exists: {processed_video.name}")

    # --- Step 2: Tracking inference ---
    try:
        _run_cmd([
            "psifx", "video", "tracking", "samurai", "inference",
            "--video", str(processed_video),
            "--mask_dir", str(dirs["masks"]),
            "--model_size", vc["model_size"],
            "--yolo_model", vc["yolo_model"],
            "--max_objects", str(vc["nb_objects"]),
            "--step", str(vc["step_size"]),
            "--device", vc["device"],
            "--overwrite"
        ], "Tracking inference", log_file)
        _log(log_file, "[OK] Tracking inference completed.")
    except Exception:
        return False

    # --- Step 2b: Tracking visualization ---
    vis_track = dirs["visual"] / "visualization_tracking.mp4"
    try:
        _run_cmd([
            "psifx", "video", "tracking", "visualization",
            "--video", str(processed_video),
            "--masks", str(dirs["masks"]),
            "--visualization", str(vis_track),
            "--overwrite"
        ], "Tracking visualization", log_file)
        _log(log_file, "[OK] Tracking visualization completed.")
    except Exception as e:
        print(f"  ⚠ Warning: Tracking visualization failed: {e}")
        _log(log_file, f"[WARNING] Tracking visualization failed.")

    # --- Step 3: Pose inference ---
    try:
        _run_cmd([
            "psifx", "video", "pose", "mediapipe", "multi-inference",
            "--video", str(processed_video),
            "--masks", str(dirs["masks"]),
            "--poses_dir", str(dirs["poses"]),
            "--device", vc["device"],
            "--overwrite"
        ], "Pose inference", log_file)
        _log(log_file, "[OK] Pose inference completed.")
    except Exception:
        return False

    # --- Step 3b: Pose visualization ---
    vis_pose = dirs["visual"] / "visualization_pose.mp4"
    try:
        _run_cmd([
            "psifx", "video", "pose", "mediapipe", "visualization",
            "--video", str(processed_video),
            "--poses", str(dirs["poses"]),
            "--visualization", str(vis_pose),
            "--confidence_threshold", "0.0",
            "--overwrite"
        ], "Pose visualization", log_file)
        _log(log_file, "[OK] Pose visualization completed.")
    except Exception as e:
        print(f"  ⚠ Warning: Pose visualization failed: {e}")
        _log(log_file, "[WARNING] Pose visualization failed.")

    # --- Step 4: Cleanup frames ---
    if not debug_keep_frames:
        try:
            if dirs["frames"].exists():
                shutil.rmtree(dirs["frames"])
                print("  → Cleaned up frames directory.")
                _log(log_file, "[OK] Frames cleaned up.")
        except Exception as e:
            print(f"  ⚠ Warning: Could not clean frames directory: {e}")
            _log(log_file, "[WARNING] Frames cleanup failed.")

    print(f"✅ Video processing complete for project: {project_name}")
    print(f"  Log file: {log_file}")
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
    project_dirs = [d for d in raw_video_root.iterdir() if d.is_dir()]

    print(f"Found {len(project_dirs)} project directories in {raw_video_root}")

    results = {}
    for project_path in sorted(project_dirs):
        success = run_video_processing(project_path, cfg)
        results[project_path.name] = "OK" if success else "FAILED/SKIPPED"

    print(f"\n{'='*70}\nVIDEO PROCESSING SUMMARY\n{'='*70}")
    for name, status in results.items():
        print(f"  {status:15s}  {name}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ==========================================================
# ===================== CUSTOM SETTINGS ====================
# ==========================================================

DEVICE = "cuda"
TEXT_PROMPT = "people"
CHUNK_SIZE = 100
IOU_THRESHOLD = 0.3

MAX_OBJECTS: Optional[int] = 2  # uses PSIFX_MAX_OBJECTS env (requires your patched sam3 tool.py)

START_TRIM_SEC = 350
END_TRIM_SEC = int(15*60)

# Optional resize (helps GPU + often improves detection stability)
RESIZE_W: Optional[int] = None
RESIZE_H: Optional[int] = None

# Optional crop bbox (pixels)
X_MIN: Optional[int] = None
Y_MIN: Optional[int] = None
X_MAX: Optional[int] = None
Y_MAX: Optional[int] = None

BASE_PATH = Path("/home/liubov/Bureau/new/29-10-2024_#2_INDIVIDUAL_[83]")
RAW_VIDEO = BASE_PATH / "camera_a.mkv"
PROCESSED_VIDEO = BASE_PATH / "processed_video.mp4"

FRAMES_DIR = BASE_PATH / "Frames"
MASK_DIR = BASE_PATH / "MaskDir"
POSES_DIR = BASE_PATH / "PosesDir"
FACES_DIR = BASE_PATH / "FacesDir"
VIS_DIR = BASE_PATH / "Visualizations"
LOG_FILE = BASE_PATH / "processing_log.log"

OVERWRITE_OUTPUT_DIRS = True
SKIP_IF_PROCESSED_EXISTS = True
ENABLE_FACE = False


# ==========================================================
# ===================== UTILITIES ==========================
# ==========================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def init_log() -> None:
    ensure_dir(LOG_FILE.parent)
    LOG_FILE.write_text("=== Processing Log ===\n", encoding="utf-8")

def log_to_file(message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def dir_is_empty(p: Path) -> bool:
    return (not p.exists()) or (p.is_dir() and next(p.iterdir(), None) is None)

def wipe_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def build_env() -> dict:
    env = os.environ.copy()
    if "HUGGINGFACE_HUB_TOKEN" in env and "HF_TOKEN" not in env:
        env["HF_TOKEN"] = env["HUGGINGFACE_HUB_TOKEN"]
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if MAX_OBJECTS is None:
        env.pop("PSIFX_MAX_OBJECTS", None)
    else:
        env["PSIFX_MAX_OBJECTS"] = str(MAX_OBJECTS)

    return env

def run_cmd(cmd: list[str], step_desc: str, env: dict) -> subprocess.CompletedProcess:
    print(f"\n>>> {step_desc}")
    print("Running command:", " ".join(map(str, cmd)))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        if res.stdout.strip():
            print(res.stdout)
        if res.stderr.strip():
            print(res.stderr)
        log_to_file(f"[OK] {step_desc}")
        return res
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {step_desc} failed with exit code {e.returncode}")
        if e.stdout:
            print("\n--- STDOUT ---\n", e.stdout)
        if e.stderr:
            print("\n--- STDERR ---\n", e.stderr)
        log_to_file(f"[FAILED] {step_desc}: {(e.stderr or '').strip()}")
        raise

def ffprobe_has_video_frames(video_path: Path, env: dict) -> bool:
    """
    Quick sanity check: does the file have video stream + frames?
    """
    if not video_path.exists():
        return False
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,avg_frame_rate,width,height",
        "-of", "default=nw=1",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        out = res.stdout or ""
        # If nb_frames is missing for some codecs, still accept if width/height exist.
        return ("width=" in out and "height=" in out)
    except Exception:
        return False


# ==========================================================
# ===================== FFMPEG COMMANDS ====================
# ==========================================================

def build_ffmpeg_processed_cmd() -> list[str]:
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", str(START_TRIM_SEC),
        "-to", str(END_TRIM_SEC),
        "-i", str(RAW_VIDEO),
        "-map", "0:0",   # Azure Kinect: COLOR stream only
        "-an",
    ]

    vf_parts: list[str] = []
    if all(v is not None for v in (X_MIN, Y_MIN, X_MAX, Y_MAX)):
        crop_w = int(X_MAX) - int(X_MIN)
        crop_h = int(Y_MAX) - int(Y_MIN)
        vf_parts.append(f"crop={crop_w}:{crop_h}:{int(X_MIN)}:{int(Y_MIN)}")

    if RESIZE_W is not None or RESIZE_H is not None:
        w = RESIZE_W if RESIZE_W is not None else -1
        h = RESIZE_H if RESIZE_H is not None else -1
        vf_parts.append(f"scale={w}:{h}")

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        str(PROCESSED_VIDEO),
    ]
    return cmd

def build_ffmpeg_frames_cmd() -> list[str]:
    """
    Re-introduce your old "Frames" logic: extract jpg frames from the trimmed window.
    """
    out_pattern = str(FRAMES_DIR / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", str(START_TRIM_SEC),
        "-to", str(END_TRIM_SEC),
        "-i", str(RAW_VIDEO),
        "-map", "0:0",
        "-vf", "fps=15",
        out_pattern,
    ]
    return cmd


# ==========================================================
# ===================== MAIN PIPELINE ======================
# ==========================================================

def main() -> int:
    for d in [FRAMES_DIR, MASK_DIR, POSES_DIR, FACES_DIR, VIS_DIR]:
        ensure_dir(d)

    init_log()
    log_to_file(f"Input: {RAW_VIDEO}")
    log_to_file(f"Processed: {PROCESSED_VIDEO}")
    log_to_file(f"ENABLE_FACE: {ENABLE_FACE}")
    log_to_file(f"MAX_OBJECTS: {MAX_OBJECTS}")
    log_to_file(f"CHUNK_SIZE: {CHUNK_SIZE}")
    log_to_file(f"DEVICE: {DEVICE}")
    log_to_file(f"TEXT_PROMPT: {TEXT_PROMPT}")

    if not RAW_VIDEO.exists():
        print(f"[ERROR] Input video not found: {RAW_VIDEO}")
        log_to_file(f"[FAILED] Input video missing: {RAW_VIDEO}")
        return 2

    env = build_env()

    # Overwrite dirs (sam3 requires empty MaskDir)
    if OVERWRITE_OUTPUT_DIRS:
        wipe_dir(MASK_DIR)
        wipe_dir(POSES_DIR)
        wipe_dir(VIS_DIR)
        if ENABLE_FACE:
            wipe_dir(FACES_DIR)

    if not dir_is_empty(MASK_DIR):
        print(f"[ERROR] MaskDir must be empty for sam3: {MASK_DIR}")
        log_to_file(f"[FAILED] MaskDir not empty: {MASK_DIR}")
        return 3

    # === Frames logic restored ===
    if FRAMES_DIR.exists() and any(FRAMES_DIR.iterdir()):
        print(f"\n>>> Frames already exist (skip extraction): {FRAMES_DIR}")
        log_to_file(f"[SKIPPED] Frame extraction skipped: {FRAMES_DIR} already populated")
    else:
        run_cmd(build_ffmpeg_frames_cmd(), "Frame extraction (trim window -> Frames/)", env)

    # === processed_video.mp4 creation (only if needed) ===
    if PROCESSED_VIDEO.exists() and SKIP_IF_PROCESSED_EXISTS:
        print(f"\n>>> Skipping processed video creation (exists): {PROCESSED_VIDEO}")
        log_to_file("[SKIPPED] processed_video.mp4 exists, ffmpeg skipped")
    else:
        run_cmd(build_ffmpeg_processed_cmd(), "FFmpeg trim -> processed_video.mp4", env)

    if not ffprobe_has_video_frames(PROCESSED_VIDEO, env):
        print(f"[ERROR] processed_video.mp4 seems invalid or has no video stream: {PROCESSED_VIDEO}")
        log_to_file("[FAILED] processed_video.mp4 invalid (ffprobe failed)")
        return 4

    # === Tracking inference (SAM3) ===
    tracking_cmd = [
        "psifx", "video", "tracking", "sam3", "inference",
        "--video", str(PROCESSED_VIDEO),
        "--mask_dir", str(MASK_DIR),
        "--text_prompt", TEXT_PROMPT,
        "--chunk_size", str(CHUNK_SIZE),
        "--iou_threshold", str(IOU_THRESHOLD),
        "--device", DEVICE,
    ]

    run_cmd(tracking_cmd, "Tracking inference (sam3)", env)

    # If no masks produced, retry with a slightly different prompt (one retry only).
    if dir_is_empty(MASK_DIR):
        log_to_file("[WARNING] MaskDir empty after first tracking run; retrying with prompt='person' and smaller chunk.")
        print("\n[WARN] MaskDir empty. Retrying tracking once with prompt='person' and chunk_size=50...")

        retry_cmd = tracking_cmd.copy()
        # replace prompt + chunk
        retry_cmd[retry_cmd.index("--text_prompt") + 1] = "person"
        retry_cmd[retry_cmd.index("--chunk_size") + 1] = "50"

        run_cmd(retry_cmd, "Tracking inference retry (sam3)", env)

    if dir_is_empty(MASK_DIR):
        print(f"[ERROR] Tracking finished but MaskDir is empty: {MASK_DIR}")
        log_to_file("[FAILED] Tracking produced no masks (after retry).")
        return 6

    # === Tracking Visualization ===
    vis_track = VIS_DIR / "visualization_tracking.mp4"
    track_vis_cmd = [
        "psifx", "video", "tracking", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--visualization", str(vis_track),
        "--labels",
        "--color",
    ]
    run_cmd(track_vis_cmd, "Tracking visualization", env)

    # === Pose inference ===
    pose_inf_cmd = [
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--poses_dir", str(POSES_DIR),
        "--device", DEVICE,
    ]
    run_cmd(pose_inf_cmd, "Pose inference (mediapipe multi-inference)", env)

    # === Pose visualization ===
    vis_pose = VIS_DIR / "visualization_pose.mp4"
    pose_vis_cmd = [
        "psifx", "video", "pose", "mediapipe", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--poses", str(POSES_DIR),
        "--visualization", str(vis_pose),
        "--confidence_threshold", "0.0",
    ]
    try:
        run_cmd(pose_vis_cmd, "Pose visualization", env)
    except Exception as e:
        log_to_file(f"[WARNING] Pose visualization failed: {e}")

    # === Face optional (unchanged) ===
    if ENABLE_FACE:
        face_cmd = [
            "psifx", "video", "face", "openface", "multi-inference",
            "--video", str(PROCESSED_VIDEO),
            "--masks", str(MASK_DIR),
            "--features_dir", str(FACES_DIR),
            "--device", "cpu",
        ]
        try:
            run_cmd(face_cmd, "Face inference (openface)", env)
        except Exception as e:
            log_to_file(f"[WARNING] Face inference failed: {e}")

    # === Cleanup Frames directory (kept) ===
    try:
        if FRAMES_DIR.exists():
            print(f"\n>>> Cleaning up Frames directory: {FRAMES_DIR}")
            shutil.rmtree(FRAMES_DIR)
            log_to_file(f"[OK] Cleaned up Frames directory: {FRAMES_DIR}")
    except Exception as e:
        log_to_file(f"[WARNING] Failed to clean up Frames directory: {e}")

    print("\n==================== Processing Complete ====================")
    print(f"Results saved to: {BASE_PATH}")
    print(f"See {LOG_FILE} for details.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        try:
            log_to_file(f"[FAILED] Unhandled exception: {e}")
        except Exception:
            pass
        print("\n[ERROR] Unhandled exception:", e)
        sys.exit(1)

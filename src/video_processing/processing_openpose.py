#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import logging
import getpass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure OpenPose is installed and in your Python path
import pyopenpose as op

# ==========================================================
# ===================== SETTINGS ===========================
# ==========================================================

DEVICE = "cuda"
TEXT_PROMPT = "person"
CHUNK_SIZE = 300
IOU_THRESHOLD = 0.15
MAX_OBJECTS: Optional[int] = 5  # allow small subjects

START_TRIM_SEC = 0
END_TRIM_SEC = int(10 * 60)

X_MIN, Y_MIN = 40, 40
X_MAX, Y_MAX = 1240, 700
RESIZE_W, RESIZE_H = None, None

BASE_PATH = Path("/home/liubov/Bureau/new/29-10-2024_#2_INDIVIDUAL_[83]")
RAW_VIDEO = BASE_PATH / "camera_b.mkv"
PROCESSED_VIDEO = BASE_PATH / "processed_video.mp4"

MASK_DIR = BASE_PATH / "MaskDir"
POSES_DIR = BASE_PATH / "PosesDir"
VIS_DIR = BASE_PATH / "Visualizations"
LOG_FILE = BASE_PATH / "processing_log.log"

# Temporal smoothing parameters
CONFIDENCE_THRESHOLD = 0.3
SMOOTH_ALPHA = 0.6

OPENPOSE_MODEL_FOLDER = "/path/to/openpose/models/"  # replace with your path

# ==========================================================
# ===================== TOKEN INPUT ========================
# ==========================================================

def ensure_hf_token():
    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
        return
    print("\nHuggingFace token required.")
    token = getpass.getpass("Enter HF_TOKEN (input hidden): ").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required.")
    os.environ["HF_TOKEN"] = token

# ==========================================================
# ===================== LOGGING ============================
# ==========================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    )
    logging.info("=" * 60)
    logging.info("PIPELINE STARTED")
    logging.info("=" * 60)

def log_configuration():
    logging.info("PARAMETERS")
    logging.info("DEVICE=%s | TEXT_PROMPT=%s | MAX_OBJECTS=%s", DEVICE, TEXT_PROMPT, MAX_OBJECTS)
    logging.info("TRIM=%s-%s sec", START_TRIM_SEC, END_TRIM_SEC)
    logging.info("CROP=(%s,%s)-(%s,%s)", X_MIN, Y_MIN, X_MAX, Y_MAX)
    logging.info("=" * 60)

# ==========================================================
# ===================== UTILITIES ==========================
# ==========================================================

def run(cmd, name, env):
    logging.info("START STEP: %s", name)
    start = time.time()
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    duration = time.time() - start
    if r.returncode != 0:
        logging.error("STEP FAILED: %s", name)
        raise RuntimeError(name)
    logging.info("END STEP: %s (%.2f sec)", name, duration)
    logging.info("-" * 60)

def wipe(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def build_env():
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if MAX_OBJECTS:
        env["PSIFX_MAX_OBJECTS"] = str(MAX_OBJECTS)
    if "HF_TOKEN" in env:
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]
    return env

def build_ffmpeg():
    crop = f"crop={X_MAX-X_MIN}:{Y_MAX-Y_MIN}:{X_MIN}:{Y_MIN}"
    vf = [crop]
    if RESIZE_W or RESIZE_H:
        vf.append(f"scale={RESIZE_W or -1}:{RESIZE_H or -1}")
    return [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", str(START_TRIM_SEC),
        "-to", str(END_TRIM_SEC),
        "-i", str(RAW_VIDEO),
        "-map", "0:0",
        "-vf", ",".join(vf),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(PROCESSED_VIDEO)
    ]

# ==========================================================
# ===================== SKELETON ROBUSTNESS =================
# ==========================================================

def smooth_keypoints(keypoints_seq, alpha=SMOOTH_ALPHA):
    smoothed = keypoints_seq.copy()
    for i in range(1, len(keypoints_seq)):
        prev, curr = smoothed[i-1], smoothed[i]
        mask = curr[:,2] >= CONFIDENCE_THRESHOLD
        smoothed[i][mask, :2] = alpha * prev[mask, :2] + (1-alpha) * curr[mask, :2]
        smoothed[i][mask, 2] = curr[mask, 2]
    return smoothed

def fill_missing_keypoints(keypoints_seq):
    seq = keypoints_seq.copy()
    frames, joints, _ = seq.shape
    for j in range(joints):
        conf = seq[:, j, 2]
        good_idx = np.where(conf >= CONFIDENCE_THRESHOLD)[0]
        if len(good_idx) == 0:
            continue
        for i in range(frames):
            if conf[i] < CONFIDENCE_THRESHOLD:
                before = good_idx[good_idx < i]
                after = good_idx[good_idx > i]
                if len(before) == 0:
                    seq[i, j, :2] = seq[after[0], j, :2]
                elif len(after) == 0:
                    seq[i, j, :2] = seq[before[-1], j, :2]
                else:
                    b_idx, a_idx = before[-1], after[0]
                    t = (i - b_idx) / (a_idx - b_idx)
                    seq[i, j, :2] = (1-t)*seq[b_idx, j, :2] + t*seq[a_idx, j, :2]
                seq[i, j, 2] = 0.1
    return seq

def run_openpose(video_path: Path, output_dir: Path):
    """Run OpenPose on a video and save keypoints as NPZ."""
    params = dict()
    params["model_folder"] = OPENPOSE_MODEL_FOLDER
    params["hand"] = False
    params["face"] = False
    params["disable_blending"] = True
    params["render_pose"] = 0

    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    cap = cv2.VideoCapture(str(video_path))
    keypoints_seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        datum = op.Datum()
        datum.cvInputData = frame
        op_wrapper.emplaceAndPop([datum])

        pose_keypoints = datum.poseKeypoints
        if pose_keypoints is None:
            # Assume 1 person, all zeros
            keypoints_seq.append(np.zeros((1, 25, 3)))
        else:
            keypoints_seq.append(pose_keypoints)

    keypoints_seq = np.array(keypoints_seq)  # shape: (frames, num_people, joints, 3)
    # For simplicity, only process the first detected person
    first_person_seq = keypoints_seq[:,0,:,:]
    # Save raw keypoints
    np.savez(output_dir / "keypoints_openpose.npz", keypoints=first_person_seq)
    cap.release()
    logging.info("OpenPose inference completed.")

def postprocess_poses(poses_dir: Path):
    """Smooth and fill missing joints."""
    for f in poses_dir.glob("*.npz"):
        data = np.load(f)
        keypoints = data["keypoints"]  # (frames, joints, 3)
        keypoints = smooth_keypoints(keypoints)
        keypoints = fill_missing_keypoints(keypoints)
        np.savez(f.with_name(f.stem + "_robust.npz"), keypoints=keypoints)
    logging.info("Pose post-processing completed.")

# ==========================================================
# ===================== PIPELINE ===========================
# ==========================================================

def main():
    ensure_hf_token()
    setup_logging()
    log_configuration()
    env = build_env()

    wipe(MASK_DIR)
    wipe(POSES_DIR)
    wipe(VIS_DIR)

    # --- Video trimming & cropping ---
    run(build_ffmpeg(), "Create processed video", env)

    # --- SAM3 tracking ---
    run([
        "psifx", "video", "tracking", "sam3", "inference",
        "--video", str(PROCESSED_VIDEO),
        "--mask_dir", str(MASK_DIR),
        "--text_prompt", TEXT_PROMPT,
        "--chunk_size", str(CHUNK_SIZE),
        "--iou_threshold", str(IOU_THRESHOLD),
        "--device", DEVICE,
    ], "SAM3 tracking", env)

    # --- Tracking visualization ---
    run([
        "psifx", "video", "tracking", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--visualization", str(VIS_DIR / "tracking.mp4"),
        "--labels", "--color"
    ], "Tracking visualization", env)

    # --- OpenPose pose inference ---
    run_openpose(PROCESSED_VIDEO, POSES_DIR)

    # --- Pose post-processing: smoothing & missing joints ---
    postprocess_poses(POSES_DIR)

    # --- Pose visualization ---
    run([
        "psifx", "video", "pose", "mediapipe", "visualization",  # or use psifx visualization for OpenPose keypoints
        "--video", str(PROCESSED_VIDEO),
        "--poses", str(POSES_DIR),
        "--visualization", str(VIS_DIR / "pose.mp4"),
        "--confidence_threshold", "0.0"
    ], "Pose visualization", env)

    logging.info("PIPELINE FINISHED SUCCESSFULLY")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
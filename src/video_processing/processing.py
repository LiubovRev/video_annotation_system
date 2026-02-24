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

# ==========================================================
# ===================== SETTINGS ===========================
# ==========================================================

DEVICE = "cuda"

TEXT_PROMPT = "person"
CHUNK_SIZE = 300
IOU_THRESHOLD = 0.15
MAX_OBJECTS: Optional[int] = 3

START_TRIM_SEC = 260
END_TRIM_SEC = int(15 * 60)

# ----------------------------------------------------------
# GLOBAL SCENE CROP
# ----------------------------------------------------------

X_MIN = 40
Y_MIN = 40
X_MAX = 1240
Y_MAX = 700

RESIZE_W = None
RESIZE_H = None

POSE_MASK_THRESHOLD = "0.0"
POSE_MODEL_COMPLEXITY = "2"

BASE_PATH = Path(
    "/home/liubov/Bureau/new/29-10-2024_#2_INDIVIDUAL_[83]"
)

RAW_VIDEO = BASE_PATH / "camera_a.mkv"
PROCESSED_VIDEO = BASE_PATH / "processed_video.mp4"

MASK_DIR = BASE_PATH / "MaskDir"
POSES_DIR = BASE_PATH / "PosesDir"
VIS_DIR = BASE_PATH / "Visualizations"
LOG_FILE = BASE_PATH / "processing_log.log"


# ==========================================================
# ===================== TOKEN INPUT ========================
# ==========================================================

def ensure_hf_token():
    """
    Ask user for HF_TOKEN if not present in environment.
    """
    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
        return

    print("\n HuggingFace token required.")
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
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 80)
    logging.info("PIPELINE STARTED")
    logging.info("=" * 80)


def log_configuration():
    logging.info("CONFIGURATION PARAMETERS")
    logging.info("-" * 80)

    logging.info("DEVICE=%s", DEVICE)
    logging.info("TEXT_PROMPT=%s", TEXT_PROMPT)
    logging.info("CHUNK_SIZE=%s", CHUNK_SIZE)
    logging.info("IOU_THRESHOLD=%s", IOU_THRESHOLD)
    logging.info("MAX_OBJECTS=%s", MAX_OBJECTS)

    logging.info("START_TRIM_SEC=%s", START_TRIM_SEC)
    logging.info("END_TRIM_SEC=%s", END_TRIM_SEC)

    logging.info(
        "CROP=(%s,%s) -> (%s,%s)",
        X_MIN, Y_MIN, X_MAX, Y_MAX
    )

    logging.info("POSE_MASK_THRESHOLD=%s", POSE_MASK_THRESHOLD)
    logging.info("POSE_MODEL_COMPLEXITY=%s", POSE_MODEL_COMPLEXITY)

    logging.info("BASE_PATH=%s", BASE_PATH)
    logging.info("RAW_VIDEO=%s", RAW_VIDEO)

    logging.info("-" * 80)


# ==========================================================
# ===================== UTILITIES ==========================
# ==========================================================

def run(cmd, name, env):
    logging.info("START STEP: %s", name)
    logging.info("COMMAND: %s", " ".join(map(str, cmd)))

    start = time.time()

    r = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    duration = time.time() - start

    if r.stdout:
        logging.info("STDOUT:\n%s", r.stdout)

    if r.stderr:
        logging.warning("STDERR:\n%s", r.stderr)

    if r.returncode != 0:
        logging.error("STEP FAILED: %s", name)
        raise RuntimeError(name)

    logging.info("END STEP: %s (%.2f sec)", name, duration)
    logging.info("-" * 80)


def wipe(p: Path):
    if p.exists():
        logging.info("Removing directory: %s", p)
        shutil.rmtree(p)

    p.mkdir(parents=True, exist_ok=True)
    logging.info("Created directory: %s", p)


def build_env():
    env = os.environ.copy()

    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True"
    )

    if MAX_OBJECTS:
        env["PSIFX_MAX_OBJECTS"] = str(MAX_OBJECTS)

    if "HF_TOKEN" in env:
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]

    return env


# ==========================================================
# ===================== FFMPEG =============================
# ==========================================================

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

    run(build_ffmpeg(), "Create processed video", env)

    run([
        "psifx", "video", "tracking", "sam3", "inference",
        "--video", str(PROCESSED_VIDEO),
        "--mask_dir", str(MASK_DIR),
        "--text_prompt", TEXT_PROMPT,
        "--chunk_size", str(CHUNK_SIZE),
        "--iou_threshold", str(IOU_THRESHOLD),
        "--device", DEVICE,
    ], "SAM3 tracking", env)

    run([
        "psifx", "video", "tracking", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--visualization",
        str(VIS_DIR / "tracking.mp4"),
        "--labels", "--color"
    ], "Tracking visualization", env)

    run([
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--poses_dir", str(POSES_DIR),
        "--mask_threshold", POSE_MASK_THRESHOLD,
        "--model_complexity", POSE_MODEL_COMPLEXITY,
        "--smooth",
        "--device", DEVICE
    ], "Pose inference", env)

    run([
        "psifx", "video", "pose", "mediapipe", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--poses", str(POSES_DIR),
        "--visualization",
        str(VIS_DIR / "pose.mp4"),
        "--confidence_threshold", "0.0"
    ], "Pose visualization", env)

    logging.info("PIPELINE FINISHED SUCCESSFULLY")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

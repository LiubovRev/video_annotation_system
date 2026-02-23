#!/usr/bin/env python3
"""
psifx pipeline

- Overwrite outputs: if folders already contain files, wipe them before running.
- Skip creating processed_video.mp4 if it already exists.

Important:
- Tracking visualization flags must be passed WITHOUT "True"/"False": use --labels --color
- If you patched psifx Sam3 tensor->numpy bug in psifx/video/tracking/sam3/tool.py, keep that patch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ==========================================================
# ===================== CONFIGURATION ======================
# ==========================================================

@dataclass(frozen=True)
class CropParams:
    # Crop bbox (pixels). Set all to None to disable.
    x_min: Optional[int] = None
    y_min: Optional[int] = None
    x_max: Optional[int] = None
    y_max: Optional[int] = None
    # Optional resize
    width: Optional[int] = None
    height: Optional[int] = None

    def enabled(self) -> bool:
        return all(v is not None for v in (self.x_min, self.y_min, self.x_max, self.y_max))


# ---- General ----
DEVICE = "cuda"
TEXT_PROMPT = "people"
CHUNK_SIZE = 300
IOU_THRESHOLD = 0.3

# ---- Time window (seconds) ----
START_TRIM_SEC = 0
END_TRIM_SEC = 100000

# ---- Optional crop/resize (ffmpeg) ----
CROP = CropParams(
    x_min=None, y_min=None, x_max=None, y_max=None,
    width=None, height=None,
)

# ---- Paths ----
BASE_PATH = Path("/home/liubov/Bureau/new/29-10-2024_#2_INDIVIDUAL_[83]")
RAW_VIDEO = BASE_PATH / "camera_a.mkv"
PROCESSED_VIDEO = BASE_PATH / "processed_video.mp4"

MASK_DIR = BASE_PATH / "MaskDir"
POSES_DIR = BASE_PATH / "PosesDir"
FACES_DIR = BASE_PATH / "FacesDir"
VIS_DIR = BASE_PATH / "Visualizations"
LOG_FILE = BASE_PATH / "processing_log.log"

# ---- Behavior ----
OVERWRITE_OUTPUT_DIRS = True        # wipe MaskDir/PosesDir/FacesDir/Visualizations before running
SKIP_IF_PROCESSED_EXISTS = True     # do NOT recreate processed_video.mp4 if it exists


# ==========================================================
# ===================== UTILITIES ==========================
# ==========================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def init_log(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text("=== Processing Log ===\n", encoding="utf-8")


def log_line(path: Path, msg: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def dir_is_empty(p: Path) -> bool:
    return (not p.exists()) or (p.is_dir() and next(p.iterdir(), None) is None)


def wipe_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], description: str) -> None:
    print(f"\n>>> {description}")
    print("Running:", " ".join(cmd))

    # ensure HF token passes through to subprocesses
    env = os.environ.copy()
    if "HUGGINGFACE_HUB_TOKEN" in env and "HF_TOKEN" not in env:
        env["HF_TOKEN"] = env["HUGGINGFACE_HUB_TOKEN"]

    try:
        res = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
        if res.stdout.strip():
            print(res.stdout)
        if res.stderr.strip():
            # warnings often go to stderr even on success
            print(res.stderr)
        log_line(LOG_FILE, f"[OK] {description}")
    except subprocess.CalledProcessError as e:
        log_line(LOG_FILE, f"[FAILED] {description} (exit={e.returncode})")
        print("\n[ERROR] Command failed.")
        print("Return code:", e.returncode)
        if e.stdout:
            print("\n--- STDOUT ---\n", e.stdout)
        if e.stderr:
            print("\n--- STDERR ---\n", e.stderr)
        raise


# ==========================================================
# ===================== FFMPEG STEP ========================
# ==========================================================

def build_ffmpeg_cmd() -> list[str]:
    """
    Robust for Azure Kinect MKV (attachments): map only COLOR stream 0:0.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", str(START_TRIM_SEC),
        "-to", str(END_TRIM_SEC),
        "-i", str(RAW_VIDEO),
        "-map", "0:0",  # COLOR only
        "-an",
    ]

    vf_parts: list[str] = []
    if CROP.enabled():
        crop_w = int(CROP.x_max) - int(CROP.x_min)
        crop_h = int(CROP.y_max) - int(CROP.y_min)
        vf_parts.append(f"crop={crop_w}:{crop_h}:{int(CROP.x_min)}:{int(CROP.y_min)}")

    if CROP.width is not None or CROP.height is not None:
        w = CROP.width if CROP.width is not None else -1
        h = CROP.height if CROP.height is not None else -1
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


# ==========================================================
# ===================== CLI ================================
# ==========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="psifx video pipeline")
    p.add_argument(
        "--face",
        action="store_true",
        help="Enable OpenFace face feature extraction + visualization (default: off).",
    )
    p.add_argument(
        "--no-overwrite-dirs",
        action="store_true",
        help="Do NOT wipe output folders before running (default: wipe).",
    )
    p.add_argument(
        "--rebuild-processed",
        action="store_true",
        help="Force rebuilding processed_video.mp4 even if it exists.",
    )
    return p.parse_args()


# ==========================================================
# ===================== PIPELINE ===========================
# ==========================================================

def main() -> int:
    args = parse_args()
    enable_face = bool(args.face)

    overwrite_dirs = not args.no_overwrite_dirs
    skip_processed_if_exists = not args.rebuild_processed

    if not RAW_VIDEO.exists():
        print(f"[ERROR] Input video not found: {RAW_VIDEO}")
        return 2

    ensure_dir(MASK_DIR)
    ensure_dir(POSES_DIR)
    ensure_dir(FACES_DIR)
    ensure_dir(VIS_DIR)
    init_log(LOG_FILE)

    log_line(LOG_FILE, f"Input: {RAW_VIDEO}")
    log_line(LOG_FILE, f"Processed: {PROCESSED_VIDEO}")
    log_line(LOG_FILE, f"Face enabled: {enable_face}")

    # Overwrite output dirs if requested (required for SAM3: mask_dir must be empty)
    if overwrite_dirs:
        wipe_dir(MASK_DIR)
        wipe_dir(POSES_DIR)
        wipe_dir(VIS_DIR)
        if enable_face:
            wipe_dir(FACES_DIR)
        else:
            # keep FacesDir untouched when face is disabled
            ensure_dir(FACES_DIR)

    # SAM3 requires empty mask_dir; enforce if user disabled overwrite
    if not dir_is_empty(MASK_DIR):
        raise SystemExit(
            f"[ERROR] Mask directory must be empty for sam3.\n"
            f"Either delete contents or run without --no-overwrite-dirs.\n"
            f"MaskDir: {MASK_DIR}"
        )

    # STEP 1: processed_video.mp4
    if PROCESSED_VIDEO.exists() and skip_processed_if_exists:
        print(f"\n>>> Skipping ffmpeg (processed video exists): {PROCESSED_VIDEO}")
        log_line(LOG_FILE, "[OK] Skipped ffmpeg (processed video exists)")
    else:
        run_cmd(build_ffmpeg_cmd(), "FFmpeg processing (trim + map 0:0)")

    # STEP 2: Tracking (Sam3)
    tracking_cmd = [
        "psifx", "video", "tracking", "sam3", "inference",
        "--video", str(PROCESSED_VIDEO),
        "--mask_dir", str(MASK_DIR),
        "--text_prompt", TEXT_PROMPT,
        "--chunk_size", str(CHUNK_SIZE),
        "--iou_threshold", str(IOU_THRESHOLD),
        "--device", DEVICE,
    ]
    run_cmd(tracking_cmd, "Tracking inference (sam3)")

    if dir_is_empty(MASK_DIR):
        raise SystemExit(
            f"[ERROR] Tracking finished but MaskDir is empty: {MASK_DIR}\n"
            f"Check sam3 output/logs."
        )

    # Tracking visualization
    vis_track = VIS_DIR / "visualization_tracking.mp4"
    track_vis_cmd = [
        "psifx", "video", "tracking", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--visualization", str(vis_track),
        "--labels",
        "--color",
    ]
    run_cmd(track_vis_cmd, "Tracking visualization")

    # STEP 3: Pose (MediaPipe multi-inference)
    pose_cmd = [
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", str(PROCESSED_VIDEO),
        "--masks", str(MASK_DIR),
        "--poses_dir", str(POSES_DIR),
        "--device", DEVICE,
    ]
    run_cmd(pose_cmd, "Pose multi-inference (mediapipe)")

    # Pose visualization
    vis_pose = VIS_DIR / "visualization_pose.mp4"
    pose_vis_cmd = [
        "psifx", "video", "pose", "mediapipe", "visualization",
        "--video", str(PROCESSED_VIDEO),
        "--poses", str(POSES_DIR),
        "--visualization", str(vis_pose),
        "--confidence_threshold", "0.0",
    ]
    run_cmd(pose_vis_cmd, "Pose visualization")

    # STEP 4: Face (optional)
    if enable_face:
        face_cmd = [
            "psifx", "video", "face", "openface", "multi-inference",
            "--video", str(PROCESSED_VIDEO),
            "--masks", str(MASK_DIR),
            "--features_dir", str(FACES_DIR),
            "--device", DEVICE,  # OpenFace will use CPU anyway; psifx accepts this arg
        ]
        run_cmd(face_cmd, "Face multi-inference (openface)")

        if dir_is_empty(FACES_DIR):
            raise SystemExit(
                f"[ERROR] Face extraction ran but FacesDir is empty: {FACES_DIR}\n"
                f"Most common causes:\n"
                f"- OpenFace not installed correctly / models missing\n"
                f"- Masks don't cover faces well\n"
            )

        vis_face = VIS_DIR / "visualization_face.mp4"
        face_vis_cmd = [
            "psifx", "video", "face", "openface", "visualization",
            "--video", str(PROCESSED_VIDEO),
            "--features", str(FACES_DIR),
            "--visualization", str(vis_face),
            "--depth", "3.0",
        ]
        run_cmd(face_vis_cmd, "Face visualization")
    else:
        print("\n>>> Face module disabled (run with --face to enable).")
        log_line(LOG_FILE, "[OK] Face module disabled")

    print("\n==================== Processing Complete ====================")
    print(f"Results saved to: {BASE_PATH}")
    print(f"MaskDir: {MASK_DIR}")
    print(f"PosesDir: {POSES_DIR}")
    if enable_face:
        print(f"FacesDir: {FACES_DIR}")
    print(f"Visualizations: {VIS_DIR}")
    print(f"Log: {LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

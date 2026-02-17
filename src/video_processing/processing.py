#!/usr/bin/env python3

import subprocess
from pathlib import Path
import sys
import traceback
import shutil

# === Custom settings ===
nb_objects = "2"
yolo_model = "yolo11m.pt"
model_size = "small"
fps = 15
step_size = 1
object_class = "0"
device = "cuda"

start_trim_sec = 0
end_trim_sec = None

base_path = Path("/home/liubov/Bureau/new/15-5-2024_#20_INDIVIDUAL_[15]")
raw_video = base_path / "camera_a.mkv"
processed_video = base_path / "processed_video.mp4"

frames_dir = base_path / "Frames"
mask_dir = base_path / "MaskDir"
poses_dir = base_path / "PosesDir"
faces_dir = base_path / "FacesDir"
vis_dir = base_path / "Visualizations"
log_file_path = base_path / "processing_log.log"

# === Ensure output directories exist ===
for d in [frames_dir, mask_dir, poses_dir, faces_dir, vis_dir]:
    d.mkdir(parents=True, exist_ok=True)
log_file_path.write_text("=== Processing Log ===\n")


def log_to_file(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")


def run_cmd(cmd, step_desc):
    print(f"\n>>> {step_desc}")
    print("Running command:", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Command succeeded.")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {step_desc} failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        log_to_file(f"[FAILED] {step_desc}: {e.stderr.strip()}")
        sys.exit(1)


# === Step 1: Trim and sample video (only if Frames folder is empty) ===
if frames_dir.exists() and any(frames_dir.iterdir()):
    print(f"Frames folder already exists and is not empty: {frames_dir}")
    log_to_file(f"[SKIPPED] Frame extraction skipped: {frames_dir} already populated")
else:
    if not processed_video.exists():
        ffmpeg_trim_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_video),
            "-ss", str(start_trim_sec),
            "-to", str(end_trim_sec),
            "-filter:v", f"fps={fps}",
            str(processed_video)
        ]
        run_cmd(ffmpeg_trim_cmd, "Trimming and sampling video")
        log_to_file(f"[OK] Processed video created: {processed_video}")
    else:
        log_to_file(f"[OK] Processed video exist: {processed_video}")

# === Step 2: Run tracking inference ===
tracking_cmd = [
    "psifx", "video", "tracking", "samurai", "inference",
    "--video", str(processed_video),
    "--mask_dir", str(mask_dir),
    "--model_size", "small",
    "--yolo_model", yolo_model,
    "--max_objects", "2",
    "--step", "1",
    "--device", "cuda",
    "--overwrite"
]

run_cmd(tracking_cmd, "Tracking inference")
log_to_file("[OK] Tracking inference completed successfully")

vis_track = vis_dir / "visualization_tracking.mp4"
track_cmd = [
    "psifx", "video", "tracking", "visualization",
    "--video", str(processed_video),
    "--masks", mask_dir,
    "--visualization", str(vis_track),
    "--overwrite"
]
run_cmd(track_cmd, "Tracking visualization")
log_to_file("[OK] Tracking visualization completed successfully")

run_cmd([
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", str(processed_video),
        "--masks", mask_dir,
        "--poses_dir", poses_dir,
        "--device", "cuda",
        "--overwrite"
    ], "Pose inference")

# === Step 3: Visualize pose ===
vis_pose = vis_dir / "visualization_pose.mp4"
pose_cmd = [
    "psifx", "video", "pose", "mediapipe", "visualization",
    "--video", str(processed_video),
    "--poses", str(poses_dir),
    "--visualization", str(vis_pose),
    "--confidence_threshold", "0.0",
    "--overwrite"
]



try:
    run_cmd(pose_cmd, "Visualizing pose")
    log_to_file("[OK] Pose visualization completed successfully")
except Exception as e:
    log_to_file(f"[WARNING] Pose visualization failed: {str(e)}")

## === Step 4: Face feature extraction ===
#face_cmd = [
    #"psifx", "video", "face", "openface", "multi-inference",
    #"--video", str(processed_video),
    #"--masks", str(mask_dir),
    #"--features_dir", str(faces_dir),
    #"--device", device,
    #"--overwrite"
#]

#try:
    #run_cmd(face_cmd, "Face inference")
    #log_to_file("[OK] Face inference completed successfully")
#except Exception as e:
    #log_to_file(f"[WARNING] Face inference failed: {str(e)}")

## === Step 5: Visualize face features ===
#vis_face = vis_dir / "visualization_face.mp4"
#face_vis_cmd = [
    #"psifx", "video", "face", "openface", "visualization",
    #"--video", str(processed_video),
    #"--features", str(faces_dir),
    #"--visualization", str(vis_face),
    #"--depth", "3.0",
    #"--f_x", "1600.0", "--f_y", "1600.0",
    #"--c_x", "960.0", "--c_y", "540.0",
    #"--overwrite"
#]

#try:
    #run_cmd(face_vis_cmd, "Visualizing face")
    #log_to_file("[OK] Face visualization completed successfully")
#except Exception as e:
    #log_to_file(f"[WARNING] Face visualization failed: {str(e)}")

# === Step 6: Cleanup frames directory ===
try:
    if frames_dir.exists():
        print(f"Cleaning up frames directory: {frames_dir}")
        shutil.rmtree(frames_dir)
        log_to_file(f"[OK] Cleaned up frames directory: {frames_dir}")
except Exception as e:
    log_to_file(f"[WARNING] Failed to clean up frames directory: {str(e)}")

# === Final Summary ===
print("\n==================== Processing Complete ====================")
num_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
#print(f"Total frames processed: {num_frames}")
print(f"Results saved to: {base_path}")
print(f"See {log_file_path} for details.")

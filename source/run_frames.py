#!/usr/bin/env python3
"""
Video processing pipeline for tracking, pose estimation, and face detection.
Processes video through multiple analysis stages with robust error handling.
"""

import subprocess
from pathlib import Path
import sys
import shutil
from datetime import datetime
from typing import List, Optional
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for video processing pipeline."""

    # Paths
    BASE_PATH = Path("/home/liubov/Bureau/new/27-3-2024_#15_INDIVIDUAL_[15]")
    RAW_VIDEO = BASE_PATH / "camera_a.mkv"
    PROCESSED_VIDEO = BASE_PATH / "processed_video.mp4"

    # Output directories
    FRAMES_DIR = BASE_PATH / "Frames"
    MASK_DIR = BASE_PATH / "MaskDir"
    POSES_DIR = BASE_PATH / "PosesDir"
    FACES_DIR = BASE_PATH / "FacesDir"
    VIS_DIR = BASE_PATH / "Visualizations"
    LOG_FILE = BASE_PATH / "processing_log.log"

    # Video processing parameters
    START_TRIM_SEC = 185
    END_TRIM_SEC = 280
    FPS = 15

    # Tracking parameters
    YOLO_MODEL = "yolo11s.pt"
    MODEL_SIZE = "small"
    MAX_OBJECTS = 3
    OBJECT_CLASS = None  # Set to None to detect all classes, or "0" for person only
    DEVICE = "cuda"

    # Pose parameters
    POSE_CONFIDENCE_THRESHOLD = 0.0

    # Face parameters
    FACE_DEPTH = 3.0
    FACE_FX = 1600.0
    FACE_FY = 1600.0
    FACE_CX = 960.0
    FACE_CY = 540.0

    # Processing options
    SKIP_IF_EXISTS = True  # Skip steps if output already exists
    CLEANUP_FRAMES = True  # Remove frames directory after processing
    CONTINUE_ON_ERROR = True  # Continue pipeline even if non-critical steps fail


# ============================================================================
# LOGGING AND UTILITIES
# ============================================================================

class Logger:
    """Handle logging to both console and file."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_log()

    def _initialize_log(self):
        """Initialize log file with header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{'='*70}
Video Processing Pipeline Log
Started: {timestamp}
{'='*70}
"""
        self.log_file.write_text(header)

    def log(self, message: str, level: str = "INFO"):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")

    def success(self, message: str):
        """Log success message."""
        self.log(message, "SUCCESS")

    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")

    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")

    def section(self, title: str):
        """Log section header."""
        separator = "=" * 70
        self.log(f"\n{separator}\n{title}\n{separator}")


# ============================================================================
# COMMAND EXECUTION
# ============================================================================

class CommandRunner:
    """Execute shell commands with logging and error handling."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def run(self, cmd: List[str], description: str,
            critical: bool = True) -> Optional[subprocess.CompletedProcess]:
        """
        Execute a command with proper error handling.

        Args:
            cmd: Command and arguments as list
            description: Human-readable description of the command
            critical: If True, exit on failure; if False, continue pipeline

        Returns:
            CompletedProcess object if successful, None if failed
        """
        self.logger.log(f"Running: {description}")
        self.logger.log(f"Command: {' '.join(map(str, cmd))}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.success(f"{description} completed")

            if result.stdout and result.stdout.strip():
                self.logger.log(f"Output: {result.stdout.strip()}")

            if result.stderr and result.stderr.strip():
                # Filter out common warnings
                stderr_lines = result.stderr.strip().split('\n')
                important_lines = [
                    line for line in stderr_lines
                    if not any(skip in line for skip in [
                        'UserWarning',
                        'torchaudio._backend.list_audio_backends',
                        'GPU device discovery failed'
                    ])
                ]
                if important_lines:
                    self.logger.log(f"Warnings: {chr(10).join(important_lines)}")

            return result

        except subprocess.CalledProcessError as e:
            self.logger.error(f"{description} failed with exit code {e.returncode}")

            if e.stdout:
                self.logger.log(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.log(f"STDERR: {e.stderr}")

            if critical:
                self.logger.error("Critical step failed. Exiting pipeline.")
                sys.exit(1)
            else:
                self.logger.warning(f"Non-critical step failed. Continuing pipeline.")
                return None


# ============================================================================
# PROCESSING STEPS
# ============================================================================

class VideoProcessor:
    """Main video processing pipeline."""

    def __init__(self, config: Config, logger: Logger, runner: CommandRunner):
        self.config = config
        self.logger = logger
        self.runner = runner
        self._ensure_directories()

    def _ensure_directories(self):
        """Create all necessary output directories."""
        for directory in [
            self.config.FRAMES_DIR,
            self.config.MASK_DIR,
            self.config.POSES_DIR,
            self.config.FACES_DIR,
            self.config.VIS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _should_skip(self, output_path: Path, step_name: str) -> bool:
        """Check if a step should be skipped based on existing output."""
        if not self.config.SKIP_IF_EXISTS:
            return False

        if output_path.exists():
            if output_path.is_dir() and any(output_path.iterdir()):
                self.logger.warning(f"Skipping {step_name}: Output already exists at {output_path}")
                return True
            elif output_path.is_file():
                self.logger.warning(f"Skipping {step_name}: Output already exists at {output_path}")
                return True

        return False

    def process_video(self):
        """Trim and resample the raw video."""
        self.logger.section("Step 1: Video Preprocessing")

        if self._should_skip(self.config.PROCESSED_VIDEO, "video preprocessing"):
            return

        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.config.RAW_VIDEO),
            "-ss", str(self.config.START_TRIM_SEC),
            "-to", str(self.config.END_TRIM_SEC),
            "-filter:v", f"fps={self.config.FPS}",
            str(self.config.PROCESSED_VIDEO)
        ]

        self.runner.run(cmd, "Trimming and resampling video", critical=True)

    def run_tracking(self):
        """Run SAMURAI tracking inference."""
        self.logger.section("Step 2: Object Tracking")

        if self._should_skip(self.config.MASK_DIR, "tracking inference"):
            return

        cmd = [
            "psifx", "video", "tracking", "samurai", "inference",
            "--video", str(self.config.PROCESSED_VIDEO),
            "--mask_dir", str(self.config.MASK_DIR),
            "--model_size", self.config.MODEL_SIZE,
            "--yolo_model", self.config.YOLO_MODEL,
            "--max_objects", str(self.config.MAX_OBJECTS),
            "--device", self.config.DEVICE,
            "--overwrite"
        ]

        # Add object class filter if specified
        if self.config.OBJECT_CLASS is not None:
            cmd.extend(["--object_class", str(self.config.OBJECT_CLASS)])

        result = self.runner.run(cmd, "Object tracking inference",
                                critical=not self.config.CONTINUE_ON_ERROR)

        # Check if masks were generated
        if result:
            mask_files = list(self.config.MASK_DIR.glob("*.npz"))
            if not mask_files:
                self.logger.error("No masks generated! YOLO may not be detecting objects.")
                self.logger.warning("Try removing --object_class filter or check your video content")
            else:
                self.logger.success(f"Generated {len(mask_files)} mask files")

    def visualize_tracking(self):
        """Create tracking visualization video."""
        self.logger.section("Step 3: Tracking Visualization")

        vis_path = self.config.VIS_DIR / "visualization_tracking.mp4"

        if self._should_skip(vis_path, "tracking visualization"):
            return

        cmd = [
            "psifx", "video", "tracking", "visualization",
            "--video", str(self.config.PROCESSED_VIDEO),
            "--masks", str(self.config.MASK_DIR),
            "--visualization", str(vis_path),
            "--overwrite"
        ]

        self.runner.run(cmd, "Creating tracking visualization",
                       critical=not self.config.CONTINUE_ON_ERROR)

    def run_pose_estimation(self):
        """Run MediaPipe pose estimation."""
        self.logger.section("Step 4: Pose Estimation")

        if self._should_skip(self.config.POSES_DIR, "pose estimation"):
            return

        cmd = [
            "psifx", "video", "pose", "mediapipe", "multi-inference",
            "--video", str(self.config.PROCESSED_VIDEO),
            "--masks", str(self.config.MASK_DIR),
            "--poses_dir", str(self.config.POSES_DIR),
            "--device", self.config.DEVICE,
            "--overwrite"
        ]

        self.runner.run(cmd, "Pose estimation inference",
                       critical=not self.config.CONTINUE_ON_ERROR)

    def visualize_pose(self):
        """Create pose visualization video."""
        self.logger.section("Step 5: Pose Visualization")

        vis_path = self.config.VIS_DIR / "visualization_pose.mp4"

        if self._should_skip(vis_path, "pose visualization"):
            return

        cmd = [
            "psifx", "video", "pose", "mediapipe", "visualization",
            "--video", str(self.config.PROCESSED_VIDEO),
            "--poses", str(self.config.POSES_DIR),
            "--visualization", str(vis_path),
            "--confidence_threshold", str(self.config.POSE_CONFIDENCE_THRESHOLD),
            "--overwrite"
        ]

        self.runner.run(cmd, "Creating pose visualization",
                       critical=not self.config.CONTINUE_ON_ERROR)

    def run_face_detection(self):
        """Run OpenFace face detection."""
        self.logger.section("Step 6: Face Detection")

        if self._should_skip(self.config.FACES_DIR, "face detection"):
            return

        cmd = [
            "psifx", "video", "face", "openface", "multi-inference",
            "--video", str(self.config.PROCESSED_VIDEO),
            "--masks", str(self.config.MASK_DIR),
            "--features_dir", str(self.config.FACES_DIR),
            "--device", self.config.DEVICE,
            "--overwrite"
        ]

        self.runner.run(cmd, "Face detection inference",
                       critical=not self.config.CONTINUE_ON_ERROR)

    def visualize_face(self):
        """Create face visualization video."""
        self.logger.section("Step 7: Face Visualization")

        vis_path = self.config.VIS_DIR / "visualization_face.mp4"

        if self._should_skip(vis_path, "face visualization"):
            return

        cmd = [
            "psifx", "video", "face", "openface", "visualization",
            "--video", str(self.config.PROCESSED_VIDEO),
            "--features", str(self.config.FACES_DIR),
            "--visualization", str(vis_path),
            "--depth", str(self.config.FACE_DEPTH),
            "--f_x", str(self.config.FACE_FX),
            "--f_y", str(self.config.FACE_FY),
            "--c_x", str(self.config.FACE_CX),
            "--c_y", str(self.config.FACE_CY),
            "--overwrite"
        ]

        self.runner.run(cmd, "Creating face visualization",
                       critical=not self.config.CONTINUE_ON_ERROR)

    def cleanup(self):
        """Clean up temporary files."""
        if not self.config.CLEANUP_FRAMES:
            return

        self.logger.section("Step 8: Cleanup")

        try:
            if self.config.FRAMES_DIR.exists():
                self.logger.log(f"Removing frames directory: {self.config.FRAMES_DIR}")
                shutil.rmtree(self.config.FRAMES_DIR)
                self.logger.success("Frames directory removed")
        except Exception as e:
            self.logger.warning(f"Failed to remove frames directory: {e}")

    def run_full_pipeline(self):
        """Execute the complete processing pipeline."""
        self.logger.section("Starting Video Processing Pipeline")

        try:
            self.process_video()
            self.run_tracking()
            self.visualize_tracking()
            self.run_pose_estimation()
            self.visualize_pose()
            self.run_face_detection()
            self.visualize_face()
            self.cleanup()

            self.logger.section("Pipeline Complete")
            self.logger.success(f"All results saved to: {self.config.BASE_PATH}")
            self.logger.success(f"Log file: {self.config.LOG_FILE}")

        except KeyboardInterrupt:
            self.logger.warning("\nPipeline interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            sys.exit(1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the video processing pipeline."""

    # Initialize components
    config = Config()
    logger = Logger(config.LOG_FILE)
    runner = CommandRunner(logger)
    processor = VideoProcessor(config, logger, runner)

    # Run the pipeline
    processor.run_full_pipeline()


if __name__ == "__main__":
    main()

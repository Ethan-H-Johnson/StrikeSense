"""
StrikeSense Pose Estimation Demo (Config-Driven)
Real-time 3D pose keypoint extraction for Jab/Cross analysis
"""

import cv2
import mediapipe as mp
import time
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys
import subprocess

# Add pose_estimation to path
sys.path.insert(0, str(Path(__file__).parent))

from keypoint_logger import KeypointLogger
from strike_detector import StrikeDetector
from audio_feedback import AudioFeedback


# ------------------------------------------------------------------
# CONFIG LOADER
# ------------------------------------------------------------------

def load_config():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config" / "pose.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# POSE ESTIMATOR
# ------------------------------------------------------------------

class PoseEstimator:
    def __init__(self, config):
        self.config = config

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        model_cfg = config["model"]

        self.pose = self.mp_pose.Pose(
            model_complexity=model_cfg["model_complexity"],
            min_detection_confidence=model_cfg["min_detection_confidence"],
            min_tracking_confidence=model_cfg["min_tracking_confidence"],
            static_image_mode=model_cfg["static_image_mode"],
        )

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            return results.pose_landmarks, results.pose_world_landmarks
        return None, None

    def draw_skeleton(self, frame, landmarks):
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame


# ------------------------------------------------------------------
# METADATA WRITER
# ------------------------------------------------------------------

def write_metadata(session_dir, config, fps, width, height):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "fps_target": fps,
        "resolution": {"width": width, "height": height},
        "mediapipe_version": mp.__version__,
        "config": config,
        "git_commit": get_git_commit(),
    }

    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 50)
    print("StrikeSense Pose Estimation (Config Driven)")
    print("=" * 50)

    config = load_config()

    pose_estimator = PoseEstimator(config)
    strike_detector = StrikeDetector()
    audio_feedback = AudioFeedback()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_target = config["recording"]["fps_target"]
    video_codec = config["recording"]["video_codec"]

    recording = False
    logger = None
    video_writer = None
    frame_idx = 0
    start_time = None
    session_dir = None

    cv2.namedWindow("StrikeSense - Pose Estimation", cv2.WINDOW_NORMAL)

    print("\n[READY] Press 'R' to start recording\n")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_2d, landmarks_3d = pose_estimator.process_frame(frame)

        # Strike detection
        if landmarks_3d:
            timestamp = time.time()
            strike, stance = strike_detector.process(landmarks_3d, timestamp)

            if strike:
                print(f"[STRIKE] {strike.upper()} detected!")
                if strike == "jab":
                    audio_feedback.play_jab()
                elif strike == "cross":
                    audio_feedback.play_cross()

        display_frame = frame.copy()
        display_frame = pose_estimator.draw_skeleton(display_frame, landmarks_2d)

        # Recording
        if recording:
            if start_time is None:
                start_time = time.time()

            rec_timestamp = time.time() - start_time

            if logger and landmarks_3d:
                logger.log_frame(rec_timestamp, frame_idx, landmarks_3d)

            if video_writer:
                video_writer.write(display_frame)

        # FPS Overlay
        fps = 1.0 / (time.time() - frame_start + 1e-6)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("StrikeSense - Pose Estimation", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            recording = not recording

            if recording:
                print("[REC] Recording started")

                root_data_dir = Path(__file__).resolve().parent.parent / "data" / "recordings"
                root_data_dir.mkdir(parents=True, exist_ok=True)

                session_name = datetime.now().strftime(
                    config["data"]["session_format"]
                )

                session_dir = root_data_dir / session_name
                session_dir.mkdir(parents=True, exist_ok=True)

                # Initialize logger
                logger = KeypointLogger(session_dir, config)

                # Video writer
                video_path = session_dir / "video.mp4"
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*video_codec),
                    fps_target,
                    (width, height)
                )

                # Metadata
                if config["data"]["include_metadata"]:
                    write_metadata(session_dir, config, fps_target, width, height)

                start_time = None
                frame_idx = 0

            else:
                print("[REC] Recording stopped")

                if logger:
                    logger.close()
                    logger = None

                if video_writer:
                    video_writer.release()
                    video_writer = None

                start_time = None

        frame_idx += 1

    cap.release()

    if logger:
        logger.close()

    if video_writer:
        video_writer.release()

    cv2.destroyAllWindows()
    print("\n[DONE] StrikeSense session ended")


if __name__ == "__main__":
    main()

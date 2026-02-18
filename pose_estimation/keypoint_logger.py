"""
KeypointLogger - Session-based CSV export for 3D pose keypoints
Phase 0 Refactored Version (Config Driven)
"""

import csv
from pathlib import Path


class KeypointLogger:
    """
    Logs 3D keypoints to CSV for biomechanical analysis.

    Writes to:
        session_dir/keypoints.csv
    """

    def __init__(self, session_dir: Path, config: dict):
        """
        Args:
            session_dir: Path to active recording session folder
            config: Loaded pose.yaml configuration
        """

        self.session_dir = Path(session_dir)
        self.config = config

        self.filepath = self.session_dir / "keypoints.csv"

        # Pull joint subset from config
        self.relevant_joints = config["landmarks"]["relevant_joints"]

        # Open CSV file
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Write header
        self.writer.writerow([
            "timestamp",
            "frame",
            "joint",
            "x",
            "y",
            "z",
            "visibility"
        ])

        self.file.flush()

        print(f"[LOGGER] Writing keypoints to: {self.filepath}")

    # ------------------------------------------------------------------

    def log_frame(self, timestamp, frame_num, world_landmarks):
        """
        Log all relevant joints for a single frame.

        Args:
            timestamp: Seconds since recording start
            frame_num: Frame index
            world_landmarks: MediaPipe world landmarks
        """

        if world_landmarks is None:
            return

        for idx_str, joint_name in self.relevant_joints.items():
            idx = int(idx_str)  # YAML keys load as strings

            if idx < len(world_landmarks.landmark):
                lm = world_landmarks.landmark[idx]

                self.writer.writerow([
                    f"{timestamp:.6f}",
                    frame_num,
                    joint_name,
                    f"{lm.x:.6f}",
                    f"{lm.y:.6f}",
                    f"{lm.z:.6f}",
                    f"{lm.visibility:.6f}",
                ])

        # Optional but good for long recordings
        self.file.flush()

    # ------------------------------------------------------------------

    def close(self):
        """Close CSV safely"""
        if self.file:
            self.file.close()
            print(f"[LOGGER] Saved keypoints: {self.filepath}")

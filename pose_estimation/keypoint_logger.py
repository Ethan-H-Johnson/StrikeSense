"""
KeypointLogger - CSV export for 3D pose keypoints
"""
import csv
from datetime import datetime
from pathlib import Path

# Joint indices for Jab/Cross analysis (MediaPipe Pose)
RELEVANT_JOINTS = {
    0: 'Nose',
    11: 'L_Shoulder',
    12: 'R_Shoulder',
    13: 'L_Elbow',
    14: 'R_Elbow',
    15: 'L_Wrist',
    16: 'R_Wrist',
    23: 'L_Hip',
    24: 'R_Hip',
    25: 'L_Knee',
    26: 'R_Knee',
    12: 'Neck'  # Approximation (spine alignment)
}

class KeypointLogger:
    """Logs 3D keypoints to CSV for biomechanical analysis"""
    
    def __init__(self, output_dir='data/recordings/keypoints'):
        """
        Initialize logger with auto-generated filename
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename: pose_YYYYMMDD_HHMMSS.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"pose_{timestamp}.csv"
        
        # Open CSV file
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write header
        self.writer.writerow([
            'timestamp', 'frame', 'joint', 'x', 'y', 'z', 'visibility'
        ])
        
        print(f"[LOGGER] Writing keypoints to: {self.filepath}")
    
    def log_frame(self, timestamp, frame_num, world_landmarks):
        """
        Log all relevant joints for a single frame
        
        Args:
            timestamp: Seconds since recording start
            frame_num: Frame number
            world_landmarks: MediaPipe world landmarks
        """
        if world_landmarks is None:
            return
        
        # Extract only relevant joints
        for idx, joint_name in RELEVANT_JOINTS.items():
            if idx < len(world_landmarks.landmark):
                lm = world_landmarks.landmark[idx]
                self.writer.writerow([
                    f"{timestamp:.3f}",
                    frame_num,
                    joint_name,
                    f"{lm.x:.4f}",
                    f"{lm.y:.4f}",
                    f"{lm.z:.4f}",
                    f"{lm.visibility:.3f}"
                ])
    
    def close(self):
        """Close the CSV file"""
        if self.file:
            self.file.close()
            print(f"[LOGGER] Saved keypoints: {self.filepath}")

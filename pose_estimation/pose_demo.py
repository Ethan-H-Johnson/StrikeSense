"""
StrikeSense Pose Estimation Demo
Real-time 3D pose keypoint extraction for Jab/Cross analysis
"""
import cv2
import mediapipe as mp
import time
from pathlib import Path
import sys

# Add pose_estimation to path
sys.path.insert(0, str(Path(__file__).parent))
from keypoint_logger import KeypointLogger

class PoseEstimator:
    """MediaPipe 3D pose estimation"""
    def __init__(self, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

    def process_frame(self, frame):
        """Process frame and return 2D + 3D landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            return results.pose_landmarks, results.pose_world_landmarks
        return None, None

    def draw_skeleton(self, frame, landmarks):
        """Draw pose skeleton on frame"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame


def main():
    """Main execution loop"""
    print("="*50)
    print("StrikeSense Pose Estimation")
    print("="*50)
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(model_complexity=1)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_target = 30
    
    # Recording state
    recording = False
    logger = None
    video_writer = None
    frame_idx = 0
    start_time = None
    
    print("\n[CONTROLS]")
    print("  R - Start/Stop Recording")
    print("  Q - Quit")
    print("\n[READY] Press 'R' to start recording\n")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Process pose
        landmarks_2d, landmarks_3d = pose_estimator.process_frame(frame)
        
        # Draw skeleton
        display_frame = frame.copy()
        display_frame = pose_estimator.draw_skeleton(display_frame, landmarks_2d)
        
        # Recording logic
        if recording:
            if start_time is None:
                start_time = time.time()
            
            timestamp = time.time() - start_time
            
            # Log keypoints
            if logger and landmarks_3d:
                logger.log_frame(timestamp, frame_idx, landmarks_3d)
            
            # Write video
            if video_writer:
                video_writer.write(display_frame)
        
        # UI Overlay
        fps = 1.0 / (time.time() - frame_start + 1e-6)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if recording:
            cv2.circle(display_frame, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (width - 70, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("StrikeSense - Pose Estimation", display_frame)
        
        # Input handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            recording = not recording
            
            if recording:
                # Start recording
                print(f"\n[REC] Recording started")
                
                # Initialize logger
                logger = KeypointLogger()
                
                # Initialize video writer
                from datetime import datetime
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = Path('data/recordings/videos') / f"pose_{timestamp_str}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps_target,
                    (width, height)
                )
                
                print(f"[REC] Video: {video_path}")
                start_time = None
                frame_idx = 0
                
            else:
                # Stop recording
                print(f"[REC] Recording stopped")
                
                if logger:
                    logger.close()
                    logger = None
                
                if video_writer:
                    video_writer.release()
                    video_writer = None
                
                start_time = None
                print()
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    if logger:
        logger.close()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print("\n[DONE] StrikeSense session ended")


if __name__ == "__main__":
    main()

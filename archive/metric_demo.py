"""
StrikeSense MVP - Punch Angle Analysis Demo
Uses YOLOv11-Pose for multi-person tracking + MediaPipe Selfie Segmentation for background blur
Deterministic scoring with frame-by-frame playback
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, asdict
from ultralytics import YOLO


@dataclass
class FrameAnalysis:
    """Store analysis data for a single frame"""
    frame_num: int
    timestamp: float
    elbow_angle: Optional[float]
    shoulder_angle: Optional[float]
    wrist_visibility: float
    elbow_visibility: float
    shoulder_visibility: float
    punch_phase: str
    score: Optional[int]
    notes: str


class PunchAngleAnalyzer:
    """Main analyzer class for punch angle detection and scoring"""
    
    # Target angles from biomechanics research (Dinu & Louis, 2020)
    TARGET_ELBOW_ANGLE = 175.0  # degrees
    PERFECT_RANGE = 5.0  # ±5 degrees for 10/10 score
    
    # Visibility thresholds (YOLO confidence)
    MIN_VISIBILITY = 0.5
    HIGH_VISIBILITY = 0.7
    
    # Punch detection thresholds
    VELOCITY_THRESHOLD = 0.05  # normalized wrist displacement per frame
    EXTENSION_ANGLE = 160.0  # degrees - indicates punch extension
    
    # YOLOv11 keypoint indices (COCO format)
    KEYPOINT_NAMES = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16
    }
    
    def __init__(self, video_path: str, use_background_blur: bool = True):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.use_background_blur = use_background_blur
        
        # YOLOv11-Pose setup
        print("Loading YOLOv11-pose model...")
        self.yolo_model = YOLO('yolo11n-pose.pt')  # nano model for speed
        
        # MediaPipe Selfie Segmentation for background blur
        if self.use_background_blur:
            print("Loading background segmentation model...")
            self.mp_selfie = mp.solutions.selfie_segmentation
            self.selfie_seg = self.mp_selfie.SelfieSegmentation(model_selection=1)
        
        # State variables
        self.tracked_person_id: Optional[int] = None
        self.current_frame_idx = 0
        self.paused = True
        self.frame_analyses: List[FrameAnalysis] = []
        self.prev_wrist_pos: Optional[Tuple[float, float]] = None
        self.peak_extension_frame: Optional[int] = None
        self.max_elbow_angle = 0.0
        
        # Frame cache
        self.all_frames = []
        self.all_blurred_frames = []  # Cache blurred frames
        
    def load_and_process_frames(self):
        """Pre-load all frames and apply background blur if enabled"""
        print("Loading video frames...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.all_frames.append(frame.copy())
            
            # Apply background blur
            if self.use_background_blur:
                blurred = self.apply_background_blur(frame)
                self.all_blurred_frames.append(blurred)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count}/{self.total_frames} frames...")
        
        print(f"Loaded {len(self.all_frames)} frames")
        if self.use_background_blur:
            print(f"Applied background blur to all frames")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def apply_background_blur(self, frame: np.ndarray, blur_strength: int = 25) -> np.ndarray:
        """Apply background blur using MediaPipe Selfie Segmentation"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_seg.process(rgb_frame)
        
        # Get segmentation mask
        mask = results.segmentation_mask
        
        # Threshold mask to binary (foreground = 1, background = 0)
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Smooth mask edges
        mask_smooth = cv2.GaussianBlur(mask_binary.astype(np.float32), (15, 15), 0)
        mask_smooth = np.stack([mask_smooth] * 3, axis=-1)
        
        # Create blurred background
        blurred_bg = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        # Composite: foreground + blurred background
        output = (frame * mask_smooth + blurred_bg * (1 - mask_smooth)).astype(np.uint8)
        
        return output
    
    def select_person(self, frame: np.ndarray) -> bool:
        """Manual person selection from detected people"""
        # Get working frame (blurred if enabled)
        work_frame = self.all_blurred_frames[0] if self.use_background_blur else frame
        
        # Detect all people with YOLO
        results = self.yolo_model.track(work_frame, persist=True, verbose=False)
        
        if not results or len(results) == 0:
            print("ERROR: No people detected in frame")
            return False
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            print("ERROR: No people detected in frame")
            return False
        
        # Draw all detected people
        display = work_frame.copy()
        people_data = []
        
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else i
            conf = float(box.conf[0])
            
            # Draw bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Person {track_id} ({conf:.2f})", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            people_data.append({
                'id': track_id,
                'bbox': (x1, y1, x2, y2),
                'conf': conf
            })
        
        # Instructions
        cv2.putText(display, "Click on the fighter to track (Mayweather)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Press 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        selected = {'id': None}
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check which bbox was clicked
                for person in people_data:
                    x1, y1, x2, y2 = person['bbox']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected['id'] = person['id']
                        cv2.destroyAllWindows()
                        return
        
        cv2.namedWindow('Select Fighter to Track')
        cv2.setMouseCallback('Select Fighter to Track', mouse_callback)
        
        while selected['id'] is None:
            cv2.imshow('Select Fighter to Track', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        
        self.tracked_person_id = selected['id']
        print(f"Tracking person ID: {self.tracked_person_id}")
        return True
    
    def get_keypoints_for_person(self, frame: np.ndarray, person_id: int) -> Optional[np.ndarray]:
        """Get keypoints for specific tracked person"""
        # Use blurred frame if enabled
        work_frame = frame
        if self.use_background_blur and self.current_frame_idx < len(self.all_blurred_frames):
            work_frame = self.all_blurred_frames[self.current_frame_idx]
        
        results = self.yolo_model.track(work_frame, persist=True, verbose=False)
        
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        
        if result.boxes is None or result.keypoints is None:
            return None
        
        # Find the tracked person
        for i, box in enumerate(result.boxes):
            track_id = int(box.id[0]) if box.id is not None else i
            
            if track_id == person_id:
                # Get keypoints for this person
                kpts = result.keypoints.data[i]  # Shape: (17, 3) - x, y, conf
                return kpts.cpu().numpy()
        
        return None
    
    def calculate_angle(self, p1: Tuple[float, float], 
                       p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def calculate_angles(self, keypoints: np.ndarray) -> Dict:
        """Calculate both full arm and upper arm angles from YOLO keypoints"""
        # Extract right arm keypoints (orthodox stance - lead hand)
        # keypoints shape: (17, 3) - [x, y, confidence]
        
        right_shoulder = keypoints[self.KEYPOINT_NAMES['right_shoulder']]
        right_elbow = keypoints[self.KEYPOINT_NAMES['right_elbow']]
        right_wrist = keypoints[self.KEYPOINT_NAMES['right_wrist']]
        
        shoulder_pos = (right_shoulder[0], right_shoulder[1])
        elbow_pos = (right_elbow[0], right_elbow[1])
        wrist_pos = (right_wrist[0], right_wrist[1])
        
        # Get confidence scores
        shoulder_conf = right_shoulder[2]
        elbow_conf = right_elbow[2]
        wrist_conf = right_wrist[2]
        
        # Full arm angle (shoulder-elbow-wrist)
        elbow_angle = None
        if wrist_conf > self.HIGH_VISIBILITY and elbow_conf > self.MIN_VISIBILITY:
            elbow_angle = self.calculate_angle(shoulder_pos, elbow_pos, wrist_pos)
        
        # Upper arm angle (shoulder-elbow-vertical)
        vertical_ref = (elbow_pos[0], elbow_pos[1] + 100)
        shoulder_angle = None
        if elbow_conf > self.MIN_VISIBILITY and shoulder_conf > self.MIN_VISIBILITY:
            shoulder_angle = self.calculate_angle(shoulder_pos, elbow_pos, vertical_ref)
        
        return {
            'elbow_angle': elbow_angle,
            'shoulder_angle': shoulder_angle,
            'wrist_visibility': float(wrist_conf),
            'elbow_visibility': float(elbow_conf),
            'shoulder_visibility': float(shoulder_conf),
            'wrist_pos': wrist_pos,
            'elbow_pos': elbow_pos,
            'shoulder_pos': shoulder_pos,
            'keypoints': keypoints  # Pass through for drawing
        }
    
    def detect_punch_phase(self, angles: Dict, frame_num: int) -> str:
        """Velocity-based punch detection"""
        wrist_pos = angles['wrist_pos']
        
        # Calculate wrist velocity
        velocity = 0.0
        if self.prev_wrist_pos is not None:
            dx = wrist_pos[0] - self.prev_wrist_pos[0]
            dy = wrist_pos[1] - self.prev_wrist_pos[1]
            velocity = np.sqrt(dx**2 + dy**2) / (np.linalg.norm([wrist_pos[0], wrist_pos[1]]) + 1e-6)
        
        self.prev_wrist_pos = wrist_pos
        
        # Determine phase
        angle = angles.get('elbow_angle') or angles.get('shoulder_angle', 0)
        
        if angle and velocity > self.VELOCITY_THRESHOLD and angle > self.EXTENSION_ANGLE:
            # Track maximum angle for peak detection
            if angle > self.max_elbow_angle:
                self.max_elbow_angle = angle
                self.peak_extension_frame = frame_num
            return "extension"
        elif angle and angle > self.EXTENSION_ANGLE:
            return "peak"
        else:
            return "guard"
    
    def calculate_score(self, angle: float) -> Tuple[int, str]:
        """Deterministic scoring based on angle deviation"""
        if angle is None:
            return None, "Angle unavailable"
        
        deviation = abs(angle - self.TARGET_ELBOW_ANGLE)
        
        if deviation <= self.PERFECT_RANGE:
            return 10, "Perfect extension"
        elif deviation <= 10:
            return 8, "Good extension, slight deviation"
        elif deviation <= 15:
            return 6, "Acceptable, noticeable deviation"
        elif deviation <= 20:
            return 4, "Poor extension angle"
        else:
            return 2, "Significant form error"
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, FrameAnalysis]:
        """Process single frame and return annotated frame + analysis"""
        # Use blurred frame as base if enabled
        if self.use_background_blur and frame_num < len(self.all_blurred_frames):
            annotated_frame = self.all_blurred_frames[frame_num].copy()
        else:
            annotated_frame = frame.copy()
        
        analysis = FrameAnalysis(
            frame_num=frame_num,
            timestamp=frame_num / self.fps,
            elbow_angle=None,
            shoulder_angle=None,
            wrist_visibility=0.0,
            elbow_visibility=0.0,
            shoulder_visibility=0.0,
            punch_phase="no_detection",
            score=None,
            notes=""
        )
        
        # Get keypoints for tracked person
        keypoints = self.get_keypoints_for_person(frame, self.tracked_person_id)
        
        if keypoints is None:
            analysis.notes = "TRACKING LOST - Person not detected"
            cv2.putText(annotated_frame, "TRACKING LOST - Press 'R' to reselect", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated_frame, analysis
        
        # Draw skeleton
        connections = [
            (5, 7), (7, 9),   # Right arm: shoulder -> elbow -> wrist
            (6, 8), (8, 10),  # Left arm
            (5, 6),           # Shoulders
            (5, 11), (6, 12), # Torso
            (11, 12),         # Hips
        ]
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > self.MIN_VISIBILITY:
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Draw connections
        for start_idx, end_idx in connections:
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]
            
            if start_kpt[2] > self.MIN_VISIBILITY and end_kpt[2] > self.MIN_VISIBILITY:
                start_pos = (int(start_kpt[0]), int(start_kpt[1]))
                end_pos = (int(end_kpt[0]), int(end_kpt[1]))
                cv2.line(annotated_frame, start_pos, end_pos, (0, 255, 0), 2)
        
        # Calculate angles
        angles = self.calculate_angles(keypoints)
        
        # Update analysis
        analysis.wrist_visibility = angles['wrist_visibility']
        analysis.elbow_visibility = angles['elbow_visibility']
        analysis.shoulder_visibility = angles['shoulder_visibility']
        
        # Determine which angle to use (adaptive)
        primary_angle = None
        angle_type = ""
        
        if angles['elbow_angle'] is not None:
            primary_angle = angles['elbow_angle']
            angle_type = "Full Arm"
            analysis.elbow_angle = primary_angle
        elif angles['wrist_visibility'] < self.HIGH_VISIBILITY and angles['shoulder_angle'] is not None:
            primary_angle = angles['shoulder_angle']
            angle_type = "Upper Arm (wrist occluded)"
            analysis.notes = "UNRELIABLE: Using upper arm only"
        
        analysis.shoulder_angle = angles['shoulder_angle']
        
        # Detect punch phase
        analysis.punch_phase = self.detect_punch_phase(angles, frame_num)
        
        # Calculate score if at peak
        if analysis.punch_phase == "peak" and primary_angle:
            score, note = self.calculate_score(primary_angle)
            analysis.score = score
            analysis.notes = note
        
        # Draw angle visualization
        if primary_angle:
            elbow_pos = (int(angles['elbow_pos'][0]), int(angles['elbow_pos'][1]))
            shoulder_pos = (int(angles['shoulder_pos'][0]), int(angles['shoulder_pos'][1]))
            wrist_pos = (int(angles['wrist_pos'][0]), int(angles['wrist_pos'][1]))
            
            # Draw angle lines (thicker, yellow)
            cv2.line(annotated_frame, shoulder_pos, elbow_pos, (0, 255, 255), 4)
            cv2.line(annotated_frame, elbow_pos, wrist_pos, (0, 255, 255), 4)
            
            # Draw angle text with background
            angle_text = f"{angle_type}: {primary_angle:.1f}deg"
            text_pos = (elbow_pos[0] + 20, elbow_pos[1] - 20)
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (text_pos[0] - 5, text_pos[1] - text_size[1] - 5),
                         (text_pos[0] + text_size[0] + 5, text_pos[1] + 5),
                         (0, 0, 0), -1)
            cv2.putText(annotated_frame, angle_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw tracking indicator
        cv2.putText(annotated_frame, f"Tracking ID: {self.tracked_person_id}", 
                   (10, annotated_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw HUD
        self.draw_hud(annotated_frame, analysis)
        
        return annotated_frame, analysis
    
    def draw_hud(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Draw heads-up display with metrics"""
        h, w = frame.shape[:2]
        
        # Background for HUD
        cv2.rectangle(frame, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 220), (255, 255, 255), 2)
        
        y_offset = 35
        cv2.putText(frame, f"Frame: {analysis.frame_num}/{self.total_frames}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Time: {analysis.timestamp:.2f}s",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Phase: {analysis.punch_phase}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 30
        if analysis.elbow_angle:
            cv2.putText(frame, f"Elbow: {analysis.elbow_angle:.1f}deg (Target: {self.TARGET_ELBOW_ANGLE})",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Wrist Conf: {analysis.wrist_visibility:.2f}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
        if analysis.score:
            color = (0, 255, 0) if analysis.score >= 8 else (0, 165, 255) if analysis.score >= 6 else (0, 0, 255)
            cv2.putText(frame, f"SCORE: {analysis.score}/10",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        y_offset += 30
        if analysis.notes:
            cv2.putText(frame, analysis.notes,
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Controls: [Space] Play | [Left/Right] Frame | [R] Reselect | [S] Save | [Q] Quit",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main analysis loop with frame-by-frame playback"""
        # Load and process frames
        self.load_and_process_frames()
        
        if not self.all_frames:
            print("Error: No frames loaded")
            return
        
        # Select person on first frame
        if not self.select_person(self.all_frames[0]):
            print("Person selection cancelled")
            return
        
        print("\n=== StrikeSense Punch Angle Analyzer ===")
        print(f"Video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Tracking Person ID: {self.tracked_person_id}")
        print(f"Background Blur: {'Enabled' if self.use_background_blur else 'Disabled'}")
        print("\nControls:")
        print("  [Space]  - Play/Pause")
        print("  [Left]   - Previous frame")
        print("  [Right]  - Next frame")
        print("  [R]      - Reselect person")
        print("  [S]      - Save current frame")
        print("  [Q]      - Quit")
        print("\nStarting analysis...\n")
        
        while True:
            # Get current frame
            if self.current_frame_idx >= len(self.all_frames):
                print("End of video reached")
                break
            
            frame = self.all_frames[self.current_frame_idx]
            
            # Process frame
            annotated, analysis = self.process_frame(frame, self.current_frame_idx)
            self.frame_analyses.append(analysis)
            
            # Display
            cv2.imshow('StrikeSense Analysis', annotated)
            
            # Handle input
            wait_time = int(1000 / self.fps) if not self.paused else 0
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord(' '):  # Space - play/pause
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Playing'}")
            elif key == 83 or key == 2555904 or key == ord('d'):  # Right arrow
                self.current_frame_idx = min(self.current_frame_idx + 1, len(self.all_frames) - 1)
                self.paused = True
            elif key == 81 or key == 2424832 or key == ord('a'):  # Left arrow
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
                self.paused = True
            elif key == ord('r'):  # Reselect person
                self.paused = True
                if self.select_person(self.all_frames[self.current_frame_idx]):
                    print(f"Reselected - now tracking ID: {self.tracked_person_id}")
            elif key == ord('s'):  # Save frame
                output_path = Path("output") / f"frame_{self.current_frame_idx:04d}.jpg"
                output_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(output_path), annotated)
                print(f"Saved: {output_path}")
            elif key == ord('q'):  # Quit
                break
            
            # Auto-advance if playing
            if not self.paused:
                self.current_frame_idx += 1
        
        cv2.destroyAllWindows()
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate JSON report and summary scorecard"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save frame-by-frame data
        report_data = {
            'video': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'target_angle': self.TARGET_ELBOW_ANGLE,
            'tracked_person_id': self.tracked_person_id,
            'background_blur': self.use_background_blur,
            'peak_extension_frame': self.peak_extension_frame,
            'frames': [asdict(fa) for fa in self.frame_analyses]
        }
        
        with open(output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate scorecard
        peak_analysis = None
        if self.peak_extension_frame:
            peak_analysis = next((fa for fa in self.frame_analyses 
                                 if fa.frame_num == self.peak_extension_frame), None)
        
        print("\n" + "="*50)
        print("FINAL SCORECARD")
        print("="*50)
        if peak_analysis and peak_analysis.score:
            print(f"Peak Extension Frame: {peak_analysis.frame_num}")
            print(f"Timestamp: {peak_analysis.timestamp:.2f}s")
            print(f"Elbow Angle: {peak_analysis.elbow_angle:.1f}° (Target: {self.TARGET_ELBOW_ANGLE}°)")
            print(f"Deviation: {abs(peak_analysis.elbow_angle - self.TARGET_ELBOW_ANGLE):.1f}°")
            print(f"\nFINAL SCORE: {peak_analysis.score}/10")
            print(f"Assessment: {peak_analysis.notes}")
        else:
            print("No peak extension detected")
        print("="*50)
        print(f"\nReport saved to: {output_dir / 'analysis_report.json'}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python demo.py <video_path> [--no-blur]")
        print("Example: python demo.py data/mayweather_jab.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    analyzer = PunchAngleAnalyzer(video_path)
    analyzer.run()
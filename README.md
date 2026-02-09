# StrikeSense

Real-time 3D pose estimation system for MMA strike analysis. Currently focused on Jab and Cross biomechanical analysis using MediaPipe Pose for 3D keypoint extraction.

## Features

- MediaPipe Pose for accurate 3D landmark detection
- 12 relevant joint extraction (shoulders, elbows, wrists, hips, knees)
- Real-time Strike Detection (Jab vs Cross)
- Audio Feedback (High/Low beeps)
- CSV export for offline analysis
- 30+ FPS performance on CPU
- Skeleton overlay visualization
- Auto-named recording files
- Resizable window with Fullscreen toggle

## Setup

### Requirements

- Python 3.11
- Webcam

### Installation

1. Navigate to pose estimation directory:
```bash
cd pose_estimation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies:
- mediapipe
- opencv-python
- numpy

## Usage

### Running the Demo

```bash
cd pose_estimation
python pose_demo.py
```

### Controls

- **R**: Start/Stop recording
- **S**: Toggle Strike Detection (ON/OFF)
- **F**: Toggle Fullscreen
- **Q**: Quit application

### Recording Workflow

1. Launch pose_demo.py
2. Position yourself in frame (skeleton should appear)
3. Press S to enable strike detection (optional)
4. Press R to start recording
5. Perform strikes (jabs, crosses)
6. Press R to stop recording
7. Files automatically saved to data/recordings/

## Output Files

### Video Files
Location: `data/recordings/videos/`
Format: `pose_YYYYMMDD_HHMMSS.mp4`
- Annotated video with skeleton overlay
- 30 FPS

### Keypoint Data
Location: `data/recordings/keypoints/`
Format: `pose_YYYYMMDD_HHMMSS.csv`

CSV Structure:
```csv
timestamp,frame,joint,x,y,z,visibility
0.033,1,L_Shoulder,0.45,-0.12,0.85,0.99
0.033,1,R_Shoulder,0.55,-0.10,0.82,0.99
```

Columns:
- timestamp: Seconds since recording start
- frame: Frame number
- joint: Joint name (L_Shoulder, R_Elbow, etc.)
- x, y, z: 3D world coordinates (origin at hips)
- visibility: Landmark confidence (0-1)

## Extracted Joints

12 relevant joints for Jab/Cross analysis:

| Joint | Biomechanical Relevance |
|-------|------------------------|
| L_Shoulder, R_Shoulder | Rotation, power generation |
| L_Elbow, R_Elbow | Extension angle |
| L_Wrist, R_Wrist | Fist trajectory, speed |
| L_Hip, R_Hip | Hip rotation, weight transfer |
| L_Knee, R_Knee | Stance stability |
| Nose | Head movement |
| Neck | Spine alignment |

## Project Structure

```
StrikeSense/
├── pose_estimation/          # Current MVP
│   ├── pose_demo.py         # Main application
│   ├── strike_detector.py   # Strike logic & Stance detection
│   ├── audio_feedback.py    # Audio feedback module
│   ├── keypoint_logger.py   # CSV export module
│   └── requirements.txt     # Dependencies
│
├── data/
│   └── recordings/
│       ├── videos/          # Annotated MP4 files
│       └── keypoints/       # CSV keypoint data
│         
└── archive/                 # Archived (SMPL & Metric experiments)
    ├── mediapipe_demo.py
    ├── metric_demo.py
    └── smpl_demo.py
```

## Technical Details

### Coordinate System
MediaPipe provides world coordinates with origin at hip center:
- X-axis: Right (positive) / Left (negative)
- Y-axis: Up (positive) / Down (negative)
- Z-axis: Forward (positive) / Backward (negative)

### Pose Estimation
- Model: MediaPipe Pose (BlazePose GHUM)
- Complexity: 1 (balanced speed/accuracy)
- Landmarks: 33 total (12 exported)

## Notes

- SMPL mesh integration was explored but removed for simplicity
- Current focus is on keypoint extraction for analytics
- MediaPipe chosen over YOLOv8/MMPose for 3D support

## License
- MIT
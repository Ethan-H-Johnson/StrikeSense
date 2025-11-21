"""
StrikeSense 3D Pose Estimation Demo
Real-time pose estimation with SMPL mesh overlay
"""
import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
import json
import time
import os

# === Configuration ===
MODEL_FOLDER = r"D:\Projects\StrikeSense\models_smplx_v1_1\models"
GENDER = 'neutral'
DEVICE = 'cpu'

class PoseEstimator:
    """Handles MediaPipe pose estimation"""
    def __init__(self, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

    def process_frame(self, frame):
        """Process frame and return landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks, results.pose_world_landmarks
        return None, None

class SMPLHandler:
    """Handles SMPL model loading and pose fitting"""
    def __init__(self, model_folder, gender='neutral', device='cpu'):
        self.device = torch.device(device)
        self.gender = gender
        
        try:
            print(f"[INFO] Loading SMPL-X model from: {model_folder}")
            self.smpl_model = smplx.create(
                model_path=model_folder,
                model_type='smplx',
                gender=gender,
                use_pca=False,
                num_betas=10
            ).to(self.device)
            
            self.faces = self.smpl_model.faces
            print(f"[INFO] Successfully loaded SMPL-X model")
            
        except Exception as e:
            print(f"[ERROR] Failed to load SMPL model: {e}")
            self.smpl_model = None
            self.faces = None

    def _vec_angle(self, v1, v2):
        """Calculate angle between two vectors"""
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def landmarks_to_smpl(self, world_landmarks):
        """Map MediaPipe landmarks to SMPL pose using improved heuristic IK"""
        if not self.smpl_model:
            return None, None

        # Extract landmarks as numpy array
        lms = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark])

        # Initialize pose parameters
        body_pose = torch.zeros([1, 63], dtype=torch.float32).to(self.device)
        global_orient = torch.zeros([1, 3], dtype=torch.float32).to(self.device)

        # === FIX: Calculate proper global orientation ===
        # Use shoulder line to determine facing direction
        left_shoulder, right_shoulder = lms[11], lms[12]
        shoulder_vec = right_shoulder - left_shoulder
        
        # Calculate rotation around Y axis (yaw)
        # Note: MediaPipe Z-axis points forward, X is right, Y is up
        yaw = np.arctan2(shoulder_vec[2], shoulder_vec[0])  # Rotation in XZ plane
        global_orient[0, 1] = yaw - np.pi/2  # Offset to align with SMPL forward

        # Get hip center for translation
        left_hip, right_hip = lms[23], lms[24]
        hip_center = (left_hip + right_hip) / 2

        # === Enhanced Joint Mapping ===
        # Map more joints for better accuracy
        joint_mappings = [
            # Format: ([parent_idx, joint_idx, child_idx], smpl_joint_idx, rotation_axis)
            
            # Arms
            ([11, 13, 15], 18, 0),  # L_Elbow (X-axis flexion)
            ([12, 14, 16], 19, 0),  # R_Elbow
            
            # Legs  
            ([23, 25, 27], 4, 0),   # L_Knee
            ([24, 26, 28], 5, 0),   # R_Knee
            
            # Shoulders (abduction/adduction)
            ([12, 11, 13], 16, 1),  # L_Shoulder (Y-axis rotation)
            ([11, 12, 14], 17, 1),  # R_Shoulder
        ]

        for (p_idx, j_idx, c_idx), smpl_joint, axis in joint_mappings:
            parent, joint, child = lms[p_idx], lms[j_idx], lms[c_idx]
            
            # Calculate angle
            v1 = parent - joint
            v2 = child - joint
            angle = self._vec_angle(v1, v2)
            
            # Convert to rotation
            flexion = np.pi - angle
            
            # Apply to SMPL (with better scaling)
            if smpl_joint > 0:
                pose_idx = (smpl_joint - 1) * 3 + axis
                if pose_idx < 63:
                    # Scale rotation for better fit
                    body_pose[0, pose_idx] = np.clip(flexion * 0.8, -np.pi, np.pi)

        # Forward pass
        output = self.smpl_model(
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True
        )
        
        vertices = output.vertices.detach().cpu().numpy()[0]
        
        # Align mesh to MediaPipe coordinate system
        vertices[:, 0] += hip_center[0]
        vertices[:, 1] += hip_center[1]
        vertices[:, 2] += hip_center[2]
        
        return vertices, self.faces

class Renderer:
    """Handles 3D mesh projection and rendering"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        f = width
        self.K = np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist = np.zeros(5)

    def draw_mesh(self, frame, vertices, faces):
        """Draw wireframe mesh overlay on frame"""
        if vertices is None or faces is None:
            return frame

        # Project 3D vertices to 2D
        v_cam = vertices.copy()
        
        # Coordinate system adjustments
        v_cam[:, 1] = -v_cam[:, 1]  # Flip Y for correct orientation
        v_cam[:, 2] += 2.5          # Push back
        v_cam *= 0.5                # Scale down

        points_2d, _ = cv2.projectPoints(v_cam, np.zeros(3), np.zeros(3), self.K, self.dist)
        points_2d = points_2d.reshape(-1, 2).astype(int)

        # Draw wireframe
        step = 5
        for face in faces[::step]:
            pt1, pt2, pt3 = points_2d[face[0]], points_2d[face[1]], points_2d[face[2]]
            
            if (0 <= pt1[0] < self.width and 0 <= pt1[1] < self.height):
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 100), 1)
                cv2.line(frame, tuple(pt2), tuple(pt3), (0, 255, 100), 1)
                cv2.line(frame, tuple(pt3), tuple(pt1), (0, 255, 100), 1)
        
        return frame

def main():
    """Main execution loop"""
    print("Initializing StrikeSense Demo...")
    pose_estimator = PoseEstimator()
    smpl_handler = SMPLHandler(MODEL_FOLDER, gender=GENDER, device=DEVICE)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    renderer = Renderer(width, height)
    
    recording = False
    writer = None
    json_data = []
    frame_idx = 0

    print("=== StrikeSense Demo Running ===")
    print("[R] Record  |  [Q] Quit")
    print("[IMPROVED IK] Better rotation + joint mapping")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pose Estimation
        mp_lms, world_lms = pose_estimator.process_frame(frame)
        
        # 2. SMPL Fitting
        vertices, faces = None, None
        if world_lms:
            vertices, faces = smpl_handler.landmarks_to_smpl(world_lms)

        # 3. Render
        disp_frame = frame.copy()
        if mp_lms:
            mp.solutions.drawing_utils.draw_landmarks(
                disp_frame, mp_lms, mp.solutions.pose.POSE_CONNECTIONS)
        
        if vertices is not None:
            disp_frame = renderer.draw_mesh(disp_frame, vertices, faces)

        # 4. UI
        fps = 1.0 / (time.time() - start_time + 1e-6)
        cv2.putText(disp_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if recording:
            cv2.circle(disp_frame, (width-30, 30), 10, (0, 0, 255), -1)

        cv2.imshow("StrikeSense", disp_frame)

        # 5. Recording
        if recording and writer:
            writer.write(disp_frame)
            if world_lms:
                json_data.append({
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "landmarks": [[lm.x, lm.y, lm.z] for lm in world_lms.landmark]
                })

        # 6. Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                fname = f"output/rec_{int(time.time())}.mp4"
                os.makedirs("output", exist_ok=True)
                writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 
                                        30, (width, height))
                json_data = []
                print(f"[REC] Recording: {fname}")
            else:
                if writer:
                    writer.release()
                json_fname = f"output/rec_{int(time.time())}.json"
                with open(json_fname, 'w') as f:
                    json.dump(json_data, f)
                print(f"[SAVED] Video and JSON")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
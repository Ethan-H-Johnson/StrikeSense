"""
Live SMPL Pose Estimation from Webcam
======================================
Captures webcam feed, detects COCO keypoints with YOLOv11,
fits SMPL body model, and visualizes in real-time.

Requirements:
pip install ultralytics opencv-python torch smplx numpy scipy matplotlib

Usage:
python smpl_webcam_demo.py --smpl_path /path/to/smpl/models
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import smplx
from scipy.optimize import minimize
import argparse
import time


# ============================================================================
# COCO to SMPL Joint Mapping
# ============================================================================

COCO_TO_SMPL_DIRECT = {
    5: 16, 6: 17,   # shoulders
    7: 18, 8: 19,   # elbows
    9: 20, 10: 21,  # wrists
    11: 1, 12: 2,   # hips
    13: 4, 14: 5,   # knees
    15: 7, 16: 8    # ankles
}


# ============================================================================
# Helper Functions
# ============================================================================

def infer_missing_joints(coco_kpts):
    """
    Compute SMPL joints not directly available in COCO keypoints.
    
    Args:
        coco_kpts: (17, 2) array of COCO keypoints [x, y]
    
    Returns:
        dict mapping SMPL joint index to inferred (x, y) coordinates
    """
    inferred = {}
    
    # Pelvis = midpoint of hips
    inferred[0] = (coco_kpts[11] + coco_kpts[12]) / 2
    
    # Neck = midpoint of shoulders
    inferred[12] = (coco_kpts[5] + coco_kpts[6]) / 2
    
    # Head = nose + upward offset (10% of torso height)
    torso_height = np.linalg.norm(inferred[12] - inferred[0])
    inferred[15] = coco_kpts[0] + np.array([0, -torso_height * 0.1])
    
    # Spine interpolation (3 segments between pelvis and neck)
    pelvis_to_neck = inferred[12] - inferred[0]
    inferred[3] = inferred[0] + pelvis_to_neck * 0.33  # spine1
    inferred[6] = inferred[0] + pelvis_to_neck * 0.66  # spine2
    inferred[9] = inferred[12]                         # spine3 = neck
    
    return inferred


def prepare_target_keypoints(coco_kpts):
    """
    Map COCO keypoints to SMPL joint space (24 joints).
    
    Args:
        coco_kpts: (17, 2) COCO keypoints
    
    Returns:
        (24, 2) array for SMPL joints
    """
    target_2d = np.zeros((24, 2))
    
    # Direct mappings
    for coco_idx, smpl_idx in COCO_TO_SMPL_DIRECT.items():
        target_2d[smpl_idx] = coco_kpts[coco_idx]
    
    # Inferred joints
    inferred = infer_missing_joints(coco_kpts)
    for smpl_idx, coords in inferred.items():
        target_2d[smpl_idx] = coords
    
    return target_2d


def project_points(joints_3d, focal_length, img_width, img_height):
    """
    Weak perspective projection: 3D joints -> 2D pixel coordinates.
    
    Args:
        joints_3d: (N, 3) array in camera coordinates
        focal_length: camera focal length in pixels
        img_width, img_height: image dimensions
    
    Returns:
        (N, 2) pixel coordinates
    """
    cx = img_width / 2
    cy = img_height / 2
    
    # Avoid division by zero
    z = np.maximum(joints_3d[:, 2], 0.1)
    
    x_2d = focal_length * joints_3d[:, 0] / z + cx
    y_2d = focal_length * joints_3d[:, 1] / z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


# ============================================================================
# SMPL Fitting (Fast Version for Real-Time)
# ============================================================================

def fit_smpl_to_keypoints_fast(coco_kpts, smpl_model, camera_params, 
                                prev_params=None, max_iter=20):
    """
    Fast SMPL fitting using scipy optimizer with warm start.
    
    Args:
        coco_kpts: (17, 2) COCO keypoints
        smpl_model: smplx.SMPL instance
        camera_params: dict with focal_length, img_width, img_height
        prev_params: previous frame parameters for initialization
        max_iter: optimization iterations (lower = faster)
    
    Returns:
        dict with 'pose' (72,), 'shape' (10,), 'translation' (3,)
    """
    # Prepare target keypoints
    target_2d = prepare_target_keypoints(coco_kpts)
    
    # Define matched joints (ignore feet, hands, collars)
    matched_indices = list(COCO_TO_SMPL_DIRECT.values()) + [0, 3, 6, 9, 12, 15]
    
    # Joint weights (emphasize upper body for boxing)
    joint_weights = np.ones(24)
    joint_weights[16:22] = 3.0  # Shoulders, elbows, wrists
    joint_weights[1:3] = 2.0    # Hips
    
    def loss_fn(params):
        # Unpack parameters
        pose = torch.tensor(params[:72], dtype=torch.float32).reshape(1, -1)
        shape = torch.tensor(params[72:82], dtype=torch.float32).reshape(1, -1)
        transl = torch.tensor(params[82:85], dtype=torch.float32).reshape(1, -1)
        
        # SMPL forward pass
        with torch.no_grad():
            smpl_output = smpl_model(
                body_pose=pose[:, 3:],
                global_orient=pose[:, :3],
                betas=shape,
                transl=transl
            )
        
        joints_3d = smpl_output.joints[0].numpy()
        
        # Project to 2D
        joints_2d = project_points(
            joints_3d, 
            camera_params['focal_length'],
            camera_params['img_width'],
            camera_params['img_height']
        )
        
        # Reprojection loss (weighted)
        reproj_error = (joints_2d[matched_indices] - target_2d[matched_indices])**2
        reproj_loss = np.sum(joint_weights[matched_indices, None] * reproj_error)
        
        # Regularization
        shape_reg = 0.01 * np.sum(params[72:82]**2)
        pose_reg = 0.001 * np.sum(params[:72]**2)
        
        return reproj_loss + shape_reg + pose_reg
    
    # Initialize parameters (warm start if available)
    if prev_params is not None:
        x0 = prev_params
    else:
        init_pose = np.zeros(72)
        init_shape = np.zeros(10)
        init_transl = np.array([0, 0, 5])  # 5m in front of camera
        x0 = np.concatenate([init_pose, init_shape, init_transl])
    
    # Optimize (fast settings for real-time)
    result = minimize(
        loss_fn,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )
    
    opt_params = result.x
    return {
        'pose': opt_params[:72],
        'shape': opt_params[72:82],
        'translation': opt_params[82:85],
        'full': opt_params  # For warm start next frame
    }


# ============================================================================
# Visualization
# ============================================================================

def draw_skeleton_2d(frame, keypoints, connections=None):
    """
    Draw COCO skeleton on frame.
    
    Args:
        frame: BGR image
        keypoints: (17, 2) COCO keypoints
        connections: list of (start_idx, end_idx) tuples
    """
    # COCO skeleton connections
    if connections is None:
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for kpt in keypoints:
        cv2.circle(frame, tuple(kpt.astype(int)), 4, (0, 0, 255), -1)
    
    return frame


def draw_smpl_overlay(frame, smpl_output, camera_params):
    """
    Draw SMPL joint projections on frame.
    
    Args:
        frame: BGR image
        smpl_output: SMPL forward pass result
        camera_params: camera parameters
    """
    joints_3d = smpl_output.joints[0].detach().numpy()
    joints_2d = project_points(
        joints_3d,
        camera_params['focal_length'],
        camera_params['img_width'],
        camera_params['img_height']
    )
    
    # Draw SMPL skeleton (simplified connections)
    smpl_connections = [
        (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine
        (1, 4), (2, 5), (4, 7), (5, 8),  # Legs
        (3, 6), (6, 9), (9, 12),  # Spine to neck
        (12, 15),  # Neck to head
        (9, 16), (9, 17),  # Shoulders
        (16, 18), (18, 20),  # Left arm
        (17, 19), (19, 21)   # Right arm
    ]
    
    for start_idx, end_idx in smpl_connections:
        pt1 = tuple(joints_2d[start_idx].astype(int))
        pt2 = tuple(joints_2d[end_idx].astype(int))
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    # Draw joints
    for joint_2d in joints_2d:
        cv2.circle(frame, tuple(joint_2d.astype(int)), 3, (255, 255, 0), -1)
    
    return frame


# ============================================================================
# Main Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_path', type=str, required=True,
                        help='Path to SMPL model directory')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--yolo_model', type=str, default='yolov11n-pose.pt',
                        help='YOLOv11 pose model')
    parser.add_argument('--focal_length', type=int, default=1000,
                        help='Camera focal length (pixels)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMPL Webcam Demo - Initializing...")
    print("=" * 60)
    
    # Load models
    print("Loading YOLOv11 pose model...")
    yolo_model = YOLO('yolo11n-pose.pt')
    
    print(f"Loading SMPL model from {args.smpl_path}...")
    smpl_model = smplx.create(
        args.smpl_path,
        model_type='smpl',
        gender='neutral',
        use_pca=False,
        batch_size=1
    )
    
    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    # Get camera properties
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    camera_params = {
        'focal_length': args.focal_length,
        'img_width': img_width,
        'img_height': img_height
    }
    
    print(f"Camera: {img_width}x{img_height} @ {fps:.1f} FPS")
    print("\nPress 'q' to quit, 's' to toggle SMPL overlay")
    print("=" * 60)
    
    # State variables
    prev_params = None
    show_smpl = True
    frame_count = 0
    fps_history = []
    
    while True:
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv11 detection
        results = yolo_model(frame, verbose=False)
        
        # Process if person detected
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints
            
            if len(keypoints_data.xy) > 0:
                # Get first person's keypoints
                coco_kpts = keypoints_data.xy[0].cpu().numpy()  # (17, 2)
                
                # Draw COCO skeleton
                frame = draw_skeleton_2d(frame, coco_kpts)
                
                # Fit SMPL (every frame)
                try:
                    smpl_params = fit_smpl_to_keypoints_fast(
                        coco_kpts,
                        smpl_model,
                        camera_params,
                        prev_params,
                        max_iter=10  # Lower for speed
                    )
                    prev_params = smpl_params['full']
                    
                    # Visualize SMPL
                    if show_smpl:
                        pose = torch.tensor(smpl_params['pose']).reshape(1, -1)
                        shape = torch.tensor(smpl_params['shape']).reshape(1, -1)
                        transl = torch.tensor(smpl_params['translation']).reshape(1, -1)
                        
                        with torch.no_grad():
                            smpl_output = smpl_model(
                                body_pose=pose[:, 3:],
                                global_orient=pose[:, :3],
                                betas=shape,
                                transl=transl
                            )
                        
                        frame = draw_smpl_overlay(frame, smpl_output, camera_params)
                
                except Exception as e:
                    cv2.putText(frame, f"SMPL fit error: {str(e)[:30]}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        
        # Display info
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Green=COCO | Blue=SMPL" if show_smpl else "Green=COCO only", 
                    (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('SMPL Webcam Demo', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_smpl = not show_smpl
            print(f"SMPL overlay: {'ON' if show_smpl else 'OFF'}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames. Average FPS: {avg_fps:.1f}")


if __name__ == '__main__':
    main()
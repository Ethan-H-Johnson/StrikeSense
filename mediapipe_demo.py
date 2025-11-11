import cv2
import mediapipe as mp
import numpy as np
import math
import time

"""
TODO: 
# Lock on to specific fighter/target
"""


# --- Configuration ---
# IMPORTANT: Replace 'your_boxing_video.mp4' with the actual filename of your video.
VIDEO_FILE = 'data\clips\mayweather_jab.mp4'
# ---------------------

# --- Helper Function to Calculate Angle ---
def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three 3D points (a, b, c), 
    where 'b' is the vertex of the angle."""
    
    # Check if all points are available (not 0,0,0) and visible (visibility > 0.5)
    if any(p.visibility < 0.5 for p in [a, b, c]):
        return None # Indicate low confidence/not visible
        
    a_xyz = np.array([a.x, a.y])
    b_xyz = np.array([b.x, b.y])
    c_xyz = np.array([c.x, c.y])
    
    # Calculate vectors
    ba = a_xyz - b_xyz
    bc = c_xyz - b_xyz
    
    # Calculate cosine of angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Ensure value is in the valid domain for arccos ([-1, 1])
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Use the 'with' statement for the Pose model
with mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_FILE}")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect (optional, often better for demos)
        # frame = cv2.flip(frame, 1) 

        # Convert the BGR frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect the pose
        results = pose.process(image)

        # Re-mark the image as writeable and convert back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- DEMO-SPECIFIC LOGIC ---
        
        elbow_angle_text = "Elbow Angle: N/A"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- Keypoint Calculation (Example: Right Elbow Angle) ---
            
            # Get the three points for the right elbow angle
            # Shoulder (12), Elbow (14), Wrist (16)
            try:
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # Calculate the angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Format the display text
                if angle is not None:
                    # Determine color for visual feedback (e.g., tight hook is < 90 deg)
                    color = (0, 0, 255) if angle > 95 else (0, 255, 0) # Red for bad, Green for good (approx)
                    elbow_angle_text = f"Elbow Angle: {angle:.1f}°"
                    
                    # Draw a colored circle on the elbow for emphasis
                    h, w, c = image.shape
                    cx, cy = int(elbow.x * w), int(elbow.y * h)
                    cv2.circle(image, (cx, cy), 10, color, -1)

            except Exception as e:
                # Handle cases where landmarks might be missing
                # print(f"Error calculating angle: {e}") 
                pass

            # Draw the pose skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # --- Display the Feature Output on the Video ---
        
        # Add the feature text to the top-left corner
        cv2.putText(image, elbow_angle_text, 
                    (50, 50),                 # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5,                      # Font scale
                    (255, 255, 255),          # White text color
                    3,                        # Thickness
                    cv2.LINE_AA)
        
        # --- Display the Video ---
        cv2.imshow('StrikeSense MVP Demo - Real-Time Analysis', image)
        
        # Press 'q' to break the loop and stop the video
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

print("Demo complete. Press any key to exit.")
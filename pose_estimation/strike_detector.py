"""
StrikeDetector - Logic for detecting and classifying Jab/Cross strikes
Includes OneEuro Filter for signal smoothing
"""
import numpy as np
import math
import time

class OneEuroFilter:
    """
    Adaptive smoothing filter for real-time applications.
    Minimizes jitter at low speeds, minimizes lag at high speeds.
    """
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0:
            return self.x_prev

        # Estimate derivative (velocity)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Calculate cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class StrikeDetector:
    """
    Detects Jab and Cross strikes using velocity and biomechanics.
    """
    def __init__(self):
        # Configuration
        self.VELOCITY_THRESH = 1.5  # m/s (approx)
        self.ELBOW_ANGLE_THRESH = 140 # degrees
        self.STANCE_CONFIDENCE_THRESH = 0.1 # m (Z-diff)
        
        # State
        self.stance = "orthodox" # Default
        self.prev_landmarks = None
        self.prev_time = None
        
        # Filters for key joints (L_Wrist, R_Wrist)
        # x, y, z for each hand
        self.filters = {
            'left': [OneEuroFilter(), OneEuroFilter(), OneEuroFilter()],
            'right': [OneEuroFilter(), OneEuroFilter(), OneEuroFilter()]
        }
        
        # Strike State Machine
        self.cooldown = 0
        self.last_strike_time = 0

    def _get_z(self, landmarks, idx):
        return landmarks.landmark[idx].z

    def _detect_stance(self, landmarks, is_striking=False):
        """
        Determine stance using Hybrid approach:
        1. Knees (Primary) - Robust to rotation
        2. Shoulders (Fallback) - Robust to occlusion
        3. Stance Lock - Prevents flipping during strikes
        """
        # If striking, LOCK the stance (prevent rotation flip)
        if is_striking:
            return self.stance

        # Get joints
        l_knee = landmarks.landmark[25]
        r_knee = landmarks.landmark[26]
        l_shoulder = landmarks.landmark[11]
        r_shoulder = landmarks.landmark[12]
        
        # Check Knee Visibility (Confidence > 0.6)
        knees_visible = (l_knee.visibility > 0.6 and r_knee.visibility > 0.6)
        
        diff = 0
        if knees_visible:
            # Use Knees (Best accuracy)
            diff = l_knee.z - r_knee.z
        else:
            # Fallback to Shoulders
            diff = l_shoulder.z - r_shoulder.z
        
        # Only update if difference is significant (Robustness)
        if abs(diff) > self.STANCE_CONFIDENCE_THRESH:
            if diff < 0: # Left is closer (smaller Z)
                self.stance = "orthodox"
            else:
                self.stance = "southpaw"
        
        return self.stance

    def _calculate_velocity(self, current, prev, dt):
        """Calculate 3D velocity magnitude"""
        dist = math.sqrt(
            (current.x - prev.x)**2 + 
            (current.y - prev.y)**2 + 
            (current.z - prev.z)**2
        )
        return dist / dt if dt > 0 else 0

    def _calculate_angle(self, a, b, c):
        """Calculate angle at b (a-b-c)"""
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def process(self, landmarks, timestamp):
        """
        Main processing loop.
        Returns: 'jab', 'cross', or None
        """
        if landmarks is None:
            return None, self.stance

        # 1. Smooth Wrist Coordinates
        l_wrist_raw = landmarks.landmark[15]
        r_wrist_raw = landmarks.landmark[16]
        
        l_wrist = type('obj', (object,), {'x':0, 'y':0, 'z':0})
        r_wrist = type('obj', (object,), {'x':0, 'y':0, 'z':0})
        
        l_wrist.x = self.filters['left'][0].filter(l_wrist_raw.x, timestamp)
        l_wrist.y = self.filters['left'][1].filter(l_wrist_raw.y, timestamp)
        l_wrist.z = self.filters['left'][2].filter(l_wrist_raw.z, timestamp)
        
        r_wrist.x = self.filters['right'][0].filter(r_wrist_raw.x, timestamp)
        r_wrist.y = self.filters['right'][1].filter(r_wrist_raw.y, timestamp)
        r_wrist.z = self.filters['right'][2].filter(r_wrist_raw.z, timestamp)

        # 2. Calculate Metrics
        result = None
        is_striking = False
        
        if self.prev_landmarks and self.prev_time:
            dt = timestamp - self.prev_time
            
            # Velocity (using internal OneEuro derivative estimate)
            l_vel = math.sqrt(self.filters['left'][0].dx_prev**2 + 
                              self.filters['left'][1].dx_prev**2 + 
                              self.filters['left'][2].dx_prev**2)
            
            r_vel = math.sqrt(self.filters['right'][0].dx_prev**2 + 
                              self.filters['right'][1].dx_prev**2 + 
                              self.filters['right'][2].dx_prev**2)

            # Check if striking (for stance lock)
            if l_vel > 1.0 or r_vel > 1.0:
                is_striking = True

            # Elbow Angles
            l_elbow_angle = self._calculate_angle(
                landmarks.landmark[11], landmarks.landmark[13], landmarks.landmark[15])
            r_elbow_angle = self._calculate_angle(
                landmarks.landmark[12], landmarks.landmark[14], landmarks.landmark[16])

            # 3. Detect Stance (with Lock)
            current_stance = self._detect_stance(landmarks, is_striking)

            # 4. Strike Detection Logic
            # Cooldown check
            if (timestamp - self.last_strike_time) > 0.3:
                
                # Determine Lead/Rear hand
                if current_stance == "orthodox":
                    lead_vel, lead_angle = l_vel, l_elbow_angle
                    rear_vel, rear_angle = r_vel, r_elbow_angle
                    lead_hand, rear_hand = "left", "right"
                else: # Southpaw
                    lead_vel, lead_angle = r_vel, r_elbow_angle
                    rear_vel, rear_angle = l_vel, l_elbow_angle
                    lead_hand, rear_hand = "right", "left"

                # Check Thresholds
                if lead_vel > self.VELOCITY_THRESH and lead_angle > self.ELBOW_ANGLE_THRESH:
                    result = 'jab'
                    self.last_strike_time = timestamp
                
                elif rear_vel > self.VELOCITY_THRESH and rear_angle > self.ELBOW_ANGLE_THRESH:
                    result = 'cross'
                    self.last_strike_time = timestamp
        else:
            # First frame
            current_stance = self._detect_stance(landmarks, False)

        self.prev_landmarks = landmarks
        self.prev_time = timestamp
        
        return result, current_stance

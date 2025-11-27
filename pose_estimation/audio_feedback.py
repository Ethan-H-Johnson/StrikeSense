"""
AudioFeedback - Handles real-time audio cues for strikes
Uses separate thread to prevent blocking the main video loop
"""
import winsound
import threading
import time

class AudioFeedback:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_play_time = 0
        self._min_interval = 0.2  # Minimum seconds between sounds

    def _play_tone(self, frequency, duration):
        """Play sound in a separate thread"""
        def run():
            with self._lock:
                # Prevent overlapping sounds
                now = time.time()
                if now - self._last_play_time < self._min_interval:
                    return
                self._last_play_time = now
                
            try:
                winsound.Beep(frequency, duration)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")

        threading.Thread(target=run, daemon=True).start()

    def play_jab(self):
        """High pitch, short beep for Jab"""
        # 1000 Hz, 100 ms
        self._play_tone(1000, 100)

    def play_cross(self):
        """Lower pitch, slightly longer beep for Cross"""
        # 600 Hz, 150 ms
        self._play_tone(600, 150)

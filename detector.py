import numpy as np

class EyeStrainDetector:
    def __init__(self):
        self.blinks = 0
        self.last_state = "open"

    def process_dummy(self):
        """
        Since we cannot use OpenCV or MediaPipe in Streamlit Cloud,
        we simulate eye strain values.
        """
        ear = np.random.uniform(0.18, 0.35)
        return ear

    def get_status(self, ear):
        if ear < 0.22:
            status = "HIGH STRAIN"
            self.blinks += 1
        else:
            status = "OPTIMAL"

        return status, self.blinks

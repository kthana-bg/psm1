import cv2
import numpy as np
import threading

class EyeStrainDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self._blink_count = 0
        self._eye_closed = False
        self._lock = threading.Lock()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        ear = 0.0
        annotated = frame.copy()

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = annotated[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            eye_count = len(eyes)

            # Simple EAR approximation (based on eye visibility)
            ear = eye_count / 2.0

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        return ear, faces, annotated

    def update_blink_state(self, ear, threshold=0.5):
        with self._lock:
            # eye closed detection
            if ear < threshold and not self._eye_closed:
                self._eye_closed = True

            elif ear >= threshold and self._eye_closed:
                self._blink_count += 1
                self._eye_closed = False

            return self._blink_count

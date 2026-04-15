import numpy as np

class EyeStrainDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.blinks = 0
        self.eye_closed = False

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        eye_count = 0

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi)
            eye_count = len(eyes)

        # simple EAR approximation
        ear = eye_count / 2.0

        return ear

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

@dataclass
class EyeMetrics:
    ear: float
    blink_count: int
    status: str
    timestamp: float

class EyeStrainDetector:
    def __init__(self, ear_threshold: float = 0.25):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.blink_active = False
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
    def calculate_ear(self, landmarks: List[List[float]], eye_indices: List[int]) -> float:
        try:
            p = [np.array(landmarks[i]) for i in eye_indices]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h = np.linalg.norm(p[0] - p[3])
            return (v1 + v2) / (2.0 * h) if h != 0 else 0.0
        except:
            return 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return 0.0, None
            
        landmarks = [[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark]
        
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        self._update_blink(ear)
        
        annotated = self._draw_landmarks(frame.copy(), landmarks)
        return ear, annotated
    
    def _update_blink(self, ear: float):
        if ear < self.ear_threshold and not self.blink_active:
            self.blink_active = True
        elif ear >= self.ear_threshold and self.blink_active:
            self.blink_count += 1
            self.blink_active = False
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: List[List[float]]) -> np.ndarray:
        h, w = frame.shape[:2]
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        return frame
    
    def get_metrics(self, ear: float) -> EyeMetrics:
        status = "OPTIMAL" if ear >= self.ear_threshold else "HIGH STRAIN" if ear > 0 else "NO FACE"
        return EyeMetrics(ear, self.blink_count, status, time.time())

class VideoProcessor:
    def __init__(self, detector: EyeStrainDetector):
        self.detector = detector
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, EyeMetrics]:
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        if self.frame_count % 2 == 0:
            small = cv2.resize(frame, (320, 240))
            ear, annotated_small = self.detector.process_frame(small)
            if annotated_small is not None:
                annotated = cv2.resize(annotated_small, (frame.shape[1], frame.shape[0]))
            else:
                annotated = frame
        else:
            ear = 0.0
            annotated = frame
        
        metrics = self.detector.get_metrics(ear)
        
        h, w = annotated.shape[:2]
        color = (50, 255, 150) if metrics.status == "OPTIMAL" else (255, 50, 50) if metrics.status == "HIGH STRAIN" else (128, 128, 128)
        cv2.rectangle(annotated, (0, 0), (w-1, h-1), color, 3)
        cv2.putText(annotated, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated, metrics

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-attachment: fixed;
    height: 100vh;
    overflow: hidden;
}
.block-container { padding: 0.5rem 1rem; height: 100vh; }
h1 { color: #E0B0FF !important; font-weight: 300 !important; text-align: center; margin: 0; }
.metric-value { font-size: 36px; color: #BB86FC; text-align: center; font-weight: bold; }
.metric-label { font-size: 10px; opacity: 0.7; text-align: center; text-transform: uppercase; }
.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    margin-bottom: 10px;
}
.status-optimal { color: #00E676 !important; }
.status-danger { color: #FF1744 !important; }
.status-warning { color: #FFD600 !important; }
video { width: 100% !important; border-radius: 16px !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)
if "detector" not in st.session_state:
    st.session_state.detector = EyeStrainDetector()
if "processor" not in st.session_state:
    st.session_state.processor = VideoProcessor(st.session_state.detector)

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col2:
    st.subheader("Analytics")
    
    ear_placeholder = st.empty()
    blink_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    coach_placeholder = st.empty()
    
    with st.expander("Settings"):
        new_threshold = st.slider("EAR Threshold", 0.15, 0.30, 0.25, 0.01)
        st.session_state.detector.ear_threshold = new_threshold
        if st.button("Reset", width="stretch"):
            st.session_state.detector.blink_count = 0
            st.session_state.history = deque([0.25] * 40, maxlen=40)

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()

def get_frame():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

if st.button("Start Camera", key="start"):
    st.session_state.camera_active = True

if st.button("Stop Camera", key="stop"):
    st.session_state.camera_active = False
    if "cap" in st.session_state:
        st.session_state.cap.release()

if st.session_state.get("camera_active", False):
    if "cap" not in st.session_state or not st.session_state.cap.isOpened():
        st.session_state.cap = get_frame()
    
    ret, frame = st.session_state.cap.read()
    if ret:
        annotated, metrics = st.session_state.processor.process(frame)
        
        st.session_state.history.append(metrics.ear)
        
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        status_class = "status-optimal" if metrics.status == "OPTIMAL" else "status-danger" if metrics.status == "HIGH STRAIN" else "status-warning"
        
        ear_placeholder.markdown(f"<div class='card'><p class='metric-label'>Current EAR</p><div class='metric-value {status_class}'>{metrics.ear:.3f}</div></div>", unsafe_allow_html=True)
        blink_placeholder.markdown(f"<div class='card'><p class='metric-label'>Total Blinks</p><div class='metric-value' style='color: #BB86FC;'>{metrics.blink_count}</div></div>", unsafe_allow_html=True)
        status_placeholder.markdown(f"<div class='card'><p class='metric-label'>Status</p><div class='metric-value {status_class}' style='font-size: 20px;'>{metrics.status}</div></div>", unsafe_allow_html=True)
        
        chart_placeholder.line_chart(list(st.session_state.history), height=120, width="stretch")
        
        if metrics.status == "HIGH STRAIN":
            coach_placeholder.error("Eye strain detected. Take a 20-20-20 break.")
        elif metrics.status == "NO FACE":
            coach_placeholder.warning("Position your face in camera view.")
        else:
            coach_placeholder.success("Monitoring active. Blink regularly.")
        
        time.sleep(0.03)
        st.rerun()
    else:
        st.error("Camera error. Check permissions.")
else:
    frame_placeholder.info("Click 'Start Camera' to begin monitoring.")

st.markdown("<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)

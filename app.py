import streamlit as st
import numpy as np
import time
import math
from typing import Optional, Tuple
from dataclasses import dataclass
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
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.blink_active = False
        self.frame_count = 0
        
    def process_frame(self, frame_bytes: Optional[bytes] = None) -> Tuple[float, str]:
        self.frame_count += 1
        
        # Simulated EAR logic for FYP logic testing
        base_ear = 0.28 + 0.05 * math.sin(self.frame_count * 0.1)
        noise = np.random.normal(0, 0.02)
        ear = max(0.15, min(0.35, base_ear + noise))
        
        if np.random.random() < 0.02: 
            ear = np.random.uniform(0.10, 0.18)
        
        self._update_blink(ear)
        
        if ear >= self.ear_threshold:
            status = "OPTIMAL"
        elif ear > 0.18:
            status = "HIGH STRAIN"
        else:
            status = "BLINKING"
            
        return ear, status
    
    def _update_blink(self, ear: float):
        if ear < 0.20 and not self.blink_active:
            self.blink_active = True
        elif ear >= 0.22 and self.blink_active:
            self.blink_count += 1
            self.blink_active = False

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
.block-container { padding: 0.5rem 1rem; height: 100vh; overflow: hidden; }
h1 { color: #E0B0FF !important; font-weight: 300 !important; text-align: center; margin: 0; font-size: 1.5rem; }
p { text-align: center; color: #B0B0B0; font-size: 0.8rem; margin: 0; }
.metric-value { font-size: 32px; color: #BB86FC; text-align: center; font-weight: bold; }
.metric-label { font-size: 10px; opacity: 0.7; text-align: center; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.card {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}
.status-optimal { color: #00E676 !important; }
.status-danger { color: #FF1744 !important; }
.status-warning { color: #FFD600 !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

if "detector" not in st.session_state:
    st.session_state.detector = EyeStrainDetector()
if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col2:
    st.subheader("Analytics")
    
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_display = st.empty()
    
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    blink_display = st.empty()
    
    st.markdown('<p class="metric-label">System Status</p>', unsafe_allow_html=True)
    status_display = st.empty()
    
    st.divider()
    st.markdown('<p class="metric-label">EAR History</p>', unsafe_allow_html=True)
    chart_display = st.empty()
    
    st.divider()
    st.subheader("Coach")
    coach_display = st.empty()
    
    with st.expander("Settings"):
        new_threshold = st.slider("EAR Threshold", 0.15, 0.30, st.session_state.detector.ear_threshold, 0.01)
        st.session_state.detector.ear_threshold = new_threshold
        if st.button("Reset Stats", use_container_width=True):
            st.session_state.detector.blink_count = 0
            st.session_state.history = deque([0.25] * 40, maxlen=40)
            st.rerun()

with col1:
    st.subheader("Live Feed")
    camera_image = st.camera_input("Capture", label_visibility="collapsed", key="camera")
    
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        
        # The analytics magic: update detector state
        ear, status = st.session_state.detector.process_frame(bytes_data)
        st.session_state.history.append(ear)
        
        status_class = "status-optimal" if status == "OPTIMAL" else "status-danger" if status == "HIGH STRAIN" else "status-warning"
        
        # Populate analytics placeholders
        ear_display.markdown(f"<div class='card'><div class='metric-value {status_class}'>{ear:.3f}</div></div>", unsafe_allow_html=True)
        blink_display.markdown(f"<div class='card'><div class='metric-value' style='color: #BB86FC;'>{st.session_state.detector.blink_count}</div></div>", unsafe_allow_html=True)
        status_display.markdown(f"<div class='card'><div class='metric-value {status_class}' style='font-size: 18px;'>{status}</div></div>", unsafe_allow_html=True)
        
        chart_display.line_chart(list(st.session_state.history), height=150)
        
        if status == "HIGH STRAIN":
            coach_display.error("Eye strain detected. Take a 20-20-20 break.")
        elif status == "BLINKING":
            coach_display.info("Blink detected! Keep it up.")
        else:
            coach_display.success("System active. Remember to blink regularly.")
            
    else:
        ear_display.markdown("<div class='card'><div class='metric-value'>--.---</div></div>", unsafe_allow_html=True)
        blink_display.markdown("<div class='card'><div class='metric-value' style='color: #BB86FC;'>0</div></div>", unsafe_allow_html=True)
        status_display.markdown("<div class='card'><div class='metric-value' style='font-size: 18px;'>WAITING</div></div>", unsafe_allow_html=True)
        coach_display.info("Click 'Take Photo' to update analytics.")

st.markdown("<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)

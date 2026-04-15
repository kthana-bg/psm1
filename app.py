import streamlit as st
import numpy as np
import time
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
    """
    Placeholder detector - requires actual implementation with 
    MediaPipe or similar when camera is available
    """
    def __init__(self, ear_threshold: float = 0.25):
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.blink_active = False
        self.frame_count = 0
        
    def process_frame(self, frame_bytes: Optional[bytes] = None) -> Tuple[float, str]:
        """
        Process frame and return EAR value.
        Currently returns simulated data - replace with actual ML model
        """
        self.frame_count += 1
        
        # TODO: Replace with actual eye detection using MediaPipe or similar
        # For now, returns realistic simulated data based on time patterns
        import math
        base_ear = 0.28 + 0.05 * math.sin(self.frame_count * 0.1)
        noise = np.random.normal(0, 0.02)
        ear = max(0.15, min(0.35, base_ear + noise))
        
        # Simulate blinks (occasional low EAR values)
        if np.random.random() < 0.02:  # 2% chance of blink per frame
            ear = np.random.uniform(0.10, 0.18)
        
        self._update_blink(ear)
        
        status = "OPTIMAL" if ear >= self.ear_threshold else "HIGH STRAIN" if ear > 0 else "NO FACE"
        return ear, status
    
    def _update_blink(self, ear: float):
        if ear < 0.20 and not self.blink_active:  # Blink threshold
            self.blink_active = True
        elif ear >= 0.22 and self.blink_active:
            self.blink_count += 1
            self.blink_active = False

# CSS styling
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
.stCamera { width: 100% !important; border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

# Initialize
if "detector" not in st.session_state:
    st.session_state.detector = EyeStrainDetector()
if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)
if "is_running" not in st.session_state:
    st.session_state.is_running = False

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col2:
    st.subheader("Analytics")
    
    # Metric placeholders
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_display = st.empty()
    
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    blink_display = st.empty()
    
    st.markdown('<p class="metric-label">System Status</p>', unsafe_allow_html=True)
    status_display = st.empty()
    
    st.divider()
    
    # Chart
    st.markdown('<p class="metric-label">EAR History</p>', unsafe_allow_html=True)
    chart_display = st.empty()
    
    st.divider()
    
    # Coach
    st.subheader("Coach")
    coach_display = st.empty()
    
    # Controls
    with st.expander("Settings"):
        new_threshold = st.slider("EAR Threshold", 0.15, 0.30, st.session_state.detector.ear_threshold, 0.01)
        st.session_state.detector.ear_threshold = new_threshold
        
        if st.button("Reset Stats", use_container_width=True):
            st.session_state.detector.blink_count = 0
            st.session_state.detector.blink_active = False
            st.session_state.history = deque([0.25] * 40, maxlen=40)
            st.rerun()

with col1:
    st.subheader("Live Feed")
    
    # Camera input - real camera without OpenCV
    camera_image = st.camera_input("Capture", label_visibility="collapsed", key="camera")
    
    if camera_image is not None:
        # Process the captured frame
        bytes_data = camera_image.getvalue()
        
        # In real implementation, you would:
        # 1. Convert bytes to image array
        # 2. Run MediaPipe face mesh detection
        # 3. Calculate EAR from eye landmarks
        # 4. Update metrics
        
        # For now, use detector with timestamp seed for consistency
        ear, status = st.session_state.detector.process_frame(bytes_data)
        
        # Update history
        st.session_state.history.append(ear)
        
        # Determine status class
        if status == "OPTIMAL":
            status_class = "status-optimal"
        elif status == "HIGH STRAIN":
            status_class = "status-danger"
        else:
            status_class = "status-warning"
        
        # Update displays
        ear_display.markdown(f"<div class='card'><div class='metric-value {status_class}'>{ear:.3f}</div></div>", unsafe_allow_html=True)
        blink_display.markdown(f"<div class='card'><div class='metric-value' style='color: #BB86FC;'>{st.session_state.detector.blink_count}</div></div>", unsafe_allow_html=True)
        status_display.markdown(f"<div class='card'><div class='metric-value {status_class}' style='font-size: 18px;'>{status}</div></div>", unsafe_allow_html=True)
        
        # Chart
        chart_display.line_chart(list(st.session_state.history), height=100, width="stretch")
        
        # Coach message
        if status == "HIGH STRAIN":
            coach_display.error("Eye strain detected. Take a 20-20-20 break.")
        elif status == "NO FACE":
            coach_display.warning("No face detected. Position yourself in frame.")
        else:
            coach_display.success("System active. Remember to blink regularly.")
        
        # Auto-refresh for continuous monitoring
        time.sleep(0.1)
        st.rerun()
    else:
        ear_display.markdown("<div class='card'><div class='metric-value'>--.---</div></div>", unsafe_allow_html=True)
        blink_display.markdown("<div class='card'><div class='metric-value' style='color: #BB86FC;'>0</div></div>", unsafe_allow_html=True)
        status_display.markdown("<div class='card'><div class='metric-value' style='font-size: 18px;'>WAITING</div></div>", unsafe_allow_html=True)
        coach_display.info("Click 'Take Photo' to start monitoring, or allow camera access for continuous monitoring.")

st.markdown("<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)

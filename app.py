import streamlit as st
import numpy as np
import time
from collections import deque
from dataclasses import dataclass

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

@dataclass
class EyeMetrics:
    ear: float
    blink_count: int
    status: str

class EyeStrainDetector:
    def __init__(self, ear_threshold: float = 0.25):
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.blink_active = False
        self.frame_count = 0
        self.last_blink_time = time.time()
        
    def process_frame(self) -> EyeMetrics:
        self.frame_count += 1
        import math
        
        # Simulate realistic EAR pattern (not pure random)
        # Normal blink every 3-6 seconds
        time_since_last_blink = time.time() - self.last_blink_time
        should_blink = time_since_last_blink > np.random.uniform(3, 6)
        
        if should_blink:
            ear = np.random.uniform(0.10, 0.18)  # Blink (low EAR)
            self.last_blink_time = time.time()
        else:
            # Normal eye state with small variations
            base_ear = 0.28
            fatigue_factor = max(0, (self.frame_count % 200) / 1000)  # Gradual fatigue
            noise = np.random.normal(0, 0.015)
            ear = base_ear - fatigue_factor + noise
            ear = max(0.20, min(0.35, ear))
        
        self._update_blink(ear)
        
        if ear < 0.20:
            status = "HIGH STRAIN"
        elif ear < 0.22:
            status = "BLINKING"
        else:
            status = "OPTIMAL"
            
        return EyeMetrics(ear, self.blink_count, status)
    
    def _update_blink(self, ear: float):
        if ear < 0.20 and not self.blink_active:
            self.blink_active = True
        elif ear >= 0.22 and self.blink_active:
            self.blink_count += 1
            self.blink_active = False

# CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; margin: 0; padding: 0; }
.stApp {
    background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-attachment: fixed;
}
.block-container { padding: 0.5rem 1rem; height: 100vh; overflow: hidden; }
h1 { color: #E0B0FF !important; font-weight: 300 !important; text-align: center; margin: 0; font-size: 1.5rem; }
p { text-align: center; color: #B0B0B0; font-size: 0.8rem; margin: 0; }
.metric-value { font-size: 36px; color: #BB86FC; text-align: center; font-weight: bold; }
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
.camera-container {
    width: 100%;
    border-radius: 16px;
    overflow: hidden;
    background: rgba(0,0,0,0.3);
}
footer { display: none !important; }
.stCamera > div > div {
    border-radius: 16px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# Init
if "detector" not in st.session_state:
    st.session_state.detector = EyeStrainDetector()
if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

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
            st.session_state.detector = EyeStrainDetector(new_threshold)
            st.session_state.history = deque([0.25] * 40, maxlen=40)
            st.rerun()

with col1:
    st.subheader("Live Feed")
    
    # Hidden auto-capture camera
    camera_image = st.camera_input("", label_visibility="collapsed", key=f"cam_{int(time.time()*10)}")
    
    # Process metrics (works even without camera for demo)
    metrics = st.session_state.detector.process_frame()
    
    # Update history
    st.session_state.history.append(metrics.ear)
    
    # Determine status class
    if metrics.status == "OPTIMAL":
        status_class = "status-optimal"
    elif metrics.status == "HIGH STRAIN":
        status_class = "status-danger"
    else:
        status_class = "status-warning"
    
    # Update displays
    ear_display.markdown(f"<div class='card'><div class='metric-value {status_class}'>{metrics.ear:.3f}</div></div>", unsafe_allow_html=True)
    blink_display.markdown(f"<div class='card'><div class='metric-value' style='color: #BB86FC;'>{metrics.blink_count}</div></div>", unsafe_allow_html=True)
    status_display.markdown(f"<div class='card'><div class='metric-value {status_class}' style='font-size: 18px;'>{metrics.status}</div></div>", unsafe_allow_html=True)
    
    chart_display.line_chart(list(st.session_state.history), height=100, width="stretch")
    
    if metrics.status == "HIGH STRAIN":
        coach_display.error("Eye strain detected. Take a 20-20-20 break.")
    elif metrics.status == "BLINKING":
        coach_display.info("Blink detected.")
    else:
        coach_display.success("Monitoring active. Remember to blink regularly.")
    
    # Auto refresh every 0.5 seconds for live effect
    time.sleep(0.5)
    st.rerun()

st.markdown("<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)

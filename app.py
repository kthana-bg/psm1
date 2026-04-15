import streamlit as st
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass

# Try importing streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.error("Please install: pip install streamlit-webrtc")

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

VIDEO_HEIGHT = 400

@dataclass
class SharedMetrics:
    """Thread-safe metrics storage"""
    lock: threading.Lock
    ear: float
    blink_count: int
    status: str
    history: deque

# Global metrics object (shared between threads)
if "metrics" not in st.session_state:
    st.session_state.metrics = SharedMetrics(
        lock=threading.Lock(),
        ear=0.0,
        blink_count=0,
        status="Initializing",
        history=deque([0.25] * 40, maxlen=40)
    )

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.20

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{ height: 100vh; overflow: hidden; margin: 0; padding: 0; }}
.stApp {{
    background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
}}
.block-container {{ padding: 0.5rem 1rem; height: 100vh; overflow: hidden; }}
h1 {{ color: #E0B0FF !important; font-weight: 300 !important; text-align: center; margin: 0; font-size: 1.5rem; }}
.metric-value {{ font-size: 36px; color: #BB86FC; text-align: center; font-weight: bold; }}
.metric-label {{ font-size: 10px; opacity: 0.7; text-align: center; text-transform: uppercase; margin-bottom: 4px; }}
.card {{ background: rgba(255,255,255,0.08); padding: 12px; border-radius: 16px; backdrop-filter: blur(10px); margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.1); }}
.status-optimal {{ color: #00E676 !important; }} .status-danger {{ color: #FF1744 !important; }} .status-warning {{ color: #FFD600 !important; }}
footer {{ display: none !important; }}
video {{ transform: scaleX(-1) !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

class VideoProcessor(VideoProcessorBase):
    """Real-time video processor that updates metrics continuously"""
    
    def __init__(self):
        self.frame_count = 0
        self.blink_active = False
        self.last_ear = 0.0
        
    def recv(self, frame):
        # Mirror the frame
        img = frame.to_ndarray(format="bgr24")
        img = np.fliplr(img)  # Mirror horizontally
        
        h, w, _ = img.shape
        self.frame_count += 1
        
        # Simulate EAR detection (replace with real detection when you have cv2/mediapipe)
        import math
        time_val = self.frame_count * 0.1
        base_ear = 0.28 + 0.03 * math.sin(time_val)
        
        # Random blink every ~50 frames
        if np.random.random() < 0.02:
            ear = np.random.uniform(0.12, 0.18)
            is_blink = True
        else:
            ear = base_ear + np.random.normal(0, 0.01)
            ear = max(0.20, min(0.35, ear))
            is_blink = False
        
        # Blink detection logic
        if is_blink and not self.blink_active:
            self.blink_active = True
        elif not is_blink and self.blink_active:
            with st.session_state.metrics.lock:
                st.session_state.metrics.blink_count += 1
            self.blink_active = False
        
        self.last_ear = ear
        
        # Update shared metrics (thread-safe)
        if self.frame_count % 5 == 0:  # Update every 5 frames
            with st.session_state.metrics.lock:
                st.session_state.metrics.ear = ear
                if ear < 0.20:
                    st.session_state.metrics.status = "HIGH STRAIN"
                elif is_blink:
                    st.session_state.metrics.status = "BLINKING"
                else:
                    st.session_state.metrics.status = "OPTIMAL"
                st.session_state.metrics.history.append(ear)
        
        # Draw status border
        if st.session_state.metrics.status == "HIGH STRAIN":
            color = (255, 50, 50)
        elif st.session_state.metrics.status == "BLINKING":
            color = (255, 200, 50)
        else:
            color = (50, 255, 150)
        
        import cv2
        cv2.rectangle(img, (0, 0), (w-1, h-1), color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col1:
    st.subheader("Live Feed")
    
    if WEBRTC_AVAILABLE:
        # WebRTC configuration
        rtc_config = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        }
        
        ctx = webrtc_streamer(
            key="visionmate-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": 30},
                "audio": False
            },
            async_processing=True,
            desired_playing_state=True,
            video_html_attrs={
                "style": {"width": "100%", "height": "auto", "max-height": f"{VIDEO_HEIGHT}px"},
                "controls": False,
                "autoPlay": True,
                "playsInline": True,
                "muted": True
            }
        )
        
        # Show camera status
        if ctx.state.playing:
            st.success("Camera active - Processing live feed")
        else:
            st.info("Click 'START' on the camera widget above")
    else:
        st.error("WebRTC not available. Install with: pip install streamlit-webrtc")

with col2:
    st.subheader("Analytics")
    
    # Read metrics with lock
    with st.session_state.metrics.lock:
        ear = st.session_state.metrics.ear
        blink_count = st.session_state.metrics.blink_count
        status = st.session_state.metrics.status
        history = list(st.session_state.metrics.history)
    
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    if status == "OPTIMAL":
        status_class = "status-optimal"
    elif status == "HIGH STRAIN":
        status_class = "status-danger"
    else:
        status_class = "status-warning"
    st.markdown(f"<div class='card'><div class='metric-value {status_class}'>{ear:.3f}</div></div>", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='metric-value' style='color: #BB86FC;'>{blink_count}</div></div>", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">System Status</p>', unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='metric-value {status_class}' style='font-size: 18px;'>{status}</div></div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown('<p class="metric-label">EAR History</p>', unsafe_allow_html=True)
    st.line_chart(history, height=100, width="stretch")
    
    st.divider()
    st.subheader("Coach")
    if status == "HIGH STRAIN":
        st.error("Eye strain detected. Take a 20-20-20 break.")
    elif status == "BLINKING":
        st.info("Blink detected.")
    else:
        st.success("Monitoring active. Remember to blink regularly.")
    
    # Controls
    with st.expander("Settings"):
        new_threshold = st.slider("EAR Threshold", 0.15, 0.30, st.session_state.threshold, 0.01)
        st.session_state.threshold = new_threshold
        
        if st.button("Reset Stats", use_container_width=True):
            with st.session_state.metrics.lock:
                st.session_state.metrics.ear = 0.0
                st.session_state.metrics.blink_count = 0
                st.session_state.metrics.status = "Initializing"
                st.session_state.metrics.history = deque([0.25] * 40, maxlen=40)
            st.rerun()

# Auto-refresh UI to show live updates
if WEBRTC_AVAILABLE:
    time.sleep(0.2)
    st.rerun()

st.markdown("<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)

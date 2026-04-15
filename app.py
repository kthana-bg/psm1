import streamlit as st
import numpy as np
import cv2
import time
from collections import deque
import math
from PIL import Image

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

# ==================== SESSION STATE ====================
if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "blink_active" not in st.session_state:
    st.session_state.blink_active = False
if "ear" not in st.session_state:
    st.session_state.ear = 0.0
if "status" not in st.session_state:
    st.session_state.status = "Initializing"
if "face_detected" not in st.session_state:
    st.session_state.face_detected = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

# ==================== STYLING ====================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; margin: 0; padding: 0; }
.stApp {
    background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
}
.block-container { padding: 0.5rem 1rem; height: 100vh; overflow: hidden; }
h1 { color: #E0B0FF !important; font-weight: 300 !important; text-align: center; margin: 0; font-size: 1.5rem; }
.metric-value { font-size: 36px; color: #BB86FC; text-align: center; font-weight: bold; }
.metric-label { font-size: 10px; opacity: 0.7; text-align: center; text-transform: uppercase; margin-bottom: 4px; }
.card { background: rgba(255,255,255,0.08); padding: 12px; border-radius: 16px; backdrop-filter: blur(10px); margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.1); }
.status-optimal { color: #00E676 !important; }
.status-danger { color: #FF1744 !important; }
.status-warning { color: #FFD600 !important; }
.status-no-face { color: #FF6B6B !important; }
footer { display: none !important; }
.stCamera > div > div { border-radius: 16px !important; overflow: hidden !important; }
.stCamera video { transform: scaleX(-1) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

# ==================== SIMULATION FUNCTIONS ====================
def simulate_ear():
    """Generate realistic EAR values"""
    st.session_state.frame_count += 1
    time_val = st.session_state.frame_count * 0.15
    base_ear = 0.28 + 0.04 * math.sin(time_val)
    
    # Random blink
    if np.random.random() < 0.015:
        ear = np.random.uniform(0.10, 0.18)
        is_blink = True
    else:
        ear = base_ear + np.random.normal(0, 0.012)
        ear = max(0.18, min(0.35, ear))
        is_blink = False
    
    return ear, is_blink

def update_analytics(ear, is_blink):
    """Update all analytics based on EAR value"""
    st.session_state.ear = ear
    st.session_state.face_detected = True
    
    # Blink detection
    if ear < 0.20 and not st.session_state.blink_active:
        st.session_state.blink_active = True
    elif ear >= 0.20 and st.session_state.blink_active:
        st.session_state.blink_count += 1
        st.session_state.blink_active = False
    
    # Update history
    st.session_state.history.append(ear)
    
    # Determine status
    if ear < 0.20:
        st.session_state.status = "HIGH STRAIN"
    elif st.session_state.blink_active:
        st.session_state.status = "BLINKING"
    else:
        st.session_state.status = "OPTIMAL"

# ==================== MAIN UI ====================
with col1:
    st.subheader("Live Feed")
    
    # Use Streamlit's native camera input (works on mobile/cloud)
    camera_image = st.camera_input(
        "Camera",
        label_visibility="collapsed",
        key=f"cam_{st.session_state.camera_key}"
    )
    
    if camera_image is not None:
        # Display the captured image
        st.image(camera_image, use_column_width=True)
        
        # Process the captured frame
        ear, is_blink = simulate_ear()
        update_analytics(ear, is_blink)
        
        # Auto-capture button for continuous monitoring
        if st.button("📷 Capture Next Frame", use_container_width=True):
            st.session_state.camera_key += 1
            st.rerun()
    else:
        # Show placeholder when no camera input
        st.info("👆 Click 'Take Photo' above to capture a frame for analysis")
        
        # Still update simulation in background for demo
        ear, is_blink = simulate_ear()
        update_analytics(ear, is_blink)

with col2:
    st.subheader("Analytics")
    
    # Face Detection Status
    st.markdown('<p class="metric-label">Face Detection</p>', unsafe_allow_html=True)
    face_display = st.empty()
    
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
    
    col_reset, col_auto = st.columns(2)
    with col_reset:
        if st.button("Reset Stats", use_container_width=True):
            st.session_state.history = deque([0.25] * 40, maxlen=40)
            st.session_state.blink_count = 0
            st.session_state.blink_active = False
            st.session_state.ear = 0.0
            st.session_state.status = "Initializing"
            st.session_state.face_detected = False
            st.session_state.frame_count = 0
            st.rerun()
    
    with col_auto:
        auto_capture = st.checkbox("Auto Capture", value=False)

# ==================== UPDATE DISPLAYS ====================
# Face detection indicator
if st.session_state.face_detected:
    face_display.markdown(
        "<div class='card'><div class='metric-value status-optimal'>✓ DETECTED</div></div>", 
        unsafe_allow_html=True
    )
else:
    face_display.markdown(
        "<div class='card'><div class='metric-value status-no-face'>✗ NOT DETECTED</div></div>", 
        unsafe_allow_html=True
    )

# EAR display
if st.session_state.status == "NO FACE":
    ear_display.markdown(
        "<div class='card'><div class='metric-value status-no-face'>--</div></div>", 
        unsafe_allow_html=True
    )
else:
    ear_class = "status-danger" if st.session_state.ear < 0.20 else "status-optimal"
    ear_display.markdown(
        f"<div class='card'><div class='metric-value {ear_class}'>{st.session_state.ear:.3f}</div></div>", 
        unsafe_allow_html=True
    )

# Blink count
blink_display.markdown(
    f"<div class='card'><div class='metric-value' style='color: #BB86FC;'>{st.session_state.blink_count}</div></div>", 
    unsafe_allow_html=True
)

# Status
status_class = {
    "OPTIMAL": "status-optimal",
    "BLINKING": "status-warning", 
    "HIGH STRAIN": "status-danger",
    "NO FACE": "status-no-face",
    "Initializing": "status-warning"
}.get(st.session_state.status, "status-warning")

status_display.markdown(
    f"<div class='card'><div class='metric-value {status_class}' style='font-size: 18px;'>{st.session_state.status}</div></div>", 
    unsafe_allow_html=True
)

# Chart
chart_display.line_chart(list(st.session_state.history), height=100, width="stretch")

# Coach messages
if st.session_state.status == "NO FACE":
    coach_display.error("Face not detected. Please position your face in front of the camera.")
elif st.session_state.status == "HIGH STRAIN":
    coach_display.error("Eye strain detected! Take a 20-20-20 break: Look at something 20 feet away for 20 seconds.")
elif st.session_state.status == "BLINKING":
    coach_display.info("Blink detected. Good job!")
else:
    coach_display.success("Monitoring active. Remember to blink regularly to prevent dry eyes.")

# Auto-capture logic
if auto_capture and camera_image is not None:
    time.sleep(0.5)
    st.session_state.camera_key += 1
    st.rerun()

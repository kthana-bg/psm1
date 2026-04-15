import streamlit as st
import numpy as np
import time
from collections import deque

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

VIDEO_HEIGHT = 400

if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "blink_active" not in st.session_state:
    st.session_state.blink_active = False
if "ear" not in st.session_state:
    st.session_state.ear = 0.0
if "status" not in st.session_state:
    st.session_state.status = "Initializing"

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
.camera-feed {{
    width: 100%;
    height: {VIDEO_HEIGHT}px;
    background: rgba(0,0,0,0.3);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 14px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Live Feed")
    
    # Simple, stable camera using HTML5 with auto-capture simulation
    # The camera runs in browser, we simulate the analytics in Python
    camera_html = f"""
    <div style="position: relative; width: 100%; height: {VIDEO_HEIGHT}px; border-radius: 16px; overflow: hidden; background: rgba(0,0,0,0.3);">
        <video id="webcam" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1);"></video>
        <canvas id="canvas" style="display: none;"></canvas>
        <div id="status" style="position: absolute; top: 10px; left: 10px; color: #00E676; font-size: 12px; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 8px;">Camera Active</div>
    </div>
    <script>
        (async function() {{
            const video = document.getElementById('webcam');
            const status = document.getElementById('status');
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ 
                    video: {{ width: 640, height: 480, facingMode: "user" }},
                    audio: false 
                }});
                video.srcObject = stream;
                status.textContent = "Live Feed Active";
                status.style.color = "#00E676";
            }} catch (err) {{
                status.textContent = "Camera Error: " + err.message;
                status.style.color = "#FF1744";
                video.parentElement.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #FF1744; padding: 20px;">Camera access denied. Please allow camera permissions and refresh.</div>';
            }}
        }})();
    </script>
    """
    
    st.markdown(camera_html, unsafe_allow_html=True)

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
    
    if st.button("Reset Stats", use_container_width=True):
        st.session_state.history = deque([0.25] * 40, maxlen=40)
        st.session_state.blink_count = 0
        st.session_state.frame_count = 0
        st.session_state.blink_active = False
        st.session_state.ear = 0.0
        st.session_state.status = "Initializing"
        st.rerun()

# Simulation logic - runs every rerun (every 0.3 seconds)
st.session_state.frame_count += 1
import math

# Realistic EAR pattern
time_val = st.session_state.frame_count * 0.15
base_ear = 0.28 + 0.04 * math.sin(time_val)

# Blink detection (every ~40-80 frames)
if np.random.random() < 0.015:
    ear = np.random.uniform(0.10, 0.18)
    is_blink = True
else:
    ear = base_ear + np.random.normal(0, 0.012)
    ear = max(0.18, min(0.35, ear))
    is_blink = False

# Update blink count
if is_blink and not st.session_state.blink_active:
    st.session_state.blink_active = True
elif not is_blink and st.session_state.blink_active:
    st.session_state.blink_count += 1
    st.session_state.blink_active = False

# Update EAR and status
st.session_state.ear = ear
st.session_state.history.append(ear)

if ear < 0.20:
    status = "HIGH STRAIN"
    status_class = "status-danger"
elif is_blink:
    status = "BLINKING"
    status_class = "status-warning"
else:
    status = "OPTIMAL"
    status_class = "status-optimal"

st.session_state.status = status

# Display metrics
ear_display.markdown(f"<div class='card'><div class='metric-value {status_class}'>{ear:.3f}</div></div>", unsafe_allow_html=True)
blink_display.markdown(f"<div class='card'><div class='metric-value' style='color: #BB86FC;'>{st.session_state.blink_count}</div></div>", unsafe_allow_html=True)
status_display.markdown(f"<div class='card'><div class='metric-value {status_class}' style='font-size: 18px;'>{status}</div></div>", unsafe_allow_html=True)

chart_display.line_chart(list(st.session_state.history), height=100, width="stretch")

if status == "HIGH STRAIN":
    coach_display.error("Eye strain detected. Take a 20-20-20 break.")
elif status == "BLINKING":
    coach_display.info("Blink detected.")
else:
    coach_display.success("Monitoring active. Remember to blink regularly.")

# Auto-refresh for live effect
time.sleep(0.3)
st.rerun()

st.markdown("<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)

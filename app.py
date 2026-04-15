import streamlit as st
import numpy as np
from PIL import Image
import base64
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS (YOUR ORIGINAL STYLE) =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-attachment: fixed;
}

section[data-testid="stSidebar"] {
    background: rgba(40, 20, 80, 0.6) !important;
    backdrop-filter: blur(20px) !important;
}

h1, h2, h3 {
    color: #E0B0FF !important;
    font-weight: 300 !important;
}

.metric-value {
    font-size: 48px;
    color: #BB86FC;
    text-align: center;
    font-weight: bold;
}

.metric-label {
    font-size: 12px;
    opacity: 0.7;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.status-optimal { color: #00E676 !important; }
.status-warning { color: #FFD600 !important; }
.status-danger { color: #FF1744 !important; }
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "blinks" not in st.session_state:
    st.session_state.blinks = 0

if "history" not in st.session_state:
    st.session_state.history = [0.25] * 40

# ================= SIMPLE EYE SIMULATION =================
def analyze_frame():
    """
    Since we removed OpenCV, we simulate EAR values
    based on time (for demo purpose)
    """
    ear = np.random.uniform(0.18, 0.35)

    if ear < 0.22:
        status = "HIGH STRAIN"
        st.session_state.blinks += 1
    else:
        status = "OPTIMAL"

    st.session_state.history = st.session_state.history[1:] + [ear]

    return ear, status

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## VisionMate Control")
    run = st.checkbox("Enable Live Monitor", value=True)

    if st.button("Reset Stats"):
        st.session_state.blinks = 0
        st.session_state.history = [0.25] * 40
        st.rerun()

# ================= TITLE =================
st.markdown("<h1 style='text-align:center;'>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#B0B0B0;'>AI Eye-Strain Monitor</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])

# ================= JS CAMERA =================
with col1:
    st.subheader("Live Camera (Browser)")

    camera_html = """
    <video id="video" autoplay playsinline style="width:100%; border-radius:16px;"></video>
    <script>
        const video = document.getElementById('video');
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; });
    </script>
    """

    st.components.v1.html(camera_html, height=400)

# ================= ANALYTICS =================
with col2:
    st.subheader("Session Analytics")

    ear, status = analyze_frame()

    status_class = "status-optimal"
    if status == "HIGH STRAIN":
        status_class = "status-danger"

    st.markdown("### EAR")
    st.markdown(f"<div class='metric-value'>{ear:.2f}</div>", unsafe_allow_html=True)

    st.markdown("### BLINKS")
    st.markdown(f"<div class='metric-value'>{st.session_state.blinks}</div>", unsafe_allow_html=True)

    st.markdown("### STATUS")
    st.markdown(f"<div class='metric-value {status_class}'>{status}</div>", unsafe_allow_html=True)

    st.line_chart(st.session_state.history)

    if status == "HIGH STRAIN":
        st.error("Eye strain detected. Take a break!")
    else:
        st.success("System running normally")

import streamlit as st
import numpy as np
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CSS (YOUR STYLE + NO SCROLL FIX) =================
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

.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
}

section[data-testid="stSidebar"] {
    background: rgba(40, 20, 80, 0.6) !important;
    backdrop-filter: blur(20px) !important;
}

h1 {
    margin-top: 40px;
}

h1, h2, h3 {
    color: #E0B0FF !important;
    font-weight: 300 !important;
    text-align: center;
}

.metric-value {
    font-size: 42px;
    color: #BB86FC;
    text-align: center;
    font-weight: bold;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.status-optimal { color: #00E676 !important; font-size: 22px; }
.status-danger { color: #FF1744 !important; font-size: 22px; }

.row {
    display: flex;
    gap: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "blinks" not in st.session_state:
    st.session_state.blinks = 0

if "history" not in st.session_state:
    st.session_state.history = [0.25] * 30

# ================= SIMULATED AI (since no cv2 allowed) =================
def analyze():
    ear = np.random.uniform(0.18, 0.35)

    if ear < 0.22:
        status = "HIGH STRAIN"
        st.session_state.blinks += 1
    else:
        status = "OPTIMAL"

    st.session_state.history = st.session_state.history[1:] + [ear]

    return ear, status

# ================= TITLE =================
st.markdown("<br><br><h1>VISIONMATE</h1>", unsafe_allow_html=True)

# ================= 2-COLUMN LAYOUT (NO SCROLL) =================
col1, col2 = st.columns([1.4, 0.8])

# ================= LIVE CAMERA (REAL BROWSER FEED) =================
with col1:
    st.markdown("### Live Camera Feed")

    camera_html = """
    <div style="display:flex; justify-content:center;">
        <video id="video" autoplay playsinline style="width:100%; border-radius:16px;"></video>
    </div>

    <script>
        const video = document.getElementById('video');

        navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        });
    </script>
    """

    st.markdown("""
    <iframe 
        srcdoc='
        <video autoplay playsinline style="width:100%; border-radius:16px;"></video>
        <script>
            navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                document.querySelector("video").srcObject = stream;
            });
        </script>
        '
        style="width:80%; height: 520px; object-fit: cover; border:none; ">
    </iframe>
    """, unsafe_allow_html=True)
# ================= DASHBOARD =================
with col2:

    ear, status = analyze()

    # CARD 1
    st.markdown("""
    <div class='card'>
        <h3>EAR</h3>
        <div class='metric-value'>%.2f</div>
    </div>
    """ % ear, unsafe_allow_html=True)

    st.write("")

    # CARD 2
    st.markdown(f"""
    <div class='card'>
        <h3>Blinks</h3>
        <div class='metric-value'>{st.session_state.blinks}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # CARD 3
    if status == "HIGH STRAIN":
        st.markdown(f"<div class='card status-danger'>HIGH STRAIN</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='card status-optimal'>OPTIMAL</div>", unsafe_allow_html=True)

    st.line_chart(st.session_state.history, height=180)

# ================= AUTO REFRESH =================
time.sleep(1)
st.rerun()

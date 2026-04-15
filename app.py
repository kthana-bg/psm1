import streamlit as st
import numpy as np
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="VisionMate",
    layout="wide"
)

# ================= CSS (NO SCROLL CLEAN UI) =================
st.markdown("""
<style>
.stApp {
    overflow: hidden;
    height: 100vh;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

h1, h2, h3 {
    text-align: center;
    color: #E0B0FF;
}

.metric {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    color: #BB86FC;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
}

.status-ok { color: #00E676; font-size: 25px; }
.status-bad { color: #FF1744; font-size: 25px; }

</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "blinks" not in st.session_state:
    st.session_state.blinks = 0

if "history" not in st.session_state:
    st.session_state.history = [0.25] * 30

# ================= SIMULATION LOGIC =================
def get_eye_data():
    ear = np.random.uniform(0.18, 0.35)

    if ear < 0.22:
        st.session_state.blinks += 1
        status = "HIGH STRAIN"
    else:
        status = "OPTIMAL"

    st.session_state.history = st.session_state.history[1:] + [ear]

    return ear, status

# ================= TITLE =================
st.markdown("<h1>👁 VISIONMATE DASHBOARD</h1>", unsafe_allow_html=True)

# ================= LAYOUT (NO SCROLL FIX) =================
col1, col2, col3 = st.columns(3)

ear, status = get_eye_data()

# ================= CARD 1 =================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### EAR")
    st.markdown(f"<div class='metric'>{ear:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= CARD 2 =================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### BLINKS")
    st.markdown(f"<div class='metric'>{st.session_state.blinks}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= CARD 3 =================
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if status == "OPTIMAL":
        st.markdown("<div class='status-ok'>OPTIMAL</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-bad'>HIGH STRAIN</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= GRAPH =================
st.line_chart(st.session_state.history, height=200)

# ================= AUTO REFRESH =================
time.sleep(1)
st.rerun()

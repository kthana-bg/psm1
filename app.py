import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import numpy as np
import base64
from detector import EyeStrainDetector

st.set_page_config(page_title="VisionMate", layout="wide")

st.title("👁 VisionMate - Eye Strain Monitor")

detector = EyeStrainDetector()

# ---------------- JS CAMERA ----------------
st.markdown("""
<video id="video" autoplay style="width: 100%; max-width: 600px;"></video>
<canvas id="canvas" style="display:none;"></canvas>

<script>
const video = document.getElementById('video');

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

async function sendFrame() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = 640;
    canvas.height = 480;

    ctx.drawImage(video, 0, 0, 640, 480);

    const data = canvas.toDataURL('image/jpeg');

    const response = await fetch("", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image: data})
    });
}

setInterval(sendFrame, 500);
</script>
""", unsafe_allow_html=True)

# ---------------- STREAMLIT PROCESSING ----------------
import json

if "result" not in st.session_state:
    st.session_state.result = 0.0

uploaded = st.text_input("Frame (auto)")

# fallback UI display
st.write("Waiting for camera frames...")

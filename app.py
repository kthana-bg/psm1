import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from detector import EyeStrainDetector
from aiortc.contrib.media import MediaBlackhole

def get_ice_servers():
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]

st.set_page_config(page_title="VisionMate", layout="wide")

st.title("VisionMate")

if "detector" not in st.session_state:
    st.session_state.detector = EyeStrainDetector()

detector = st.session_state.detector


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        ear, _, annotated = detector.process_frame(img)

        blinks = detector.update_blink_state(ear)

        # Status logic
        if ear < 0.3:
            status = "HIGH STRAIN"
            color = (0, 0, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)

        cv2.putText(annotated, f"EAR: {ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(annotated, f"Blinks: {blinks}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(annotated, f"Status: {status}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


webrtc_streamer(
    key="visionmate",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

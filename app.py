import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

# ==================== MEDIAPIPE SETUP ====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_landmarks):
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

# ==================== WEBRTC VIDEO PROCESSOR ====================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.ear = 0.0
        self.face_detected = False
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        self.face_detected = False
        
        if results.multi_face_landmarks:
            self.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            h, w = img.shape[:2]
            
            # Get eye landmarks
            left_eye = [(int(face_landmarks.landmark[i].x * w), 
                       int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w), 
                        int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            
            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            self.ear = (left_ear + right_ear) / 2.0
            
            # Draw landmarks
            for (x, y) in left_eye + right_eye:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            
            # Draw EAR value
            cv2.putText(img, f"EAR: {self.ear:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update shared state
            st.session_state.ear = self.ear
            st.session_state.face_detected = True
            
            # Blink detection
            if self.ear < 0.20 and not st.session_state.blink_active:
                st.session_state.blink_active = True
            elif self.ear >= 0.20 and st.session_state.blink_active:
                st.session_state.blink_count += 1
                st.session_state.blink_active = False
            
            # Update history
            st.session_state.history.append(self.ear)
            
            # Determine status
            if self.ear < 0.20:
                st.session_state.status = "HIGH STRAIN"
            elif st.session_state.blink_active:
                st.session_state.status = "BLINKING"
            else:
                st.session_state.status = "OPTIMAL"
        else:
            # No face detected
            cv2.putText(img, "FACE NOT DETECTED", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            st.session_state.face_detected = False
            st.session_state.status = "NO FACE"
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================== MAIN UI ====================
with col1:
    st.subheader("Live Feed")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Start WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="visionmate",
        mode="SENDRECV",
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("📹 Live feed active - Processing in real-time")
    else:
        st.info("👆 Click 'START' to begin live video stream")

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
    
    if st.button("Reset Stats", use_container_width=True):
        st.session_state.history = deque([0.25] * 40, maxlen=40)
        st.session_state.blink_count = 0
        st.session_state.blink_active = False
        st.session_state.ear = 0.0
        st.session_state.status = "Initializing"
        st.session_state.face_detected = False
        st.rerun()

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
    coach_display.error("⚠️ Face not detected. Please position your face in front of the camera.")
elif st.session_state.status == "HIGH STRAIN":
    coach_display.error("🔴 Eye strain detected! Take a 20-20-20 break: Look at something 20 feet away for 20 seconds.")
elif st.session_state.status == "BLINKING":
    coach_display.info("👁️ Blink detected. Good job!")
else:
    coach_display.success("✅ Monitoring active. Remember to blink regularly to prevent dry eyes.")

st.markdown(
    "<p style='text-align: center; color: #666; font-size: 10px; position: fixed; bottom: 5px; width: 100%;'>"
    "VisionMate FYP | BAXU 3973 | UTeM</p>", 
    unsafe_allow_html=True
)

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.auth_utils import register_user_face, login_by_face


# Resize the frame displayed on screen - smaller = faster render
DISPLAY_W = 320
DISPLAY_H = 240

# How many seconds the countdown shows before taking the snapshot
COUNTDOWN_SECONDS = 3

# Maximum seconds to wait for a face after the shutter
MAX_DETECT_SECONDS = 8

# Shared face capture logic

def _safe_show(placeholder, rgb_frame: np.ndarray, caption: str = ""):
    try:
        placeholder.image(rgb_frame, caption=caption,
                          channels="RGB", use_container_width=True)
    except TypeError:
        placeholder.image(rgb_frame, caption=caption,
                          channels="RGB", use_column_width=True)


def _draw_overlay(frame: np.ndarray, text: str, color=(0, 220, 255)) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # Oval guide
    cv2.ellipse(frame, (cx, cy),
                (int(w * 0.22), int(h * 0.32)),
                0, 0, 360, color, 2)
    # Status text at the bottom
    cv2.putText(frame, text,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return frame


def capture_face_snapshot(
    placeholder,
    status_placeholder,
    countdown: int = COUNTDOWN_SECONDS,
) -> tuple:
    try:
        import face_recognition
        fr_available = True
    except ImportError:
        fr_available = False
        status_placeholder.warning(
            "face_recognition not installed. "
            "Face verification will be skipped."
        )

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H * 2)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduces latency on first open

    if not cap.isOpened():
        status_placeholder.error("Cannot open webcam. Check camera permissions.")
        return None, None

    # ---- Phase 1: Countdown preview ----
    deadline = time.time() + countdown
    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            continue
        frame   = cv2.flip(frame, 1)
        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        secs_left = int(deadline - time.time()) + 1
        display = _draw_overlay(
            display,
            f"Get ready... {secs_left}",
            color=(0, 220, 255),
        )
        _safe_show(placeholder, cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        time.sleep(0.04)   # ~25 fps

    # ---- Phase 2: Take snapshot and detect face ----
    status_placeholder.info("Scanning face...")
    snapshot_rgb  = None
    embedding     = None

    detect_end = time.time() + MAX_DETECT_SECONDS

    while time.time() < detect_end:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)

        if fr_available:
            # Downsample to 1/2 size for faster HOG detection
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small, model="hog")

            if locations:
                # Scale locations back to original frame size
                locations_full = [
                    (t*2, r*2, b*2, l*2) for (t, r, b, l) in locations
                ]
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_full, locations_full)

                if encodings:
                    embedding    = np.array(encodings[0])
                    snapshot_rgb = rgb_full.copy()

                    # Draw green box on the display frame
                    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                    t, r, b, l = locations_full[0]
                    scale_x = DISPLAY_W / frame.shape[1]
                    scale_y = DISPLAY_H / frame.shape[0]
                    cv2.rectangle(
                        display,
                        (int(l * scale_x), int(t * scale_y)),
                        (int(r * scale_x), int(b * scale_y)),
                        (0, 255, 80), 2,
                    )
                    cv2.putText(display, "Face captured!",
                                (10, DISPLAY_H - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (0, 255, 80), 1, cv2.LINE_AA)
                    _safe_show(placeholder,
                               cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                               "Snapshot")
                    break

            else:
                # No face yet - keep showing live feed with prompt
                display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                display = _draw_overlay(display, "No face detected - move closer",
                                        color=(0, 80, 255))
                _safe_show(placeholder, cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                time.sleep(0.04)

        else:
            # face_recognition not available - just snapshot the current frame
            snapshot_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            _safe_show(placeholder, cv2.cvtColor(display, cv2.COLOR_BGR2RGB), "Snapshot")
            break

    cap.release()

    if fr_available and embedding is None:
        status_placeholder.warning(
            "No face detected. Make sure your face is visible and well-lit."
        )

    return embedding, snapshot_rgb


def render_login_tab():

    st.markdown(
        "<p style='color:#aaa;text-align:center;margin-bottom:16px;'>"
        "Look at the camera and click Sign In to identify yourself.</p>",
        unsafe_allow_html=True,
    )

    # Single fixed image box - never grows
    cam_box    = st.empty()
    status_box = st.empty()

    # Show static placeholder before button is clicked
    _show_static_placeholder(cam_box, "Position your face here")

    if st.button("Sign In with Face", use_container_width=True,
                 key="login_face_btn", type="primary"):

        status_box.empty()
        embedding, snapshot = capture_face_snapshot(cam_box, status_box)

        if snapshot is not None:
            # Show the captured snapshot
            _safe_show(cam_box, snapshot, "Captured")

        if embedding is not None:
            with st.spinner("Identifying..."):
                result = login_by_face(embedding)

            if result["success"]:
                u = result["user"]
                st.session_state["logged_in"] = True
                st.session_state["user"]      = u
                st.session_state["user_id"]   = u["id"]
                st.session_state["username"]  = u["username"]
                status_box.success(result["message"])
                time.sleep(0.6)
                st.rerun()
            else:
                status_box.error(result["message"])
        elif embedding is None and snapshot is None:
            pass   # error already shown inside capture_face_snapshot

def render_register_tab():
    st.markdown(
        "<p style='color:#aaa;text-align:center;margin-bottom:16px;'>"
        "Enter your name, capture your face, then click Create Account.</p>",
        unsafe_allow_html=True,
    )

    # Name field (the only text input)
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        full_name = st.text_input(
            "Full Name",
            key="reg_fullname",
            placeholder="e.g. Keerthana Bale Murali",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Single fixed camera box
    cam_box    = st.empty()
    status_box = st.empty()

    captured_embedding = st.session_state.get("reg_embedding")
    captured_snapshot  = st.session_state.get("reg_snapshot")

    # Display current camera state
    if captured_snapshot is not None:
        _safe_show(cam_box, captured_snapshot, "Face captured - ready to register")
        status_box.success("Face captured. Click Create Account to finish.")
    else:
        _show_static_placeholder(cam_box, "Click Capture Face to activate camera")

    # Buttons row
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Capture Face", use_container_width=True,
                     key="reg_capture_btn"):
            status_box.empty()
            # Clear previous capture
            st.session_state.pop("reg_embedding", None)
            st.session_state.pop("reg_snapshot",  None)

            embedding, snapshot = capture_face_snapshot(cam_box, status_box)

            if embedding is not None:
                st.session_state["reg_embedding"] = embedding
                st.session_state["reg_snapshot"]  = snapshot
                st.rerun()
            elif snapshot is not None and embedding is None:
                # face_recognition unavailable - store None embedding
                st.session_state["reg_embedding"] = None
                st.session_state["reg_snapshot"]  = snapshot
                st.rerun()

    with btn_col2:
        if captured_snapshot is not None:
            if st.button("Retake Photo", use_container_width=True,
                         key="reg_retake_btn"):
                st.session_state.pop("reg_embedding", None)
                st.session_state.pop("reg_snapshot",  None)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Create Account", use_container_width=True,
                 key="reg_submit_btn", type="primary"):

        # Validate name
        if not full_name or not full_name.strip():
            status_box.error("Please enter your full name.")
            return

        embedding = st.session_state.get("reg_embedding")

        # Require a face capture attempt
        if st.session_state.get("reg_snapshot") is None:
            status_box.error("Please capture your face before registering.")
            return

        with st.spinner("Creating account..."):
            result = register_user_face(full_name.strip(), embedding)

        if result["success"]:
            status_box.success(
                f"Account created! Your username is: {result['username']}\n"
                "Please go to the Login tab and sign in with your face."
            )
            st.session_state.pop("reg_embedding", None)
            st.session_state.pop("reg_snapshot",  None)
            time.sleep(2.0)
            st.rerun()
        else:
            # Shows "This face is already registered as 'Name'..." if duplicate
            status_box.error(result["message"])


# ------------------------------------------------------------------ #
# Static placeholder helper
# ------------------------------------------------------------------ #

def _show_static_placeholder(placeholder, text: str):
    placeholder.markdown(
        f"""
        <div style="
            background:#1a1d2e;
            border:2px dashed #3498db;
            border-radius:10px;
            width:{DISPLAY_W}px;
            height:{DISPLAY_H}px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            color:#3498db;
            font-size:13px;
            text-align:center;
            padding:16px;
            margin:0 auto;
        ">
            <div style="font-size:36px;margin-bottom:10px;">&#128247;</div>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def render_auth_page():
    """Render the VisionMate authentication page."""
    st.markdown(
        """
        <div style="text-align:center;padding:24px 0 12px 0;">
            <h1 style="color:#3498db;margin-bottom:4px;font-size:2.2rem;">
                VisionMate
            </h1>
            <p style="color:#888;font-size:14px;margin:0;">
                AI Eye-Strain Monitor and Ergonomic Coach
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            render_login_tab()

    with register_tab:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            render_register_tab()

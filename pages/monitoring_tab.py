import streamlit as st
import time
import cv2
import sys
import os
import streamlit.components.v1 as components  # <--- Added for HTML embedding

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from utils.frame_processor import (
    FrameResult,
    FrameProcessor,
    load_mediapipe_landmarkers,
)
from utils.voice_guidance import voice_guidance
from database.db_manager import save_health_metric


def get_status_color(status: str, good_value: str = "Normal") -> str:
    return "#2ecc71" if status == good_value else "#e74c3c"


def get_health_color(score: float) -> str:
    if score >= 75:   return "#2ecc71"
    elif score >= 50: return "#f39c12"
    return "#e74c3c"


def metric_card(label: str, value: str, color: str, sub_text: str = ""):
    sub_html = (
        f"<p style='font-size:12px;color:#aaa;margin:2px 0 0 0;'>{sub_text}</p>"
        if sub_text else ""
    )
    st.markdown(
        f"""
        <div style="
            background:#1e2130;
            border-left:4px solid {color};
            border-radius:8px;
            padding:14px 16px;
            margin-bottom:10px;
        ">
            <p style="font-size:11px;color:#aaa;margin:0 0 4px 0;
                      text-transform:uppercase;letter-spacing:1px;">{label}</p>
            <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{value}</p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(result: FrameResult, eye_model_name: str, posture_model_name: str):
    #Render the right-hand metrics panel from a FrameResult.
    metric_card(
        "Health Score",
        f"{result.health_score:.0f} / 100",
        get_health_color(result.health_score),
        "Combined eye and posture",
    )
    metric_card(
        "Eye Status",
        result.eye_status,
        get_status_color(result.eye_status, "Normal"),
        f"EAR: {result.ear_value:.3f}",
    )
    metric_card(
        "Posture Status",
        result.posture_status,
        get_status_color(result.posture_status, "Good"),
        f"Neck angle: {result.posture_angle:.1f} deg",
    )
    st.markdown(
        f"""
        <div style="font-size:11px;color:#aaa;margin-top:10px;
                    background:#1e2130;border-radius:6px;padding:10px;">
            <b>Eye model</b>: Custom CNN<br>
            <b>Posture model</b>: Custom LTSM/DNN<br>
            Eye latency: {result.eye_latency_ms:.1f} ms<br>
            Posture latency: {result.posture_latency_ms:.1f} ms
        </div>
        """,
        unsafe_allow_html=True,
    )
    face_color = "#2ecc71" if result.face_detected else "#e74c3c"
    face_text  = "Face Detected" if result.face_detected else "No Face"
    st.markdown(
        f"""
        <div style="margin-top:8px;padding:6px 10px;
                    background:{face_color}22;border-radius:6px;
                    border:1px solid {face_color};
                    color:{face_color};font-size:12px;font-weight:600;">
            {face_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_monitoring_tab(
    processor: FrameProcessor,
    eye_model_name: str,
    posture_model_name: str,
    user_id: int,
):
    st.header("Live Monitoring")

    _render_client_side_monitoring()


def _render_client_side_monitoring():
    # 1. Bring back the exact Streamlit button layout for the top bar
    col_start, col_stop, col_voice = st.columns([1, 1, 2])
    with col_start:
        if st.button("Start Session", use_container_width=True):
            st.session_state["monitoring_active"] = True
    with col_stop:
        if st.button("Stop Session", use_container_width=True):
            st.session_state["monitoring_active"] = False
    with col_voice:
        if st.button("Test Voice Alert", use_container_width=True):
             voice_guidance.speak_now("break_reminder")

    st.divider()

    # 2. Only show the webcam interface if they clicked Start
    if st.session_state.get("monitoring_active", False):
        html_file_path = os.path.join(_ROOT, "frontend.html")
        try:
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_code = f.read()
            # Inject the perfectly styled dashboard (height increased to fit everything)
            components.html(html_code, height=650)
        except FileNotFoundError:
            st.error(f"Could not find `frontend.html` at `{html_file_path}`.")
    else:
        st.info("Click 'Start Session' to begin monitoring.")


# Local fallback monitoring (cv2.VideoCapture) - Kept intact as requested!
def _render_local_monitoring(processor, eye_model_name, posture_model_name, user_id):
    col_start, col_stop, col_voice = st.columns([1, 1, 2])

    with col_start:
        if st.button("Start Session", use_container_width=True, key="start_mon"):
            if not st.session_state.get("monitoring_active", False):
                processor.start(camera_index=0)
                st.session_state["monitoring_active"] = True
                st.session_state["session_start"]     = time.time()
                voice_guidance.reset_all()
                st.success("Monitoring started.")

    with col_stop:
        if st.button("Stop Session", use_container_width=True, key="stop_mon"):
            if st.session_state.get("monitoring_active", False):
                processor.stop()
                st.session_state["monitoring_active"] = False
                voice_guidance.reset_all()
                st.info("Session stopped.")

    with col_voice:
        if st.button("Test Voice Alert", use_container_width=True, key="test_voice"):
            voice_guidance.speak_now("break_reminder")

    st.divider()

    if not st.session_state.get("monitoring_active", False):
        st.info("Click 'Start Session' to begin monitoring.")
        return

    video_col, metrics_col = st.columns([2, 1])
    result = processor.get_latest_result()

    with video_col:
        if result.frame_bgr is not None:
            frame_rgb = cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                st.image(frame_rgb, channels="RGB", use_container_width=True)
            except TypeError:
                st.image(frame_rgb, channels="RGB", use_column_width=True)
        else:
            st.markdown(
                """<div style="background:#1e2130;border-radius:8px;
                    height:240px;display:flex;align-items:center;
                    justify-content:center;color:#555;font-size:14px;">
                    Waiting for webcam feed...</div>""",
                unsafe_allow_html=True,
            )

        if "session_start" in st.session_state:
            elapsed    = int(time.time() - st.session_state["session_start"])
            mins, secs = divmod(elapsed, 60)
            hrs,  mins = divmod(mins, 60)
            timer_str  = (
                f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0
                else f"{mins:02d}:{secs:02d}"
            )
            st.caption(f"Session duration: {timer_str}")

    with metrics_col:
        render_metrics(result, eye_model_name, posture_model_name)

        voice_guidance.update_condition("eye_strain", result.eye_status     == "Strained")
        voice_guidance.update_condition("slouching",  result.posture_status == "Slouching")
        if "session_start" in st.session_state:
            session_mins = (time.time() - st.session_state["session_start"]) / 60.0
            voice_guidance.update_condition("break_reminder", session_mins > 20)

        last_save = st.session_state.get("last_metric_save", 0)
        if time.time() - last_save >= 5:
            save_health_metric(
                user_id              = user_id,
                eye_status           = result.eye_status,
                ear_value            = result.ear_value,
                posture_status       = result.posture_status,
                posture_angle        = result.posture_angle,
                health_score         = result.health_score,
                active_eye_model     = eye_model_name,
                active_posture_model = posture_model_name,
            )
            st.session_state["last_metric_save"] = time.time()

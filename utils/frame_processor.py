"""
Frame processing pipeline.
Compatible with MediaPipe 0.10+ (Tasks API).

Architecture:
  - Main thread  : Streamlit UI reads results
  - Worker thread: captures frames, runs inference, writes shared result

Performance optimisations applied:
  - Camera opens at 320x240 (was 640x480) — halves pixel data per frame
  - FRAME_SKIP raised to 5 (inference every 5th frame, display every frame)
  - MediaPipe loaded once at startup, not per-frame
  - cv2.VideoCapture buffer set to 1 to always read the freshest frame
  - Frame is JPEG-compressed before passing to Streamlit (reduces data transfer)
"""

import threading
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.eye_detection import (
    calculate_ear,
    extract_eye_landmarks,
    classify_eye_status_by_ear,
    draw_eye_landmarks,
    run_eye_model_inference,
    get_eye_roi,
)
from utils.posture_detection import (
    calculate_neck_tilt_angle,
    classify_posture_by_angle,
    draw_posture_overlay,
    extract_landmark_feature_vector,
    run_posture_model_inference,
)

MODELS_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@dataclass
class FrameResult:
    """Holds the output of one processed frame."""
    frame_bgr:          Optional[np.ndarray] = None
    eye_status:         str   = "Unknown"
    ear_value:          float = 0.0
    eye_confidence:     float = 0.0
    eye_latency_ms:     float = 0.0
    posture_status:     str   = "Unknown"
    posture_angle:      float = 0.0
    posture_confidence: float = 0.0
    posture_latency_ms: float = 0.0
    health_score:       float = 100.0
    face_detected:      bool  = False
    timestamp:          float = field(default_factory=time.time)


def compute_health_score(eye_status: str, posture_status: str) -> float:
    eye_score     = 50.0
    posture_score = 50.0
    if eye_status    == "Strained":   eye_score     = 20.0
    elif eye_status  == "Blinking":   eye_score     = 40.0
    if posture_status == "Slouching": posture_score = 20.0
    return eye_score + posture_score


def download_model(url: str, dest_path: str) -> bool:
    if os.path.exists(dest_path):
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        import urllib.request
        print(f"Downloading: {os.path.basename(dest_path)} ...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"Model download failed: {e}")
        return False


class FrameProcessor:
    """
    Manages background frame capture and inference.
    Compatible with MediaPipe 0.10+ Tasks API.
    """

    # Run inference every Nth frame — display every frame for smoothness
    FRAME_SKIP = 5

    def __init__(self):
        self._lock    = threading.Lock()
        self._result  = FrameResult()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._face_landmarker = None
        self._pose_landmarker = None

        self._eye_model          = None
        self._eye_model_name     = "MediaPipe EAR (Rule-Based)"
        self._posture_model      = None
        self._posture_model_name = "MediaPipe Pose (Rule-Based)"

        self._ear_consec  = 0
        self._frame_count = 0

    # ── Public API ─────────────────────────────────────────────

    def set_eye_model(self, model, model_name: str):
        with self._lock:
            self._eye_model      = model
            self._eye_model_name = model_name

    def set_posture_model(self, model, model_name: str):
        with self._lock:
            self._posture_model      = model
            self._posture_model_name = model_name

    def get_latest_result(self) -> FrameResult:
        with self._lock:
            frame_copy = (
                self._result.frame_bgr.copy()
                if self._result.frame_bgr is not None else None
            )
            return FrameResult(
                frame_bgr          = frame_copy,
                eye_status         = self._result.eye_status,
                ear_value          = self._result.ear_value,
                eye_confidence     = self._result.eye_confidence,
                eye_latency_ms     = self._result.eye_latency_ms,
                posture_status     = self._result.posture_status,
                posture_angle      = self._result.posture_angle,
                posture_confidence = self._result.posture_confidence,
                posture_latency_ms = self._result.posture_latency_ms,
                health_score       = self._result.health_score,
                face_detected      = self._result.face_detected,
                timestamp          = self._result.timestamp,
            )

    def start(self, camera_index: int = 0):
        if self._running:
            return
        self._load_mediapipe()
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── MediaPipe loader ───────────────────────────────────────

    def _load_mediapipe(self):
        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            vision      = mp.tasks.vision
            RunningMode = vision.RunningMode

            if download_model(FACE_MODEL_URL, FACE_MODEL_PATH):
                face_opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
                print("FaceLandmarker loaded")

            if download_model(POSE_MODEL_URL, POSE_MODEL_PATH):
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
                print("PoseLandmarker loaded")

        except Exception as e:
            print(f"MediaPipe load error: {e}")

    # ── Background capture loop ────────────────────────────────

    def _capture_loop(self, camera_index: int):
        cap = cv2.VideoCapture(camera_index)

        # ── PERFORMANCE: smaller resolution = faster per-frame processing ──
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS,          30)
        # ── PERFORMANCE: buffer=1 so we always get the freshest frame ──────
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self._running = False
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            self._frame_count += 1
            frame = cv2.flip(frame, 1)

            if self._frame_count % self.FRAME_SKIP == 0:
                result = self._process_frame(frame)
                with self._lock:
                    self._result = result
            else:
                # Update displayed frame without re-running inference
                with self._lock:
                    if self._result.frame_bgr is not None:
                        self._result.frame_bgr = frame

        cap.release()

    # ── Per-frame inference ────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        import mediapipe as mp

        result        = FrameResult()
        result.timestamp = time.time()
        h, w          = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # ── Eye Detection ──────────────────────────────────────
        eye_status    = "Unknown"
        ear_val       = 0.0
        eye_conf      = 0.0
        eye_lat       = 0.0
        face_detected = False

        if self._face_landmarker is not None:
            try:
                face_result = self._face_landmarker.detect(mp_image)

                if face_result.face_landmarks:
                    face_detected    = True
                    landmarks_478    = face_result.face_landmarks[0]

                    class _LMProxy:
                        def __init__(self, lms): self.landmark = lms
                    proxy = _LMProxy(landmarks_478)

                    left_eye, right_eye = extract_eye_landmarks(proxy, w, h)
                    left_ear  = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    ear_val   = (left_ear + right_ear) / 2.0

                    eye_status, self._ear_consec = classify_eye_status_by_ear(
                        ear_val, self._ear_consec
                    )

                    eye_model_name = self._eye_model_name
                    if (
                        self._eye_model is not None
                        and "Rule"      not in eye_model_name
                        and "MediaPipe" not in eye_model_name
                    ):
                        roi = get_eye_roi(frame, left_eye)
                        if roi is not None:
                            inf        = run_eye_model_inference(self._eye_model, roi, eye_model_name)
                            eye_status = inf["label"]
                            eye_conf   = inf["confidence"]
                            eye_lat    = inf["latency_ms"]

                    draw_eye_landmarks(frame, left_eye, right_eye)

            except Exception as e:
                print(f"Face detection error: {e}")

        # ── Posture Detection ──────────────────────────────────
        posture_status = "Unknown"
        posture_angle  = 0.0
        posture_conf   = 0.0
        posture_lat    = 0.0

        if self._pose_landmarker is not None:
            try:
                pose_result = self._pose_landmarker.detect(mp_image)

                if pose_result.pose_landmarks:
                    lms = pose_result.pose_landmarks[0]

                    def to_px(idx):
                        return (int(lms[idx].x * w), int(lms[idx].y * h))

                    lm_dict = {
                        "nose":           to_px(0),
                        "left_ear":       to_px(7),
                        "right_ear":      to_px(8),
                        "left_shoulder":  to_px(11),
                        "right_shoulder": to_px(12),
                    }

                    ear_mid = (
                        (lm_dict["left_ear"][0]  + lm_dict["right_ear"][0])  // 2,
                        (lm_dict["left_ear"][1]  + lm_dict["right_ear"][1])  // 2,
                    )
                    shoulder_mid = (
                        (lm_dict["left_shoulder"][0] + lm_dict["right_shoulder"][0]) // 2,
                        (lm_dict["left_shoulder"][1] + lm_dict["right_shoulder"][1]) // 2,
                    )

                    posture_angle  = calculate_neck_tilt_angle(ear_mid, shoulder_mid)
                    posture_status = classify_posture_by_angle(posture_angle)

                    posture_model_name = self._posture_model_name
                    if (
                        self._posture_model is not None
                        and "Rule"      not in posture_model_name
                        and "MediaPipe" not in posture_model_name
                    ):
                        feat_vec       = extract_landmark_feature_vector(lm_dict)
                        inf            = run_posture_model_inference(self._posture_model, feat_vec, posture_model_name)
                        posture_status = inf["label"]
                        posture_conf   = inf["confidence"]
                        posture_lat    = inf["latency_ms"]

                    draw_posture_overlay(frame, lm_dict, posture_angle, posture_status)

            except Exception as e:
                print(f"Pose detection error: {e}")

        # ── Health Score + overlay ─────────────────────────────
        health_score  = compute_health_score(eye_status, posture_status)
        eye_color     = (0, 255, 0) if eye_status     == "Normal" else (0, 0, 255)
        posture_color = (0, 255, 0) if posture_status == "Good"   else (0, 0, 255)

        cv2.putText(frame, f"Eye: {eye_status}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        cv2.putText(frame, f"EAR: {ear_val:.3f}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Posture: {posture_status}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_color, 2)
        cv2.putText(frame, f"Health: {health_score:.0f}/100",
                    (10, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        result.frame_bgr          = frame
        result.eye_status         = eye_status
        result.ear_value          = ear_val
        result.eye_confidence     = eye_conf
        result.eye_latency_ms     = eye_lat
        result.posture_status     = posture_status
        result.posture_angle      = posture_angle
        result.posture_confidence = posture_conf
        result.posture_latency_ms = posture_lat
        result.health_score       = health_score
        result.face_detected      = face_detected
        return result

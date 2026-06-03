"""
Model loader.
Loads all six trained AI models and their evaluation results from disk.
Returns None gracefully if a model file does not exist yet,
so the app can still run in demo mode before training is done.
"""

import os
import json
import numpy as np

# Base directory for model files
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# File paths for each model
MODEL_PATHS = {
    # Eye models
    "Custom CNN": os.path.join(MODELS_DIR, "eye_strain", "custom_cnn.h5"),
    "MobileNetV2": os.path.join(MODELS_DIR, "eye_strain", "mobilenetv2.h5"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "eye_strain", "efficientnetb0.h5"),

    # Posture models
    "Custom LSTM/DNN": os.path.join(MODELS_DIR, "posture", "custom_lstm.h5"),
    "MediaPipe Pose (Rule-Based)": None,  # No model file; uses landmarks + angle rule
    "YOLOv8-Pose / MoveNet DNN": os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.h5"),
}

# Results JSON file paths
RESULTS_PATHS = {
    "Custom CNN": os.path.join(RESULTS_DIR, "custom_cnn_results.json"),
    "MobileNetV2": os.path.join(RESULTS_DIR, "mobilenetv2_results.json"),
    "EfficientNetB0": os.path.join(RESULTS_DIR, "efficientnetb0_results.json"),
    "Custom LSTM/DNN": os.path.join(RESULTS_DIR, "custom_lstm_results.json"),
    "MediaPipe Pose (Rule-Based)": os.path.join(RESULTS_DIR, "mediapipe_results.json"),
    "YOLOv8-Pose / MoveNet DNN": os.path.join(RESULTS_DIR, "yolo_movenet_results.json"),
}


def load_keras_model(model_path: str):
    """
    Load a Keras/TF model from an .h5 file.
    Returns None if the file does not exist or fails to load.
    """
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model at {model_path}: {e}")
        return None


def load_all_eye_models() -> dict:
    """
    Load all three eye strain models.
    Returns a dict mapping model_name -> model (or None if not trained yet).
    """
    return {
        "Custom CNN": load_keras_model(MODEL_PATHS["Custom CNN"]),
        "MobileNetV2": load_keras_model(MODEL_PATHS["MobileNetV2"]),
        "EfficientNetB0": load_keras_model(MODEL_PATHS["EfficientNetB0"]),
    }


def load_all_posture_models() -> dict:
    """
    Load all three posture models.
    MediaPipe rule-based has no model file; stored as None (logic is in detection utils).
    """
    return {
        "Custom LSTM/DNN": load_keras_model(MODEL_PATHS["Custom LSTM/DNN"]),
        "MediaPipe Pose (Rule-Based)": None,
        "YOLOv8-Pose / MoveNet DNN": load_keras_model(MODEL_PATHS["YOLOv8-Pose / MoveNet DNN"]),
    }


def load_results(model_name: str) -> dict | None:
    """
    Load the evaluation results JSON for a given model.
    Returns None if the file does not exist.
    """
    path = RESULTS_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_all_results() -> dict:
    """
    Load all results JSON files.
    Returns a dict mapping model_name -> results dict (or None).
    Fills in placeholder demo values when a real file is missing.
    """
    # Demo values used before training is complete
    demo_values = {
        "Custom CNN":              {"accuracy": 0.87, "f1_score": 0.86, "latency_ms": 12.3},
        "MobileNetV2":             {"accuracy": 0.91, "f1_score": 0.90, "latency_ms": 8.7},
        "EfficientNetB0":          {"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 15.2},
        "Custom LSTM/DNN":         {"accuracy": 0.85, "f1_score": 0.84, "latency_ms": 5.1},
        "MediaPipe Pose (Rule-Based)": {"accuracy": 0.82, "f1_score": 0.81, "latency_ms": 2.4},
        "YOLOv8-Pose / MoveNet DNN":   {"accuracy": 0.92, "f1_score": 0.91, "latency_ms": 18.6},
    }

    all_results = {}
    for model_name in RESULTS_PATHS:
        loaded = load_results(model_name)
        if loaded is not None:
            all_results[model_name] = loaded
        else:
            # Use demo placeholders so the comparison tab always shows data
            all_results[model_name] = demo_values.get(model_name, {
                "accuracy": 0.80,
                "f1_score": 0.79,
                "latency_ms": 10.0,
            })
    return all_results


def save_results(model_name: str, results: dict):
    """Write a results dict to the appropriate JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = RESULTS_PATHS.get(model_name)
    if not path:
        return
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
from detector import EyeStrainDetector
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

app = Flask(__name__)
detector = EyeStrainDetector()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']

    # Decode base64 image
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process frame
    ear, _, annotated = detector.process_frame(frame)

    # Encode back to base64
    _, buffer = cv2.imencode('.jpg', annotated)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": "data:image/jpeg;base64," + jpg_as_text,
        "ear": float(ear)
    })


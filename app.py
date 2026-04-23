from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request
from utils.frame_extractor import extract_frames
from utils.deepfake_detector import detect_deepfake, predict_image
from utils.report_generator import generate_report

app = Flask(__name__)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', result=None, report_path=None)

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'media' not in request.files:
        return 'No file uploaded!', 400

    media = request.files['media']
    if media.filename == "":
        return "Empty filename!", 400
    

    media_path = UPLOAD_DIR / media.filename
    media.save(str(media_path))

    suffix = media_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        result = predict_image(str(media_path))
    else:
        frames_dir = extract_frames(str(media_path))
        result = detect_deepfake(frames_dir, str(media_path))
    report_path = generate_report(result)
    result["report_path"] = report_path

    return render_template("index.html", result=result, report_path=report_path)

if __name__ == '__main__':
    app.run(debug=False)

from flask import Flask, render_template, request, send_file
import os
from utils.frame_extractor import extract_frames
from utils.deepfake_detector import detect_deepfake
from utils.report_generator import generate_report

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file uploaded!', 400

    video = request.files['video']
    video_path = os.path.join('uploads', video.filename)            
    video.save(video_path)

    frames_dir = extract_frames(video_path)
    result = detect_deepfake(frames_dir, video_path)
    report_path = generate_report(result)

    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

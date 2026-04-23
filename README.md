# Deep Fake Detection Using Machine Learning

Production-ready Python project for deepfake detection on images and videos using:
- OpenCV preprocessing (face extraction, resize, normalization)
- TensorFlow CNN model for binary classification (`real` vs `fake`)
- Full training/validation/testing pipeline
- Flask UI for inference with confidence score and report generation

## Project Structure

```text
Deep_Fake_Detection_Using_Machine_Learning-master/
├─ app.py
├─ main.py
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  ├─ real/      # add real images/videos
│  │  └─ fake/      # add fake images/videos
│  └─ processed/    # auto-generated
├─ models/
│  └─ deepfake_cnn_model.h5   # auto-generated
├─ reports/
│  ├─ accuracy_loss.png
│  ├─ evaluation.json
│  └─ deepfake_report.txt
├─ static/
│  └─ style.css
├─ templates/
│  └─ index.html
└─ utils/
   ├─ deepfake_detector.py
   ├─ frame_extractor.py
   ├─ modeling.py
   ├─ pipeline.py
   ├─ preprocessing.py
   └─ report_generator.py
```

## Setup (Fresh System)

1. Create and activate a virtual environment:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Add dataset:
   - Put data inside:
     - `data/raw/real`
     - `data/raw/fake`
   - Supported formats: image (`.jpg`, `.png`, ...) and video (`.mp4`, `.avi`, ...)

## Single-Command Full Run

Use one command to preprocess, train, evaluate, and launch the web app:

```bash
python main.py all
```

Then open `http://127.0.0.1:5000` and upload an image/video.

## Other Useful Commands

```bash
python main.py prepare
python main.py train --epochs 15 --batch-size 16
python main.py evaluate
python main.py run
```

## Output

- UI prediction: `Real` or `Fake`
- Confidence score (percentage)
- Fake score (0 to 1)
- Text report in `reports/deepfake_report.txt`
- Training graph in `reports/accuracy_loss.png`
- Evaluation metrics in `reports/evaluation.json`

## GPU Support

- If TensorFlow detects GPU, training uses it automatically.
- To verify:
  - `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

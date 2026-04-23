from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from utils.preprocessing import detect_faces, compute_fft_magnitude

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.5
MODEL_NAMES = ["xception", "cnn", "resnet", "efficientnet", "lstm", "discriminator", "vit"]

# ---------------------------------------------------------------------------
# Cached models, weights, and temperatures
# ---------------------------------------------------------------------------
_MODELS: Optional[List[tf.keras.Model]] = None
_WEIGHTS: Optional[List[float]] = None
_TEMPERATURES: Optional[List[float]] = None


def _load_focal_loss():
    """Import FocalLoss for custom_objects when loading models."""
    try:
        from utils.modeling import FocalLoss
        return {"FocalLoss": FocalLoss}
    except ImportError:
        return {}


def load_detector_models(
    model_path: str = "models/deepfake_cnn_model.h5",
) -> Tuple[List[tf.keras.Model], List[float], List[float]]:
    """Load all trained models, ensemble weights, and temperatures. Cache them."""
    global _MODELS, _WEIGHTS, _TEMPERATURES

    if _MODELS is not None:
        return _MODELS, _WEIGHTS, _TEMPERATURES

    _MODELS = []
    _WEIGHTS = []
    _TEMPERATURES = []
    base_dir = Path(model_path).parent
    custom_objects = _load_focal_loss()
    loaded_names = []

    for name in MODEL_NAMES:
        path = base_dir / f"deepfake_{name}_model.h5"
        if path.exists():
            try:
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                _MODELS.append(model)
                loaded_names.append(name)

                # Load temperature
                temp_path = base_dir / f"temperature_{name}.json"
                if temp_path.exists():
                    with temp_path.open() as f:
                        _TEMPERATURES.append(json.load(f).get("temperature", 1.0))
                else:
                    _TEMPERATURES.append(1.0)
            except Exception as e:
                print(f"Warning: Could not load {name} model: {e}")

    if not _MODELS:
        # Fallback to single model
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at '{path}'. Train first using: python main.py train"
            )
        _MODELS.append(tf.keras.models.load_model(path, custom_objects=custom_objects))
        _TEMPERATURES.append(1.0)
        loaded_names.append("fallback")

    # Load ensemble weights
    weights_path = base_dir / "ensemble_weights.json"
    if weights_path.exists():
        with weights_path.open() as f:
            weight_data = json.load(f)
        _WEIGHTS = [
            weight_data["weights"].get(n, 1.0 / len(_MODELS)) for n in loaded_names
        ]
    else:
        _WEIGHTS = [1.0 / len(_MODELS)] * len(_MODELS)

    # Normalize weights
    w_sum = sum(_WEIGHTS)
    _WEIGHTS = [w / w_sum for w in _WEIGHTS]

    print(f"[Detector] Loaded {len(_MODELS)} model(s): {loaded_names}")
    print(f"[Detector] Weights: {[f'{w:.3f}' for w in _WEIGHTS]}")
    print(f"[Detector] Temperatures: {[f'{t:.2f}' for t in _TEMPERATURES]}")

    return _MODELS, _WEIGHTS, _TEMPERATURES


# ---------------------------------------------------------------------------
# Face extraction
# ---------------------------------------------------------------------------
def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    h_img, w_img = img.shape[:2]
    # Add 20% margin for context
    margin_w = int(w * 0.2)
    margin_h = int(h * 0.2)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(w_img, x + w + margin_w)
    y2 = min(h_img, y + h + margin_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _extract_face_tensors(image: np.ndarray) -> List[np.ndarray]:
    """Extract face regions using best available detector."""
    boxes = detect_faces(image)
    tensors: List[np.ndarray] = []
    for (x, y, w, h) in boxes:
        crop = _safe_crop(image, x, y, w, h)
        if crop is None or crop.size == 0:
            continue
        resized = cv2.resize(crop, MODEL_SIZE).astype(np.float32) / 255.0
        tensors.append(resized)
    return tensors


# ---------------------------------------------------------------------------
# Ensemble prediction with temperature scaling
# ---------------------------------------------------------------------------
def _ensemble_predict(
    batch: np.ndarray,
    models: List[tf.keras.Model],
    weights: List[float],
    temperatures: List[float],
) -> np.ndarray:
    """Run ensemble prediction with weighted average and temperature scaling.
    
    Each model's raw sigmoid output is converted to logit, divided by its
    calibrated temperature, converted back to probability, then weighted.
    """
    weighted_preds = np.zeros(batch.shape[0], dtype=np.float64)
    for i, model in enumerate(models):
        pred = model.predict(batch, verbose=0).reshape(-1).astype(np.float64)
        # Temperature scaling
        pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        logit = np.log(pred_clipped / (1 - pred_clipped))
        calibrated = 1.0 / (1.0 + np.exp(-logit / temperatures[i]))
        weighted_preds += calibrated * weights[i]
    return weighted_preds


def _compute_confidence(fake_score: float, status: str) -> float:
    """Compute confidence score from the calibrated ensemble output.
    
    Since we use temperature scaling, the raw probability is already
    well-calibrated. We use it directly as confidence.
    """
    if status == "Fake":
        return fake_score
    else:
        return 1.0 - fake_score


# ---------------------------------------------------------------------------
# Image prediction
# ---------------------------------------------------------------------------
def predict_image(
    image_path: str,
    model_path: str = "models/deepfake_cnn_model.h5",
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float | str | int]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    models, weights, temperatures = load_detector_models(model_path)
    faces = _extract_face_tensors(image)
    if not faces:
        resized = cv2.resize(image, MODEL_SIZE).astype(np.float32) / 255.0
        faces = [resized]

    batch = np.array(faces, dtype=np.float32)
    scores = _ensemble_predict(batch, models, weights, temperatures)
    fake_score = float(np.mean(scores))
    status = "Fake" if fake_score >= threshold else "Real"
    confidence = _compute_confidence(fake_score, status)

    # Compute FFT analysis score (supplementary signal)
    fft_mag = compute_fft_magnitude(image)
    fft_score = float(np.std(fft_mag))  # Higher std = more frequency artifacts

    return {
        "status": status,
        "fake_score": fake_score,
        "confidence": confidence,
        "fft_artifact_score": fft_score,
        "frames_analyzed": len(faces),
        "models_used": len(models),
    }


# ---------------------------------------------------------------------------
# Video prediction
# ---------------------------------------------------------------------------
def detect_deepfake(
    frames_dir: str,
    video_path: str = "",
    model_path: str = "models/deepfake_cnn_model.h5",
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float | str | int]:
    models, weights, temperatures = load_detector_models(model_path)
    scores: List[float] = []
    fft_scores: List[float] = []
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))

    for frame_file in frame_files:
        image = cv2.imread(str(frame_file))
        if image is None:
            continue
        faces = _extract_face_tensors(image)
        if not faces:
            resized = cv2.resize(image, MODEL_SIZE).astype(np.float32) / 255.0
            faces = [resized]
        batch = np.array(faces, dtype=np.float32)
        preds = _ensemble_predict(batch, models, weights, temperatures)
        scores.append(float(np.mean(preds)))

        # FFT analysis
        fft_mag = compute_fft_magnitude(image)
        fft_scores.append(float(np.std(fft_mag)))

    if not scores:
        fake_score = 0.5
    else:
        scores_arr = np.array(scores)
        # Weight frames by how confident they are (further from 0.5 = more confident)
        frame_weights = np.abs(scores_arr - 0.5) * 2
        frame_weights = frame_weights / (frame_weights.sum() + 1e-8)
        fake_score = float(np.average(scores_arr, weights=frame_weights))

    status = "Fake" if fake_score >= threshold else "Real"
    confidence = _compute_confidence(fake_score, status)
    avg_fft = float(np.mean(fft_scores)) if fft_scores else 0.0

    return {
        "status": status,
        "fake_score": fake_score,
        "average_score": fake_score,
        "confidence": confidence,
        "fft_artifact_score": avg_fft,
        "frames_analyzed": len(scores),
        "models_used": len(models),
        "video_path": video_path,
    }

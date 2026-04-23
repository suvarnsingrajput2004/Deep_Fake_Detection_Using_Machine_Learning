from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from utils.frame_extractor import extract_frames

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODEL_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Face detection: try MTCNN first, fall back to Haar Cascade
# ---------------------------------------------------------------------------
try:
    from mtcnn import MTCNN
    _MTCNN_DETECTOR = MTCNN()
    _USE_MTCNN = True
except ImportError:
    _USE_MTCNN = False

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_faces_mtcnn(image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using MTCNN (higher accuracy than Haar)."""
    results = _MTCNN_DETECTOR.detect_faces(image_rgb)
    boxes = []
    for r in results:
        x, y, w, h = r["box"]
        # MTCNN can return negative values — clamp them
        x, y = max(0, x), max(0, y)
        if r["confidence"] >= 0.9:
            boxes.append((x, y, w, h))
    return boxes


def _detect_faces_haar(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using Haar Cascade (fallback)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [(x, y, w, h) for (x, y, w, h) in faces]


def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using best available detector."""
    if _USE_MTCNN:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = _detect_faces_mtcnn(image_rgb)
        if boxes:
            return boxes
    # Fallback to Haar
    return _detect_faces_haar(image_bgr)


# ---------------------------------------------------------------------------
# FFT frequency analysis (deepfakes leave artifacts in frequency domain)
# ---------------------------------------------------------------------------
def compute_fft_magnitude(image_bgr: np.ndarray, size: Tuple[int, int] = MODEL_SIZE) -> np.ndarray:
    """Compute FFT magnitude spectrum — GAN-generated faces have distinct frequency patterns.
    
    Returns a single-channel image (same size as input) containing the
    log-scaled magnitude spectrum, normalized to [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    # Normalize to [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)


# ---------------------------------------------------------------------------
# Face extraction with alignment
# ---------------------------------------------------------------------------
def _extract_face(image: np.ndarray, size: Tuple[int, int] = MODEL_SIZE) -> np.ndarray:
    """Extract the largest face from an image, or fall back to the whole image."""
    boxes = detect_faces(image)
    if boxes:
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        # Add 20% margin around the face for context
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        h_img, w_img = image.shape[:2]
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(w_img, x + w + margin_w)
        y2 = min(h_img, y + h + margin_h)
        crop = image[y1:y2, x1:x2]
    else:
        crop = image
    return cv2.resize(crop, size)


# ---------------------------------------------------------------------------
# Deepfake-specific augmentations
# ---------------------------------------------------------------------------
def _apply_jpeg_compression(image: np.ndarray, quality: int = None) -> np.ndarray:
    """Simulate JPEG compression artifacts (common in deepfakes)."""
    if quality is None:
        quality = np.random.randint(30, 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply random Gaussian blur."""
    ksize = np.random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def augment_for_deepfake(image: np.ndarray) -> np.ndarray:
    """Apply deepfake-specific augmentations randomly."""
    aug = image.copy()
    if np.random.random() < 0.3:
        aug = _apply_jpeg_compression(aug)
    if np.random.random() < 0.2:
        aug = _apply_gaussian_blur(aug)
    if np.random.random() < 0.2:
        # Random brightness/contrast shift
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.randint(-20, 20)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    return aug


# ---------------------------------------------------------------------------
# Source iteration
# ---------------------------------------------------------------------------
def _iter_sources(path: Path) -> Iterable[Path]:
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in VALID_IMAGE_EXTS.union(VALID_VIDEO_EXTS):
            yield file


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_processed_dataset(
    raw_root: str = "data/raw",
    out_root: str = "data/processed",
    split=(0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    raw_root_path = Path(raw_root)
    out_root_path = Path(out_root)
    class_names = ["real", "fake"]

    if not raw_root_path.exists():
        raise FileNotFoundError(
            f"Missing '{raw_root}'. Create folders: data/raw/real and data/raw/fake."
        )

    if out_root_path.exists():
        shutil.rmtree(out_root_path)

    for phase in ("train", "val", "test"):
        for cls in class_names:
            (out_root_path / phase / cls).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for cls in class_names:
        cls_path = raw_root_path / cls
        if not cls_path.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_path}")

        samples = list(_iter_sources(cls_path))
        rng.shuffle(samples)
        if not samples:
            continue

        n = len(samples)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        phase_map = (
            [("train", s) for s in samples[:n_train]]
            + [("val", s) for s in samples[n_train : n_train + n_val]]
            + [("test", s) for s in samples[n_train + n_val :]]
        )

        out_counter = 0
        for phase, sample in phase_map:
            if sample.suffix.lower() in VALID_IMAGE_EXTS:
                img = cv2.imread(str(sample))
                if img is None:
                    continue
                face = _extract_face(img)
                cv2.imwrite(str(out_root_path / phase / cls / f"{out_counter:06d}.jpg"), face)
                out_counter += 1
            else:
                temp_frames = extract_frames(str(sample), output_dir="frames", every_n=12, max_frames=40)
                for frame in sorted(Path(temp_frames).glob("*.jpg")):
                    img = cv2.imread(str(frame))
                    if img is None:
                        continue
                    face = _extract_face(img)
                    cv2.imwrite(str(out_root_path / phase / cls / f"{out_counter:06d}.jpg"), face)
                    out_counter += 1

    print(f"[Preprocessing] Dataset built at '{out_root_path}' using "
          f"{'MTCNN' if _USE_MTCNN else 'Haar Cascade'} face detection.")

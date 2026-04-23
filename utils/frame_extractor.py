from __future__ import annotations

import shutil
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str = "frames", every_n: int = 5, max_frames: int = 200) -> str:
    """
    Extract sampled frames from a video.
    - every_n: keep 1 frame every N frames.
    - max_frames: hard cap to keep inference fast and memory-safe.
    """
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % max(1, every_n) == 0:
            cv2.imwrite(str(out_path / f"frame_{saved:05d}.jpg"), frame)
            saved += 1
            if saved >= max_frames:
                break
        frame_idx += 1
    capture.release()
    return str(out_path)

import cv2, os

def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames_dir = 'frames/'
    os.makedirs(frames_dir, exist_ok=True)

    while success:
        cv2.imwrite(os.path.join(frames_dir, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1
    return frames_dir

import os, cv2, numpy as np, tensorflow as tf, dlib

model = tf.keras.models.load_model('models/deepfake_cnn_model.h5')

def detect_deepfake(frames_dir, video_path):
    detector = dlib.get_frontal_face_detector()
    fake_scores = []
    
    for frame_file in sorted(os.listdir(frames_dir)):
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = cv2.resize(img[y:y+h, x:x+w], (128, 128))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            
            score = model.predict(face_img, verbose=0)[0][0]
            print(f"Frame: {frame_file} | Prediction Score: {score:.4f}")  # <-- Debug Print
            fake_scores.append(score)
    
    avg_score = float(np.mean(fake_scores)) if fake_scores else 0.0
    print(f"\nAverage Score for Video '{video_path}': {avg_score:.4f}")
    
    THRESHOLD = 0.7  # <-- Increase Threshold (You can test with 0.75 or 0.8 also)
    status = 'Fake' if avg_score > THRESHOLD else 'Real'
    
    return {'average_score': avg_score, 'status': status}

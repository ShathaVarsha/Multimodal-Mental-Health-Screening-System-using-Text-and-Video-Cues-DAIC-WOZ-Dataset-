"""
Simple camera feature extraction using OpenCV Haar Cascades
This is a fallback for systems without advanced facial landmark detection
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def process_webcam_frame_opencv(frame_data: str) -> dict:
    """
    Process webcam frame using OpenCV
    Returns AU-like features (22), pose (6), gaze (6) with confidence score
    """
    try:
        # Decode base64
        img_data = base64.b64decode(frame_data.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Extract features
            au_features = extract_au_features(face_roi, w, h)
            pose_features = extract_pose(x, y, w, h, img_array.shape)
            gaze_features = extract_gaze(face_roi, w, h)
            
            # Calculate confidence based on face size and feature detection
            face_size = w * h
            img_size = gray.shape[0] * gray.shape[1]
            size_ratio = face_size / img_size
            confidence = min(size_ratio * 10, 1.0)  # Normalize to 0-1
            
            return {
                "au_features": au_features, 
                "pose_features": pose_features, 
                "gaze_features": gaze_features,
                "confidence": confidence,
                "face_detected": True
            }
        else:
            return {
                "au_features": np.zeros(22), 
                "pose_features": np.zeros(6), 
                "gaze_features": np.zeros(6),
                "confidence": 0.0,
                "face_detected": False
            }
    except Exception as e:
        return {
            "au_features": np.zeros(22), 
            "pose_features": np.zeros(6), 
            "gaze_features": np.zeros(6),
            "confidence": 0.0,
            "face_detected": False,
            "error": str(e)
        }

def extract_au_features(face_gray, w, h):
    """Extract 22 AU-like features"""
    features = np.zeros(22)
    
    # Eyes
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
    features[0] = min(len(eyes) / 2.0, 1.0)
    features[1] = features[0]
    
    # Smile
    smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20)
    features[4] = min(len(smiles), 1.0)
    features[5] = features[4]
    
    # Brightness
    brightness = np.mean(face_gray) / 255.0
    features[6] = brightness
    features[7] = brightness
    
    # Fill rest
    for i in range(8, 22):
        features[i] = features[i % 8] * 0.9
    
    return np.clip(features, 0, 1)

def extract_pose(x, y, w, h, img_shape):
    """Extract 6 pose features"""
    img_h, img_w = img_shape[:2]
    center_x = (x + w/2) / img_w
    center_y = (y + h/2) / img_h
    yaw = (center_x - 0.5) * 2
    pitch = (center_y - 0.5) * 2
    roll = 0.0
    z = 1.0 - min((w*h)/(img_w*img_h)*10, 1.0)
    return np.array([pitch, yaw, roll, center_x, center_y, z])

def extract_gaze(face_gray, w, h):
    """Extract 6 gaze features"""
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
    features = np.zeros(6)
    
    if len(eyes) >= 1:
        ex, ey, ew, eh = eyes[0]
        features[0] = (ex + ew/2) / w
        features[1] = (ey + eh/2) / h
        features[2] = ew / w
    
    if len(eyes) >= 2:
        ex, ey, ew, eh = eyes[1]
        features[3] = (ex + ew/2) / w
        features[4] = (ey + eh/2) / h
        features[5] = ew / w
    else:
        features[3:6] = features[0:3]
    
    return features

# Test
if __name__ == "__main__":
    print("OpenCV Haar Cascades loaded successfully")
    print("Face cascade:", face_cascade.empty() == False)
    print("Eye cascade:", eye_cascade.empty() == False)
    print("Smile cascade:", smile_cascade.empty() == False)

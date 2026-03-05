"""
Video Processor Service
Processes recorded video blobs and extracts facial features for micro-expression analysis
"""
import numpy as np
import cv2
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


class VideoProcessor:
    """Processes video files and extracts facial features"""
    
    def __init__(self):
        """Initialize video processor"""
        self.temp_dir = Path(tempfile.gettempdir()) / "assessment_videos"
        self.temp_dir.mkdir(exist_ok=True)
        
    def process_video_blob(self, video_blob: bytes, session_id: str, 
                          question_responses: List[Dict]) -> Dict:
        """
        Process recorded video blob and extract facial features
        
        Args:
            video_blob: Raw video file bytes
            session_id: Assessment session ID
            question_responses: List of response dicts with timing info
            
        Returns:
            Dictionary with extracted features and detected micro-expressions
        """
        try:
            print(f"[{session_id}] Processing video blob ({len(video_blob)} bytes)...")
            
            # Save blob to temp file
            video_file = self._save_video_blob(video_blob, session_id)
            print(f"[{session_id}] Video saved to: {video_file}")
            
            # Extract frames and features
            frames, frame_times = self._extract_frames(video_file)
            print(f"[{session_id}] Extracted {len(frames)} frames, {len(frame_times)} timestamps")
            
            if len(frames) == 0:
                print(f"[{session_id}] WARNING: No frames extracted from video")
                return {
                    'success': False,
                    'error': 'No frames extracted from video',
                    'video_processed': False,
                    'fallback_available': True
                }
            
            print(f"[{session_id}] Video duration: {frame_times[-1]:.2f}s, FPS estimated: {self._estimate_fps(frame_times):.1f}")
            
            # Detect faces and extract facial features per frame
            facial_features = self._extract_facial_features(frames, frame_times)
            print(f"[{session_id}] Extracted facial features from {len(frames)} frames")
            
            faces_detected = sum(1 for f in facial_features if f.get('face_detected', False))
            print(f"[{session_id}] Face detected in {faces_detected}/{len(facial_features)} frames ({100*faces_detected/len(frames):.1f}%)")
            
            # Generate AU-like patterns from facial features
            au_patterns = self._generate_au_patterns(facial_features, question_responses)
            print(f"[{session_id}] Generated AU patterns shape: {au_patterns.shape}")
            
            # Create response-level analysis
            response_analysis = self._analyze_by_response(
                au_patterns, frame_times, question_responses
            )
            
            success = faces_detected > 0
            
            return {
                'success': success,
                'video_processed': True,
                'total_frames': len(frames),
                'duration_seconds': float(frame_times[-1] if len(frame_times) > 0 else 0),
                'fps': self._estimate_fps(frame_times),
                'au_patterns': au_patterns.tolist() if hasattr(au_patterns, 'tolist') else au_patterns,
                'response_microexpressions': response_analysis,
                'facial_features_detected': faces_detected > 0,
                'face_detection_rate': float(faces_detected / len(frames)) if len(frames) > 0 else 0.0
            }
            
        except Exception as e:
            print(f"[{session_id}] Video processing error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'video_processed': False,
                'fallback_available': True,
                'traceback': traceback.format_exc()
            }
    
    def _save_video_blob(self, video_blob: bytes, session_id: str) -> Path:
        """Save video blob to temporary file"""
        video_file = self.temp_dir / f"{session_id}_{datetime.now().timestamp()}.webm"
        with open(video_file, 'wb') as f:
            f.write(video_blob)
        return video_file
    
    def _extract_frames(self, video_file: Path, max_frames: int = 300) -> Tuple[List, np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_file: Path to video file
            max_frames: Maximum frames to extract
            
        Returns:
            Tuple of (frames list, frame_times array)
        """
        frames = []
        frame_times = []
        
        try:
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                return [], np.array([])
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Default fallback
            
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for processing
                frame_resized = cv2.resize(frame_rgb, (320, 240))
                frames.append(frame_resized)
                
                # Calculate frame time
                frame_time = frame_count / fps
                frame_times.append(frame_time)
                
                frame_count += 1
            
            cap.release()
            
            return frames, np.array(frame_times)
        
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return [], np.array([])
    
    def _extract_facial_features(self, frames: List, frame_times: np.ndarray) -> List[Dict]:
        """
        Extract facial features from frames using basic CV techniques
        
        Args:
            frames: List of image frames
            frame_times: Array of frame timestamps
            
        Returns:
            List of detected facial features per frame
        """
        features_per_frame = []
        
        # Load face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded successfully
        if face_cascade.empty():
            print(f"WARNING: Could not load face cascade from {cascade_path}")
            print("Trying alternative cascade path...")
            # Try alternative location
            import os
            alt_cascade = cv2.data.haarcascades.replace('\\haarcascades', '/haarcascades') + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(alt_cascade)
        
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        print(f"Cascade loaded: face_cascade.empty()={face_cascade.empty()}, eye_cascade.empty()={eye_cascade.empty()}")
        
        for idx, frame in enumerate(frames):
            frame_features = {
                'frame_idx': idx,
                'timestamp': float(frame_times[idx]),
                'face_detected': False,
                'face_area': 0,
                'mouth_openness': 0.0,
                'eye_aspect_ratio': 0.0,
                'head_bounding_box': None,
                'motion_intensity': 0.0,
                'brightness': float(np.mean(frame))
            }
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Try multiple detection scales if first attempt fails
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            # If no faces found, try with more permissive parameters
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
            
            # Last resort: very permissive
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(gray, 1.05, 1, minSize=(15, 15))
            
            if len(faces) > 0:
                frame_features['face_detected'] = True
                
                # Get largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                frame_features['head_bounding_box'] = [int(x), int(y), int(w), int(h)]
                frame_features['face_area'] = float(w * h)
                
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Estimate mouth openness from lower part of face
                mouth_roi = face_roi[int(h*0.6):, :]
                if mouth_roi.size > 0:
                    mouth_openness = float(np.std(mouth_roi)) / 255.0
                    frame_features['mouth_openness'] = min(1.0, max(0.0, mouth_openness))
                
                # Detect eyes
                eyes = eye_cascade.detectMultiScale(face_roi)
                if len(eyes) >= 2:
                    # Estimate eye aspect ratio from detected eyes
                    eye_heights = [e[3] for e in eyes[:2]]
                    eye_widths = [e[2] for e in eyes[:2]]
                    eye_aspect_ratio = np.mean(eye_heights) / (np.mean(eye_widths) + 1e-6)
                    frame_features['eye_aspect_ratio'] = float(min(1.0, eye_aspect_ratio))
            
            # Calculate motion intensity (frame difference)
            if idx > 0:
                prev_gray = cv2.cvtColor(frames[idx-1], cv2.COLOR_RGB2GRAY)
                motion = cv2.absdiff(prev_gray, gray)
                motion_intensity = float(np.mean(motion)) / 255.0
                frame_features['motion_intensity'] = motion_intensity
            
            features_per_frame.append(frame_features)
        
        return features_per_frame
    
    def _generate_au_patterns(self, facial_features: List[Dict], 
                             question_responses: List[Dict]) -> np.ndarray:
        """
        Generate AU-like patterns from extracted facial features
        
        Args:
            facial_features: List of facial features per frame
            question_responses: List of response info
            
        Returns:
            Array of AU patterns (frames x AUs)
        """
        if not facial_features:
            return np.zeros((1, 17))
        
        n_frames = len(facial_features)
        n_aus = 17  # Standard AU count
        au_patterns = np.zeros((n_frames, n_aus))
        
        # Extract feature signals
        mouth_openness = np.array([f['mouth_openness'] for f in facial_features])
        eye_aspect = np.array([f['eye_aspect_ratio'] for f in facial_features])
        motion_intensity = np.array([f['motion_intensity'] for f in facial_features])
        brightness = np.array([f['brightness'] for f in facial_features])
        
        # Normalize signals
        mouth_openness = self._normalize_signal(mouth_openness)
        eye_aspect = self._normalize_signal(eye_aspect)
        motion_intensity = self._normalize_signal(motion_intensity)
        
        # Map facial features to AUs (simplified mapping)
        # AU1 (inner brow raiser) - emotion intensity
        au_patterns[:, 0] = motion_intensity * 0.7
        
        # AU4 (brow lowerer) - concentration/emotion
        au_patterns[:, 3] = (motion_intensity + eye_aspect) / 2 * 0.6
        
        # AU6 (cheek raiser) - smile intensity
        au_patterns[:, 5] = (1 - eye_aspect) * mouth_openness * 0.5
        
        # AU12 (lip corner puller) - mouth movement
        au_patterns[:, 11] = mouth_openness * 0.7
        
        # AU15 (lip corner depressor) - sadness
        au_patterns[:, 14] = (1 - mouth_openness) * motion_intensity * 0.4
        
        # AU17 (chin raiser) - mouth tension
        au_patterns[:, 16] = motion_intensity * (1 - mouth_openness) * 0.5
        
        # Add random variations to simulate natural AU fluctuations
        noise = np.random.normal(0, 0.05, au_patterns.shape)
        au_patterns = np.clip(au_patterns + noise, 0, 1)
        
        return au_patterns
    
    def _analyze_by_response(self, au_patterns: np.ndarray, 
                            frame_times: np.ndarray,
                            question_responses: List[Dict]) -> Dict:
        """
        Analyze micro-expressions for each response
        
        Args:
            au_patterns: Array of AU patterns over time
            frame_times: Frame timestamps
            question_responses: Response info with timing
            
        Returns:
            Dictionary of micro-expressions per response
        """
        response_analysis = {}
        
        for idx, response in enumerate(question_responses):
            response_key = f"response_{idx + 1}"
            
            # Create AU pattern for this response
            # Use entire video segment or specific window
            au_segment = au_patterns
            
            # Calculate feature activations for this response
            au_means = np.mean(au_segment, axis=0)
            au_stds = np.std(au_segment, axis=0)
            
            # Determine dominant AUs (those with highest activation)
            top_au_indices = np.argsort(au_means)[-3:][::-1]
            
            response_analysis[response_key] = {
                'response_text': response.get('text', ''),
                'response_scale': response.get('scale', 0),
                'au_intensities': au_means.tolist(),
                'au_variance': au_stds.tolist(),
                'dominant_aus': top_au_indices.tolist(),
                'motion_level': float(np.mean(au_stds)),
                'expression_stability': float(1.0 - np.std(au_means))
            }
        
        return response_analysis
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to 0-1 range"""
        if len(signal) == 0:
            return signal
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val == min_val:
            return np.zeros_like(signal)
        return (signal - min_val) / (max_val - min_val)
    
    def _estimate_fps(self, frame_times: np.ndarray) -> float:
        """Estimate FPS from frame times"""
        if len(frame_times) < 2:
            return 0.0
        time_diffs = np.diff(frame_times)
        avg_frame_time = np.mean(time_diffs)
        if avg_frame_time > 0:
            return float(1.0 / avg_frame_time)
        return 0.0

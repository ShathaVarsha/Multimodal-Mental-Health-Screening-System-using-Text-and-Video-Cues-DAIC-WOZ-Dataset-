"""
Feature extraction from facial and audio data
Processes CLNF (Facial Action Units), HOG, COVAREP, and other features
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class FeatureExtractor:
    """Extracts and processes multimodal features from video data"""
    
    # Feature dimensions for DAIC-WOZ dataset
    AU_FEATURES = 17  # Action Units
    HOG_FEATURES = 489  # Histogram of Gradients
    POSE_FEATURES = 6  # Head pose (3 angles + 3D position)
    GAZE_FEATURES = 4  # Gaze direction
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.feature_cache = {}
    
    def extract_participant_features(self, participant_id: int, segment: str = "all") -> Dict:
        """
        Extract all features for a participant
        
        Args:
            participant_id: Unique participant ID
            segment: "all", "utterance" or specific segment
            
        Returns:
            Dictionary with processed features
        """
        participant_dir = self.data_dir / str(participant_id)
        if not participant_dir.exists():
            return {"error": f"Participant {participant_id} not found"}
        
        features = {
            "participant_id": participant_id,
            "action_units": self.load_action_units(participant_dir),
            "hog": self.load_hog_features(participant_dir),
            "pose": self.load_pose_features(participant_dir),
            "gaze": self.load_gaze_features(participant_dir),
            "audio": self.load_audio_features(participant_dir),
            "transcript": self.load_transcript(participant_dir)
        }
        
        return features
    
    def load_action_units(self, participant_dir: Path) -> Optional[np.ndarray]:
        """Load Action Unit features (CLNF_AUs.txt)"""
        au_file = participant_dir / f"{participant_dir.name}_CLNF_AUs.txt"
        if not au_file.exists():
            return None
        
        try:
            data = np.loadtxt(au_file, skiprows=0)
            return self.normalize_features(data)
        except Exception as e:
            print(f"Error loading AUs: {e}")
            return None
    
    def load_hog_features(self, participant_dir: Path) -> Optional[np.ndarray]:
        """Load HOG (Histogram of Gradients) features"""
        hog_file = participant_dir / f"{participant_dir.name}_CLNF_hog.txt"
        if not hog_file.exists():
            return None
        
        try:
            data = np.loadtxt(hog_file, skiprows=0)
            return self.normalize_features(data)
        except Exception as e:
            print(f"Error loading HOG: {e}")
            return None
    
    def load_pose_features(self, participant_dir: Path) -> Optional[np.ndarray]:
        """Load head pose features"""
        pose_file = participant_dir / f"{participant_dir.name}_CLNF_pose.txt"
        if not pose_file.exists():
            return None
        
        try:
            data = np.loadtxt(pose_file, skiprows=0)
            return self.normalize_features(data)
        except Exception as e:
            print(f"Error loading pose: {e}")
            return None
    
    def load_gaze_features(self, participant_dir: Path) -> Optional[np.ndarray]:
        """Load gaze direction features"""
        gaze_file = participant_dir / f"{participant_dir.name}_CLNF_gaze.txt"
        if not gaze_file.exists():
            return None
        
        try:
            data = np.loadtxt(gaze_file, skiprows=0)
            return self.normalize_features(data)
        except Exception as e:
            print(f"Error loading gaze: {e}")
            return None
    
    def load_audio_features(self, participant_dir: Path) -> Optional[Dict]:
        """Load COVAREP audio features"""
        covarep_file = participant_dir / f"{participant_dir.name}_COVAREP.csv"
        if not covarep_file.exists():
            return None
        
        try:
            data = pd.read_csv(covarep_file)
            return self.aggregate_audio_features(data)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def load_transcript(self, participant_dir: Path) -> Optional[str]:
        """Load transcript text"""
        trans_file = participant_dir / f"{participant_dir.name}_TRANSCRIPT.csv"
        if not trans_file.exists():
            return None
        
        try:
            data = pd.read_csv(trans_file)
            if 'value' in data.columns:
                return " ".join(data['value'].astype(str).tolist())
            return None
        except Exception as e:
            print(f"Error loading transcript: {e}")
            return None
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        if len(features) == 0:
            return features
        
        min_val = np.min(features)
        max_val = np.max(features)
        
        if max_val - min_val <= 1e-10:
            return np.zeros_like(features)
        
        return (features - min_val) / (max_val - min_val)
    
    @staticmethod
    def aggregate_audio_features(covarep_df: pd.DataFrame) -> Dict:
        """Aggregate audio features (mean, std, min, max)"""
        features = {}
        for col in covarep_df.columns:
            if col != 'frame' and pd.api.types.is_numeric_dtype(covarep_df[col]):
                features[f"{col}_mean"] = float(covarep_df[col].mean())
                features[f"{col}_std"] = float(covarep_df[col].std())
                features[f"{col}_min"] = float(covarep_df[col].min())
                features[f"{col}_max"] = float(covarep_df[col].max())
        
        return features
    
    def compute_feature_statistics(self, features: Dict) -> Dict:
        """Compute statistical summaries of extracted features"""
        stats = {}
        
        for key, feature_data in features.items():
            if isinstance(feature_data, np.ndarray):
                stats[f"{key}_mean"] = float(np.mean(feature_data))
                stats[f"{key}_std"] = float(np.std(feature_data))
                stats[f"{key}_min"] = float(np.min(feature_data))
                stats[f"{key}_max"] = float(np.max(feature_data))
        
        return stats

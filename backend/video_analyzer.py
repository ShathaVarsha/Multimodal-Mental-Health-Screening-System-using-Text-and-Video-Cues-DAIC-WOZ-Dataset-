"""
Video analysis module
Processes video files and extracts facial features for depression assessment
"""
import os
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np

class VideoAnalyzer:
    """Processes video files and extracts depression-related features"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def analyze_video(self, participant_id: int) -> Dict:
        """
        Analyze video for a participant
        
        Args:
            participant_id: Participant ID
            
        Returns:
            Dictionary with video analysis results
        """
        results = {
            'participant_id': participant_id,
            'depression_indicators': {},
            'micro_expressions': [],
            'au_patterns': {},
            'processing_status': 'success'
        }
        
        participant_dir = self.data_dir / str(participant_id)
        if not participant_dir.exists():
            results['processing_status'] = 'error'
            results['error'] = f"Participant directory not found: {participant_id}"
            return results
        
        # Load AU (Action Unit) data
        aus = self._load_au_data(participant_dir)
        if aus is not None:
            # Calculate depression indicators from AU patterns
            depression_score = self._calculate_depression_from_au(aus)
            results['depression_indicators']['depression_severity'] = depression_score
            results['depression_indicators']['primary_indicators'] = self._identify_au_indicators(aus)
            results['au_patterns']['mean'] = float(np.mean(aus))
            results['au_patterns']['std'] = float(np.std(aus))
            results['au_patterns']['activation_ratio'] = float(np.sum(aus > 0.1) / len(aus))
        
        return results
    
    def _load_au_data(self, participant_dir: Path) -> Optional[np.ndarray]:
        """Load Action Unit data from file"""
        au_file = participant_dir / f"{participant_dir.name}_CLNF_AUs.txt"
        
        if not au_file.exists():
            return None
        
        try:
            data = np.loadtxt(au_file, skiprows=0)
            # Data should be frames x AUs (typically 17 AUs)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return data
        except Exception as e:
            print(f"Error loading AU data: {e}")
            return None
    
    @staticmethod
    def _calculate_depression_from_au(au_data: np.ndarray) -> float:
        """
        Calculate depression severity score from AU activations
        
        Depression-related AUs:
        - AU1 (inner brow raiser): sadness/concern
        - AU4 (brow lowerer): anger/concentration/sadness
        - AU6 (cheek raiser): smile intensity
        - AU12 (lip corner puller): smile
        - AU15 (lip corner depressor): sadness
        
        Returns:
            Score 0-1 indicating depression severity
        """
        # Check AU indices
        depression_aus = {
            1: 0.15,   # AU1 - inner brow
            4: 0.20,   # AU4 - brow lower (strong indicator)
            6: -0.10,  # AU6 - smile (negative weight - reduced smiling)
            12: -0.10, # AU12 - smile (negative weight)
            15: 0.25   # AU15 - sadness (strong indicator)
        }
        
        score = 0.0
        
        # Calculate mean activation for each AU
        for au_idx, weight in depression_aus.items():
            if au_idx - 1 < au_data.shape[1]:  # AU indices are 1-based
                au_activation = np.mean(au_data[:, au_idx - 1])
                score += au_activation * weight
        
        # Normalize to 0-1 range
        score = max(0.0, min(1.0, (score + 0.5) / 1.0))
        
        return float(score)
    
    @staticmethod
    def _identify_au_indicators(au_data: np.ndarray) -> List[str]:
        """Identify which AUs are activated above threshold"""
        indicators = []
        
        # Calculate mean activation per AU
        au_means = np.mean(au_data, axis=0)
        threshold = np.median(au_means) + np.std(au_means)
        
        au_names = {
            0: "Inner Brow Raiser",
            3: "Brow Lowerer",
            5: "Cheek Raiser",
            11: "Lip Corner Puller",
            14: "Lip Corner Depressor"
        }
        
        for au_idx, threshold_val in enumerate(au_means):
            if threshold_val > threshold and au_idx in au_names:
                indicators.append(f"{au_names[au_idx]} (AU{au_idx+1})")
        
        return indicators
    
    def extract_temporal_features(self, participant_id: int) -> Dict:
        """Extract temporal dynamics of facial expressions"""
        participant_dir = self.data_dir / str(participant_id)
        
        results = {
            'participant_id': participant_id,
            'temporal_features': {}
        }
        
        aus = self._load_au_data(participant_dir)
        if aus is None:
            return results
        
        # Calculate temporal statistics
        au_diff = np.diff(aus, axis=0)  # Changes over time
        
        results['temporal_features'] = {
            'mean_velocity': float(np.mean(np.abs(au_diff))),
            'peak_velocity': float(np.max(np.abs(au_diff))),
            'velocity_std': float(np.std(np.abs(au_diff))),
            'movement_frequency': float(np.sum(np.abs(au_diff) > np.std(au_diff))),
            'total_frames': int(aus.shape[0])
        }
        
        return results

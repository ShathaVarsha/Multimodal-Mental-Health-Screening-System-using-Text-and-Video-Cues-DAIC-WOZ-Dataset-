"""
Micro-expression detection service
Identifies brief facial micro-expressions using AU patterns
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

class MicroExpressionDetector:
    """Detects and classifies micro-expressions from Action Unit activations"""
    
    # Individual AU descriptions
    AU_DESCRIPTIONS = {
        1: 'Inner brow raiser',
        2: 'Outer brow raiser',
        3: 'Brow lowerer',
        4: 'Brow lowerer (additional)',
        5: 'Upper lid raiser',
        6: 'Cheek raiser',
        7: 'Lid tightener',
        8: 'Lips toward teeth',
        9: 'Nose wrinkler',
        10: 'Upper lip raiser',
        11: 'Nasolabial deepener',
        12: 'Lip corner puller',
        13: 'Cheek puffer',
        14: 'Dimpler',
        15: 'Lip corner depressor',
        16: 'Lower lip depressor',
        17: 'Chin raiser',
        18: 'Lip puckerer',
        19: 'Tongue show',
        20: 'Lip stretcher',
        21: 'Neck tightener',
        22: 'Lip funneler',
        23: 'Lip tightener',
        24: 'Lip press',
        25: 'Lips part',
        26: 'Jaw drop',
        27: 'Mouth stretch',
        28: 'Lip suck'
    }
    
    # Detection parameters optimized for AVEC2017 AU data
    PEAK_DETECTION_THRESHOLD = 0.05
    PATTERN_ACTIVATION_THRESHOLD = 0.12
    CONFIDENCE_THRESHOLD = 0.2
    MINIMUM_FRAMES = 2
    
    def __init__(self):
        self.detection_history = []
        self.per_question_detections = {}
        self.confidence_threshold = self.CONFIDENCE_THRESHOLD
        self.current_question_id = None
    
    def detect_microexpressions(self, au_intensities: np.ndarray, 
                                frame_times: np.ndarray,
                                question_id: Optional[str] = None,
                                time_window: float = 2.0) -> List[Dict]:
        """Detect individual AU activations from AU sequence data"""
        if question_id:
            self.current_question_id = question_id
            if question_id not in self.per_question_detections:
                self.per_question_detections[question_id] = []
        
        detections = []
        
        if au_intensities.shape[0] == 0 or au_intensities.ndim != 2:
            return detections
        
        # Apply time window filter
        if time_window and len(frame_times) > 0 and len(frame_times) > 30:
            window_mask = frame_times <= (frame_times[0] + time_window)
            au_intensities = au_intensities[window_mask]
            frame_times = frame_times[window_mask]
        
        # Detect individual AU activations
        for au_idx in range(au_intensities.shape[1]):
            au_signal = au_intensities[:, au_idx]
            
            # Find activation periods for this AU
            active_frames = np.where(au_signal > self.PATTERN_ACTIVATION_THRESHOLD)[0]
            
            if len(active_frames) > 0:
                groups = self._group_consecutive_frames(active_frames)
                
                for group in groups:
                    if len(group) >= self.MINIMUM_FRAMES:
                        start_frame = group[0]
                        end_frame = group[-1]
                        peak_frame = group[int(len(group) / 2)]
                        
                        confidence = float(np.mean(au_signal[group]))
                        
                        if confidence >= self.confidence_threshold:
                            au_name = self.AU_DESCRIPTIONS.get(au_idx, f'AU{au_idx}')
                            
                            detection = {
                                'expression': f'AU{au_idx} - {au_name}',
                                'au_index': au_idx,
                                'au_name': au_name,
                                'confidence': confidence,
                                'start_time': float(frame_times[start_frame]),
                                'peak_time': float(frame_times[peak_frame]),
                                'end_time': float(frame_times[min(end_frame, len(frame_times)-1)]),
                                'duration': float(frame_times[min(end_frame, len(frame_times)-1)] - frame_times[start_frame]),
                                'intensity': float(au_signal[peak_frame]),
                                'description': au_name,
                                'question_id': question_id
                            }
                            
                            detections.append(detection)
                            
                            if question_id:
                                self.per_question_detections[question_id].append(detection)
        
        self.detection_history.extend(detections)
        return detections
    
    def _find_au_peaks(self, au_intensities: np.ndarray) -> Dict[int, List[int]]:
        """Find peak activations for each AU"""
        peaks = {}
        
        for au_idx in range(au_intensities.shape[1]):
            au_signal = au_intensities[:, au_idx]
            peak_frames = []
            
            for i in range(1, len(au_signal) - 1):
                if au_signal[i] > au_signal[i-1] and au_signal[i] > au_signal[i+1]:
                    if au_signal[i] > self.PEAK_DETECTION_THRESHOLD:
                        peak_frames.append(i)
            
            if peak_frames:
                peaks[au_idx] = peak_frames
        
        return peaks
    
    def _match_pattern(self, au_intensities: np.ndarray, au_pattern: List[int],
                       peaks: Dict) -> List[Dict]:
        """Match AU pattern against detected peaks (not used with individual AU detection)"""
        matches = []
        return matches
    
    @staticmethod
    def _group_consecutive_frames(frames: np.ndarray) -> List[List[int]]:
        """Group consecutive frame indices"""
        if len(frames) == 0:
            return []
        
        groups = []
        current_group = [frames[0]]
        
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] <= 2:
                current_group.append(frames[i])
            else:
                groups.append(current_group)
                current_group = [frames[i]]
        
        groups.append(current_group)
        return groups
    
    def get_micro_expression_summary(self) -> Dict:
        """Get statistics of detected AU activations"""
        if not self.detection_history:
            return {"total_detections": 0}
        
        summary = {"total_detections": len(self.detection_history)}
        
        au_counts = {}
        for detection in self.detection_history:
            au_expr = detection['expression']
            au_counts[au_expr] = au_counts.get(au_expr, 0) + 1
        
        summary['by_au'] = au_counts
        summary['mean_intensity'] = float(np.mean([d['intensity'] for d in self.detection_history]))
        summary['au_list'] = sorted(au_counts.keys())
        
        return summary
    
    def get_per_question_summary(self) -> Dict:
        """Get AU activation summary per question"""
        per_question_summary = {}
        
        for question_id, detections in self.per_question_detections.items():
            if detections:
                au_counts = {}
                for detection in detections:
                    au_expr = detection['expression']
                    au_counts[au_expr] = au_counts.get(au_expr, 0) + 1
                
                per_question_summary[question_id] = {
                    'total_detections': len(detections),
                    'aus_detected': au_counts,
                    'mean_intensity': float(np.mean([d['intensity'] for d in detections])),
                    'au_list': sorted(au_counts.keys())
                }
        
        return per_question_summary
    
    def _generate_clinical_interpretation(self, au_counts: Dict) -> str:
        """Generate overall AU activation summary"""
        total = sum(au_counts.values())
        if total == 0:
            return "No AU activations detected"
        
        return f"Total AU activations: {total} across {len(au_counts)} unique action units"
    
    def _get_question_clinical_note(self, detections: List[Dict]) -> str:
        """Generate AU activation note for specific question"""
        if not detections:
            return "No AU activations detected"
        
        au_counts = {}
        for d in detections:
            au_expr = d['expression']
            au_counts[au_expr] = au_counts.get(au_expr, 0) + 1
        
        au_list = ", ".join(sorted(au_counts.keys()))
        return f"AUs detected: {au_list}"

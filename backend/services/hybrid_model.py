"""
Hybrid depression detection model
Combines facial expressions (video) and natural language features
"""
import numpy as np
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path

class HybridDepressionModel:
    """
    Multi-modal depression severity classifier
    Integrates video (facial AU) and text features (audio disabled)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.au_model = None
        self.audio_model = None
        self.text_model = None
        self.video_scale_model = None
        self.video_scale_scaler = None
        self.fusion_weights = {
            'video': 0.6,
            'text': 0.4
        }
        self.model_path = Path(model_path) if model_path else Path("ml_training/saved_models")
        self.is_trained = False
        
        if self.model_path.exists():
            self.load_models()
    
    def predict_depression_severity(self, features: Dict) -> Dict:
        """
        Predict depression severity from video + text features
        
        Args:
            features: Dict with 'video' and 'text' keys
            
        Returns:
            Prediction with confidence scores and severity classification
        """
        predictions = {
            'severity': 'unknown',
            'confidence': 0.0,
            'component_scores': {},
            'reasoning': []
        }
        
        # Get individual modality predictions
        video_score = self._predict_from_video(features.get('video', {}))
        audio_score = None
        text_score = self._predict_from_text(features.get('text', {}))
        
        predictions['component_scores'] = {
            'video': video_score,
            'audio': audio_score,
            'text': text_score
        }
        
        # Fuse predictions
        if video_score is not None or text_score is not None:
            fused_score = self._fuse_predictions(video_score, None, text_score)
            predictions['severity'] = self._score_to_severity(fused_score)
            predictions['confidence'] = min(fused_score[1], 0.95)  # Confidence from prediction
            predictions['risk_level'] = self._calculate_risk_level(fused_score, predictions)
        
        return predictions
    
    def _predict_from_video(self, video_features: Dict) -> Optional[Tuple[float, float]]:
        """
        Predict from facial expression features
        Returns: (prediction_score, confidence)
        """
        if not video_features:
            return None
        
        # Extract AUs if available
        aus = video_features.get('action_units')
        if aus is None:
            return None
        
        # Key AUs for depression: brow lowering (AU4), sadness (AU1, AU15)
        depression_aus = [1, 4, 15]
        sadness_score = 0
        
        try:
            if isinstance(aus, (list, np.ndarray)):
                aus_array = np.array(aus)
                if len(aus_array) > max(depression_aus):
                    sadness_score = np.mean([aus_array[au] for au in depression_aus if au < len(aus_array)])
        except:
            pass
        
        # Normalize to [0, 1]
        prediction_score = min(sadness_score, 1.0)
        confidence = 0.6  # Baseline confidence for facial features
        
        return (prediction_score, confidence)
    
    def _predict_from_audio(self, audio_features: Dict) -> Optional[Tuple[float, float]]:
        """
        Audio modality is disabled.
        """
        return None
    
    def _predict_from_text(self, text_features: Dict) -> Optional[Tuple[float, float]]:
        """
        Predict from linguistic features
        Returns: (prediction_score, confidence)
        """
        if not text_features:
            return None
        
        # Depression markers in text:
        # - First person pronouns (rumination)
        # - Negative words
        # - Low diversity
        
        depression_score = 0.0
        
        if 'first_person_ratio' in text_features:
            fpp = float(text_features['first_person_ratio'])
            depression_score += min(fpp * 0.5, 0.2)  # Higher first person = more rumination
        
        if 'negative_word_ratio' in text_features:
            neg = float(text_features['negative_word_ratio'])
            depression_score += min(neg * 1.5, 0.3)  # Negative words
        
        if 'semantic_diversity' in text_features:
            diversity = float(text_features['semantic_diversity'])
            if diversity < 0.4:  # Low semantic diversity
                depression_score += 0.2
        
        confidence = 0.55
        return (min(depression_score, 1.0), confidence)
    
    def _fuse_predictions(self, video_score: Optional[Tuple], 
                         audio_score: Optional[Tuple],
                         text_score: Optional[Tuple]) -> Tuple[float, float]:
        """Fuse predictions using video + text weighted average (audio ignored)."""
        
        scores = []
        weights = []
        
        if video_score is not None:
            scores.append(video_score[0])
            weights.append(self.fusion_weights['video'])
        
        if text_score is not None:
            scores.append(text_score[0])
            weights.append(self.fusion_weights['text'])
        
        if not scores:
            return (0.0, 0.0)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        fused_score = np.average(scores, weights=weights)
        fused_confidence = np.mean([s[1] for s in [video_score, text_score] if s is not None])
        
        return (fused_score, fused_confidence)
    
    @staticmethod
    def _score_to_severity(score_tuple: Tuple[float, float]) -> str:
        """Convert prediction score to severity level"""
        score = score_tuple[0]
        
        if score < 0.2:
            return "minimal"
        elif score < 0.4:
            return "mild"
        elif score < 0.6:
            return "moderate"
        elif score < 0.8:
            return "moderately_severe"
        else:
            return "severe"
    
    @staticmethod
    def _calculate_risk_level(score_tuple: Tuple[float, float], predictions: Dict) -> str:
        """
        Determine risk level (low, moderate, high, crisis)
        Considers suicide indicators and severity
        """
        score = score_tuple[0]
        
        # Check for crisis indicators
        has_suicide_thoughts = predictions.get('has_suicide_thoughts', False)
        
        if has_suicide_thoughts:
            return "crisis"
        elif score > 0.8:
            return "high"
        elif score > 0.5:
            return "moderate"
        else:
            return "low"
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            au_model_path = self.model_path / "au_model.pkl"
            if au_model_path.exists():
                with open(au_model_path, 'rb') as f:
                    self.au_model = pickle.load(f)

            # Load a trained video model for AU-pattern -> depression-scale inference
            candidate_video_models = [
                self.model_path / "au_expert.pkl",
                self.model_path / "hybrid_optimized" / "au_expert.pkl",
                self.model_path / "hybrid_proper_split" / "au_expert.pkl",
                self.model_path / "hybrid_107_expression" / "model.pkl",
                self.model_path / "ensemble_discriminative" / "ensemble_model.pkl"
            ]
            for candidate in candidate_video_models:
                if candidate.exists():
                    with open(candidate, 'rb') as f:
                        self.video_scale_model = pickle.load(f)
                    break

            candidate_scalers = [
                self.model_path / "scaler.pkl",
                self.model_path / "hybrid_optimized" / "scaler.pkl",
                self.model_path / "hybrid_proper_split" / "scaler.pkl",
                self.model_path / "hybrid_107_expression" / "scaler.pkl",
                self.model_path / "ensemble_discriminative" / "scaler.pkl"
            ]
            for candidate in candidate_scalers:
                if candidate.exists():
                    with open(candidate, 'rb') as f:
                        self.video_scale_scaler = pickle.load(f)
                    break
            
            text_model_path = self.model_path / "text_model.pkl"
            if text_model_path.exists():
                with open(text_model_path, 'rb') as f:
                    self.text_model = pickle.load(f)
            
            self.is_trained = True
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            self.is_trained = False
    
    def save_models(self, path: Optional[str] = None):
        """Save trained models to disk"""
        save_path = Path(path) if path else self.model_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.au_model:
                with open(save_path / "au_model.pkl", 'wb') as f:
                    pickle.dump(self.au_model, f)
            
            if self.text_model:
                with open(save_path / "text_model.pkl", 'wb') as f:
                    pickle.dump(self.text_model, f)
        except Exception as e:
            print(f"Error saving models: {e}")

    def predict_video_scale_from_au_patterns(self, au_patterns) -> Dict:
        """
        Predict depression severity scale from AU patterns using trained video PKL model.
        Returns scale in both 0-3 and 0-27 ranges for UI consumption.
        """
        if self.video_scale_model is None:
            return {
                'used_model': False,
                'reason': 'No trained video model (.pkl) loaded',
                'scale_0_3': None,
                'scale_0_27': None,
                'confidence': 0.0
            }

        if au_patterns is None:
            return {
                'used_model': False,
                'reason': 'No AU patterns available',
                'scale_0_3': None,
                'scale_0_27': None,
                'confidence': 0.0
            }

        try:
            au_array = np.array(au_patterns, dtype=np.float32)
            if au_array.size == 0:
                return {
                    'used_model': False,
                    'reason': 'Empty AU patterns',
                    'scale_0_3': None,
                    'scale_0_27': None,
                    'confidence': 0.0
                }

            if au_array.ndim == 1:
                au_array = au_array.reshape(-1, 1)

            feature_vector = self._build_video_feature_vector(au_array)
            expected_features = getattr(self.video_scale_model, 'n_features_in_', len(feature_vector))
            adjusted = self._adjust_feature_vector(feature_vector, int(expected_features))
            X = np.array([adjusted], dtype=np.float32)

            if self.video_scale_scaler is not None and hasattr(self.video_scale_scaler, 'transform'):
                try:
                    X = self.video_scale_scaler.transform(X)
                except Exception:
                    pass

            confidence = 0.65
            model_raw = None
            if hasattr(self.video_scale_model, 'predict_proba'):
                probabilities = self.video_scale_model.predict_proba(X)[0]
                predicted_index = int(np.argmax(probabilities))
                classes = getattr(self.video_scale_model, 'classes_', None)
                model_raw = classes[predicted_index] if classes is not None else predicted_index
                confidence = float(np.max(probabilities))
            elif hasattr(self.video_scale_model, 'predict'):
                prediction = self.video_scale_model.predict(X)
                model_raw = prediction[0]
            else:
                return {
                    'used_model': False,
                    'reason': 'Loaded video model does not support predict methods',
                    'scale_0_3': None,
                    'scale_0_27': None,
                    'confidence': 0.0
                }

            scale_0_3 = self._map_model_output_to_scale(model_raw)
            scale_0_27 = int(round((scale_0_3 / 3.0) * 27))

            return {
                'used_model': True,
                'reason': 'Predicted by trained video .pkl model',
                'raw_prediction': str(model_raw),
                'scale_0_3': int(scale_0_3),
                'scale_0_27': int(max(0, min(scale_0_27, 27))),
                'confidence': float(min(max(confidence, 0.0), 1.0))
            }
        except Exception as exc:
            return {
                'used_model': False,
                'reason': f'Video model inference failed: {exc}',
                'scale_0_3': None,
                'scale_0_27': None,
                'confidence': 0.0
            }

    @staticmethod
    def _build_video_feature_vector(au_array: np.ndarray) -> list[float]:
        """Aggregate AU frame sequence into fixed-length statistical vector."""
        if au_array.ndim == 1:
            au_array = au_array.reshape(-1, 1)

        means = np.mean(au_array, axis=0)
        stds = np.std(au_array, axis=0)
        maxima = np.max(au_array, axis=0)
        minima = np.min(au_array, axis=0)

        vector = np.concatenate([means, stds, maxima, minima]).astype(np.float32)
        return vector.tolist()

    @staticmethod
    def _adjust_feature_vector(vector: list[float], expected_features: int) -> list[float]:
        if expected_features <= 0:
            return vector
        if len(vector) == expected_features:
            return vector
        if len(vector) < expected_features:
            return vector + [0.0] * (expected_features - len(vector))
        return vector[:expected_features]

    @staticmethod
    def _map_model_output_to_scale(predicted_label) -> int:
        if isinstance(predicted_label, np.generic):
            predicted_label = predicted_label.item()

        if isinstance(predicted_label, str):
            normalized = predicted_label.strip().lower()
            severity_map = {
                'minimal': 0,
                'none': 0,
                'mild': 1,
                'moderate': 2,
                'moderately_severe': 3,
                'severe': 3
            }
            if normalized in severity_map:
                return severity_map[normalized]
            try:
                numeric = float(normalized)
                return max(0, min(int(round(numeric)), 3))
            except Exception:
                return 1

        if isinstance(predicted_label, (int, np.integer)):
            return max(0, min(int(predicted_label), 3))

        if isinstance(predicted_label, (float, np.floating)):
            numeric = float(predicted_label)
            if 0.0 <= numeric <= 1.0:
                return max(0, min(int(round(numeric * 3)), 3))
            return max(0, min(int(round(numeric)), 3))

        return 1

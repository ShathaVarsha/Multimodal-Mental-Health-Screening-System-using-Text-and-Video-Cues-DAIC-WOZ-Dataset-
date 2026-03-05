"""
Fusion engine for multimodal depression assessment
Integrates questionnaire, video, and text analysis results
"""
from typing import Dict, Optional
from datetime import datetime

class FusionEngine:
    """
    Combines results from multiple modalities (questionnaire, video, text)
    into integrated depression assessment
    """
    
    def __init__(self):
        self.modality_weights = {
            'questionnaire': 0.35,  # PHQ-9 is well-validated
            'video': 0.35,          # Facial expressions are reliable
            'text': 0.20,           # Linguistic features are supplementary
            'demographic': 0.10     # Optional: age, gender adjustments
        }
    
    def fuse_assessment(self, phq_score: Dict, video_analysis: Dict,
                       text_analysis: Dict, demographic: Optional[Dict] = None) -> Dict:
        """
        Fuse multimodal assessment results
        
        Args:
            phq_score: PHQ-9 questionnaire results
            video_analysis: Video/facial expression analysis
            text_analysis: Transcript linguistic analysis
            demographic: Optional demographic information
            
        Returns:
            Integrated depression assessment with severity and risk level
        """
        
        assessment_result = {
            'timestamp': datetime.now().isoformat(),
            'modality_scores': {},
            'fusion_method': 'weighted_average',
            'overall_severity': 'unknown',
            'overall_probability': 0.0,
            'confidence': 0.0,
            'consistency_check': {},
            'risk_level': 'unknown'
        }
        
        # 1. Normalize PHQ-9 score to 0-1 probability
        phq_prob = self._phq_to_probability(phq_score.get('total_score', 0))
        assessment_result['modality_scores']['questionnaire'] = phq_prob
        assessment_result['modality_scores']['questionnaire_raw'] = phq_score.get('total_score', 0)
        
        # 2. Extract video depression probability
        video_prob = video_analysis.get('depression_indicators', {}).get('depression_severity', 0.5)
        assessment_result['modality_scores']['video'] = video_prob
        
        # 3. Extract text depression probability from sentiment/linguistic features
        text_prob = self._text_to_probability(text_analysis)
        assessment_result['modality_scores']['text'] = text_prob
        
        # 4. Perform weighted fusion
        fusion_score = self._weighted_fusion(
            questionnaire_score=phq_prob,
            video_score=video_prob,
            text_score=text_prob
        )
        
        assessment_result['overall_probability'] = fusion_score
        assessment_result['overall_severity'] = self._probability_to_severity(fusion_score)
        
        # 5. Check consistency across modalities
        consistency = self._check_consistency(phq_prob, video_prob, text_prob)
        assessment_result['consistency_check'] = consistency
        
        # 6. Calculate confidence based on agreement
        assessment_result['confidence'] = consistency['agreement_score']
        
        # 7. Assess risk level
        risk_level = self._assess_risk_level(
            probability=fusion_score,
            phq_suicide_risk=phq_score.get('suicide_risk', False),
            consistency=consistency['agreement_score']
        )
        assessment_result['risk_level'] = risk_level
        
        return assessment_result
    
    @staticmethod
    def _phq_to_probability(phq_score: int) -> float:
        """Convert PHQ-9 score (0-27) to depression probability (0-1)"""
        # Rough scaling: higher PHQ score = higher probability
        # Linear mapping: 0 -> 0.05, 27 -> 0.95
        probability = min(0.95, max(0.05, phq_score / 27.0))
        return probability
    
    @staticmethod
    def _text_to_probability(text_analysis: Dict) -> float:
        """
        Convert linguistic features to depression probability
        Considers sentiment, negative word density, and rumination indicators
        """
        if not text_analysis:
            return 0.5  # Neutral if no data
        
        score = 0.0
        
        # Sentiment: negative sentiment indicates depression
        sentiment = text_analysis.get('sentiment_score', 0)
        score += (1 - sentiment) / 2 * 0.4  # 0-0.4 range
        
        # Negative word density
        neg_ratio = text_analysis.get('negative_word_ratio', 0)
        score += min(neg_ratio * 2, 0.3)  # 0-0.3 range
        
        # Depression risk features (rumination, existential concerns)
        depression_features = text_analysis.get('depression_risk_features', {})
        rumination = depression_features.get('rumination_score', 0)
        score += rumination * 0.3  # 0-0.3 range
        
        return min(score, 1.0)
    
    def _weighted_fusion(self, questionnaire_score: float, video_score: float,
                        text_score: float) -> float:
        """Fuse scores using weighted average"""
        import numpy as np
        
        scores = np.array([questionnaire_score, video_score, text_score])
        weights = np.array([
            self.modality_weights['questionnaire'],
            self.modality_weights['video'],
            self.modality_weights['text']
        ])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        fused = np.average(scores, weights=weights)
        
        return float(fused)
    
    @staticmethod
    def _probability_to_severity(probability: float) -> str:
        """Convert probability to severity category"""
        if probability < 0.2:
            return "minimal"
        elif probability < 0.4:
            return "mild"
        elif probability < 0.6:
            return "moderate"
        elif probability < 0.8:
            return "moderately_severe"
        else:
            return "severe"
    
    @staticmethod
    def _check_consistency(q_score: float, v_score: float, t_score: float) -> Dict:
        """Check if modalities agree on depression assessment"""
        import numpy as np
        
        scores = np.array([q_score, v_score, t_score])
        
        # Calculate agreement metrics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # If standard deviation is low, modalities agree well
        # Normalize std to 0-1 agreement score
        max_std = 0.5  # Max expected standard deviation
        agreement_score = max(0.0, 1.0 - (std_score / max_std))
        agreement_score = min(1.0, agreement_score)
        
        return {
            'questionnaire_score': q_score,
            'video_score': v_score,
            'text_score': t_score,
            'score_mean': mean_score,
            'score_std': std_score,
            'agreement_score': agreement_score,
            'all_agree': std_score < 0.2,
            'majority_agree': True  # All three modalities present
        }
    
    @staticmethod
    def _assess_risk_level(probability: float, phq_suicide_risk: bool,
                          consistency: float) -> str:
        """Assess suicide/crisis risk level"""
        
        if phq_suicide_risk:
            return "high_risk"
        
        if probability > 0.8:
            return "high_risk"
        elif probability > 0.6:
            return "moderate_risk"
        elif probability > 0.4:
            return "low_risk"
        else:
            return "minimal_risk"
    
    def generate_clinical_summary(self, fused_result: Dict) -> str:
        """Generate human-readable clinical summary"""
        severity = fused_result['overall_severity']
        probability = fused_result['overall_probability']
        consistency = fused_result['consistency_check']['agreement_score']
        
        summary = f"""
        DEPRESSION ASSESSMENT SUMMARY
        =============================
        Overall Severity: {severity.upper().replace('_', ' ')}
        Depression Probability: {probability:.1%}
        Modality Consistency: {consistency:.0%}
        Risk Level: {fused_result['risk_level'].upper().replace('_', ' ')}
        
        Modality Scores:
        - Questionnaire (PHQ-9): {fused_result['modality_scores']['questionnaire']:.1%}
        - Video/Facial Expression: {fused_result['modality_scores']['video']:.1%}
        - Text/Linguistic: {fused_result['modality_scores']['text']:.1%}
        """
        
        return summary

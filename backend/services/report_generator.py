"""
Report generation service
Creates comprehensive depression screening reports from analysis results
"""
from typing import Dict, Optional
from datetime import datetime
import json

class ReportGenerator:
    """Generates comprehensive depression screening reports"""
    
    def __init__(self):
        self.report_template = {
            'report_id': None,
            'generated_at': None,
            'participant_id': None,
            'session_id': None,
            'summary': None,
            'questionnaire': None,
            'video_analysis': None,
            'text_analysis': None,
            'integrated_assessment': None,
            'recommendations': None,
            'crisis_flag': False
        }
    
    def generate_full_report(self, session_data: Dict) -> Dict:
        """
        Generate comprehensive depression screening report
        
        Args:
            session_data: Complete session data with all analyses
            
        Returns:
            Comprehensive report dictionary
        """
        report = self.report_template.copy()
        
        # Basic info
        report['report_id'] = f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report['generated_at'] = datetime.now().isoformat()
        report['participant_id'] = session_data.get('participant_id')
        report['session_id'] = session_data.get('session_id')
        
        # Questionnaire component
        if 'phq_data' in session_data:
            report['questionnaire'] = self._format_questionnaire_section(session_data['phq_data'])
        
        # Video analysis component
        if 'video_analysis' in session_data:
            report['video_analysis'] = self._format_video_section(session_data['video_analysis'])
        
        # Text analysis component
        if 'text_analysis' in session_data:
            report['text_analysis'] = self._format_text_section(session_data['text_analysis'])
        
        # Integrated assessment
        report['integrated_assessment'] = self._generate_integrated_assessment(session_data)
        
        # Crisis assessment
        report['crisis_flag'] = self._assess_crisis_risk(session_data)
        
        # Clinical recommendations
        report['recommendations'] = self._generate_recommendations(session_data, report['crisis_flag'])
        
        # Overall summary
        report['summary'] = self._generate_summary(report)
        
        return report
    
    @staticmethod
    def _format_questionnaire_section(phq_data: Dict) -> Dict:
        """Format PHQ-9 questionnaire results"""
        if not phq_data:
            return {}
        
        return {
            'phq9_score': phq_data.get('total_score', 0),
            'severity': phq_data.get('severity', 'unknown'),
            'suicide_risk_indicator': phq_data.get('suicide_risk', False),
            'interpretation': ReportGenerator._interpret_phq_score(phq_data.get('total_score', 0))
        }
    
    @staticmethod
    def _interpret_phq_score(score: int) -> str:
        """Interpret PHQ-9 score"""
        if score < 5:
            return "No significant depressive symptoms identified"
        elif score < 10:
            return "Mild depressive symptoms present; monitoring recommended"
        elif score < 15:
            return "Moderate depressive symptoms; clinical follow-up suggested"
        elif score < 20:
            return "Moderately severe depression; professional treatment recommended"
        else:
            return "Severe depression; urgent professional evaluation recommended"
    
    @staticmethod
    def _format_video_section(video_data: Dict) -> Dict:
        """Format facial expression analysis results with feature importance"""
        if not video_data:
            return {}
        
        # Calculate feature contributions to depression probability
        feature_contributions = ReportGenerator._calculate_feature_contributions(video_data)
        
        return {
            'depression_indicators': video_data.get('depression_indicators', {}),
            'micro_expressions_detected': video_data.get('micro_expressions', []),
            'au_patterns': video_data.get('au_patterns', {}),
            'feature_contributions': feature_contributions,
            'interpretation': ReportGenerator._interpret_facial_expression(video_data)
        }
    
    @staticmethod
    def _calculate_feature_contributions(video_data: Dict) -> Dict:
        """Calculate which features contributed most to depression probability"""
        
        # Feature importance mappings with clinical descriptions
        AU_CLINICAL_IMPORTANCE = {
            'AU01': {'name': 'Inner Brow Raiser', 'depression_link': 'worry, sadness, distress', 'weight': 0.15},
            'AU02': {'name': 'Outer Brow Raiser', 'depression_link': 'surprise, worry, fear', 'weight': 0.10},
            'AU04': {'name': 'Brow Lowerer', 'depression_link': 'sadness, anger, concentration difficulties', 'weight': 0.18},
            'AU05': {'name': 'Upper Lid Raiser', 'depression_link': 'fear, alertness, hypervigilance', 'weight': 0.08},
            'AU06': {'name': 'Cheek Raiser', 'depression_link': 'genuine happiness (absence indicates anhedonia)', 'weight': 0.12},
            'AU07': {'name': 'Lid Tightener', 'depression_link': 'tension, stress, genuine emotion', 'weight': 0.07},
            'AU09': {'name': 'Nose Wrinkler', 'depression_link': 'disgust, irritability', 'weight': 0.06},
            'AU10': {'name': 'Upper Lip Raiser', 'depression_link': 'disgust, contempt', 'weight': 0.06},
            'AU12': {'name': 'Lip Corner Puller (Smile)', 'depression_link': 'happiness (reduced in depression)', 'weight': 0.14},
            'AU14': {'name': 'Dimpler', 'depression_link': 'genuine smile, positive affect', 'weight': 0.09},
            'AU15': {'name': 'Lip Corner Depressor', 'depression_link': 'sadness, unhappiness, negative affect', 'weight': 0.17},
            'AU17': {'name': 'Chin Raiser', 'depression_link': 'doubt, sadness, distress', 'weight': 0.10},
            'AU20': {'name': 'Lip Stretcher', 'depression_link': 'fear, tension', 'weight': 0.08},
            'AU23': {'name': 'Lip Tightener', 'depression_link': 'anger, frustration, suppressed emotion', 'weight': 0.09},
            'AU24': {'name': 'Lip Pressor', 'depression_link': 'stress, tension, holding back', 'weight': 0.08},
            'AU25': {'name': 'Lips Part', 'depression_link': 'surprise, shock, mouth breathing', 'weight': 0.05},
            'AU26': {'name': 'Jaw Drop', 'depression_link': 'surprise, shock, mouth breathing', 'weight': 0.05},
            'AU45': {'name': 'Blink', 'depression_link': 'stress, fatigue, avoidance', 'weight': 0.06}
        }
        
        POSE_CLINICAL_IMPORTANCE = {
            'pitch': {'name': 'Head Pitch (Up/Down)', 'depression_link': 'downward gaze avoidance, shame, low energy', 'weight': 0.20},
            'yaw': {'name': 'Head Yaw (Left/Right)', 'depression_link': 'avoidance behaviors, disengagement', 'weight': 0.12},
            'roll': {'name': 'Head Roll (Tilt)', 'depression_link': 'tension, discomfort', 'weight': 0.08},
            'movement_variability': {'name': 'Head Movement Variability', 'depression_link': 'psychomotor retardation (low movement)', 'weight': 0.18},
            'downward_tendency': {'name': 'Downward Head Tendency', 'depression_link': 'shame, low self-esteem, withdrawal', 'weight': 0.22}
        }
        
        GAZE_CLINICAL_IMPORTANCE = {
            'gaze_avoidance': {'name': 'Gaze Avoidance', 'depression_link': 'social anxiety, shame, low self-esteem', 'weight': 0.25},
            'gaze_stability': {'name': 'Gaze Stability', 'depression_link': 'concentration difficulties, agitation (if unstable)', 'weight': 0.18},
            'downward_gaze': {'name': 'Downward Gaze Pattern', 'depression_link': 'sadness, guilt, shame', 'weight': 0.22},
            'eye_contact_duration': {'name': 'Eye Contact Duration', 'depression_link': 'social withdrawal, anxiety (if reduced)', 'weight': 0.20},
            'blink_rate': {'name': 'Blink Rate', 'depression_link': 'stress, fatigue, cognitive load (if elevated)', 'weight': 0.15}
        }
        
        contributions = {
            'top_contributors': [],
            'summary': '',
            'by_category': {
                'action_units': {'contribution_percentage': 0, 'top_features': []},
                'head_pose': {'contribution_percentage': 0, 'top_features': []},
                'gaze_patterns': {'contribution_percentage': 0, 'top_features': []}
            },
            'clinical_interpretation': ''
        }
        
        # Get video data components
        au_patterns = video_data.get('au_patterns', {})
        depression_indicators = video_data.get('depression_indicators', {})
        depression_prob = depression_indicators.get('depression_severity', 0)
        
        if depression_prob == 0:
            contributions['summary'] = "No significant depression indicators detected in facial features."
            return contributions
        
        all_feature_scores = []
        
        # === ACTION UNIT CONTRIBUTIONS ===
        au_total_contribution = 0
        if au_patterns:
            for au_code, au_info in AU_CLINICAL_IMPORTANCE.items():
                # Check if this AU was present in the analysis
                au_value = au_patterns.get(au_code, 0)
                
                if au_value > 0:
                    # Calculate contribution score
                    contribution_score = au_value * au_info['weight'] * depression_prob
                    
                    if contribution_score > 0.02:  # Only include significant contributors
                        feature_detail = {
                            'feature_code': au_code,
                            'feature_name': au_info['name'],
                            'category': 'Action Unit',
                            'activation_level': round(au_value, 3),
                            'contribution_score': round(contribution_score, 4),
                            'contribution_percentage': round(contribution_score / depression_prob * 100, 1) if depression_prob > 0 else 0,
                            'clinical_significance': au_info['depression_link'],
                            'interpretation': ReportGenerator._interpret_au_contribution(au_code, au_value, au_info['depression_link'])
                        }
                        all_feature_scores.append(feature_detail)
                        au_total_contribution += contribution_score
        
        # === HEAD POSE CONTRIBUTIONS ===
        pose_total_contribution = 0
        pose_data = video_data.get('pose_features', {})
        if pose_data:
            for pose_key, pose_info in POSE_CLINICAL_IMPORTANCE.items():
                pose_value = pose_data.get(pose_key, 0)
                
                if abs(pose_value) > 0.1:  # Significant pose deviation
                    contribution_score = abs(pose_value) * pose_info['weight'] * depression_prob
                    
                    if contribution_score > 0.02:
                        feature_detail = {
                            'feature_code': pose_key,
                            'feature_name': pose_info['name'],
                            'category': 'Head Pose',
                            'activation_level': round(pose_value, 3),
                            'contribution_score': round(contribution_score, 4),
                            'contribution_percentage': round(contribution_score / depression_prob * 100, 1) if depression_prob > 0 else 0,
                            'clinical_significance': pose_info['depression_link'],
                            'interpretation': ReportGenerator._interpret_pose_contribution(pose_key, pose_value, pose_info['depression_link'])
                        }
                        all_feature_scores.append(feature_detail)
                        pose_total_contribution += contribution_score
        
        # === GAZE PATTERN CONTRIBUTIONS ===
        gaze_total_contribution = 0
        gaze_data = video_data.get('gaze_features', {})
        if gaze_data:
            for gaze_key, gaze_info in GAZE_CLINICAL_IMPORTANCE.items():
                gaze_value = gaze_data.get(gaze_key, 0)
                
                if abs(gaze_value) > 0.1:
                    contribution_score = abs(gaze_value) * gaze_info['weight'] * depression_prob
                    
                    if contribution_score > 0.02:
                        feature_detail = {
                            'feature_code': gaze_key,
                            'feature_name': gaze_info['name'],
                            'category': 'Gaze Pattern',
                            'activation_level': round(gaze_value, 3),
                            'contribution_score': round(contribution_score, 4),
                            'contribution_percentage': round(contribution_score / depression_prob * 100, 1) if depression_prob > 0 else 0,
                            'clinical_significance': gaze_info['depression_link'],
                            'interpretation': ReportGenerator._interpret_gaze_contribution(gaze_key, gaze_value, gaze_info['depression_link'])
                        }
                        all_feature_scores.append(feature_detail)
                        gaze_total_contribution += contribution_score
        
        # Sort by contribution score (highest first)
        all_feature_scores.sort(key=lambda x: x['contribution_score'], reverse=True)
        
        # Get top 10 contributors
        contributions['top_contributors'] = all_feature_scores[:10]
        
        # Calculate category percentages
        total_contribution = au_total_contribution + pose_total_contribution + gaze_total_contribution
        
        if total_contribution > 0:
            contributions['by_category']['action_units']['contribution_percentage'] = round(au_total_contribution / total_contribution * 100, 1)
            contributions['by_category']['head_pose']['contribution_percentage'] = round(pose_total_contribution / total_contribution * 100, 1)
            contributions['by_category']['gaze_patterns']['contribution_percentage'] = round(gaze_total_contribution / total_contribution * 100, 1)
            
            # Top features by category
            contributions['by_category']['action_units']['top_features'] = [f for f in all_feature_scores if f['category'] == 'Action Unit'][:5]
            contributions['by_category']['head_pose']['top_features'] = [f for f in all_feature_scores if f['category'] == 'Head Pose'][:3]
            contributions['by_category']['gaze_patterns']['top_features'] = [f for f in all_feature_scores if f['category'] == 'Gaze Pattern'][:3]
        
        # Generate summary
        if contributions['top_contributors']:
            top3 = contributions['top_contributors'][:3]
            summary_parts = []
            for feat in top3:
                summary_parts.append(f"{feat['feature_name']} ({feat['contribution_percentage']}%)")
            
            contributions['summary'] = f"Primary contributors to depression probability: {', '.join(summary_parts)}"
            
            # Clinical interpretation
            dominant_category = max(
                contributions['by_category'].items(),
                key=lambda x: x[1]['contribution_percentage']
            )[0]
            
            category_interpretations = {
                'action_units': "Facial muscle movements (Action Units) showed the strongest depression indicators, particularly in expression patterns associated with sadness and reduced positive affect.",
                'head_pose': "Head position and movement patterns were the primary indicators, suggesting psychomotor changes consistent with depression.",
                'gaze_patterns': "Eye gaze behavior was the most prominent feature, indicating potential social withdrawal, avoidance, or concentration difficulties."
            }
            
            contributions['clinical_interpretation'] = category_interpretations.get(dominant_category, "Multiple facial features contributed to depression assessment.")
            
            # Add specific pattern interpretation
            if contributions['by_category']['action_units']['contribution_percentage'] > 40:
                contributions['clinical_interpretation'] += " Elevated sadness-related facial expressions detected."
            
            if contributions['by_category']['head_pose']['contribution_percentage'] > 30:
                contributions['clinical_interpretation'] += " Postural indicators suggest reduced energy or motivation."
            
            if contributions['by_category']['gaze_patterns']['contribution_percentage'] > 30:
                contributions['clinical_interpretation'] += " Gaze patterns indicate possible social anxiety or avoidance behaviors."
        else:
            contributions['summary'] = "Depression probability based on overall facial pattern rather than specific features."
            contributions['clinical_interpretation'] = "Assessment based on holistic facial expression patterns."
        
        return contributions
    
    @staticmethod
    def _interpret_au_contribution(au_code: str, activation: float, clinical_link: str) -> str:
        """Generate interpretation for Action Unit contribution"""
        activation_level = "high" if activation > 0.7 else "moderate" if activation > 0.4 else "mild"
        return f"{activation_level.capitalize()} activation detected; associated with {clinical_link}"
    
    @staticmethod
    def _interpret_pose_contribution(pose_key: str, value: float, clinical_link: str) -> str:
        """Generate interpretation for pose contribution"""
        if 'pitch' in pose_key.lower() and value < -0.3:
            return f"Head pitched downward (value: {value:.2f}); {clinical_link}"
        elif 'movement' in pose_key.lower() and value < 0.3:
            return f"Reduced head movement (value: {value:.2f}); {clinical_link}"
        else:
            return f"Observed pattern (value: {value:.2f}); {clinical_link}"
    
    @staticmethod
    def _interpret_gaze_contribution(gaze_key: str, value: float, clinical_link: str) -> str:
        """Generate interpretation for gaze contribution"""
        if 'avoidance' in gaze_key.lower() and value > 0.5:
            return f"Elevated gaze avoidance (value: {value:.2f}); {clinical_link}"
        elif 'downward' in gaze_key.lower() and value > 0.5:
            return f"Frequent downward gaze (value: {value:.2f}); {clinical_link}"
        elif 'stability' in gaze_key.lower() and value < 0.4:
            return f"Reduced gaze stability (value: {value:.2f}); {clinical_link}"
        else:
            return f"Observed pattern (value: {value:.2f}); {clinical_link}"
    
    @staticmethod
    def _interpret_facial_expression(video_data: Dict) -> str:
        """Interpret facial expression findings"""
        indicators = video_data.get('depression_indicators', {})
        
        depression_index = indicators.get('depression_severity', 0)
        micro_expr_count = len(video_data.get('micro_expressions', []))
        
        if depression_index < 0.3:
            return "Facial expressions generally consistent with non-depressed affect"
        elif depression_index < 0.6:
            return "Facial expressions show some depression-related characteristics"
        else:
            return "Facial expressions show significant depression-related indicators"
    
    @staticmethod
    def _format_text_section(text_data: Dict) -> Dict:
        """Format linguistic analysis results"""
        if not text_data:
            return {}
        
        return {
            'sentiment_score': text_data.get('sentiment_score', 0),
            'negative_word_density': text_data.get('negative_word_ratio', 0),
            'semantic_diversity': text_data.get('semantic_diversity', 0),
            'depression_risk_features': text_data.get('depression_risk_features', {}),
            'interpretation': ReportGenerator._interpret_linguistic(text_data)
        }
    
    @staticmethod
    def _interpret_linguistic(text_data: Dict) -> str:
        """Interpret linguistic patterns"""
        sentiment = text_data.get('sentiment_score', 0)
        neg_ratio = text_data.get('negative_word_ratio', 0)
        
        if sentiment > 0.2 and neg_ratio < 0.05:
            return "Linguistic patterns generally positive"
        elif sentiment < -0.2 and neg_ratio > 0.15:
            return "Linguistic patterns show significant negative bias; rumination present"
        else:
            return "Linguistic patterns show moderate concerning features"
    
    @staticmethod
    def _generate_integrated_assessment(session_data: Dict) -> Dict:
        """Integrate findings from all modalities"""
        assessment = {
            'overall_depression_probability': 0.0,
            'modality_agreement': "",
            'confidence_level': 0.0,
            'key_findings': []
        }
        
        # Calculate overall probability
        scores = []
        weights = []
        
        if 'video_analysis' in session_data:
            video_score = session_data['video_analysis'].get('depression_indicators', {}).get('depression_severity', 0)
            scores.append(video_score)
            weights.append(0.4)
        
        if 'phq_data' in session_data:
            phq_score = session_data['phq_data'].get('total_score', 0) / 27  # Normalize to 0-1
            scores.append(phq_score)
            weights.append(0.4)
        
        if 'text_analysis' in session_data:
            text_score = session_data['text_analysis'].get('sentiment_score', 0)
            # Convert -1..1 to 0..1
            text_score = (1 - text_score) / 2
            scores.append(text_score)
            weights.append(0.2)
        
        if scores:
            import numpy as np
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()
            assessment['overall_depression_probability'] = float(np.average(scores, weights=weights_array))
            assessment['confidence_level'] = 0.7  # Moderate confidence for multimodal assessment
        
        return assessment
    
    @staticmethod
    def _assess_crisis_risk(session_data: Dict) -> bool:
        """Assess suicide/crisis risk"""
        # Check suicide indicator in questionnaire
        if session_data.get('phq_data', {}).get('suicide_risk', False):
            return True
        
        # Check for extreme depression indicators
        depression_prob = session_data.get('integrated_assessment', {}).get('overall_depression_probability', 0)
        if depression_prob > 0.85:
            return True
        
        return False
    
    @staticmethod
    def _generate_recommendations(session_data: Dict, crisis_flag: bool) -> Dict:
        """Generate comprehensive clinical recommendations based on severity and diagnosis"""
        recommendations = {
            'immediate_actions': [],
            'clinical_referral': [],
            'treatment_options': [],
            'lifestyle_interventions': [],
            'follow_up': None,
            'resources': [],
            'medications_to_consider': [],
            'therapy_recommendations': [],
            'self_care_strategies': []
        }
        
        # Get severity indicators
        phq_score = session_data.get('phq_data', {}).get('total_score', 0)
        depression_prob = session_data.get('integrated_assessment', {}).get('overall_depression_probability', 0)
        
        # Analyze for comorbidities
        video_data = session_data.get('video_analysis', {})
        micro_expr = video_data.get('micro_expressions', [])
        fear_count = sum(1 for me in micro_expr if me.get('expression') == 'fear')
        
        # Check for PTSD indicators
        ptsd_indicators = ReportGenerator._assess_ptsd_indicators(session_data, fear_count)
        
        # CRISIS INTERVENTION (Severe/Immediate Risk)
        if crisis_flag:
            recommendations['immediate_actions'] = [
                "🚨 CRISIS PROTOCOL ACTIVATED - Immediate intervention required",
                "Conduct comprehensive suicide risk assessment using Columbia Scale",
                "Contact emergency services if immediate danger present (911)",
                "Consider emergency psychiatric evaluation and possible hospitalization",
                "Implement safety planning: Remove means, establish 24/7 supervision",
                "Contact emergency contacts and support network immediately",
                "Do NOT leave individual alone until professional evaluation completed"
            ]
            recommendations['resources'] = [
                "🆘 National Suicide Prevention Lifeline: 988 (24/7)",
                "🆘 Crisis Text Line: Text HOME to 741741 (24/7)",
                "🆘 Veterans Crisis Line: 988 then Press 1 (for veterans)",
                "🆘 SAMHSA National Helpline: 1-800-662-4357 (24/7)",
                "🆘 Emergency Services: 911",
                "🆘 Local Emergency Room - Walk-in psychiatric evaluation available"
            ]
            recommendations['clinical_referral'] = [
                "Emergency psychiatric evaluation within 24 hours",
                "Inpatient psychiatric hospitalization may be necessary",
                "Intensive outpatient program (IOP) if hospitalization declined",
                "Daily check-ins with mental health professional"
            ]
            
        # SEVERE DEPRESSION (PHQ-9: 20-27, Probability > 0.75)
        elif phq_score >= 20 or depression_prob > 0.75:
            recommendations['immediate_actions'] = [
                "Schedule psychiatric evaluation within 1 week",
                "Assess suicide risk at every contact",
                "Consider partial hospitalization program (PHP) if functioning impaired",
                "Implement safety monitoring and support system"
            ]
            recommendations['treatment_options'] = [
                "MEDICATION: Antidepressant therapy strongly recommended",
                "  - SSRIs (first-line): Sertraline, Escitalopram, Fluoxetine",
                "  - SNRIs: Venlafaxine, Duloxetine for severe cases",
                "  - Augmentation strategies if first trial inadequate",
                "  - Consider atypical antipsychotics for severe/psychotic features",
                "PSYCHOTHERAPY: Evidence-based therapy essential",
                "  - Cognitive Behavioral Therapy (CBT) - 12-16 weekly sessions",
                "  - Behavioral Activation - immediate engagement in activities",
                "  - Interpersonal Therapy (IPT) if relationship issues prominent",
                "COMBINATION: Medication + Therapy shows best outcomes for severe depression"
            ]
            recommendations['clinical_referral'] = [
                "Psychiatrist for medication management (urgent - within 1 week)",
                "Licensed therapist (PhD/PsyD/LCSW) for weekly psychotherapy",
                "Consider ECT evaluation if medication-resistant",
                "Transcranial Magnetic Stimulation (TMS) for treatment-resistant depression"
            ]
            recommendations['follow_up'] = "Weekly follow-up for first month, then bi-weekly"
            recommendations['lifestyle_interventions'] = [
                "⚠️ Avoid alcohol and recreational drugs - worsens depression",
                "Sleep schedule: Maintain regular sleep-wake times (even if difficult)",
                "Nutrition: Regular meals, reduce sugar/caffeine",
                "Physical activity: Gentle walks 10-15 minutes daily (builds to 30 min)",
                "Social connection: Schedule 1-2 brief contacts daily (calls/texts acceptable)"
            ]
            
        # MODERATELY SEVERE DEPRESSION (PHQ-9: 15-19, Probability 0.60-0.75)
        elif phq_score >= 15 or depression_prob > 0.60:
            recommendations['immediate_actions'] = [
                "Schedule mental health evaluation within 2 weeks",
                "Monitor suicide risk - discuss at each appointment",
                "Consider intensive outpatient program (IOP) if available"
            ]
            recommendations['treatment_options'] = [
                "MEDICATION: Antidepressant recommended",
                "  - SSRIs first-line: Sertraline 50-200mg, Escitalopram 10-20mg",
                "  - Expected response: 4-6 weeks for full effect",
                "  - Trial period: 8-12 weeks at therapeutic dose before switching",
                "PSYCHOTHERAPY: Strong recommendation",
                "  - CBT: 12-20 sessions, focus on thought patterns and behaviors",
                "  - Problem-Solving Therapy: Practical approach for daily challenges",
                "  - Mindfulness-Based Cognitive Therapy (MBCT): Prevent relapse",
                "COMBINATION: Medication + therapy recommended for faster/better results"
            ]
            recommendations['therapy_recommendations'] = [
                "Weekly therapy sessions initially (45-50 minutes)",
                "Learn cognitive restructuring techniques",
                "Develop behavioral activation schedule",
                "Practice mindfulness and relaxation strategies"
            ]
            recommendations['clinical_referral'] = [
                "Psychiatrist or primary care physician for medication evaluation (within 2 weeks)",
                "Licensed therapist for evidence-based psychotherapy (within 2 weeks)",
                "Consider group therapy for additional support"
            ]
            recommendations['follow_up'] = "Bi-weekly follow-up for first 6 weeks, then monthly"
            recommendations['lifestyle_interventions'] = [
                "Sleep hygiene: 7-9 hours nightly, consistent schedule",
                "Exercise: 30 minutes moderate activity 5x/week (walking, biking, swimming)",
                "Nutrition: Mediterranean diet pattern, omega-3 fatty acids",
                "Social activity: Schedule activities 3-4 times/week (even if unmotivated)",
                "Light therapy: 30 minutes morning bright light (especially for seasonal pattern)",
                "Limit alcohol to <1-2 drinks/week, avoid completely if possible"
            ]
            
        # MODERATE DEPRESSION (PHQ-9: 10-14, Probability 0.45-0.60)
        elif phq_score >= 10 or depression_prob > 0.45:
            recommendations['immediate_actions'] = [
                "Schedule mental health consultation within 3-4 weeks",
                "Screen for suicide risk at consultation",
                "Monitor symptom progression"
            ]
            recommendations['treatment_options'] = [
                "PSYCHOTHERAPY: First-line recommendation",
                "  - CBT: 8-16 sessions, evidence-based for moderate depression",
                "  - Behavioral Activation: Focus on re-engaging with life activities",
                "  - ACT (Acceptance Commitment Therapy): Build psychological flexibility",
                "MEDICATION: Consider if:",
                "  - Psychotherapy not available or preferred",
                "  - Symptoms not improving after 6-8 weeks of therapy",
                "  - History of good response to medication",
                "  - Patient preference for medication",
                "DIGITAL INTERVENTIONS:",
                "  - Evidence-based apps: Moodpath, Sanvello, Woebot",
                "  - Online CBT programs: MoodGYM, Beating the Blues"
            ]
            recommendations['therapy_recommendations'] = [
                "Weekly or bi-weekly therapy sessions (50 minutes)",
                "Cognitive restructuring for negative thought patterns",
                "Behavioral activation - schedule pleasant activities",
                "Develop coping skills toolkit"
            ]
            recommendations['clinical_referral'] = [
                "Licensed therapist (LCSW, LPC, PhD/PsyD) for psychotherapy",
                "Primary care physician for medical evaluation and medication if needed",
                "Consider counselor or social worker if therapist unavailable"
            ]
            recommendations['follow_up'] = "Monthly check-ins for first 3 months"
            recommendations['lifestyle_interventions'] = [
                "Exercise: 150 minutes moderate activity per week (very effective for moderate depression)",
                "Sleep routine: Regular bedtime/waketime, avoid screens before bed",
                "Social connections: Regular contact with friends/family",
                "Stress management: Yoga, meditation, progressive muscle relaxation",
                "Nutrition: Balanced diet, limit processed foods and sugar",
                "Sunlight exposure: 15-30 minutes daily outdoor time",
                "Journaling: Daily mood tracking and gratitude practice"
            ]
            recommendations['self_care_strategies'] = [
                "Daily routine: Maintain structure even when unmotivated",
                "Pleasant activities: Schedule 1-2 enjoyable activities daily",
                "Social support: Reach out to trusted friend/family member",
                "Relaxation: 10-minute daily mindfulness or deep breathing",
                "Sleep: Prioritize consistent 7-9 hours nightly",
                "Avoid isolation: Leave house at least once daily"
            ]
            
        # MILD DEPRESSION (PHQ-9: 5-9, Probability 0.30-0.45)
        elif phq_score >= 5 or depression_prob > 0.30:
            recommendations['immediate_actions'] = [
                "Monitor symptoms for 2-4 weeks for progression",
                "Implement lifestyle interventions and self-care strategies",
                "Consider preventive mental health consultation"
            ]
            recommendations['treatment_options'] = [
                "WATCHFUL WAITING with active monitoring",
                "LIFESTYLE INTERVENTIONS: Often sufficient for mild depression",
                "  - Exercise: As effective as medication for mild depression",
                "  - Sleep optimization: Critical foundation",
                "  - Social engagement: Protective factor",
                "PSYCHOEDUCATION: Understanding depression and coping strategies",
                "PREVENTIVE THERAPY: If symptoms persist beyond 6-8 weeks",
                "  - Brief CBT (6-8 sessions)",
                "  - Problem-solving therapy",
                "  - Online self-help programs",
                "MEDICATION: Generally NOT first-line for mild depression"
            ]
            recommendations['clinical_referral'] = [
                "Primary care physician for general health evaluation",
                "Mental health counselor if symptoms persist or worsen",
                "Consider workplace EAP (Employee Assistance Program) if available"
            ]
            recommendations['follow_up'] = "Re-assess in 4-6 weeks; earlier if worsening"
            recommendations['lifestyle_interventions'] = [
                "⭐ Exercise: 30-45 minutes, 5x/week (KEY intervention for mild depression)",
                "Sleep: Consistent schedule, 7-9 hours, good sleep hygiene",
                "Social activities: Regular social engagement, avoid isolation",
                "Stress reduction: Identify and address stressors",
                "Hobbies/interests: Re-engage with enjoyable activities",
                "Mindfulness: Daily meditation or mindfulness practice (10-20 min)",
                "Limit substances: Reduce/eliminate alcohol, caffeine moderation",
                "Nature exposure: Spend time outdoors regularly (proven benefit)"
            ]
            recommendations['self_care_strategies'] = [
                "Daily structure: Maintain regular routine for meals, sleep, activities",
                "Pleasant events scheduling: Plan enjoyable activities several times/week",
                "Social connection: Reach out to friends/family regularly",
                "Gratitude practice: List 3 positives daily",
                "Physical activity: Walk, bike, swim, dance - find what you enjoy",
                "Creative outlets: Music, art, writing, crafts",
                "Limit negative inputs: Reduce news/social media if distressing",
                "Volunteer: Helping others can improve mood"
            ]
            recommendations['resources'] = [
                "📱 Mental health apps: Headspace, Calm, Insight Timer (meditation)",
                "📚 Self-help books: 'Feeling Good' by David Burns, 'Mind Over Mood'",
                "🌐 Online resources: depression.org, NAMI.org, MentalHealth.gov",
                "☎️ SAMHSA Helpline: 1-800-662-4357 (information and referrals)"
            ]
            
        # NO SIGNIFICANT DEPRESSION (PHQ-9: 0-4, Probability < 0.30)
        else:
            recommendations['immediate_actions'] = [
                "No immediate clinical intervention needed",
                "Continue healthy lifestyle practices",
                "Monitor for any future symptom development"
            ]
            recommendations['treatment_options'] = [
                "PREVENTION FOCUS: Maintain mental wellness",
                "Continue current coping strategies that are working",
                "Build resilience through stress management skills"
            ]
            recommendations['follow_up'] = "Routine mental health screening in 1 year, or sooner if concerns arise"
            recommendations['lifestyle_interventions'] = [
                "Maintain regular exercise routine",
                "Preserve good sleep hygiene",
                "Continue social connections and support network",
                "Practice stress management techniques",
                "Engage in meaningful activities and hobbies"
            ]
        
        # PTSD-SPECIFIC RECOMMENDATIONS (if indicators present)
        if ptsd_indicators['likely_ptsd']:
            recommendations['ptsd_specific'] = ReportGenerator._generate_ptsd_recommendations(ptsd_indicators)
        
        # ANXIETY COMORBIDITY
        if fear_count > 3 or session_data.get('anxiety_indicators', False):
            recommendations['anxiety_specific'] = [
                "Screen for anxiety disorders (GAD-7 questionnaire)",
                "Consider CBT with exposure therapy component",
                "Relaxation techniques: Progressive muscle relaxation, deep breathing",
                "Medications: SSRIs also treat anxiety; consider buspirone if SSRI contraindicated",
                "Mindfulness-based stress reduction (MBSR) programs"
            ]
        
        # GENERAL RESOURCES (for all levels)
        if not crisis_flag:
            recommendations['resources'].extend([
                "📞 SAMHSA National Helpline: 1-800-662-4357 (treatment referrals, 24/7)",
                "🌐 NAMI (National Alliance on Mental Illness): nami.org or 1-800-950-6264",
                "🌐 Depression and Bipolar Support Alliance: dbsalliance.org",
                "🌐 Anxiety and Depression Association: adaa.org",
                "🌐 Psychology Today Therapist Finder: psychologytoday.com/us/therapists",
                "🌐 SAMHSA Treatment Locator: findtreatment.gov",
                "💼 Employee Assistance Program (EAP): Check if available through employer",
                "🏥 Community Mental Health Centers: Often sliding scale fees",
                "🎓 University counseling centers: Low-cost options for students"
            ])
        
        return recommendations
    
    @staticmethod
    def _assess_ptsd_indicators(session_data: Dict, fear_count: int) -> Dict:
        """Assess for PTSD indicators based on expression patterns"""
        indicators = {
            'likely_ptsd': False,
            'ptsd_type': None,
            'confidence': 0.0,
            'supporting_evidence': []
        }
        
        # Get microexpression data
        video_data = session_data.get('video_analysis', {})
        micro_expressions = video_data.get('micro_expressions', [])
        
        # Count relevant expressions
        fear_expr = sum(1 for me in micro_expressions if me.get('expression') == 'fear')
        suppressed_expr = sum(1 for me in micro_expressions if me.get('expression') == 'suppressed_emotion')
        
        # Check for PTSD indicators
        if fear_expr >= 3:
            indicators['supporting_evidence'].append(f"Elevated fear microexpressions ({fear_expr} detected)")
        
        if suppressed_expr >= 3:
            indicators['supporting_evidence'].append(f"High emotional suppression ({suppressed_expr} detected)")
        
        # Check text for trauma language
        text_data = session_data.get('text_analysis', {})
        trauma_keywords = ['trauma', 'flashback', 'nightmare', 'scared', 'afraid', 'helpless']
        # This would need actual text analysis
        
        # Determine if PTSD likely
        if len(indicators['supporting_evidence']) >= 2:
            indicators['likely_ptsd'] = True
            indicators['confidence'] = min(0.9, len(indicators['supporting_evidence']) * 0.3)
            
            # Classify PTSD type (simplified)
            if fear_expr > 5:
                indicators['ptsd_type'] = 'acute_stress'
            else:
                indicators['ptsd_type'] = 'chronic_ptsd'
        
        return indicators
    
    @staticmethod
    def _generate_ptsd_recommendations(ptsd_indicators: Dict) -> Dict:
        """Generate PTSD-specific recommendations"""
        ptsd_recommendations = {
            'screening': [
                "⚠️ PTSD indicators detected - formal screening recommended",
                "Administer PCL-5 (PTSD Checklist) for comprehensive assessment",
                "Screen for trauma history using trauma-focused interview",
                "Assess for dissociative symptoms (derealization, depersonalization)"
            ],
            'trauma_focused_therapy': [
                "🎯 GOLD STANDARD: Evidence-based trauma-focused psychotherapy",
                "  - Prolonged Exposure (PE): 8-15 weekly 90-minute sessions",
                "  - Cognitive Processing Therapy (CPT): 12 weekly sessions",
                "  - Eye Movement Desensitization & Reprocessing (EMDR): 6-12 sessions",
                "  - Trauma-Focused CBT: For trauma-related depression/anxiety",
                "Do NOT use general talk therapy alone - trauma-focused treatment essential"
            ],
            'medications': [
                "SSRIs: Sertraline and Paroxetine (FDA-approved for PTSD)",
                "Prazosin: For PTSD-related nightmares (1-20mg at bedtime)",
                "Avoid benzodiazepines: Not recommended for PTSD, risk of dependence",
                "Consider medications for comorbid depression/anxiety"
            ],
            'stabilization': [
                "Safety planning if self-harm risk present",
                "Grounding techniques for dissociation/flashbacks",
                "Emotion regulation skills training",
                "Sleep hygiene and nightmare management"
            ],
            'referrals': [
                "Trauma specialist therapist (PE, CPT, or EMDR trained)",
                "Psychiatrist familiar with trauma treatment if medications needed",
                "Veteran-specific services if military trauma (VA system)",
                "Consider residential PTSD program if severe and not responding"
            ],
            'resources': [
                "🎖️ Veterans Crisis Line: 988, press 1 (for veterans)",
                "☎️ RAINN (sexual assault): 1-800-656-4673 (24/7)",
                "🌐 PTSD National Center: ptsd.va.gov",
                "🌐 International Society for Traumatic Stress: istss.org (find therapist)",
                "🌐 Sidran Institute: sidran.org (trauma resources)"
            ]
        }
        
        # Add type-specific recommendations
        if ptsd_indicators.get('ptsd_type') == 'acute_stress':
            ptsd_recommendations['acute_specific'] = [
                "Trauma occurred recently - early intervention critical",
                "Psychological First Aid (PFA) approach",
                "Early CPT or EMDR within 3 months of trauma shows best results",
                "Monitor for development of chronic PTSD symptoms"
            ]
        elif ptsd_indicators.get('ptsd_type') == 'chronic_ptsd':
            ptsd_recommendations['chronic_specific'] = [
                "Chronic PTSD pattern - comprehensive trauma treatment program",
                "Prolonged Exposure or CPT recommended (strong evidence base)",
                "Address comorbid depression concurrently",
                "May require longer treatment duration (3-6 months or more)",
                "Consider intensive outpatient trauma program"
            ]
        
        return ptsd_recommendations
    
    @staticmethod
    def _generate_summary(report: Dict) -> str:
        """Generate executive summary with feature importance highlights"""
        summary_parts = []
        
        # Questionnaire summary
        if 'questionnaire' in report and report['questionnaire']:
            q_severity = report['questionnaire'].get('severity', 'unknown')
            summary_parts.append(f"PHQ-9 Assessment: {q_severity} depression")
        
        # Integrated assessment
        if 'integrated_assessment' in report and report['integrated_assessment']:
            prob = report['integrated_assessment']['overall_depression_probability']
            summary_parts.append(f"Multimodal depression probability: {prob:.1%}")
        
        # Video analysis feature contributions
        if 'video_analysis' in report and report['video_analysis']:
            feature_contrib = report['video_analysis'].get('feature_contributions', {})
            if feature_contrib and feature_contrib.get('top_contributors'):
                top_feature = feature_contrib['top_contributors'][0]
                summary_parts.append(f"Primary video indicator: {top_feature['feature_name']} ({top_feature['contribution_percentage']}% contribution)")
        
        # Crisis status
        if report['crisis_flag']:
            summary_parts.append("⚠️ CRISIS RISK IDENTIFIED - Immediate action required")
        
        return "; ".join(summary_parts) if summary_parts else "Assessment incomplete"
    
    @staticmethod
    def export_report_json(report: Dict, filepath: str) -> bool:
        """Export report to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False

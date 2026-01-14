"""
Enhanced Report Generator with Detailed Facial Behavior Analysis
Generates comprehensive PDF reports for clinical use
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64


class DetailedReportGenerator:
    """Generate comprehensive clinical reports with facial behavior analysis"""
    
    def __init__(self, output_dir='outputs/reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Severity thresholds
        self.severity_levels = {
            (0, 4): "Minimal",
            (5, 9): "Mild",
            (10, 14): "Moderate",
            (15, 19): "Moderately Severe",
            (20, 24): "Severe"
        }
    
    def generate_report(self, session_data, responses, phq8_score, facial_features_per_question):
        """
        Generate detailed PDF report
        
        Args:
            session_data: Session metadata
            responses: List of question-answer pairs
            phq8_score: Predicted PHQ-8 score
            facial_features_per_question: List of facial features for each answer
        
        Returns:
            path to generated PDF file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"depression_screening_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF
        doc = SimpleDocTemplate(filepath, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("DEPRESSION SCREENING ASSESSMENT REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata table
        severity = self._get_severity(phq8_score)
        confidence = self._calculate_confidence(facial_features_per_question)
        
        metadata = [
            ["Report Generated:", datetime.now().strftime("%B %d, %Y at %H:%M")],
            ["Session ID:", session_data.get('session_id', 'N/A')],
            ["PHQ-8 Score:", f"{phq8_score:.1f} / 24"],
            ["Severity Level:", severity],
            ["Model Confidence:", f"{confidence:.1%}"],
            ["Facial Analysis:", "✓ Enabled" if len(facial_features_per_question) > 0 else "✗ Not Used"]
        ]
        
        t = Table(metadata, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Score visualization
        story.append(Paragraph("SEVERITY ASSESSMENT", heading_style))
        score_chart = self._create_score_visualization(phq8_score)
        if score_chart:
            story.append(Image(score_chart, width=5*inch, height=1.5*inch))
        story.append(Spacer(1, 0.2*inch))
        
        # Clinical interpretation
        story.append(Paragraph("CLINICAL INTERPRETATION", heading_style))
        interpretation = self._get_clinical_interpretation(phq8_score, severity)
        story.append(Paragraph(interpretation, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Facial Behavior Analysis (if available)
        if len(facial_features_per_question) > 0:
            story.append(Paragraph("FACIAL BEHAVIOR ANALYSIS", heading_style))
            
            # Aggregate statistics
            behavior_stats = self._calculate_behavioral_statistics(facial_features_per_question)
            story.append(Paragraph(f"<b>Overall Behavioral Patterns:</b>", styles['BodyText']))
            for key, value in behavior_stats.items():
                story.append(Paragraph(f"• {key}: {value}", styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
            
            # Graphs
            story.append(Paragraph("<b>Facial Expression Patterns Over Time:</b>", styles['BodyText']))
            expression_chart = self._create_expression_timeline(facial_features_per_question)
            if expression_chart:
                story.append(Image(expression_chart, width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.3*inch))
        
        # Question-by-Question Analysis
        story.append(PageBreak())
        story.append(Paragraph("DETAILED RESPONSE ANALYSIS", heading_style))
        
        for idx, response_data in enumerate(responses, 1):
            question = response_data.get('question', f'Question {idx}')
            answer = response_data.get('answer', 'No response')
            facial_features = facial_features_per_question[idx - 1] if idx <= len(facial_features_per_question) else None
            
            # Question header
            q_style = ParagraphStyle('Question', parent=styles['BodyText'], 
                                      fontName='Helvetica-Bold', fontSize=12, 
                                      textColor=colors.HexColor('#667eea'))
            story.append(Paragraph(f"Question {idx}: {question}", q_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Answer
            story.append(Paragraph(f"<b>Response:</b> {answer}", styles['BodyText']))
            story.append(Spacer(1, 0.1*inch))
            
            # Facial analysis for this question
            if facial_features:
                facial_summary = self._summarize_facial_features(facial_features)
                story.append(Paragraph(f"<b>Facial Behavior Observations:</b>", styles['BodyText']))
                for observation in facial_summary:
                    story.append(Paragraph(f"  • {observation}", styles['BodyText']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("RECOMMENDATIONS", heading_style))
        recommendations = self._generate_recommendations(phq8_score, severity)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['BodyText']))
            story.append(Spacer(1, 0.1*inch))
        
        # Disclaimer
        story.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['BodyText'],
                                          fontSize=9, textColor=colors.grey,
                                          alignment=TA_JUSTIFY)
        story.append(Paragraph(
            "<b>IMPORTANT DISCLAIMER:</b> This screening tool is for informational purposes only "
            "and is not a substitute for professional medical advice, diagnosis, or treatment. "
            "Results should be reviewed by a qualified mental health professional. "
            "If you are experiencing thoughts of self-harm or suicide, please contact emergency "
            "services immediately (988 Suicide & Crisis Lifeline).",
            disclaimer_style
        ))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _get_severity(self, score):
        """Get severity level from PHQ-8 score"""
        for (min_score, max_score), severity in self.severity_levels.items():
            if min_score <= score <= max_score:
                return severity
        return "Unknown"
    
    def _calculate_confidence(self, facial_features):
        """Calculate model confidence based on data availability"""
        base_confidence = 0.75  # Text-only baseline
        
        if len(facial_features) > 0:
            # Boost confidence if facial data available
            valid_features = sum(1 for f in facial_features if f is not None and any(f))
            confidence_boost = (valid_features / len(facial_features)) * 0.15
            return min(base_confidence + confidence_boost, 0.95)
        
        return base_confidence
    
    def _create_score_visualization(self, score):
        """Create horizontal bar showing score on PHQ-8 scale"""
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # Define severity regions
        regions = [
            (0, 4, 'Minimal', '#28a745'),
            (5, 9, 'Mild', '#ffc107'),
            (10, 14, 'Moderate', '#fd7e14'),
            (15, 19, 'Mod. Severe', '#dc3545'),
            (20, 24, 'Severe', '#6f42c1')
        ]
        
        # Draw regions
        for start, end, label, color in regions:
            ax.barh(0, end - start + 1, left=start, height=0.5, 
                    color=color, alpha=0.3, edgecolor='black', linewidth=0.5)
            ax.text((start + end) / 2, 0, label, ha='center', va='center', 
                    fontsize=9, fontweight='bold')
        
        # Draw score marker
        ax.plot(score, 0, marker='v', markersize=15, color='red', zorder=10)
        ax.text(score, 0.35, f'{score:.1f}', ha='center', fontsize=12, 
                fontweight='bold', color='red')
        
        ax.set_xlim(-1, 25)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(range(0, 25, 5))
        ax.set_yticks([])
        ax.set_xlabel('PHQ-8 Score', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def _calculate_behavioral_statistics(self, facial_features_list):
        """Calculate aggregate behavioral statistics"""
        stats = {}
        
        # Check if we have valid data
        valid_data = [f for f in facial_features_list if f is not None and len(f) >= 34]
        
        if len(valid_data) == 0:
            return {"Data Availability": "Insufficient facial data for analysis"}
        
        # Convert to numpy array (assuming 34 features)
        features_array = np.array([f[:34] for f in valid_data])
        
        # AU features (indices 0-21)
        au_features = features_array[:, :22]
        avg_expressiveness = np.mean(np.std(au_features, axis=0))
        stats["Facial Expressiveness"] = f"{avg_expressiveness:.3f} (higher = more expressive)"
        
        # Smile detection (AU12 - index 11)
        if au_features.shape[1] > 11:
            avg_smile = np.mean(au_features[:, 11])
            stats["Smile Frequency"] = f"{avg_smile:.2f} (0-1 scale)"
        
        # Eye openness (AU5, AU7 - indices 4, 6)
        if au_features.shape[1] > 6:
            avg_eye_openness = np.mean(au_features[:, [4, 6]])
            stats["Average Eye Openness"] = f"{avg_eye_openness:.2f} (0-1 scale)"
        
        # Head pose variability (indices 22-27)
        if features_array.shape[1] >= 28:
            pose_features = features_array[:, 22:28]
            pose_variability = np.mean(np.std(pose_features, axis=0))
            stats["Head Movement Variability"] = f"{pose_variability:.3f}"
        
        # Gaze features (indices 28-33)
        if features_array.shape[1] >= 34:
            gaze_features = features_array[:, 28:34]
            gaze_stability = np.mean(np.std(gaze_features, axis=0))
            stats["Gaze Stability"] = f"{gaze_stability:.3f} (lower = more stable)"
        
        return stats
    
    def _create_expression_timeline(self, facial_features_list):
        """Create timeline of facial expressions across questions"""
        valid_data = [f for f in facial_features_list if f is not None and len(f) >= 34]
        
        if len(valid_data) == 0:
            return None
        
        features_array = np.array([f[:34] for f in valid_data])
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        questions = range(1, len(valid_data) + 1)
        
        # Plot 1: Smile (AU12)
        if features_array.shape[1] > 11:
            axes[0].plot(questions, features_array[:, 11], marker='o', color='#28a745', linewidth=2)
            axes[0].set_ylabel('Smile\nIntensity', fontsize=10, fontweight='bold')
            axes[0].set_ylim(0, 1)
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Eye Openness (average of AU5, AU7)
        if features_array.shape[1] > 6:
            eye_openness = np.mean(features_array[:, [4, 6]], axis=1)
            axes[1].plot(questions, eye_openness, marker='s', color='#667eea', linewidth=2)
            axes[1].set_ylabel('Eye\nOpenness', fontsize=10, fontweight='bold')
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Overall Expressiveness (std of AU features)
        au_features = features_array[:, :22]
        expressiveness = np.std(au_features, axis=1)
        axes[2].plot(questions, expressiveness, marker='^', color='#fd7e14', linewidth=2)
        axes[2].set_ylabel('Facial\nExpressiveness', fontsize=10, fontweight='bold')
        axes[2].set_xlabel('Question Number', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Facial Expression Patterns Across Questions', fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def _summarize_facial_features(self, features):
        """Generate human-readable summary of facial features"""
        if features is None or len(features) < 34:
            return ["No facial data captured for this response"]
        
        observations = []
        
        # Smile (AU12 - index 11)
        if len(features) > 11:
            smile = features[11]
            if smile > 0.6:
                observations.append("Strong positive facial expression (smile detected)")
            elif smile > 0.3:
                observations.append("Moderate positive expression")
            elif smile < 0.1:
                observations.append("Minimal facial expression / flat affect")
        
        # Eye openness (AU5, AU7)
        if len(features) > 6:
            eye_avg = (features[4] + features[6]) / 2
            if eye_avg < 0.3:
                observations.append("Reduced eye openness (possible fatigue or low energy)")
            elif eye_avg > 0.7:
                observations.append("Alert, engaged eye contact")
        
        # Brow movement (AU1, AU2, AU4 - worry/concentration indicators)
        if len(features) > 3:
            brow = max(features[0], features[1], features[3])
            if brow > 0.5:
                observations.append("Elevated brow activity (possible worry or concentration)")
        
        # Head pose (pitch - looking down)
        if len(features) >= 23:
            pitch = features[22]
            if pitch < -0.3:
                observations.append("Head tilted downward (possible low mood indicator)")
            elif pitch > 0.3:
                observations.append("Head tilted upward (engaged posture)")
        
        # Gaze stability
        if len(features) >= 34:
            gaze_variance = np.std(features[28:34])
            if gaze_variance > 0.5:
                observations.append("Variable gaze direction (possible discomfort or avoidance)")
            elif gaze_variance < 0.2:
                observations.append("Stable, direct gaze (good engagement)")
        
        if not observations:
            observations.append("Neutral facial expression patterns observed")
        
        return observations
    
    def _get_clinical_interpretation(self, score, severity):
        """Generate clinical interpretation text"""
        interpretations = {
            "Minimal": (
                "The assessment indicates minimal depressive symptoms. The individual is functioning well "
                "with no significant impairment. While symptoms may be present, they are minor and do not "
                "substantially interfere with daily activities or quality of life."
            ),
            "Mild": (
                "The assessment indicates mild depressive symptoms. The individual may experience some "
                "difficulty with daily functioning, mood fluctuations, or reduced energy. While these symptoms "
                "are noticeable, they typically do not severely impair work, social, or personal activities. "
                "Early intervention and monitoring are recommended."
            ),
            "Moderate": (
                "The assessment indicates moderate depressive symptoms. The individual is likely experiencing "
                "noticeable impairment in social, occupational, or other important areas of functioning. "
                "Symptoms such as low mood, reduced interest, sleep disturbances, and concentration difficulties "
                "are present and affecting quality of life. Professional evaluation and treatment are recommended."
            ),
            "Moderately Severe": (
                "The assessment indicates moderately severe depressive symptoms. The individual is experiencing "
                "significant impairment in multiple life domains. Symptoms are substantially interfering with work, "
                "relationships, and daily activities. There may be persistent feelings of worthlessness, guilt, "
                "or thoughts of death. Immediate professional intervention is strongly recommended."
            ),
            "Severe": (
                "The assessment indicates severe depressive symptoms. The individual is likely experiencing "
                "profound impairment in most or all areas of functioning. Symptoms are intense and pervasive, "
                "potentially including suicidal ideation or inability to perform basic self-care. "
                "<b>IMMEDIATE CLINICAL INTERVENTION IS REQUIRED.</b> If suicidal thoughts are present, "
                "contact emergency services immediately (988 Suicide & Crisis Lifeline)."
            )
        }
        
        return interpretations.get(severity, "Unable to determine severity classification.")
    
    def _generate_recommendations(self, score, severity):
        """Generate personalized recommendations"""
        recommendations = []
        
        if severity in ["Minimal", "Mild"]:
            recommendations.extend([
                "Continue monitoring symptoms over the next 2-4 weeks",
                "Maintain regular sleep schedule (7-9 hours per night)",
                "Engage in regular physical activity (30 minutes, 3-5 times per week)",
                "Practice stress-reduction techniques (meditation, deep breathing, yoga)",
                "Maintain social connections with supportive friends and family",
                "Consider lifestyle modifications: balanced diet, reduced alcohol/caffeine",
                "Schedule follow-up screening in 4 weeks to monitor symptom progression"
            ])
        
        elif severity == "Moderate":
            recommendations.extend([
                "<b>Seek professional evaluation from a mental health provider within 1-2 weeks</b>",
                "Consider evidence-based psychotherapy (Cognitive Behavioral Therapy, Interpersonal Therapy)",
                "Discuss treatment options with healthcare provider (therapy, medication, or combination)",
                "Establish consistent daily routines (sleep, meals, physical activity)",
                "Avoid alcohol and recreational drugs, which can worsen symptoms",
                "Engage support network (family, friends, support groups)",
                "Monitor for worsening symptoms and seek immediate help if suicidal thoughts emerge",
                "Schedule follow-up screening in 2 weeks"
            ])
        
        else:  # Moderately Severe or Severe
            recommendations.extend([
                "<b>URGENT: Contact mental health crisis services or healthcare provider TODAY</b>",
                "<b>If experiencing suicidal thoughts, call 988 Suicide & Crisis Lifeline immediately</b>",
                "Do not delay seeking professional help - same-day or next-day appointment recommended",
                "Consider intensive treatment options: outpatient program, day treatment, or hospitalization if needed",
                "Inform trusted family member or friend about current mental state",
                "Remove access to means of self-harm (firearms, medications)",
                "Avoid being alone; ensure continuous supervision if suicidal ideation present",
                "Follow treatment plan prescribed by mental health professional",
                "Weekly follow-up appointments until stabilization achieved"
            ])
        
        # Add behavioral recommendations based on facial analysis
        recommendations.append("")
        recommendations.append("<b>Behavioral Health Recommendations:</b>")
        
        if score > 10:
            recommendations.append("Practice mindfulness and emotion regulation techniques")
            recommendations.append("Monitor nonverbal communication patterns with therapist")
            recommendations.append("Consider body-focused therapies (somatic experiencing, EMDR if trauma-related)")
        
        return recommendations


# Convenience function
def generate_clinical_report(session_data, responses, phq8_score, facial_features):
    """
    Generate comprehensive clinical PDF report
    
    Args:
        session_data: Session metadata dict
        responses: List of response dicts with 'question' and 'answer' keys
        phq8_score: Predicted PHQ-8 score (0-24)
        facial_features: List of facial feature arrays (one per question)
    
    Returns:
        Path to generated PDF file
    """
    generator = DetailedReportGenerator()
    return generator.generate_report(session_data, responses, phq8_score, facial_features)

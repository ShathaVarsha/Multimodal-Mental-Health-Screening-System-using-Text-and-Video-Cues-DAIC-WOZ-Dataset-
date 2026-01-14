"""
=================================================================
STEP 7: MODEL 4 - REPORT GENERATOR
=================================================================
Generate comprehensive depression assessment reports

This script:
1. Loads predictions from Model 3
2. Creates severity-based report templates
3. Generates personalized assessment reports
4. Includes behavioral observations
5. Provides recommendations

Inputs:
  - outputs/model3_predictions.pkl (from Step 6)
  - outputs/session_aggregates.pkl (from Step 2)

Outputs:
  - outputs/report_templates.json (template library)
  - outputs/sample_reports/ (generated reports)
  - outputs/report_evaluation.json (metrics)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from utils import *

# =============================================================================
# REPORT TEMPLATES
# =============================================================================

REPORT_TEMPLATES = {
    "minimal": {
        "severity": "Minimal Depression",
        "phq_range": "0-4",
        "description": "Your responses indicate minimal or no symptoms of depression.",
        "summary": "Based on your responses, you appear to be experiencing minimal symptoms of depression. This is a positive indicator of your current mental health status.",
        "recommendations": [
            "Continue maintaining healthy lifestyle habits",
            "Stay connected with supportive friends and family",
            "Engage in regular physical activity",
            "Practice stress management techniques",
            "Maintain a regular sleep schedule"
        ],
        "follow_up": "While your current assessment shows minimal symptoms, please don't hesitate to reach out for support if you notice changes in your mood or functioning."
    },
    "mild": {
        "severity": "Mild Depression",
        "phq_range": "5-9",
        "description": "Your responses suggest mild symptoms of depression that may be affecting your daily life.",
        "summary": "Your assessment indicates mild depressive symptoms. While these symptoms are manageable, they warrant attention and self-care strategies.",
        "recommendations": [
            "Consider talking to a mental health professional",
            "Increase physical activity (aim for 30 min daily)",
            "Practice mindfulness or meditation",
            "Maintain social connections",
            "Monitor your sleep patterns",
            "Consider keeping a mood journal",
            "Limit alcohol and caffeine intake"
        ],
        "follow_up": "We recommend monitoring your symptoms over the next few weeks. If symptoms persist or worsen, please consult with a healthcare provider."
    },
    "moderate": {
        "severity": "Moderate Depression",
        "phq_range": "10-14",
        "description": "Your responses indicate moderate symptoms of depression that are likely impacting your daily functioning.",
        "summary": "Your assessment shows moderate depressive symptoms that warrant professional attention. These symptoms may be significantly affecting your quality of life.",
        "recommendations": [
            "Schedule an appointment with a mental health professional",
            "Consider evidence-based therapies (CBT, IPT)",
            "Discuss treatment options with your doctor",
            "Reach out to trusted friends or family for support",
            "Engage in structured daily activities",
            "Prioritize self-care and healthy routines",
            "Join a support group if available",
            "Avoid isolation and maintain social connections"
        ],
        "follow_up": "Professional support is recommended. Please reach out to a mental health provider to discuss treatment options that may include therapy and/or medication."
    },
    "moderately_severe": {
        "severity": "Moderately Severe Depression",
        "phq_range": "15-19",
        "description": "Your responses suggest moderately severe symptoms of depression that are significantly affecting your life.",
        "summary": "Your assessment indicates moderately severe depression. Professional treatment is strongly recommended to help you manage these symptoms effectively.",
        "recommendations": [
            "Seek professional help immediately",
            "Contact your healthcare provider or therapist",
            "Consider both therapy and medication options",
            "Inform trusted family members or friends about your situation",
            "Avoid making major life decisions while experiencing these symptoms",
            "Create a safety plan with your healthcare provider",
            "Attend all scheduled appointments",
            "Consider intensive outpatient programs if recommended"
        ],
        "follow_up": "Immediate professional support is strongly recommended. Please contact a mental health provider as soon as possible. If you have thoughts of self-harm, please call a crisis hotline immediately.",
        "crisis_resources": True
    },
    "severe": {
        "severity": "Severe Depression",
        "phq_range": "20-24",
        "description": "Your responses indicate severe symptoms of depression requiring immediate professional attention.",
        "summary": "Your assessment shows severe depression. It is crucial that you receive professional help immediately. You do not have to face this alone - effective treatments are available.",
        "recommendations": [
            "Seek immediate professional help",
            "Contact emergency services if you have thoughts of self-harm",
            "Reach out to a crisis hotline",
            "Inform family members or close friends",
            "Do not be alone - stay with trusted individuals",
            "Follow all treatment recommendations from healthcare providers",
            "Consider hospitalization if recommended",
            "Engage with intensive treatment programs"
        ],
        "follow_up": "Immediate intervention is critical. Please contact emergency services or a crisis hotline if you are experiencing thoughts of self-harm. Professional treatment including therapy and medication is essential.",
        "crisis_resources": True,
        "urgent": True
    }
}

CRISIS_RESOURCES = {
    "National Suicide Prevention Lifeline": "1-800-273-8255",
    "Crisis Text Line": "Text HOME to 741741",
    "SAMHSA National Helpline": "1-800-662-4357",
    "Emergency": "911"
}

# =============================================================================
# REPORT GENERATOR CLASS
# =============================================================================

class ReportGenerator:
    """Generate depression assessment reports"""
    
    def __init__(self):
        """Initialize report generator"""
        self.logger = setup_logging(LOG_FILE, LOG_LEVEL)
        print_section("STEP 7: MODEL 4 - REPORT GENERATOR")
        
        self.templates = REPORT_TEMPLATES
        self.predictions_data = None
        self.session_data = None
        
        self.generated_reports = []
    
    def load_data(self):
        """Load predictions and session data"""
        print_step(1, "Loading Prediction Data")
        
        # Load predictions from Model 3
        predictions_file = OUTPUTS_DIR / "model3_predictions.pkl"
        if not predictions_file.exists():
            print(f"❌ Predictions not found: {predictions_file}")
            print(f"   Please run: python step6_model3_fusion.py")
            return False
        
        self.predictions_data = load_pickle(predictions_file)
        
        # Load session data
        session_file = OUTPUTS_DIR / "session_aggregates.pkl"
        if session_file.exists():
            self.session_data = load_pickle(session_file)
        
        print(f"✓ Data loaded")
        print(f"  Sessions: {len(self.predictions_data['session_ids'])}")
        print(f"  Predictions: {len(self.predictions_data['predictions'])}")
        
        return True
    
    def get_severity_level(self, phq8_score):
        """Determine severity level from PHQ-8 score"""
        if phq8_score < 5:
            return "minimal"
        elif phq8_score < 10:
            return "mild"
        elif phq8_score < 15:
            return "moderate"
        elif phq8_score < 20:
            return "moderately_severe"
        else:
            return "severe"
    
    def extract_behavioral_observations(self, session_id):
        """Extract behavioral observations from session data"""
        observations = []
        
        if self.session_data is None:
            return observations
        
        # Find session
        session = self.session_data[self.session_data["session_id"] == session_id]
        if len(session) == 0:
            return observations
        
        session = session.iloc[0]
        
        # Check gaze patterns
        if "gaze_downward" in session:
            if session["gaze_downward"] > 0.5:
                observations.append("Frequent downward gaze patterns observed, which may indicate low mood")
        
        # Check facial expressions (Action Units)
        if "au_depression_score" in session:
            if session["au_depression_score"] > 0.3:
                observations.append("Facial expressions showing signs of sadness or distress")
        
        if "au_smile_score" in session:
            if session["au_smile_score"] < 0.2:
                observations.append("Limited positive facial expressions observed")
        
        # Check head pose
        if "pose_range" in session:
            if session["pose_range"] < 10:
                observations.append("Reduced head movement, suggesting low energy or engagement")
        
        return observations
    
    def generate_report(self, session_id, phq8_score, predicted_score=None):
        """Generate comprehensive report for a session"""
        # Determine severity
        severity_level = self.get_severity_level(phq8_score)
        template = self.templates[severity_level]
        
        # Create report
        report = {
            "session_id": int(session_id),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phq8_score": float(phq8_score),
            "predicted_score": float(predicted_score) if predicted_score else None,
            "severity": template["severity"],
            "phq_range": template["phq_range"],
            "description": template["description"],
            "summary": template["summary"],
            "recommendations": template["recommendations"],
            "follow_up": template["follow_up"]
        }
        
        # Add behavioral observations
        observations = self.extract_behavioral_observations(session_id)
        if observations:
            report["behavioral_observations"] = observations
        
        # Add crisis resources if needed
        if template.get("crisis_resources"):
            report["crisis_resources"] = CRISIS_RESOURCES
        
        # Add urgency flag
        if template.get("urgent"):
            report["urgent"] = True
        
        return report
    
    def generate_all_reports(self):
        """Generate reports for all sessions"""
        print_step(2, "Generating Assessment Reports")
        
        session_ids = self.predictions_data["session_ids"]
        actuals = self.predictions_data["actuals"]
        predictions = self.predictions_data["predictions"]
        
        for i, session_id in enumerate(session_ids):
            actual = actuals[i]
            predicted = predictions[i]
            
            # Generate report using actual score
            report = self.generate_report(session_id, actual, predicted)
            self.generated_reports.append(report)
            
            print(f"  Session {session_id}: PHQ-8={actual:.1f}, Predicted={predicted:.1f}, Severity={report['severity']}")
        
        print(f"\n✓ Generated {len(self.generated_reports)} reports")
    
    def save_reports(self):
        """Save report templates and generated reports"""
        print_step(3, "Saving Reports")
        
        # Save templates
        save_json(self.templates, OUTPUTS_DIR / "report_templates.json")
        print(f"  ✓ Templates saved")
        
        # Create reports directory
        reports_dir = OUTPUTS_DIR / "sample_reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save individual reports
        for report in self.generated_reports:
            session_id = report["session_id"]
            filename = reports_dir / f"report_session_{session_id}.json"
            save_json(report, filename)
        
        print(f"  ✓ {len(self.generated_reports)} reports saved to {reports_dir}")
        
        # Save summary
        summary = {
            "total_reports": len(self.generated_reports),
            "severity_distribution": {},
            "average_phq8": np.mean([r["phq8_score"] for r in self.generated_reports]),
            "average_predicted": np.mean([r["predicted_score"] for r in self.generated_reports if r["predicted_score"]])
        }
        
        # Count severity levels
        for report in self.generated_reports:
            severity = report["severity"]
            summary["severity_distribution"][severity] = summary["severity_distribution"].get(severity, 0) + 1
        
        save_json(summary, OUTPUTS_DIR / "report_evaluation.json")
        
        print(f"\n  Summary:")
        print(f"    Average PHQ-8: {summary['average_phq8']:.2f}")
        print(f"    Average Predicted: {summary['average_predicted']:.2f}")
        print(f"    Severity Distribution:")
        for severity, count in summary["severity_distribution"].items():
            print(f"      {severity}: {count}")
        
        print(f"\n✓ Reports saved successfully!")
        print(f"  → All models trained! Run web interface: python web_interface_8.py")
    
    def generate_formatted_report(self, report):
        """Generate human-readable formatted report"""
        lines = []
        lines.append("=" * 70)
        lines.append("DEPRESSION ASSESSMENT REPORT")
        lines.append("=" * 70)
        lines.append(f"\nSession ID: {report['session_id']}")
        lines.append(f"Date: {report['date']}")
        lines.append(f"\nPHQ-8 Score: {report['phq8_score']:.1f} ({report['phq_range']})")
        if report.get('predicted_score'):
            lines.append(f"Predicted Score: {report['predicted_score']:.1f}")
        
        lines.append(f"\n{'─' * 70}")
        lines.append(f"SEVERITY: {report['severity'].upper()}")
        lines.append(f"{'─' * 70}")
        
        lines.append(f"\n{report['description']}")
        lines.append(f"\n{report['summary']}")
        
        if report.get('behavioral_observations'):
            lines.append(f"\n{'─' * 70}")
            lines.append("BEHAVIORAL OBSERVATIONS:")
            lines.append(f"{'─' * 70}")
            for obs in report['behavioral_observations']:
                lines.append(f"• {obs}")
        
        lines.append(f"\n{'─' * 70}")
        lines.append("RECOMMENDATIONS:")
        lines.append(f"{'─' * 70}")
        for rec in report['recommendations']:
            lines.append(f"• {rec}")
        
        lines.append(f"\n{'─' * 70}")
        lines.append("FOLLOW-UP:")
        lines.append(f"{'─' * 70}")
        lines.append(report['follow_up'])
        
        if report.get('crisis_resources'):
            lines.append(f"\n{'─' * 70}")
            lines.append("CRISIS RESOURCES:")
            lines.append(f"{'─' * 70}")
            for name, number in CRISIS_RESOURCES.items():
                lines.append(f"• {name}: {number}")
        
        if report.get('urgent'):
            lines.append(f"\n{'!' * 70}")
            lines.append("⚠️  URGENT: Please seek immediate professional help  ⚠️")
            lines.append(f"{'!' * 70}")
        
        lines.append("\n" + "=" * 70)
        lines.append("This assessment is for informational purposes only and does not")
        lines.append("constitute a clinical diagnosis. Please consult with a qualified")
        lines.append("mental health professional for proper evaluation and treatment.")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def print_sample_reports(self):
        """Print sample formatted reports"""
        print_step(4, "Sample Reports")
        
        # Print one report from each severity level if available
        severity_samples = {}
        for report in self.generated_reports:
            severity_level = self.get_severity_level(report["phq8_score"])
            if severity_level not in severity_samples:
                severity_samples[severity_level] = report
        
        for severity_level in ["minimal", "mild", "moderate", "moderately_severe", "severe"]:
            if severity_level in severity_samples:
                print(f"\n{'=' * 70}")
                print(f"SAMPLE REPORT: {severity_level.upper()}")
                print("=" * 70)
                formatted = self.generate_formatted_report(severity_samples[severity_level])
                print(formatted)
                
                # Save formatted report
                reports_dir = OUTPUTS_DIR / "sample_reports"
                session_id = severity_samples[severity_level]["session_id"]
                with open(reports_dir / f"report_session_{session_id}.txt", "w", encoding="utf-8") as f:
                    f.write(formatted)

# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_report_generation():
    """Test report generation with sample scores"""
    print_section("TESTING REPORT GENERATION")
    
    generator = ReportGenerator()
    
    # Test each severity level
    test_scores = [2, 7, 12, 17, 22]
    test_names = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
    
    print("Testing report generation for each severity level:\n")
    
    for score, name in zip(test_scores, test_names):
        report = generator.generate_report(
            session_id=999,
            phq8_score=score,
            predicted_score=score + np.random.uniform(-1, 1)
        )
        
        print(f"PHQ-8 Score: {score} → Severity: {report['severity']}")
        print(f"  Recommendations: {len(report['recommendations'])}")
        if report.get('crisis_resources'):
            print(f"  ⚠️  Crisis resources included")
        if report.get('urgent'):
            print(f"  ⚠️  URGENT FLAG")
        print()
    
    print("✓ Report generation test complete")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    # Initialize
    generator = ReportGenerator()
    
    # Load data
    if not generator.load_data():
        # If no predictions available, run test
        print("\n⚠️  No predictions available. Running test mode...\n")
        test_report_generation()
        return
    
    # Generate reports
    generator.generate_all_reports()
    
    # Save reports
    generator.save_reports()
    
    # Print samples
    generator.print_sample_reports()
    
    # Run test
    test_report_generation()
    
    print_section("MODEL 4 COMPLETE - ALL MODELS TRAINED!")

if __name__ == "__main__":
    main()
